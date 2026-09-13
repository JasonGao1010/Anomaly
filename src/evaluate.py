"""Official STU evaluation and exact global AP loss attribution for single scans."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from numba import njit

from .data import FramePrediction, FrozenDataset, _atomic_json, host_disk, source_identity
from .protocol import PROJECT_ROOT, load_protocol
from .scene import STUSequence, LabelMode
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


def score_bits(scores, score_kind):
    """Order finite signed float32 values without rounding or sigmoid saturation."""
    values = np.asarray(scores, dtype=np.float32).copy()
    values[values == 0] = 0
    bits = values.view(np.uint32)
    if score_kind == "logit":
        bits = np.where(
            bits & np.uint32(0x80000000), ~bits, bits ^ np.uint32(0x80000000)
        )
    elif score_kind != "probability":
        raise ValueError("unknown score kind")
    return bits


def bits_score(bits, score_kind):
    bits = np.asarray(bits, dtype=np.uint32)
    if score_kind == "logit":
        bits = np.where(
            bits & np.uint32(0x80000000), bits ^ np.uint32(0x80000000), ~bits
        )
    return bits.view(np.float32)


def packed_scores(scores, target, *, score_kind="probability"):
    """Encode exact nonnegative float32 score bits and one binary label, losslessly."""
    scores = np.asarray(scores, dtype=np.float32)
    target = np.asarray(target)
    if (
        scores.shape != target.shape
        or scores.ndim != 1
        or not np.isfinite(scores).all()
        or (score_kind == "probability" and np.any((scores < 0) | (scores > 1)))
        or np.any((target != 0) & (target != 1))
    ):
        raise ValueError("metric records require finite scores and binary labels")
    # Positive float bits have the same order as their values; canonicalize -0.
    return (score_bits(scores, score_kind).astype(np.uint64) << 1) | target.astype(
        np.uint64
    )


def score_groups(ordered, chunk_size=1 << 20):
    """Yield exact score ties in descending order, including ties crossing chunks."""
    pending = None
    for end in range(len(ordered), 0, -chunk_size):
        values = np.array(ordered[max(0, end - chunk_size) : end][::-1])
        bits = values >> 1
        starts = np.r_[0, np.flatnonzero(bits[1:] != bits[:-1]) + 1]
        counts = np.diff(np.r_[starts, len(values)]).astype(np.int64)
        positives = np.add.reduceat((values & 1).astype(np.int64), starts)
        bits = bits[starts].astype(np.uint32)
        if pending is not None:
            if bits[0] == pending[0]:
                counts[0] += pending[1]
                positives[0] += pending[2]
            else:
                yield tuple(
                    np.asarray([x], dtype=d)
                    for x, d in zip(
                        pending, (np.uint32, np.int64, np.int64), strict=True
                    )
                )
        pending = (bits[-1], counts[-1], positives[-1])
        if len(bits) > 1:
            yield bits[:-1], counts[:-1], positives[:-1]
    if pending is not None:
        yield tuple(
            np.asarray([x], dtype=d)
            for x, d in zip(pending, (np.uint32, np.int64, np.int64), strict=True)
        )


def exact_metrics(
    ordered,
    *,
    chunk_size=1 << 20,
    prevalence=None,
    fpr_limit=0.01,
    score_kind="probability",
    observe=None,
    fpr_limits=(),
):
    """Exact point pooling with bounded RAM; ordered is an ascending uint64 array.

    No score quantization is used. ROC drops the same collinear threshold nodes
    as sklearn's default roc_curve before applying the upstream strict TPR > .95.
    """
    positive = sum(
        int(np.sum(ordered[start : start + chunk_size] & 1, dtype=np.int64))
        for start in range(0, len(ordered), chunk_size)
    )
    negative = len(ordered) - positive
    return metrics_from_groups(
        score_groups(ordered, chunk_size), positive=positive, negative=negative,
        prevalence=prevalence, fpr_limit=fpr_limit, score_kind=score_kind,
        observe=observe, fpr_limits=fpr_limits,
    )


def metrics_from_groups(
    groups,
    *,
    positive,
    negative,
    prevalence=None,
    fpr_limit=0.01,
    score_kind="probability",
    observe=None,
    fpr_limits=(),
):
    """Reduce complete float32 ties without expanding their point counts.

    Each block is (uint32 score bits, int64 counts, int64 positive counts).
    Bits follow score_bits' ordering and must strictly descend across all blocks;
    ties from separate frames must be merged before this reduction.
    """
    if any(not 0 <= limit <= 1 for limit in (fpr_limit, *fpr_limits)) or (
        prevalence is not None and not 0 < prevalence < 1
    ):
        raise ValueError("invalid diagnostic prevalence or FPR limit")
    if any(
        not isinstance(count, (int, np.integer)) or count < 0
        for count in (positive, negative)
    ):
        raise ValueError("class totals require nonnegative integer counts")
    positive, negative = int(positive), int(negative)
    result = {
        "AP": None,
        "AUROC": None,
        "FPR95": None,
        "normal_count": negative,
        "anomaly_count": positive,
        "recall_at_fpr_limit": None,
    }
    if prevalence is not None:
        result.update(standardized_AP=None, recall_at_fpr_limit=None)
    if not positive or not negative:
        if positive:
            result.update(AP=100.0)
        return result
    tp = fp = 0
    ap = area = standardized_ap = 0.0
    points = {limit: dict(recall=0.0, FPR=0.0, threshold=None, tp=0, fp=0)
              for limit in (fpr_limit, *fpr_limits)}
    fpr95 = None
    previous = None
    high_recall = None
    first = True
    last_bits = None
    for bits, counts, pos in groups:
        bits, counts, pos = map(np.asarray, (bits, counts, pos))
        if (
            bits.ndim != 1 or counts.shape != bits.shape or pos.shape != bits.shape
            or bits.dtype != np.uint32 or counts.dtype != np.int64
            or pos.dtype != np.int64
            or np.any(counts <= 0) or np.any(pos < 0) or np.any(pos > counts)
        ):
            raise ValueError("score groups require uint32 bits and valid int64 counts")
        if not len(bits):
            continue
        if np.any(bits[1:] >= bits[:-1]) or (
            last_bits is not None and bits[0] >= last_bits
        ):
            raise ValueError("score groups must be complete ties in descending order")
        last_bits = bits[-1]
        neg = counts - pos
        tps = tp + np.cumsum(pos, dtype=np.int64)
        fps = fp + np.cumsum(neg, dtype=np.int64)
        recall, fpr = tps / positive, fps / negative
        if observe is not None:
            # Observers see the same complete ties used by the authoritative AP.
            observe(bits, pos, tps, fps, positive, negative)
        ap += float(np.sum(np.diff(np.r_[tp / positive, recall]) * tps / (tps + fps)))
        if prevalence is not None:
            # Constant class weights retain every score and change only prevalence.
            precision = (
                prevalence * recall / (prevalence * recall + (1 - prevalence) * fpr)
            )
            standardized_ap += float(np.sum(pos / positive * precision))
        for limit, point in points.items():
            feasible = np.flatnonzero(fpr <= limit)
            if len(feasible) and tps[feasible[-1]] > point["tp"]:
                # Complete ties are indivisible; retain the first threshold at the best recall.
                index = int(np.searchsorted(tps, tps[feasible[-1]]))
                points[limit] = dict(
                    recall=float(recall[index]) * 100,
                    FPR=float(fpr[index]) * 100,
                    threshold=float(bits_score(bits[index], score_kind)),
                    tp=int(tps[index]),
                    fp=int(fps[index]),
                )
        area += float(
            np.sum(
                np.diff(np.r_[fp / negative, fpr])
                * (recall + np.r_[tp / positive, recall[:-1]])
                * 0.5
            )
        )
        if previous is not None and fpr95 is None:
            p, n, r, f, was_first, last_point = previous
            if (was_first or p != pos[0] or n != neg[0]) and r > 0.95:
                fpr95 = f
                high_recall = last_point
        keep = (pos[:-1] != pos[1:]) | (neg[:-1] != neg[1:])
        if first and len(keep):
            keep[0] = True
        eligible = np.flatnonzero(keep & (recall[:-1] > 0.95))
        if fpr95 is None and len(eligible):
            index = int(eligible[0])
            fpr95 = float(fpr[index])
            high_recall = dict(
                threshold=float(bits_score(bits[index], score_kind)),
                tp=int(tps[index]),
                fp=int(fps[index]),
            )
        previous = (
            int(pos[-1]),
            int(neg[-1]),
            float(recall[-1]),
            float(fpr[-1]),
            first and len(pos) == 1,
            dict(
                threshold=float(bits_score(bits[-1], score_kind)),
                tp=int(tps[-1]),
                fp=int(fps[-1]),
            ),
        )
        tp, fp = int(tps[-1]), int(fps[-1])
        first = False
    if tp != positive or fp != negative:
        raise ValueError("score-group counts do not match class totals")
    if fpr95 is None:
        fpr95 = previous[3]  # The final ROC threshold is always retained.
        high_recall = previous[5]
    result.update(AP=ap * 100, AUROC=area * 100, FPR95=fpr95 * 100)
    operating_point = points[fpr_limit]
    result["recall_at_fpr_limit"] = operating_point
    if fpr_limits:
        result["operating_points"] = {
            f"{limit:g}": {**point, "FPR_limit": limit * 100,
                            "precision": 100 * point["tp"] / (point["tp"] + point["fp"])
                            if point["tp"] + point["fp"] else None}
            for limit, point in points.items()
        }
    result["official_high_recall"] = {
        **high_recall,
        "fn": positive - high_recall["tp"],
        "tn": negative - high_recall["fp"],
        "recall": high_recall["tp"] / positive * 100,
        "FPR": high_recall["fp"] / negative * 100,
    }
    if prevalence is not None:
        result.update(
            standardized_AP=standardized_ap * 100,
            recall_at_fpr_limit=operating_point,
        )
    return result


def pooled_files(
    paths,
    *,
    ranges=None,
    prevalence=None,
    score_kind="probability",
    observe=None,
    fpr_limits=(),
):
    """Sort exact records on disk, then reduce them in bounded chunks."""
    sizes = [path.stat().st_size for path in paths]
    if any(size % 8 for size in sizes):
        raise ValueError("truncated exact evaluation records")
    if ranges is None:
        ranges = [(0, size // 8) for size in sizes]
    if len(ranges) != len(paths) or any(
        start < 0 or count < 0 or (start + count) * 8 > size
        for (start, count), size in zip(ranges, sizes, strict=True)
    ):
        raise ValueError("evaluation record range exceeds its source file")
    size = sum(count * 8 for _, count in ranges)
    if not size:
        return exact_metrics(
            np.empty(0, np.uint64), prevalence=prevalence, score_kind=score_kind,
            fpr_limits=fpr_limits,
        )
    with tempfile.TemporaryFile(dir=paths[0].parent) as stream:
        stream.truncate(size)
        ordered = np.memmap(stream, dtype=np.uint64, mode="r+", shape=(size // 8,))
        offset = 0
        for path, (start, count) in zip(paths, ranges, strict=True):
            with path.open("rb") as source:
                source.seek(start * 8)
                while count:
                    block = np.fromfile(
                        source, dtype=np.uint64, count=min(count, 1 << 20)
                    )
                    if not len(block):
                        raise ValueError("truncated evaluation record range")
                    ordered[offset : offset + len(block)] = block
                    offset += len(block)
                    count -= len(block)
        # Numeric in-place quicksort avoids point-count-sized index/ROC arrays.
        ordered.sort(kind="quicksort")
        result = exact_metrics(
            ordered, prevalence=prevalence, score_kind=score_kind, observe=observe,
            fpr_limits=fpr_limits,
        )
        del ordered
    return result


def evaluation_targets(points, semantic):
    """The official point filter, without its anomaly-frame eligibility gate."""
    distance = np.linalg.norm(points, axis=1)
    inside = (distance >= PointOODMetricsCalculator.min_eval_distance) & (
        distance <= PointOODMetricsCalculator.max_eval_distance
    )
    target = np.where(semantic == 0, -1, np.where(semantic == 2, 1, 0))
    return np.where(inside, target, -1)


def official_metrics(calculator):
    # Keep the upstream ROC convention (including its strict TPR > 0.95 test).
    metrics = calculator.compute_metrics()
    return {
        key: float(metrics[key])
        if key in metrics and np.isfinite(metrics[key])
        else None
        for key in ("AP", "AUROC", "FPR95")
    }


def diagnostic_bin(count, distance):
    if count < 5:
        return None
    if distance is None or not 2.5 <= distance <= 50:
        raise ValueError("eligible anomalies require an official-range distance")
    i = int(np.searchsorted([20, 100, 500], count, side="right"))
    j = int(np.searchsorted([10, 20, 35], distance, side="right"))
    return f"{i}_{j}"


class APAttribution:
    """Keep precision and required FPR at complete positive-score ties."""

    def __init__(self):
        self.bits, self.precision, self.required_fpr = [], [], []
        self._lookup = None

    def __call__(self, bits, pos, tps, fps, positive, negative):
        use = pos > 0
        self.bits.append(bits[use])
        self.precision.append(tps[use] / (tps[use] + fps[use]))
        self.required_fpr.append(fps[use] / negative)
        self._lookup = None

    def values(self, scores, *, score_kind="logit"):
        if self._lookup is None:
            bits = np.concatenate(self.bits) if self.bits else np.empty(0, np.uint32)
            order = np.argsort(bits)
            precision = (
                np.concatenate(self.precision) if self.precision else np.empty(0)
            )
            required = (
                np.concatenate(self.required_fpr) if self.required_fpr else np.empty(0)
            )
            self._lookup = bits[order], precision[order], required[order]
        bits, precision, required = self._lookup
        query = score_bits(scores, score_kind)
        indexes = np.searchsorted(bits, query)
        if np.any(indexes >= len(bits)) or not np.array_equal(bits[indexes], query):
            raise ValueError("anomaly scores are absent from the global ranking")
        return precision[indexes], required[indexes]


def official_frame(source, prediction):
    """Return complete file-slot scores and official targets for this one scan."""
    if source.labels is None:
        raise ValueError("official evaluation requires semantic labels")
    scores = prediction.restore(source)
    target = evaluation_targets(source.xyzi[:, :3], source.labels.semantic)
    eligible = (
        int(np.count_nonzero(target == 1))
        >= PointOODMetricsCalculator.min_num_points_to_eval
    )
    return scores, target, eligible


def synthetic_targets(frozen, *, official=False):
    """Filter frozen insertion labels without reinterpreting native semantics."""
    target = frozen.anomaly_target
    if not official:
        return target, True
    distance = np.linalg.norm(frozen.source.xyzi[:, :3], axis=1)
    inside = (distance >= PointOODMetricsCalculator.min_eval_distance) & (
        distance <= PointOODMetricsCalculator.max_eval_distance
    )
    target = np.where(inside, target, -1).astype(np.int8)
    eligible = np.count_nonzero(target == 1) >= PointOODMetricsCalculator.min_num_points_to_eval
    return target, bool(eligible)


def evaluate_frames(frames, *, directory=None, observe=None, check_resources=None):
    """Pool eligible points exactly, sorting one temporary file in place."""
    rows, count, seen = [], 0, set()
    with tempfile.TemporaryFile(dir=directory) as stream:
        for source, prediction in frames:
            identity = (source.partition, source.sequence_id, source.frame_id)
            if identity in seen:
                raise ValueError("duplicate source scan in metric pooling")
            seen.add(identity)
            scores, target, eligible = official_frame(source, prediction)
            valid = target >= 0
            row = dict(
                sequence=source.sequence_id,
                frame=source.frame_id,
                eligible=eligible,
                anomaly_points=int(np.count_nonzero(target == 1)),
                normal_points=int(np.count_nonzero(target == 0)),
            )
            rows.append(row)
            if eligible:
                packed = packed_scores(scores[valid], target[valid], score_kind="logit")
                packed.tofile(stream)
                count += len(packed)
            if check_resources is not None:
                check_resources()
        stream.flush()
        if count:
            ordered = np.memmap(stream, dtype=np.uint64, mode="r+", shape=(count,))
            ordered.sort(kind="quicksort")
            result = exact_metrics(ordered, score_kind="logit", observe=observe)
            del ordered
        else:
            result = exact_metrics(np.empty(0, np.uint64), score_kind="logit")
    result.update(frames=len(rows), eligible_frames=sum(r["eligible"] for r in rows))
    return result, rows


def prediction_frames(data_root, prediction_root, sequence_ids, protocol=None):
    protocol = load_protocol() if protocol is None else protocol
    for sequence_id in sequence_ids:
        source = STUSequence.open(
            data_root,
            protocol=protocol,
            partition="val",
            sequence_id=sequence_id,
            label_mode=LabelMode.REQUIRED,
        )
        for frame in source:
            path = (
                Path(prediction_root)
                / "val"
                / str(sequence_id)
                / f"{frame.frame_id:06d}.npz"
            )
            yield frame, FramePrediction.load(path, frame)


_COUNT_DTYPE = np.dtype([("bits", "<u4"), ("count", "<i8"), ("positive", "<i8")])


@njit
def _compress_counts(ordered, records):
    """Collapse complete ties in a sorted packed block; no floating arithmetic."""
    used = 0
    for index in range(len(ordered) - 1, -1, -1):
        key = np.uint32(ordered[index] >> 1)
        positive = np.int64(ordered[index] & 1)
        if used and records[used - 1]["bits"] == key:
            records[used - 1]["count"] += 1
            records[used - 1]["positive"] += positive
        else:
            records[used]["bits"] = key
            records[used]["count"] = 1
            records[used]["positive"] = positive
            used += 1
    return used


@njit
def _merge_counts(left, right, output):
    """Merge descending runs, stopping before either unread suffix is needed."""
    i = j = used = 0
    while i < len(left) and j < len(right):
        a, b = left[i]["bits"], right[j]["bits"]
        if a >= b:
            output[used] = left[i]
            i += 1
            if a == b:
                output[used]["count"] += right[j]["count"]
                output[used]["positive"] += right[j]["positive"]
                j += 1
        else:
            output[used] = right[j]
            j += 1
        used += 1
    return i, j, used


def _evaluation_space(additional_bytes):
    # Existing runs are already reflected in SizeRemaining; only reserve the new output.
    volume = host_disk()
    if additional_bytes > volume["SizeRemaining"] - volume["reserve_bytes"]:
        raise OSError("exact evaluation merge would invade the host E: 10 GB reserve")


class ScoreCounts:
    """Exact sorted tie counts with bounded RAM and compressed external merges.

    The numeric workspace is bounded by max_bytes, plus fixed codec/runtime
    overhead and caller-owned inputs. Every record is a uint32 score key and two
    int64 counts (20 bytes), independent of empty float32 score regions. Full
    input predictions are never retained. Before each spill or merge, reserve
    the worst-case compressed output while both input runs still exist.
    """

    def __init__(self, max_bytes=128 * 2**20, *, directory=None, check_resources=None):
        if not isinstance(max_bytes, int) or max_bytes < 4096:
            raise ValueError("exact counting workspace must be at least 4096 bytes")
        self.max_bytes = max_bytes
        self.capacity = max_bytes // 64
        self.block_size = max(1, min(1 << 16, max_bytes // 256))
        self.positive = self.negative = self.buffered = 0
        self.buffer, self.runs = [], []
        self.temporary = tempfile.TemporaryDirectory(prefix="score-counts-", dir=directory)
        self.directory = Path(self.temporary.name)
        self.check_resources = check_resources
        self.serial = self.spills = self.merges = self.disk_bytes = self.peak_disk_bytes = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        self.buffer.clear()
        self.runs.clear()
        self.temporary.cleanup()
        self.closed = True

    def _output(self, rows):
        # gzip's deflate overhead is below this conservative bound, including headers.
        raw = rows * _COUNT_DTYPE.itemsize
        required = raw + raw // 1000 + 1024
        if self.check_resources is not None:
            self.check_resources(required)
        elif self.disk_bytes + required >= 2**30:
            _evaluation_space(required)
        if required > shutil.disk_usage(self.directory).free:
            raise OSError("insufficient temporary storage for exact score-count merge")
        self.serial += 1
        return self.directory / f"{self.serial}.gz"

    def _record_output(self, path, rows):
        size = path.stat().st_size
        self.disk_bytes += size
        self.peak_disk_bytes = max(self.peak_disk_bytes, self.disk_bytes)
        return path, rows, size

    def _blocks(self, run):
        with gzip.open(run[0], "rb") as stream:
            while block := stream.read(self.block_size * _COUNT_DTYPE.itemsize):
                if len(block) % _COUNT_DTYPE.itemsize:
                    raise ValueError("truncated exact score-count run")
                yield np.frombuffer(block, _COUNT_DTYPE)

    def _merge(self, left, right):
        path = self._output(left[1] + right[1])
        output = np.empty(self.block_size * 2, _COUNT_DTYPE)
        empty = np.empty(0, _COUNT_DTYPE)
        rows = 0
        a, b = self._blocks(left), self._blocks(right)
        x, y = next(a, empty), next(b, empty)
        try:
            with gzip.open(path, "wb", compresslevel=1) as stream:
                while len(x) and len(y):
                    i, j, used = _merge_counts(x, y, output)
                    stream.write(output[:used].tobytes())
                    rows += used
                    x, y = x[i:], y[j:]
                    if not len(x):
                        x = next(a, empty)
                    if not len(y):
                        y = next(b, empty)
                for current, blocks in ((x, a), (y, b)):
                    stream.write(current.tobytes())
                    rows += len(current)
                    for block in blocks:
                        stream.write(block.tobytes())
                        rows += len(block)
        finally:
            a.close()
            b.close()
        result = self._record_output(path, rows)
        for source in (left, right):
            source[0].unlink()
            self.disk_bytes -= source[2]
        self.merges += 1
        return result

    def _flush(self):
        if not self.buffered:
            return
        ordered = np.concatenate(self.buffer)
        self.buffer.clear()
        self.buffered = 0
        ordered.sort(kind="quicksort")
        records = np.empty(len(ordered), _COUNT_DTYPE)
        used = _compress_counts(ordered, records)
        del ordered
        path = self._output(used)
        with gzip.open(path, "wb", compresslevel=1) as stream:
            for start in range(0, used, self.block_size):
                stream.write(records[start:min(used, start + self.block_size)].tobytes())
        del records
        run = self._record_output(path, used)
        self.spills += 1
        # Binary levels limit the number of live runs and avoid a full-pool rewrite per frame.
        level = 0
        while level < len(self.runs) and self.runs[level] is not None:
            run = self._merge(self.runs[level], run)
            self.runs[level] = None
            level += 1
        if level == len(self.runs):
            self.runs.append(run)
        else:
            self.runs[level] = run

    def add(self, scores, target):
        if self.closed:
            raise ValueError("score counter is closed")
        scores, target = np.asarray(scores), np.asarray(target)
        if scores.ndim != 1 or target.shape != scores.shape:
            raise ValueError("metric records require matching one-dimensional arrays")
        for start in range(0, len(scores), self.block_size):
            packed = packed_scores(scores[start:start + self.block_size],
                                   target[start:start + self.block_size], score_kind="logit")
            if self.buffered + len(packed) > self.capacity:
                self._flush()
            positive = int(np.sum(packed & 1, dtype=np.int64))
            self.positive += positive
            self.negative += len(packed) - positive
            self.buffer.append(packed)
            self.buffered += len(packed)

    def groups(self):
        self._flush()
        present = [run for run in self.runs if run is not None]
        if not present:
            return
        merged = present[0]
        for run in present[1:]:
            merged = self._merge(merged, run)
        self.runs = [merged]
        for records in self._blocks(merged):
            yield records["bits"], records["count"], records["positive"]

    def metrics(self):
        return metrics_from_groups(self.groups(), positive=self.positive,
                                   negative=self.negative, score_kind="logit")

    def storage(self):
        return dict(numeric_workspace_bytes=self.max_bytes,
                    fixed_codec_and_runtime_overhead_excluded=True,
                    peak_temporary_bytes=self.peak_disk_bytes, spills=self.spills,
                    merges=self.merges, record_bytes=_COUNT_DTYPE.itemsize)


def checkpoint_frames(data_root, checkpoint_path, sequence_ids, protocol):
    import torch
    from .model import ScanTransform
    from .train import load_checkpoint
    torch.set_num_threads(4)
    model, saved = load_checkpoint(checkpoint_path)
    model.eval()
    transform = ScanTransform(saved["config"], state=saved["preprocessing"], workers=8)
    for identifier in sequence_ids:
        sequence = STUSequence.open(data_root, protocol=protocol, partition="val",
                                    sequence_id=identifier, label_mode=LabelMode.REQUIRED)
        for source in sequence:
            yield source, model.predict(source, transform)


def prepare_fixed(data_root, selection):
    """Resolve the declared frames and reuse only existing geometric witness records."""
    from .train import select_samples
    datasets, indices = {}, {}
    for split in ("train", "validation"):
        datasets[split] = FrozenDataset(PROJECT_ROOT / "results/synthetic", data_root, split)
        indices[split] = select_samples(datasets[split], selection[split])
        for record in selection[split]:
            if source_identity(datasets[split].sequence[record["frame"]]) != record["source_identity"]:
                raise ValueError("fixed synthetic selection refers to a changed original scan")
    wanted = {(r["identity"], r["source_identity"]) for split in datasets for r in selection[split]}
    cached = json.loads((PROJECT_ROOT / "results/coverage/geometry.json").read_text())
    measured = {(r["world_identity"], r["source_identity"]): r for r in cached["observations"]
                if (r["world_identity"], r["source_identity"]) in wanted}
    geometry = {(r["identity"], r["frame"]): measured[(r["identity"], r["source_identity"])]
                for split in datasets for r in selection[split]
                if (r["identity"], r["source_identity"]) in measured}
    protocol = load_protocol()
    sequences = {key: STUSequence.open(data_root, protocol=protocol, partition="val",
        sequence_id=int(key), label_mode=LabelMode.REQUIRED) for key in selection["val"]}
    for key, frames in selection["val"].items():
        if len(set(frames)) != len(frames) or any(frame not in sequences[key].frame_ids for frame in frames):
            raise ValueError("fixed real selection contains duplicate or nonexistent frames")
    return dict(selection=selection, datasets=datasets, indices=indices, geometry=geometry, sequences=sequences)


def fixed_summary(rows, *, directory=None):
    """Pool complete valid labels; report the common half-weighted detection objective."""
    sums, totals, logits = np.zeros(2), np.zeros(2, np.int64), np.zeros(2)
    with ScoreCounts(directory=directory, check_resources=_evaluation_space) as counts:
        for row in rows:
            scores, target = row["scores"], row["target"]
            valid = target >= 0
            counts.add(scores[valid], target[valid])
            for label in (0, 1):
                values = scores[target == label].astype(np.float64)
                totals[label] += len(values)
                logits[label] += values.sum()
                sums[label] += np.logaddexp(0., (1 - 2 * label) * values).sum()
        result = counts.metrics()
    means = [sums[k] / totals[k] if totals[k] else 0. for k in (0, 1)]
    result.update(detection_loss=.5 * sum(means), normal_loss=means[0], anomaly_loss=means[1],
        mean_score={name: float(logits[k] / totals[k]) if totals[k] else None
                    for k, name in enumerate(("normal", "anomaly"))}, frames=len(rows))
    return result


def threshold_counts(rows, threshold):
    """A missing pooled operating threshold means no positive predictions."""
    counts = dict(tp=0, fp=0, normal=0, anomaly=0)
    for row in rows:
        target, scores = row["target"], row["scores"]
        positive = scores >= threshold if threshold is not None else np.zeros(len(scores), bool)
        counts["tp"] += int(((target == 1) & positive).sum())
        counts["fp"] += int(((target == 0) & positive).sum())
        counts["normal"] += int((target == 0).sum())
        counts["anomaly"] += int((target == 1).sum())
    return dict(counts, threshold=threshold,
        FPR=100 * counts["fp"] / counts["normal"] if counts["normal"] else None,
        recall=100 * counts["tp"] / counts["anomaly"] if counts["anomaly"] else None)


def evaluate_fixed(model, transform, prepared, *, include_real=False, directory=None):
    """Fixed development scopes; callers preserve the training state around evaluation."""
    if model.training:
        raise ValueError("fixed evaluation requires inference mode")
    import time
    started = time.perf_counter()
    result, train_threshold = {}, None
    for split in ("train", "validation"):
        rows, official, witness = [], [], {name: [] for name in ("smooth", "rough", "sparse", "changed_normal")}
        records = prepared["selection"][split]
        for record, index in zip(records, prepared["indices"][split], strict=True):
            frozen = prepared["datasets"][split][index]
            scores = model.predict(frozen.source, transform).restore(frozen.source)
            target, _ = synthetic_targets(frozen)
            row = dict(scores=scores, target=target, role=record["role"], world=record["identity"], frame=record["frame"])
            rows.append(row)
            filtered, eligible = synthetic_targets(frozen, official=True)
            if eligible:
                official.append(dict(row, target=filtered))
            measured = prepared["geometry"].get((record["identity"], record["frame"]))
            if measured is not None:
                for name in witness:
                    slots = (measured["changed_normal_source_slots"] if name == "changed_normal" else
                        measured["contrasts"][name]["normal_source_slots"] + measured["contrasts"][name]["central_anomaly_slots"])
                    slots = np.unique(np.asarray(slots, np.int64))
                    if len(slots):
                        witness[name].append(dict(scores=scores[slots], target=target[slots]))
        full = fixed_summary(rows, directory=directory)
        if split == "train":
            train_threshold = full["recall_at_fpr_limit"]["threshold"]
        roles = {}
        for role in sorted({r["role"] for r in rows}):
            group = [r for r in rows if r["role"] == role]
            roles[role] = dict(curve=fixed_summary(group, directory=directory),
                              at_training_threshold=threshold_counts(group, train_threshold))
        result[split] = dict(full=full, official=fixed_summary(official, directory=directory),
            at_training_threshold=threshold_counts(rows, train_threshold),
            zero_anomaly=threshold_counts([r for r in rows if not np.any(r["target"] == 1)], train_threshold),
            roles=roles, witnesses={name: dict(curve=fixed_summary(group, directory=directory),
                at_training_threshold=threshold_counts(group, train_threshold)) for name, group in witness.items()},
            scope="full valid synthetic labels include zero-anomaly and fewer-than-five-return frames; role curves use their own point pools; all reported group counts use the same 206 full-pool threshold",
            witness_scope="existing measured slot lists only; not all complex normals; absent geometry is unmeasured")
        print(json.dumps(dict(event="fixed_evaluation", split=split, detection_loss=full["detection_loss"],
            AP=full["AP"], FPR95=full["FPR95"], frames=len(rows))), flush=True)
    if include_real:
        rows = []
        for key, frames in prepared["selection"]["val"].items():
            for frame in frames:
                source = prepared["sequences"][key][frame]
                scores, target, eligible = official_frame(source, model.predict(source, transform))
                if not eligible:
                    raise ValueError(f"predeclared real frame {key}/{frame} is no longer official-eligible")
                rows.append(dict(scores=scores, target=target, sequence=int(key)))
        full = fixed_summary(rows, directory=directory)
        own_threshold = full["recall_at_fpr_limit"]["threshold"]
        result["val"] = dict(full=full, at_training_threshold=threshold_counts(rows, train_threshold),
            sequences={key: dict(curve=fixed_summary([r for r in rows if r["sequence"] == int(key)], directory=directory),
                at_global_real_threshold=threshold_counts([r for r in rows if r["sequence"] == int(key)], own_threshold))
                for key in prepared["selection"]["val"]},
            scope="predeclared 152-frame development subset; each complete official point pool retained; not full val19")
        print(json.dumps(dict(event="fixed_evaluation", split="val", detection_loss=full["detection_loss"],
            AP=full["AP"], FPR95=full["FPR95"], frames=len(rows))), flush=True)
    result["seconds"] = time.perf_counter() - started
    return result


def evaluate_synthetic(data_root, checkpoint_path, *, directory=None):
    """Full 201 pool and official filtering, each with its own global score curve."""
    import torch
    from .model import ScanTransform
    from .train import load_checkpoint
    _evaluation_space(0)
    torch.set_num_threads(4)
    model, saved = load_checkpoint(checkpoint_path)
    model.eval()
    transform = ScanTransform(saved["config"], state=saved["preprocessing"], workers=8)
    dataset = FrozenDataset(PROJECT_ROOT / "results/synthetic", data_root, "validation")
    eligible = 0
    with ScoreCounts(directory=directory, check_resources=_evaluation_space) as full, \
            ScoreCounts(directory=directory, check_resources=_evaluation_space) as official:
        for index in range(len(dataset)):
            frozen = dataset[index]
            source = frozen.source
            prediction = model.predict(source, transform)
            scores = prediction.restore(source)
            target, _ = synthetic_targets(frozen)
            valid = target >= 0
            full.add(scores[valid], target[valid])
            target, accepted = synthetic_targets(frozen, official=True)
            if accepted:
                valid = target >= 0
                official.add(scores[valid], target[valid])
                eligible += 1
            if (index + 1) % 100 == 0:
                _evaluation_space(0)
                print(json.dumps(dict(synthetic_frames=index + 1)), flush=True)
        full_metrics = full.metrics()
        full_storage = full.storage()
        full.close()  # Release the full-scope run before the second final merge.
        official_metrics_result = official.metrics()
        official_storage = official.storage()
    remaining_after_cleanup = host_disk()["SizeRemaining"]
    return dict(source_sequence=201, worlds=len({identity for _, identity, _ in dataset.samples}), world_frames=len(dataset),
        eligible_world_frames=eligible, full=full_metrics, official=official_metrics_result,
        full_point_set="all_real_inserted_returns_and_valid_original_normal_targets_without_range_or_frame_filter",
        official_point_set="same_frozen_insertion_targets_with_2.5_to_50_m_and_at_least_5_inserted_return_filter",
        exact_count_storage=dict(full=full_storage, official=official_storage,
            algorithm="compressed_sorted_float32_tie_counts_with_bounded_RAM_and_external_merges",
            new_output_bound="20_bytes_per_input_tie_record_plus_conservative_gzip_overhead; input_runs_retained_until_merge_succeeds",
            disk_policy="shared_host_E_free_space_checked_before_each_output; preserve_10_GB; fail_safely_if_exact_merge_cannot_fit",
            host_E_remaining_after_cleanup=remaining_after_cleanup),
        scope="synthetic_validation_with_repeated_source_backgrounds; not_real_STU_validation",
        checkpoint=str(checkpoint_path.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--predictions", type=Path)
    inputs.add_argument("--checkpoint", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="evaluate complete frozen 201 validation worlds")
    parser.add_argument("--fixed", type=Path, help="evaluate a declared micro-learning checkpoint scope")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, action="append")
    args = parser.parse_args()
    if args.fixed is not None:
        if args.checkpoint is None or args.synthetic or args.sequence is not None:
            parser.error("--fixed requires --checkpoint and its declared development selection")
        import torch
        from .model import ScanTransform
        from .train import load_checkpoint, evaluation_state
        declaration = json.loads(args.fixed.read_text())
        torch.set_num_threads(4)
        model, saved = load_checkpoint(args.checkpoint)
        if saved.get("micro") != declaration:
            parser.error("fixed evaluation declaration differs from the saved experiment")
        prepared = prepare_fixed(args.data_root, declaration["selection"])
        transform = ScanTransform(saved["config"], state=saved["preprocessing"], workers=8)
        args.output.mkdir(parents=True, exist_ok=True)
        with evaluation_state(model):
            result = evaluate_fixed(model, transform, prepared, directory=args.output,
                include_real=saved["step"] in declaration["evaluation"]["real_steps"])
        _atomic_json(args.output / f'{saved["step"]}.json',
            dict(step=saved["step"], checkpoint=str(args.checkpoint.resolve()), **result))
        return
    if args.synthetic:
        if args.checkpoint is None or args.sequence is not None:
            parser.error("--synthetic requires --checkpoint and uses the fixed 201 world split")
        args.output.mkdir(parents=True, exist_ok=True)
        result = evaluate_synthetic(args.data_root, args.checkpoint, directory=args.output)
        _atomic_json(args.output / "synthetic.json", result)
        print(json.dumps(result, indent=2))
        return
    protocol = load_protocol()
    sequences = tuple(args.sequence) if args.sequence else protocol.public_sequence_ids
    if len(set(sequences)) != len(sequences):
        parser.error("duplicate sequences")
    for sequence in sequences:
        protocol.sequence("val", sequence)
    # File-size bounds include every raw slot; the actual official subset is smaller.
    upper = sum(
        p.stat().st_size // 2
        for sequence in sequences
        for p in (args.data_root / "val" / str(sequence) / "velodyne").glob("*.bin")
    )
    if upper >= 2**30:
        disk = host_disk()
        if upper > disk["SizeRemaining"] - disk["reserve_bytes"]:
            raise OSError("evaluation temporary storage would invade the E: reserve")
    args.output.mkdir(parents=True, exist_ok=True)
    last_check = [0]

    def check_resources():
        last_check[0] += 1
        if upper >= 2**30 and last_check[0] % 100 == 0:
            host_disk()

    result, _ = evaluate_frames(
        checkpoint_frames(args.data_root, args.checkpoint, sequences, protocol)
        if args.checkpoint is not None else prediction_frames(args.data_root, args.predictions, sequences, protocol),
        directory=args.output,
        check_resources=check_resources,
    )
    result.update(
        partition="val",
        sequences=list(sequences),
        scope="full_public_validation"
        if set(sequences) == set(protocol.public_sequence_ids)
        else "development_subset",
        **({"checkpoint": str(args.checkpoint.resolve())} if args.checkpoint is not None
           else {"prediction_root": str(args.predictions.resolve())}),
    )
    _atomic_json(args.output / "global.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
