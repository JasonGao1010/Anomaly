"""Official STU evaluation and exact global AP loss attribution for single scans."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from numba import njit

from .data import FramePrediction, FrozenDataset, _atomic_json, host_disk, source_identity, runtime_resources
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


def official_targets(source):
    """Official frame eligibility depends only on the unchanged public labels."""
    if source.labels is None:
        raise ValueError("official evaluation requires semantic labels")
    target = evaluation_targets(source.xyzi[:, :3], source.labels.semantic)
    eligible = (
        int(np.count_nonzero(target == 1))
        >= PointOODMetricsCalculator.min_num_points_to_eval
    )
    return target, eligible


def official_frame(source, prediction):
    """Allow omitted inference only when the official rule excludes the whole frame."""
    target, eligible = official_targets(source)
    if prediction is None and eligible:
        raise ValueError("official-eligible scan requires complete predictions")
    scores = prediction.restore(source) if prediction is not None else None
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


def evaluate_frames(frames, *, directory=None, observe=None, check_resources=None,
                    per_sequence=False, capture=None):
    """Pool the complete official point set with bounded exact tie counting."""
    rows, seen = [], set()
    with ExitStack() as stack:
        counts = stack.enter_context(ScoreCounts(directory=directory, check_resources=_evaluation_space))
        sequences = {}
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
                counts.add(scores[valid], target[valid])
                if per_sequence:
                    key = source.sequence_id
                    if key not in sequences:
                        sequences[key] = stack.enter_context(ScoreCounts(
                            max_bytes=16 * 2**20, directory=directory, check_resources=_evaluation_space))
                    sequences[key].add(scores[valid], target[valid])
                if capture is not None and (source.sequence_id, source.frame_id) in capture:
                    capture[source.sequence_id, source.frame_id] = scores
            if check_resources is not None:
                check_resources()
        result = counts.metrics(observe=observe)
        result["exact_count_storage"] = counts.storage()
        if per_sequence:
            threshold = (result["recall_at_fpr_limit"] or {}).get("threshold")
            high_threshold = (result.get("official_high_recall") or {}).get("threshold")
            result["per_sequence"] = {str(key): dict(curve=value.metrics(),
                at_global_threshold=value.at_threshold(threshold),
                at_global_high_recall=value.at_threshold(high_threshold))
                for key, value in sequences.items()}
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
        self.buffer_copies = 1
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
        records["count"][:used] *= self.buffer_copies
        records["positive"][:used] *= self.buffer_copies
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

    def add(self, scores, target, *, copies=1):
        if self.closed:
            raise ValueError("score counter is closed")
        if not isinstance(copies, int) or copies < 1:
            raise ValueError("score multiplicity must be a positive integer")
        # Identical complete scans retain every world's integer metric weight.
        if copies != self.buffer_copies:
            self._flush()
            self.buffer_copies = copies
        scores, target = np.asarray(scores), np.asarray(target)
        if scores.ndim != 1 or target.shape != scores.shape:
            raise ValueError("metric records require matching one-dimensional arrays")
        for start in range(0, len(scores), self.block_size):
            packed = packed_scores(scores[start:start + self.block_size],
                                   target[start:start + self.block_size], score_kind="logit")
            if self.buffered + len(packed) > self.capacity:
                self._flush()
            positive = int(np.sum(packed & 1, dtype=np.int64))
            self.positive += positive * copies
            self.negative += (len(packed) - positive) * copies
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

    def metrics(self, *, observe=None):
        return metrics_from_groups(self.groups(), positive=self.positive,
                                   negative=self.negative, score_kind="logit", observe=observe)

    def at_threshold(self, threshold):
        tp = fp = 0
        if threshold is not None:
            for bits, count, positive in self.groups():
                selected = bits_score(bits, "logit") >= threshold
                tp += int(positive[selected].sum())
                fp += int((count[selected] - positive[selected]).sum())
        return dict(threshold=threshold, tp=tp, fp=fp, normal=self.negative, anomaly=self.positive,
                    FPR=100 * fp / self.negative if self.negative else None,
                    recall=100 * tp / self.positive if self.positive else None)

    def storage(self):
        return dict(numeric_workspace_bytes=self.max_bytes,
                    fixed_codec_and_runtime_overhead_excluded=True,
                    peak_temporary_bytes=self.peak_disk_bytes, spills=self.spills,
                    merges=self.merges, record_bytes=_COUNT_DTYPE.itemsize)


class EvaluationFrames:
    """Bounded worker preparation; ordered loading preserves complete scan identity."""

    def __init__(self, data_root, sequence_ids, protocol, config, preprocessing):
        self.sequences = {identifier: STUSequence.open(data_root, protocol=protocol, partition="val",
            sequence_id=identifier, label_mode=LabelMode.REQUIRED) for identifier in sequence_ids}
        self.samples = [(identifier, frame) for identifier, sequence in self.sequences.items()
                        for frame in sequence.frame_ids]
        self.config, self.preprocessing, self.transform = config, preprocessing, None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from .model import ScanTransform
        if self.transform is None:
            self.transform = ScanTransform(self.config, state=self.preprocessing, workers=4)
        identifier, frame = self.samples[index]
        source = self.sequences[identifier][frame]
        _, eligible = official_targets(source)
        return source, self.transform(source) if eligible else None


def checkpoint_frames(data_root, checkpoint_path, sequence_ids, protocol):
    import torch
    from torch.utils.data import DataLoader
    from .train import load_checkpoint
    torch.set_num_threads(4)
    model, saved = load_checkpoint(checkpoint_path)
    model.eval()
    dataset = EvaluationFrames(data_root, sequence_ids, protocol, saved["config"], saved["preprocessing"])
    loader = DataLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=1,
        multiprocessing_context="spawn", pin_memory=True,
        generator=torch.Generator().manual_seed(83))
    started, evaluated = time.perf_counter(), 0
    for index, (source, scan) in enumerate(loader, 1):
        yield source, model.predict(source, prepared=scan) if scan is not None else None
        evaluated += int(scan is not None)
        if scan is not None and evaluated % 25 == 0:
            resources = runtime_resources()
            print(f"完整val19 扫描 {index}/{len(dataset)} 有效帧={evaluated} "
                  f"耗时={(time.perf_counter() - started) / 60:.1f}min "
                  f"可用内存={resources['memory_available_bytes'] / 1e9:.1f}GB", flush=True)


def evaluate_validation(data_root, *, checkpoint_path=None, prediction_root=None,
                        sequences=None, directory=None, capture=None):
    protocol = load_protocol()
    sequences = tuple(sequences) if sequences is not None else protocol.public_sequence_ids
    if len(set(sequences)) != len(sequences):
        raise ValueError("duplicate validation sequences")
    for sequence in sequences:
        protocol.sequence("val", sequence)
    if (checkpoint_path is None) == (prediction_root is None):
        raise ValueError("choose one checkpoint or prediction source")
    disk = host_disk()
    _evaluation_space(512 * 2**20)
    if directory is not None:
        Path(directory).mkdir(parents=True, exist_ok=True)
    checked = 0

    def check_resources():
        nonlocal checked
        checked += 1
        if checked % 100 == 0:
            host_disk()

    started = time.perf_counter()
    result, _ = evaluate_frames(
        checkpoint_frames(data_root, checkpoint_path, sequences, protocol) if checkpoint_path is not None
        else prediction_frames(data_root, prediction_root, sequences, protocol),
        directory=directory, check_resources=check_resources, per_sequence=True, capture=capture)
    result.update(partition="val", sequences=list(sequences), seconds=time.perf_counter() - started,
        host_E_before=disk, host_E_after=host_disk(),
        scope="full_public_validation" if set(sequences) == set(protocol.public_sequence_ids) else "development_subset",
        **({"checkpoint": str(Path(checkpoint_path).resolve())} if checkpoint_path is not None
           else {"prediction_root": str(Path(prediction_root).resolve())}))
    return result


def prepare_fixed(data_root, selection, *, synthetic_splits=("train", "validation")):
    """Resolve the declared frames and reuse only existing geometric witness records."""
    from .train import select_samples
    if list(synthetic_splits) not in (["train"], ["train", "validation"]):
        raise ValueError("fixed scopes require train first, with optional validation")
    datasets, indices = {}, {}
    for split in synthetic_splits:
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


def print_metrics(label, result):
    """Keep console metrics short; full precision and point counts remain in JSON outputs."""
    metrics = [(key, result[key], 5 if key == "FPR95" else 3) for key in ("AP", "FPR95", "AUROC")]
    metrics.append(("R@1%", result["recall_at_fpr_limit"]["recall"], 3))
    values = " ".join(f"{key}={value:.{digits}f}%" if value is not None else f"{key}=无定义"
                      for key, value, digits in metrics)
    print(f"{label} {values}", flush=True)


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


def retained_witness_slots(frozen, original, measured):
    groups = dict(sparse=measured["contrasts"]["sparse"]["normal_source_slots"],
                  changed_normal=measured["changed_normal_source_slots"])
    groups = {key: np.unique(np.asarray(value, np.int64)) for key, value in groups.items()}
    slots = np.union1d(*groups.values())
    if np.any((slots < 0) | (slots >= original.slot_count)):
        raise ValueError("normal witness slot is outside the original scan")
    if (frozen.source.slot_count != original.slot_count
            or np.any(frozen.inserted_mask[slots] | frozen.occluded_original_mask[slots]
                      | original.zero_slot_mask[slots] | (frozen.anomaly_target[slots] != 0))
            or np.any(original.labels.semantic_target[slots] == 255)
            or not np.array_equal(original.xyzi[slots], frozen.source.xyzi[slots])
            or not np.array_equal(original.labels.packed[slots], frozen.source.labels.packed[slots])):
        raise ValueError("normal witness changed physical return or valid normal label")
    return groups


def paired_normal_summary(records, threshold):
    """Four transitions use one checkpoint's same pooled threshold on both scans."""
    before = np.concatenate([r["before"] for r in records]) if records else np.empty(0)
    after = np.concatenate([r["after"] for r in records]) if records else np.empty(0)
    if not (np.isfinite(before).all() and np.isfinite(after).all()):
        raise FloatingPointError("nonfinite paired-normal score")
    high0 = before >= threshold if threshold is not None else np.zeros(len(before), bool)
    high1 = after >= threshold if threshold is not None else np.zeros(len(after), bool)
    delta = after - before
    return dict(frames=len(records), normal=len(before), threshold=threshold,
        before_fp=int(high0.sum()), after_fp=int(high1.sum()),
        both_high=int((high0 & high1).sum()), low_to_high=int((~high0 & high1).sum()),
        high_to_low=int((high0 & ~high1).sum()), both_low=int((~high0 & ~high1).sum()),
        mean_before=float(before.mean()) if len(before) else None,
        mean_after=float(after.mean()) if len(after) else None,
        delta=dict(mean=float(delta.mean()), median=float(np.median(delta)),
            p10=float(np.quantile(delta, .1)), p90=float(np.quantile(delta, .9))) if len(delta) else None)


def paired_operating_points(fixed):
    """Reuse identical witness scores at transferred and own pooled thresholds."""
    transfer = {}
    threshold = fixed["train"]["full"]["recall_at_fpr_limit"]["threshold"]
    for split in ("train", "validation"):
        row = fixed[split]
        if row["at_training_threshold"]["threshold"] != threshold:
            raise ValueError("fixed result did not transfer the same 206 threshold")
        transfer[split] = dict(operating_point=row["at_training_threshold"], groups={
            name: paired_normal_summary(row["paired_normals"][name]["records"], threshold)
            for name in ("sparse", "changed_normal")})
    full = fixed["validation"]["full"]
    own = full["recall_at_fpr_limit"]
    return dict(threshold_transfer=transfer, validation_at_own_1pct=dict(
        operating_point=dict(own, normal=full["normal_count"], anomaly=full["anomaly_count"]),
        groups={name: paired_normal_summary(fixed["validation"]["paired_normals"][name]["records"], own["threshold"])
                for name in ("sparse", "changed_normal")}),
        scope="own 201 pooled <=1% FPR is retrospective ranking diagnosis; identical witness slots and scores; no subgroup threshold fitting; unchanged threshold before and after insertion")


def evaluate_normal_pairs(model, transform, prepared, split, threshold, *, inserted_scores=None):
    """Reuse frozen witness identities; save only the small paired score lists."""
    if model.training:
        raise ValueError("paired evaluation requires inference mode")
    records = dict(sparse=[], changed_normal=[])
    dataset = prepared["datasets"][split]
    for record, index in zip(prepared["selection"][split], prepared["indices"][split], strict=True):
        key = (record["identity"], record["frame"])
        measured = prepared["geometry"].get(key)
        if measured is None:
            continue
        original, frozen = dataset.sequence[record["frame"]], dataset[index]
        if source_identity(original) != record["source_identity"]:
            raise ValueError("paired-normal original scan identity changed")
        groups = retained_witness_slots(frozen, original, measured)
        if not any(len(slots) for slots in groups.values()):
            continue
        before = model.predict(original, transform).restore(original)
        after = (inserted_scores[key] if inserted_scores is not None else
                 model.predict(frozen.source, transform).restore(frozen.source))
        for name, slots in groups.items():
            if len(slots):
                records[name].append(dict(world=record["identity"], frame=record["frame"],
                    source_identity=record["source_identity"], slots=slots.tolist(),
                    before=before[slots].astype(np.float64).tolist(), after=after[slots].astype(np.float64).tolist()))
    result = {name: dict(summary=paired_normal_summary(rows, threshold), records=rows)
              for name, rows in records.items()}
    result["scope"] = "same unchanged valid normal file slots; full original and inserted inference; existing partial witness lists may overlap; one 206 threshold for both scans and both splits"
    for name in records:
        summary = result[name]["summary"]
        print(f"正常配对 {split}/{name} n={summary['normal']} "
              f"误报={summary['before_fp']}→{summary['after_fp']}", flush=True)
    return result


def evaluate_fixed(model, transform, prepared, *, include_real=False, directory=None, include_pairs=False,
                   synthetic_scores=None, real_scores=None, include_normalization=False):
    """Fixed development scopes; callers preserve the training state around evaluation."""
    if model.training:
        raise ValueError("fixed evaluation requires inference mode")
    import time
    started = time.perf_counter()
    result, train_threshold = {}, None
    for split in prepared["datasets"]:
        rows, official, witness = [], [], {name: [] for name in ("smooth", "rough", "sparse", "changed_normal")}
        scan_rows, differences = [], []
        compare_statistics = include_normalization and split == "train"
        records = prepared["selection"][split]
        for record, index in zip(records, prepared["indices"][split], strict=True):
            frozen = prepared["datasets"][split][index]
            key = (record["identity"], record["frame"])
            scores = synthetic_scores.get(key) if split == "validation" and synthetic_scores is not None else None
            scan = transform(frozen.source) if compare_statistics else None
            if scores is None:
                prediction = model.predict(frozen.source, prepared=scan) if compare_statistics else model.predict(frozen.source, transform)
                scores = prediction.restore(frozen.source)
            target, _ = synthetic_targets(frozen)
            row = dict(scores=scores, target=target, role=record["role"], world=record["identity"], frame=record["frame"])
            rows.append(row)
            if len(rows) % 8 == 0 or len(rows) == len(records):
                print(f"合成评价 {split} {len(rows)}/{len(records)}", flush=True)
            if compare_statistics:
                from .train import normalization_mode
                # Reuse exact geometry and keep the parent model in inference mode; restore BN buffers per scan.
                with normalization_mode(model, current_scan=True):
                    current = model.predict(frozen.source, prepared=scan).restore(frozen.source)
                scan_rows.append(dict(row, scores=current))
                differences.append(dict(identity=record["identity"], frame=record["frame"], role=record["role"],
                    by_label={str(label): dict(count=int((target == label).sum()),
                        mean=float((current[target == label] - scores[target == label]).astype(np.float64).mean()),
                        absolute_mean=float(np.abs(current[target == label] - scores[target == label]).astype(np.float64).mean()))
                        for label in (0, 1) if np.any(target == label)}))
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
        if compare_statistics:
            controls = {}
            for name, group in (("saved_running_statistics", rows), ("current_scan_statistics", scan_rows)):
                summary = full if group is rows else fixed_summary(group, directory=directory)
                threshold = summary["recall_at_fpr_limit"]["threshold"]
                controls[name] = dict(full=summary,
                    roles={role: threshold_counts([r for r in group if r["role"] == role], threshold) for role in roles})
            result[split]["normalization"] = dict(controls, score_changes=differences)
        if include_pairs:
            result[split]["paired_normals"] = evaluate_normal_pairs(model, transform, prepared, split, train_threshold,
                inserted_scores={(r["world"], r["frame"]): r["scores"] for r in rows})
        print_metrics(f"合成{len(rows)}帧 {split}", full)
    if include_real:
        rows = []
        for key, frames in prepared["selection"]["val"].items():
            for frame in frames:
                source = prepared["sequences"][key][frame]
                scores = real_scores.get((int(key), frame)) if real_scores is not None else None
                if scores is None:
                    scores = model.predict(source, transform).restore(source)
                target, eligible = official_targets(source)
                if not eligible:
                    raise ValueError(f"predeclared real frame {key}/{frame} is no longer official-eligible")
                rows.append(dict(scores=scores, target=target, sequence=int(key)))
            print(f"真实子集 {len(rows)}/{sum(map(len, prepared['selection']['val'].values()))}", flush=True)
        full = fixed_summary(rows, directory=directory)
        own_threshold = full["recall_at_fpr_limit"]["threshold"]
        result["val"] = dict(full=full, at_training_threshold=threshold_counts(rows, train_threshold),
            sequences={key: dict(curve=fixed_summary([r for r in rows if r["sequence"] == int(key)], directory=directory),
                at_global_real_threshold=threshold_counts([r for r in rows if r["sequence"] == int(key)], own_threshold))
                for key in prepared["selection"]["val"]},
            scope="predeclared 152-frame development subset; each complete official point pool retained; not full val19")
        print_metrics(f"真实{len(rows)}帧", full)
    result["seconds"] = time.perf_counter() - started
    return result


def evaluate_scores(model, transform, prepared, *, directory=None, include_normalization=False):
    """Same-forward score decomposition, optionally paired with per-scan BN statistics."""
    from .train import normalization_mode
    if model.training or list(prepared["datasets"]) != ["train"]:
        raise ValueError("score diagnosis requires inference mode and the fixed 206/real scope")
    modes = ["saved_running_statistics"]
    if include_normalization:
        modes.append("current_scan_statistics")
    started = time.perf_counter()
    results, training_thresholds = {}, {}

    def frames(split):
        if split == "train":
            for record, index in zip(prepared["selection"][split], prepared["indices"][split], strict=True):
                frozen = prepared["datasets"][split][index]
                target, _ = synthetic_targets(frozen)
                measured = prepared["geometry"].get((record["identity"], record["frame"]))
                witnesses = {}
                if measured is not None:
                    original = prepared["datasets"][split].sequence[record["frame"]]
                    witnesses = retained_witness_slots(frozen, original, measured)
                yield frozen.source, target, dict(world=record["identity"], frame=record["frame"],
                    source_identity=record["source_identity"], group=record["role"], witnesses=witnesses)
        else:
            for key, ids in prepared["selection"]["val"].items():
                for frame in ids:
                    source = prepared["sequences"][key][frame]
                    target, eligible = official_targets(source)
                    if not eligible:
                        raise ValueError(f"predeclared real frame {key}/{frame} is no longer official-eligible")
                    yield source, target, dict(sequence=int(key), frame=frame,
                        source_identity=source_identity(source), group=key, witnesses={})

    forwards, identities = 0, {}
    for split in ("train", "val"):
        rows = {mode: {name: [] for name in ("base", "relation", "final")} for mode in modes}
        identities[split] = []
        for index, (source, target, record) in enumerate(frames(split), 1):
            scan = transform(source)
            identities[split].append({key: value for key, value in record.items() if key != "witnesses"})
            for mode in modes:
                # Only BN uses scan statistics; no labels enter the forward, and no state accumulates across scans.
                with normalization_mode(model, current_scan=mode == "current_scan_statistics"):
                    output = model.predict(source, prepared=scan, components=True)
                scores = {key: value.restore(source) for key, value in output.items()}
                valid = ~source.zero_slot_mask
                if not np.array_equal((scores["base"] + scores["relation"])[valid], scores["final"][valid]):
                    raise ValueError("same-forward score decomposition changed the final score")
                for name, values in scores.items():
                    rows[mode][name].append(dict(record, scores=values, target=target))
                forwards += 1
            if index % 8 == 0:
                print(json.dumps(dict(event="score_diagnosis", split=split, frames=index,
                    model_forwards=forwards, seconds=time.perf_counter() - started)), flush=True)
        results[split] = {}
        for mode, components in rows.items():
            results[split][mode] = {}
            for name, group in components.items():
                full = fixed_summary(group, directory=directory)
                threshold = full["recall_at_fpr_limit"]["threshold"]
                if split == "train":
                    training_thresholds[mode, name] = threshold
                groups = {}
                for key in sorted({row["group"] for row in group}):
                    subset = [row for row in group if row["group"] == key]
                    groups[key] = dict(curve=fixed_summary(subset, directory=directory),
                        at_global_threshold=threshold_counts(subset, threshold))
                result = dict(full=full, groups=groups,
                    at_training_threshold=threshold_counts(group, training_thresholds[mode, name]))
                if split == "train":
                    result["zero_anomaly"] = threshold_counts(
                        [row for row in group if not np.any(row["target"] == 1)], threshold)
                    result["witnesses"] = {}
                    for witness in ("sparse", "changed_normal"):
                        subset = []
                        for row in group:
                            slots = row["witnesses"].get(witness, np.empty(0, np.int64))
                            if len(slots):
                                subset.append(dict(scores=row["scores"][slots], target=row["target"][slots]))
                        result["witnesses"][witness] = threshold_counts(subset, threshold)
                results[split][mode][name] = result
                print(json.dumps(dict(event="score_metrics", split=split, normalization=mode,
                    component=name, **full)), flush=True)
    return dict(results, identities=identities, model_forwards=forwards,
        seconds=time.perf_counter() - started,
        scope="fixed 206 complete valid synthetic labels and fixed real152 complete official labels; no201; not full val19",
        interpretation="base and relation are post-hoc components of one jointly trained model, not training ablations; relation alone is an uncalibrated logit correction",
        normalization="saved running statistics versus unlabeled current-scan BatchNorm only; restore buffers and RNG after each scan; all other modules stay in inference mode",
        operating_points="each component and mode has its own pooled curve; group counts use that scope's global threshold; no subgroup threshold fitting")


class SyntheticEvaluationFrames:
    """Prepare changed full scans; an unchanged delta is exactly its source scan."""

    def __init__(self, data_root, config, preprocessing):
        self.dataset = FrozenDataset(PROJECT_ROOT / "results/synthetic", data_root, "validation")
        self.config, self.preprocessing, self.transform = config, preprocessing, None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from .model import ScanTransform
        if self.transform is None:
            self.transform = ScanTransform(self.config, state=self.preprocessing, workers=4)
        frozen = self.dataset[index]
        changed = (frozen.inserted_mask | frozen.occluded_original_mask).any()
        return frozen, self.transform(frozen.source) if changed else None


def evaluate_synthetic(data_root, checkpoint_path, *, directory=None, capture=None):
    """Full 201 pool and official filtering, each with its own global score curve."""
    import torch
    from torch.utils.data import DataLoader
    from .model import ScanTransform
    from .train import load_checkpoint
    _evaluation_space(0)
    torch.set_num_threads(4)
    model, saved = load_checkpoint(checkpoint_path)
    model.eval()
    transform = ScanTransform(saved["config"], state=saved["preprocessing"], workers=4)
    dataset = SyntheticEvaluationFrames(data_root, saved["config"], saved["preprocessing"])
    loader = DataLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=1,
        multiprocessing_context="spawn", pin_memory=True, generator=torch.Generator().manual_seed(83))
    # This cache cannot cross checkpoint, calibration or source-dataset boundaries.
    originals, copies = {}, {}
    eligible = predictions = unchanged = zero_frames = few_frames = 0
    started = time.perf_counter()
    with ScoreCounts(directory=directory, check_resources=_evaluation_space) as full, \
            ScoreCounts(directory=directory, check_resources=_evaluation_space) as official:
        for index, (frozen, prepared) in enumerate(loader):
            source = frozen.source
            target, _ = synthetic_targets(frozen)
            anomaly = int((target == 1).sum())
            zero_frames += anomaly == 0
            few_frames += 1 <= anomaly <= 4
            if prepared is None:
                key = source.frame_id
                if key not in originals:
                    scores = model.predict(source, transform).restore(source)
                    originals[key] = scores, target
                    predictions += 1
                scores, original_target = originals[key]
                if not np.array_equal(target, original_target):
                    raise ValueError("unchanged-world targets differ from the cached source")
                copies[key] = copies.get(key, 0) + 1
                unchanged += 1
            else:
                scores = model.predict(source, prepared=prepared).restore(source)
                predictions += 1
                valid = target >= 0
                full.add(scores[valid], target[valid])
            if capture is not None and (frozen.world_identity, source.frame_id) in capture:
                capture[frozen.world_identity, source.frame_id] = scores
            filtered, accepted = synthetic_targets(frozen, official=True)
            if accepted:
                valid = filtered >= 0
                official.add(scores[valid], filtered[valid])
                eligible += 1
            if (index + 1) % 100 == 0:
                _evaluation_space(0)
                print(json.dumps(dict(event="full_synthetic", synthetic_frames=index + 1,
                    model_forwards=predictions, unchanged_world_frames=unchanged,
                    seconds=time.perf_counter() - started, resources=runtime_resources())), flush=True)
        for key, (scores, target) in originals.items():
            valid = target >= 0
            full.add(scores[valid], target[valid], copies=copies[key])
        full_metrics = full.metrics()
        full_storage = full.storage()
        full.close()  # Release the full-scope run before the second final merge.
        official_metrics_result = official.metrics()
        official_storage = official.storage()
    remaining_after_cleanup = host_disk()["SizeRemaining"]
    return dict(source_sequence=201, worlds=len({identity for _, identity, _ in dataset.dataset.samples}), world_frames=len(dataset),
        eligible_world_frames=eligible, full=full_metrics, official=official_metrics_result,
        zero_anomaly_world_frames=int(zero_frames), one_to_four_anomaly_world_frames=int(few_frames),
        model_forwards=predictions, unchanged_world_frames=unchanged, cached_source_frames=len(originals),
        seconds=time.perf_counter() - started,
        full_point_set="all_real_inserted_returns_and_valid_original_normal_targets_without_range_or_frame_filter",
        official_point_set="same_frozen_insertion_targets_with_2.5_to_50_m_and_at_least_5_inserted_return_filter",
        exact_count_storage=dict(full=full_storage, official=official_storage,
            algorithm="compressed_sorted_float32_tie_counts_with_bounded_RAM_and_external_merges",
            new_output_bound="20_bytes_per_input_tie_record_plus_conservative_gzip_overhead; input_runs_retained_until_merge_succeeds",
            disk_policy="shared_host_E_free_space_checked_before_each_output; preserve_10_GB; fail_safely_if_exact_merge_cannot_fit",
            host_E_remaining_after_cleanup=remaining_after_cleanup),
        scope="synthetic_validation_with_repeated_source_backgrounds; not_real_STU_validation",
        checkpoint=str(Path(checkpoint_path).resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--predictions", type=Path)
    inputs.add_argument("--checkpoint", type=Path)
    inputs.add_argument("--pair-thresholds", type=Path, help="reuse a fixed result's saved paired scores at both pooled operating thresholds")
    parser.add_argument("--synthetic", action="store_true", help="evaluate complete frozen 201 validation worlds")
    parser.add_argument("--fixed", type=Path, help="evaluate a declared finite-learning checkpoint scope")
    parser.add_argument("--parent-reference", action="store_true", help="evaluate the declared V2 parent on repaired fixed synthetic frames")
    parser.add_argument("--pairs", type=Path, help="only paired normal witnesses; reuse the threshold from this checkpoint's fixed result")
    parser.add_argument("--scores", action="store_true", help="same-forward base, relation and final scores on the declared 206/real scope")
    parser.add_argument("--scan-statistics", action="store_true", help="with --scores, also compare per-scan BatchNorm statistics")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, action="append")
    args = parser.parse_args()
    if args.parent_reference and (args.fixed is None or args.checkpoint is None or args.scores or args.pairs is not None):
        parser.error("--parent-reference requires only --fixed and its declared parent checkpoint")
    if ((args.scores and (args.fixed is None or args.checkpoint is None or args.pairs is not None))
            or (args.scan_statistics and not args.scores)):
        parser.error("--scores requires --fixed and --checkpoint without --pairs; --scan-statistics requires --scores")
    if args.pair_thresholds is not None:
        if args.synthetic or args.fixed is not None or args.pairs is not None or args.sequence is not None:
            parser.error("--pair-thresholds only reuses one completed fixed result")
        fixed = json.loads(args.pair_thresholds.read_text())
        result = paired_operating_points(fixed)
        args.output.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.output / f'{fixed["step"]}_thresholds.json', dict(
            step=fixed["step"], checkpoint=str((args.pair_thresholds.parent / fixed["checkpoint"]).resolve()),
            reference_metrics=str(args.pair_thresholds.resolve()), **result))
        return
    if args.data_root is None:
        parser.error("prediction evaluation requires --data-root")
    if args.fixed is not None:
        if args.checkpoint is None or args.synthetic or args.sequence is not None:
            parser.error("--fixed requires --checkpoint and its declared development selection")
        import torch
        from .model import ScanTransform
        from .train import load_checkpoint, load_experiment, evaluation_state, experiment_config, validate_stage_parent
        declaration = load_experiment(args.fixed)
        torch.set_num_threads(4)
        model, saved = load_checkpoint(args.checkpoint)
        if args.parent_reference:
            if args.checkpoint.resolve() != (PROJECT_ROOT / declaration["warm_start"]["checkpoint"]).resolve():
                parser.error("parent reference requires the declared warm-start checkpoint")
            validate_stage_parent(saved, experiment_config(declaration), declaration)
        elif saved.get("experiment", saved.get("micro")) != declaration:
            parser.error("fixed evaluation declaration differs from the saved experiment")
        prepared = prepare_fixed(args.data_root, declaration["selection"],
            synthetic_splits=declaration["evaluation"].get("synthetic_splits", ["train", "validation"]))
        transform = ScanTransform(saved["config"], state=saved["preprocessing"], workers=8)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.scores:
            from .train import forward_state
            initial_state = forward_state(model)
            torch.cuda.reset_peak_memory_stats()
            print(json.dumps(dict(event="score_diagnosis_start", checkpoint=str(args.checkpoint),
                step=saved["step"], scan_statistics=args.scan_statistics,
                resources=runtime_resources())), flush=True)
        with evaluation_state(model):
            if args.scores:
                result = evaluate_scores(model, transform, prepared, directory=args.output,
                    include_normalization=args.scan_statistics)
                if any(not torch.equal(value.cpu(), saved["model"][key])
                       for key, value in model.state_dict().items()):
                    raise ValueError("score diagnosis changed saved model parameters or buffers")
                result["model_and_buffers_unchanged"] = True
                result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            elif args.pairs is not None:
                reference = json.loads(args.pairs.read_text())
                if (reference["step"] != saved["step"]
                        or (args.pairs.parent / reference["checkpoint"]).resolve() != args.checkpoint.resolve()):
                    parser.error("paired threshold result must refer to this same checkpoint")
                threshold = reference["train"]["full"]["recall_at_fpr_limit"]["threshold"]
                result = {split: evaluate_normal_pairs(model, transform, prepared, split, threshold)
                          for split in prepared["datasets"]}
                result["reference_metrics"] = str(args.pairs.resolve())
            else:
                result = evaluate_fixed(model, transform, prepared, directory=args.output,
                    include_real=not args.parent_reference and saved["step"] in declaration["evaluation"]["real_steps"],
                    include_pairs=args.parent_reference or saved["step"] in declaration["evaluation"].get("paired_normal_steps", []))
            if args.parent_reference:
                if any(not torch.equal(value.cpu(), saved["model"][key]) for key, value in model.state_dict().items()):
                    raise ValueError("parent reference changed model parameters or buffers")
                result.update(reference_for=declaration, model_and_buffers_unchanged=True,
                              scope="mean1152 on repaired fixed206 source-frame choices; real152 and full val19 retain historical references")
        if args.scores:
            after = forward_state(model)
            for key in ("buffers", "torch_rng", "cuda_rng"):
                torch.testing.assert_close(after[key], initial_state[key], rtol=0, atol=0)
            left, right = after["numpy_rng"], initial_state["numpy_rng"]
            if (after["python_rng"] != initial_state["python_rng"] or left[0] != right[0]
                    or left[2:] != right[2:] or not torch.equal(left[1], right[1])):
                raise ValueError("score diagnosis changed a Python or NumPy random stream")
            result["buffers_and_all_rng_restored"] = True
            result["resources_after"] = runtime_resources()
        suffix = "_scores" if args.scores else "_pairs" if args.pairs is not None else ""
        name = f'{saved["step"]}{suffix}.json'
        _atomic_json(args.output / name,
            dict(step=saved["step"], checkpoint=str(args.checkpoint.resolve()), **result))
        return
    if args.pairs is not None:
        parser.error("--pairs requires --fixed and --checkpoint")
    if args.synthetic:
        if args.checkpoint is None or args.sequence is not None:
            parser.error("--synthetic requires --checkpoint and uses the fixed 201 world split")
        args.output.mkdir(parents=True, exist_ok=True)
        result = evaluate_synthetic(args.data_root, args.checkpoint, directory=args.output)
        _atomic_json(args.output / "synthetic.json", result)
        print(json.dumps(result, indent=2))
        return
    result = evaluate_validation(args.data_root, checkpoint_path=args.checkpoint,
        prediction_root=args.predictions, sequences=args.sequence, directory=args.output)
    _atomic_json(args.output / "global.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
