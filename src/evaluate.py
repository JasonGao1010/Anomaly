"""Official STU evaluation and exact global AP loss attribution for single scans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from .data import FramePrediction, _atomic_json, host_disk
from .protocol import load_protocol
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
):
    """Exact point pooling with bounded RAM; ordered is an ascending uint64 array.

    No score quantization is used. ROC drops the same collinear threshold nodes
    as sklearn's default roc_curve before applying the upstream strict TPR > .95.
    """
    if not 0 <= fpr_limit <= 1 or (prevalence is not None and not 0 < prevalence < 1):
        raise ValueError("invalid diagnostic prevalence or FPR limit")
    positive = sum(
        int(np.sum(ordered[start : start + chunk_size] & 1, dtype=np.int64))
        for start in range(0, len(ordered), chunk_size)
    )
    negative = len(ordered) - positive
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
    operating_point = dict(recall=0.0, FPR=0.0, threshold=None, tp=0, fp=0)
    fpr95 = None
    previous = None
    high_recall = None
    first = True
    for bits, counts, pos in score_groups(ordered, chunk_size):
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
        feasible = np.flatnonzero(fpr <= fpr_limit)
        if len(feasible) and tps[feasible[-1]] > operating_point["tp"]:
            # Complete score ties are indivisible, including ties across read chunks.
            index = int(np.searchsorted(tps, tps[feasible[-1]]))
            operating_point = dict(
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
    if fpr95 is None:
        fpr95 = previous[3]  # The final ROC threshold is always retained.
        high_recall = previous[5]
    result.update(AP=ap * 100, AUROC=area * 100, FPR95=fpr95 * 100)
    result["recall_at_fpr_limit"] = operating_point
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
            np.empty(0, np.uint64), prevalence=prevalence, score_kind=score_kind
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
            ordered, prevalence=prevalence, score_kind=score_kind, observe=observe
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, action="append")
    args = parser.parse_args()
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
        prediction_frames(args.data_root, args.predictions, sequences, protocol),
        directory=args.output,
        check_resources=check_resources,
    )
    result.update(
        partition="val",
        sequences=list(sequences),
        scope="full_public_validation"
        if set(sequences) == set(protocol.public_sequence_ids)
        else "development_subset",
        prediction_root=str(args.predictions.resolve()),
    )
    _atomic_json(args.output / "global.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
