"""Official STU evaluation and exact global AP loss attribution for single scans."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from numba import njit

from .data import (FramePrediction, FrozenDataset, _atomic_json, binary_target, binary_normal_groups,
                   detection_range, low_support_slots, host_disk, source_identity, runtime_resources)
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
    inside = detection_range(points)
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


def synthetic_targets(frozen, *, official=False, binary_view=None):
    """Keep historical labels separate from the explicitly requested V3 binary view."""
    target = binary_target(frozen.source, frozen.inserted_mask) if binary_view == "official_range_v3" else frozen.anomaly_target
    if not official:
        return target, True
    distance = np.linalg.norm(frozen.source.xyzi[:, :3], axis=1)
    inside = (distance >= PointOODMetricsCalculator.min_eval_distance) & (
        distance <= PointOODMetricsCalculator.max_eval_distance
    )
    target = np.where(inside, target, -1).astype(np.int8)
    eligible = np.count_nonzero(target == 1) >= PointOODMetricsCalculator.min_num_points_to_eval
    return target, bool(eligible)


REAL_GROUPS = "instance_observations_v1"
DISTANCE_GROUPS = ("2.5-10", "10-20", "20-35", "35-50")
SUPPORT_GROUPS = ("low_lt8", "adequate_ge8")
INSTANCE_GROUPS = ("1-4", "5-19", "20-99", "100+", "unknown_id0")
ANOMALY_FIELDS = ("source_slot", "instance", "instance_returns", "distance_m", "low_support", "scores")


def real_group_definition():
    return dict(format=REAL_GROUPS, distance_m=[2.5, 10., 20., 35., 50.],
        distance_intervals="left-closed/right-open;50m included", instance_returns=[1, 5, 20, 100],
        support=dict(radius_m=2., minimum_other_distinct_positions=8, input="complete actual scan before label/range filtering"),
        instance_unit="sequence-frame-nonzero_instance: instance observations, not cross-frame unique objects",
        instance_equal_point_recall="mean of each represented known instance observation's point recall within the selected group; not object detection rate",
        unknown_id0="included in point recall and a separate count group; excluded from instance counts and instance-equal recall",
        point_grouping="instance size counts official-range returns; distance/support belong to each point; one observation may span distance/support groups",
        frame_scope="official eligible frames only; no additional frames with fewer than5 in-range anomalies")


def anomaly_record(source, scores, *, geometry=None):
    """Bind scores and all group memberships to the same official anomaly slots."""
    target, eligible = official_targets(source)
    if not eligible:
        raise ValueError("weak-anomaly groups require an official-eligible frame")
    slots = np.flatnonzero(target == 1).astype(np.int32)
    identity = source_identity(source)
    if geometry is not None:
        if (geometry["source_identity"] != identity or not np.array_equal(geometry["source_slot"], slots)
                or not np.array_equal(geometry["instance"], source.labels.instance[slots])):
            raise ValueError("anomaly geometry cache changed source, labels or physical slots")
        result = {key: value for key, value in geometry.items() if key != "scores"}
    else:
        instance = source.labels.instance[slots]
        _, inverse, counts = np.unique(instance, return_inverse=True, return_counts=True)
        result = dict(sequence=source.sequence_id, frame=source.frame_id, source_identity=identity,
            source_slot=slots, instance=instance.copy(), instance_returns=np.where(instance > 0, counts[inverse], -1).astype(np.int32),
            distance_m=np.linalg.norm(source.xyzi[slots, :3], axis=1),
            low_support=np.isin(slots, low_support_slots(source, 2., 8, workers=1, query_slots=slots)))
    values = np.asarray(scores, np.float32)
    if values.shape != slots.shape or not np.isfinite(values).all():
        raise ValueError("weak-anomaly scores must cover every official anomaly exactly once")
    return dict(result, scores=values.copy())


def anomaly_cache(path, checkpoint, *, records=None):
    """Small resumable cache of anomaly scores and source-bound geometric groups."""
    path, checkpoint = Path(path), Path(checkpoint).resolve()
    stamp = checkpoint.stat().st_mtime_ns
    if records is None:
        if not path.exists():
            return {}
        with np.load(path, allow_pickle=False) as saved:
            if (saved["format"].item() != REAL_GROUPS or saved["checkpoint"].item() != str(checkpoint)
                    or int(saved["checkpoint_mtime_ns"]) != stamp):
                raise ValueError("anomaly cache belongs to different weights or grouping rules")
            arrays = {key: saved[key] for key in ANOMALY_FIELDS}
            offsets, identities = saved["offsets"], saved["source_identity"]
            result = {}
            for i, (sequence, frame) in enumerate(saved["frames"]):
                begin, end = offsets[i:i + 2]
                result[int(sequence), int(frame)] = dict(sequence=int(sequence), frame=int(frame),
                    source_identity=str(identities[i]),
                    **{key: arrays[key][begin:end].copy() for key in ANOMALY_FIELDS})
            return result
    rows = sorted(records, key=lambda row: (row["sequence"], row["frame"]))
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, format=np.asarray(REAL_GROUPS), checkpoint=np.asarray(str(checkpoint)),
                checkpoint_mtime_ns=np.int64(stamp), frames=np.array([(r["sequence"], r["frame"]) for r in rows], np.int32),
                source_identity=np.array([r["source_identity"] for r in rows]),
                offsets=np.r_[0, np.cumsum([len(r["scores"]) for r in rows])],
                **{key: np.concatenate([r[key] for r in rows]) for key in ANOMALY_FIELDS})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def weak_anomaly_metrics(records, full):
    """Pool points, then average within-instance recalls without inventing an object detector."""
    rows = sorted(records, key=lambda row: (row["sequence"], row["frame"]))
    if len({(r["sequence"], r["frame"]) for r in rows}) != len(rows):
        raise ValueError("duplicate frame in weak-anomaly evaluation")
    values = {key: np.concatenate([r[key] for r in rows]) if rows else np.empty(0) for key in ANOMALY_FIELDS}
    frame_index = np.repeat(np.arange(len(rows)), [len(r["scores"]) for r in rows])
    known = values["instance"] > 0
    instance_index = np.full(len(known), -1, np.int32)
    observations, instance_index[known] = np.unique(np.column_stack((frame_index[known], values["instance"][known])),
                                                   axis=0, return_inverse=True)
    masks = {"all": np.ones(len(known), bool)}
    count_bin = np.searchsorted([5, 20, 100], values["instance_returns"], side="right")
    distance_bin = np.searchsorted([10, 20, 35], values["distance_m"], side="right")
    masks.update({f"instance_returns/{name}": known & (count_bin == i) for i, name in enumerate(INSTANCE_GROUPS[:4])})
    masks["instance_returns/unknown_id0"] = ~known
    masks.update({f"distance/{name}": distance_bin == i for i, name in enumerate(DISTANCE_GROUPS)})
    masks.update({f"support/{name}": values["low_support"] == (i == 0) for i, name in enumerate(SUPPORT_GROUPS)})
    points = len(known)
    if points != full["anomaly_count"] or len(rows) != full["eligible_frames"]:
        raise ValueError("weak-anomaly denominator differs from the complete official evaluation")
    operating = {}
    for name in ("official_high_recall", "recall_at_fpr_limit"):
        if full.get(name) is None:
            operating[name] = None
            continue
        threshold = full[name]["threshold"]
        high = values["scores"] >= threshold if threshold is not None else np.zeros(points, bool)
        groups = {}
        for key, use in masks.items():
            selected = use & known
            total = np.bincount(instance_index[selected], minlength=len(observations))
            hit = np.bincount(instance_index[selected & high], minlength=len(observations))
            represented = total > 0
            n, tp = int(use.sum()), int((use & high).sum())
            groups[key] = dict(points=n, tp=tp, fn=n-tp, point_recall=100*tp/n if n else None,
                unknown_instance_points=int((use & ~known).sum()), instance_observations=int(represented.sum()),
                instance_equal_point_recall=float(100*np.mean(hit[represented]/total[represented])) if represented.any() else None)
        if groups["all"]["tp"] != full[name]["tp"]:
            raise ValueError("weak-anomaly decisions differ from the saved global operating point")
        operating[name] = dict(threshold=threshold, global_operating_point=full[name], groups=groups)
    return dict(definition=real_group_definition(), eligible_frames=len(rows), anomaly_points=points,
        instance_observations=len(observations), unknown_instance_points=int((~known).sum()), operating_points=operating)


def normal_group_counts(source, scores, threshold):
    """All normal groups share the full scan's support and one transferred global threshold."""
    slots = np.flatnonzero(binary_target(source) == 0)
    sparse = np.isin(slots, low_support_slots(source, 2., 8, workers=1, query_slots=slots))
    distance = np.searchsorted([10, 20, 35], np.linalg.norm(source.xyzi[slots, :3], axis=1), side="right")
    high = scores[slots] >= threshold
    groups = {"all": np.ones(len(slots), bool)}
    for i, name in enumerate(SUPPORT_GROUPS):
        groups[f"support/{name}"] = sparse == (i == 0)
    for j, name in enumerate(DISTANCE_GROUPS):
        groups[f"distance/{name}"] = distance == j
        for i, support in enumerate(SUPPORT_GROUPS):
            groups[f"support_distance/{support}/{name}"] = (distance == j) & (sparse == (i == 0))
    return {name: np.array([use.sum(), (use & high).sum()], np.int64) for name, use in groups.items()}


def compare_group_metrics(candidate, reference, *, normal=False):
    """Only compare identical scientific groups and their complete point/observation denominators."""
    if candidate["definition"] != reference["definition"]:
        raise ValueError("candidate and1152 use different group definitions")
    if normal:
        pairs = [("normal201", candidate["groups"], reference["groups"])]
        counts, metrics = ("normal",), ("FPR",)
    else:
        if (candidate["operating_points"].keys() != reference["operating_points"].keys()
                or any((value is None) != (reference["operating_points"][name] is None)
                       for name, value in candidate["operating_points"].items())):
            raise ValueError("candidate and1152 global operating points have different support")
        pairs = [(name, value["groups"], reference["operating_points"][name]["groups"])
                 for name, value in candidate["operating_points"].items() if value is not None]
        counts, metrics = ("points", "unknown_instance_points", "instance_observations"), ("point_recall", "instance_equal_point_recall")
    result = {}
    for name, left, right in pairs:
        if left.keys() != right.keys() or any(left[key][count] != right[key][count] for key in left for count in counts):
            raise ValueError("candidate and1152 grouped denominators differ")
        result[name] = {key: {metric + "_change_pp": left[key][metric] - right[key][metric]
                             if left[key][metric] is not None and right[key][metric] is not None else None
                             for metric in metrics} for key in left}
    return result


def evaluate_frames(frames, *, directory=None, observe=None, check_resources=None,
                    per_sequence=False, capture=None, anomaly_records=None, anomaly_geometry=None):
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
                if anomaly_records is not None:
                    geometry = None
                    if anomaly_geometry is not None:
                        geometry = anomaly_geometry.get((source.sequence_id, source.frame_id))
                        if geometry is None:
                            raise ValueError("1152 reference lacks this eligible frame's anomaly geometry")
                    anomaly_records.append(anomaly_record(source, scores[target == 1], geometry=geometry))
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

    def __init__(self, data_root, sequence_ids, protocol, config, preprocessing, *, normal_source=False, skip_frames=()):
        self.normal_source = normal_source
        self.skip_frames = set(skip_frames)
        self.sequences = {identifier: STUSequence.open(data_root, protocol=protocol, partition="train" if normal_source else "val",
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
        eligible = True if self.normal_source else official_targets(source)[1]
        return source, self.transform(source) if eligible and (identifier, frame) not in self.skip_frames else None


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
                        sequences=None, directory=None, capture=None, save_capture=False,
                        real_groups=False, reference=None):
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
    anomalies = [] if real_groups else None
    geometry = anomaly_cache(reference["weak_prediction_cache"], reference["checkpoint"]) if real_groups and reference else None
    result, _ = evaluate_frames(
        checkpoint_frames(data_root, checkpoint_path, sequences, protocol) if checkpoint_path is not None
        else prediction_frames(data_root, prediction_root, sequences, protocol),
        directory=directory, check_resources=check_resources, per_sequence=True, capture=capture,
        anomaly_records=anomalies, anomaly_geometry=geometry)
    result.update(partition="val", sequences=list(sequences), seconds=time.perf_counter() - started,
        host_E_before=disk, host_E_after=host_disk(),
        scope="full_public_validation" if set(sequences) == set(protocol.public_sequence_ids) else "development_subset",
        **({"checkpoint": str(Path(checkpoint_path).resolve())} if checkpoint_path is not None
           else {"prediction_root": str(Path(prediction_root).resolve())}))
    if real_groups:
        result["weak_anomaly"] = weak_anomaly_metrics(anomalies, result)
        if reference is not None:
            result["weak_anomaly"]["versus1152"] = compare_group_metrics(result["weak_anomaly"], reference["weak_anomaly"])
        if checkpoint_path is not None:
            path = Path(checkpoint_path).with_name(Path(checkpoint_path).stem + "_weak.npz")
            anomaly_cache(path, checkpoint_path, records=anomalies)
            result["weak_prediction_cache"] = str(path.resolve())
    if save_capture:
        if checkpoint_path is None or capture is None:
            raise ValueError("persisted fixed predictions require a checkpoint and explicit frame identities")
        path = Path(checkpoint_path).with_name(Path(checkpoint_path).stem + "_real.npz")
        captured_scores(data_root, path, capture=capture)
        result["fixed_predictions"] = str(path.resolve())
    return result


def captured_scores(data_root, path, *, capture=None):
    """Persist/recover fixed full-return scores with physical identities, never score components."""
    path = Path(path)
    saved = np.load(path, allow_pickle=False) if capture is None else None
    try:
        frames = np.array(sorted(capture), np.int32) if capture is not None else saved["frames"]
        readers = {int(key): STUSequence.open(data_root, protocol=load_protocol(), partition="val",
            sequence_id=int(key), label_mode=LabelMode.REQUIRED) for key in set(frames[:, 0])}
        identities, slots, scores, result = [], [], [], {}
        for i, (key, frame) in enumerate(frames):
            source = readers[int(key)][int(frame)]
            identity = source_identity(source)
            if capture is None:
                begin, end = saved["offsets"][i:i + 2]
                rows = saved["source_slot"][begin:end]
                if identity != saved["source_identity"][i] or not np.array_equal(rows, source.real_slots):
                    raise ValueError("saved fixed predictions no longer identify the same physical returns")
                result[int(key), int(frame)] = FramePrediction("val", int(key), int(frame), rows,
                    saved["scores"][begin:end]).restore(source)
            else:
                values = capture[int(key), int(frame)]
                if values is None or not np.isfinite(values[source.real_slots]).all():
                    raise ValueError("complete evaluation did not capture all declared real returns")
                identities.append(identity)
                slots.append(source.real_slots.astype(np.int32))
                scores.append(values[source.real_slots].astype(np.float32))
        if capture is None:
            return result
        np.savez_compressed(path, frames=frames, source_identity=np.array(identities),
            offsets=np.r_[0, np.cumsum(list(map(len, slots)))], source_slot=np.concatenate(slots), scores=np.concatenate(scores))
    finally:
        if saved is not None:
            saved.close()


def evaluate_normal_source(model, transform, data_root, threshold, *, capture=None, reference=None):
    """201 is a pure-normal transfer check at this model's complete-val19 threshold."""
    import torch
    from torch.utils.data import DataLoader
    if model.training or threshold is None or not np.isfinite(threshold):
        raise ValueError("normal-source evaluation requires inference and a finite transferred threshold")
    dataset = EvaluationFrames(data_root, [201], load_protocol(), dict(model=model.config),
                               transform.state_dict(), normal_source=True)
    loader = DataLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=1,
        multiprocessing_context="spawn", pin_memory=True, generator=torch.Generator().manual_seed(83))
    normal, false_positive, native, native_high = 0, 0, 0, 0
    frames, grouped, started = [], {}, time.perf_counter()
    for i, (source, scan) in enumerate(loader, 1):
        scores = model.predict(source, prepared=scan).restore(source)
        target = binary_target(source)
        use = target == 0
        raw2 = ~source.zero_slot_mask & (source.labels.semantic == 2) & detection_range(source.xyzi[:, :3])
        n, fp = int(use.sum()), int((scores[use] >= threshold).sum())
        groups = normal_group_counts(source, scores, threshold)
        for name, counts in groups.items():
            grouped[name] = grouped.get(name, np.zeros(2, np.int64)) + counts
        normal += n
        false_positive += fp
        native += int(raw2.sum())
        native_high += int((scores[raw2] >= threshold).sum())
        frames.append(dict(frame=source.frame_id, source_identity=source_identity(source), normal=n, fp=fp,
                           groups={key: dict(normal=int(value[0]), fp=int(value[1])) for key, value in groups.items()}))
        if capture is not None and (201, source.frame_id) in capture:
            capture[201, source.frame_id] = scores.copy()
        if i % 25 == 0 or i == len(dataset):
            host_disk()
            print(f"原始201 {i}/{len(dataset)} 正常={normal} FP={false_positive} 用时={(time.perf_counter()-started)/60:.1f}min", flush=True)
    result = dict(sequence=201, frames=frames, frame_count=len(frames), normal_count=normal,
        fp=false_positive, FPR=100 * false_positive / normal if normal else None, threshold=threshold,
        definition=dict(real_group_definition(), frame_scope="all682 original201 frames; no anomaly-count eligibility gate"),
        groups={name: dict(normal=int(n), fp=int(fp), FPR=100*int(fp)/int(n) if n else None)
                for name, (n, fp) in grouped.items()},
        native_raw2=dict(points=native, above_threshold=native_high), seconds=time.perf_counter()-started,
        scope="all682 original201 frames; raw0 ignored, native raw2 separately reported, other actual returns normal within2.5-50m; no AP/AUROC/FPR95 or201 threshold fitting")
    if result["groups"]["all"] != dict(normal=normal, fp=false_positive, FPR=result["FPR"]):
        raise ValueError("normal201 grouped counts differ from its complete binary point set")
    if reference is not None:
        identities = lambda rows: [(r["frame"], r["source_identity"]) for r in rows]
        if identities(frames) != identities(reference["frames"]):
            raise ValueError("normal201 and1152 source identities differ")
        result["versus1152"] = compare_group_metrics(result, reference, normal=True)
    return result


def model_selection(directory, parent):
    """Select only joint improvements; retain and disclose other non-dominated tradeoffs."""
    rows = []
    for step in (256, 512, 1024):
        path = Path(directory) / f"{step}_val.json"
        if path.exists():
            value = json.loads(path.read_text())
            if (value["normal_count"], value["anomaly_count"]) != (parent["normal_count"], parent["anomaly_count"]):
                raise ValueError("candidate and parent evaluation denominators differ")
            rows.append(dict(step=step, AP=value["AP"], FPR95=value["FPR95"],
                recall=value["recall_at_fpr_limit"]["recall"], AUROC=value["AUROC"]))
    baseline = dict(step=1152, AP=parent["AP"], FPR95=parent["FPR95"],
                    recall=parent["recall_at_fpr_limit"]["recall"], AUROC=parent["AUROC"])
    def dominates(a, b):
        good = (a["AP"] >= b["AP"], a["FPR95"] <= b["FPR95"], a["recall"] >= b["recall"])
        return all(good) and any(a[k] != b[k] for k in ("AP", "FPR95", "recall"))
    joint = sorted([r for r in rows if dominates(r, baseline)], key=lambda r: (-r["AP"], r["FPR95"], -r["recall"]))
    return dict(parent=baseline, evaluated=rows, jointly_improved=joint, preferred=joint[0] if joint else None,
        nondominated_tradeoffs=[r for r in rows if not dominates(r, baseline) and
                               not any(dominates(other, r) for other in [baseline] + rows)])


def evaluate_parent_anomalies(model, saved, data_root, declaration, full, path):
    """Reuse same-point parent logits first; infer each missing eligible scan only once."""
    import torch
    from torch.utils.data import DataLoader
    checkpoint = PROJECT_ROOT / declaration["warm_start"]["checkpoint"]
    if model.training or Path(full["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("parent groups require inference and its historical complete-val19 reference")
    cached = anomaly_cache(path, checkpoint)
    directory = PROJECT_ROOT / declaration["diagnostic_from"]
    manifest_path = directory / "selection.json"
    legacy, mode, component = {}, None, None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (Path(manifest["reference"]).resolve() != checkpoint.resolve()
                or checkpoint.stat().st_mtime_ns > manifest_path.stat().st_mtime_ns):
            raise ValueError("diagnostic scores no longer identify the historical1152 weights")
        mode, component = manifest["modes"].index("parent"), manifest["components"].index("final")
        legacy = {(r["sequence"], r["frame"]): r for r in manifest["frames"] if (directory / r["file"]).exists()}
    dataset = EvaluationFrames(data_root, full["sequences"], load_protocol(), saved["config"], saved["preprocessing"],
                               skip_frames=set(cached) | set(legacy))
    loader = DataLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=1,
        multiprocessing_context="spawn", pin_memory=True, generator=torch.Generator().manual_seed(83))
    rows, reused, inferred, started = [], dict(anomaly_cache=0, diagnostic_cache=0), 0, time.perf_counter()
    try:
        for source, scan in loader:
            target, eligible = official_targets(source)
            if not eligible:
                continue
            key = (source.sequence_id, source.frame_id)
            if key in cached:
                record = anomaly_record(source, cached[key]["scores"], geometry=cached[key])
                reused["anomaly_cache"] += 1
            elif key in legacy:
                item = legacy[key]
                valid = np.flatnonzero(target >= 0)
                with np.load(directory / item["file"], allow_pickle=False) as values:
                    if (item["source_identity"] != source_identity(source)
                            or not np.array_equal(values["source_slot"], valid)
                            or not np.array_equal(values["target"], target[valid])
                            or not np.array_equal(values["instance"], source.labels.instance[valid])):
                        raise ValueError("diagnostic parent scores changed source, official labels or point identities")
                    scores = values["scores"][mode, component, values["target"] == 1]
                record = anomaly_record(source, scores)
                reused["diagnostic_cache"] += 1
            else:
                # Retain the historical full-query path, including exact threshold-boundary logits.
                scores = model.predict(source, prepared=scan).restore(source)
                record = anomaly_record(source, scores[target == 1])
                inferred += 1
            cached[key] = record
            rows.append(record)
            if len(rows) % 128 == 0:
                anomaly_cache(path, checkpoint, records=cached.values())
                host_disk()
            if len(rows) % 25 == 0 or len(rows) == full["eligible_frames"]:
                print(f"1152真实分组 {len(rows)}/{full['eligible_frames']} 复用={sum(reused.values())} "
                      f"补推理={inferred} 用时={(time.perf_counter()-started)/60:.1f}min", flush=True)
    finally:
        anomaly_cache(path, checkpoint, records=cached.values())
    result = weak_anomaly_metrics(rows, full)
    result.update(reused_frames=reused, model_forwards=inferred, seconds=time.perf_counter()-started)
    return result


def prepare_reference(data_root, declaration, directory=None):
    """Establish the parent on V3 label support once; historical result files stay untouched."""
    import torch
    from .train import load_checkpoint, evaluation_state, validate_stage_parent, experiment_config
    from .model import ScanTransform
    directory = Path(directory) if directory is not None else PROJECT_ROOT / declaration["reference_directory"]
    path = directory / "1152.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["reference_for"] != declaration or result["binary_view"] != "official_range_v3":
            raise ValueError("parent reference uses another V3 declaration or binary view")
        return result
    _evaluation_space(512 * 2**20)
    model, saved = load_checkpoint(PROJECT_ROOT / declaration["warm_start"]["checkpoint"])
    validate_stage_parent(saved, experiment_config(declaration), declaration)
    transform = ScanTransform(saved["config"], state=saved["preprocessing"], workers=4)
    prepared = prepare_fixed(data_root, declaration["selection"], binary_view="official_range_v3")
    parent = json.loads((PROJECT_ROOT / "results/keep/mean/global.json").read_text())
    threshold = parent["official_high_recall"]["threshold"]
    raw_scores = {(201, r["frame"]): None for r in declaration["selection"]["validation"]}
    directory.mkdir(parents=True, exist_ok=True)
    with evaluation_state(model):
        weak_path = directory / "1152_weak.npz"
        weak = evaluate_parent_anomalies(model, saved, data_root, declaration, parent, weak_path)
        normal = evaluate_normal_source(model, transform, data_root, threshold, capture=raw_scores)
        result = evaluate_fixed(model, transform, prepared, include_pairs=True, directory=directory,
                                raw_scores=raw_scores, real_threshold=threshold)
    if any(not torch.equal(value.cpu(), saved["model"][key]) for key, value in model.state_dict().items()):
        raise ValueError("V3 parent reference changed historical model tensors")
    result.update(step=1152, checkpoint=str((PROJECT_ROOT / declaration["warm_start"]["checkpoint"]).resolve()),
        reference_for=declaration, binary_view="official_range_v3", normal201=normal, full_val19=parent,
        weak_anomaly=weak, weak_prediction_cache=str(weak_path.resolve()),
        model_and_buffers_unchanged=True, scope="independent parent1152 on new binary synthetic support and original201; real groups reuse point caches and infer missing eligible scans, keeping historical complete-val19 metrics and thresholds")
    _atomic_json(path, result)
    return result


def prepare_fixed(data_root, selection, *, synthetic_splits=("train", "validation"), binary_view=None):
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
    return dict(selection=selection, datasets=datasets, indices=indices, geometry=geometry, sequences=sequences, binary_view=binary_view)


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


def evaluate_normal_pairs(model, transform, prepared, split, threshold, *, inserted_scores=None, raw_scores=None):
    """Reuse frozen witness identities; save only the small paired score lists."""
    if model.training:
        raise ValueError("paired evaluation requires inference mode")
    records = dict(sparse=[], changed_normal=[])
    raw_scores = {} if raw_scores is None else raw_scores
    dataset = prepared["datasets"][split]
    for record, index in zip(prepared["selection"][split], prepared["indices"][split], strict=True):
        key = (record["identity"], record["frame"])
        measured = prepared["geometry"].get(key)
        if measured is None and prepared.get("binary_view") != "official_range_v3":
            continue
        original, frozen = dataset.sequence[record["frame"]], dataset[index]
        if source_identity(original) != record["source_identity"]:
            raise ValueError("paired-normal original scan identity changed")
        if prepared.get("binary_view") == "official_range_v3":
            normal, sparse, changed = fixed_binary_groups(prepared, frozen, original)
            groups = dict(sparse=sparse, changed_normal=changed)
        else:
            groups = retained_witness_slots(frozen, original, measured)
        if not any(len(slots) for slots in groups.values()):
            continue
        raw_key = (original.sequence_id, original.frame_id)
        if raw_scores.get(raw_key) is None:
            raw_scores[raw_key] = model.predict(original, transform).restore(original)
        before = raw_scores[raw_key]
        after = (inserted_scores[key] if inserted_scores is not None else
                 model.predict(frozen.source, transform).restore(frozen.source))
        for name, slots in groups.items():
            if len(slots):
                records[name].append(dict(world=record["identity"], frame=record["frame"],
                    source_identity=record["source_identity"], slots=slots.tolist(),
                    before=before[slots].astype(np.float64).tolist(), after=after[slots].astype(np.float64).tolist()))
    result = {name: dict(summary=paired_normal_summary(rows, threshold), records=rows)
              for name, rows in records.items()}
    result["scope"] = ("all V3 low-support and disturbed retained normal slots; identical physical return before/after; same supplied threshold" if prepared.get("binary_view") == "official_range_v3" else
        "same unchanged valid normal file slots; full original and inserted inference; existing partial witness lists may overlap; one 206 threshold for both scans and both splits")
    for name in records:
        summary = result[name]["summary"]
        print(f"正常配对 {split}/{name} n={summary['normal']} "
              f"误报={summary['before_fp']}→{summary['after_fp']}", flush=True)
    return result


def fixed_binary_groups(prepared, frozen, original):
    cache = prepared.setdefault("binary_groups", {})
    key = (frozen.world_identity, original.frame_id)
    if key not in cache:
        identity = source_identity(original)
        native = prepared.setdefault("native_sparse", {})
        if identity not in native:
            native[identity] = low_support_slots(original, 2., 8, workers=4)
        cache[key] = binary_normal_groups(frozen, original, native[identity])[0]
    return cache[key]


def evaluate_fixed(model, transform, prepared, *, include_real=False, directory=None, include_pairs=False,
                   synthetic_scores=None, real_scores=None, include_normalization=False, raw_scores=None,
                   real_threshold=None):
    """Fixed development scopes; callers preserve the training state around evaluation."""
    if model.training:
        raise ValueError("fixed evaluation requires inference mode")
    import time
    started = time.perf_counter()
    result, train_threshold = {}, None
    binary_view = prepared.get("binary_view")
    raw_scores = {} if raw_scores is None else raw_scores
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
            target, _ = synthetic_targets(frozen, binary_view=binary_view)
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
            filtered, eligible = synthetic_targets(frozen, official=True, binary_view=binary_view)
            if eligible:
                official.append(dict(row, target=filtered))
            measured = prepared["geometry"].get((record["identity"], record["frame"]))
            if measured is not None:
                for name in witness:
                    if binary_view == "official_range_v3" and name in {"sparse", "changed_normal"}:
                        continue
                    slots = (measured["changed_normal_source_slots"] if name == "changed_normal" else
                        measured["contrasts"][name]["normal_source_slots"] + measured["contrasts"][name]["central_anomaly_slots"])
                    slots = np.unique(np.asarray(slots, np.int64))
                    if len(slots):
                        witness[name].append(dict(scores=scores[slots], target=target[slots]))
            if binary_view == "official_range_v3":
                original = prepared["datasets"][split].sequence[record["frame"]]
                _, sparse, changed = fixed_binary_groups(prepared, frozen, original)
                for name, slots in (("sparse", sparse), ("changed_normal", changed)):
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
                inserted_scores={(r["world"], r["frame"]): r["scores"] for r in rows}, raw_scores=raw_scores)
        if binary_view == "official_range_v3":
            result[split].update(binary_view=binary_view,
                scope="V3 binary labels within2.5-50m; every actual nonzero/non2 raw label normal; inserted returns anomalous; no frame eligibility gate in full pool",
                witness_scope="complete low-support and disturbed post-normal groups; other historical geometry witnesses retain their own identities",
                anomaly_count_groups={f"{low}-{high}": fixed_summary([r for r in rows
                    if low <= np.count_nonzero(r["target"] == 1) < high], directory=directory)
                    for low, high in ((0, 1), (1, 5), (5, 20), (20, 100), (100, np.inf))})
            if real_threshold is not None:
                result[split]["at_real95"] = dict(overall=threshold_counts(rows, real_threshold),
                    normals={name: threshold_counts(witness[name], real_threshold) for name in ("sparse", "changed_normal")},
                    anomaly_count_groups={f"{low}-{high}": threshold_counts([r for r in rows
                        if low <= np.count_nonzero(r["target"] == 1) < high], real_threshold)
                        for low, high in ((0, 1), (1, 5), (5, 20), (20, 100), (100, np.inf))})
                if include_pairs:
                    result[split]["at_real95"]["paired_normals"] = {name: paired_normal_summary(
                        result[split]["paired_normals"][name]["records"], real_threshold) for name in ("sparse", "changed_normal")}
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


def paired_transitions(target, left, right, thresholds):
    """Count decisions on identical slots; each model uses its declared threshold."""
    accepted = [values >= threshold if threshold is not None else np.zeros(len(target), bool)
                for values, threshold in zip((left, right), thresholds, strict=True)]
    result = {}
    for label, name in ((0, "normal"), (1, "anomaly")):
        use = target == label
        a, b = (value[use] for value in accepted)
        result[name] = dict(points=int(use.sum()), both=int((a & b).sum()),
            added=int((~a & b).sum()), removed=int((a & ~b).sum()), neither=int((~a & ~b).sum()))
    return result


def paired_point_details(arrays, index, manifest, precision, required):
    """Localize ranking and decision changes without another model forward."""
    scores, target, xyzi = (arrays[key] for key in ("scores", "target", "xyzi"))
    anomaly, normal = target == 1, target == 0
    distance = np.linalg.norm(xyzi[:, :3], axis=1)
    # A shell cap of 16 cannot change the predicate 'fewer than 8 in all shells'.
    sparse = arrays["shell_counts"].sum(axis=1) < 8
    sequence = np.array([r["sequence"] for r in manifest["frames"]])[index]
    count = np.bincount(index[anomaly], minlength=len(manifest["frames"]))[index]
    groups = {"all": np.ones(len(target), bool)}
    for seq in sorted(set(sequence)):
        own = sequence == seq
        groups[str(seq)] = own
        for low, high in ((2.5, 10), (10, 20), (20, 35), (35, 50.00001)):
            groups[f"{seq}/range/{low}-{high}"] = own & (distance >= low) & (distance < high)
        groups[f"{seq}/support/below8"] = own & sparse
        groups[f"{seq}/support/at_least8"] = own & ~sparse
        groups[f"{seq}/intensity/below0.05"] = own & (xyzi[:, 3] < .05)
        groups[f"{seq}/intensity/at_least0.05"] = own & (xyzi[:, 3] >= .05)
        groups[f"{seq}/frame_anomaly_count/at_least100"] = own & (count >= 100)
    thresholds = manifest["full_validation_thresholds"]
    added = normal & (scores[0, 2] < thresholds[0]) & (scores[1, 2] >= thresholds[1])
    groups["new_normal_false_positives"] = added
    result = {}
    for name, use in groups.items():
        pos, neg = use[anomaly], use & normal
        item = dict(decisions=paired_transitions(target[use], scores[0, 2, use], scores[1, 2, use], thresholds))
        if pos.any():
            item["anomaly"] = dict(points=int(pos.sum()),
                AP_change_pp=float(100 * (precision[1, 2, pos] - precision[0, 2, pos]).sum() / anomaly.sum()),
                mean_precision=[float(precision[i, 2, pos].mean()) for i in range(3)],
                mean_normal_overtakes=[float(required[i, 2, pos].mean() * normal.sum()) for i in range(3)],
                mean_scores=scores[:, :, use & anomaly].mean(axis=2, dtype=np.float64).tolist())
        if neg.any():
            item["normal"] = dict(points=int(neg.sum()), mean_scores=scores[:, :, neg].mean(axis=2, dtype=np.float64).tolist())
        result[name] = item
    cases = []
    positive_rows = np.flatnonzero(anomaly)
    for k, record in enumerate(manifest["frames"]):
        candidates = positive_rows[index[anomaly] == k]
        delta = (precision[1, 2] - precision[0, 2])[index[anomaly] == k]
        selected = [("anomaly_rank_loss", candidates[int(np.argmin(delta))])]
        false_positives = np.flatnonzero((index == k) & added)
        if len(false_positives):
            selected.append(("new_normal_false_positive", false_positives[np.argmax(scores[1, 2, false_positives])]))
        for kind, row in selected:
            cases.append(dict(kind=kind, sequence=record["sequence"], frame=record["frame"],
                source_slot=int(arrays["source_slot"][row]), instance=int(arrays["instance"][row]),
                semantic=int(arrays["semantic"][row]), xyzi=xyzi[row].tolist(), range_m=float(distance[row]),
                shell_counts=arrays["shell_counts"][row].tolist(), scores=scores[:, :, row].tolist()))
    return dict(groups=result, cases=cases,
        definitions="official point range; full-scan distinct-position support at 2m; intensity0.05 and count100 are descriptive diagnostic cuts, not new model thresholds")


def paired_score_summary(directory, manifest):
    """Reuse bounded, lossless scores; attribute AP changes to the same anomalies."""
    directory = Path(directory)
    frames = []
    for record in manifest["frames"]:
        with np.load(directory / record["file"], allow_pickle=False) as saved:
            frames.append({key: saved[key] for key in saved.files})
    arrays = {key: np.concatenate([row[key] for row in frames], axis=2 if key == "scores" else 0)
              for key in frames[0]}
    index = np.concatenate([np.full(len(row["target"]), i, np.int16) for i, row in enumerate(frames)])
    del frames
    scores, target = arrays["scores"], arrays["target"]
    positive = target == 1
    sequence = np.array([r["sequence"] for r in manifest["frames"]])[index]
    old = np.array([r["historical"] for r in manifest["frames"]])[index]
    modes, components = manifest["modes"], manifest["components"]
    groups = {"all": np.ones(len(target), bool), "historical": old, "additional": ~old}
    groups.update({str(key): sequence == key for key in sorted(set(sequence))})
    results, precision, required = {}, np.empty((3, 3, int(positive.sum()))), np.empty((3, 3, int(positive.sum())))
    local_precision = np.empty((3, int(positive.sum())))
    for group, use in groups.items():
        if not use.any():
            continue
        results[group] = {}
        for i, mode in enumerate(modes):
            results[group][mode] = {}
            for j, component in enumerate(components):
                values, labels = scores[i, j, use], target[use]
                observer = APAttribution()
                curve = exact_metrics(np.sort(packed_scores(values, labels, score_kind="logit")),
                    score_kind="logit", observe=observer)
                curve["mean_score"] = {name: float(values[labels == label].mean(dtype=np.float64))
                                       for label, name in ((0, "normal"), (1, "anomaly"))}
                for label, name in ((0, "normal"), (1, "anomaly")):
                    curve[name + "_loss"] = float(np.logaddexp(0., (1 - 2 * label) *
                        values[labels == label].astype(np.float64)).mean())
                results[group][mode][component] = curve
                if group == "all":
                    precision[i, j], required[i, j] = observer.values(values[labels == 1])
                elif group not in ("historical", "additional") and component == "final":
                    local_precision[i, use[positive]] = observer.values(values[labels == 1])[0]
            print_metrics(f"配对诊断 {group} {mode}", results[group][mode]["final"])
    # Mean precision at each complete positive-score tie is exactly pooled AP.
    for i, mode in enumerate(modes):
        for j, component in enumerate(components):
            if not np.isclose(100 * precision[i, j].mean(), results["all"][mode][component]["AP"], atol=1e-10, rtol=0):
                raise ValueError("point attribution does not reproduce exact pooled AP")
    np.savez_compressed(directory / "anomalies.npz", frame_index=index[positive],
        source_slot=arrays["source_slot"][positive], sequence=sequence[positive],
        instance=arrays["instance"][positive], xyzi=arrays["xyzi"][positive],
        precision=precision, required_fpr=required, within_sequence_precision=local_precision)
    thresholds = manifest["full_validation_thresholds"]
    local_thresholds = [results["all"][mode]["final"]["official_high_recall"]["threshold"] for mode in modes]
    transitions = {}
    for group, use in groups.items():
        if not use.any():
            continue
        transitions[group] = dict(
            historical_global95=paired_transitions(target[use], scores[0, 2, use], scores[1, 2, use], thresholds),
            diagnostic95=paired_transitions(target[use], scores[0, 2, use], scores[1, 2, use], local_thresholds[:2]),
            diagnostic95_bn=paired_transitions(target[use], scores[1, 2, use], scores[2, 2, use], local_thresholds[1:]))
    frame_results = []
    for k, record in enumerate(manifest["frames"]):
        use, anomaly_use = index == k, index[positive] == k
        frame_results.append(dict(record,
            decisions=paired_transitions(target[use], scores[0, 2, use], scores[1, 2, use], thresholds),
            AP_change_pp=float(100 * (precision[1, 2, anomaly_use] - precision[0, 2, anomaly_use]).sum() / positive.sum()),
            normal_overtakes_mean=[float(required[i, 2, anomaly_use].mean() * (target == 0).sum()) for i in range(3)]))
    return dict(curves=results, transitions=transitions, frames=frame_results,
        points=paired_point_details(arrays, index, manifest, precision, required),
        normal_count=int((target == 0).sum()), anomaly_count=int(positive.sum()),
        diagnostic95_thresholds=local_thresholds,
        attribution="mean complete-tie precision equals AP; required_fpr counts same-pool normals scoring at least each anomaly",
        boundary="targeted development diagnosis; cannot attribute the full-val19 AP loss or identify a training cause by itself")


def evaluate_diagnostic(data_root, checkpoint, source_directory, directory, *, capture=None, reference=None):
    """Reuse fixed diagnostic identities and parent scores; retain only new final scores."""
    from .model import ScanTransform
    from .train import load_checkpoint, evaluation_state
    checkpoint, source_directory, directory = map(Path, (checkpoint, source_directory, directory))
    reference = Path(reference).resolve() if reference is not None else None
    if reference == checkpoint.resolve():
        reference = None
    manifest = json.loads((source_directory / "selection.json").read_text())
    frames = manifest["frames"]
    result_path = directory / f"{checkpoint.stem}_diagnostic.json"
    score_path = result_path.with_suffix(".npz")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result["frames"] != frames or Path(result["checkpoint"]).resolve() != checkpoint.resolve()
                or result["reference"] != (str(reference) if reference is not None else None)
                or Path(result["source_directory"]).resolve() != source_directory.resolve()
                or checkpoint.stat().st_mtime_ns > result_path.stat().st_mtime_ns):
            raise ValueError("cached diagnostic checkpoint or source identities changed")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    readers = {key: STUSequence.open(data_root, protocol=load_protocol(), partition="val",
        sequence_id=key, label_mode=LabelMode.REQUIRED) for key in {r["sequence"] for r in frames}}
    capture = {} if capture is None else capture
    missing = any(capture.get((r["sequence"], r["frame"])) is None for r in frames)
    model, saved = load_checkpoint(checkpoint) if missing else (None, None)
    transform = ScanTransform(saved["config"], state=saved["preprocessing"], workers=4) if missing else None
    scores, labels, sparse, parent_scores = [], [], [], []
    forwards, started = 0, time.perf_counter()
    for i, record in enumerate(frames, 1):
        source = readers[record["sequence"]][record["frame"]]
        if source_identity(source) != record["source_identity"]:
            raise ValueError("diagnostic source differs from the original fixed frame")
        target, eligible = official_targets(source)
        valid = target >= 0
        if not eligible:
            raise ValueError("fixed diagnostic frame is no longer eligible")
        values = capture.get((record["sequence"], record["frame"]))
        if values is None:
            with evaluation_state(model):
                values, _, _ = official_frame(source, model.predict(source, transform))
            forwards += 1
        with np.load(source_directory / record["file"], allow_pickle=False) as original:
            if (not np.array_equal(np.flatnonzero(valid), original["source_slot"])
                    or not np.array_equal(target[valid], original["target"])):
                raise ValueError("diagnostic labels or return slots differ from the paired reference")
            parent_scores.append(original["scores"][0, 2].copy())
            sparse.append(original["shell_counts"].sum(axis=1) < 8)
        scores.append(np.asarray(values[valid], dtype=np.float32))
        labels.append(target[valid])
        if i % 24 == 0 or i == len(frames):
            print(f"固定诊断 {i}/{len(frames)} 新推理={forwards} 用时={time.perf_counter()-started:.1f}s", flush=True)
    target, support = np.concatenate(labels), np.concatenate(sparse)
    index = np.concatenate([np.full(len(row), i, np.int16) for i, row in enumerate(labels)])
    scores = np.concatenate(scores)
    modes = dict(parent=np.concatenate(parent_scores))
    thresholds = [manifest["full_validation_thresholds"][0]]
    if reference is not None and Path(reference).resolve() != checkpoint.resolve():
        reference_path = Path(reference).with_name(Path(reference).stem + "_diagnostic.json")
        baseline = json.loads(reference_path.read_text())
        if (baseline["frames"] != frames or Path(baseline["checkpoint"]).resolve() != Path(reference).resolve()
                or Path(reference).stat().st_mtime_ns > reference_path.stat().st_mtime_ns):
            raise ValueError("control reference changed its checkpoint or fixed point identities")
        with np.load(reference_path.with_suffix(".npz"), allow_pickle=False) as original:
            modes["A"] = original["scores"]
        thresholds.append(baseline["full_validation_thresholds"][-1])
    modes["candidate"] = scores
    full = json.loads(checkpoint.with_name(checkpoint.stem + "_val.json").read_text())
    if Path(full["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("diagnostic threshold does not belong to this complete validation result")
    thresholds.append(full["official_high_recall"]["threshold"])
    sequence = np.array([r["sequence"] for r in frames])[index]
    positive = target == 1
    groups = {"all": np.ones(len(target), bool)}
    groups.update({str(key): sequence == key for key in sorted(set(sequence))})
    curves, precision, overtakes = {}, {}, {}
    for group, use in groups.items():
        curves[group] = {}
        for name, values in modes.items():
            observer = APAttribution()
            curve = exact_metrics(np.sort(packed_scores(values[use], target[use], score_kind="logit")),
                score_kind="logit", observe=observer)
            for label, kind in ((0, "normal"), (1, "anomaly")):
                selected = values[use & (target == label)].astype(np.float64)
                curve[kind + "_loss"] = float(np.logaddexp(0., (1 - 2 * label) * selected).mean())
            curves[group][name] = curve
            if group == "all":
                precision[name], rate = observer.values(values[positive])
                overtakes[name] = rate * (target == 0).sum()
            if name == "candidate":
                print_metrics(f"固定诊断 {group}", curve)
    # Subgroup attribution uses the same complete diagnostic ranking, not subgroup AP.
    counts = np.bincount(index[positive], minlength=len(frames))[index]
    groups.update(normal_support_below8=support, normal_support_at_least8=~support)
    for key in sorted(set(sequence)):
        for low, high in ((5, 20), (20, 100), (100, np.inf)):
            groups[f"{key}/frame_anomaly_count/{low}-{high}"] = (sequence == key) & (counts >= low) & (counts < high)
    details = {}
    for group, use in groups.items():
        pos = use[positive]
        item = dict(normal_points=int((use & ~positive).sum()), anomaly_points=int(pos.sum()), comparisons={})
        for i, name in enumerate(modes):
            if name == "candidate":
                continue
            item["comparisons"][name] = dict(
                full95=paired_transitions(target[use], modes[name][use], scores[use], [thresholds[i], thresholds[-1]]),
                diagnostic95=paired_transitions(target[use], modes[name][use], scores[use],
                    [curves["all"][name]["official_high_recall"]["threshold"], curves["all"]["candidate"]["official_high_recall"]["threshold"]]),
                AP_change_pp=float(100 * (precision["candidate"][pos] - precision[name][pos]).sum() / positive.sum()))
        if pos.any():
            item["mean_normal_overtakes"] = {name: float(values[pos].mean()) for name, values in overtakes.items()}
        details[group] = item
    np.savez_compressed(score_path, scores=scores)
    result = dict(checkpoint=str(checkpoint.resolve()), reference=str(reference) if reference is not None else None,
        source_directory=str(source_directory.resolve()), frames=frames, curves=curves, groups=details,
        modes=list(modes), full_validation_thresholds=thresholds, model_forwards=forwards,
        reused_frames=len(frames)-forwards, seconds=time.perf_counter()-started,
        normal_count=int((target == 0).sum()), anomaly_count=int(positive.sum()),
        boundary="fixed real development diagnosis; subgroup precision contributions use the same pooled ranking; full-val19 decides whether the checkpoint improves on1152")
    _atomic_json(result_path, result)
    return result


def compare_scores(data_root, checkpoint, reference, declaration, sequences, extra_frames, directory, *, additional=()):
    """Compare the declared parent, V2, and V2 with only parent BN buffers."""
    import torch
    from .model import ScanTransform
    from .train import load_checkpoint, evaluation_state, normalization_mode, validate_stage_parent, experiment_config
    if not sequences or len(set(sequences)) != len(sequences) or extra_frames < 0:
        raise ValueError("comparison needs distinct explicit sequences and a nonnegative extra-frame count")
    directory = Path(directory)
    previous = None
    if directory.exists() and any(directory.iterdir()):
        if not additional:
            raise ValueError("comparison exists; reuse its scores or explicitly add diagnostic frames")
        previous = json.loads((directory / "selection.json").read_text())
        for key, path in (("checkpoint", checkpoint), ("reference", reference)):
            if (Path(previous[key]).resolve() != Path(path).resolve()
                    or Path(path).stat().st_mtime_ns > (directory / "selection.json").stat().st_mtime_ns):
                raise ValueError("cached comparison checkpoint changed")
    model, saved = load_checkpoint(checkpoint)
    parent, historical = load_checkpoint(reference)
    if saved.get("experiment", saved.get("micro")) != declaration:
        raise ValueError("comparison declaration differs from the evaluated checkpoint")
    if Path(reference).resolve() != (PROJECT_ROOT / declaration["warm_start"]["checkpoint"]).resolve():
        raise ValueError("comparison reference is not the declared stage parent")
    validate_stage_parent(historical, experiment_config(declaration), declaration)
    if saved["config"]["model"] != historical["config"]["model"]:
        raise ValueError("shared preprocessing requires identical model input configuration")
    for key, value in saved["preprocessing"].items():
        other = historical["preprocessing"][key]
        if not (torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other):
            raise ValueError(f"parent and evaluated preprocessing differ: {key}")
    with (PROJECT_ROOT / "results/profile/tables/frames.csv").open(encoding="utf-8-sig") as stream:
        eligible = list(csv.DictReader(stream))
    frames = []
    for sequence in sequences:
        original = declaration["selection"]["val"][str(sequence)]
        remaining = sorted(int(row["frame"]) for row in eligible
                           if int(row["sequence"]) == sequence and int(row["state"]) == 3
                           and int(row["frame"]) not in original)
        extra = [remaining[i] for i in np.linspace(0, len(remaining) - 1,
                 min(extra_frames, len(remaining))).astype(int)] if remaining else []
        frames.extend(dict(sequence=sequence, frame=frame, historical=frame in original,
            file=f"{sequence}_{frame}.npz") for frame in sorted(original + extra))
    cached = {(r["sequence"], r["frame"]): r for r in previous["frames"]} if previous else {}
    if previous and not {(r["sequence"], r["frame"]) for r in frames}.issubset(cached):
        raise ValueError("extension changed the original diagnostic selection")
    if previous:
        frames = list(previous["frames"])
    for sequence, frame in additional:
        if sequence not in sequences:
            raise ValueError("additional frame is outside the explicit diagnostic sequences")
        if not any(r["sequence"] == sequence and r["frame"] == frame for r in frames):
            frames.append(dict(sequence=sequence, frame=frame, historical=False, file=f"{sequence}_{frame}.npz"))
    # Scores, point identities and geometry stay below a conservative per-slot bound.
    volume = host_disk()
    peak_bytes = len(frames) * 131072 * 96 + 2**30
    if peak_bytes > volume["SizeRemaining"] - volume["reserve_bytes"]:
        raise OSError("paired scores and metric workspaces would invade the host E: reserve")
    metric_paths = [Path(reference).parent / "global.json", Path(checkpoint).with_name(Path(checkpoint).stem + "_val.json")]
    metrics = [json.loads(path.read_text()) for path in metric_paths]
    for path, record, expected in zip(metric_paths, metrics, (reference, checkpoint), strict=True):
        if Path(record["checkpoint"]).resolve() != Path(expected).resolve():
            raise ValueError(f"historical threshold checkpoint differs: {path}")
    manifest = dict(checkpoint=str(Path(checkpoint).resolve()), reference=str(Path(reference).resolve()),
        steps=[historical["step"], saved["step"]], frames=frames,
        selection="retain historical eight per sequence; add equally spaced eligible frame indices excluding them; chosen before inference",
        modes=["parent", "v2", "v2_parent_bn"], components=["base", "relation", "final"],
        full_validation_metrics=[str(path) for path in metric_paths],
        full_validation_thresholds=[record["official_high_recall"]["threshold"] for record in metrics],
        host_E_before=volume, peak_write_budget_bytes=peak_bytes, resources_before=runtime_resources())
    if previous:
        manifest["previous_scope"] = dict(frames=len(previous["frames"]),
            model_forwards=previous["model_forwards"], inference_seconds=previous["inference_seconds"],
            curves=json.loads((directory / "comparison.json").read_text())["curves"])
        manifest["pilot"] = previous["pilot"]
        manifest["additional_frames"] = [list(pair) for pair in additional]
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(directory / "selection.json", manifest)
    transform = ScanTransform(saved["config"], state=saved["preprocessing"], workers=4)
    readers = {key: STUSequence.open(data_root, protocol=load_protocol(), partition="val",
               sequence_id=key, label_mode=LabelMode.REQUIRED) for key in sequences}
    started = time.perf_counter()
    forwards = 0
    torch.cuda.reset_peak_memory_stats()
    with evaluation_state(parent), evaluation_state(model):
        for k, record in enumerate(frames, 1):
            source = readers[record["sequence"]][record["frame"]]
            target, valid_frame = official_targets(source)
            if not valid_frame:
                raise ValueError("selected diagnostic frame no longer satisfies official eligibility")
            if (record["sequence"], record["frame"]) in cached:
                if source_identity(source) != record["source_identity"]:
                    raise ValueError("cached diagnostic source changed")
                continue
            scan, valid = transform(source), target >= 0
            record["source_identity"] = source_identity(source)
            scores = []
            for evaluated, buffers in ((parent, None), (model, None), (model, parent)):
                with normalization_mode(evaluated, reference=buffers):
                    output = evaluated.predict(source, prepared=scan, components=True)
                parts = np.stack([output[name].restore(source)[valid] for name in manifest["components"]])
                if not np.array_equal(parts[0] + parts[1], parts[2]):
                    raise ValueError("same-forward components do not reconstruct the final float32 scores")
                scores.append(parts)
                forwards += 1
            if "pilot" not in manifest:
                plain = model.predict(source, prepared=scan).restore(source)[valid]
                if not np.array_equal(plain, scores[1][2]):
                    raise ValueError("component capture differs from the authoritative prediction")
                manifest["pilot"] = dict(sequence=record["sequence"], frame=record["frame"],
                    ordinary_prediction_exactly_equal=True, extra_forwards=1,
                    elapsed_seconds=time.perf_counter() - started, resources=runtime_resources())
                forwards += 1
            real_valid = valid[source.real_slots]
            neighbors = scan["neighbors"][scan["geometry_inverse"]][real_valid].numpy()
            shells = (neighbors.reshape(len(neighbors), 3, -1) >= 0).sum(axis=2).astype(np.uint8)
            np.savez_compressed(directory / record["file"], scores=np.stack(scores),
                target=target[valid].astype(np.int8), source_slot=np.flatnonzero(valid).astype(np.int32),
                xyzi=source.xyzi[valid], semantic=source.labels.semantic[valid], instance=source.labels.instance[valid],
                shell_counts=shells, condition=scan["condition"][real_valid].numpy())
            print(f"配对推理 {k}/{len(frames)} 序列={record['sequence']} 帧={record['frame']} 本次前向={forwards} 用时={time.perf_counter()-started:.1f}s", flush=True)
            if k % 24 == 0:
                host_disk()
        for evaluated, state in ((parent, historical), (model, saved)):
            if any(not torch.equal(value.cpu(), state["model"][key]) for key, value in evaluated.state_dict().items()):
                raise ValueError("paired inference changed checkpoint parameters or buffers")
    manifest.update(inference_seconds=time.perf_counter() - started + (previous["inference_seconds"] if previous else 0),
                    model_forwards=forwards + (previous["model_forwards"] if previous else 0), reused_frames=len(cached),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), model_and_buffers_unchanged=True)
    _atomic_json(directory / "selection.json", manifest)
    result = paired_score_summary(directory, manifest)
    result.update(inference=manifest, seconds=time.perf_counter() - started,
                  host_E_after=host_disk(), resources_after=runtime_resources())
    _atomic_json(directory / "comparison.json", result)
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
    binary_view = saved["config"]["training"].get("binary_view")
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
            target, _ = synthetic_targets(frozen, binary_view=binary_view)
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
            filtered, accepted = synthetic_targets(frozen, official=True, binary_view=binary_view)
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
        binary_view=binary_view,
        full_point_set=("V3_binary_actual_returns_within2.5-50m_without_frame_filter" if binary_view == "official_range_v3" else
                        "all_real_inserted_returns_and_valid_original_normal_targets_without_range_or_frame_filter"),
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
    parser.add_argument("--arm", help="arm explicitly listed in the fixed experiment declaration")
    parser.add_argument("--parent-reference", action="store_true", help="evaluate the declared parent on the stage's repaired frames and binary view")
    parser.add_argument("--pairs", type=Path, help="only paired normal witnesses; reuse the threshold from this checkpoint's fixed result")
    parser.add_argument("--scores", action="store_true", help="same-forward base, relation and final scores on the declared 206/real scope")
    parser.add_argument("--scan-statistics", action="store_true", help="with --scores, also compare per-scan BatchNorm statistics")
    parser.add_argument("--compare", type=Path, help="with --scores, compare this stage's parent and a parent-BN-only intervention on explicit real sequences")
    parser.add_argument("--extra-frames", type=int, default=16, help="additional equally spaced eligible frames per comparison sequence")
    parser.add_argument("--frame", action="append", default=[], help="explicit sequence/frame to add to an existing paired comparison")
    parser.add_argument("--diagnostic-from", type=Path, help="reuse this paired diagnosis's fixed frames and parent scores")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, action="append")
    args = parser.parse_args()
    if args.arm is not None and args.fixed is None:
        parser.error("--arm requires --fixed")
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
    if args.diagnostic_from is not None:
        if (args.checkpoint is None or args.fixed is not None or args.synthetic or args.scores
                or args.compare is not None or args.sequence is not None or args.pairs is not None):
            parser.error("--diagnostic-from requires only a checkpoint and existing fixed diagnosis")
        import torch
        torch.set_num_threads(4)
        evaluate_diagnostic(args.data_root, args.checkpoint, args.diagnostic_from, args.output)
        return
    if args.compare is not None:
        if not args.scores or args.scan_statistics or args.synthetic or args.parent_reference or not args.sequence:
            parser.error("--compare requires --scores --fixed --checkpoint and explicit --sequence values")
        import torch
        from .train import load_experiment
        torch.set_num_threads(4)
        try:
            additional = [tuple(map(int, value.split("/"))) for value in args.frame]
            if any(len(pair) != 2 for pair in additional):
                raise ValueError
        except ValueError:
            parser.error("--frame must be sequence/frame")
        compare_scores(args.data_root, args.checkpoint, args.compare, load_experiment(args.fixed, args.arm),
                       args.sequence, args.extra_frames, args.output, additional=additional)
        return
    if args.frame:
        parser.error("--frame requires --compare")
    if args.fixed is not None:
        if args.checkpoint is None or args.synthetic or args.sequence is not None:
            parser.error("--fixed requires --checkpoint and its declared development selection")
        import torch
        from .model import ScanTransform
        from .train import load_checkpoint, load_experiment, evaluation_state, experiment_config, validate_stage_parent
        declaration = load_experiment(args.fixed, args.arm)
        torch.set_num_threads(4)
        if args.parent_reference and declaration["format"] == "ajae-v3-learning":
            if args.checkpoint.resolve() != (PROJECT_ROOT / declaration["warm_start"]["checkpoint"]).resolve():
                parser.error("V3 parent reference requires its declared1152 checkpoint")
            prepare_reference(args.data_root, declaration, args.output)
            return
        model, saved = load_checkpoint(args.checkpoint)
        if args.parent_reference:
            if args.checkpoint.resolve() != (PROJECT_ROOT / declaration["warm_start"]["checkpoint"]).resolve():
                parser.error("parent reference requires the declared warm-start checkpoint")
            validate_stage_parent(saved, experiment_config(declaration), declaration)
        elif saved.get("experiment", saved.get("micro")) != declaration:
            parser.error("fixed evaluation declaration differs from the saved experiment")
        prepared = prepare_fixed(args.data_root, declaration["selection"],
            synthetic_splits=declaration["evaluation"].get("synthetic_splits", ["train", "validation"]),
            binary_view=declaration["evaluation"].get("binary_view"))
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
                threshold = (reference["train"]["at_real95"]["overall"]["threshold"] if declaration["format"] == "ajae-v3-learning" else
                             reference["train"]["full"]["recall_at_fpr_limit"]["threshold"])
                result = {split: evaluate_normal_pairs(model, transform, prepared, split, threshold)
                          for split in prepared["datasets"]}
                if declaration["format"] == "ajae-v3-learning":
                    result["overall_operating_points"] = {split: reference[split]["at_real95"]["overall"]
                                                          for split in prepared["datasets"]}
                result["reference_metrics"] = str(args.pairs.resolve())
            else:
                raw_scores, real_threshold, normal201 = {}, None, None
                if declaration["format"] == "ajae-v3-learning":
                    full = json.loads(args.checkpoint.with_name(args.checkpoint.stem + "_val.json").read_text())
                    real_threshold = full["official_high_recall"]["threshold"]
                    raw_scores = {(201, r["frame"]): None for r in declaration["selection"]["validation"]}
                    reference = prepare_reference(args.data_root, declaration)
                    normal201 = evaluate_normal_source(model, transform, args.data_root, real_threshold, capture=raw_scores,
                                                       reference=reference["normal201"])
                result = evaluate_fixed(model, transform, prepared, directory=args.output,
                    include_real=not args.parent_reference and saved["step"] in declaration["evaluation"]["real_steps"],
                    include_pairs=args.parent_reference or saved["step"] in declaration["evaluation"].get("paired_normal_steps", []),
                    raw_scores=raw_scores, real_threshold=real_threshold,
                    real_scores=captured_scores(args.data_root, args.checkpoint.with_name(args.checkpoint.stem + "_real.npz"))
                        if declaration["format"] == "ajae-v3-learning" else None)
                if normal201 is not None:
                    result["normal201"] = normal201
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
        prediction_root=args.predictions, sequences=args.sequence, directory=args.output, real_groups=True)
    _atomic_json(args.output / "global.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
