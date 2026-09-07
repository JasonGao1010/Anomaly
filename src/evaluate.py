"""Zero-update paired diagnostics and full frozen 201 development validation."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import io
import zlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import tempfile
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .data import FrozenWindowDataset, PredictionBatch, WindowPartition, _atomic_json
from .model import AJAE, joint_voxelize
from .protocol import PROJECT_ROOT, load_protocol
from .scene import STUSequence, LabelMode
from .train import balanced_loss, fixed_check, host_disk, score_distribution
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


CURRENT_FRAMES = (
    15,
    43,
    71,
    99,
    127,
    155,
    183,
    211,
    239,
    267,
    295,
    323,
    351,
    379,
    407,
    435,
    463,
    491,
    519,
    547,
    575,
    603,
    650,
)
NORMAL_THRESHOLD = 0.5
CHECK_SEED = 23


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
):
    """Exact point pooling with bounded RAM; ordered is an ascending uint64 array.

    No score quantization is used. ROC drops the same collinear threshold nodes
    as sklearn's default roc_curve before applying the upstream strict TPR > .95.
    """
    if prevalence is not None and not (0 < prevalence < 1 and 0 <= fpr_limit <= 1):
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
    first = True
    for bits, counts, pos in score_groups(ordered, chunk_size):
        neg = counts - pos
        tps = tp + np.cumsum(pos, dtype=np.int64)
        fps = fp + np.cumsum(neg, dtype=np.int64)
        recall, fpr = tps / positive, fps / negative
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
            p, n, r, f, was_first = previous
            if (was_first or p != pos[0] or n != neg[0]) and r > 0.95:
                fpr95 = f
        keep = (pos[:-1] != pos[1:]) | (neg[:-1] != neg[1:])
        if first and len(keep):
            keep[0] = True
        eligible = np.flatnonzero(keep & (recall[:-1] > 0.95))
        if fpr95 is None and len(eligible):
            fpr95 = float(fpr[eligible[0]])
        previous = (
            int(pos[-1]),
            int(neg[-1]),
            float(recall[-1]),
            float(fpr[-1]),
            first and len(pos) == 1,
        )
        tp, fp = int(tps[-1]), int(fps[-1])
        first = False
    if fpr95 is None:
        fpr95 = previous[3]  # The final ROC threshold is always retained.
    result.update(AP=ap * 100, AUROC=area * 100, FPR95=fpr95 * 100)
    result["recall_at_fpr_limit"] = operating_point
    if prevalence is not None:
        result.update(
            standardized_AP=standardized_ap * 100,
            recall_at_fpr_limit=operating_point,
        )
    return result


def normal_files(paths, *, float32=False, score_kind="probability"):
    """Select exact float32 order statistics by bytes, with no full sorting copy."""
    dtype = np.uint32 if float32 else np.uint64
    itemsize = np.dtype(dtype).itemsize
    sizes = [path.stat().st_size for path in paths]
    if any(size % itemsize for size in sizes):
        raise ValueError("truncated normal score records")
    count = sum(sizes) // itemsize
    if not count:
        return normal_statistics(np.empty(0), score_kind=score_kind)
    positions = [(count - 1) * q for q in (0.5, 0.95)]
    ranks = sorted({int(f(p)) for p in positions for f in (np.floor, np.ceil)})
    # Positive finite float32 bit order equals numerical order; no score is quantized.
    nodes = {0: {rank: rank for rank in ranks}}
    high = 0
    for shift in (24, 16, 8, 0):
        histograms = {prefix: np.zeros(256, dtype=np.int64) for prefix in nodes}
        for path in paths:
            with path.open("rb") as stream:
                while len(block := np.fromfile(stream, dtype=dtype, count=1 << 20)):
                    bits = (
                        score_bits(block.view(np.float32), score_kind)
                        if float32
                        else (block >> 1).astype(np.uint32)
                    )
                    if shift == 24:
                        high += int(
                            np.count_nonzero(
                                bits
                                >= score_bits(
                                    np.array(
                                        [0 if score_kind == "logit" else 0.5],
                                        np.float32,
                                    ),
                                    score_kind,
                                )[0]
                            )
                        )
                    for prefix, hist in histograms.items():
                        selected = (
                            bits
                            if shift == 24
                            else bits[(bits >> (shift + 8)) == prefix]
                        )
                        hist += np.bincount((selected >> shift) & 255, minlength=256)
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        next_nodes = {}
        for prefix, entries in nodes.items():
            cumulative = np.cumsum(histograms[prefix])
            for rank, relative in entries.items():
                byte = int(np.searchsorted(cumulative, relative, side="right"))
                before = int(cumulative[byte - 1]) if byte else 0
                next_nodes.setdefault((prefix << 8) | byte, {})[rank] = (
                    relative - before
                )
        nodes = next_nodes
    values = {
        rank: bits_score(np.uint32(bits), score_kind)
        for bits, entries in nodes.items()
        for rank in entries
    }
    quantiles = []
    for q, position in zip((0.5, 0.95), positions):
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        pair = np.array([values[lo], values[hi]], dtype=np.float32)
        quantiles.append(
            float(np.median(pair) if q == 0.5 else np.quantile(pair, position - lo))
        )
    suffix = "0" if score_kind == "logit" else "0_5"
    return dict(
        point_count=count,
        median=quantiles[0],
        p95=quantiles[1],
        **{f"count_ge_{suffix}": high, f"fraction_ge_{suffix}": high / count},
    )


def pooled_files(
    paths,
    *,
    normal=False,
    normal_float32=False,
    ranges=None,
    prevalence=None,
    score_kind="probability",
):
    """Sort exact records on disk, then reduce them in bounded chunks."""
    if normal:
        return normal_files(paths, float32=normal_float32, score_kind=score_kind)
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
        result = exact_metrics(ordered, prevalence=prevalence, score_kind=score_kind)
        del ordered
    return result


def anomaly_losses(logits, target, current, official_target):
    """Count and sum observation-level losses; history occurrences are not fused."""
    positive = target == 1
    current = torch.as_tensor(current.copy(), device=logits.device)
    official = torch.zeros_like(positive)
    official[current] = torch.as_tensor(official_target == 1, device=logits.device)
    masks = {
        "all": positive,
        "history": positive & ~current,
        "current": positive & current,
        "current_official_range": official,
    }
    values = F.binary_cross_entropy_with_logits(
        logits.float(), torch.ones_like(logits, dtype=torch.float32), reduction="none"
    )
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        total = float(values[mask].double().sum())
        result[name] = {
            "point_count": count,
            "loss_sum": total,
            "point_mean": total / count if count else None,
        }
    return result


def predict_window(model, window, inputs, seed, *, split_losses=False):
    target = torch.tensor(window.labels.anomaly_target, device=inputs.features.device)
    with fixed_check(model, seed):
        begin = time.perf_counter()
        nre = getattr(model, "score_kind", "probability") == "logit"
        if nre:
            evidence = model(window, inputs=inputs, return_evidence=True)
            logits = evidence.score.float()
        else:
            logits = model(window, inputs=inputs).float()
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - begin
        loss, parts = balanced_loss(logits, target)
        losses = {
            key: float(parts[key]) if key in parts else None
            for key in ("normal", "anomaly")
        }
        losses["total"] = float(loss)
        if nre and window.labels.semantic_target is not None:
            from .data import normal_group_targets

            current = torch.tensor(window.current_mask, device=logits.device)
            truth = torch.tensor(
                normal_group_targets(window.labels), device=logits.device
            )
            use = current & (truth >= 0)
            prediction = model.head.active_groups[evidence.normal_logits.argmax(1)]
            losses["semantic_confusion"] = (
                torch.bincount(20 * truth[use] + prediction[use], minlength=400)
                .reshape(20, 20)
                .cpu()
                .tolist()
            )
        scopes = None
        if split_losses:
            current = window.current_mask
            official = evaluation_targets(
                window.points.coordinates[current], window.labels.semantic[current]
            )
            scopes = anomaly_losses(logits, target, current, official)
        scores = (logits if nre else logits.sigmoid()).cpu().numpy()
    return scores, losses, scopes, inference_seconds


def select_samples(pool):
    """Select the earlier median legal window without inspecting any labels."""
    if (
        pool.name != "validation"
        or pool.source_sequence_id != 201
        or len(pool.segments) != 23
    ):
        raise ValueError(
            "diagnostic requires the frozen 23-segment 201 validation pool"
        )
    result, offset = [], 0
    for segment in range(len(pool.segments)):
        starts = pool.window_starts(segment)
        middle = (len(starts) - 1) // 2
        start = starts[middle]
        result.append(
            {
                "segment_index": segment,
                "dataset_index": offset + middle,
                "window_start": start,
                "current_frame": start + 4,
                "frame_ids": list(range(start, start + 5)),
                "check_seed": CHECK_SEED + segment,
                "synthetic_sequence_id": pool.synthetic_sequence_id(0),
                "normal_sequence_id": "train/201",
            }
        )
        offset += len(starts)
    if tuple(item["current_frame"] for item in result) != CURRENT_FRAMES:
        raise ValueError(
            "validation segment boundaries differ from the requested sample list"
        )
    return result


def evaluation_targets(points, semantic):
    """The official point filter, without its anomaly-frame eligibility gate."""
    distance = np.linalg.norm(points, axis=1)
    inside = (distance >= PointOODMetricsCalculator.min_eval_distance) & (
        distance <= PointOODMetricsCalculator.max_eval_distance
    )
    target = np.where(semantic == 0, -1, np.where(semantic == 2, 1, 0))
    return np.where(inside, target, -1)


def normal_statistics(scores, *, score_kind="probability"):
    suffix = "0" if score_kind == "logit" else "0_5"
    if not len(scores):
        return {
            "point_count": 0,
            "median": None,
            "p95": None,
            f"fraction_ge_{suffix}": None,
            f"count_ge_{suffix}": 0,
        }
    count = int(
        np.count_nonzero(scores >= (0 if score_kind == "logit" else NORMAL_THRESHOLD))
    )
    return {
        "point_count": len(scores),
        "median": float(np.median(scores)),
        "p95": float(np.quantile(scores, 0.95)),
        f"count_ge_{suffix}": count,
        f"fraction_ge_{suffix}": count / len(scores),
    }


def official_metrics(calculator):
    # Keep the upstream ROC convention (including its strict TPR > 0.95 test).
    metrics = calculator.compute_metrics()
    return {
        key: float(metrics[key])
        if key in metrics and np.isfinite(metrics[key])
        else None
        for key in ("AP", "AUROC", "FPR95")
    }


def synthetic_metrics(points, scores, semantic, pooled):
    target = evaluation_targets(points, semantic)
    counts = {
        "normal_count": int((target == 0).sum()),
        "anomaly_count": int((target == 1).sum()),
    }
    single = PointOODMetricsCalculator()
    single.update(points, scores, semantic)
    eligible = bool(single.all_labels)
    if eligible:
        # Counts used for reporting must match the actual official evaluation rows.
        np.testing.assert_array_equal(single.all_labels[0], target[target != -1])
        pooled.all_labels.extend(single.all_labels)
        pooled.all_scores.extend(single.all_scores)
    return {
        **counts,
        "eligible": eligible,
        "AP": official_metrics(single)["AP"],
        "ineligible_reason": None
        if eligible
        else "fewer_than_5_official_anomaly_points",
    }


def assert_unchanged(model, reference):
    current = model.state_dict()
    if current.keys() != reference.keys() or any(
        not torch.equal(value.detach().cpu(), reference[name])
        for name, value in current.items()
    ):
        raise RuntimeError("zero-update inference changed model parameters or buffers")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("zero-update inference produced parameter gradients")


def file_hash(path, *, discard_cache=False):
    with path.open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256").hexdigest()
        if discard_cache:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return result


def summarize(rows, pooled, normal_scores):
    summary = {}
    names = tuple(pooled)
    for name in names:
        synthetic = [row["models"][name] for row in rows if row["view"] == "synthetic"]
        normal = [row for row in rows if row["view"] == "normal"]
        aps = [
            item["current"]["AP"] for item in synthetic if item["current"]["eligible"]
        ]
        scores = (
            np.concatenate(normal_scores[name]) if normal_scores[name] else np.empty(0)
        )
        worst = sorted(
            normal,
            key=lambda row: (
                row["models"][name]["current"]["fraction_ge_0_5"]
                if row["models"][name]["current"]["point_count"]
                else -1
            ),
            reverse=True,
        )
        losses = {}
        for key in ("normal", "anomaly", "total"):
            values = [
                item["loss"][key] for item in synthetic if item["loss"][key] is not None
            ]
            losses[key] = {
                "window_count": len(values),
                "window_mean": float(np.mean(values)) if values else None,
            }
        summary[name] = {
            "synthetic": {
                "window_count": len(synthetic),
                "eligible_windows": len(aps),
                **official_metrics(pooled[name]),
                "per_window_AP_median": float(np.median(aps)) if aps else None,
                "pooled_normal_points": sum(
                    int((x == 0).sum()) for x in pooled[name].all_labels
                ),
                "pooled_anomaly_points": sum(
                    int((x == 1).sum()) for x in pooled[name].all_labels
                ),
                "full_window_loss": losses,
                "official_point_scores": score_distribution(
                    np.concatenate(pooled[name].all_scores)
                    if pooled[name].all_scores
                    else np.empty(0),
                    np.concatenate(pooled[name].all_labels)
                    if pooled[name].all_labels
                    else np.empty(0),
                ),
            },
            "normal": {
                "window_count": len(normal),
                **normal_statistics(scores),
                "worst_windows": [
                    {
                        "current_frame": row["current_frame"],
                        **row["models"][name]["current"],
                    }
                    for row in worst[:5]
                ],
            },
        }
    changes = {"improved": 0, "unchanged": 0, "declined": 0, "ineligible": 0}
    for row in rows:
        if row["view"] != "synthetic":
            continue
        a, b = (row["models"][name]["current"] for name in names)
        if (a["eligible"], a["normal_count"], a["anomaly_count"]) != (
            b["eligible"],
            b["normal_count"],
            b["anomaly_count"],
        ):
            raise RuntimeError(
                "checkpoint comparisons used different evaluation points"
            )
        key = (
            "ineligible"
            if not a["eligible"]
            else (
                "improved"
                if b["AP"] > a["AP"]
                else "declined"
                if b["AP"] < a["AP"]
                else "unchanged"
            )
        )
        changes[key] += 1
    return {
        "models": summary,
        "comparison": {"reference": names[0], "candidate": names[1]},
        "per_window_AP_changes": changes,
    }


def run(
    data_root,
    checkpoints,
    output,
    *,
    checkpoint_paths=None,
    samples_file=None,
    expected_attempts=None,
):
    started = time.perf_counter()
    paths = checkpoint_paths or {
        "initial": checkpoints / "initial.pt",
        "final": checkpoints / "final.pt",
    }
    if output.exists() or any(
        output.resolve().is_relative_to(path.resolve().parent)
        for path in paths.values()
    ):
        raise FileExistsError(
            "use a new output directory outside the training evidence"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("the unchanged LitePT implementation requires CUDA")
    torch.set_num_threads(1)
    volume = host_disk()
    # 92 full five-scan outputs plus temporary NPZ buffers and modest metadata.
    disk_budget = 2 * 2**30
    if volume["SizeRemaining"] - disk_budget < volume["reserve_bytes"]:
        raise OSError("diagnostic output budget would invade the E: free-space reserve")
    protocol = load_protocol()
    samples = select_samples(protocol.validation_pool)
    source_manifest = None
    if samples_file is not None:
        source_manifest = json.loads(samples_file.read_text())
        if (
            source_manifest["samples"] != samples
            or source_manifest["normal_threshold"] != NORMAL_THRESHOLD
            or source_manifest["voxel_size"] != 0.05
        ):
            raise ValueError(
                "saved development-validation view differs from the fixed selection"
            )
        samples = source_manifest["samples"]
    begin = time.perf_counter()
    dataset = FrozenWindowDataset(data_root, protocol, pool_name="validation")
    if dataset.gradient_updates_allowed:
        raise RuntimeError(
            "the validation data role unexpectedly allows gradient updates"
        )
    worlds = [
        item
        for item in dataset.manifest["segments"]
        if item["synthetic_sequence_index"] == 0
    ]
    if source_manifest is not None and source_manifest["worlds"] != worlds:
        raise ValueError("validation worlds differ from the saved diagnostic view")
    partition = WindowPartition(
        dataset.source_sequence, CURRENT_FRAMES[0], CURRENT_FRAMES[-1]
    )
    initialization_seconds = time.perf_counter() - begin
    models, references, checkpoint_records = {}, {}, {}
    begin = time.perf_counter()
    if len(paths) != 2:
        raise ValueError("paired evaluation requires exactly two model states")
    for name, path in paths.items():
        digest = file_hash(path)
        # Only the two locally produced, user-designated checkpoint files are loaded.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["state"]
        valid_step = (
            state["planned_attempts"] == expected_attempts
            if expected_attempts is not None
            else state["successful_updates"] == {"initial": 0, "final": 200}[name]
        )
        if not valid_step or payload["config"]["voxel_size"] != 0.05:
            raise ValueError(
                "checkpoint does not match the requested training state or voxel size"
            )
        reference = payload["model"]
        model = AJAE(voxel_size=0.05).cuda().eval().requires_grad_(False)
        model.load_state_dict(reference, strict=True)
        assert_unchanged(model, reference)
        models[name], references[name] = model, reference
        checkpoint_records[name] = {
            "file": str(path.resolve()),
            "sha256": digest,
            "training_state": payload["state"],
        }
        del payload
    checkpoint_loading_seconds = time.perf_counter() - begin
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "purpose": "zero_update_paired_201_transfer_diagnostic_not_full_validation",
        "synthetic_sequence_index": 0,
        "sample_rule": "earlier_median_legal_window_per_segment",
        "samples": samples,
        "normal_threshold": NORMAL_THRESHOLD,
        "check_seed_rule": "23 + segment_index, identical across views and checkpoints",
        "voxel_size": 0.05,
        "inference_precision": "unchanged_AJAE_eval_path",
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "GPU": torch.cuda.get_device_name(),
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "checkpoints": checkpoint_records,
        "source_sample_manifest": None
        if samples_file is None
        else {"file": str(samples_file.resolve()), "sha256": file_hash(samples_file)},
        "planned_training_attempts": expected_attempts,
        "host_disk_before": volume,
        "disk_budget_bytes": disk_budget,
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            name: file_hash(PROJECT_ROOT / name)
            for name in (
                "src/evaluate.py",
                "src/model.py",
                "src/train.py",
                "protocol.json",
                "vendor/stu/compute_point_level_ood.py",
            )
        },
        "worlds": worlds,
    }
    _atomic_json(output / "samples.json", manifest)
    log = (output / "results.jsonl").open("x", encoding="utf-8", buffering=1)
    rows = []
    pooled = {name: PointOODMetricsCalculator() for name in models}
    normal_scores = {name: [] for name in models}
    status, error = "completed", None
    torch.cuda.reset_peak_memory_stats()

    def stop_signal(signum, _frame):
        raise InterruptedError(f"diagnostic interrupted by signal {signum}")

    handlers = {
        s: signal.signal(s, stop_signal)
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
    }
    signal.alarm(1800)
    print(
        json.dumps(
            {
                "event": "start",
                "samples": samples,
                "checkpoints": checkpoint_records,
                "dataset_initialization_seconds": initialization_seconds,
                "checkpoint_loading_seconds": checkpoint_loading_seconds,
            }
        ),
        flush=True,
    )
    try:
        for sample in samples:
            for view in ("synthetic", "normal"):
                begin = time.perf_counter()
                window = (
                    dataset[sample["dataset_index"]]
                    if view == "synthetic"
                    else partition.for_output(sample["current_frame"])
                )
                expected_id = sample[f"{view}_sequence_id"]
                if (
                    window.observation_sequence_id != expected_id
                    or list(window.frame_ids) != sample["frame_ids"]
                ):
                    raise ValueError(
                        "actual window differs from the predeclared paired sample"
                    )
                load_seconds = time.perf_counter() - begin
                begin = time.perf_counter()
                inputs = joint_voxelize(window, 0.05, device="cuda")
                torch.cuda.synchronize()
                prepare_seconds = time.perf_counter() - begin
                current = window.current_mask
                xyz, semantic = (
                    window.points.coordinates[current],
                    window.labels.semantic[current],
                )
                current_target = evaluation_targets(xyz, semantic)
                if view == "normal" and np.any(window.labels.anomaly_target == 1):
                    raise ValueError(
                        "raw normal 201 unexpectedly contains anomaly labels"
                    )
                row = {
                    "view": view,
                    "segment_index": sample["segment_index"],
                    "current_frame": sample["current_frame"],
                    "sequence_id": expected_id,
                    "point_count": window.points.count,
                    "current_point_count": int(current.sum()),
                    "voxel_count": len(inputs.features),
                    "load_seconds": load_seconds,
                    "prepare_seconds": prepare_seconds,
                    "models": {},
                }
                for name, model in models.items():
                    scores, losses, _, inference_seconds = predict_window(
                        model, window, inputs, sample["check_seed"]
                    )
                    begin = time.perf_counter()
                    record = PredictionBatch.from_window(window, scores)
                    relative = (
                        Path("predictions")
                        / view
                        / name
                        / f"frame_{sample['current_frame']:06d}.npz"
                    )
                    saved = record.save(output / relative, window=window)
                    saved["file"] = relative.as_posix()
                    if view == "synthetic":
                        metrics = synthetic_metrics(
                            xyz, scores[current], semantic, pooled[name]
                        )
                    else:
                        values = scores[current][current_target == 0]
                        metrics = normal_statistics(values)
                        normal_scores[name].append(values)
                    row["models"][name] = {
                        "loss": losses,
                        "full_window_scores": score_distribution(
                            scores, window.labels.anomaly_target
                        ),
                        "current": metrics,
                        "prediction": saved,
                        "inference_seconds": inference_seconds,
                        "scoring_and_saving_seconds": time.perf_counter() - begin,
                    }
                    del scores, record
                rows.append(row)
                line = json.dumps(row, allow_nan=False, separators=(",", ":"))
                log.write(line + "\n")
                print(line, flush=True)
                del inputs, window, xyz, semantic, current_target
            if (sample["segment_index"] + 1) % 5 == 0:
                print(
                    json.dumps(
                        {
                            "event": "resource",
                            "completed_pairs": len(rows) // 2,
                            "host_disk": host_disk(),
                            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_rss_bytes": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            * 1024,
                        }
                    ),
                    flush=True,
                )
        for name, model in models.items():
            assert_unchanged(model, references[name])
            if file_hash(paths[name]) != checkpoint_records[name]["sha256"]:
                raise RuntimeError("source checkpoint changed during the diagnostic")
    except BaseException as exception:
        status, error = "stopped_error", f"{type(exception).__name__}: {exception}"
        raise
    finally:
        signal.alarm(0)
        log.close()
        result = {
            "purpose": manifest["purpose"],
            "status": status,
            "error": error,
            "completed_windows": len(rows),
            "optimizer_updates": 0,
            "model_parameters_and_buffers_unchanged": status == "completed",
            **summarize(rows, pooled, normal_scores),
            "resources": {
                "wall_seconds": time.perf_counter() - started,
                "dataset_initialization_seconds": initialization_seconds,
                "checkpoint_loading_seconds": checkpoint_loading_seconds,
                "window_load_seconds": sum(row["load_seconds"] for row in rows),
                "voxel_prepare_seconds": sum(row["prepare_seconds"] for row in rows),
                "inference_seconds": {
                    name: sum(row["models"][name]["inference_seconds"] for row in rows)
                    for name in models
                },
                "scoring_and_saving_seconds": sum(
                    value["scoring_and_saving_seconds"]
                    for row in rows
                    for value in row["models"].values()
                ),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            },
        }
        _atomic_json(output / "summary.json", result)
        print(json.dumps({"event": "finished", **result}, allow_nan=False), flush=True)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return result


def full_samples(pool):
    """All legal frozen windows, followed by exactly one complete raw 201 pass."""
    v2 = pool.namespace == "v2"
    selected = (
        set() if v2 else {sample["dataset_index"] for sample in select_samples(pool)}
    )
    result = []
    index = 0
    for sequence in range(pool.synthetic_sequence_count):
        for segment in range(len(pool.segments)):
            for start in pool.window_starts(segment):
                result.append(
                    {
                        "view": "synthetic",
                        "sequence_index": sequence,
                        "segment_index": segment,
                        "dataset_index": index,
                        "current_frame": start + 4,
                        "frame_ids": list(range(start, start + 5)),
                        "sequence_id": pool.synthetic_sequence_id(sequence),
                        "check_seed": CHECK_SEED + sequence
                        if v2
                        else CHECK_SEED + sequence * 23 + segment,
                        "scope": "full"
                        if v2
                        else "selected_23"
                        if index in selected
                        else "sequence_0_remaining"
                        if sequence == 0
                        else "sequences_1_3",
                    }
                )
                index += 1
    for current in range(4, 682):
        segment = next(
            i
            for i, span in enumerate(pool.segments)
            if span.start <= current < span.stop
        )
        result.append(
            {
                "view": "normal",
                "sequence_index": None,
                "segment_index": segment,
                "dataset_index": None,
                "current_frame": current,
                "frame_ids": list(range(current - 4, current + 1)),
                "sequence_id": "train/201",
                "check_seed": CHECK_SEED + segment,
                "scope": "normal",
            }
        )
    if index != pool.total_window_count or len(result) != pool.total_window_count + 678:
        raise ValueError("full 201 validation pool size differs")
    return result


def prepare_window(dataset, partition, sample, *, voxelize=True):
    begin = time.perf_counter()
    if sample["view"] == "real":
        for index, sequence in dataset.items():
            if index != sample["sequence_index"] or sample["current_frame"] == 0:
                sequence._frames.clear()
    window = (
        dataset[sample["dataset_index"]]
        if sample["view"] == "synthetic"
        else (
            dataset[sample["sequence_index"]] if sample["view"] == "real" else partition
        ).for_output(sample["current_frame"])
    )
    if (
        window.observation_sequence_id != sample["sequence_id"]
        or list(window.frame_ids) != sample["frame_ids"]
    ):
        raise ValueError("actual full-validation window differs from its declaration")
    if sample["view"] == "normal" and np.any(window.labels.anomaly_target == 1):
        raise ValueError("raw normal 201 contains anomaly labels")
    loaded = time.perf_counter() - begin
    begin = time.perf_counter()
    inputs = joint_voxelize(window, 0.05) if voxelize else None
    return window, inputs, loaded, time.perf_counter() - begin


def append_records(output, relative, values):
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        offset = stream.tell() // values.dtype.itemsize
        values.tofile(stream)
        stream.flush()
        os.fdatasync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return dict(
        file=relative.as_posix(),
        offset=offset,
        count=len(values),
        itemsize=values.dtype.itemsize,
        sha256=hashlib.sha256(values.tobytes()).hexdigest(),
    )


def official_current_scores(prediction, slot_count):
    """Restore file slots with a label-independent zero for absent returns."""
    current = prediction.online_mask
    slots = prediction.source_slot[current]
    if len(slots) and int(slots.max()) >= slot_count:
        raise ValueError("current prediction lies outside original scan slots")
    result = np.zeros(slot_count, np.float32)
    result[slots] = prediction.anomaly_score[current]
    return result


def save_window(
    output, sample, window, scores, losses, scopes, timings, *, prediction_from=None
):
    """One bounded writer retains all predictions and only current metric records."""
    begin = time.perf_counter()
    view = sample["view"]
    score_kind = sample.get("score_kind", "probability")
    directory = (
        Path(view)
        if view == "normal"
        else Path(view) / f"{sample['sequence_index']:03d}"
    )
    relative = (
        Path("predictions") / directory / f"frame_{sample['current_frame']:06d}.npz"
    )
    batch = PredictionBatch.from_window(window, scores, score_kind=score_kind)
    if prediction_from is None:
        prediction = batch.save(output / relative, window=window)
    else:
        source, previous = prediction_from
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        os.link(source, output / relative)
        prediction = dict(previous)
    prediction["file"] = relative.as_posix()
    # A bounded Python writer must not accumulate unbounded host-backed file pages.
    with (output / relative).open("rb") as stream:
        os.fdatasync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    current = window.current_mask
    xyz, semantic = window.points.coordinates[current], window.labels.semantic[current]
    current_values = scores[current]
    if view == "real":
        source = window.current_frame.source
        xyz, semantic = source.xyzi[:, :3], source.labels.semantic
        current_values = official_current_scores(batch, source.slot_count)
    current_target = evaluation_targets(xyz, semantic)
    calculator = PointOODMetricsCalculator()
    if view in {"synthetic", "real"}:
        metrics = synthetic_metrics(xyz, current_values, semantic, calculator)
        keys = (
            packed_scores(
                calculator.all_scores[0],
                calculator.all_labels[0],
                score_kind=score_kind,
            )
            if metrics["eligible"]
            else np.empty(0, np.uint64)
        )
        if view == "real":
            metrics["raw_anomaly_count"] = int(
                (window.labels.semantic[current] == 2).sum()
            )
            metrics["normal"] = normal_statistics(
                current_values[current_target == 0], score_kind=score_kind
            )
            metrics["anomaly"] = normal_statistics(
                current_values[current_target == 1], score_kind=score_kind
            )
            metrics["raw_slot_count"] = window.current_frame.source.slot_count
            normal_values = (
                current_values[current_target == 0]
                if metrics["raw_anomaly_count"] == 0
                else np.empty(0, np.float32)
            )
            normal_records = append_records(
                output,
                Path("current")
                / str(sample["sequence_index"])
                / f"normal_{sample['scope']}.bin",
                normal_values.astype(np.float32, copy=False),
            )
            metric_path = (
                Path("current")
                / str(sample["sequence_index"])
                / f"{sample['scope']}.bin"
            )
        else:
            subset = "selected" if sample["scope"] == "selected_23" else "remaining"
            metric_path = (
                Path("current")
                / f"{sample['sequence_index']:03d}"
                / f"segment_{sample.get('segment_index', 0):02d}_{subset}.bin"
            )
    else:
        values = current_values[current_target == 0]
        metrics = normal_statistics(values, score_kind=score_kind)
        keys = packed_scores(
            values, np.zeros(len(values), dtype=np.int8), score_kind=score_kind
        )
        metric_path = Path("current/normal.bin")
    records = append_records(output, metric_path, keys)
    return {
        **sample,
        "point_count": window.points.count,
        "current_point_count": int(current.sum()),
        "loss": losses,
        "anomaly_loss_scopes": scopes,
        "current": metrics,
        "prediction": prediction,
        "evaluation_records": records,
        **({"normal_records": normal_records} if view == "real" else {}),
        **timings,
        "scoring_and_saving_seconds": time.perf_counter() - begin,
    }


def semantic_summary(rows):
    matrices = [
        r["loss"]["semantic_confusion"]
        for r in rows
        if "semantic_confusion" in r["loss"]
    ]
    if not matrices:
        return None
    matrix = np.sum(matrices, axis=0, dtype=np.int64)
    union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
    iou = np.divide(matrix.diagonal(), union, out=np.full(20, np.nan), where=union > 0)
    return dict(
        scope="current visible binary-normal points in raw 201; no distance filter; other_normal=19",
        point_count=int(matrix.sum()),
        confusion=matrix.tolist(),
        per_group_IoU=[float(x) if np.isfinite(x) else None for x in iou],
        mean_IoU=float(np.nanmean(iou)),
        original_19_mean_IoU=float(np.nanmean(iou[:19])),
    )


def loss_summary(rows):
    losses, scopes = {}, {}
    for name in ("normal", "anomaly", "total"):
        values = [row["loss"][name] for row in rows if row["loss"][name] is not None]
        losses[name] = {
            "window_count": len(values),
            "window_mean": float(np.mean(values)) if values else None,
        }
    for name in ("all", "history", "current", "current_official_range"):
        items = [row["anomaly_loss_scopes"][name] for row in rows]
        count = sum(item["point_count"] for item in items)
        total = sum(item["loss_sum"] for item in items)
        means = [item["point_mean"] for item in items if item["point_count"]]
        scopes[name] = {
            "point_count": count,
            "loss_sum": total,
            "point_mean": total / count if count else None,
            "window_count": len(means),
            "window_mean": float(np.mean(means)) if means else None,
        }
    return {"full_window_loss": losses, "anomaly_loss_scopes": scopes}


def summarize_full(rows, output, *, check_resources=None):
    synthetic = [row for row in rows if row["view"] == "synthetic"]
    normal = [row for row in rows if row["view"] == "normal"]
    kind = rows[0].get("score_kind", "probability") if rows else "probability"
    fraction_key = "fraction_ge_0" if kind == "logit" else "fraction_ge_0_5"

    def group(items):
        if check_resources is not None:
            check_resources()
        paths = sorted({output / row["evaluation_records"]["file"] for row in items})
        aps = [row["current"]["AP"] for row in items if row["current"]["eligible"]]
        metrics = pooled_files(paths, score_kind=kind)
        if metrics["normal_count"] != sum(
            row["current"]["normal_count"]
            for row in items
            if row["current"]["eligible"]
        ) or metrics["anomaly_count"] != sum(
            row["current"]["anomaly_count"]
            for row in items
            if row["current"]["eligible"]
        ):
            raise RuntimeError(
                "exact pooling and eligible current point counts disagree"
            )
        return {
            "window_count": len(items),
            "eligible_windows": len(aps),
            **metrics,
            "per_window_AP_median": float(np.median(aps)) if aps else None,
            "ineligible_reason": None
            if aps
            else "no_window_has_5_official_anomaly_points",
            **loss_summary(items),
        }

    worlds = []
    for sequence in sorted({r["sequence_index"] for r in synthetic}):
        for segment in sorted(
            {
                r.get("segment_index", 0)
                for r in synthetic
                if r["sequence_index"] == sequence
            }
        ):
            items = [
                row
                for row in synthetic
                if row["sequence_index"] == sequence
                and row.get("segment_index", 0) == segment
            ]
            worlds.append(
                {
                    "sequence_index": sequence,
                    "segment_index": segment,
                    **group(items),
                    "windows": [
                        {"current_frame": row["current_frame"], **row["current"]}
                        for row in items
                    ],
                }
            )
        print(
            json.dumps({"event": "world_metrics", "completed_worlds": len(worlds)}),
            flush=True,
        )
    _atomic_json(output / "worlds.json", {"worlds": worlds})
    subsets = {
        name: group([row for row in synthetic if row["scope"] == name])
        for name in sorted({r["scope"] for r in synthetic})
    }
    sequences = {
        str(index): group([row for row in synthetic if row["sequence_index"] == index])
        for index in sorted({r["sequence_index"] for r in synthetic})
    }
    complete = group(synthetic)
    normal_metrics = pooled_files(
        sorted({output / r["evaluation_records"]["file"] for r in normal}),
        normal=True,
        score_kind=kind,
    )
    world_aps = [world["AP"] for world in worlds if world["AP"] is not None]
    worst = sorted(
        normal, key=lambda row: row["current"][fraction_key] or 0, reverse=True
    )
    return {
        "score_kind": kind,
        "normal_201_semantics": semantic_summary(normal),
        "synthetic": complete,
        "subsets": subsets,
        "sequences": sequences,
        "world_count": len(worlds),
        "world_AP_q25": float(np.quantile(world_aps, 0.25)) if world_aps else None,
        "world_AP_median": float(np.median(world_aps)) if world_aps else None,
        "worlds_below_10_AP": [
            {
                key: world[key]
                for key in (
                    "sequence_index",
                    "segment_index",
                    "AP",
                    "anomaly_count",
                    "eligible_windows",
                )
            }
            for world in worlds
            if world["AP"] is not None and world["AP"] < 10
        ],
        "worlds_without_eligible_windows": sum(
            not world["eligible_windows"] for world in worlds
        ),
        "normal": {
            "window_count": len(normal),
            **normal_metrics,
            "sequence": [
                {"current_frame": row["current_frame"], **row["current"]}
                for row in normal
            ],
            "worst_windows": [
                {"current_frame": row["current_frame"], **row["current"]}
                for row in worst[:10]
            ],
            "around_407": [
                {"current_frame": row["current_frame"], **row["current"]}
                for row in normal
                if 397 <= row["current_frame"] <= 417
            ],
        },
    }


def monitor_samples(pool):
    """First, earlier middle and last legal window of every frozen 201 world."""
    currents = set()
    for segment in range(len(pool.segments)):
        starts = pool.window_starts(segment)
        currents.update(starts[i] + 4 for i in (0, (len(starts) - 1) // 2, -1))
    samples = [s for s in full_samples(pool) if s["current_frame"] in currents]
    if len(samples) != 345 or sum(s["view"] == "normal" for s in samples) != 69:
        raise ValueError("the fixed 201 monitor must contain 276 + 69 windows")
    return samples


def reusable_predictions(directories, model_sha256):
    """Reuse only completed predictions from the identical model and point task."""
    result = {}
    for directory in directories:
        if not (directory / "summary.json").exists():
            continue
        manifest = json.loads((directory / "samples.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        if (
            summary["status"] != "completed"
            or manifest["identity"].get("model_sha256") != model_sha256
            or manifest["identity"].get("rotary_cache_precision")
            != "autocast_separated"
        ):
            continue
        for line in (directory / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["sequence_id"], row["current_frame"])
            result.setdefault(key, (directory, row))
    return result


def evaluate_samples(
    model, dataset, samples, output, *, identity, check_resources, reuse=None
):
    """The same bounded, zero-update evaluator serves monitoring and full selection."""
    from .train import write_progress

    started = time.perf_counter()
    real = identity.get("scope") == "real_val"
    reuse = reuse or {}
    score_kind = getattr(model, "score_kind", "probability")
    if score_kind == "logit":
        identity = {
            **identity,
            "score_kind": "logit",
            "normal_observation_threshold": 0.0,
            "rotary_cache_precision": "autocast_separated",
        }
    manifest = {
        "identity": identity,
        "samples": samples,
        "worlds": [] if real else dataset.manifest["segments"],
    }
    partition = None if real else WindowPartition(dataset.source_sequence, 4, 681)
    reference = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    if not real and dataset.gradient_updates_allowed:
        raise RuntimeError("201 must not permit gradient updates")
    rows = []
    if (output / "samples.json").exists():
        if json.loads((output / "samples.json").read_text()) != manifest:
            raise ValueError("resume evaluation identity or sample list differs")
        result_path = output / "results.jsonl"
        result_path.touch(exist_ok=True)
        # A committed JSONL row is the boundary for both prediction and metric writes.
        valid_bytes = 0
        with result_path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                rows.append(json.loads(line))
                valid_bytes += len(line)
        if len(rows) > len(samples):
            raise ValueError(
                "evaluation log contains more rows than the fixed sample list"
            )
        with result_path.open("r+b") as stream:
            stream.truncate(valid_bytes)
        ends = {}
        retained = set()
        for sample, row in zip(samples, rows):
            if any(row[key] != value for key, value in sample.items()):
                raise ValueError("committed evaluation rows differ from fixed order")
            pred = output / row["prediction"]["file"]
            if file_hash(pred, discard_cache=True) != row["prediction"]["file_sha256"]:
                raise ValueError("committed prediction changed")
            retained.add(pred)
            for key in ("evaluation_records", "normal_records"):
                if key not in row:
                    continue
                records = row[key]
                path = output / records["file"]
                itemsize = records.get("itemsize", 8)
                if records["offset"] * itemsize != ends.get(path, 0):
                    raise ValueError("evaluation record offsets are discontinuous")
                ends[path] = (records["offset"] + records["count"]) * itemsize
                with path.open("rb") as stream:
                    stream.seek(records["offset"] * itemsize)
                    block = stream.read(records["count"] * itemsize)
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                if hashlib.sha256(block).hexdigest() != records["sha256"]:
                    raise ValueError("committed exact evaluation records changed")
        for path in (output / "current").rglob("*.bin"):
            length = ends.get(path, 0)
            if path.stat().st_size < length:
                raise ValueError("committed evaluation records were truncated")
            with path.open("r+b") as stream:
                stream.truncate(length)
        for path in (output / "predictions").rglob("*.npz"):
            if path not in retained:
                path.unlink()  # Only the interrupted, uncommitted write is removed.
        summary_path = output / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary["status"] == "completed" and len(rows) == len(samples):
                assert_unchanged(model, reference)
                return summary
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "samples.json", manifest)
    status, error, summary = "running", None, {}
    try:
        with (
            (output / "results.jsonl").open("a", buffering=1) as log,
            ThreadPoolExecutor(max_workers=1) as loader,
            ThreadPoolExecutor(max_workers=1) as writer,
        ):
            prepared = None
            pending_write = None
            if len(rows) < len(samples):
                check_resources()
                prepared = loader.submit(
                    prepare_window,
                    dataset,
                    partition,
                    samples[len(rows)],
                    voxelize=(
                        samples[len(rows)].get("sequence_id"),
                        samples[len(rows)]["current_frame"],
                    )
                    not in reuse,
                )
            for index in range(len(rows), len(samples)):
                check_resources()
                sample = samples[index]
                if score_kind == "logit":
                    sample = {**sample, "score_kind": "logit"}
                window, cpu_inputs, load_seconds, prepare_seconds = prepared.result()
                prepared = None
                if index + 1 < len(samples):
                    prepared = loader.submit(
                        prepare_window,
                        dataset,
                        partition,
                        samples[index + 1],
                        voxelize=(
                            samples[index + 1].get("sequence_id"),
                            samples[index + 1]["current_frame"],
                        )
                        not in reuse,
                    )
                begin = time.perf_counter()
                # Reused scores need point-identity validation, not another voxel grid or GPU copy.
                inputs = cpu_inputs.to("cuda") if cpu_inputs is not None else None
                if inputs is not None:
                    torch.cuda.synchronize()
                transfer_seconds = time.perf_counter() - begin
                key = (sample.get("sequence_id"), sample["current_frame"])
                reused = reuse.get(key)
                prediction_from = None
                if reused:
                    directory, old = reused
                    if any(
                        sample[k] != old[k]
                        for k in (
                            "sequence_id",
                            "current_frame",
                            "frame_ids",
                            "check_seed",
                        )
                    ):
                        raise ValueError(
                            "reused prediction has different input identity"
                        )
                    source = directory / old["prediction"]["file"]
                    prior = PredictionBatch.load(
                        source,
                        window=window,
                        expected_sha256=old["prediction"]["file_sha256"],
                    )
                    if prior.score_kind != score_kind:
                        raise ValueError("reused prediction has a different score kind")
                    scores, losses, scopes, inference_seconds = (
                        prior.anomaly_score,
                        old["loss"],
                        old["anomaly_loss_scopes"],
                        0.0,
                    )
                    prediction_from = source, old["prediction"]
                if not reused:
                    scores, losses, scopes, inference_seconds = predict_window(
                        model, window, inputs, sample["check_seed"], split_losses=True
                    )
                timings = {
                    "load_seconds": load_seconds,
                    "prepare_seconds": prepare_seconds,
                    "transfer_seconds": transfer_seconds,
                    "inference_seconds": inference_seconds,
                    "voxel_count": old["voxel_count"]
                    if reused
                    else len(inputs.features),
                    **(
                        dict(
                            reused_prediction=str(source),
                            reuse_identity_verified=True,
                        )
                        if reused
                        else {}
                    ),
                }
                if pending_write is not None:
                    row = pending_write.result()
                    log.write(
                        json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                    )
                    rows.append(row)
                pending_write = writer.submit(
                    save_window,
                    output,
                    sample,
                    window,
                    scores,
                    losses,
                    scopes,
                    timings,
                    prediction_from=prediction_from,
                )
                del inputs, cpu_inputs, window, scores
                if (index + 1) % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "evaluation_progress",
                                "directory": str(output),
                                "windows": index + 1,
                                "total": len(samples),
                                "wall_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
            if pending_write is not None:
                row = pending_write.result()
                log.write(
                    json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                )
                rows.append(row)
        assert_unchanged(model, reference)
        # An interrupted summary is recomputed from the committed predictions.
        if (output / "worlds.json").exists():
            (output / "worlds.json").unlink()
        if real:
            for sequence in dataset.values():
                sequence._frames.clear()
        summary = (
            summarize_real(
                rows,
                output,
                check_resources=check_resources,
                continuous=not (identity.get("monitor") or identity.get("interim")),
            )
            if real
            else summarize_full(rows, output, check_resources=check_resources)
        )
        status = "completed"
    except BaseException as exception:
        status, error = "stopped_error", f"{type(exception).__name__}: {exception}"
        raise
    finally:
        result = {
            "status": status,
            "error": error,
            "completed_windows": len(rows),
            "reused_windows": sum("reused_prediction" in row for row in rows),
            "optimizer_updates": 0,
            "model_parameters_and_buffers_unchanged": status == "completed",
            **summary,
            "resources": {
                "session_wall_seconds": time.perf_counter() - started,
                **{
                    key: sum(row[key] for row in rows)
                    for key in (
                        "load_seconds",
                        "prepare_seconds",
                        "transfer_seconds",
                        "inference_seconds",
                        "scoring_and_saving_seconds",
                    )
                },
            },
        }
        write_progress(output / "summary.json", result)
    return result


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_baseline(initial, monitors):
    """Bind B reuse to the same inputs, complete prediction rows and exact metrics."""
    directory = PROJECT_ROOT / "runs/history/validation"
    manifest = json.loads((directory / "samples.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    plan = json.loads((PROJECT_ROOT / "runs/history/coverage/plan.json").read_text())
    if file_hash(initial) != plan["initial_checkpoint"]["sha256"]:
        raise ValueError("original initialization differs from coverage evidence")
    checkpoint = Path(manifest["checkpoint"]["file"])
    if (
        file_hash(checkpoint) != manifest["checkpoint"]["sha256"]
        or summary["status"] != "completed"
        or summary["completed_windows"] != 3038
        or summary["optimizer_updates"] != 0
        or not summary["model_parameters_and_buffers_unchanged"]
    ):
        raise ValueError("B has no verified complete zero-update validation")
    protocol = load_protocol()
    if manifest["samples"] != full_samples(protocol.validation_pool):
        raise ValueError("B complete validation samples or check seeds differ")
    frozen = json.loads(
        (PROJECT_ROOT / "artifacts/data/v1/validation_manifest.json").read_text()
    )
    if manifest["worlds"] != frozen["segments"]:
        raise ValueError("B validation worlds differ from the frozen manifest")
    for name in (
        "protocol.json",
        "vendor/stu/compute_point_level_ood.py",
    ):
        if file_hash(PROJECT_ROOT / name) != manifest["source_sha256"][name]:
            raise ValueError(
                f"B scientific input or metric implementation changed: {name}"
            )
    # Voxel sorting may change implementation after exact-input regression; network classes stay fixed.
    old_model = subprocess.check_output(
        ["git", "show", manifest["base_commit"] + ":src/model.py"], text=True
    )

    def classes(source):
        return [
            ast.dump(node)
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef)
        ]

    if classes(old_model) != classes((PROJECT_ROOT / "src/model.py").read_text()):
        raise ValueError("B network architecture or forward behavior changed")
    rows = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text().splitlines()
    ]
    if len(rows) != 3038:
        raise ValueError("B prediction row coverage differs")
    monitor_ids = {(s["sequence_id"], s["current_frame"]) for s in monitors}
    prediction_bound = monitor_bound = monitor_records = 0
    for sample, row in zip(manifest["samples"], rows, strict=True):
        if any(row[key] != value for key, value in sample.items()):
            raise ValueError("B prediction identities or check seeds differ")
        path = directory / row["prediction"]["file"]
        if file_hash(path, discard_cache=True) != row["prediction"]["file_sha256"]:
            raise ValueError("B retained prediction content changed")
        with zipfile.ZipFile(path) as archive:
            overhead = sum(
                info.compress_size + 2 * len(info.filename) + 128
                for info in archive.infolist()
                if info.filename != "anomaly_score.npy"
            )
        # Bound new score storage by raw float32 bytes plus DEFLATE expansion.
        bound = overhead + row["point_count"] * 4 * 1.001 + 1024
        prediction_bound += bound
        if (sample["sequence_id"], sample["current_frame"]) in monitor_ids:
            monitor_bound += bound
            monitor_records += row["evaluation_records"]["count"] * 8
    paths = sorted((directory / "current").rglob("*.bin"))
    current_bytes = sum(path.stat().st_size for path in paths)
    official = pooled_files([path for path in paths if path.name != "normal.bin"])
    normal = pooled_files([directory / "current/normal.bin"], normal=True)
    if any(
        abs(official[key] - summary["synthetic"][key]) > 1e-10
        for key in ("AP", "AUROC", "FPR95")
    ) or any(normal[key] != summary["normal"][key] for key in normal):
        raise ValueError("independent exact B metric reduction disagrees")
    evidence = {
        name: file_hash(directory / name)
        for name in ("samples.json", "results.jsonl", "summary.json", "worlds.json")
    }
    estimated_peak = int(
        2 * (prediction_bound + current_bytes)
        + 10 * (monitor_bound + monitor_records)
        + 15 * checkpoint.stat().st_size
        + current_bytes
        + 0.5 * 2**30
    )
    return {
        "checkpoint": str(checkpoint),
        "sha256": manifest["checkpoint"]["sha256"],
        "evaluation": str(directory),
        "evidence_sha256": evidence,
        "verified_complete_predictions": len(rows),
        "recomputed_exact_metrics": official,
        "normal": normal,
    }, max(35 * 2**30, estimated_peak)


def select_full_candidates(output, selected, baseline):
    from .train import choose_candidate

    candidates, paired, normal_series = [], {}, {}
    entries = [
        {"name": "B", "epoch": 0, **baseline},
        *[
            {**candidate, "evaluation": str(output / "validation" / candidate["name"])}
            for candidate in selected
        ],
    ]
    for entry in entries:
        directory = Path(entry["evaluation"])
        summary = json.loads((directory / "summary.json").read_text())
        if summary["status"] != "completed" or summary["completed_windows"] != 3038:
            raise ValueError("incomplete full validation cannot enter final selection")
        worlds = json.loads((directory / "worlds.json").read_text())["worlds"]
        rows = [
            json.loads(line)
            for line in (directory / "results.jsonl").read_text().splitlines()
        ]
        normal = [r for r in rows if r["view"] == "normal"]
        aps = [world["AP"] for world in worlds if world["AP"] is not None]
        rolling = []
        for start in range(len(normal) - 20):
            block = normal[start : start + 21]
            count = sum(row["current"]["point_count"] for row in block)
            high = sum(row["current"]["count_ge_0_5"] for row in block)
            rolling.append(
                {
                    "first_frame": block[0]["current_frame"],
                    "last_frame": block[-1]["current_frame"],
                    "fraction_ge_0_5": high / count,
                    "count_ge_0_5": high,
                    "point_count": count,
                }
            )
        candidate = {
            "name": entry["name"],
            "epoch": entry["epoch"],
            "scope": "complete_201",
            **{key: summary["synthetic"][key] for key in ("AP", "AUROC", "FPR95")},
            "normal_fraction": summary["normal"]["fraction_ge_0_5"],
            "normal_median": summary["normal"]["median"],
            "normal_p95": summary["normal"]["p95"],
            "world_AP_q25": float(np.quantile(aps, 0.25)),
            "world_AP_median": float(np.median(aps)),
            "worlds_below_10_AP": [
                {
                    k: w[k]
                    for k in ("sequence_index", "segment_index", "AP", "anomaly_count")
                }
                for w in worlds
                if w["AP"] is not None and w["AP"] < 10
            ],
            "worst_21_consecutive_frames": max(
                rolling, key=lambda r: r["fraction_ge_0_5"]
            ),
            "checkpoint": entry["checkpoint"],
            "sha256": file_hash(Path(entry["checkpoint"])),
            "evaluation": str(directory),
        }
        candidates.append(candidate)
        normal_series[entry["name"]] = [
            {"current_frame": r["current_frame"], **r["current"]} for r in normal
        ]
        for world in worlds:
            key = f"{world['sequence_index']}:{world['segment_index']}"
            paired.setdefault(key, {})[entry["name"]] = {
                k: world[k]
                for k in (
                    "AP",
                    "AUROC",
                    "FPR95",
                    "eligible_windows",
                    "anomaly_count",
                    "normal_count",
                )
            }
    for values in paired.values():
        for name, metrics in values.items():
            if any(
                metrics[key] != values["B"][key]
                for key in ("anomaly_count", "normal_count", "eligible_windows")
            ):
                raise ValueError(
                    "paired world evaluation contains different point populations"
                )
            metrics["AP_change_from_B"] = metrics["AP"] - values["B"]["AP"]
    winner = choose_candidate(candidates)
    result = {
        "candidates": candidates,
        "selected": winner,
        "paired_worlds": paired,
        "normal_sequences": normal_series,
        "real_anomaly_evaluated": False,
        "score_definition": "sigmoid of unchanged AJAE full-window point logits; no calibration or fusion",
        "inference": "five ordered causal scans, 0.05 m voxels, unchanged eval path; only current-frame scores enter official metrics",
        "boundary": "one training seed and development-selected candidates; neither cross-seed stability nor five-frame mechanism attribution",
    }
    from .train import write_progress

    write_progress(output / "comparison.json", result)
    return result


def summarize_real(rows, output, *, check_resources, continuous=True):
    kind = rows[0].get("score_kind", "probability") if rows else "probability"
    fraction_key = "fraction_ge_0" if kind == "logit" else "fraction_ge_0_5"
    count_key = "count_ge_0" if kind == "logit" else "count_ge_0_5"

    def group(items):
        check_resources()
        records = [row["evaluation_records"] for row in items]
        metrics = pooled_files(
            [output / r["file"] for r in records],
            ranges=[(r["offset"], r["count"]) for r in records],
            score_kind=kind,
        )
        eligible = [row for row in items if row["current"]["eligible"]]
        for key in ("normal_count", "anomaly_count"):
            if metrics[key] != sum(row["current"][key] for row in eligible):
                raise RuntimeError(
                    "real pooled point counts disagree with frame records"
                )
        return dict(frame_count=len(items), eligible_frames=len(eligible), **metrics)

    def normal(items):
        paths = sorted({output / row["normal_records"]["file"] for row in items})
        values = pooled_files(paths, normal=True, normal_float32=True, score_kind=kind)
        selected = [r for r in items if r["current"]["raw_anomaly_count"] == 0]
        if values["point_count"] != sum(
            r["current"]["normal"]["point_count"] for r in selected
        ):
            raise RuntimeError("normal-only frame records disagree with pooled scores")
        return dict(frame_count=len(selected), **values)

    def excerpt(row):
        return dict(current_frame=row["current_frame"], **row["current"])

    if not continuous:
        # A time-selected monitor cannot establish contiguous visibility or miss durations.
        return dict(
            score_kind=kind,
            scope="fixed_real_development_monitor",
            all_frames=group(rows),
            normal_without_anomaly_returns=normal(rows),
            sequences={
                str(index): dict(
                    all_frames=group([r for r in rows if r["sequence_index"] == index]),
                    normal_without_anomaly_returns=normal(
                        [r for r in rows if r["sequence_index"] == index]
                    ),
                )
                for index in sorted({r["sequence_index"] for r in rows})
            },
        )

    def longest_run(items, predicate):
        best, current = [], []
        for row in items:
            if predicate(row):
                current.append(row)
                if len(current) > len(best):
                    best = current.copy()
            else:
                current = []
        return [excerpt(row) for row in best]

    sequences = {}
    for sequence in sorted({row["sequence_index"] for row in rows}):
        items = [row for row in rows if row["sequence_index"] == sequence]
        full = [row for row in items if row["current_frame"] >= 4]
        visible = [
            i for i, row in enumerate(items) if row["current"]["raw_anomaly_count"]
        ]
        sequences[str(sequence)] = dict(
            all_frames=group(items),
            full_history=group(full),
            normal_without_anomaly_returns=normal(items),
            first_visible=items[visible[0]]["current_frame"] if visible else None,
            last_visible=items[visible[-1]]["current_frame"] if visible else None,
            first_visible_phase=[excerpt(r) for r in items[visible[0] : visible[0] + 5]]
            if visible
            else [],
            after_last_visible=[
                excerpt(r) for r in items[visible[-1] + 1 : visible[-1] + 6]
            ]
            if visible
            else [],
            longest_few_point_miss=longest_run(
                items,
                lambda r: (
                    1 <= r["current"]["anomaly_count"] <= 4
                    and r["current"]["anomaly"][count_key] == 0
                ),
            ),
            longest_eligible_complete_miss=longest_run(
                items,
                lambda r: (
                    r["current"]["eligible"] and r["current"]["anomaly"][count_key] == 0
                ),
            ),
            worst_normal_frames=[
                excerpt(r)
                for r in sorted(
                    [r for r in items if r["current"]["raw_anomaly_count"] == 0],
                    key=lambda r: r["current"]["normal"][fraction_key] or 0,
                    reverse=True,
                )[:5]
            ],
        )
        print(
            json.dumps({"event": "real_sequence_metrics", "sequence": sequence}),
            flush=True,
        )
    return dict(
        score_kind=kind,
        all_frames=group(rows),
        full_history=group([r for r in rows if r["current_frame"] >= 4]),
        startup=group([r for r in rows if r["current_frame"] < 4]),
        normal_without_anomaly_returns=normal(rows),
        sequences=sequences,
        ineligible_frames=dict(
            zero_official_anomaly_points=sum(
                r["current"]["anomaly_count"] == 0 for r in rows
            ),
            one_to_four_official_anomaly_points=sum(
                1 <= r["current"]["anomaly_count"] <= 4 for r in rows
            ),
        ),
    )


def compare_real_results(current, baseline):
    """Compare the same public frames and points; score thresholds retain their meaning."""
    result = {}
    for scope in ("all_frames", "full_history", "startup"):
        new, old = current[scope], baseline[scope]
        for key in ("frame_count", "eligible_frames", "normal_count", "anomaly_count"):
            if new[key] != old[key]:
                raise ValueError("NRE and v1 real evaluation populations differ")
        result[scope] = {
            metric: dict(
                v1=old[metric],
                nre=new[metric],
                difference_percentage_points=new[metric] - old[metric],
            )
            for metric in ("AP", "AUROC", "FPR95")
        }
    new, old = (r["normal_without_anomaly_returns"] for r in (current, baseline))
    if any(new[k] != old[k] for k in ("frame_count", "point_count")):
        raise ValueError("NRE and v1 normal-only real observations differ")
    result["normal_without_anomaly_returns"] = dict(
        frame_count=new["frame_count"],
        point_count=new["point_count"],
        v1_probability_ge_0_5=old["fraction_ge_0_5"],
        nre_logit_ge_0=new["fraction_ge_0"],
        difference_percentage_points=100
        * (new["fraction_ge_0"] - old["fraction_ge_0_5"]),
    )
    return result


def check_startup(model, data_root, protocol, output, checkpoint_sha256):
    identity = dict(
        checkpoint_sha256=checkpoint_sha256,
        source_sha256={
            name: file_hash(PROJECT_ROOT / name)
            for name in (
                "src/scene.py",
                "src/data.py",
                "src/model.py",
                "src/evaluate.py",
            )
        },
    )
    kind = getattr(model, "score_kind", "probability")
    if kind == "logit":
        identity["score_kind"] = kind
    path = output / "startup/checks.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["identity"] != identity:
            raise ValueError("startup check implementation or candidate changed")
        return result
    rows = []
    for sequence_id in (206, 201):
        sequence = STUSequence.open(
            data_root,
            protocol=protocol,
            partition="train",
            sequence_id=sequence_id,
            label_mode=LabelMode.REQUIRED,
        )
        for current in (0, 1, 2, 3, 4, 5, sequence.frame_count - 1):
            window = sequence.for_output(current)
            inputs = joint_voxelize(window)
            scores, _, _, _ = predict_window(
                model, window, inputs.to("cuda"), CHECK_SEED
            )
            batch = PredictionBatch.from_window(window, scores, score_kind=kind)
            prediction_path = (
                output / "startup" / str(sequence_id) / f"{current:06d}.npz"
            )
            record = batch.save(prediction_path, window=window)
            recovered = PredictionBatch.load(
                prediction_path, window=window, expected_sha256=record["file_sha256"]
            )
            np.testing.assert_array_equal(recovered.anomaly_score, scores)
            assert recovered.score_kind == kind
            expected_groups = np.arange(4 - current, 5) if current < 4 else np.arange(5)
            np.testing.assert_array_equal(
                np.unique(window.points.scan_group), expected_groups
            )
            assert list(window.frame_ids) == list(
                range(max(0, current - 4), current + 1)
            )
            if current < 4:
                assert not np.any(inputs.features.numpy()[:, 4 : 4 + (4 - current)])
            maximum_difference = None
            if current >= 4:
                old = sequence.window(current - 4)
                old_inputs = joint_voxelize(old)
                for key in (
                    "coordinates",
                    "grid_coord",
                    "features",
                    "point_to_voxel",
                    "point_features",
                ):
                    assert torch.equal(getattr(inputs, key), getattr(old_inputs, key))
                old_scores, _, _, _ = predict_window(
                    model, old, old_inputs.to("cuda"), CHECK_SEED
                )
                np.testing.assert_array_equal(scores, old_scores)
                maximum_difference = float(np.max(np.abs(scores - old_scores)))
                del old, old_inputs, old_scores
            raw = official_current_scores(
                recovered, window.current_frame.source.slot_count
            )
            text_buffer = io.StringIO()
            np.savetxt(text_buffer, raw, fmt="%.9g")
            text_buffer.seek(0)
            np.testing.assert_array_equal(
                np.loadtxt(text_buffer).astype(np.float32), raw
            )
            source = window.current_frame.source
            np.testing.assert_array_equal(
                raw[source.real_slots], scores[window.current_mask]
            )
            assert np.isfinite(raw).all() and np.all(raw[source.zero_slot_mask] == 0)
            rows.append(
                dict(
                    sequence_id=sequence_id,
                    current_frame=current,
                    frame_ids=list(window.frame_ids),
                    scan_groups=expected_groups.tolist(),
                    window_points=window.points.count,
                    raw_slots=len(raw),
                    full_path_max_difference=maximum_difference,
                    prediction=record,
                )
            )
            del window, inputs, scores, batch, recovered, raw, text_buffer
        sequence._frames.clear()
    result = dict(
        identity=identity,
        rows=rows,
        status="passed",
        note="206/201 input and inference checks only; no real-val score selection",
    )
    _atomic_json(path, result)
    return result


def real_inventory(data_root, protocol, check_resources):
    sequences, inventory, samples = {}, [], []
    total_prediction_bytes = 0
    for sequence_id in protocol.public_sequence_ids:
        sequence = STUSequence.open(
            data_root,
            protocol=protocol,
            partition="val",
            sequence_id=sequence_id,
            label_mode=LabelMode.REQUIRED,
        )
        sequences[sequence_id] = sequence
        stats = dict(
            sequence_id=sequence_id,
            frame_count=sequence.frame_count,
            raw_slots=0,
            visible_points=0,
            window_points=0,
            eligible_frames=0,
            eligible_points=0,
            normal_only_points=0,
            prediction_bound_bytes=0,
        )
        digest = hashlib.sha256()
        for name in ("calib.txt", "poses.txt"):
            digest.update((sequence.sequence_dir / name).read_bytes())
        for current in sequence.frame_ids:
            check_resources()
            source = sequence.source_frame(current)
            digest.update(np.int32(current).tobytes())
            digest.update(source.xyzi.tobytes())
            digest.update(source.labels.packed.tobytes())
            target = evaluation_targets(source.xyzi[:, :3], source.labels.semantic)
            count = source.real_count
            repetitions = min(5, sequence.frame_count - current)
            # Worst-case lossless score bytes; identity compression is measured on actual slots.
            slots = np.diff(source.real_slots, prepend=np.int32(0))
            slot_bytes = len(zlib.compress(slots.tobytes(), 1)) + 64
            frame_bytes = (
                len(zlib.compress(np.full(count, current, np.int32).tobytes(), 1)) + 64
            )
            stats["prediction_bound_bytes"] += repetitions * (
                int(4.004 * count) + slot_bytes + frame_bytes
            )
            stats["raw_slots"] += source.slot_count
            stats["visible_points"] += count
            stats["window_points"] += repetitions * count
            anomalies = int((target == 1).sum())
            stats["eligible_frames"] += anomalies >= 5
            stats["eligible_points"] += (
                int((target >= 0).sum()) if anomalies >= 5 else 0
            )
            stats["normal_only_points"] += (
                int((target == 0).sum())
                if not np.any(source.labels.semantic[source.real_slots] == 2)
                else 0
            )
            samples.append(
                dict(
                    view="real",
                    sequence_index=sequence_id,
                    dataset_index=None,
                    sequence_id=f"val/{sequence_id}",
                    current_frame=current,
                    frame_ids=list(range(max(0, current - 4), current + 1)),
                    check_seed=CHECK_SEED + sequence_id,
                    scope="startup" if current < 4 else "full",
                )
            )
        stats["source_content_sha256"] = digest.hexdigest()
        total_prediction_bytes += stats["prediction_bound_bytes"]
        inventory.append(stats)
        sequence._frames.clear()
        print(json.dumps({"event": "real_inventory", **stats}), flush=True)
    official_bytes = 8 * sum(row["eligible_points"] for row in inventory)
    normal_bytes = 4 * sum(row["normal_only_points"] for row in inventory)
    # One atomic prediction, startup checks, metadata, allocator slack; no text duplicate.
    budget = dict(
        predictions=total_prediction_bytes + len(samples) * 8192,
        official_records=official_bytes,
        normal_records=normal_bytes,
        temporary_sort=official_bytes,
        buffers_and_startup=2**30,
    )
    budget["peak_new_bytes"] = sum(budget.values())
    return sequences, inventory, samples, budget


def nre_storage(data_root, config):
    """Bound all future float32 outputs from actual slots; scan raw 201 only once."""
    import zlib
    import zipfile
    from collections import Counter
    from .data import observation_pool

    _, manifest = observation_pool("validation")
    sequence = STUSequence.open(
        data_root,
        protocol=load_protocol(),
        partition="train",
        sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    sequence._cache_frames = 1
    arrays = []
    for r in manifest["segments"]:
        with np.load(r["file"], allow_pickle=False) as payload:
            arrays.append(
                {
                    k: payload[k].copy()
                    for k in ("frame_offsets", "changed_slots", "changed_xyzi")
                }
            )
    samples = json.loads((PROJECT_ROOT / config["monitor"]).read_text())["windows"]
    selected = {
        key: {
            s["current_frame"]
            for s in samples
            if s["view"] == ("normal" if key == "normal" else "synthetic")
            and (key == "normal" or s["sequence_index"] == key)
        }
        for key in ["normal", *range(8)]
    }
    repetitions = {
        key: Counter(t for c in currents for t in range(c - 4, c + 1))
        for key, currents in selected.items()
    }
    full = Counter(t for c in range(4, 682) for t in range(c - 4, c + 1))
    predictions = records = monitor_predictions = monitor_records = 0
    for t in range(682):
        raw = sequence.source_frame(t)
        target = evaluation_targets(raw.xyzi[:, :3], raw.labels.semantic)
        if np.any(target == 1):
            raise ValueError("normal 201 contains anomaly supervision")
        for key in ["normal", *range(8)]:
            normal = int((target == 0).sum())
            anomaly = 0
            visible = ~raw.zero_slot_mask.copy()
            if key != "normal":
                a = arrays[key]
                first, last = a["frame_offsets"][t : t + 2]
                slots, xyz = (
                    a["changed_slots"][first:last],
                    a["changed_xyzi"][first:last, :3],
                )
                visible[slots] = np.any(xyz != 0, axis=1)
                normal -= int((target[slots] == 0).sum())
                distance = np.linalg.norm(xyz, axis=1)
                anomaly = int(((distance >= 2.5) & (distance <= 50)).sum())
            slots = np.flatnonzero(visible).astype(np.int32)
            bound = int(4.004 * len(slots)) + sum(
                len(zlib.compress(v.tobytes(), 1)) + 128
                for v in (
                    np.diff(slots, prepend=np.int32(0)),
                    np.full(len(slots), t, np.int32),
                )
            )
            predictions += full[t] * bound
            monitor_predictions += repetitions[key][t] * bound
            current = 8 * (
                normal + anomaly if anomaly >= 5 else normal if key == "normal" else 0
            )
            records += current if t >= 4 else 0
            monitor_records += current if t in selected[key] else 0
        if (t + 1) % 200 == 0:
            print(
                json.dumps(
                    dict(event="nre_storage_inventory", source=201, frames=t + 1)
                ),
                flush=True,
            )
    real = PROJECT_ROOT / "runs/eval/v1/real"
    rows = {
        (r["sequence_index"], r["current_frame"]): r
        for r in [
            json.loads(line)
            for line in (real / "results.jsonl").read_text().splitlines()
        ]
    }
    for sample in (s for s in samples if s["view"] == "real"):
        r = rows[(sample["sequence_index"], sample["current_frame"])]
        with zipfile.ZipFile(real / r["prediction"]["file"]) as archive:
            identities = sum(
                archive.getinfo(k + ".npy").compress_size
                for k in ("source_frame", "source_slot_delta")
            )
        monitor_predictions += int(4.004 * r["point_count"]) + identities
        monitor_records += (
            8 * r["evaluation_records"]["count"] + 4 * r["normal_records"]["count"]
        )
    real_peak = json.loads((real / "inventory.json").read_text())["budget"][
        "peak_new_bytes"
    ]
    labels = json.loads((PROJECT_ROOT / config["labels"]).read_text())
    # Original backbone + exact NRE head size; eight states include recovery and atomic replacement.
    head_parameters = (
        117 * 96
        + 96
        + 192
        + 96 * 64
        + 64
        + len(labels["active_groups"]) * (4 * 64 + 1)
        + 64 * 32
        + 32
        + 32
        + 1
    )
    initial = torch.load(
        PROJECT_ROOT / config["initial_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    backbone_parameters = sum(
        v.numel() for k, v in initial["model"].items() if k.startswith("backbone.")
    )
    parts = dict(
        four_monitors=4 * (monitor_predictions + monitor_records + 464 * 8192),
        full_synthetic_and_normal=predictions + records + 6102 * 8192,
        checkpoints_recovery_and_logs=8 * 12 * (backbone_parameters + head_parameters)
        + 128 * 2**20,
        final_real_peak=real_peak,
        additional_buffers=512 * 2**20,
    )
    return dict(
        source="completed v2 slots and retained complete real input inventory; no score compression assumed",
        manifest_sha256=file_hash(
            PROJECT_ROOT / "artifacts/data/v2/validation/manifest.json"
        ),
        monitor_sha256=file_hash(PROJECT_ROOT / config["monitor"]),
        score_kind="logit",
        parts=parts,
        synthetic_sort_bytes=records,
        peak_new_bytes=sum(parts.values()) + max(0, records - real_peak),
        host_disk=host_disk(),
    )


def nre_candidate(checkpoint, payload, recipe, *, interim=False):
    """Distinguish a healthy completed monitor from the final selected candidate."""
    plan_path = checkpoint.parent / "plan.json"
    plan = json.loads(plan_path.read_text())
    state = payload["state"]
    if (
        payload["config"]["purpose"] != "AJAE-NRE"
        or payload["config"]["nre"] != recipe
        or payload["plan_sha256"] != file_hash(plan_path)
        or state["next_position"] != 0
        or payload["config"]["voxel_size"] != 0.05
        or state["status"] == "numerical_error"
        or any(not torch.isfinite(v).all() for v in payload["model"].values())
    ):
        raise ValueError("NRE evaluation requires a healthy completed candidate")
    if interim:
        if not any(
            c["visit"] == state["planned_attempts"]
            and Path(c["checkpoint"]).resolve() == checkpoint.resolve()
            for c in state["monitor_candidates"]
        ):
            raise ValueError(
                "interim evaluation requires a completed monitor checkpoint"
            )
    else:
        selection = json.loads((checkpoint.parent / "selection.json").read_text())
        if (
            selection["checkpoint_sha256"] != file_hash(checkpoint)
            or Path(selection["selected"]["checkpoint"]).resolve()
            != checkpoint.resolve()
            or [c["visit"] for c in selection["candidates"]]
            != [7120, 14240, 21360, 28480]
            or state["planned_attempts"] != selection["selected"]["visit"]
        ):
            raise ValueError(
                "NRE final evaluation requires the fixed selected candidate"
            )
    return plan


def interim_inventory(data_root, protocol, baseline, check_resources):
    """Select every official scoring frame from the retained full-run observations."""
    rows = [
        json.loads(line)
        for line in (baseline / "results.jsonl").read_text().splitlines()
    ]
    samples = json.loads((baseline / "samples.json").read_text())["samples"]
    if len(rows) != 8659 or len(samples) != len(rows):
        raise ValueError("the interim scope requires the complete retained v1 run")
    selected, predictions = [], 0
    for sample, row in zip(samples, rows, strict=True):
        if any(row[k] != v for k, v in sample.items()):
            raise ValueError("retained v1 observations differ from their sample list")
        if not row["current"]["eligible"]:
            continue
        selected.append(sample)
        with zipfile.ZipFile(baseline / row["prediction"]["file"]) as archive:
            identities = sum(
                archive.getinfo(k + ".npy").compress_size
                for k in ("source_frame", "source_slot_delta")
            )
        predictions += int(4.004 * row["point_count"]) + identities + 8192
    eligible = [r for r in rows if r["current"]["eligible"]]
    if (
        len(selected) != 1960
        or sum(r["current"]["normal_count"] for r in eligible) != 193792470
        or sum(r["current"]["anomaly_count"] for r in eligible) != 87499
        or {s["sequence_index"] for s in selected} != set(protocol.public_sequence_ids)
    ):
        raise ValueError("the complete official scoring population differs")
    sequences = {}
    for index in protocol.public_sequence_ids:
        check_resources()
        sequences[index] = STUSequence.open(
            data_root,
            protocol=protocol,
            partition="val",
            sequence_id=index,
            label_mode=LabelMode.REQUIRED,
        )
    records = 8 * (193792470 + 87499)
    budget = dict(
        predictions=predictions,
        official_records=records,
        normal_records=0,
        temporary_sort=records,
        buffers_and_startup=2**30,
        persistent_bound_bytes=predictions + records + 2**30,
        peak_new_bytes=predictions + 2 * records + 2**30,
    )
    inventory = dict(
        source="retained full v1 observations; only official eligible frames selected",
        baseline_results_sha256=file_hash(baseline / "results.jsonl"),
        baseline_inventory_sha256=file_hash(baseline / "inventory.json"),
        frames=len(selected),
        startup_frames=sum(s["current_frame"] < 4 for s in selected),
        normal_count=193792470,
        anomaly_count=87499,
    )
    return sequences, inventory, selected, budget


def compare_interim(output, baseline, monitor, summary, check_resources):
    """Compare identical official point populations and retain development subsets."""
    rows = [json.loads(s) for s in (output / "results.jsonl").read_text().splitlines()]
    old_rows = [
        json.loads(s) for s in (baseline / "results.jsonl").read_text().splitlines()
    ]
    old = {(r["sequence_id"], r["current_frame"]): r for r in old_rows}
    monitor_rows = [
        json.loads(s) for s in (monitor / "results.jsonl").read_text().splitlines()
    ]
    monitor_ids = {(r["sequence_id"], r["current_frame"]) for r in monitor_rows}
    baseline_summary = json.loads((baseline / "summary.json").read_text())
    if len(rows) != 1960 or {(r["sequence_id"], r["current_frame"]) for r in rows} != {
        key for key, r in old.items() if r["current"]["eligible"]
    }:
        raise ValueError(
            "interim predictions must cover exactly all official eligible frames"
        )
    for row in rows:
        previous = old[(row["sequence_id"], row["current_frame"])]
        if any(
            row[k] != previous[k] for k in ("frame_ids", "check_seed", "point_count")
        ) or any(
            row["current"][k] != previous["current"][k]
            for k in ("eligible", "normal_count", "anomaly_count", "raw_slot_count")
        ):
            raise ValueError("interim and v1 current point populations differ")

    def pair(current, previous):
        if any(current[k] != previous[k] for k in ("normal_count", "anomaly_count")):
            raise ValueError("paired metric point counts differ")
        return {
            metric: dict(
                v1=previous[metric],
                nre=current[metric],
                difference_percentage_points=current[metric] - previous[metric],
            )
            for metric in ("AP", "AUROC", "FPR95")
        }

    subsets = {}
    for name, inside in (("monitor", True), ("outside_monitor", False)):
        selected = [
            r
            for r in rows
            if ((r["sequence_id"], r["current_frame"]) in monitor_ids) == inside
        ]
        metrics = []
        for directory, records, kind in (
            (output, selected, "logit"),
            (
                baseline,
                [old[(r["sequence_id"], r["current_frame"])] for r in selected],
                "probability",
            ),
        ):
            check_resources()
            refs = [r["evaluation_records"] for r in records]
            metrics.append(
                pooled_files(
                    [directory / r["file"] for r in refs],
                    ranges=[(r["offset"], r["count"]) for r in refs],
                    score_kind=kind,
                )
            )
        subsets[name] = dict(
            frame_count=len(selected),
            sequence_count=len({r["sequence_index"] for r in selected}),
            normal_count=metrics[0]["normal_count"],
            anomaly_count=metrics[0]["anomaly_count"],
            comparison=pair(*metrics),
        )
        if len(selected) != (152 if inside else 1808):
            raise ValueError("the fixed monitor partition differs")
    normal_rows = [r for r in monitor_rows if r["current"]["raw_anomaly_count"] == 0]
    previous_normal = [old[(r["sequence_id"], r["current_frame"])] for r in normal_rows]
    normal = json.loads((monitor / "summary.json").read_text())[
        "normal_without_anomaly_returns"
    ]
    if (
        len(normal_rows) != 152
        or sum(r["current"]["normal"]["point_count"] for r in previous_normal)
        != normal["point_count"]
    ):
        raise ValueError("normal monitor populations differ")
    return dict(
        scope="same official scoring points; outside-monitor frames remain development data",
        overall=pair(summary["all_frames"], baseline_summary["all_frames"]),
        subsets=subsets,
        sequences={
            key: pair(
                value["all_frames"], baseline_summary["sequences"][key]["all_frames"]
            )
            for key, value in summary["sequences"].items()
        },
        normal_monitor=dict(
            source=str(monitor),
            interpretation="historical monitor with training-warmed rotary cache; not recomputed by this interim evaluation",
            frame_count=len(normal_rows),
            point_count=normal["point_count"],
            nre_logit_ge_0=normal["fraction_ge_0"],
            v1_probability_ge_0_5=sum(
                r["current"]["normal"]["count_ge_0_5"] for r in previous_normal
            )
            / normal["point_count"],
            full_normal_phases_evaluated=False,
        ),
    )


def run_real(
    data_root, output, *, startup_only=False, checkpoint=None, nre=None, interim=False
):
    from .train import FullResources, write_progress

    protocol = load_protocol()
    if interim and (not nre or checkpoint is None or startup_only):
        raise ValueError("interim evaluation requires one explicit NRE checkpoint")
    rule = protocol.data["real_anomaly_development_validation"]
    checkpoint = checkpoint if nre else PROJECT_ROOT / rule["checkpoint"]
    digest = file_hash(checkpoint, discard_cache=True)
    if not nre and digest != rule["checkpoint_sha256"]:
        raise ValueError("the fixed epoch-seven candidate bytes differ")
    if not torch.cuda.is_available():
        raise RuntimeError("the unchanged inference path requires CUDA")
    torch.set_num_threads(1)
    resources = FullResources(
        lambda event, **values: print(
            json.dumps({"event": event, **values}), flush=True
        )
    )
    snapshot = resources()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    plan = nre_candidate(checkpoint, payload, nre, interim=interim) if nre else None
    candidate = (
        next(
            (
                c
                for c in payload["state"].get("monitor_candidates", [])
                if c["visit"] == payload["state"]["planned_attempts"]
            ),
            None,
        )
        if nre
        else None
    )
    if nre and not interim:
        candidate = json.loads((checkpoint.parent / "selection.json").read_text())[
            "selected"
        ]
    model = (
        AJAE(
            0.05,
            **(
                dict(
                    normal_groups=plan["label_statistics"]["active_groups"],
                    point_chunk_size=nre["model"]["point_chunk_size"],
                )
                if nre
                else {}
            ),
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    model.load_state_dict(payload["model"], strict=True)
    reference = payload["model"]
    del payload
    checks = check_startup(model, data_root, protocol, output, digest)
    assert_unchanged(model, reference)
    if startup_only:
        return checks
    baseline = PROJECT_ROOT / "runs/eval/v1/real"
    sequences, inventory, samples, budget = (
        interim_inventory(data_root, protocol, baseline, resources)
        if interim
        else real_inventory(data_root, protocol, resources)
    )
    volume = host_disk()
    existing = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if (
        volume["SizeRemaining"] - max(0, budget["peak_new_bytes"] - existing)
        < volume["reserve_bytes"]
    ):
        raise OSError(f"real-val peak output would invade E: reserve: {budget}")
    _path = output / "inventory.json"
    record = dict(
        sequences=inventory,
        budget=budget,
        resources=snapshot,
        host_disk_before_inference=volume,
        startup_checks_sha256=file_hash(output / "startup/checks.json"),
    )
    if interim:
        # The interim sorting array is temporary; its saved predictions remain during training.
        seen, retained = set(), 0
        for p in (PROJECT_ROOT / nre["paths"]["training"]).rglob("*"):
            if p.is_file():
                stat = p.stat()
                if (stat.st_dev, stat.st_ino) not in seen:
                    seen.add((stat.st_dev, stat.st_ino))
                    retained += stat.st_size
        remaining = max(0, plan["storage"]["peak_new_bytes"] - retained)
        combined = max(
            budget["peak_new_bytes"], budget["persistent_bound_bytes"] + remaining
        )
        record["subsequent_training_and_evaluation"] = dict(
            remaining_original_bound=remaining,
            interim_persistent_bound=budget["persistent_bound_bytes"],
            combined_peak_new_bytes=combined,
            projected_free_bytes=volume["SizeRemaining"] - max(0, combined - existing),
        )
        if (
            record["subsequent_training_and_evaluation"]["projected_free_bytes"]
            < volume["reserve_bytes"]
        ):
            raise OSError(
                "interim plus remaining full plan would invade the E: reserve"
            )
    if not _path.exists():
        _atomic_json(_path, record)
    else:
        previous = json.loads(_path.read_text())
        if previous["sequences"] != inventory or previous["budget"] != budget:
            raise ValueError(
                "real input identity or storage estimate changed on resume"
            )
    print(
        json.dumps({"event": "real_ready", "frames": len(samples), "budget": budget}),
        flush=True,
    )
    summary = evaluate_samples(
        model,
        sequences,
        samples,
        output,
        identity=dict(
            scope="real_val",
            checkpoint=str(checkpoint),
            sha256=digest,
            model_sha256=model_digest(model),
            inventory_sha256=file_hash(_path),
            protocol_sha256=file_hash(protocol.path),
            **(
                dict(
                    interim=True,
                    purpose="official scoring coverage at an intermediate checkpoint",
                )
                if interim
                else {}
            ),
        ),
        check_resources=resources,
        reuse=reusable_predictions(
            [
                Path(candidate["evaluation"]) / "real",
                PROJECT_ROOT / nre["paths"]["evaluation"] / "interim_14240",
            ],
            model_digest(model),
        )
        if nre
        else None,
    )
    if interim:
        summary["scope"] = "official_eligible_development_interim"
        summary.pop("normal_without_anomaly_returns", None)
        for sequence in summary["sequences"].values():
            sequence.pop("normal_without_anomaly_returns", None)
        summary["v1_comparison"] = compare_interim(
            output, baseline, Path(candidate["evaluation"]) / "real", summary, resources
        )
    elif nre and checkpoint.stem == "visit_14240":
        earlier = (
            PROJECT_ROOT / nre["paths"]["evaluation"] / "interim_14240/summary.json"
        )
        if earlier.exists():
            prior = json.loads(earlier.read_text())["all_frames"]
            if any(
                summary["all_frames"][k] != prior[k]
                for k in ("AP", "AUROC", "FPR95", "normal_count", "anomaly_count")
            ):
                raise ValueError(
                    "selected 14240 full evaluation differs from its identical reused interim scoring points"
                )
            summary["interim_scoring_consistency"] = dict(
                source=str(earlier), metrics_identical=True
            )
    assert_unchanged(model, reference)
    if file_hash(checkpoint, discard_cache=True) != digest:
        raise RuntimeError("candidate changed during real validation")
    summary["final_resources"] = resources()
    write_progress(output / "summary.json", summary)
    return summary


def export_official(data_root, evaluation, sequence_id, output):
    protocol = load_protocol()
    if sequence_id not in protocol.public_sequence_ids:
        raise ValueError("official export is restricted to public val")
    rows = [
        json.loads(line)
        for line in (evaluation / "results.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if r["sequence_index"] == sequence_id]
    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="val",
        sequence_id=sequence_id,
        label_mode=LabelMode.REQUIRED,
    )
    if [r["current_frame"] for r in rows] != list(sequence.frame_ids):
        raise ValueError("official export requires every original frame")
    volume = host_disk()
    bound = sum(r["current"]["raw_slot_count"] for r in rows) * 16 + 2**28
    if volume["SizeRemaining"] - bound < volume["reserve_bytes"]:
        raise OSError("official text export would invade the E: reserve")
    directory = output / str(sequence_id)
    directory.mkdir(parents=True, exist_ok=False)
    for row in rows:
        window = sequence.for_output(row["current_frame"])
        batch = PredictionBatch.load(
            evaluation / row["prediction"]["file"],
            window=window,
            expected_sha256=row["prediction"]["file_sha256"],
        )
        scores = official_current_scores(batch, row["current"]["raw_slot_count"])
        path = directory / f"{row['current_frame']:06d}.txt"
        with path.open("x") as stream:
            np.savetxt(stream, scores, fmt="%.9g")
            stream.flush()
            os.fdatasync(stream.fileno())
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return directory


def run_full(data_root, checkpoint, output, *, nre=None):
    from .train import FullResources

    if not torch.cuda.is_available():
        raise RuntimeError("the unchanged LitePT implementation requires CUDA")
    torch.set_num_threads(1)
    protocol = load_protocol()
    digest = file_hash(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state"]
    formal = payload["config"]["purpose"] == "AJAE-FullTrain-v1"
    if nre:
        from .data import observation_pool

        plan = nre_candidate(checkpoint, payload, nre)
        pool, _ = observation_pool("validation")
        if plan["storage"]["manifest_sha256"] != file_hash(
            PROJECT_ROOT / "artifacts/data/v2/validation/manifest.json"
        ):
            raise ValueError("the v2 validation pool changed since training")
        peak = (
            plan["storage"]["parts"]["full_synthetic_and_normal"]
            + plan["storage"]["synthetic_sort_bytes"]
            + 2**29
        )
    elif formal:
        if (
            not state["completed_epochs"]
            or state["next_position"] != 0
            or state["phase"] == "monitor"
        ):
            raise ValueError(
                "full selection requires a completed training and monitoring epoch"
            )
        plan_path = checkpoint.parent / "plan.json"
        if file_hash(plan_path) != payload["plan_sha256"]:
            raise ValueError("candidate training plan changed")
        selection = json.loads((checkpoint.parent / "selection.json").read_text())
        if checkpoint.stem not in selection["candidate_names"]:
            raise ValueError(
                "only the two predeclared completed candidates may be fully evaluated"
            )
    else:
        comparison = json.loads(
            (PROJECT_ROOT / "runs/history/coverage/check_1280/samples.json").read_text()
        )
        if (
            digest != comparison["checkpoints"]["B"]["sha256"]
            or state["successful_updates"] != 1280
        ):
            raise ValueError("historical full validation requires the fixed B state")
    if payload["config"]["voxel_size"] != 0.05:
        raise ValueError("the fixed voxel size changed")
    if not nre:
        pool, peak = protocol.validation_pool, 13 * 2**30
    samples = full_samples(pool)
    resources = FullResources(
        lambda event, **values: print(
            json.dumps({"event": event, **values}), flush=True
        )
    )
    snapshot = resources()
    # Include full-window predictions and the temporary exact sorting array.
    existing = (
        sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if output.exists()
        else 0
    )
    if (
        snapshot["host_disk"]["SizeRemaining"] - max(0, peak - existing)
        < snapshot["host_disk"]["reserve_bytes"]
    ):
        raise OSError("complete candidate evaluation would invade the E: reserve")
    dataset = FrozenWindowDataset(
        data_root, protocol, pool_name="validation", version="v2" if nre else "v1"
    )
    model = (
        AJAE(
            0.05,
            **(
                dict(
                    normal_groups=plan["label_statistics"]["active_groups"],
                    point_chunk_size=nre["model"]["point_chunk_size"],
                )
                if nre
                else {}
            ),
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    model.load_state_dict(payload["model"], strict=True)
    assert_unchanged(model, payload["model"])
    del payload
    gc.collect()
    try:
        result = evaluate_samples(
            model,
            dataset,
            samples,
            output,
            identity={
                "checkpoint": str(checkpoint.resolve()),
                "sha256": digest,
                "scope": "complete_201",
                "model_sha256": model_digest(model),
            },
            check_resources=resources,
        )
        if file_hash(checkpoint) != digest:
            raise RuntimeError("candidate checkpoint changed during inference")
    finally:
        del model, dataset
        gc.collect()
        torch.cuda.empty_cache()
    return result


COUNT_BINS = ((5, 20), (20, 100), (100, 500), (500, None))
DISTANCE_BINS = ((2.5, 10), (10, 20), (20, 35), (35, 50))


def diagnostic_bin(count, distance):
    if count < 5:
        return None
    if distance is None or not 2.5 <= distance <= 50:
        raise ValueError("eligible anomalies require an official-range distance")
    i = int(np.searchsorted([20, 100, 500], count, side="right"))
    j = int(np.searchsorted([10, 20, 35], distance, side="right"))
    return f"{i}_{j}"


def diagnostic_frames(data_root, protocol, directory, rows, *, synthetic=False):
    """Read current observations only; saved inference supplies every score."""
    if synthetic:
        dataset = FrozenWindowDataset(data_root, protocol, pool_name="validation")
    else:
        source_sequence = STUSequence.open(
            data_root,
            protocol=protocol,
            partition="val",
            sequence_id=rows[0]["sequence_index"],
            label_mode=LabelMode.REQUIRED,
        )
    frames = []
    for row in rows:
        current = row["current"]
        count = current["anomaly_count"]
        frame = {
            "domain": "synthetic" if synthetic else "real",
            "sequence_id": row["sequence_id"],
            "current_frame": row["current_frame"],
            "world_identity": None,
            "normal_count": current["normal_count"],
            "anomaly_count": count,
            "eligible": current["eligible"],
            "anomaly_distance_median": None,
            "stratum": None,
            "detected_anomaly_ge_0_5": 0,
            "evaluation_records": row["evaluation_records"],
        }
        if synthetic:
            segment, start = dataset.segment_for_window(row["dataset_index"])
            if (
                start + 4 != row["current_frame"]
                or segment.metadata["synthetic_sequence_id"] != row["sequence_id"]
            ):
                raise ValueError("saved prediction and synthetic observation disagree")
            frame["world_identity"] = segment.metadata["world_identity"]
        if count:
            source = (
                segment.frame(row["current_frame"])
                if synthetic
                else source_sequence.source_frame(row["current_frame"])
            )
            target = evaluation_targets(source.xyzi[:, :3], source.labels.semantic)
            if (
                int((target == 1).sum()) != count
                or int((target == 0).sum()) != current["normal_count"]
                or current["eligible"] != (count >= 5)
            ):
                raise ValueError(
                    "current observation counts differ from saved evaluation"
                )
            distance = float(
                np.median(np.linalg.norm(source.xyzi[target == 1, :3], axis=1))
            )
            frame["anomaly_distance_median"] = distance
            frame["stratum"] = diagnostic_bin(count, distance)
            if current["eligible"]:
                record = row["evaluation_records"]
                with (directory / record["file"]).open("rb") as stream:
                    stream.seek(record["offset"] * 8)
                    values = np.fromfile(stream, dtype=np.uint64, count=record["count"])
                # Preserve the original point order, labels and exact float32 scores.
                np.testing.assert_array_equal(values & 1, target[target != -1])
                scores = (values >> 1).astype(np.uint32).view(np.float32)
                if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                    raise ValueError("saved evaluation contains invalid scores")
                detected = int(np.count_nonzero((values & 1) & (scores >= 0.5)))
            elif synthetic:
                window = segment.window(start)
                batch = PredictionBatch.load(
                    directory / row["prediction"]["file"], window=window
                )
                anomaly = target[batch.source_slot[batch.online_mask]] == 1
                detected = int(
                    np.count_nonzero(
                        batch.anomaly_score[batch.online_mask][anomaly] >= 0.5
                    )
                )
            else:
                detected = current["anomaly"]["count_ge_0_5"]
            frame["detected_anomaly_ge_0_5"] = detected
        frames.append(frame)
    print(
        json.dumps(
            {
                "event": "diagnostic_observations",
                "domain": frames[0]["domain"],
                "sequence": "all" if synthetic else rows[0]["sequence_id"],
                "frames": len(frames),
            }
        ),
        flush=True,
    )
    return frames


def diagnostic_coverage(frames):
    sequences = sorted({r["sequence_id"] for r in frames})
    worlds = sorted({r["world_identity"] for r in frames if r["world_identity"]})
    positive = sum(r["anomaly_count"] for r in frames)
    negative = sum(r["normal_count"] for r in frames)
    return dict(
        frame_count=len(frames),
        sequence_count=len(sequences),
        sequences=sequences,
        world_count=len(worlds),
        worlds=worlds,
        anomaly_count=positive,
        normal_count=negative,
        prevalence=positive / (positive + negative) if positive + negative else None,
        detected_anomaly_ge_0_5=sum(r["detected_anomaly_ge_0_5"] for r in frames),
        frames_without_detection_ge_0_5=sum(
            r["detected_anomaly_ge_0_5"] == 0 for r in frames
        ),
    )


def diagnostic_group(directory, frames, prevalence):
    records = [r["evaluation_records"] for r in frames]
    metrics = pooled_files(
        [directory / r["file"] for r in records],
        ranges=[(r["offset"], r["count"]) for r in records],
        prevalence=prevalence,
    )
    coverage = diagnostic_coverage(frames)
    for key in ("anomaly_count", "normal_count"):
        if metrics[key] != coverage[key]:
            raise ValueError(
                "pooled scores differ from the stratum's observation counts"
            )
    return {**coverage, **metrics}


def run_diagnostic(data_root, output):
    """One fixed prevalence/count/distance comparison, with no model loading."""
    started = time.monotonic()
    protocol = load_protocol()
    directories = dict(
        synthetic=PROJECT_ROOT / "runs/eval/v1/synthetic",
        real=PROJECT_ROOT / "runs/eval/v1/real",
    )
    manifests = {
        name: json.loads((p / "samples.json").read_text())
        for name, p in directories.items()
    }
    summaries = {
        name: json.loads((p / "summary.json").read_text())
        for name, p in directories.items()
    }
    checkpoint = protocol.data["real_anomaly_development_validation"][
        "checkpoint_sha256"
    ]
    if any(m["identity"]["sha256"] != checkpoint for m in manifests.values()) or (
        len({m["identity"]["model_sha256"] for m in manifests.values()}) != 1
    ):
        raise ValueError("diagnosis requires the same fixed epoch-seven predictions")
    rows = {}
    for name, directory in directories.items():
        records = [
            json.loads(line)
            for line in (directory / "results.jsonl").read_text().splitlines()
        ]
        rows[name] = [
            r for r in records if r["view"] == name and r["current_frame"] >= 4
        ]
        expected = [
            r
            for r in manifests[name]["samples"]
            if r["view"] == name and r["current_frame"] >= 4
        ]
        keys = [(r["sequence_id"], r["current_frame"]) for r in rows[name]]
        if len(set(keys)) != len(keys) or set(keys) != {
            (r["sequence_id"], r["current_frame"]) for r in expected
        }:
            raise ValueError("diagnostic input is not the complete saved sample set")
        if summaries[name]["status"] != "completed":
            raise ValueError("diagnosis requires completed inference")
    reference = summaries["real"]["full_history"]
    positive, negative = reference["anomaly_count"], reference["normal_count"]
    prevalence = positive / (positive + negative)
    record_bytes = sum(
        r["evaluation_records"]["count"] * 8 for domain in rows.values() for r in domain
    )
    disk = host_disk()
    # Disjoint strata together need at most the total record size, even in parallel.
    peak_bytes = record_bytes + 64 * 2**20
    if disk["SizeRemaining"] - peak_bytes < disk["reserve_bytes"]:
        raise OSError("diagnostic sorting would invade the host E: reserve")
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    workers = min(4, len(os.sched_getaffinity(0)), max(1, available // (2 * 2**30)))
    spec = dict(
        checkpoint_sha256=checkpoint,
        model_sha256=manifests["real"]["identity"]["model_sha256"],
        prediction_directories={k: str(v) for k, v in directories.items()},
        reference_anomaly_count=positive,
        reference_normal_count=negative,
        reference_prevalence=prevalence,
        count_bins=COUNT_BINS,
        distance_bins=DISTANCE_BINS,
        distance_closure="left_closed_right_open_except_50_included",
        fpr_limit=0.01,
        score_rule="score >= threshold; complete ties; no interpolation; null threshold means +inf",
        weights="positive pi/P, negative (1-pi)/N separately within every reported pool",
        scope="synthetic complete 201; real val current_frame >= 4; official eligible points and frames",
        official_real_all_frames=summaries["real"]["all_frames"],
        interpretation="descriptive conditional comparison, not causal matching or an official score",
        host_disk=disk,
        predicted_peak_extra_bytes=peak_bytes,
        available_memory_bytes=available,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        workers=workers,
        numeric_threads=1,
        model_forward_calls=0,
        optimizer_updates=0,
    )
    _atomic_json(
        output / "spec.json", spec
    )  # Persist the requested rules before metrics.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        synthetic = pool.submit(
            diagnostic_frames,
            data_root,
            protocol,
            directories["synthetic"],
            rows["synthetic"],
            synthetic=True,
        )
        real = [
            pool.submit(
                diagnostic_frames,
                data_root,
                protocol,
                directories["real"],
                [r for r in rows["real"] if r["sequence_id"] == sequence],
            )
            for sequence in sorted({r["sequence_id"] for r in rows["real"]})
        ]
        frames = dict(
            synthetic=synthetic.result(),
            real=[r for task in real for r in task.result()],
        )
    with (output / "frames.jsonl").open("x") as stream:
        for domain in frames.values():
            for frame in domain:
                stream.write(json.dumps(frame) + "\n")
    eligible = {
        name: [r for r in domain if r["eligible"]] for name, domain in frames.items()
    }
    counts = diagnostic_coverage(eligible["real"])
    if (counts["anomaly_count"], counts["normal_count"]) != (positive, negative):
        raise ValueError(
            "reference prevalence differs from the full-history observations"
        )
    result = dict(overall={}, strata={}, few_points={}, zero_anomaly_frames={})
    for name, domain in frames.items():
        result["few_points"][name] = diagnostic_coverage(
            [r for r in domain if 1 <= r["anomaly_count"] <= 4]
        )
        result["zero_anomaly_frames"][name] = sum(
            r["anomaly_count"] == 0 for r in domain
        )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {
            name: pool.submit(diagnostic_group, directories[name], domain, prevalence)
            for name, domain in eligible.items()
        }
        for name, task in jobs.items():
            result["overall"][name] = task.result()
        for name, previous in (
            ("synthetic", summaries["synthetic"]["synthetic"]),
            ("real", reference),
        ):
            for key in ("AP", "AUROC", "FPR95"):
                if abs(result["overall"][name][key] - previous[key]) > 1e-10:
                    raise ValueError(
                        "reused scores do not reproduce the original pooled metrics"
                    )
        jobs = {}
        for i in range(4):
            for j in range(4):
                key = f"{i}_{j}"
                jobs[key] = {
                    name: pool.submit(
                        diagnostic_group,
                        directories[name],
                        [r for r in domain if r["stratum"] == key],
                        prevalence,
                    )
                    for name, domain in eligible.items()
                }
        for key, tasks in jobs.items():
            cell = {name: task.result() for name, task in tasks.items()}
            cell["common_coverage"] = all(c["frame_count"] > 0 for c in cell.values())
            result["strata"][key] = cell
            print(
                json.dumps(
                    {
                        "event": "diagnostic_stratum",
                        "stratum": key,
                        "common_coverage": cell["common_coverage"],
                    }
                ),
                flush=True,
            )
    result.update(
        status="completed",
        reference_prevalence=prevalence,
        wall_seconds=time.monotonic() - started,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        final_host_disk=host_disk(),
    )
    _atomic_json(output / "summary.json", result)
    plot_diagnostic(output, result)
    print(
        json.dumps(
            {"event": "diagnostic_completed", "wall_seconds": result["wall_seconds"]}
        ),
        flush=True,
    )
    return result


def plot_diagnostic(output, summary):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt

    for family, filename in (
        ("SimSun", "simsun.ttc"),
        ("Times New Roman", "times.ttf"),
    ):
        path = Path("/mnt/c/Windows/Fonts") / filename
        if path.exists():
            font_manager.fontManager.addfont(path)
        font_manager.findfont(family, fallback_to_default=False)
    with plt.rc_context(
        {"font.family": ["Times New Roman", "SimSun"], "pdf.fonttype": 42}
    ):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad("0.88")
        for row, (domain, name) in enumerate((("synthetic", "合成"), ("real", "真实"))):
            for column, (metric, title) in enumerate(
                (
                    ("AUROC", "AUROC"),
                    ("standardized_AP", "统一占比 AP"),
                    ("recall_at_fpr_limit", "误报率不超过 1% 时的召回率"),
                )
            ):
                ax = axes[row, column]
                values = np.full((4, 4), np.nan)
                for i in range(4):
                    for j in range(4):
                        cell = summary["strata"][f"{i}_{j}"][domain]
                        value = cell[metric]
                        if isinstance(value, dict):
                            value = value["recall"]
                        values[i, j] = np.nan if value is None else value
                        label = (
                            "空"
                            if value is None
                            else f"{value:.2f}\n{cell['frame_count']} 帧"
                        )
                        ax.text(
                            j,
                            i,
                            label,
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white"
                            if value is not None and value < 45
                            else "black",
                        )
                chart = ax.imshow(values, vmin=0, vmax=100, cmap=cmap)
                ax.set_title(f"{name}：{title}", fontsize=12)
                ax.set_xticks(
                    range(4), ["[2.5, 10)", "[10, 20)", "[20, 35)", "[35, 50]"]
                )
                ax.set_yticks(range(4), ["5–19", "20–99", "100–499", "≥500"])
                ax.set_xlabel("异常点距离中位数（米）")
                if column == 0:
                    ax.set_ylabel("当前帧异常点数")
        fig.colorbar(chart, ax=axes, shrink=0.8, label="百分比")
        fig.suptitle(
            "第七轮候选的条件分层诊断\n真实区间从第 4 帧开始；各组统一异常占比为 0.0451738%；空表示没有合格帧",
            fontsize=14,
        )
        fig.savefig(output / "results.pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoints", type=Path, default=Path("runs/history/learning")
    )
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--diagnose", action="store_true")
    mode.add_argument("--export-official", type=Path)
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--nre", action="store_true")
    parser.add_argument(
        "--interim",
        action="store_true",
        help="cover all official eligible real frames at an explicit healthy NRE checkpoint",
    )
    args = parser.parse_args()
    if args.nre and (not (args.real or args.full) or args.checkpoint is None):
        parser.error("--nre requires --real or --full and an explicit --checkpoint")
    if args.interim and (not (args.nre and args.real) or args.startup_only):
        parser.error("--interim requires --nre --real and cannot use --startup-only")
    recipe = (
        json.loads((PROJECT_ROOT / "protocols/nre/config.json").read_text())
        if args.nre
        else None
    )
    if args.diagnose:
        run_diagnostic(
            args.data_root, args.output or Path("runs/diagnostics/v1/conditions")
        )
    elif args.real:
        run_real(
            args.data_root,
            args.output
            or Path("runs/eval/nre" if args.nre else "runs/eval/v1")
            / (
                args.checkpoint.stem.replace("visit_", "interim_")
                if args.interim
                else "real"
            ),
            startup_only=args.startup_only,
            checkpoint=args.checkpoint,
            nre=recipe,
            interim=args.interim,
        )
    elif args.export_official:
        if args.sequence is None or args.output is None:
            parser.error("official export requires --sequence and --output")
        export_official(
            args.data_root, args.export_official, args.sequence, args.output
        )
    elif args.full:
        run_full(
            args.data_root,
            args.checkpoint or Path("runs/history/coverage/B/final.pt"),
            args.output
            or Path(
                "runs/eval/nre/synthetic" if args.nre else "runs/history/validation"
            ),
            nre=recipe,
        )
    else:
        run(
            args.data_root,
            args.checkpoints,
            args.output or Path("runs/history/transfer"),
        )
