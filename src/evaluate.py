"""Zero-update paired diagnostics and full frozen 201 development validation."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
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


def packed_scores(scores, target):
    """Encode exact nonnegative float32 score bits and one binary label, losslessly."""
    scores = np.asarray(scores, dtype=np.float32)
    target = np.asarray(target)
    if (
        scores.shape != target.shape
        or scores.ndim != 1
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or np.any((target != 0) & (target != 1))
    ):
        raise ValueError("metric records require finite scores and binary labels")
    # Positive float bits have the same order as their values; canonicalize -0.
    scores = scores.copy()
    scores[scores == 0] = 0
    return (scores.view(np.uint32).astype(np.uint64) << 1) | target.astype(np.uint64)


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


def exact_metrics(ordered, *, chunk_size=1 << 20):
    """Exact point pooling with bounded RAM; ordered is an ascending uint64 array.

    No score quantization is used. ROC drops the same collinear threshold nodes
    as sklearn's default roc_curve before applying the upstream strict TPR > .95.
    """
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
    }
    if not positive or not negative:
        if positive:
            result.update(AP=100.0)
        return result
    tp = fp = 0
    ap = area = 0.0
    fpr95 = None
    previous = None
    first = True
    for _, counts, pos in score_groups(ordered, chunk_size):
        neg = counts - pos
        tps = tp + np.cumsum(pos, dtype=np.int64)
        fps = fp + np.cumsum(neg, dtype=np.int64)
        recall, fpr = tps / positive, fps / negative
        ap += float(np.sum(np.diff(np.r_[tp / positive, recall]) * tps / (tps + fps)))
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
    return result


def pooled_files(paths, *, normal=False):
    """Sort exact records on disk, then reduce them in bounded chunks."""
    size = sum(path.stat().st_size for path in paths)
    if size % 8:
        raise ValueError("truncated exact evaluation records")
    if not size:
        return (
            normal_statistics(np.empty(0))
            if normal
            else exact_metrics(np.empty(0, np.uint64))
        )
    with tempfile.TemporaryFile(dir=paths[0].parent) as stream:
        stream.truncate(size)
        ordered = np.memmap(stream, dtype=np.uint64, mode="r+", shape=(size // 8,))
        offset = 0
        for path in paths:
            with path.open("rb") as source:
                while len(block := np.fromfile(source, dtype=np.uint64, count=1 << 20)):
                    ordered[offset : offset + len(block)] = block
                    offset += len(block)
        # Numeric in-place quicksort avoids point-count-sized index/ROC arrays.
        ordered.sort(kind="quicksort")
        if normal:

            def quantile(array, q):
                position = (len(array) - 1) * q
                lo, hi = int(np.floor(position)), int(np.ceil(position))
                values = (
                    (np.asarray(array[[lo, hi]]) >> 1)
                    .astype(np.uint32)
                    .view(np.float32)
                )
                return float(
                    np.median(values)
                    if q == 0.5
                    else np.quantile(values, position - lo)
                )

            count = len(ordered) - int(
                np.searchsorted(
                    ordered,
                    packed_scores(np.array([NORMAL_THRESHOLD]), np.array([0]))[0],
                )
            )
            result = {
                "point_count": len(ordered),
                "median": quantile(ordered, 0.5),
                "p95": quantile(ordered, 0.95),
                "count_ge_0_5": count,
                "fraction_ge_0_5": count / len(ordered),
            }
        else:
            result = exact_metrics(ordered)
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
        logits = model(window, inputs=inputs).float()
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - begin
        loss, parts = balanced_loss(logits, target)
        losses = {
            key: float(parts[key]) if key in parts else None
            for key in ("normal", "anomaly")
        }
        losses["total"] = float(loss)
        scopes = None
        if split_losses:
            current = window.current_mask
            official = evaluation_targets(
                window.points.coordinates[current], window.labels.semantic[current]
            )
            scopes = anomaly_losses(logits, target, current, official)
        scores = logits.sigmoid().cpu().numpy()
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


def normal_statistics(scores):
    if not len(scores):
        return {
            "point_count": 0,
            "median": None,
            "p95": None,
            "fraction_ge_0_5": None,
            "count_ge_0_5": 0,
        }
    count = int(np.count_nonzero(scores >= NORMAL_THRESHOLD))
    return {
        "point_count": len(scores),
        "median": float(np.median(scores)),
        "p95": float(np.quantile(scores, 0.95)),
        "count_ge_0_5": count,
        "fraction_ge_0_5": count / len(scores),
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
    selected = {sample["dataset_index"] for sample in select_samples(pool)}
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
                        "check_seed": CHECK_SEED + sequence * 23 + segment,
                        "scope": "selected_23"
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
    if index != 2360 or len(result) != 3038:
        raise ValueError("full 201 validation pool size differs")
    return result


def prepare_window(dataset, partition, sample):
    begin = time.perf_counter()
    window = (
        dataset[sample["dataset_index"]]
        if sample["view"] == "synthetic"
        else partition.for_output(sample["current_frame"])
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
    inputs = joint_voxelize(window, 0.05)
    return window, inputs, loaded, time.perf_counter() - begin


def save_window(output, sample, window, scores, losses, scopes, timings):
    """One bounded writer retains all predictions and only current metric records."""
    begin = time.perf_counter()
    view = sample["view"]
    directory = (
        Path(view)
        if view == "normal"
        else Path(view) / f"{sample['sequence_index']:03d}"
    )
    relative = (
        Path("predictions") / directory / f"frame_{sample['current_frame']:06d}.npz"
    )
    prediction = PredictionBatch.from_window(window, scores).save(
        output / relative, window=window
    )
    prediction["file"] = relative.as_posix()
    # A bounded Python writer must not accumulate unbounded host-backed file pages.
    with (output / relative).open("rb") as stream:
        os.fdatasync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    current = window.current_mask
    xyz, semantic = window.points.coordinates[current], window.labels.semantic[current]
    current_target = evaluation_targets(xyz, semantic)
    calculator = PointOODMetricsCalculator()
    if view == "synthetic":
        metrics = synthetic_metrics(xyz, scores[current], semantic, calculator)
        keys = (
            packed_scores(calculator.all_scores[0], calculator.all_labels[0])
            if metrics["eligible"]
            else np.empty(0, np.uint64)
        )
        subset = "selected" if sample["scope"] == "selected_23" else "remaining"
        metric_path = (
            Path("current")
            / f"{sample['sequence_index']:03d}"
            / f"segment_{sample['segment_index']:02d}_{subset}.bin"
        )
    else:
        values = scores[current][current_target == 0]
        metrics = normal_statistics(values)
        keys = packed_scores(values, np.zeros(len(values), dtype=np.int8))
        metric_path = Path("current/normal.bin")
    path = output / metric_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        offset = stream.tell() // 8
        keys.tofile(stream)
    return {
        **sample,
        "point_count": window.points.count,
        "current_point_count": int(current.sum()),
        "loss": losses,
        "anomaly_loss_scopes": scopes,
        "current": metrics,
        "prediction": prediction,
        "evaluation_records": {
            "file": metric_path.as_posix(),
            "offset": offset,
            "count": len(keys),
            "sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
        },
        **timings,
        "scoring_and_saving_seconds": time.perf_counter() - begin,
    }


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

    def group(items):
        if check_resources is not None:
            check_resources()
        paths = sorted({output / row["evaluation_records"]["file"] for row in items})
        aps = [row["current"]["AP"] for row in items if row["current"]["eligible"]]
        metrics = pooled_files(paths)
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
    for sequence in range(4):
        for segment in range(23):
            items = [
                row
                for row in synthetic
                if row["sequence_index"] == sequence and row["segment_index"] == segment
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
        for name in ("selected_23", "sequence_0_remaining", "sequences_1_3")
    }
    sequences = {
        str(index): group([row for row in synthetic if row["sequence_index"] == index])
        for index in range(4)
    }
    complete = group(synthetic)
    normal_metrics = pooled_files([output / "current/normal.bin"], normal=True)
    world_aps = [world["AP"] for world in worlds if world["AP"] is not None]
    worst = sorted(
        normal, key=lambda row: row["current"]["fraction_ge_0_5"] or 0, reverse=True
    )
    return {
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


def evaluate_samples(model, dataset, samples, output, *, identity, check_resources):
    """The same bounded, zero-update evaluator serves monitoring and full selection."""
    from .train import write_progress

    started = time.perf_counter()
    manifest = {
        "identity": identity,
        "samples": samples,
        "worlds": dataset.manifest["segments"],
    }
    partition = WindowPartition(dataset.source_sequence, 4, 681)
    reference = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    if dataset.gradient_updates_allowed:
        raise RuntimeError("201 must not permit gradient updates")
    rows = []
    if output.exists():
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
            if file_hash(pred) != row["prediction"]["file_sha256"]:
                raise ValueError("committed prediction changed")
            retained.add(pred)
            records = row["evaluation_records"]
            path = output / records["file"]
            if records["offset"] != ends.get(path, 0):
                raise ValueError("evaluation record offsets are discontinuous")
            ends[path] = records["offset"] + records["count"]
            with path.open("rb") as stream:
                stream.seek(records["offset"] * 8)
                block = stream.read(records["count"] * 8)
            if hashlib.sha256(block).hexdigest() != records["sha256"]:
                raise ValueError("committed exact evaluation records changed")
        for path in (output / "current").rglob("*.bin"):
            length = ends.get(path, 0) * 8
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
        output.mkdir(parents=True)
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
                    prepare_window, dataset, partition, samples[len(rows)]
                )
            for index in range(len(rows), len(samples)):
                check_resources()
                sample = samples[index]
                window, cpu_inputs, load_seconds, prepare_seconds = prepared.result()
                prepared = None
                if index + 1 < len(samples):
                    prepared = loader.submit(
                        prepare_window, dataset, partition, samples[index + 1]
                    )
                begin = time.perf_counter()
                inputs = cpu_inputs.to("cuda")
                torch.cuda.synchronize()
                transfer_seconds = time.perf_counter() - begin
                scores, losses, scopes, inference_seconds = predict_window(
                    model, window, inputs, sample["check_seed"], split_losses=True
                )
                timings = {
                    "load_seconds": load_seconds,
                    "prepare_seconds": prepare_seconds,
                    "transfer_seconds": transfer_seconds,
                    "inference_seconds": inference_seconds,
                    "voxel_count": len(inputs.features),
                }
                if pending_write is not None:
                    row = pending_write.result()
                    log.write(
                        json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                    )
                    rows.append(row)
                pending_write = writer.submit(
                    save_window, output, sample, window, scores, losses, scopes, timings
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
        summary = summarize_full(rows, output, check_resources=check_resources)
        status = "completed"
    except BaseException as exception:
        status, error = "stopped_error", f"{type(exception).__name__}: {exception}"
        raise
    finally:
        result = {
            "status": status,
            "error": error,
            "completed_windows": len(rows),
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
    directory = PROJECT_ROOT / "runs/validation"
    manifest = json.loads((directory / "samples.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    plan = json.loads((PROJECT_ROOT / "runs/coverage/plan.json").read_text())
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
        (PROJECT_ROOT / "artifacts/data/validation_manifest.json").read_text()
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


def run_full(data_root, checkpoint, output):
    from .train import FullResources

    if not torch.cuda.is_available():
        raise RuntimeError("the unchanged LitePT implementation requires CUDA")
    torch.set_num_threads(1)
    protocol = load_protocol()
    samples = full_samples(protocol.validation_pool)
    digest = file_hash(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state"]
    formal = payload["config"]["purpose"] == "AJAE-FullTrain-v1"
    if formal:
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
            (PROJECT_ROOT / "runs/coverage/check_1280/samples.json").read_text()
        )
        if (
            digest != comparison["checkpoints"]["B"]["sha256"]
            or state["successful_updates"] != 1280
        ):
            raise ValueError("historical full validation requires the fixed B state")
    if payload["config"]["voxel_size"] != 0.05:
        raise ValueError("the fixed voxel size changed")
    resources = FullResources(
        lambda event, **values: print(
            json.dumps({"event": event, **values}), flush=True
        )
    )
    snapshot = resources()
    # One complete candidate is below 11 GiB plus exact sorting and write buffers.
    existing = (
        sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if output.exists()
        else 0
    )
    if (
        snapshot["host_disk"]["SizeRemaining"] - max(0, 13 * 2**30 - existing)
        < snapshot["host_disk"]["reserve_bytes"]
    ):
        raise OSError("complete candidate evaluation would invade the E: reserve")
    dataset = FrozenWindowDataset(data_root, protocol, pool_name="validation")
    model = AJAE(0.05).cuda().eval().requires_grad_(False)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("runs/learn"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/coverage/B/final.pt")
    )
    args = parser.parse_args()
    if args.full:
        run_full(
            args.data_root, args.checkpoint, args.output or Path("runs/validation")
        )
    else:
        run(args.data_root, args.checkpoints, args.output or Path("runs/transfer"))
