"""Zero-update comparison on 23 predeclared synthetic/raw 201 window pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import time

import numpy as np
import torch

from .data import FrozenWindowDataset, PredictionBatch, WindowPartition, _atomic_json
from .model import AJAE, joint_voxelize
from .protocol import PROJECT_ROOT, load_protocol
from .train import balanced_loss, fixed_check, host_disk
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


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(rows, pooled, normal_scores):
    summary = {}
    for name in ("initial", "final"):
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
        a, b = (row["models"][name]["current"] for name in ("initial", "final"))
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
    return {"models": summary, "per_window_AP_changes": changes}


def run(data_root, checkpoints, output):
    started = time.perf_counter()
    if output.exists() or output.resolve().is_relative_to(checkpoints.resolve()):
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
    begin = time.perf_counter()
    dataset = FrozenWindowDataset(data_root, protocol, pool_name="validation")
    if dataset.gradient_updates_allowed:
        raise RuntimeError(
            "the validation data role unexpectedly allows gradient updates"
        )
    partition = WindowPartition(
        dataset.source_sequence, CURRENT_FRAMES[0], CURRENT_FRAMES[-1]
    )
    initialization_seconds = time.perf_counter() - begin
    models, references, checkpoint_records = {}, {}, {}
    begin = time.perf_counter()
    for name, expected_steps in (("initial", 0), ("final", 200)):
        path = checkpoints / f"{name}.pt"
        digest = file_hash(path)
        # Only the two locally produced, user-designated checkpoint files are loaded.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload["state"]["successful_updates"] != expected_steps
            or payload["config"]["voxel_size"] != 0.05
        ):
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
        "worlds": [
            item
            for item in dataset.manifest["segments"]
            if item["synthetic_sequence_index"] == 0
        ],
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
                target = torch.tensor(window.labels.anomaly_target, device="cuda")
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
                    with fixed_check(model, sample["check_seed"]):
                        begin = time.perf_counter()
                        logits = model(window, inputs=inputs).float()
                        torch.cuda.synchronize()
                        inference_seconds = time.perf_counter() - begin
                        begin = time.perf_counter()
                        loss, parts = balanced_loss(logits, target)
                        scores = logits.sigmoid().cpu().numpy()
                        losses = {
                            key: float(parts[key]) if key in parts else None
                            for key in ("normal", "anomaly")
                        }
                        losses["total"] = float(loss)
                    del logits, loss, parts
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
                del inputs, target, window, xyz, semantic, current_target
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
            if (
                file_hash(checkpoints / f"{name}.pt")
                != checkpoint_records[name]["sha256"]
            ):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("runs/learn"))
    parser.add_argument("--output", type=Path, default=Path("runs/transfer"))
    args = parser.parse_args()
    run(args.data_root, args.checkpoints, args.output)
