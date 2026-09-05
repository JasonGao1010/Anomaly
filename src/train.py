"""Fixed-sample learning and equal-budget training-coverage diagnostics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import FrozenWindowDataset, PredictionBatch, _atomic_json
from .model import AJAE, joint_voxelize
from .protocol import PROJECT_ROOT, load_protocol
from vendor.stu.compute_point_level_ood import (
    PointOODMetricsCalculator,
    average_precision_score,
)


SEGMENTS = (0, 2, 4, 6, 8, 10, 12, 15)
CURRENT_FRAMES = (27, 83, 139, 195, 251, 307, 363, 448)
CONFIG = {
    "purpose": "training_subset_fitting_diagnostic_only",
    "seed": 23,
    "pool": "train",
    "synthetic_sequence_index": 0,
    "segments": SEGMENTS,
    "current_frames": CURRENT_FRAMES,
    "voxel_size": 0.05,
    "planned_steps": 200,
    "check_steps": (0, 40, 80, 120, 160, 200),
    "check_repeats": 2,
    "check_seed_rule": "23 + window_index, restored after every forward",
    "batch_size": 1,
    "gradient_accumulation": 1,
    "optimizer": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "max_grad_norm": 1.0,
    "initial_loss_scale": 128.0,
    "consecutive_overflow_limit": 3,
    "timeout_seconds": 1800,
    "loss": "mean of the two class means; single-class mean if one class absent",
    "training_precision": "float16_autocast_float32_logits_loss",
    "inference_precision": "existing_AJAE_eval_path",
    "initialization": "random_no_checkpoint",
    "augmentation": None,
}


def balanced_loss(logits, target):
    """Ignore has no loss gradient; each present binary class has equal weight."""
    if logits.shape != target.shape or logits.ndim != 1:
        raise ValueError("logits and targets must be matching point vectors")
    parts = {}
    for name, label in (("normal", 0), ("anomaly", 1)):
        selected = target == label
        if selected.any():
            values = logits[selected].float()
            parts[name] = F.binary_cross_entropy_with_logits(
                values, torch.full_like(values, label)
            )
    if not parts:
        raise ValueError("window has no valid binary supervision")
    return torch.stack(tuple(parts.values())).mean(), parts


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def fixed_check(model, seed):
    """Evaluation must not advance training RNGs or BatchNorm running buffers."""
    state = random_state()
    modes = [(module, module.training) for module in model.modules()]
    buffers = {
        name: value.clone()
        for name, value in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    try:
        seed_all(seed)
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        restore_random_state(state)
        for module, mode in modes:
            module.training = mode
        current = dict(model.named_buffers())
        if any(
            not torch.equal(value, current[name]) for name, value in buffers.items()
        ):
            raise RuntimeError("evaluation changed BatchNorm running statistics")


def shuffled_schedule(seed=23, window_count=8, planned_steps=200):
    # A private generator isolates sampling from network and evaluation RNGs.
    if window_count < 1 or planned_steps < 1 or planned_steps % window_count:
        raise ValueError("planned steps must be a positive number of complete passes")
    generator = np.random.default_rng(seed)
    return [
        int(i)
        for _ in range(planned_steps // window_count)
        for i in generator.permutation(window_count)
    ]


def training_samples(pool, expanded=False):
    """Choose terminal windows by identity, never by labels or model performance."""
    if pool.name != "train" or pool.source_sequence_id != 206:
        raise ValueError("only the frozen 206 training pool may update parameters")
    result = []
    sequences = range(pool.synthetic_sequence_count) if expanded else (0,)
    segments = range(len(pool.segments)) if expanded else SEGMENTS
    for sequence in sequences:
        for segment in segments:
            current = pool.segments[segment].stop - 1
            index = (
                sequence * pool.windows_per_sequence
                + sum(len(pool.window_starts(s)) for s in range(segment + 1))
                - 1
            )
            result.append(
                {
                    "synthetic_sequence_index": sequence,
                    "segment_index": segment,
                    "dataset_index": index,
                    "sequence_id": pool.synthetic_sequence_id(sequence),
                    "current_frame": current,
                    "frame_ids": list(range(current - 4, current + 1)),
                    "check_seed": 23 + sequence * len(pool.segments) + segment,
                }
            )
    return result


def select_windows(dataset):
    if not dataset.gradient_updates_allowed or dataset.pool.source_sequence_id != 206:
        raise ValueError("only the frozen 206 training pool may update parameters")
    windows = []
    for sample in training_samples(dataset.pool):
        window = dataset[sample["dataset_index"]]
        if (
            window.observation_sequence_id != sample["sequence_id"]
            or list(window.frame_ids) != sample["frame_ids"]
        ):
            raise ValueError("the predeclared training window identity differs")
        windows.append(window)
    return windows


def prepare_samples(data_root, protocol, selected, workers):
    """Cache immutable windows/voxel inputs on CPU, with independent reader caches."""
    local = threading.local()
    completed = 0
    lock = threading.Lock()

    def prepare(sample):
        nonlocal completed
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        if available < 3 * 2**30:
            raise MemoryError("CPU input cache would leave insufficient working memory")
        if not hasattr(local, "dataset"):
            local.dataset = FrozenWindowDataset(data_root, protocol, pool_name="train")
        window = local.dataset[sample["dataset_index"]]
        if (
            window.observation_sequence_id != sample["sequence_id"]
            or list(window.frame_ids) != sample["frame_ids"]
        ):
            raise ValueError(
                "prepared input differs from its selected training identity"
            )
        inputs = joint_voxelize(window, CONFIG["voxel_size"])
        target = torch.tensor(window.labels.anomaly_target)
        with lock:
            completed += 1
            if completed % 16 == 0 or completed == len(selected):
                print(
                    json.dumps(
                        {
                            "event": "input_prepared",
                            "completed": completed,
                            "total": len(selected),
                        }
                    ),
                    flush=True,
                )
        return window, inputs, target

    # NumPy geometry and file I/O share no mutable reader state or training RNGs.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(prepare, selected))


def cuda_sample(sample):
    window, inputs, target = sample
    return window, inputs.to("cuda"), target.to("cuda")


def score_distribution(scores, target):
    normal, anomaly = scores[target == 0], scores[target == 1]
    return {
        "normal_median": float(np.median(normal)) if len(normal) else None,
        "normal_p95": float(np.quantile(normal, 0.95)) if len(normal) else None,
        "anomaly_median": float(np.median(anomaly)) if len(anomaly) else None,
    }


def current_metrics(points, scores, semantic):
    """Use the retained official filter and AP, including the >=5 anomaly rule."""
    calculator = PointOODMetricsCalculator()
    calculator.update(points, scores, semantic)
    if not calculator.all_labels:
        return {"AP": None, "ineligible_reason": "fewer_than_5_official_anomaly_points"}
    target, values = calculator.all_labels[0], calculator.all_scores[0]
    return {
        "AP": float(average_precision_score(target, values) * 100),
        "normal_count": int((target == 0).sum()),
        "anomaly_count": int((target == 1).sum()),
        "scores": score_distribution(values, target),
    }


def optimizer_update(loss, model, optimizer, scaler):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None for p in parameters):
        raise RuntimeError("a trainable parameter has no gradient")
    finite = bool(torch.stack([p.grad.isfinite().all() for p in parameters]).all())
    norm = None
    if finite:
        # Inspect unscaled gradients before clipping; overflow is never clipped away.
        norm = torch.nn.utils.get_total_norm([p.grad for p in parameters])
        if not torch.isfinite(norm):
            raise FloatingPointError("finite gradients have a nonfinite total norm")
        torch.nn.utils.clip_grads_with_norm_(parameters, CONFIG["max_grad_norm"], norm)
    calls = []
    hook = optimizer.register_step_post_hook(lambda *_: calls.append(True))
    scale_before = scaler.get_scale()
    try:
        scaler.step(optimizer)
        scaler.update()
    finally:
        hook.remove()
    updated = bool(calls)
    if updated != finite:
        raise RuntimeError("optimizer call and unscaled gradient finiteness disagree")
    return {
        "updated": updated,
        "finite_gradients": finite,
        "grad_norm_before_clip": float(norm) if norm is not None else None,
        "scale_before": scale_before,
        "scale_after": scaler.get_scale(),
    }


def parameter_changes(model, before):
    result = {}
    for prefix in ("backbone", "head"):
        changed, squared = 0, 0.0
        for name, parameter in model.named_parameters():
            if name.startswith(prefix + "."):
                delta = parameter.detach().cpu() - before[name]
                changed += int(torch.count_nonzero(delta))
                squared += float(delta.double().square().sum())
        if not changed:
            raise RuntimeError(f"first successful step did not change {prefix}")
        result[prefix] = {"changed_elements": changed, "delta_l2": squared**0.5}
    return result


def host_disk():
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-Volume -DriveLetter E | Select-Object Size,SizeRemaining | ConvertTo-Json -Compress",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    volume = json.loads(result.stdout)
    reserve = max(volume["Size"] * 0.05, 10 * 2**30)
    if volume["SizeRemaining"] <= reserve:
        raise OSError("host E: has reached the required free-space reserve")
    return {**volume, "reserve_bytes": reserve}


def evaluate(model, samples, step, output, emit, *, repeats=2, selected=None):
    primary = []
    for index, sample in enumerate(samples):
        window, inputs, target = cuda_sample(sample)
        observations = []
        seed = (
            CONFIG["seed"] + index
            if selected is None
            else selected[index]["check_seed"]
        )
        for repeat in range(repeats):
            with fixed_check(model, seed):
                logits = model(window, inputs=inputs).float()
                loss, parts = balanced_loss(logits, target)
                scores = logits.sigmoid().cpu().numpy()
                losses = {
                    name: float(parts[name]) if name in parts else None
                    for name in ("normal", "anomaly")
                }
                losses["total"] = float(loss)
            del logits, loss, parts
            record = PredictionBatch.from_window(window, scores)
            relative = (
                Path("predictions")
                / f"step_{step:03d}"
                / window.observation_sequence_id.rsplit("/", 1)[-1]
                / (f"frame_{window.current_frame_id:06d}_repeat_{repeat}.npz")
            )
            saved = record.save(output / relative, window=window)
            saved["file"] = relative.as_posix()
            mask = window.current_mask
            metrics = {
                "loss": losses,
                "full_window_scores": score_distribution(
                    scores, window.labels.anomaly_target
                ),
                "current": current_metrics(
                    window.points.coordinates[mask],
                    scores[mask],
                    window.labels.semantic[mask],
                ),
            }
            emit(
                "evaluation",
                step=step,
                window_index=index,
                sequence_id=window.observation_sequence_id,
                current_frame=window.current_frame_id,
                repeat=repeat,
                prediction=saved,
                **metrics,
            )
            observations.append((scores, metrics))
        if repeats == 2:
            difference = np.abs(observations[0][0] - observations[1][0])
            first, second = (item[1] for item in observations)
            ap0, ap1 = first["current"]["AP"], second["current"]["AP"]
            emit(
                "repeat_variation",
                step=step,
                current_frame=window.current_frame_id,
                max_abs_score=float(difference.max()),
                mean_abs_score=float(difference.mean()),
                AP_abs_difference=abs(ap0 - ap1)
                if ap0 is not None and ap1 is not None
                else None,
                loss_abs_difference={
                    k: abs(first["loss"][k] - second["loss"][k])
                    if first["loss"][k] is not None
                    else None
                    for k in first["loss"]
                },
            )
        primary.append(observations[0][1])
        del inputs, target, observations, scores, record
    aps = [
        item["current"]["AP"] for item in primary if item["current"]["AP"] is not None
    ]
    means = {}
    for key in ("normal", "anomaly", "total"):
        values = [
            item["loss"][key] for item in primary if item["loss"][key] is not None
        ]
        means[key] = float(np.mean(values)) if values else None
    result = {
        "step": step,
        "window_count": len(primary),
        "loss_window_mean": means,
        "loss_window_median": {
            key: float(
                np.median(
                    [
                        item["loss"][key]
                        for item in primary
                        if item["loss"][key] is not None
                    ]
                )
            )
            if any(item["loss"][key] is not None for item in primary)
            else None
            for key in ("normal", "anomaly", "total")
        },
        "eligible_windows": len(aps),
        "AP_mean": float(np.mean(aps)) if aps else None,
        "AP_median": float(np.median(aps)) if aps else None,
    }
    emit("check_summary", **result, host_disk=host_disk())
    return result


def run(data_root: Path, output: Path, *, group=None, initial=None, workers=1):
    wall_started = time.monotonic()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("this unchanged LitePT implementation requires CUDA")
    torch.set_num_threads(1)
    protocol = load_protocol()
    config = dict(CONFIG)
    if group is not None:
        if group not in ("A", "B") or initial is None:
            raise ValueError(
                "coverage training requires group A/B and the original initial.pt"
            )
        config.update(
            {
                "purpose": "equal_planned_budget_training_coverage_diagnostic",
                "group": group,
                "synthetic_sequence_index": None if group == "B" else 0,
                "segments": tuple(range(16)) if group == "B" else SEGMENTS,
                "current_frames": tuple(
                    s.stop - 1 for s in protocol.training_pool.segments
                )
                if group == "B"
                else CURRENT_FRAMES,
                "planned_steps": 1280,
                "check_steps": (640, 1280),
                "check_repeats": 1,
                "check_seed_rule": "23 + sequence_index * 16 + segment_index (training subset); saved 201 seeds",
                "initialization": "load_original_initial_model_and_RNG_fresh_optimizer_and_scaler",
            }
        )
    selected = training_samples(protocol.training_pool, expanded=group == "B")
    schedule = shuffled_schedule(config["seed"], len(selected), config["planned_steps"])
    volume = host_disk()
    # All repeats retain all point identities and scores; include checkpoint/temp headroom.
    peak_disk = 5 * 2**30 if group is not None else 3 * 2**30
    if volume["SizeRemaining"] - peak_disk < volume["reserve_bytes"]:
        raise OSError("predictions and checkpoints would invade the E: disk reserve")
    begin = time.monotonic()
    print(
        json.dumps(
            {
                "event": "preparing",
                "group": group,
                "window_count": len(selected),
                "workers": workers,
            }
        ),
        flush=True,
    )
    samples = prepare_samples(data_root, protocol, selected, workers)
    preparation_seconds = time.monotonic() - begin
    seed_all(config["seed"])
    model = AJAE(config["voxel_size"]).cuda().train()
    initial_payload = None
    if initial is not None:
        initial_payload = torch.load(initial, map_location="cpu", weights_only=False)
        if (
            initial_payload["state"]["planned_attempts"] != 0
            or initial_payload["state"]["successful_updates"] != 0
            or initial_payload["config"]["voxel_size"] != config["voxel_size"]
        ):
            raise ValueError(
                "coverage groups must start from the original untrained checkpoint"
            )
        model.load_state_dict(initial_payload["model"], strict=True)
        if any(
            not torch.equal(value.cpu(), initial_payload["model"][name])
            for name, value in model.state_dict().items()
        ):
            raise RuntimeError("initial parameters or normalization buffers differ")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=config["initial_loss_scale"])
    if initial_payload is not None:
        # Model construction may consume randomness; reset only after all initialization.
        restore_random_state(initial_payload["random_state"])
        if optimizer.state or scaler.state_dict() != initial_payload["scaler"]:
            raise RuntimeError("optimizer or scaler is not in the original fresh state")
    output.mkdir(parents=True, exist_ok=False)
    log = (output / "metrics.jsonl").open("x", encoding="utf-8", buffering=1)
    started = time.monotonic()

    def emit(event, **values):
        record = {
            "event": event,
            "elapsed_seconds": time.monotonic() - started,
            **values,
        }
        line = json.dumps(record, allow_nan=False, separators=(",", ":"))
        log.write(line + "\n")
        if event != "train_step" or values["step"] % 40 == 0:
            print(line, flush=True)

    state = {
        "planned_attempts": 0,
        "successful_updates": 0,
        "overflow_skips": 0,
        "consecutive_overflows": 0,
        "status": "running",
        "peak_train_allocated_bytes": 0,
        "peak_train_reserved_bytes": 0,
        "processed_points": 0,
        "processed_voxels": 0,
        "training_seconds": 0.0,
        "input_transfer_seconds": 0.0,
    }
    identity = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/train.py",
                "src/model.py",
                "src/evaluate.py",
                "protocol.json",
                "vendor/stu/compute_point_level_ood.py",
            )
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scikit-learn", "spconv-cu126", "flash-attn")
        },
        "GPU": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "host_disk": volume,
        "estimated_peak_disk_bytes": peak_disk,
        "initial_checkpoint": None
        if initial is None
        else {
            "file": str(initial.resolve()),
            "sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
            "parameters_and_buffers_exactly_loaded": True,
            "restored_original_random_state": True,
            "fresh_optimizer_and_scaler": True,
        },
        "preparation_seconds": preparation_seconds,
        "preparation_workers": workers,
        "input_cache": "CPU immutable full windows and deterministic voxel inputs; one window on GPU",
        "selected_samples": selected,
        "frozen_segments": [
            record
            for record in json.loads(
                (PROJECT_ROOT / "artifacts/data/train_manifest.json").read_text()
            )["segments"]
            if (record["synthetic_sequence_index"], record["segment_index"])
            in {
                (sample["synthetic_sequence_index"], sample["segment_index"])
                for sample in selected
            }
        ],
    }
    emit("configuration", config=config, identity=identity, schedule=schedule)
    for index, (window, inputs, target) in enumerate(samples):
        emit(
            "sample",
            window_index=index,
            sequence_id=window.observation_sequence_id,
            current_frame=window.current_frame_id,
            point_count=window.points.count,
            voxel_count=len(inputs.features),
            current_point_count=int(window.current_mask.sum()),
            normal_count=int((target == 0).sum()),
            anomaly_count=int((target == 1).sum()),
            ignore_count=int((target == -1).sum()),
        )

    def save_state(name):
        with (output / name).open("xb") as stream:
            torch.save(
                {
                    "config": config,
                    "identity": identity,
                    "state": dict(state),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "random_state": random_state(),
                    "schedule": schedule,
                    "next_schedule_index": state["planned_attempts"],
                },
                stream,
            )

    def stop_signal(signum, _frame):
        raise InterruptedError(f"experiment interrupted by signal {signum}")

    previous_handlers = {
        s: signal.signal(s, stop_signal)
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
    }
    signal.alarm(CONFIG["timeout_seconds"])
    fit_summary = None
    try:
        if group is None:
            save_state("initial.pt")
        # Keep the official preparation path as the numerical reference for reuse.
        window, inputs, _ = cuda_sample(samples[0])
        with fixed_check(model, CONFIG["seed"]):
            direct = model(window).cpu()
        with fixed_check(model, CONFIG["seed"]):
            prepared = model(window, inputs=inputs).cpu()
        delta = (direct - prepared).abs()
        emit(
            "prepared_input_regression",
            max_abs_logit=float(delta.max()),
            mean_abs_logit=float(delta.mean()),
        )
        torch.testing.assert_close(prepared, direct, atol=1e-6, rtol=1e-5)
        del direct, prepared, delta, inputs, _
        if group is None:
            evaluate(model, samples, 0, output, emit)
        initial_parameters = {
            name: p.detach().cpu().clone() for name, p in model.named_parameters()
        }
        for step, index in enumerate(schedule, 1):
            begin = time.monotonic()
            state["planned_attempts"] = step
            window, inputs, target = cuda_sample(samples[index])
            torch.cuda.synchronize()
            transfer_seconds = time.monotonic() - begin
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(window, inputs=inputs)
                loss, parts = balanced_loss(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            update = optimizer_update(loss, model, optimizer, scaler)
            if update["updated"]:
                state["successful_updates"] += 1
                state["consecutive_overflows"] = 0
                if state["successful_updates"] == 1:
                    emit(
                        "first_parameter_update",
                        step=step,
                        changes=parameter_changes(model, initial_parameters),
                    )
                    del initial_parameters
            else:
                state["overflow_skips"] += 1
                state["consecutive_overflows"] += 1
            torch.cuda.synchronize()
            step_seconds = time.monotonic() - begin
            state["processed_points"] += window.points.count
            state["processed_voxels"] += len(inputs.features)
            state["training_seconds"] += step_seconds
            state["input_transfer_seconds"] += transfer_seconds
            for key, value in (
                ("peak_train_allocated_bytes", torch.cuda.max_memory_allocated()),
                ("peak_train_reserved_bytes", torch.cuda.max_memory_reserved()),
            ):
                state[key] = max(state[key], value)
            emit(
                "train_step",
                step=step,
                window_index=index,
                sequence_id=window.observation_sequence_id,
                current_frame=window.current_frame_id,
                seconds=step_seconds,
                point_count=window.points.count,
                voxel_count=len(inputs.features),
                loss=float(loss),
                class_loss={name: float(value) for name, value in parts.items()},
                **update,
                **state,
            )
            del logits, loss, parts, inputs, target
            optimizer.zero_grad(set_to_none=True)
            if state["consecutive_overflows"] >= CONFIG["consecutive_overflow_limit"]:
                state["status"] = "stopped_consecutive_overflow"
                break
            if step % 80 == 0:
                emit(
                    "resource",
                    step=step,
                    host_disk=host_disk(),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                )
            if step in config["check_steps"]:
                if group is None:
                    evaluate(model, samples, step, output, emit)
                elif step == 640:
                    save_state("step_0640.pt")
        else:
            state["status"] = "completed"
            if group is not None:
                from .evaluate import assert_unchanged

                reference = {
                    name: value.cpu().clone()
                    for name, value in model.state_dict().items()
                }
                fit_summary = evaluate(
                    model, samples, 1280, output, emit, repeats=1, selected=selected
                )
                assert_unchanged(model, reference)
    except BaseException as error:
        state["status"] = "stopped_error"
        emit("error", error_type=type(error).__name__, message=str(error), **state)
        raise
    finally:
        signal.alarm(0)
        optimizer.zero_grad(set_to_none=True)
        save_state("final.pt")
        result = {
            **state,
            "training_subset_evaluation": fit_summary,
            "resources": {
                "wall_seconds": time.monotonic() - wall_started,
                "preparation_seconds": preparation_seconds,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            },
        }
        _atomic_json(output / "summary.json", result)
        emit("finished", **state)
        log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return result


def run_coverage(data_root, output, initial, samples_file, workers):
    """One initialized model, two coverage groups; no decisions from interim scores."""
    from .evaluate import file_hash, run as compare, select_samples

    if output.exists() or any(
        output.resolve().is_relative_to(path.resolve().parent)
        for path in (initial, samples_file)
    ):
        raise FileExistsError(
            "coverage experiment needs a new directory outside prior evidence"
        )
    if not 1 <= workers <= len(os.sched_getaffinity(0)):
        raise ValueError("preparation workers must fit the actual CPU affinity")
    volume = host_disk()
    # 320 full-window predictions + four model/optimizer states, including write buffers.
    if volume["SizeRemaining"] - 5 * 2**30 < volume["reserve_bytes"]:
        raise OSError("coverage experiment output budget would invade the E: reserve")
    protocol = load_protocol()
    prior = json.loads(samples_file.read_text())
    if prior["samples"] != select_samples(protocol.validation_pool):
        raise ValueError("saved 201 samples are not the predeclared development view")
    if file_hash(initial) != prior["checkpoints"]["initial"]["sha256"]:
        raise ValueError("initial model differs from the previous paired diagnostic")
    protected = {
        str(path.resolve()): file_hash(path)
        for path in (
            initial,
            initial.parent / "final.pt",
            samples_file,
            samples_file.parent / "summary.json",
            samples_file.parent / "results.jsonl",
        )
    }
    groups = {
        name: training_samples(protocol.training_pool, expanded=name == "B")
        for name in ("A", "B")
    }
    plan = {
        "purpose": "overall_training_coverage_at_equal_planned_update_budget",
        "primary_step": 1280,
        "check_steps": [640, 1280],
        "planned_steps_per_group": 1280,
        "initial_checkpoint": {
            "file": str(initial.resolve()),
            "sha256": file_hash(initial),
        },
        "validation_manifest": {
            "file": str(samples_file.resolve()),
            "sha256": file_hash(samples_file),
        },
        "validation_samples": prior["samples"],
        "groups": {
            name: {
                "samples": samples,
                "schedule": shuffled_schedule(23, len(samples), 1280),
            }
            for name, samples in groups.items()
        },
        "evaluation_order": "save at 640/1280; finish A then B; pair frozen checkpoints by step",
        "preparation_workers": workers,
        "host_disk_before": volume,
        "estimated_peak_output_bytes": 5 * 2**30,
        "source_sha256": {
            name: file_hash(PROJECT_ROOT / name)
            for name in (
                "src/train.py",
                "src/evaluate.py",
                "src/model.py",
                "protocol.json",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", plan)
    result = {"status": "running", "groups": {}, "checks": {}}
    started = time.monotonic()
    try:
        for name in groups:
            result["groups"][name] = run(
                data_root, output / name, group=name, initial=initial, workers=workers
            )
            if result["groups"][name]["status"] != "completed":
                raise RuntimeError(
                    f"group {name} stopped before completing its fixed budget"
                )
            gc.collect()
            torch.cuda.empty_cache()
        for step in plan["check_steps"]:
            filename = "final.pt" if step == 1280 else "step_0640.pt"
            result["checks"][str(step)] = compare(
                data_root,
                output,
                output / f"check_{step:04d}",
                checkpoint_paths={name: output / name / filename for name in groups},
                samples_file=samples_file,
                expected_attempts=step,
            )
            gc.collect()
            torch.cuda.empty_cache()
        for path, digest in protected.items():
            if file_hash(Path(path)) != digest:
                raise RuntimeError(f"previous experiment evidence changed: {path}")
        result["prior_evidence_unchanged"] = True
        result["equal_successful_updates"] = (
            result["groups"]["A"]["successful_updates"]
            == result["groups"]["B"]["successful_updates"]
        )
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="stopped_error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        _atomic_json(output / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="sequential A/B equal-budget coverage contrast",
    )
    parser.add_argument("--initial", type=Path, default=Path("runs/learn/initial.pt"))
    parser.add_argument(
        "--validation-samples", type=Path, default=Path("runs/transfer/samples.json")
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.coverage:
        run_coverage(
            args.data_root,
            args.output or Path("runs/coverage"),
            args.initial,
            args.validation_samples,
            args.workers,
        )
    else:
        run(args.data_root, args.output or Path("runs/learn"), workers=args.workers)
