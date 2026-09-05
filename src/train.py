"""Fixed eight-window fitting diagnostic, not validation or full-pool training."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import FrozenWindowDataset, PredictionBatch
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


def shuffled_schedule(seed=23):
    # A private generator isolates sampling from network and evaluation RNGs.
    generator = np.random.default_rng(seed)
    return [int(i) for _ in range(25) for i in generator.permutation(8)]


def select_windows(dataset):
    if not dataset.gradient_updates_allowed or dataset.pool.source_sequence_id != 206:
        raise ValueError("only the frozen 206 training pool may update parameters")
    windows = []
    for segment, current in zip(SEGMENTS, CURRENT_FRAMES, strict=True):
        index = sum(len(dataset.pool.window_starts(s)) for s in range(segment + 1)) - 1
        window = dataset[index]
        if (
            window.observation_sequence_id != dataset.pool.synthetic_sequence_id(0)
            or window.current_frame_id != current
            or window.frame_ids != tuple(range(current - 4, current + 1))
        ):
            raise ValueError("the predeclared training window identity differs")
        windows.append(window)
    return windows


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


def evaluate(model, samples, step, output, emit):
    primary = []
    for index, (window, inputs, target) in enumerate(samples):
        repeats = []
        for repeat in range(CONFIG["check_repeats"]):
            with fixed_check(model, CONFIG["seed"] + index):
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
                current_frame=window.current_frame_id,
                repeat=repeat,
                prediction=saved,
                **metrics,
            )
            repeats.append((scores, metrics))
        difference = np.abs(repeats[0][0] - repeats[1][0])
        first, second = (item[1] for item in repeats)
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
        primary.append(first)
    aps = [
        item["current"]["AP"] for item in primary if item["current"]["AP"] is not None
    ]
    means = {}
    for key in ("normal", "anomaly", "total"):
        values = [
            item["loss"][key] for item in primary if item["loss"][key] is not None
        ]
        means[key] = float(np.mean(values)) if values else None
    emit(
        "check_summary",
        step=step,
        loss_window_mean=means,
        eligible_windows=len(aps),
        AP_mean=float(np.mean(aps)) if aps else None,
        AP_median=float(np.median(aps)) if aps else None,
        host_disk=host_disk(),
    )


def run(data_root: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("this unchanged LitePT implementation requires CUDA")
    torch.set_num_threads(1)
    protocol = load_protocol()
    dataset = FrozenWindowDataset(data_root, protocol, pool_name="train")
    windows = select_windows(dataset)
    volume = host_disk()
    # All repeats retain all point identities and scores; include checkpoint/temp headroom.
    peak_disk = sum(w.points.count for w in windows) * 12 * 12 + 2 * 2**30
    if volume["SizeRemaining"] - peak_disk < volume["reserve_bytes"]:
        raise OSError("predictions and checkpoints would invade the E: disk reserve")
    seed_all(CONFIG["seed"])
    model = AJAE(CONFIG["voxel_size"]).cuda().train()
    samples = [
        (
            window,
            joint_voxelize(window, model.voxel_size, device="cuda"),
            torch.tensor(window.labels.anomaly_target, device="cuda"),
        )
        for window in windows
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        betas=CONFIG["betas"],
        eps=CONFIG["eps"],
        weight_decay=CONFIG["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=CONFIG["initial_loss_scale"])
    schedule = shuffled_schedule(CONFIG["seed"])
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
        print(line, flush=True)

    state = {
        "planned_attempts": 0,
        "successful_updates": 0,
        "overflow_skips": 0,
        "consecutive_overflows": 0,
        "status": "running",
        "peak_train_allocated_bytes": 0,
        "peak_train_reserved_bytes": 0,
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
        "frozen_segments": [
            record
            for record in dataset.manifest["segments"]
            if record["synthetic_sequence_index"] == 0
            and record["segment_index"] in SEGMENTS
        ],
    }
    emit("configuration", config=CONFIG, identity=identity, schedule=schedule)
    for index, (window, inputs, target) in enumerate(samples):
        emit(
            "sample",
            window_index=index,
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
                    "config": CONFIG,
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
    try:
        save_state("initial.pt")
        # Keep the official preparation path as the numerical reference for reuse.
        window, inputs, _ = samples[0]
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
        del direct, prepared, delta
        evaluate(model, samples, 0, output, emit)
        initial_parameters = {
            name: p.detach().cpu().clone() for name, p in model.named_parameters()
        }
        for step, index in enumerate(schedule, 1):
            begin = time.monotonic()
            state["planned_attempts"] = step
            window, inputs, target = samples[index]
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
            for key, value in (
                ("peak_train_allocated_bytes", torch.cuda.max_memory_allocated()),
                ("peak_train_reserved_bytes", torch.cuda.max_memory_reserved()),
            ):
                state[key] = max(state[key], value)
            emit(
                "train_step",
                step=step,
                window_index=index,
                current_frame=window.current_frame_id,
                seconds=time.monotonic() - begin,
                loss=float(loss),
                class_loss={name: float(value) for name, value in parts.items()},
                **update,
                **state,
            )
            del logits, loss, parts
            optimizer.zero_grad(set_to_none=True)
            if state["consecutive_overflows"] >= CONFIG["consecutive_overflow_limit"]:
                state["status"] = "stopped_consecutive_overflow"
                break
            if step in CONFIG["check_steps"]:
                evaluate(model, samples, step, output, emit)
        else:
            state["status"] = "completed"
    except BaseException as error:
        state["status"] = "stopped_error"
        emit("error", error_type=type(error).__name__, message=str(error), **state)
        raise
    finally:
        signal.alarm(0)
        optimizer.zero_grad(set_to_none=True)
        save_state("final.pt")
        emit("finished", **state)
        log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/learn"))
    args = parser.parse_args()
    run(args.data_root, args.output)
