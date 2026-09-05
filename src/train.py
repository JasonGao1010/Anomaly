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
import tempfile
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

FULL_CONFIG = {
    **{
        key: CONFIG[key]
        for key in (
            "seed",
            "voxel_size",
            "batch_size",
            "gradient_accumulation",
            "optimizer",
            "weight_decay",
            "betas",
            "eps",
            "max_grad_norm",
            "initial_loss_scale",
            "consecutive_overflow_limit",
            "loss",
            "training_precision",
            "inference_precision",
            "augmentation",
        )
    },
    "purpose": "AJAE-FullTrain-v1",
    "epochs": 10,
    "windows_per_epoch": 3080,
    "planned_steps": 30800,
    "warmup_updates": 200,
    "warmup_initial_lr": 3e-5,
    "lr_levels": [3e-4, 1e-4, 3e-5],
    "ap_tolerance_percentage_points": 0.1,
    "plateau_patience_epochs": 2,
    "minimum_epochs": 4,
    "minimum_low_lr_epochs": 2,
    "timeout_seconds": 6 * 3600,
    "recovery_interval": 500,
    "recovery_versions": 2,
    "rss_target_bytes": 6 * 2**30,
    "rss_stop_bytes": 8 * 2**30,
    "minimum_available_memory_bytes": 4 * 2**30,
    "planned_disk_bytes": 35 * 2**30,
    "normal_threshold": 0.5,
    "selection": "AP within 0.1 percentage points of maximum; lower FPR95, lower normal fraction >=0.5, earlier checkpoint",
}


def write_progress(path, payload):
    """Replace mutable progress atomically; immutable evidence uses _atomic_json."""
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def full_learning_rate(state):
    updates = state["successful_updates"]
    if updates < FULL_CONFIG["warmup_updates"]:
        # Update 1 uses the initial rate; update 200 reaches the main rate.
        fraction = updates / (FULL_CONFIG["warmup_updates"] - 1)
        return 3e-5 + (3e-4 - 3e-5) * fraction
    return FULL_CONFIG["lr_levels"][state["lr_level"]]


def advance_full_schedule(state, ap):
    """AP is in percentage units; only a strict >0.1 gain resets stagnation."""
    if ap is None or not np.isfinite(ap):
        raise FloatingPointError("fixed monitoring has no finite pooled AP")
    reference = state["reference_ap"]
    improved = reference is None or ap > reference + 0.1
    if improved:
        state["reference_ap"] = ap
        state["bad_epochs"] = 0
    else:
        state["bad_epochs"] += 1
    if state["lr_level"] == 2:
        state["low_lr_epochs"] += 1
        state["low_lr_bad_epochs"] = 0 if improved else state["low_lr_bad_epochs"] + 1
    stopped = (
        state["completed_epochs"] >= 4
        and state["low_lr_epochs"] >= 2
        and state["low_lr_bad_epochs"] >= 2
    )
    if state["bad_epochs"] >= 2 and state["lr_level"] < 2:
        state["lr_level"] += 1
        state["bad_epochs"] = 0
    return improved, stopped


def choose_candidate(candidates):
    if not candidates or len({c["scope"] for c in candidates}) != 1:
        raise ValueError("candidate selection requires one common evaluation scope")
    for candidate in candidates:
        if any(
            not np.isfinite(candidate[key])
            for key in ("AP", "FPR95", "normal_fraction")
        ):
            raise ValueError("candidate ranking metrics must be finite")
    best = max(candidate["AP"] for candidate in candidates)
    close = [
        candidate for candidate in candidates if best - candidate["AP"] <= 0.1 + 1e-12
    ]
    return min(close, key=lambda c: (c["FPR95"], c["normal_fraction"], c["epoch"]))


class FullResources:
    """Check actual host limits and the persistent training/monitoring time budget."""

    def __init__(
        self, emit=lambda *args, **kwargs: None, elapsed=lambda: 0, *, timed=False
    ):
        self.emit, self.elapsed, self.timed = emit, elapsed, timed
        self.last_host_check = -float("inf")
        self.latest = {}

    def __call__(self):
        if self.timed and self.elapsed() >= FULL_CONFIG["timeout_seconds"]:
            raise TimeoutError(
                "six-hour cumulative training and monitoring budget exhausted"
            )
        memory = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(
                ("MemAvailable:", "MemTotal:", "SwapFree:", "SwapTotal:")
            )
        }
        rss = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        )
        if rss >= FULL_CONFIG["rss_stop_bytes"]:
            raise MemoryError("resident memory reached the 8 GiB stop limit")
        if memory["MemAvailable"] < FULL_CONFIG["minimum_available_memory_bytes"]:
            raise MemoryError(
                "WSL available memory is below 4 GiB; stop further loading"
            )
        if time.monotonic() - self.last_host_check < 60:
            return self.latest
        volume = host_disk()
        host = json.loads(
            subprocess.check_output(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress",
                ],
                text=True,
                timeout=20,
            )
        )
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=20,
        ).strip()
        self.latest = {
            "rss_bytes": rss,
            "wsl_memory": memory,
            "host_memory_kib": host,
            "host_disk": volume,
            "gpu": gpu,
            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(),
            "process_cpu_seconds": time.process_time(),
        }
        self.last_host_check = time.monotonic()
        self.emit("resource", **self.latest)
        if (
            host["FreePhysicalMemory"] * 1024
            < FULL_CONFIG["minimum_available_memory_bytes"]
        ):
            raise MemoryError(
                "Windows available memory is below 4 GiB; stop further loading"
            )
        if volume["SizeRemaining"] < volume["reserve_bytes"] + 2 * 2**30:
            raise OSError("E: reserve plus recovery/sorting headroom is exhausted")
        return self.latest


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


def training_samples(pool, expanded=False, *, full=False):
    """Choose windows by identity, never by labels or model performance."""
    if pool.name != "train" or pool.source_sequence_id != 206:
        raise ValueError("only the frozen 206 training pool may update parameters")
    result = []
    sequences = range(pool.synthetic_sequence_count) if expanded or full else (0,)
    segments = range(len(pool.segments)) if expanded or full else SEGMENTS
    for sequence in sequences:
        for segment in segments:
            starts = pool.window_starts(segment)
            offset = sequence * pool.windows_per_sequence + sum(
                len(pool.window_starts(s)) for s in range(segment)
            )
            for position in range(len(starts)) if full else (len(starts) - 1,):
                current = starts[position] + 4
                result.append(
                    {
                        "synthetic_sequence_index": sequence,
                        "segment_index": segment,
                        "dataset_index": offset + position,
                        "sequence_id": pool.synthetic_sequence_id(sequence),
                        "current_frame": current,
                        "frame_ids": list(range(current - 4, current + 1)),
                        "check_seed": 23 + sequence * len(pool.segments) + segment,
                        **({"view": "synthetic"} if full else {}),
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


def run_fulltrain(data_root, output, initial, *, resume=False, updated_code=False):
    """Execute the predeclared full-pool training and two-stage candidate selection."""
    from .evaluate import (
        assert_unchanged,
        evaluate_samples,
        file_hash,
        full_samples,
        model_digest,
        monitor_samples,
        prepare_window,
        run_full,
        select_full_candidates,
        verify_baseline,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("the unchanged LitePT implementation requires CUDA")
    if output.exists() != resume:
        raise FileExistsError("use a new full-training directory, or explicit --resume")
    if any(
        output.resolve().is_relative_to((PROJECT_ROOT / p).resolve())
        for p in ("runs/learn", "runs/coverage", "runs/validation", "runs/transfer")
    ):
        raise ValueError("formal training must preserve earlier evidence directories")
    torch.set_num_threads(1)
    protocol = load_protocol()
    samples = training_samples(protocol.training_pool, full=True)
    monitors = monitor_samples(protocol.validation_pool)
    if len(samples) != 3080:
        raise ValueError("formal training must use all 3080 frozen windows")
    resources = FullResources()
    snapshot = resources()
    baseline, disk_estimate = verify_baseline(initial, monitors)
    plan_path = output / "plan.json"
    source_names = [
        "protocol.json",
        "src/train.py",
        "src/evaluate.py",
        "src/model.py",
        "src/data.py",
        "src/scene.py",
        "src/protocol.py",
        "artifacts/data/train_manifest.json",
        "artifacts/data/validation_manifest.json",
        "artifacts/data/qualification.json",
    ]
    source_names += [
        str(p.relative_to(PROJECT_ROOT))
        for p in (PROJECT_ROOT / "vendor").rglob("*.py")
    ]
    sources = {name: file_hash(PROJECT_ROOT / name) for name in sorted(source_names)}
    if resume:
        plan = json.loads(plan_path.read_text())
        resumed_payload = torch.load(
            output / "last.pt", map_location="cpu", weights_only=False
        )
        previous_sources = resumed_payload.get(
            "execution_sha256", plan["source_sha256"]
        )
        changed_sources = {
            name: {"previous": previous_sources.get(name), "current": digest}
            for name, digest in sources.items()
            if previous_sources.get(name) != digest
        }
        if changed_sources and (
            not updated_code
            or set(changed_sources)
            - {"src/train.py", "src/evaluate.py", "src/data.py", "src/model.py"}
        ):
            raise ValueError(
                "changed execution requires --updated-code after equivalent-input regression; scientific model/data identities must remain fixed"
            )
        if (
            plan["samples"] != samples
            or plan["monitor_samples"] != monitors
            or plan["baseline"] != baseline
            or plan["config"] != json.loads(json.dumps(FULL_CONFIG))
        ):
            raise ValueError(
                "resume source, data, initialization, or configuration changed"
            )
        written = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        remaining_disk = max(0, plan["estimated_peak_disk_bytes"] - written)
    else:
        generator = np.random.default_rng(23)
        schedule = [int(i) for _ in range(10) for i in generator.permutation(3080)]
        plan = {
            "config": FULL_CONFIG,
            "samples": samples,
            "monitor_samples": monitors,
            "full_validation_samples": full_samples(protocol.validation_pool),
            "schedule": schedule,
            "sampler_random_state": generator.bit_generator.state,
            "source_sha256": sources,
            "baseline": baseline,
            "initial_checkpoint": {
                "file": str(initial.resolve()),
                "sha256": file_hash(initial),
            },
            "estimated_peak_disk_bytes": disk_estimate,
            "resources_before": snapshot,
            "cpu_topology": subprocess.check_output(["lscpu"], text=True),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads": 1,
            "streaming": {
                "active_windows": 1,
                "prefetch_windows": 1,
                "raw_frame_cache": 16,
                "synthetic_frame_cache": 5,
                "sparse_segment_cache_bytes": 256 * 2**20,
            },
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "numpy",
                    "scikit-learn",
                    "spconv-cu126",
                    "flash-attn",
                )
            },
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        remaining_disk = disk_estimate
    volume = host_disk()  # Re-query immediately before any substantial writes.
    if volume["SizeRemaining"] - remaining_disk < volume["reserve_bytes"]:
        raise OSError("measured full-training peak would invade the E: reserve")
    train_data = FrozenWindowDataset(
        data_root, protocol, pool_name="train", segment_cache_bytes=256 * 2**20
    )
    validation_data = FrozenWindowDataset(data_root, protocol, pool_name="validation")
    if (
        not train_data.gradient_updates_allowed
        or validation_data.gradient_updates_allowed
    ):
        raise ValueError("206/201 update roles differ from the frozen protocol")
    if not resume:
        output.mkdir(parents=True)
        _atomic_json(plan_path, plan)
    seed_all(23)
    model = AJAE(0.05).cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    if resume:
        payload = resumed_payload
        if payload["plan_sha256"] != file_hash(plan_path):
            raise ValueError("recovery checkpoint belongs to another training plan")
        state = payload["state"]
        if state["status"] == "numerical_error":
            raise ValueError(
                "numerical failure is preserved; automatic recipe retry is forbidden"
            )
        if state["status"] in ("resource_limit", "interrupted", "execution_error"):
            state.setdefault("interruptions", []).append(
                {
                    key: state.get(key)
                    for key in (
                        "status",
                        "error",
                        "planned_attempts",
                        "elapsed_seconds",
                    )
                }
            )
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        restore_random_state(payload["random_state"])
        if changed_sources:
            state.setdefault("execution_updates", []).append(
                {
                    "next_step": state["planned_attempts"] + 1,
                    "source_changes": changed_sources,
                    "sparse_segment_cache_bytes": 256 * 2**20,
                }
            )
        del resumed_payload
    else:
        payload = torch.load(initial, map_location="cpu", weights_only=False)
        if (
            payload["state"]["planned_attempts"]
            or payload["state"]["successful_updates"]
            or payload["config"]["seed"] != 23
            or payload["config"]["voxel_size"] != 0.05
        ):
            raise ValueError("initial.pt must be the original untrained seed-23 state")
        model.load_state_dict(payload["model"], strict=True)
        assert_unchanged(model, payload["model"])
        restore_random_state(payload["random_state"])
        if optimizer.state or scaler.state_dict() != payload["scaler"]:
            raise RuntimeError("formal optimizer and loss scaler must be fresh")
        state = {
            "planned_attempts": 0,
            "successful_updates": 0,
            "overflow_skips": 0,
            "consecutive_overflows": 0,
            "completed_epochs": 0,
            "next_position": 0,
            "phase": "training",
            "status": "running",
            "lr_level": 0,
            "reference_ap": None,
            "bad_epochs": 0,
            "low_lr_epochs": 0,
            "low_lr_bad_epochs": 0,
            "elapsed_seconds": 0.0,
            "visits": [[0] * 3080 for _ in range(10)],
            "updates": [[0] * 3080 for _ in range(10)],
            "epoch_results": [],
            "monitor_candidates": [],
            "epoch_loss_sums": [
                {key: 0.0 for key in ("normal", "anomaly", "total")} for _ in range(10)
            ],
            "epoch_loss_counts": [
                {key: 0 for key in ("normal", "anomaly", "total")} for _ in range(10)
            ],
        }
    del payload
    gc.collect()
    if resume:
        records = []
        valid_bytes = 0
        with (output / "metrics.jsonl").open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                records.append(json.loads(line))
                valid_bytes += len(line)
        with (output / "metrics.jsonl").open("r+b") as stream:
            stream.truncate(valid_bytes)
        # Preserve the cost of any lost work after the latest periodic checkpoint.
        state["elapsed_seconds"] = max(
            [state["elapsed_seconds"], *[r["elapsed_seconds"] for r in records]]
        )
        last_attempt = next(
            (r["step"] for r in reversed(records) if r["event"] == "train_step"), 0
        )
        state["discarded_attempts_on_recovery"] = state.get(
            "discarded_attempts_on_recovery", 0
        ) + max(0, last_attempt - state["planned_attempts"])
        del records
    session_started, prior_elapsed = time.monotonic(), state["elapsed_seconds"]
    log = (output / "metrics.jsonl").open("a" if resume else "x", buffering=1)

    def elapsed():
        return prior_elapsed + time.monotonic() - session_started

    def emit(event, **values):
        row = {"event": event, "elapsed_seconds": elapsed(), **values}
        line = json.dumps(row, allow_nan=False, separators=(",", ":"))
        log.write(line + "\n")
        if event != "train_step" or values["step"] <= 32 or values["step"] % 100 == 0:
            print(line, flush=True)

    def save_state(path):
        state["elapsed_seconds"] = elapsed()
        checkpoint = {
            "config": FULL_CONFIG,
            "plan_sha256": file_hash(plan_path),
            "state": state,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "random_state": random_state(),
            "schedule": plan["schedule"],
            "execution_sha256": sources,
            "sampler_random_state": plan["sampler_random_state"],
            "next_schedule_index": state["planned_attempts"],
        }
        with tempfile.NamedTemporaryFile(
            dir=output, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                torch.save(checkpoint, stream)
                stream.flush()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    resources = FullResources(emit, elapsed, timed=True)
    interrupted = []
    handlers = {
        s: signal.signal(s, lambda signum, frame: interrupted.append(signum))
        for s in (signal.SIGINT, signal.SIGTERM)
    }

    def check_resources():
        if interrupted:
            raise InterruptedError(
                f"interrupted by signal {interrupted[0]} at a resumable boundary"
            )
        return resources()

    state["status"] = "running" if state["phase"] != "selection" else state["status"]
    state.pop("error", None)
    initial_parameters = None
    if not state["successful_updates"]:
        initial_parameters = {
            name: p.detach().cpu().clone() for name, p in model.named_parameters()
        }
    try:
        emit(
            "start",
            resumed=resume,
            plan_sha256=file_hash(plan_path),
            next_step=state["planned_attempts"] + 1,
            execution_sha256=sources,
            sparse_segment_cache_bytes=256 * 2**20,
        )
        if not resume:
            save_state(output / "last.pt")
        while state["completed_epochs"] < 10 and state["phase"] != "selection":
            epoch = state["completed_epochs"] + 1
            if state["next_position"] == 3080:
                state["phase"] = "monitor"
            if state["phase"] == "training":
                sequence = plan["schedule"][(epoch - 1) * 3080 : epoch * 3080]
                with ThreadPoolExecutor(max_workers=1) as loader:
                    check_resources()
                    prepared = loader.submit(
                        prepare_window,
                        train_data,
                        None,
                        samples[sequence[state["next_position"]]],
                    )
                    for position in range(state["next_position"], 3080):
                        check_resources()
                        begin = time.monotonic()
                        index = sequence[position]
                        sample = samples[index]
                        window, cpu_inputs, load_seconds, prepare_seconds = (
                            prepared.result()
                        )
                        wait_seconds = time.monotonic() - begin
                        del prepared
                        if position + 1 < 3080:
                            prepared = loader.submit(
                                prepare_window,
                                train_data,
                                None,
                                samples[sequence[position + 1]],
                            )
                        begin = time.monotonic()
                        inputs = cpu_inputs.to("cuda")
                        target = torch.tensor(
                            window.labels.anomaly_target, device="cuda"
                        )
                        torch.cuda.synchronize()
                        transfer_seconds = time.monotonic() - begin
                        lr = full_learning_rate(state)
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        optimizer.zero_grad(set_to_none=True)
                        model.train()
                        begin = time.monotonic()
                        with torch.autocast("cuda", dtype=torch.float16):
                            logits = model(window, inputs=inputs)
                            loss, parts = balanced_loss(logits, target)
                        if not torch.isfinite(loss):
                            emit(
                                "failed_window",
                                epoch=epoch,
                                position=position,
                                sample=sample,
                                reason="nonfinite loss",
                                finite_logits=bool(torch.isfinite(logits).all()),
                            )
                            raise FloatingPointError("nonfinite training loss")
                        update = optimizer_update(loss, model, optimizer, scaler)
                        torch.cuda.synchronize()
                        compute_seconds = time.monotonic() - begin
                        state["planned_attempts"] += 1
                        state["next_position"] = position + 1
                        state["visits"][epoch - 1][index] += 1
                        for key, value in {"total": loss, **parts}.items():
                            state["epoch_loss_sums"][epoch - 1][key] += float(
                                value.detach()
                            )
                            state["epoch_loss_counts"][epoch - 1][key] += 1
                        if update["updated"]:
                            state["successful_updates"] += 1
                            state["updates"][epoch - 1][index] += 1
                            state["consecutive_overflows"] = 0
                            if initial_parameters is not None:
                                emit(
                                    "first_parameter_update",
                                    changes=parameter_changes(
                                        model, initial_parameters
                                    ),
                                )
                                initial_parameters = None
                        else:
                            state["overflow_skips"] += 1
                            state["consecutive_overflows"] += 1
                        emit(
                            "train_step",
                            step=state["planned_attempts"],
                            epoch=epoch,
                            position=position,
                            sample=sample,
                            lr=lr,
                            loss=float(loss),
                            class_loss={k: float(v) for k, v in parts.items()},
                            normal_count=int((target == 0).sum()),
                            anomaly_count=int((target == 1).sum()),
                            ignore_count=int((target == -1).sum()),
                            point_count=window.points.count,
                            voxel_count=len(inputs.features),
                            load_seconds=load_seconds,
                            prepare_seconds=prepare_seconds,
                            input_wait_seconds=wait_seconds,
                            transfer_seconds=transfer_seconds,
                            compute_seconds=compute_seconds,
                            successful_updates=state["successful_updates"],
                            **update,
                        )
                        del inputs, cpu_inputs, target, logits, loss, parts, window
                        optimizer.zero_grad(set_to_none=True)
                        if state["planned_attempts"] == 32:
                            resources.last_host_check = -float("inf")
                            emit(
                                "first_32_steps",
                                visits=32,
                                updates=state["successful_updates"],
                                overflow_skips=state["overflow_skips"],
                                wall_seconds=elapsed(),
                            )
                            check_resources()
                        if state["consecutive_overflows"] >= 3:
                            raise FloatingPointError(
                                "three consecutive gradient overflows"
                            )
                        if state["planned_attempts"] % 500 == 0:
                            path = (
                                output / f"recover_{state['planned_attempts']:05d}.pt"
                            )
                            save_state(path)
                            # Keep the latest two scheduled recovery states.
                            for obsolete in sorted(output.glob("recover_*.pt"))[:-2]:
                                obsolete.unlink()
                            temporary = output / "last.link"
                            os.link(path, temporary)
                            os.replace(temporary, output / "last.pt")
                state["phase"] = "monitor"
                save_state(output / "last.pt")
            check_resources()
            monitor_path = output / f"monitor_{epoch:02d}"
            result = evaluate_samples(
                model,
                validation_data,
                monitors,
                monitor_path,
                identity={
                    "plan_sha256": file_hash(plan_path),
                    "epoch": epoch,
                    "model_sha256": model_digest(model),
                    "scope": "fixed_345",
                },
                check_resources=check_resources,
            )
            if state["visits"][epoch - 1] != [1] * 3080:
                raise RuntimeError(
                    "completed epoch did not visit every window exactly once"
                )
            state["completed_epochs"] = epoch
            improved, plateau = advance_full_schedule(state, result["synthetic"]["AP"])
            epoch_path = output / f"epoch_{epoch:02d}.pt"
            candidate = {
                "name": f"epoch_{epoch:02d}",
                "epoch": epoch,
                "scope": "fixed_345",
                "AP": result["synthetic"]["AP"],
                "AUROC": result["synthetic"]["AUROC"],
                "FPR95": result["synthetic"]["FPR95"],
                "normal_fraction": result["normal"]["fraction_ge_0_5"],
                "checkpoint": str(epoch_path.resolve()),
                "evaluation": str(monitor_path.resolve()),
            }
            state["monitor_candidates"].append(candidate)
            state["epoch_results"].append(
                {
                    "epoch": epoch,
                    "visits": sum(state["visits"][epoch - 1]),
                    "updates": sum(state["updates"][epoch - 1]),
                    "training_loss": {
                        key: {
                            "window_count": state["epoch_loss_counts"][epoch - 1][key],
                            "window_mean": state["epoch_loss_sums"][epoch - 1][key]
                            / state["epoch_loss_counts"][epoch - 1][key]
                            if state["epoch_loss_counts"][epoch - 1][key]
                            else None,
                        }
                        for key in ("normal", "anomaly", "total")
                    },
                    "substantive_improvement": improved,
                    "next_lr": full_learning_rate(state),
                    **candidate,
                }
            )
            state["next_position"] = 0
            state["phase"] = "selection" if plateau or epoch == 10 else "training"
            if state["phase"] == "selection":
                state["status"] = (
                    "monitor_plateau"
                    if plateau
                    else "epoch_budget_without_predefined_plateau"
                )
            save_state(epoch_path)
            save_state(output / "last.pt")
            emit("epoch_complete", **state["epoch_results"][-1], status=state["status"])
    except BaseException as error:
        state["status"] = (
            "interrupted"
            if isinstance(error, InterruptedError)
            else "resource_limit"
            if isinstance(
                error, (TimeoutError, MemoryError, OSError, torch.cuda.OutOfMemoryError)
            )
            else "numerical_error"
            if isinstance(error, FloatingPointError)
            else "execution_error"
        )
        state["error"] = f"{type(error).__name__}: {error}"
        emit(
            "stopped",
            status=state["status"],
            error=state["error"],
            next_position=state["next_position"],
        )
    finally:
        optimizer.zero_grad(set_to_none=True)
        save_state(output / "last.pt")
        write_progress(output / "summary.json", state)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        log.close()
    model = optimizer = scaler = train_data = validation_data = initial_parameters = (
        None
    )
    inputs = cpu_inputs = target = logits = loss = parts = window = prepared = None
    gc.collect()
    torch.cuda.empty_cache()
    if not state["monitor_candidates"]:
        return state
    # Partial epochs never enter selection. Resource stops still retain completed candidates.
    best = choose_candidate(state["monitor_candidates"])
    last = state["monitor_candidates"][-1]
    selected = list(
        {candidate["name"]: candidate for candidate in (best, last)}.values()
    )
    selection_path = output / "selection.json"
    if selection_path.exists():
        previous = json.loads(selection_path.read_text())
        if previous["candidate_names"] != [c["name"] for c in selected]:
            raise ValueError(
                "full validation candidates were already fixed differently"
            )
    else:
        _atomic_json(
            selection_path,
            {
                "candidate_names": [c["name"] for c in selected],
                "monitor_best": best["name"],
                "last_completed": last["name"],
                "candidates": selected,
            },
        )
    for candidate in selected:
        run_full(
            data_root,
            Path(candidate["checkpoint"]),
            output / "validation" / candidate["name"],
        )
    comparison = select_full_candidates(output, selected, baseline)
    selection_summary = {
        key: comparison[key]
        for key in ("candidates", "selected", "real_anomaly_evaluated")
    }
    selection_summary["comparison_file"] = str(output / "comparison.json")
    write_progress(output / "summary.json", {**state, "selection": selection_summary})
    return {**state, "selection": selection_summary}


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
    parser.add_argument("--full", action="store_true", help="execute AJAE-FullTrain-v1")
    parser.add_argument(
        "--updated-code",
        action="store_true",
        help="resume a verified equivalent execution update while retaining the original plan",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the same full-training state and cumulative budget",
    )
    args = parser.parse_args()
    if args.resume and not args.full:
        parser.error("--resume requires --full")
    if args.updated_code and not args.resume:
        parser.error("--updated-code requires --resume")
    if args.full:
        if args.coverage:
            parser.error("--full and --coverage are separate experiments")
        run_fulltrain(
            args.data_root,
            args.output or Path("runs/fulltrain_v1"),
            args.initial,
            resume=args.resume,
            updated_code=args.updated_code,
        )
    elif args.coverage:
        run_coverage(
            args.data_root,
            args.output or Path("runs/coverage"),
            args.initial,
            args.validation_samples,
            args.workers,
        )
    else:
        run(args.data_root, args.output or Path("runs/learn"), workers=args.workers)
