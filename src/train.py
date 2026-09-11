"""Train V1 for one complete frozen epoch, then evaluate every prescribed scan."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset

from .data import FrozenDataset, FrozenFrame, FramePrediction, _atomic_json, host_disk
from .evaluate import (evaluate_frames, exact_metrics, official_frame, packed_scores,
                       prediction_frames)
from .model import V1, model_input
from .protocol import load_protocol
from .scene import STUSequence, LabelMode
from .supervision import (auxiliary_path, boundary_loss, detection_loss,
                          load_training_auxiliary, loss_point_weights, sampling_loss, surface_loss,
                          training_identity)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def to_device(arrays):
    return {key: value.cuda(non_blocking=True) for key, value in arrays.items()}


def worker_init(_):
    torch.set_num_threads(1)


def loader(dataset, order, workers, seed):
    # An independent generator keeps worker seeds out of the model's dropout stream.
    return DataLoader(dataset, batch_size=None, sampler=order, num_workers=workers,
                      pin_memory=True, worker_init_fn=worker_init,
                      generator=torch.Generator().manual_seed(seed),
                      **(dict(prefetch_factor=1) if workers else {}))


class PreparedDataset(Dataset):
    def __init__(self, protocol, data_root, split):
        self.protocol, self.split = protocol, split
        self.frozen = FrozenDataset(protocol["dataset"]["directory"], data_root, split)
        self.identity = training_identity(protocol)
        self.input_frame, self.input_cache = None, {}

    def __len__(self):
        return len(self.frozen)

    def __getitem__(self, index):
        path, identity, frame = self.frozen.samples[index]
        original = self.frozen.sequence[frame]
        sample = FrozenFrame.load(path, original, identity)
        if self.input_frame != frame:
            self.input_frame = frame
            self.input_cache.clear()
        auxiliary = load_training_auxiliary(auxiliary_path(self.protocol, self.split, path),
                                            sample, original, path, self.identity)
        views = []
        target = sample.anomaly_target
        for aux in auxiliary:
            slots = aux["source_slot"]
            xyzi = sample.source.xyzi[slots]
            fingerprint = hashlib.sha256(xyzi.tobytes() + slots.tobytes()).hexdigest()
            if fingerprint not in self.input_cache:
                self.input_cache[fingerprint] = model_input(xyzi, slots, self.protocol["model"],
                                              self.protocol["supervision"]["common"]["sampling_scale"])
            scan = self.input_cache[fingerprint]
            supervision = {key: torch.from_numpy(np.ascontiguousarray(value))
                           for key, value in aux.items() if key not in {"source_slot", "phase"}}
            supervision["labels"] = torch.from_numpy(target[slots])
            # Identity is metadata only. The network receives the separate scan dictionary.
            views.append(dict(scan=scan, target=supervision, fingerprint=fingerprint))
        return dict(index=index, world=path.parent.parent.name, identity=identity,
                    frame=frame, source=sample.source, views=views)


class RealDataset(Dataset):
    def __init__(self, protocol, data_root):
        self.protocol = protocol
        self.sequences = {sequence: STUSequence.open(data_root, protocol=load_protocol(),
                          partition="val", sequence_id=sequence, label_mode=LabelMode.FORBIDDEN)
                          for sequence in load_protocol().public_sequence_ids}
        self.samples = [(sequence, frame) for sequence, source in self.sequences.items()
                        for frame in source.frame_ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence, frame = self.samples[index]
        source = self.sequences[sequence][frame]
        if source.labels is not None:
            raise ValueError("real inference must never load labels")
        scan = model_input(source.xyzi[source.real_slots], source.real_slots,
                           self.protocol["model"], self.protocol["supervision"]["common"]["sampling_scale"])
        return dict(source=source, scan=scan)


def loss_view_weights(views):
    """Empty supervision does not dilute other views; a valid zero error still counts."""
    weights = [dict(detection=1 / len(views)) for _ in views]
    for name, mask in (("boundary", "boundary_valid"), ("surface", "surface_valid"),
                       ("sampling", "sampling_consistency_valid")):
        active = []
        for view in views:
            target = view["target"]
            known = (target["labels"] == 0) | (target["labels"] == 1)
            active.append(mask in target and bool((known & target[mask]).any()))
        count = sum(active)
        for weight, present in zip(weights, active):
            weight[name] = int(present) / max(count, 1)
    return weights


def view_losses(output, target, weights, dense=None):
    labels = target["labels"]
    result = dict(detection=detection_loss(output["logits"], labels),
                  boundary=boundary_loss(output["boundary"], target["boundary_distance"],
                                         labels, target["boundary_valid"]),
                  surface=surface_loss(output["surface"], target["surface_offset_z"],
                                       labels, target["surface_valid"]))
    if dense is not None:
        result["sampling"] = sampling_loss(output["logits"], dense, target["dense_row"].long(),
                                            labels, target["sampling_consistency_valid"])
    return {name: value * weights[name] for name, value in result.items()}


def backward_sample(model, sample, scaler, coefficients, accumulation, auxiliary_scale=1.):
    dense, values = None, defaultdict(float)
    for view, weights in zip(sample["views"], loss_view_weights(sample["views"])):
        scan, target = to_device(view["scan"]), to_device(view["target"])
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(scan)
        losses = view_losses(output, target, weights, dense)
        objective = losses["detection"]
        for name, coefficient in coefficients.items():
            if name in losses:
                # The scale changes supervision only; all heads retain detection gradients.
                objective = objective + auxiliary_scale * coefficient * losses[name]
        if not torch.isfinite(objective):
            raise FloatingPointError(f"nonfinite loss at {sample['world']}/{sample['frame']}")
        scaler.scale(objective / accumulation).backward()
        if dense is None:
            # Only C2's teacher is detached; base detection has already backpropagated.
            dense = output["logits"].detach()
        for name, value in losses.items():
            values[name] += float(value.detach())
        del output, scan, target, losses, objective
    return dict(values)


def optimizer_for(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], betas=config["betas"])
    scaler = torch.amp.GradScaler("cuda", init_scale=config["initial_loss_scale"])
    return optimizer, scaler


def update(model, optimizer, scaler, config, attempt):
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.grad is not None]
    # Split PyTorch's clipping operation so an overflowing norm cannot zero finite gradients.
    norm = torch.nn.utils.get_total_norm([p.grad for p in parameters])
    finite_norm = bool(torch.isfinite(norm))
    if finite_norm:
        torch.nn.utils.clip_grads_with_norm_(parameters, config["gradient_clip_norm"], norm)
    else:
        finite_elements = bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())
        if finite_elements:
            raise FloatingPointError("finite gradient elements produced a nonfinite total norm; update refused")
        if not scaler.is_enabled():
            raise FloatingPointError("nonfinite gradient elements without AMP skip protection; update refused")
        # unscale_ already recorded the nonfinite elements; scaler.step must skip this optimizer.
    warm = min(1., attempt / config["warmup_updates"])
    factor = config["warmup_start_factor"] + (1 - config["warmup_start_factor"]) * warm
    for group in optimizer.param_groups:
        group["lr"] = config["learning_rate"] * factor
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    success = scaler.get_scale() >= before
    if success and not finite_norm:
        raise RuntimeError("AMP failed to skip nonfinite gradients")
    return dict(success=success, gradient_norm=float(norm) if finite_norm else None,
                skip_reason=None if success else "nonfinite_gradient_elements",
                loss_scale=scaler.get_scale(), learning_rate=optimizer.param_groups[0]["lr"])


def runtime():
    return dict(cpu_affinity=len(psutil.Process().cpu_affinity()), memory_available=psutil.virtual_memory().available,
                swap_used=psutil.swap_memory().used, gpu=torch.cuda.get_device_name(),
                gpu_free=torch.cuda.mem_get_info()[0], gpu_peak=torch.cuda.max_memory_allocated(),
                disk=host_disk())


def require_space(bytes_needed):
    disk = host_disk()
    if bytes_needed > disk["SizeRemaining"] - disk["reserve_bytes"]:
        raise OSError("this stage's peak writes would enter the physical E: reserve")
    return disk


def execution_config(protocol):
    paths = [Path(__file__), Path(__file__).with_name("model.py")]
    paths.extend(sorted(Path("vendor/litept").rglob("*.py")))
    result = dict(model=protocol["model"], training=protocol["training"],
                supervision_identity=training_identity(protocol),
                dataset_sha256=hashlib.sha256((Path(protocol["dataset"]["directory"]) / "manifest.json").read_bytes()).hexdigest(),
                implementation={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                packages={name: importlib.metadata.version(name) for name in
                          ("torch", "spconv", "torch-scatter", "flash-attn", "numpy", "scipy", "numba")})
    if "auxiliary_scale" in protocol["training"]:
        result["packages"]["timm"] = importlib.metadata.version("timm")
        result["environment"] = dict(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
            driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
            torch_threads=torch.get_num_threads(),
            variables={key: os.environ.get(key) for key in ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    return result


def save_checkpoint(path, model, optimizer, scaler, config, order, visited, successes, history):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                 configuration=config, order=order, visited=visited, successful_updates=successes,
                 attempted_updates=len(history), history=history, epoch=1,
                 complete=visited == len(order), rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    temporary = path.with_suffix(".pending")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_checkpoint(path, model, optimizer, scaler, config, order, *, initial=False):
    state = torch.load(path, map_location="cpu", weights_only=False)
    actual, expected = state["configuration"], config
    if initial:
        if (any(state[key] != 0 for key in ("visited", "successful_updates", "attempted_updates"))
                or state["history"] or state["complete"]):
            raise ValueError("paired initialization must precede every training update")
        actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
        for definition in (actual, expected):
            definition["training"].pop("auxiliary_scale", None)
    if actual != expected or state["order"] != order:
        raise ValueError("resume would change the scientific execution")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state


def assert_state_close(actual, expected, *, atol=0., rtol=0.):
    """Compare checkpoint tensors numerically and all discrete identities exactly."""
    if isinstance(expected, torch.Tensor):
        actual, expected = actual.detach().cpu(), expected.detach().cpu()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return float((actual - expected).abs().max()) if expected.is_floating_point() and expected.numel() else 0.
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        return max((assert_state_close(actual[k], v, atol=atol, rtol=rtol)
                    for k, v in expected.items()), default=0.)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        return max((assert_state_close(a, b, atol=atol, rtol=rtol)
                    for a, b in zip(actual, expected)), default=0.)
    else:
        assert actual == expected
    return 0.


def check_resume(protocol, run, samples, indices, initial):
    """Compare a real V1 next update with and without reconstruction from disk."""
    settings = protocol["training"]
    tolerance = settings["implementation_check"]["resume_tolerance"]
    config = execution_config(protocol)
    accumulation = settings["gradient_accumulation"]
    order = [indices[i % len(indices)] for i in range(2 * accumulation)]
    selected = dict(zip(indices, samples))

    def group(model, optimizer, scaler, start):
        values = defaultdict(float)
        optimizer.zero_grad(set_to_none=True)
        for index in order[start:start + accumulation]:
            for name, value in backward_sample(model, selected[index], scaler,
                            settings["loss_coefficients"], accumulation).items():
                values[name] += value / accumulation
        result = update(model, optimizer, scaler, settings, start // accumulation + 1)
        return dict(indices=order[start:start + accumulation], losses=dict(values), **result)

    seed_all(settings["seed"])
    model = V1(protocol["model"]).cuda().train()
    model.load_state_dict(initial)
    optimizer, scaler = optimizer_for(model, settings)
    with tempfile.TemporaryDirectory(dir=run) as directory:
        path, reference = Path(directory) / "resume.pt", Path(directory) / "continuous.pt"
        first = group(model, optimizer, scaler, 0)
        save_checkpoint(path, model, optimizer, scaler, config, order, accumulation,
                        int(first["success"]), [first])
        second = group(model, optimizer, scaler, accumulation)
        if not first["success"] or not second["success"]:
            raise ValueError("resume check must exercise two actual optimizer updates")
        save_checkpoint(reference, model, optimizer, scaler, config, order, len(order), 2, [first, second])
        del model, optimizer, scaler
        # Construction consumes RNG; restoration must put every stream back at the group boundary.
        model = V1(protocol["model"]).cuda().train()
        optimizer, scaler = optimizer_for(model, settings)
        state = restore_checkpoint(path, model, optimizer, scaler, config, order)
        assert_state_close(model.state_dict(), state["model"])
        assert_state_close(optimizer.state_dict(), state["optimizer"])
        assert_state_close(scaler.state_dict(), state["scaler"])
        assert_state_close(dict(python=random.getstate(), numpy=np.random.get_state(),
                           torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()), state["rng"])
        resumed = group(model, optimizer, scaler, state["visited"])
        save_checkpoint(path, model, optimizer, scaler, config, order, len(order),
                        state["successful_updates"] + int(resumed["success"]), state["history"] + [resumed])
        del state, model, optimizer, scaler
        expected = torch.load(reference, map_location="cpu", weights_only=False)
        actual = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("order", "visited", "successful_updates", "attempted_updates", "complete", "scaler", "rng"):
            assert_state_close(actual[key], expected[key])
        for key in ("indices", "success", "skip_reason", "loss_scale", "learning_rate"):
            assert_state_close(resumed[key], second[key])
        parameter_error = assert_state_close(actual["model"], expected["model"], **tolerance["parameters"])
        optimizer_error = assert_state_close(actual["optimizer"], expected["optimizer"], **tolerance["optimizer"])
        for key in second["losses"]:
            torch.testing.assert_close(torch.tensor(resumed["losses"][key], dtype=torch.float64),
                                       torch.tensor(second["losses"][key], dtype=torch.float64), **tolerance["reductions"])
        torch.testing.assert_close(torch.tensor(resumed["gradient_norm"], dtype=torch.float64),
                                   torch.tensor(second["gradient_norm"], dtype=torch.float64), **tolerance["reductions"])
    return dict(groups=2, accumulation=accumulation, next_group_indices=second["indices"],
                successful_updates=2, parameters_max_absolute_difference=parameter_error,
                optimizer_max_absolute_difference=optimizer_error, tolerance=tolerance,
                learning_rate=second["learning_rate"], restored_start_state_exact=True, scaler_and_rng_exact=True,
                sample_order_and_update_flags_exact=True, continuous_next_update=second,
                resumed_next_update=resumed)


def train_epoch(protocol, data_root, run, *, initial=None):
    seed_all(protocol["training"]["seed"])
    config = execution_config(protocol)
    if initial is None:
        check = json.loads((run / "check.json").read_text())
        if check["configuration"] != config or not check["passed"]:
            raise ValueError("the current implementation has not passed its isolated checks")
    auxiliary = json.loads((Path(protocol["training"]["supervision_directory"]) / "manifest.json").read_text())
    if auxiliary["status"] != "complete" or auxiliary["configuration"] != config["supervision_identity"]:
        raise ValueError("complete final-pool supervision is required before formal updates")
    recorded_config = run / "configuration.json"
    if recorded_config.exists() and json.loads(recorded_config.read_text()) != config:
        raise ValueError("this run directory already records a different formal execution")
    settings = protocol["training"]
    seed_all(settings["seed"])
    dataset = PreparedDataset(protocol, data_root, "train")
    if len(dataset) != settings["base_frames_per_epoch"]:
        raise ValueError("the formal epoch must visit exactly the final root membership")
    order = np.random.default_rng(settings["seed"]).permutation(len(dataset)).tolist()
    model = V1(protocol["model"]).cuda().train()
    optimizer, scaler = optimizer_for(model, settings)
    visited = successes = 0
    history = []
    checkpoint = run / "epoch1.pt"
    if checkpoint.exists():
        state = restore_checkpoint(checkpoint, model, optimizer, scaler, config, order)
        visited, successes, history = state["visited"], state["successful_updates"], state["history"]
        del state
    elif initial is not None:
        state = restore_checkpoint(initial, model, optimizer, scaler, config, order, initial=True)
        for actual, expected in ((model.state_dict(), state["model"]),
                                 (optimizer.state_dict(), state["optimizer"]), (scaler.state_dict(), state["scaler"]),
                                 (dict(python=random.getstate(), numpy=np.random.get_state(),
                                       torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()), state["rng"])):
            assert_state_close(actual, expected)
        _atomic_json(run / "start.json", dict(shared_initial=str(initial), state_exact=True, visited=0,
                     auxiliary_scale=settings["auxiliary_scale"], order=order, configuration=config))
        del state
    if visited == len(order):
        if not (run / "training.json").exists():
            _atomic_json(run / "training.json", dict(base_frames=visited, views=3*visited,
                         attempted_updates=len(history), successful_updates=successes,
                         skipped_updates=len(history)-successes, history=history, resources=runtime()))
        return
    remaining = (settings["storage"]["checkpoint_and_pending_write_bytes"] if initial is not None
                 else settings["storage"]["peak_additional_bytes"] - auxiliary["bytes"])
    require_space(remaining)
    if not checkpoint.exists():
        # Check-run weights are never used: record the untouched formal initialization.
        if initial is None:
            torch.save(dict(model=model.state_dict(), configuration=config), run / "initial.pt")
        _atomic_json(run / "configuration.json", config)
    start = time.monotonic()
    batches = loader(dataset, order[visited:], settings["loader_workers"], settings["seed"])
    loss_sum = defaultdict(float)
    count = 0
    accumulation = min(settings["gradient_accumulation"], len(order) - visited)
    optimizer.zero_grad(set_to_none=True)
    for sample in batches:
        if sample["index"] != order[visited]:
            raise ValueError("epoch visitation order changed")
        values = backward_sample(model, sample, scaler, settings["loss_coefficients"], accumulation,
                                 settings.get("auxiliary_scale", 1.))
        for name, value in values.items():
            loss_sum[name] += value
        visited += 1
        count += 1
        if count == accumulation:
            result = update(model, optimizer, scaler, settings, len(history) + 1)
            successes += int(result["success"])
            history.append(dict(attempt=len(history)+1, visited=visited, **result,
                                losses={name: value/count for name, value in loss_sum.items()}))
            count = 0
            loss_sum.clear()
            accumulation = min(settings["gradient_accumulation"], len(order) - visited)
            if len(history) % settings["checkpoint_interval_updates"] == 0 or visited == len(order):
                save_checkpoint(checkpoint, model, optimizer, scaler, config, order, visited, successes, history)
                status = dict(event="train", visited=visited, total=len(order), successful_updates=successes,
                              last=history[-1], seconds=time.monotonic()-start, resources=runtime())
                print(json.dumps(status), flush=True)
    if count or visited != len(order):
        raise ValueError("epoch ended with an incomplete gradient group or missing frame")
    _atomic_json(run / "training.json", dict(base_frames=visited, views=3*visited,
                 attempted_updates=len(history), successful_updates=successes,
                 skipped_updates=len(history)-successes, history=history,
                 seconds=time.monotonic()-start, resources=runtime()))


def paired_epochs(protocol, data_root, directory):
    """Train both arms from one complete pretraining state, preserving every branch."""
    protocol = copy.deepcopy(protocol)
    settings = protocol["training"]
    if settings["epochs_this_run"] != 1 or not all(protocol["model"]["enhancements"].values()):
        raise ValueError("paired supervision requires one epoch and all unchanged V1 branches")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings.update(run_directory=str(directory), auxiliary_scale=1.)
    seed_all(settings["seed"])
    config = execution_config(protocol)
    dataset = PreparedDataset(protocol, data_root, "train")
    if len(dataset) != settings["base_frames_per_epoch"]:
        raise ValueError("paired epoch membership differs from the prescribed training data")
    order = np.random.default_rng(settings["seed"]).permutation(len(dataset)).tolist()
    initial = directory / "initial.pt"
    if not initial.exists():
        require_space(15_000_000_000)  # Includes both checkpoints and one streaming evaluation.
        model = V1(protocol["model"]).cuda().train()
        optimizer, scaler = optimizer_for(model, settings)
        historical = torch.load(Path("results/v1/initial.pt"), map_location="cpu", weights_only=False)
        assert_state_close(model.state_dict(), historical["model"])
        del historical
        save_checkpoint(initial, model, optimizer, scaler, config, order, 0, 0, [])
        _atomic_json(directory / "experiment.json", dict(question="effect of existing auxiliary supervision",
            arms=dict(joint=1, detection=0), seed=settings["seed"], epochs_per_arm=1,
            initial_weights_equal_historical=True, shared_initial=str(initial),
            original_A_reused=False, configuration=config, resources=runtime(),
            historical_A_limitation="pretraining optimizer, scaler and complete RNG/environment were not recorded",
            interpretation="one paired seed and one epoch; both arms retain conditional geometry and all prediction heads"))
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    del dataset
    for arm, scale in (("joint", 1.), ("detection", 0.)):
        run = directory / arm
        run.mkdir(exist_ok=True)
        settings["auxiliary_scale"] = scale
        recorded = run / "protocol.json"
        if recorded.exists() and json.loads(recorded.read_text()) != protocol:
            raise ValueError("paired run protocol differs from the recorded arm")
        _atomic_json(recorded, protocol)
        print(json.dumps(dict(event="paired_arm", arm=arm, auxiliary_scale=scale)), flush=True)
        torch.cuda.reset_peak_memory_stats()
        train_epoch(protocol, data_root, run, initial=initial)
        torch.cuda.empty_cache()


@torch.inference_mode()
def predict(model, scan):
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(to_device(scan))
    result = {key: value.float().cpu().numpy() for key, value in output.items()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("inference produced nonfinite scores or auxiliary predictions")
    return result


def metric_arrays(scores, labels):
    valid = labels >= 0
    ordered = packed_scores(scores[valid], labels[valid], score_kind="logit")
    ordered.sort()
    return exact_metrics(ordered, score_kind="logit")


def check_summary(model, samples):
    model.eval()
    report = []
    for sample in samples:
        losses, predictions, errors = defaultdict(float), [], {}
        dense = None
        for v, (view, weights) in enumerate(zip(sample["views"], loss_view_weights(sample["views"]))):
            output = predict(model, view["scan"])
            predictions.append(output)
            tensors = {key: torch.from_numpy(value) for key, value in output.items()}
            values = view_losses(tensors, view["target"], weights,
                                 None if dense is None else torch.from_numpy(dense))
            for name, value in values.items():
                losses[name] += float(value)
            add_errors(errors, v, output, view["target"], dense)
            if v == 0:
                dense = output["logits"]
        view = sample["views"][0]
        labels = view["target"]["labels"].numpy()
        constants = {}
        for name, target_key, valid_key in (("boundary", "boundary_distance", "boundary_valid"),
                                            ("surface", "surface_offset_z", "surface_valid")):
            target = view["target"][target_key]
            valid = view["target"][valid_key]
            weights = loss_point_weights(view["target"]["labels"], valid, dtype=torch.float32,
                                        **(dict(boundary_target=target) if name == "boundary" else {}))
            selected = weights > 0
            if selected.any():
                order = torch.argsort(target[selected])
                median = torch.searchsorted(weights[selected][order].cumsum(0), .5 * weights.sum()).clamp_max(len(order)-1)
                constant = float(target[selected][order[median]])
                constants[name] = dict(best_weighted_L1_constant=constant,
                    constant_loss=float(((target-constant).abs()*weights).sum()),
                    prediction_std=float(predictions[0][name][selected.numpy()].std()),
                    prediction_unique=int(len(np.unique(predictions[0][name][selected.numpy()]))))
        report.append(dict(world=sample["world"], frame=sample["frame"],
                           losses=dict(losses), metrics=metric_arrays(dense, labels),
                           class_mean_logit={str(c):float(dense[labels == c].mean()) if np.any(labels == c) else None for c in (0,1)},
                           constant_references=constants, auxiliary=finalize_errors(errors)))
    return report


def implementation_check(protocol, data_root, run):
    settings = protocol["training"]
    specification = settings["implementation_check"]
    configuration = execution_config(protocol)
    previous = run / "check.json"
    if previous.exists():
        record = json.loads(previous.read_text())
        if record["configuration"] == configuration and record["passed"]:
            return
    seed_all(settings["seed"])
    dataset = PreparedDataset(protocol, data_root, "train")
    indices = [next(i for i,(path,_,frame) in enumerate(dataset.frozen.samples)
                    if path.parent.parent.name == row["world"] and frame == row["frame"])
               for row in specification["training_samples"]]
    samples = [dataset[i] for i in indices]
    model = V1(protocol["model"]).cuda().eval()
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    before = check_summary(model, samples)
    point_checks = []
    for sample in samples:
        scan = sample["views"][0]["scan"]
        first, second = predict(model, scan), predict(model, scan)
        if any(not np.array_equal(first[key], second[key]) for key in first):
            raise ValueError("repeated inference differs; exact-input prediction reuse is unsafe")
        # Slot IDs choose deterministic geometric ties only; their numeric values are not features.
        slots = sample["source"].real_slots
        renamed = model_input(sample["source"].xyzi[slots], slots + 1000000,
                              protocol["model"], protocol["supervision"]["common"]["sampling_scale"])
        if any(not torch.equal(scan[key], renamed[key]) for key in scan):
            raise ValueError("numeric source-slot identity leaked into a model input")
        labels = sample["views"][0]["target"]["labels"].numpy()
        for sparse_view in sample["views"][1:]:
            rows = sparse_view["target"]["dense_row"].long()
            if (not torch.equal(sparse_view["scan"]["xyzi"], scan["xyzi"][rows])
                    or not torch.equal(sparse_view["target"]["labels"], sample["views"][0]["target"]["labels"][rows])):
                raise ValueError("a sparse view changed a retained point or its detection label")
        inverse = scan["voxel_inverse"].numpy()
        counts = len(scan["voxel_count"])
        normals = np.bincount(inverse[labels == 0], minlength=counts)
        anomalies = np.bincount(inverse[labels == 1], minlength=counts)
        mixed = np.flatnonzero((normals > 0) & (anomalies > 0))
        differences = []
        for voxel in mixed:
            rows = np.flatnonzero(inverse == voxel)
            a, n = rows[labels[rows] == 1][0], rows[labels[rows] == 0][0]
            differences.append(dict(anomaly_slot=int(slots[a]), normal_slot=int(slots[n]),
                               anomaly_logit=float(first["logits"][a]), normal_logit=float(first["logits"][n])))
        point_checks.append(dict(world=sample["world"], frame=sample["frame"], points=len(labels),
                                 mixed_voxels=len(mixed), mixed_examples=differences[:5],
                                 differing_mixed_scores=sum(r["anomaly_logit"] != r["normal_logit"] for r in differences),
                                 repeated_inference_exact=True, slot_renaming_input_exact=True))
    # Hold context fixed and vary only the within-voxel offset at the point restoration layer.
    fixture = samples[0]["views"][0]["scan"]
    captured = {}
    handle = model.fuse[0].register_forward_pre_hook(lambda module, inputs: captured.update(fused_input=inputs[0]))
    scan = to_device(fixture)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(scan)
    handle.remove()
    if captured["fused_input"].shape[0] != len(scan["xyzi"]):
        raise ValueError("voxelization removed point-level output rows")
    pair = captured["fused_input"][:1].detach().float().repeat(2,1)
    pair[:, -3:] = torch.tensor([[0.,0.,0.],[.25,0.,0.]], device="cuda")
    with torch.autocast("cuda", enabled=False):
        fused = model.fuse(pair)
        b, s = model.boundary(fused).sigmoid(), model.surface(fused)
        scores = model.anomaly(torch.cat((fused,b,s), dim=1)).squeeze(-1)
    if scores[0] == scores[1]:
        raise ValueError("different within-voxel point offsets are forced to share a score")
    offset_check = dict(offsets_in_voxel_units=[[0,0,0],[.25,0,0]], logits=scores.detach().cpu().tolist())
    del pair, fused, b, s, scores
    del output, captured, scan
    gradient_checks = []
    for sample in samples:
        for v, view in enumerate(sample["views"]):
            target = to_device(view["target"])
            hard = (target["labels"] == 1) & ~target["boundary_valid"] & ~target["surface_valid"]
            if v:
                hard &= ~target["sampling_consistency_valid"]
            if not hard.any():
                continue
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(to_device(view["scan"]))
            loss = detection_loss(output["logits"][hard], target["labels"][hard])
            gradients = torch.autograd.grad(loss, (model.anomaly[-1].weight, model.fuse[0].weight))
            norms = [float(g.float().norm()) for g in gradients]
            if not all(np.isfinite(n) and n > 0 for n in norms):
                raise ValueError("auxiliary-invalid anomaly points cannot train the detection path")
            gradient_checks.append(dict(world=sample["world"], frame=sample["frame"], view=v,
                                        anomaly_points=int(hard.sum()), detector_and_fusion_gradient_norm=norms))
            del output, loss, gradients, target
    if not any(row["view"] > 0 for row in gradient_checks):
        raise ValueError("fixed checks did not exercise an anomaly with all three auxiliary masks disabled")
    validation = PreparedDataset(protocol, data_root, "validation")
    special_selection = specification["special_slot_sample"]
    special_index = next(i for i,(path,_,frame) in enumerate(validation.frozen.samples)
                         if path.parent.parent.name == special_selection["world"] and frame == special_selection["frame"])
    special = validation[special_index]
    special_views = []
    for v, view in enumerate(special["views"]):
        first, second = predict(model, view["scan"]), predict(model, view["scan"])
        if any(not np.array_equal(first[key], second[key]) for key in first):
            raise ValueError("201 alias handling changes repeated inference")
        if len(first["logits"]) != len(view["target"]["labels"]):
            raise ValueError("201 aliases lost original detection rows")
        special_views.append(dict(view=v, actual_rows=len(first["logits"]),
                                  distinct_positions=int(len(np.unique(view["scan"]["xyzi"].numpy()[:,:3], axis=0))),
                                  repeated_inference_exact=True))
    del special, validation
    dense = torch.tensor([.2, -.3], device="cuda", requires_grad=True)
    sparse = torch.tensor([-.1, .7], device="cuda", requires_grad=True)
    labels = torch.tensor([0,1], device="cuda")
    consistent = sampling_loss(sparse, dense, torch.arange(2, device="cuda"), labels, torch.ones(2, dtype=torch.bool, device="cuda"))
    teacher_gradient = torch.autograd.grad(consistent, dense, allow_unused=True)[0]
    if teacher_gradient is not None:
        raise ValueError("C2 teacher is not detached")
    if not torch.autograd.grad(detection_loss(dense, labels), dense)[0].abs().sum() > 0:
        raise ValueError("C2 detach disabled base detection gradients")
    # Disable enhancement computations, not the encoder or the direct detection head.
    model.enhancements = {name:False for name in model.enhancements}
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(to_device(samples[0]["views"][0]["scan"]))
    loss = detection_loss(output["logits"], samples[0]["views"][0]["target"]["labels"].cuda())
    base_gradients = torch.autograd.grad(loss, (model.anomaly[-1].weight, model.backbone.embedding.stem.conv.weight))
    base_norms = [float(g.float().norm()) for g in base_gradients]
    if any(n <= 0 or not np.isfinite(n) for n in base_norms):
        raise ValueError("the detector cannot train independently of all three enhancements")
    base_check = dict(points=len(output["logits"]), detector_and_encoder_gradient_norm=base_norms,
                      disabled_enhancement_predictions_zero=bool((output["boundary"] == 0).all() and (output["surface"] == 0).all()))
    del output, loss, base_gradients, model
    real = RealDataset(protocol, data_root)[0]
    if real["source"].labels is not None:
        raise ValueError("real input exposes labels")
    arms = {}
    storage_checks = {}
    for arm in specification["arms"]:
        seed_all(settings["seed"])
        model = V1(protocol["model"]).cuda()
        model.load_state_dict(initial)
        model.eval()
        real_output = predict(model, real["scan"])
        FramePrediction("val", real["source"].sequence_id, real["source"].frame_id,
                        real["source"].real_slots, real_output["logits"]).validate(real["source"])
        optimizer, scaler = optimizer_for(model, settings)
        coefficients = settings["loss_coefficients"] if arm == "joint" else {name:0. for name in settings["loss_coefficients"]}
        history = []
        start = time.monotonic()
        model.train()
        for step in range(specification["updates_per_arm"]):
            sample = samples[step % len(samples)]
            optimizer.zero_grad(set_to_none=True)
            values = backward_sample(model, sample, scaler, coefficients, 1)
            result = update(model, optimizer, scaler, settings, step+1)
            history.append(dict(step=step+1, losses=values, **result))
            if (step+1) % 16 == 0:
                print(json.dumps(dict(event="isolated_check", arm=arm, step=step+1, losses=values,
                                      seconds=time.monotonic()-start)), flush=True)
        after = check_summary(model, samples)
        if arm == "joint":
            with tempfile.TemporaryDirectory(dir=run) as temporary:
                temporary = Path(temporary)
                source = real["source"]
                prediction = FramePrediction("val", source.sequence_id, source.frame_id, source.real_slots, real_output["logits"])
                prediction.save(temporary / "prediction.npz", source)
                restored = FramePrediction.load(temporary / "prediction.npz", source)
                if not np.array_equal(prediction.anomaly_score, restored.anomaly_score):
                    raise ValueError("saving FramePrediction altered a raw model score")
                storage_checks = dict(prediction_scores_exact=True)
        learned = all(b["losses"]["detection"] < a["losses"]["detection"]
                      and b["class_mean_logit"]["1"] > b["class_mean_logit"]["0"]
                      for a,b in zip(before,after))
        arms[arm] = dict(after=after, history=history, detection_learned=learned,
                         successful_updates=sum(r["success"] for r in history), seconds=time.monotonic()-start)
        del model, optimizer, scaler
    resume = check_resume(protocol, run, samples, indices, initial)
    passed = all(arm["detection_learned"] for arm in arms.values())
    _atomic_json(previous, dict(configuration=configuration, passed=passed, scope=specification,
                 before=before, arms=arms, point_checks=point_checks, gradient_checks=gradient_checks,
                 base_only=base_check, isolated_voxel_offset=offset_check,
                 synthetic_validation_201=dict(**special_selection, views=special_views),
                 real_label_forbidden=dict(sequence=real["source"].sequence_id,
                 frame=real["source"].frame_id, actual_points=len(real_output["logits"])),
                 C2_teacher_detached_only_in_consistency=True,
                 storage_checks=storage_checks,
                 resumed_next_update=resume,
                 formal_initialization="fresh seeded module defaults; all check weights discarded", resources=runtime()))
    if not passed:
        raise ValueError("isolated training did not yet demonstrate positive/negative learnability")


def add_errors(report, view_index, output, target, dense=None):
    labels = target["labels"].numpy()
    for name, predicted, truth, mask in (
        ("boundary", output["boundary"], target["boundary_distance"].numpy(), target["boundary_valid"].numpy()),
        ("surface", output["surface"], target["surface_offset_z"].numpy(), target["surface_valid"].numpy()),
    ):
        for category in (0, 1):
            key = f"view_{view_index}/{name}/{'normal' if category == 0 else 'anomaly'}"
            selected = mask & (labels == category)
            entry = report.setdefault(key, dict(valid=0, absolute_error_sum=0., total=0))
            entry["valid"] += int(selected.sum())
            entry["total"] += int((labels == category).sum())
            entry["absolute_error_sum"] += float(np.abs(predicted[selected] - truth[selected]).sum(dtype=np.float64))
            if name == "boundary":
                band = selected & (truth < 1)
                entry["band_valid"] = entry.get("band_valid", 0) + int(band.sum())
                entry["band_absolute_error_sum"] = entry.get("band_absolute_error_sum", 0.) + float(np.abs(predicted[band]-truth[band]).sum(dtype=np.float64))
    if dense is not None:
        # Float32 sigmoid matches the authoritative consistency loss.
        sparse = torch.from_numpy(output["logits"]).sigmoid().numpy()
        base = torch.from_numpy(dense[target["dense_row"].numpy()]).sigmoid().numpy()
        for category in (0, 1):
            key = f"view_{view_index}/sampling/{'normal' if category == 0 else 'anomaly'}"
            selected = target["sampling_consistency_valid"].numpy() & (labels == category)
            entry = report.setdefault(key, dict(valid=0, squared_error_sum=0., retained=0))
            entry["valid"] += int(selected.sum())
            entry["retained"] += int((labels == category).sum())
            entry["squared_error_sum"] += float(np.square(sparse[selected]-base[selected]).sum(dtype=np.float64))


def finalize_errors(errors):
    for entry in errors.values():
        if "absolute_error_sum" in entry:
            entry["MAE"] = entry["absolute_error_sum"] / entry["valid"] if entry["valid"] else None
        if "squared_error_sum" in entry:
            entry["MSE"] = entry["squared_error_sum"] / entry["valid"] if entry["valid"] else None
        if "band_valid" in entry:
            entry["band_MAE"] = entry["band_absolute_error_sum"] / entry["band_valid"] if entry["band_valid"] else None
    return errors


def bind_predictions(protocol, run):
    checkpoint = run / "epoch1.pt"
    binding = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   dataset_sha256=execution_config(protocol)["dataset_sha256"], score="raw_float32_logit")
    path = run / "predictions" / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError("saved predictions belong to a different checkpoint or final dataset")
    else:
        if path.parent.exists() and any(path.parent.rglob("*.npz")):
            raise ValueError("unidentified predictions cannot be resumed")
        _atomic_json(path, binding)
    return binding


def load_trained(protocol, run):
    seed_all(protocol["training"]["seed"])
    state = torch.load(run / "epoch1.pt", map_location="cpu", weights_only=False)
    if not state["complete"] or state["configuration"] != execution_config(protocol):
        raise ValueError("evaluation requires the complete current epoch-end checkpoint")
    model = V1(protocol["model"]).cuda().eval()
    model.load_state_dict(state["model"])
    bind_predictions(protocol, run)
    return model


def evaluate_synthetic(protocol, data_root, run):
    storage = protocol["training"]["storage"]
    stored = sum(p.stat().st_size for p in (run / "predictions" / "synthetic").rglob("*.npz"))
    require_space(max(0, storage["synthetic_prediction_upper_bytes"] - stored) + storage["largest_metric_temporary_bytes"])
    model = load_trained(protocol, run)
    dataset = PreparedDataset(protocol, data_root, "validation")
    if len(dataset) != 13640:
        raise ValueError("synthetic evaluation must contain all 13640 base frames")
    # Frame-major iteration bounds reuse to one normal source frame. Cached predictions
    # depend only on exact current xyzi/slots and the fixed checkpoint, never on labels.
    order = sorted(range(len(dataset)), key=lambda i: (dataset.frozen.samples[i][2], i))
    batches = loader(dataset, order, protocol["training"]["loader_workers"], protocol["training"]["seed"])
    errors, rows, cache = {}, [], {}
    frame_key = None
    reused = count = 0
    start = time.monotonic()
    with tempfile.TemporaryFile(dir=run) as stream:
        for sample in batches:
            if sample["frame"] != frame_key:
                cache.clear()
                frame_key = sample["frame"]
            dense = None
            for v, view in enumerate(sample["views"]):
                key = view["fingerprint"]
                if key in cache:
                    output = cache[key]
                    reused += 1
                else:
                    output = predict(model, view["scan"])
                    cache[key] = output
                add_errors(errors, v, output, view["target"], dense)
                if v == 0:
                    dense = output["logits"]
            source = sample["source"]
            destination = run / "predictions" / "synthetic" / sample["world"] / f"{sample['frame']:06d}.npz"
            prediction = FramePrediction(source.partition, source.sequence_id, source.frame_id, source.real_slots, dense)
            if destination.exists():
                previous = FramePrediction.load(destination, source)
                if not np.array_equal(previous.anomaly_score, dense):
                    raise ValueError("resumed synthetic prediction differs from the same checkpoint")
            else:
                prediction.save(destination, source)
            labels = sample["views"][0]["target"]["labels"].numpy()
            valid = labels >= 0
            packed = packed_scores(dense[valid], labels[valid], score_kind="logit")
            packed.tofile(stream)
            count += len(packed)
            rows.append(dict(world=sample["world"], frame=sample["frame"], actual_points=len(dense),
                             anomaly_points=int((labels == 1).sum()), normal_points=int((labels == 0).sum())))
            if len(rows) % 200 == 0:
                print(json.dumps(dict(event="synthetic", frames=len(rows), total=len(dataset),
                      reused_forwards=reused, seconds=time.monotonic()-start, resources=runtime())), flush=True)
        stream.flush()
        ordered = np.memmap(stream, dtype=np.uint64, mode="r+", shape=(count,))
        ordered.sort(kind="quicksort")
        metrics = exact_metrics(ordered, score_kind="logit")
        del ordered
    _atomic_json(run / "synthetic.json", dict(scope="all valid synthetic detection labels; all 13640 base frames; not real STU scores",
                 frames=len(rows), metrics=metrics, auxiliary=finalize_errors(errors), frame_counts=rows,
                 reused_forwards=reused, seconds=time.monotonic()-start, resources=runtime()))


def infer_real(protocol, data_root, run):
    stored = sum(p.stat().st_size for p in (run / "predictions" / "real").rglob("*.npz"))
    require_space(max(0, protocol["training"]["storage"]["real_prediction_upper_bytes"] - stored))
    model = load_trained(protocol, run)
    dataset = RealDataset(protocol, data_root)
    if len(dataset) != 8659:
        raise ValueError("real inference must contain 19 sequences and all 8659 frames")
    order = []
    for index, (sequence, frame) in enumerate(dataset.samples):
        if not (run / "predictions" / "real" / "val" / str(sequence) / f"{frame:06d}.npz").exists():
            order.append(index)
    start = time.monotonic()
    frames = len(dataset) - len(order)
    for sample in loader(dataset, order, protocol["training"]["loader_workers"], protocol["training"]["seed"]):
        source = sample["source"]
        output = predict(model, sample["scan"])
        prediction = FramePrediction(source.partition, source.sequence_id, source.frame_id, source.real_slots, output["logits"])
        prediction.save(run / "predictions" / "real" / "val" / str(source.sequence_id) / f"{source.frame_id:06d}.npz", source)
        frames += 1
        if frames % 200 == 0:
            print(json.dumps(dict(event="real_inference", frames=frames, total=len(dataset),
                                  seconds=time.monotonic()-start, resources=runtime())), flush=True)
    _atomic_json(run / "inference.json", dict(frames=frames, sequences=len(dataset.sequences),
                 labels_loaded=False, score="unmodified global float32 anomaly logit", resources=runtime()))


def evaluate_real(protocol, data_root, run):
    require_space(protocol["training"]["storage"]["real_prediction_upper_bytes"])
    binding = bind_predictions(protocol, run)
    root = run / "predictions" / "real"
    sequences = load_protocol().public_sequence_ids
    checked = 0
    def resources():
        nonlocal checked
        checked += 1
        if checked % 500 == 0:
            print(json.dumps(dict(event="real_metrics", frames=checked, resources=runtime())), flush=True)
    metrics, rows = evaluate_frames(prediction_frames(data_root, root, sequences), directory=run,
                                   check_resources=resources)
    if len(rows) != 8659:
        raise ValueError("real metrics did not visit all prescribed frames")
    threshold = metrics["recall_at_fpr_limit"]["threshold"]
    groups = {}
    for source, prediction in prediction_frames(data_root, root, sequences):
        scores, target, eligible = official_frame(source, prediction)
        if not eligible:
            continue
        detected = scores >= threshold if threshold is not None else np.zeros(len(scores), bool)
        ranges = np.linalg.norm(source.xyzi[:, :3].astype(np.float64), axis=1)
        masks = {f"sequence/{source.sequence_id}": target >= 0}
        for lo, hi in ((2.5, 10), (10, 20), (20, 35), (35, 50.000001)):
            masks[f"range/{lo:g}-{min(hi,50):g}"] = (ranges >= lo) & (ranges < hi) & (target >= 0)
        for semantic in np.unique(source.labels.semantic[target == 0]):
            masks[f"normal_semantic/{int(semantic)}"] = (target == 0) & (source.labels.semantic == semantic)
        for key, use in masks.items():
            entry = groups.setdefault(key, dict(normal=0, anomaly=0, fp=0, tp=0))
            entry["normal"] += int(np.sum(use & (target == 0)))
            entry["anomaly"] += int(np.sum(use & (target == 1)))
            entry["fp"] += int(np.sum(use & (target == 0) & detected))
            entry["tp"] += int(np.sum(use & (target == 1) & detected))
    _atomic_json(run / "real.json", dict(binding=binding, metrics=metrics, frames=rows,
                 diagnostics=dict(scope="official eligible frames at the single global R@1%FPR threshold",
                                  threshold=threshold, groups=groups), resources=runtime()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("protocol/v1.json"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--stage", choices=("check", "train", "paired", "synthetic", "real", "evaluate", "all"), default="check")
    parser.add_argument("--paired-directory", type=Path, default=Path("results/paired"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    torch.set_num_threads(protocol["training"]["torch_threads"])
    if args.stage == "paired":
        paired_epochs(protocol, args.data_root, args.paired_directory)
        return
    run = Path(protocol["training"]["run_directory"])
    run.mkdir(parents=True, exist_ok=True)
    if args.stage in {"check", "all"}:
        implementation_check(protocol, args.data_root, run)
    if args.stage in {"train", "all"}:
        train_epoch(protocol, args.data_root, run)
    if args.stage in {"synthetic", "all"}:
        evaluate_synthetic(protocol, args.data_root, run)
    if args.stage in {"real", "all"}:
        infer_real(protocol, args.data_root, run)
    if args.stage in {"evaluate", "all"}:
        evaluate_real(protocol, args.data_root, run)


if __name__ == "__main__":
    main()
