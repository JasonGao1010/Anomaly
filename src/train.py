"""Train V1 for one complete frozen epoch, then evaluate every prescribed scan."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
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


def view_losses(output, target, dense=None):
    labels = target["labels"]
    result = dict(detection=detection_loss(output["logits"], labels),
                  boundary=boundary_loss(output["boundary"], target["boundary_distance"],
                                         labels, target["boundary_valid"]),
                  surface=surface_loss(output["surface"], target["surface_offset_z"],
                                       labels, target["surface_valid"]))
    if dense is not None:
        result["sampling"] = sampling_loss(output["logits"], dense, target["dense_row"].long(),
                                            labels, target["sampling_consistency_valid"])
    return result


def backward_sample(model, sample, scaler, coefficients, accumulation):
    dense, values = None, defaultdict(float)
    for view in sample["views"]:
        scan, target = to_device(view["scan"]), to_device(view["target"])
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(scan)
        losses = view_losses(output, target, dense)
        objective = losses["detection"] / 3
        for name, coefficient in coefficients.items():
            if name in losses:
                objective = objective + coefficient * losses[name] / (2 if name == "sampling" else 3)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"nonfinite loss at {sample['world']}/{sample['frame']}")
        scaler.scale(objective / accumulation).backward()
        if dense is None:
            # Only C2's teacher is detached; base detection has already backpropagated.
            dense = output["logits"].detach()
        for name, value in losses.items():
            values[name] += float(value.detach()) / (2 if name == "sampling" else 3)
        del output, scan, target, losses, objective
    return dict(values)


def optimizer_for(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], betas=config["betas"])
    scaler = torch.amp.GradScaler("cuda", init_scale=config["initial_loss_scale"])
    return optimizer, scaler


def update(model, optimizer, scaler, config, attempt):
    warm = min(1., attempt / config["warmup_updates"])
    factor = config["warmup_start_factor"] + (1 - config["warmup_start_factor"]) * warm
    for group in optimizer.param_groups:
        group["lr"] = config["learning_rate"] * factor
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return dict(success=scaler.get_scale() >= before,
                gradient_norm=float(norm) if torch.isfinite(norm) else None,
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
    return dict(model=protocol["model"], training=protocol["training"],
                supervision_identity=training_identity(protocol),
                dataset_sha256=hashlib.sha256((Path(protocol["dataset"]["directory"]) / "manifest.json").read_bytes()).hexdigest(),
                implementation={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                packages={name: importlib.metadata.version(name) for name in
                          ("torch", "spconv", "torch-scatter", "flash-attn", "numpy", "scipy", "numba")})


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


def train_epoch(protocol, data_root, run):
    config = execution_config(protocol)
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
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["configuration"] != config or state["order"] != order:
            raise ValueError("resume would change the scientific execution")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        visited, successes, history = state["visited"], state["successful_updates"], state["history"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        del state
    if visited == len(order):
        _atomic_json(run / "training.json", dict(base_frames=visited, views=3*visited,
                     attempted_updates=len(history), successful_updates=successes,
                     skipped_updates=len(history)-successes, history=history, resources=runtime()))
        return
    remaining = protocol["training"]["storage"]["peak_additional_bytes"] - auxiliary["bytes"]
    require_space(remaining)
    if not checkpoint.exists():
        # Check-run weights are never used: record the untouched formal initialization.
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
        values = backward_sample(model, sample, scaler, settings["loss_coefficients"], accumulation)
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
                 skipped_updates=len(history)-successes, history=history, resources=runtime()))


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
        for v, view in enumerate(sample["views"]):
            output = predict(model, view["scan"])
            predictions.append(output)
            tensors = {key: torch.from_numpy(value) for key, value in output.items()}
            values = view_losses(tensors, view["target"], None if dense is None else torch.from_numpy(dense))
            for name, value in values.items():
                losses[name] += float(value) / (2 if name == "sampling" else 3)
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
                check_order = [indices[i % len(indices)] for i in range(len(history))]
                save_checkpoint(temporary / "check.pt", model, optimizer, scaler, configuration,
                                check_order, len(check_order), sum(r["success"] for r in history), history)
                state = torch.load(temporary / "check.pt", map_location="cpu", weights_only=False)
                if (any(not torch.equal(state["model"][k], v.cpu()) for k,v in model.state_dict().items())
                        or state["scaler"] != scaler.state_dict() or state["order"] != check_order):
                    raise ValueError("checkpoint round-trip changed model, scaler or sample identities")
                optimizer.load_state_dict(state["optimizer"])
                storage_checks = dict(prediction_scores_exact=True, checkpoint_parameters_exact=True,
                                      optimizer_restored=True, scaler_and_order_exact=True)
                del state
        learned = all(b["losses"]["detection"] < a["losses"]["detection"]
                      and b["class_mean_logit"]["1"] > b["class_mean_logit"]["0"]
                      for a,b in zip(before,after))
        arms[arm] = dict(after=after, history=history, detection_learned=learned,
                         successful_updates=sum(r["success"] for r in history), seconds=time.monotonic()-start)
        del model, optimizer, scaler
    passed = all(arm["detection_learned"] for arm in arms.values())
    _atomic_json(previous, dict(configuration=configuration, passed=passed, scope=specification,
                 before=before, arms=arms, point_checks=point_checks, gradient_checks=gradient_checks,
                 base_only=base_check, isolated_voxel_offset=offset_check,
                 synthetic_validation_201=dict(**special_selection, views=special_views),
                 real_label_forbidden=dict(sequence=real["source"].sequence_id,
                 frame=real["source"].frame_id, actual_points=len(real_output["logits"])),
                 C2_teacher_detached_only_in_consistency=True,
                 storage_checks=storage_checks,
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
    parser.add_argument("--stage", choices=("check", "train", "synthetic", "real", "evaluate", "all"), default="check")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    torch.set_num_threads(protocol["training"]["torch_threads"])
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
