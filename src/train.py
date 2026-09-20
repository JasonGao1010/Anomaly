"""Run a bounded V4 training segment with explicit sampling and validation intervals."""

import argparse
import copy
from collections import Counter
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from .data import (VERSION, PILOT_VERSION, CONTINUATION_VERSION, NATIVE_VERSION, file_sha256, identity,
                   load_manifest, replace_background, write_json)
from .evaluate import PreparedScans, autocast, better, evaluate, memory_available, precision
from .model import (POINT_CHUNK, Segmentor, balanced_loss, ranking_loss, to_device,
                    LITEPT_COMMIT, WEIGHTS_REVISION, WEIGHTS_SHA256)


EPOCHS = (8, 4)
BATCH_SIZE = 8
PAIRED_UPDATES = 500
PEAK_LR = ((2e-4, 2e-3), (2e-5, 2e-4))
STOP = False
ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_order(count, seed, stage, epoch):
    # Independent of model construction, validation and worker RNG consumption.
    generator = np.random.default_rng(np.random.SeedSequence([seed, stage, epoch]))
    return generator.permutation(count).tolist()


def effective_batches(order, rank=0, world_size=1):
    for start in range(0, len(order), BATCH_SIZE):
        yield order[start:start + BATCH_SIZE][rank::world_size]


def pilot_order(manifest, seed, updates, *, sampling=None, segment=0, paired=False, background=None):
    """Source quotas stay fixed; a new segment gets its own reproducible permutation."""
    if manifest.get("version") == NATIVE_VERSION:
        order = epoch_order(len(manifest["records"]), seed, 1, 0) + epoch_order(len(manifest["records"]), seed, 1, 1)
        if updates != math.ceil(len(order) / BATCH_SIZE):
            raise ValueError("native training visits every fixed record exactly twice")
        return order
    if background:
        reference = load_manifest(background["manifest"], "train")
        saved = json.loads(Path(background["sampling"]).read_text())
        if saved["train_manifest"] != reference["sha256"] or len(saved["order"]) != updates * BATCH_SIZE:
            raise ValueError("background reference must be the actual completed training order")
        indices = {identity(row): i for i, row in enumerate(manifest["records"])
                   if row["group"] != "normal_nuscenes"}
        scenes = {}
        for i, row in enumerate(manifest["records"]):
            if row["group"] == "normal_nuscenes":
                scenes.setdefault(row["scene"], []).append(i)
        if not scenes:
            raise ValueError("expanded background pool is empty")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 611, 8]))
        remaining = {scene: rng.permutation(rows).tolist() for scene, rows in sorted(scenes.items())}
        draws = []
        # Each round visits every scene once; frames within a scene are used before reuse.
        while len(draws) < updates:
            for scene in rng.permutation(sorted(scenes)):
                if not remaining[scene]:
                    remaining[scene] = rng.permutation(scenes[scene]).tolist()
                draws.append(remaining[scene].pop())
                if len(draws) == updates:
                    break
        replacements = iter(draws)
        return [next(replacements) if reference["records"][i]["group"] == "normal_nuscenes"
                else indices[identity(reference["records"][i])] for i in saved["order"]]
    if paired:
        if segment != 0 or sampling != dict(base=6, normal_nuscenes=1, normal_stu=1):
            raise ValueError("the paired control replaces only P1's two targeted scans")
        if updates > PAIRED_UPDATES:
            # Extending the budget must not reshuffle the measured 500-update control.
            prefix = pilot_order(manifest, seed, PAIRED_UPDATES, sampling=sampling, paired=True)
            return prefix + pilot_order(manifest, seed, updates - PAIRED_UPDATES,
                                        sampling=sampling, segment=1)
        reference = pilot_order(manifest, seed, updates)
        used_base = {i for i in reference if manifest["records"][i]["group"] == "base"}
        available = [i for i, row in enumerate(manifest["records"])
                     if row["group"] == "base" and i not in used_base]
        if len(available) < 2 * updates:
            raise ValueError("not enough unused base scans for unique paired replacements")
        # Keep all six shared scans at their exact P1 update and microbatch positions.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 611]))
        replacements = iter(rng.permutation(available)[:2 * updates].tolist())
        return [next(replacements) if manifest["records"][i]["group"] == "targeted" else i
                for i in reference]
    sampling = sampling or dict(base=4, targeted=2, normal_nuscenes=1, normal_stu=1)
    if sum(sampling.values()) != BATCH_SIZE or min(sampling.values()) < 1:
        raise ValueError("source quotas must be positive and sum to eight")
    # Segment zero preserves the exact P1 sampler and its old-data comparison seed.
    entropy = [seed, 2064] + ([segment] if segment else [])
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    streams = []
    for group, quota in sampling.items():
        indices = np.asarray([i for i, row in enumerate(manifest["records"]) if row["group"] == group])
        if not len(indices):
            raise ValueError(f"pilot training group is empty: {group}")
        cycles = math.ceil(updates * quota / len(indices))
        draws = np.concatenate([rng.permutation(indices) for _ in range(cycles)])[:updates * quota]
        streams.append(draws.reshape(updates, quota))
    batches = np.concatenate(streams, axis=1)
    return np.concatenate([rng.permutation(batch) for batch in batches]).tolist()


def source_counts(manifest, order):
    result = {}
    for index in order:
        row = manifest["records"][index]
        counts = result.setdefault(row["group"], dict(frames=0, normal=0, anomaly=0))
        counts["frames"] += 1
        counts["normal"] += row["normal"]
        counts["anomaly"] += row["anomaly"]
    return result


def lr_factor(step, total):
    """One-based planned update: exact initial, warmup and final endpoints."""
    if not 1 <= step <= total:
        raise ValueError("scheduler step outside the fixed execution budget")
    warmup = math.ceil(.05 * total)
    if step == total:
        return .01
    if step == warmup:
        return 1.
    if step <= warmup:
        return .1 + .9 * (step - 1) / (warmup - 1)
    return .01 + .99 * .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def ranking_weight(step, total):
    return min(1., max(0., (step / total - .1) / .1))


def forward_loss(model, samples, counts, *, rank_weight=0., rank_seed=0, auc_weight=.1, fpr95_weight=.1):
    """Retain both full-scan graphs; concatenate scores only for the training objective."""
    predictions = [model(sample) for sample in samples]
    prediction = torch.cat(predictions)
    targets = torch.cat([sample["targets"] for sample in samples])
    bce = balanced_loss(prediction, targets, counts)
    if rank_weight:
        rank, details = ranking_loss(prediction, targets, rank_seed,
                                     auc_weight=auc_weight, fpr95_weight=fpr95_weight)
    else:
        rank, details = bce * 0, {}
    return bce + rank_weight * rank, dict(bce=bce.detach(), **details)


def optimizer_for(model, stage):
    groups = {}
    normalizations = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    no_decay = {id(p) for m in model.modules() if isinstance(m, normalizations)
                for p in m.parameters(recurse=False)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"all V4 model parameters must be trainable: {name}")
        loaded = name.startswith("backbone.") if stage == 1 else not name.startswith(
            ("interaction.", "interaction_weight."))
        rate = PEAK_LR[stage - 1][0 if loaded else 1]
        decay = 0. if name.endswith(".bias") or id(parameter) in no_decay else .005
        groups.setdefault((rate, decay), []).append(parameter)
    return torch.optim.AdamW([dict(params=params, lr=rate * .1, peak_lr=rate, weight_decay=decay)
                              for (rate, decay), params in groups.items()],
                             betas=(.9, .999), eps=1e-8)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(saved, device):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    if saved["cuda"] is not None:
        torch.cuda.set_rng_state(saved["cuda"], device)


def rank_info():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def broadcast_object(value):
    if dist.is_initialized():
        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]
    return value


def sync_buffers(model):
    if dist.is_initialized():
        for value in model.buffers():
            dist.broadcast(value, src=0)


def sync_gradients(model):
    if not dist.is_initialized():
        return
    # Losses already use global class counts: SUM, without DDP's extra averaging.
    parameters = list(model.parameters())
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in parameters])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    start = 0
    for parameter in parameters:
        parameter.grad = flat[start:start + parameter.numel()].view_as(parameter).clone()
        start += parameter.numel()


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def host_disk():
    executable = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not executable.exists():
        raise RuntimeError("Windows E: free space cannot be verified on this host")
    result = subprocess.run([str(executable), "-NoProfile", "-Command",
                             "Get-Volume -DriveLetter E | Select-Object Size,SizeRemaining | ConvertTo-Json -Compress"],
                            check=True, capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout.strip())


def disk_check(remaining_peak=0):
    disk = host_disk()
    if disk["SizeRemaining"] < 10_000_000_000 + remaining_peak:
        raise RuntimeError(f"Windows E: remaining {disk['SizeRemaining']} bytes cannot cover "
                           f"{remaining_peak} more bytes and the fixed 10 GB reserve")
    return disk


def runtime_snapshot():
    def command(args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    return dict(cpu_affinity=sorted(os.sched_getaffinity(0)),
                cpu=command(["lscpu"]), memory_available=memory_available(),
                memory=command(["free", "-b"]), windows_e=host_disk(),
                gpu=command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                             "--format=csv,noheader"]),
                processes=command(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                   "--format=csv,noheader"]))


def code_record():
    files = sorted(list((ROOT / "src").glob("*.py")) + list((ROOT / "vendor").rglob("*.py")))
    dependencies = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "timm", "addict",
                 "flash_attn", "torch_scatter", "spconv-cu126", "cumm-cu126"):
        dependencies[name] = importlib.metadata.version(name)
    return dict(files={str(p.relative_to(ROOT)): file_sha256(p) for p in files},
                dependencies=dependencies, python=sys.version,
                litept_commit=LITEPT_COMMIT, stu_commit="8f0f09c2ca4bf7b665e0ae5919b4092ddae140a2",
                weights_revision=WEIGHTS_REVISION, weights_sha256=WEIGHTS_SHA256)


def configuration(train, val, device, world_size, *, updates=None, initial=None,
                  eval_every=None, recipe="mixed", optimizer_state="reset", segment=0, objective="bce", branch=None):
    if branch:
        parent = torch.load(initial, map_location="cpu", weights_only=False)
        previous = parent["config"]
        if (recipe != "native" or objective != "metrics" or updates != 500 or eval_every != 500
                or world_size != 1 or parent["successful_updates"] != 500 or parent["planned_updates"] != 500
                or parent["next_batch"] != 0 or parent["epoch"] != 1 or not parent["complete"]
                or previous["train_manifest"] != train["sha256"] or previous["val_manifest"] != val["sha256"]
                or previous["recipe"] != "native" or previous["objective"] != "metrics"
                or {int(s["step"]) for s in parent["optimizer"]["state"].values()} != {500}):
            raise ValueError("diagnostic branches require the complete native metric state at update 500")
        result = copy.deepcopy(previous)
        result.update(code=code_record(), initial=str(initial.resolve()), initial_sha256=file_sha256(initial),
                      parent_configuration=identity(previous), branch=branch, start_update=500,
                      updates=1000, additional_updates=500, schedule_updates=previous["updates"],
                      epochs=None, scan_visits=4000, optimizer_state="inherit", lr_scale=.3 if branch == "lr" else 1.,
                      validation="inherited update 500 plus one full validation at update 1000",
                      reference_sampling=str(initial.parent / "sampling.json"))
        if branch == "ap":
            result["loss"].update(auc_weight=0., fpr95_weight=0.)
        return result
    result = dict(version=CONTINUATION_VERSION if updates is not None else VERSION, data_version=train["version"], seeds=[0],
                train_manifest=train["sha256"], val_manifest=val["sha256"],
                val_directory=val["directory"], samples=len(train["records"]), epochs=EPOCHS,
                batch_size=BATCH_SIZE, microbatch=1, peak_lr=PEAK_LR, weight_decay=.005,
                adam_betas=(.9, .999), adam_eps=1e-8, gradient_clip=1.,
                warmup_fraction=.05, initial_lr_fraction=.1, final_lr_fraction=.01,
                augmentation=False, point_chunk=POINT_CHUNK, grid_size=.05,
                precision=str(precision(device)), sparse_precision="float32", world_size=world_size,
                code=code_record())
    if recipe == "native":
        visits = 2 * len(train["records"])
        if (train["version"] != NATIVE_VERSION or updates != math.ceil(visits / BATCH_SIZE)
                or optimizer_state != "reset" or file_sha256(initial) != WEIGHTS_SHA256 or world_size != 1):
            raise ValueError("native training requires two complete passes and fresh public nuScenes initialization")
        result.update(version=NATIVE_VERSION, updates=updates, epochs=2,
                      initial=str(initial.resolve()), initial_sha256=WEIGHTS_SHA256,
                      peak_lr=PEAK_LR[0], scan_visits=visits,
                      sampling="two complete shuffled passes; final batch may contain fewer than eight scans",
                      sampling_segment=0, recipe=recipe, optimizer_state="reset", eval_every=eval_every,
                      microbatch=2, objective=objective,
                      loss=dict(bce="class means over all effective-batch supervised points",
                                pairs="consecutive pairs, averaged within each effective batch",
                                tau=1., ap_weight=1., auc_weight=.1, fpr95_weight=.1,
                                positive_anchors=256, positive_references="all", normal_top=512, normal_random=3584,
                                normal_weights="top: 1; rest: population / sample; normalize by full normal count",
                                auc_positives="uniform anchors", threshold_recall=.95,
                                threshold_gradient="implicit", bce_only_fraction=.1, ramp_end_fraction=.2,
                                rank_seed="seed * 100000000 + update * 8 + pair_index; independent torch generator"),
                      validation="full STU validation every interval and at the fixed endpoint; no inherited anomaly-task weights")
        return result
    if updates is not None:
        sampling = {"mixed": dict(base=4, targeted=2, normal_nuscenes=1, normal_stu=1),
                    "old": dict(base=8), "paired": dict(base=6, normal_nuscenes=1, normal_stu=1),
                    "background": dict(base=6, normal_nuscenes=1, normal_stu=1)}[recipe]
        result.update(updates=updates, epochs=None, initial=str(initial.resolve()),
                      initial_sha256=file_sha256(initial), peak_lr=2e-5,
                      samples=sum(row["group"] in sampling for row in train["records"]),
                      sampling=sampling, sampling_segment=segment, recipe=recipe,
                      optimizer_state=optimizer_state, eval_every=eval_every or updates,
                      validation="full_set_at_each_interval; inherited_full_validation_is_step_zero")
        if recipe == "paired":
            reference_path = ROOT / "results/train/p1/0/pilot/config.json"
            reference = json.loads(reference_path.read_text())["configuration"]
            keys = ("train_manifest", "val_manifest", "initial_sha256", "batch_size", "microbatch",
                    "peak_lr", "weight_decay", "adam_betas", "adam_eps", "gradient_clip", "warmup_fraction",
                    "initial_lr_fraction", "final_lr_fraction", "augmentation", "precision", "world_size")
            if (optimizer_state != "reset" or segment != 0 or updates < PAIRED_UPDATES
                    or reference["updates"] != PAIRED_UPDATES or (eval_every or updates) != PAIRED_UPDATES
                    or any(identity(result[k]) != identity(reference[k]) for k in keys)):
                raise ValueError("paired runs keep P1's initialization and optimizer settings; validate every 500 updates")
            result["paired_reference"] = str(reference_path)
            result["paired_prefix_updates"] = PAIRED_UPDATES
        if recipe == "background":
            reference_dir = ROOT / "results/train/main/0/pilot"
            reference = json.loads((reference_dir / "config.json").read_text())["configuration"]
            keys = ("updates", "eval_every", "val_manifest", "initial_sha256", "batch_size", "microbatch",
                    "peak_lr", "weight_decay", "adam_betas", "adam_eps", "gradient_clip", "warmup_fraction",
                    "initial_lr_fraction", "final_lr_fraction", "augmentation", "precision", "world_size", "sampling")
            if (optimizer_state != "reset" or segment != 0
                    or train["reference_train_manifest"] != reference["train_manifest"]
                    or any(identity(result[k]) != identity(reference[k]) for k in keys)):
                raise ValueError("background comparison must preserve the main run's optimization and other sources")
            result.update(normal_manifest=train["normal_manifest"], normal_path=train["normal_path"],
                          background_reference=dict(manifest=train["reference_train_path"],
                                                    sampling=str(reference_dir / "sampling.json")),
                          reference_sampling_sha256=file_sha256(reference_dir / "sampling.json"))
    return result


def stop_requested(signum, frame):
    global STOP
    STOP = True


def capture(model, optimizer, scaler, state, config, device, **extra):
    rank, world_size = rank_info()
    local_rng = rng_state(device)
    states = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(local_rng, states, dst=0)
    else:
        states = [local_rng]
    if rank:
        return None
    return dict(version=config.get("version", VERSION), mode=model.mode, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                rng=states, config=config, **state, **extra)


def validate_all(model, val, device, workers, score_path=None):
    rank, _ = rank_info()
    sync_buffers(model)
    result = None
    if rank == 0:
        try:
            result = dict(ok=True, result=evaluate(model, val, device, workers, score_path=score_path))
        except Exception as error:
            result = dict(ok=False, error=f"{type(error).__name__}: {error}")
    result = broadcast_object(result)
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result["result"]


def write_result(directory, state, config):
    write_json(directory / "result.json", dict(version=config.get("version", VERSION), seed=state["seed"], method=state["method"],
               complete=state["complete"], epochs=config.get("epochs", EPOCHS), updates=config.get("updates"),
               best_epoch=state["best_epoch"], best_metrics=state["best_metrics"],
               best_update=min(state["best_epoch"] * config.get("eval_every", config.get("updates", 0)),
                               config.get("updates", 0)) if config.get("updates") else None,
               final_metrics=state.get("final_metrics"),
               objective=config.get("objective", "bce"),
               branch=config.get("branch"), start_update=config.get("start_update", 0),
               additional_updates=config.get("additional_updates"),
               gradient_steps=state.get("gradient_steps"), clipped_steps=state.get("clipped_steps"),
               gradient_norm_sum=state.get("gradient_norm_sum"), gradient_norm_max=state.get("gradient_norm_max"),
               training_seconds=state.get("training_seconds"), validation_seconds=state.get("validation_seconds"),
               planned_updates=state["planned_updates"], successful_updates=state["successful_updates"],
               overflows=state["overflows"], parameters=state["parameters"],
               train_manifest_sha256=config.get("train_manifest"), val_manifest_sha256=config.get("val_manifest"),
               validation_improved_from_epoch0=(state["best_epoch"] > 0) if state["stage"] == 2 else None))


def train_stage(args, train, val, seed, method, device, config):
    global STOP
    if seed != 0:
        raise ValueError("F240-R2 fixes the sole experiment seed to 0")
    rank, world_size = rank_info()
    native = method == "conditional"
    branch = config.get("branch")
    pilot = method in ("pilot", "conditional")
    stage = 1 if method == "base" or native else 2
    parent = torch.load(args.initial, map_location="cpu", weights_only=False) if (pilot and not native) or branch else None
    mode = "conditional" if native else parent["mode"] if pilot else "base" if method in ("base", "continue") else method
    directory = args.output / str(seed) / method
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    last, best_path = directory / "last.pt", directory / "best.pt"
    seed_all(seed)
    model = Segmentor(mode)
    load_record = None
    resume = last.exists()
    if resume:
        if not args.resume:
            raise ValueError(f"{last} exists; use --resume to continue its exact state")
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if identity(saved["config"]) != identity(config) or saved["seed"] != seed or saved["method"] != method:
            raise ValueError("resume configuration, code, dependencies or data differ")
        if saved["complete"]:
            if rank == 0:
                write_result(directory, saved, config)
                print(f"completed: seed={seed} method={method}", flush=True)
            return True
        model.load_state_dict(saved["model"], strict=True)
    elif branch:
        model.load_state_dict(parent["model"], strict=True)
        load_record = dict(parent=str(args.initial), parent_sha256=config["initial_sha256"],
                           inherited_full_validation=parent["validation"], inherited_optimizer_updates=500,
                           inherited_rng=True, inherited_sampling_offset=4000)
    elif stage == 1:
        load_record = model.load_pretrained(args.initial if native else args.weights)
    elif pilot:
        if (not parent.get("complete") or not parent.get("selected") or
                parent["validation"]["manifest_sha256"] != val["sha256"]):
            raise ValueError("pilot initial model must have completed full validation on this exact set")
        model.load_state_dict(parent["model"], strict=True)
        load_record = dict(parent=str(args.initial), inherited_full_validation=parent["validation"])
    else:
        parent_path = args.output / str(seed) / "base/best.pt"
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        if not parent.get("complete") or not parent.get("selected") or parent["seed"] != seed:
            raise ValueError("stage two requires this seed's selected, completed stage-one checkpoint")
        if identity(parent["config"]) != identity(config):
            raise ValueError("stage-one and stage-two scientific configuration differ")
        state = model.state_dict()
        inherited = parent["model"]
        expected_new = {k for k in state if k.startswith(("interaction.", "interaction_weight."))}
        if set(state) - set(inherited) != expected_new or set(inherited) - set(state):
            raise ValueError("invalid stage transition parameter set")
        state.update(inherited)
        model.load_state_dict(state, strict=True)
        load_record = dict(parent=str(parent_path), parent_sha256=file_sha256(parent_path),
                           inherited=sorted(inherited), new=sorted(expected_new))
    model.to(device)
    optimizer = optimizer_for(model, stage)
    scaler = torch.amp.GradScaler("cuda", enabled=precision(device) == torch.float16)
    steps_per_epoch = config.get("eval_every", config["updates"]) if pilot else math.ceil(len(train["records"]) / BATCH_SIZE)
    total = config["updates"] if pilot else EPOCHS[stage - 1] * steps_per_epoch
    schedule_total = config.get("schedule_updates", total)
    epochs = math.ceil(total / steps_per_epoch)
    state = dict(seed=seed, method=method, stage=stage, epoch=0, next_batch=0, planned_updates=0,
                 successful_updates=0, overflows=0, best_epoch=None, best_metrics=None,
                 epoch_loss=0., epoch_points=[0, 0], epoch_frames=0, complete=False,
                 parameters=sum(p.numel() for p in model.parameters()), final_metrics=None,
                 training_seconds=0., validation_seconds=0., gradient_steps=0, clipped_steps=0,
                 gradient_norm_sum=0., gradient_norm_max=0.)
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        for key in state:
            state[key] = saved.get(key, state[key])
        restore_rng(saved["rng"][rank], device)
        del saved
    else:
        seed_all(seed + stage * 1000 + rank * 10000)
        if branch:
            optimizer.load_state_dict(parent["optimizer"])
            scaler.load_state_dict(parent["scaler"])
            state.update(epoch=1, planned_updates=500, successful_updates=500,
                         best_epoch=1, best_metrics=parent["validation"]["metrics"])
            restore_rng(parent["rng"][rank], device)
            saved = capture(model, optimizer, scaler, state, config, device,
                            selected=True, validation=parent["validation"])
            if rank == 0:
                atomic_save(best_path, saved)
                write_json(directory / "epoch1.json", dict(inherited=True, **parent["validation"]))
            del saved
        elif pilot and config.get("optimizer_state") == "inherit":
            if (parent["config"]["world_size"] != world_size or parent["config"]["sampling"] != config["sampling"]
                    or parent["config"]["train_manifest"] != train["sha256"]):
                raise ValueError("optimizer continuation requires the same data, recipe and world size")
            optimizer.load_state_dict(parent["optimizer"])
            scaler.load_state_dict(parent["scaler"])
            restore_rng(parent["rng"][rank], device)
            load_record["inherited_optimizer_updates"] = parent["successful_updates"]
        if rank == 0:
            write_json(directory / "config.json", dict(configuration=config, load=load_record,
                                                       seed=seed, method=method, parameters=state["parameters"]))
        if stage == 2:
            result = parent["validation"] if pilot else validate_all(model, val, device, args.workers)
            state.update(best_metrics=result["metrics"], best_epoch=0)
            saved = capture(model, optimizer, scaler, state, config, device, selected=True, validation=result)
            if rank == 0:
                atomic_save(best_path, saved)
                atomic_save(last, dict(saved, selected=False))
                write_json(directory / "epoch0.json", result)
            del saved
    dataset = PreparedScans(train)
    if pilot:
        del parent
        full_order = pilot_order(train, seed, schedule_total, sampling=config.get("sampling"),
                                 segment=config.get("sampling_segment", 0), paired=config.get("recipe") == "paired",
                                 background=config.get("background_reference"))
        if native and rank == 0:
            if branch and full_order != json.loads(Path(config["reference_sampling"]).read_text())["order"]:
                raise ValueError("branch scan order differs from the original recorded stream")
            executed = full_order[config.get("start_update", 0) * BATCH_SIZE:total * BATCH_SIZE]
            write_json(directory / "sampling.json", dict(train_manifest=train["sha256"], order=full_order,
                sources=source_counts(train, executed), distinct_records=len(set(executed)),
                executed_order=executed, start_update=config.get("start_update", 0),
                passes=None if branch else 2, scans_per_pass=len(train["records"]), visits=len(executed)))
        if config.get("recipe") == "paired" and rank == 0:
            paired_updates = min(total, PAIRED_UPDATES)
            reference = pilot_order(train, seed, paired_updates)
            write_json(directory / "sampling.json", dict(reference=config["paired_reference"],
                       train_manifest=train["sha256"], preserved_visits=6 * paired_updates,
                       replaced_visits=2 * paired_updates, paired_prefix_updates=paired_updates,
                       reference_order=reference, order=full_order,
                       replacement="without replacement from base scans unused by P1 in the first 500 updates",
                       extension="same-pool 6/1/1 permutations with sampling segment 1" if total > paired_updates else None))
        if config.get("recipe") == "background" and rank == 0:
            reference = load_manifest(config["background_reference"]["manifest"], "train")
            reference_order = json.loads(Path(config["background_reference"]["sampling"]).read_text())["order"]
            visits = [train["records"][i] for i in full_order if train["records"][i]["group"] == "normal_nuscenes"]
            write_json(directory / "sampling.json", dict(reference=config["background_reference"],
                       reference_train_manifest=reference["sha256"], train_manifest=train["sha256"],
                       preserved_visits=7 * total, replaced_visits=total, reference_order=reference_order,
                       order=full_order, reference_sources=source_counts(reference, reference_order),
                       sources=source_counts(train, full_order), scene_visits=dict(Counter(r["scene"] for r in visits)),
                       distinct_nuscenes_frames=len({r["token"] for r in visits}),
                       replacement="scene-balanced rounds; use unseen frames within each scene before reuse"))
    for epoch in range(state["epoch"], epochs):
        start = time.perf_counter()
        order = full_order[epoch * steps_per_epoch * BATCH_SIZE:
                           min(total, (epoch + 1) * steps_per_epoch) * BATCH_SIZE] if pilot else epoch_order(
                               len(dataset), seed, stage, epoch)
        batches = list(effective_batches(order, rank, world_size))
        first_batch = state["next_batch"]
        local_indices = [i for group in batches[first_batch:] for i in group]
        loader = DataLoader(dataset, batch_size=None, sampler=local_indices, num_workers=args.workers,
                            pin_memory=device.type == "cuda", persistent_workers=False,
                            **({"prefetch_factor": 1} if args.workers else {}),
                            generator=torch.Generator().manual_seed(seed + epoch * 31 + stage * 1000 + rank))
        iterator = iter(loader)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        for batch_number in range(first_batch, len(batches)):
            update_start = time.perf_counter()
            local_indices = batches[batch_number]
            samples = [next(iterator) for _ in local_indices]
            counts = torch.zeros(2, dtype=torch.int64, device=device)
            for sample in samples:
                observed = torch.stack([(sample["targets"] == label).sum() for label in (0, 1)])
                counts += observed.to(device)
            if world_size > 1:
                dist.all_reduce(counts)
            global_indices = order[batch_number * BATCH_SIZE:(batch_number + 1) * BATCH_SIZE]
            expected = [sum(train["records"][i][key] for i in global_indices) for key in ("normal", "anomaly")]
            if counts.tolist() != expected:
                raise ValueError("effective-batch class counts changed")
            optimizer.zero_grad(set_to_none=True)
            step = state["planned_updates"] + 1
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * lr_factor(step, schedule_total) * config.get("lr_scale", 1.)
            sync_buffers(model)
            loss_sum = torch.zeros((), device=device)
            components, pair_details = dict(bce=0., ap=0., auc=0., fpr95=0.), []
            pair_size = config["microbatch"] if native else 1
            pair_count = math.ceil(len(samples) / pair_size)
            ramp = ranking_weight(step, schedule_total) if config.get("objective") == "metrics" else 0.
            for pair_index, begin in enumerate(range(0, len(samples), pair_size)):
                pair = [to_device(sample, device) for sample in samples[begin:begin + pair_size]]
                with autocast(device):
                    loss, details = forward_loss(model, pair, counts, rank_weight=ramp / pair_count,
                                                rank_seed=seed * 100000000 + step * 8 + pair_index,
                                                auc_weight=config.get("loss", {}).get("auc_weight", .1),
                                                fpr95_weight=config.get("loss", {}).get("fpr95_weight", .1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite segmentation loss")
                if details.get("recall") is not None and abs(float(details["recall"]) - .95) > 2e-6:
                    raise FloatingPointError("smooth recall threshold did not reach 0.95")
                scaler.scale(loss).backward()
                loss_sum += loss.detach()
                for key in components:
                    components[key] += float(details.get(key, 0.)) / (1 if key == "bce" else pair_count)
                pair_details.append({key: float(value) if isinstance(value, torch.Tensor) else value
                                     for key, value in details.items() if key not in components})
                del pair, loss, details
            sync_gradients(model)
            if world_size > 1:
                dist.all_reduce(loss_sum)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            state["gradient_steps"] += 1
            state["clipped_steps"] += int(norm > 1.)
            state["gradient_norm_sum"] += float(norm)
            state["gradient_norm_max"] = max(state["gradient_norm_max"], float(norm))
            finite = bool(torch.isfinite(norm))
            if not scaler.is_enabled() and not finite:
                raise FloatingPointError("nonfinite BF16 gradient; no silent skipped update")
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.is_enabled() and scaler.get_scale() < before
            state["planned_updates"] += 1
            state["successful_updates"] += int(not overflow)
            state["overflows"] += int(overflow)
            state["epoch_loss"] += loss_sum.item()
            state["epoch_points"] = [a + b for a, b in zip(state["epoch_points"], counts.tolist())]
            state["epoch_frames"] += len(global_indices)
            state["next_batch"] = batch_number + 1
            state["training_seconds"] += time.perf_counter() - update_start
            need_stop = torch.tensor(int(STOP), device=device)
            if world_size > 1:
                dist.all_reduce(need_stop, op=dist.ReduceOp.MAX)
            check_disk = step % args.save_every == 0
            if check_disk:
                error = None
                if rank == 0:
                    try:
                        disk_check(1_000_000_000)
                    except Exception as exc:
                        error = str(exc)
                error = broadcast_object(error)
                if error:
                    print(error, flush=True)
                    need_stop.fill_(1)
            if check_disk or need_stop:
                saved = capture(model, optimizer, scaler, state, config, device, selected=False)
                if rank == 0:
                    atomic_save(last, saved)
                del saved
            if rank == 0 and (step % 25 == 0 or overflow or need_stop):
                row = dict(event="update", seed=seed, method=method, epoch=epoch + 1,
                           batch=batch_number + 1, batches=steps_per_epoch, loss=loss_sum.item(),
                           normal=int(counts[0]), anomaly=int(counts[1]), planned=step,
                           successful=state["successful_updates"], overflow=bool(overflow),
                           lr=[g["lr"] for g in optimizer.param_groups],
                           elapsed_seconds=time.perf_counter() - start,
                           objective=config.get("objective", "bce"), ranking_weight=ramp,
                           loss_components=components, pairs=pair_details,
                           gradient_norm=float(norm), clipped_steps=state["clipped_steps"],
                           peak_vram_bytes=torch.cuda.max_memory_allocated(device))
                with (directory / "log.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
            if need_stop:
                return False
        del iterator, loader
        if state["epoch_frames"] != len(order):
            raise ValueError("epoch did not visit every fixed sample exactly once")
        # Save completed training before validation so an evaluation failure is resumable.
        state["next_batch"] = len(batches)
        saved = capture(model, optimizer, scaler, state, config, device, selected=False)
        if rank == 0:
            atomic_save(last, saved)
        del saved
        validation_start = time.perf_counter()
        result = validate_all(model, val, device, args.workers,
                              score_path=args.score_path if (epoch + 1) * steps_per_epoch >= total else None)
        state["validation_seconds"] += time.perf_counter() - validation_start
        state["final_metrics"] = result["metrics"]
        selected = better(result["metrics"], state["best_metrics"])
        if selected:
            state.update(best_metrics=result["metrics"], best_epoch=epoch + 1)
        report = dict(epoch=epoch + 1, mean_batch_loss=state["epoch_loss"] / len(batches),
                      frames=state["epoch_frames"], class_points=state["epoch_points"],
                      source_points=source_counts(train, order) if pilot else None,
                      planned_updates=state["planned_updates"], successful_updates=state["successful_updates"],
                      overflows=state["overflows"], validation=result, selected=selected,
                      seconds=time.perf_counter() - start, peak_vram_bytes=torch.cuda.max_memory_allocated(device))
        state.update(epoch=epoch + 1, next_batch=0, epoch_loss=0., epoch_points=[0, 0], epoch_frames=0)
        saved = capture(model, optimizer, scaler, state, config, device, selected=False, validation=result)
        if rank == 0:
            if selected:
                atomic_save(best_path, dict(saved, selected=True))
            # Commit the new best before advancing the resumable epoch boundary.
            atomic_save(last, saved)
            write_json(directory / f"epoch{epoch + 1}.json", report)
            print(json.dumps(dict(seed=seed, method=method, **report)), flush=True)
        del saved
    state["complete"] = True
    saved = capture(model, optimizer, scaler, state, config, device, selected=False)
    if rank == 0:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        best["complete"] = True
        atomic_save(best_path, best)
        atomic_save(last, saved)
        write_result(directory, state, config)
    del saved, model, optimizer, dataset
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
    return True


def preflight(args, train, val, device, config, resources):
    """Measure the actual mixed update without changing the initial checkpoint."""
    seed_all(0)
    dataset = PreparedScans(train)
    order = pilot_order(train, 0, args.updates, sampling=config["sampling"],
                          segment=config["sampling_segment"], paired=config.get("recipe") == "paired",
                          background=config.get("background_reference"))
    indices = order[:BATCH_SIZE]
    if config.get("recipe") == "native":
        model = Segmentor("conditional").to(device)
        model.load_pretrained(args.initial)
    else:
        parent = torch.load(args.initial, map_location="cpu", weights_only=False)
        model = Segmentor(parent["mode"]).to(device)
        model.load_state_dict(parent["model"], strict=True)
        del parent
    model.train()
    counts = torch.tensor([sum(train["records"][i][key] for i in indices)
                           for key in ("normal", "anomaly")], device=device)
    records = []
    torch.cuda.reset_peak_memory_stats(device)
    pair_size = config["microbatch"] if config.get("recipe") == "native" else 1
    groups = [indices[begin:begin + pair_size] for begin in range(0, len(indices), pair_size)]
    if pair_size == 2:
        groups.append(max((order[i:i + 2] for i in range(0, len(order), 2)),
                          key=lambda pair: sum(train["records"][i]["points"] for i in pair)))
    for pair_index, selected in enumerate(groups):
        start = time.perf_counter()
        stress = pair_index * pair_size >= len(indices)
        if stress:
            model.zero_grad(set_to_none=True)
            counts = torch.tensor([sum(train["records"][i][key] for i in selected)
                                   for key in ("normal", "anomaly")], device=device)
        pair = [to_device(dataset[index], device) for index in selected]
        with autocast(device):
            loss, details = forward_loss(model, pair, counts,
                                        rank_weight=(1. if stress else pair_size / len(indices))
                                        if config.get("objective") == "metrics" else 0.,
                                        rank_seed=8 + pair_index)
        loss.backward()
        torch.cuda.synchronize(device)
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                          for p in model.parameters()):
            raise ValueError("nonfinite mixed-batch backward")
        records.append(dict(indices=selected, stress=stress, groups=[train["records"][i]["group"] for i in selected],
                            points=[len(sample["xyzi"]) for sample in pair], seconds=time.perf_counter() - start,
                            loss=float(loss.detach()),
                            details={key: float(value) if isinstance(value, torch.Tensor) else value
                                     for key, value in details.items()}))
        del pair, loss, details
    result = dict(version=config["version"], configuration=config, resources=resources,
                  scans=records, parameter_updates=0,
                  mixed_batch_seconds=sum(row["seconds"] for row in records if not row["stress"]),
                  peak_vram_bytes=torch.cuda.max_memory_allocated(device))
    write_json(args.output / "preflight.json", result)
    print(json.dumps({key: result[key] for key in
                     ("mixed_batch_seconds", "peak_vram_bytes", "parameter_updates", "scans")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("results/data/train.json"))
    parser.add_argument("--val-manifest", type=Path, default=Path("assets/val.json"))
    parser.add_argument("--normal-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/train/main"))
    parser.add_argument("--initial", type=Path, default=Path("results/train/r2/0/base/best.pt"))
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--recipe", choices=("mixed", "old", "paired", "background", "native"), default="paired")
    parser.add_argument("--objective", choices=("bce", "metrics"), default="bce")
    parser.add_argument("--branch", choices=("replay", "ap", "lr"))
    parser.add_argument("--score-path", type=Path, help="export exact endpoint validation scores in official point order")
    parser.add_argument("--optimizer-state", choices=("inherit", "reset"), default="reset")
    parser.add_argument("--sampling-segment", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", choices=(0,), default=[0],
                        help="this pilot uses only seed 0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if (args.workers < 0 or args.threads < 1 or args.save_every < 1 or args.updates < 1
            or args.eval_every < 1 or args.sampling_segment < 0 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("invalid runtime resource settings or duplicate seeds")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        parser.error("LitePT sparse convolution/FlashAttention requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
    rank, world_size = rank_info()
    if world_size > BATCH_SIZE:
        parser.error("at most eight GPUs for effective batch eight")
    train, val = load_manifest(args.train_manifest, "train"), load_manifest(args.val_manifest, "val")
    if (args.recipe == "background") != bool(args.normal_manifest):
        parser.error("background comparison requires --normal-manifest and no other recipe accepts it")
    if args.normal_manifest:
        train = replace_background(train, args.normal_manifest, args.train_manifest)
    if train["version"] not in (PILOT_VERSION, NATIVE_VERSION):
        parser.error("training requires a mixed or native two-domain manifest")
    if args.branch and (args.recipe != "native" or args.check):
        parser.error("diagnostic branches require native training, not preflight")
    if args.recipe == "native" and not args.branch:
        args.updates = math.ceil(2 * len(train["records"]) / BATCH_SIZE)
    elif args.recipe != "native" and args.objective != "bce":
        parser.error("the paired metric objective is only defined for the native experiment")
    config = configuration(train, val, device, world_size, updates=args.updates, initial=args.initial,
                           eval_every=args.eval_every, recipe=args.recipe,
                           optimizer_state=args.optimizer_state, segment=args.sampling_segment,
                           objective=args.objective, branch=args.branch)
    resources = runtime_snapshot() if rank == 0 else None
    if args.workers * world_size + args.threads * world_size > len(os.sched_getaffinity(0)):
        parser.error("worker and BLAS thread counts exceed the available CPU affinity")
    processes = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(processes, os.getpid())
    else:
        processes = [os.getpid()]
    gpu_processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()
    other = [int(p.strip()) for p in gpu_processes if p.strip().isdigit() and int(p.strip()) not in processes]
    if other:
        raise RuntimeError(f"other CUDA processes must finish before this run: {other}")
    # Include best/last optimizer states and the largest atomic replacement.
    disk_check((2_000_000_000 if args.score_path else 1_000_000_000) if not args.check else 100_000_000)
    free, _ = torch.cuda.mem_get_info(device)
    if free < 7_000_000_000:
        raise RuntimeError(f"full-scan training verification needs a free GPU; only {free / 1e9:.1f} GB available")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "resources.json", resources)
    if args.check:
        if world_size != 1:
            parser.error("--check uses one GPU; accumulation equivalence is tested separately")
        preflight(args, train, val, device, config, resources)
        return
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    if rank == 0:
        print(json.dumps(dict(version=config["version"], seeds=args.seeds,
                              methods=["conditional" if args.recipe == "native" else "pilot"],
                              samples=config["samples"], sampling=config["sampling"],
                              eval_every=args.eval_every, optimizer_state=config["optimizer_state"],
                              start_update=config.get("start_update", 0), planned_updates=config["updates"],
                              output=str(args.output.resolve()))), flush=True)
    for seed in args.seeds:
        if not train_stage(args, train, val, seed, "conditional" if args.recipe == "native" else "pilot", device, config):
            return
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
