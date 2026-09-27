"""Normal-only SERVE training, normal selection and optional score calibration."""

from collections import defaultdict
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import identity, write_json
from .evaluate import memory_available
from .model import to_device

NORMAL_SELECTION = "maximum mIoU over the fixed ground-truth-present normal development classes; minimum normal objective breaks exact ties; no anomaly labels"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(saved, device):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    if saved["cuda"] is not None:
        torch.cuda.set_rng_state(saved["cuda"], device)


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
                                   "--format=csv,noheader"]),
                cpu_processes=command(["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"]))


@torch.no_grad()
def normal_development(model, records, indices, device, workers, *, queries=4096, calibration=None, deadline=None, seed=206):
    from .data import NormalScans
    model.eval()
    if calibration is not None:
        if (list(indices) != list(range(len(records))) or not records
                or any(row.get("source") != "normal_stu" or row.get("scene") != "201" for row in records)):
            raise ValueError("calibration reuse requires the complete ordered STU 201 development population")
        calibration.update(frames=[], scores=[], data_identity=identity(records), variant=model.variant)
    data = NormalScans(records, queries=queries, seed=seed)
    loader = DataLoader(data, batch_size=None, sampler=indices, num_workers=workers,
                        pin_memory=device.type == "cuda", prefetch_factor=1 if workers else None)
    totals, count = defaultdict(float), 0
    diagnostics = {}
    class_changes = np.zeros((19, 2), np.int64)
    strata = {name: dict(edges=edges, counts=np.zeros((len(edges) + 1, 5), np.int64),
                        confusion=np.zeros((len(edges) + 1, 19, 19), np.int64))
              for name, edges in (("distance_metres", [10., 20., 30., 40.]),
                                  ("returns_per_angular_cell", [1, 3, 7, 15]))}
    confusions = {name: np.zeros((19, 19), np.int64) for name in ("joint", "semantic")}
    start = time.monotonic()
    for frame, sample in enumerate(loader):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("deadline reached before complete normal development; selected weights are saved; retry with a new deadline")
        if calibration is not None:
            chosen, eligible = normal_reference_points(sample["allowed"], frame, calibration["seed"])
        sample = to_device(sample, device)
        loss, detail = model.loss(sample)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite normal development loss")
        if calibration is not None:
            values = model.development_raw_score[torch.as_tensor(chosen, device=device)]
            if values.shape != (len(chosen),) or not bool(torch.isfinite(values).all()):
                raise ValueError("nonfinite or misaligned reused normal reference scores")
            if len(chosen):
                calibration["scores"].append(values.float().cpu().numpy())
            calibration["frames"].append(dict(frame=records[frame]["frame"], eligible=eligible, retained=len(chosen)))
        totals["objective"] += float(loss)
        for key, value in detail.items():
            totals[key] += float(value)
        for key, value in model.development_diagnostics.items():
            value = np.asarray(value, dtype=np.float64)
            if key not in diagnostics:
                diagnostics[key] = np.zeros_like(value)
            if diagnostics[key].shape != value.shape or not np.isfinite(value).all():
                raise ValueError(f"invalid additive normal diagnostic: {key}")
            diagnostics[key] += value
        # Formal and semantic-only decisions share one backbone evaluation.
        predictions = dict(joint=model.development_prediction,
                           semantic=model.development_semantic_prediction)
        allowed = sample["allowed"]
        single = allowed.sum(1) == 1
        truth = allowed[single].long().argmax(-1)
        valid = allowed.any(1)
        correct = {}
        for name, prediction in predictions.items():
            if prediction.shape != (len(allowed),):
                raise ValueError("normal development predictions must cover every input return")
            confusions[name] += torch.bincount(truth * 19 + prediction[single], minlength=361).reshape(19, 19).cpu().numpy()
            correct[name] = allowed[valid].gather(1, prediction[valid, None]).squeeze(1)
            totals[name + "_set_correct"] += int(correct[name].sum())
        totals["corrected_points"] += int((correct["joint"] & ~correct["semantic"]).sum())
        totals["worsened_points"] += int((~correct["joint"] & correct["semantic"]).sum())
        totals["set_points"] += int(valid.sum())
        joint_correct = predictions["joint"][single] == truth
        semantic_correct = predictions["semantic"][single] == truth
        for column, selected in enumerate((joint_correct & ~semantic_correct, ~joint_correct & semantic_correct)):
            class_changes[:, column] += torch.bincount(truth[selected], minlength=19).cpu().numpy()
        distance = sample["xyzi"][:, :3].norm(dim=1)
        group = sample["observation"]["group"]
        density = torch.bincount(group)[group]
        for name, values in (("distance_metres", distance), ("returns_per_angular_cell", density)):
            row = strata[name]
            bins = torch.bucketize(values.contiguous(), values.new_tensor(row["edges"]), right=False)
            normal_bins = bins[valid]
            columns = (torch.ones_like(correct["joint"]), correct["joint"], correct["semantic"],
                       correct["joint"] & ~correct["semantic"], ~correct["joint"] & correct["semantic"])
            for column, selected in enumerate(columns):
                row["counts"][:, column] += torch.bincount(normal_bins[selected], minlength=len(row["edges"]) + 1).cpu().numpy()
            combined = bins[single] * 361 + truth * 19 + predictions["joint"][single]
            row["confusion"] += torch.bincount(combined, minlength=(len(row["edges"]) + 1) * 361).reshape(-1, 19, 19).cpu().numpy()
        count += 1
    if not count or not totals["set_points"]:
        raise ValueError("normal development has no reliably labeled normal points")
    summaries = {}
    for name, confusion in confusions.items():
        support = confusion.sum(1)
        union = confusion.sum(0) + confusion.sum(1) - np.diag(confusion)
        summaries[name] = dict(set_accuracy=totals[name + "_set_correct"] / totals["set_points"],
            iou=[float(confusion[c, c] / union[c]) if union[c] else None for c in range(19)],
            mean_iou_present=float(np.mean(np.diag(confusion)[union > 0] / union[union > 0])) if (union > 0).any() else None,
            mean_iou_gt=float(np.mean(np.diag(confusion)[support > 0] / union[support > 0])) if (support > 0).any() else None,
            ground_truth_points=support.tolist(), absent_classes=np.flatnonzero(support == 0).tolist(),
            confusion=confusion.tolist())
    for row in strata.values():
        row["counts"] = row["counts"].tolist()
        row["confusion"] = row["confusion"].tolist()
        row["count_columns"] = ["points", "correct", "semantic_correct", "corrected", "worsened"]
        row["bins"] = "right-closed intervals split at edges; the first and last bins include the remaining tails"
    counts = ("joint_set_correct", "semantic_set_correct", "set_points", "corrected_points", "worsened_points")
    measured = {key: value / count for key, value in totals.items() if key not in counts}
    from .evaluate import predictive_summary
    return dict(**measured, **summaries["joint"], semantic_only=summaries["semantic"],
                joint_comparison={key: int(totals[key]) for key in ("set_points", "corrected_points", "worsened_points")},
                per_class_comparison=dict(corrected_points=class_changes[:, 0].tolist(),
                                          worsened_points=class_changes[:, 1].tolist(),
                                          population="reliable singleton-label points; class index is ground truth"),
                semantic_definition="formal inference decision for this variant; semantic_only is an internal diagnostic of these same weights, not an independently trained baseline",
                iou_definition="pooled reliable singleton-label points; mean_iou_gt averages the fixed ground-truth-present classes; mean_iou_present averages nonzero unions; set accuracy also includes coarse labels",
                strata=strata, observation_diagnostics={key: value.tolist() for key, value in diagnostics.items()},
                predictive=predictive_summary(diagnostics),
                observation_diagnostics_scope="additive statistics over the deterministic reliable supervision queries; normal semantic confusion covers every reliable input point",
                scans=count, seconds=time.monotonic() - start)


def normal_selection(measured):
    """Fixed labeled classes select normal semantics before predictive likelihood."""
    quality, objective = measured["mean_iou_gt"], measured["objective"]
    if quality is None or not np.isfinite([quality, objective]).all():
        raise ValueError("normal model selection requires finite ground-truth-class mIoU and loss")
    return float(quality), -float(objective)


def normal_replay_pools(source, target):
    from .data import STU_NORMAL_SEMANTICS
    present = set()
    for row in target:
        if row.get("source") != "normal_stu" or row.get("scene") != "206":
            raise ValueError("normal replay priorities must be derived from STU 206 training labels")
        labels = np.unique(np.fromfile(row["label"], dtype=np.uint32) & 0xFFFF)
        present.update(STU_NORMAL_SEMANTICS[int(label)] for label in labels if int(label) in STU_NORMAL_SEMANTICS)
    pools = defaultdict(list)
    for index, row in enumerate(source):
        for category in row.get("refinement_classes", []):
            if category not in present:
                pools[category].append(index)
    return dict(pools), sorted(present)


def normal_order(source_count, target_count, epochs, seed, stage, replay_pools=None):
    if stage not in ("source", "target") or min(source_count, target_count, epochs) < 1:
        raise ValueError("normal order requires two nonempty domains and positive epochs")
    replay_rng = np.random.default_rng(np.random.SeedSequence([seed, 929]))
    general = replay_rng.permutation(source_count)
    pool_classes = sorted(replay_pools or {})
    pools, pointers = {}, defaultdict(int)
    for category in pool_classes:
        values = np.unique(replay_pools[category])
        if not len(values) or (values < 0).any() or (values >= source_count).any():
            raise ValueError("normal replay pools must contain valid source record indices")
        pools[category] = np.random.default_rng(np.random.SeedSequence([seed, 917, category])).permutation(values)
    replay_count, general_count = 0, 0
    order = []
    for epoch in range(epochs):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(stage == "target"), epoch]))
        ids = rng.permutation(source_count if stage == "source" else target_count)
        for visit, index in enumerate(ids):
            order.append((epoch, int(index)))
            if stage == "target" and visit % 4 == 3:
                # Half of the fixed replay slots preserve explicitly refined normal
                # classes; the remainder traverse the whole source pool without replacement.
                if pool_classes and replay_count % 2 == 0:
                    category = pool_classes[(replay_count // 2) % len(pool_classes)]
                    selected = pools[category][pointers[category] % len(pools[category])]
                    pointers[category] += 1
                else:
                    if general_count and general_count % source_count == 0:
                        general = replay_rng.permutation(source_count)
                    selected = general[general_count % source_count]
                    general_count += 1
                order.append((epoch, target_count + int(selected)))
                replay_count += 1
    return order


def normal_reference(path, config):
    """A matched run must complete the same visits and learning-rate schedule."""
    path = Path(path)
    other = json.loads((path / "config.json").read_text())
    stages = json.loads((path / "stages.json").read_text())
    keys = ("version", "architecture", "data_identity", "classes", "source_mapping", "target_mapping", "source_epochs", "target_epochs", "batch", "seed",
            "eval_every", "target_eval_every", "queries", "source_replay_fraction", "source_replay", "initial_sha256", "budget", "selection")
    # JSON stores integer mapping keys as strings; compare the same representation.
    comparable = json.loads(json.dumps(config))
    changed = [key for key in keys if identity(comparable.get(key)) != identity(other.get(key))]
    if changed:
        raise ValueError(f"normal comparison changes data, initialization or training budget: {changed}")
    for stage, budget in config["budget"].items():
        result = stages.get(stage, {})
        if (not result.get("budget_complete") or result.get("trained_frames") != budget["visits"]
                or result.get("trained_updates") != budget["updates"]
                or result.get("planned_frames") != budget["visits"]
                or result.get("planned_updates") != budget["updates"]):
            raise ValueError(f"reference {stage} did not complete the common nominal update schedule")
    return other, stages


def normal_optimizer(model, stage):
    groups = []
    for backbone in (True, False):
        rate = ((1e-4, 8e-4) if stage == "source" else (3e-5, 2e-4))[int(not backbone)]
        parameters = [p for name, p in model.named_parameters() if name.startswith("backbone.") == backbone]
        groups.append(dict(params=parameters, lr=rate, peak_lr=rate))
    return torch.optim.AdamW(groups, weight_decay=.005, eps=1e-6)


def normal_reference_points(allowed, frame, seed):
    """Identical uniform reference points for fresh and reused normal inference."""
    valid = allowed.any(1).nonzero().flatten().cpu().numpy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, frame]))
    return np.sort(rng.choice(valid, min(len(valid), 2048), replace=False)), len(valid)


@torch.no_grad()
def normal_calibration(model, records, device, workers, output, *, seed=206, reference=None, deadline=None):
    from .data import NormalScans
    from .model import CALIBRATION_PROBABILITIES, NORMAL_SCORE_VERSION
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("deadline reached before normal calibration; selected weights are saved; retry with a new deadline")
    frames = []
    model.eval()
    start = time.monotonic()
    if not records or any(row.get("source") != "normal_stu" or row.get("scene") != "201" for row in records):
        raise ValueError("joint normal references must use only the STU 201 normal development sequence")
    reused = reference is not None
    if reused:
        if (reference["seed"] != seed or reference["variant"] != model.variant
                or reference["data_identity"] != identity(records)
                or [row["frame"] for row in reference["frames"]] != [row["frame"] for row in records]):
            raise ValueError("reused calibration must match this model, seed and full normal population")
        scores, frames = reference["scores"], reference["frames"]
        # A skipped sequential DataLoader iterator would have consumed one CPU
        # base seed. Preserve that RNG transition without loading the scans twice.
        torch.empty((), dtype=torch.int64).random_()
    else:
        scores = []
        loader = DataLoader(NormalScans(records, queries=1, seed=seed), batch_size=None, num_workers=workers,
                            pin_memory=device.type == "cuda", prefetch_factor=1 if workers else None)
        for frame, sample in enumerate(loader):
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("deadline reached before complete normal calibration; selected weights are saved; retry with a new deadline")
            chosen, eligible = normal_reference_points(sample["allowed"], frame, seed)
            if len(chosen):
                sample = to_device(sample, device)
                values = model.components(sample, torch.as_tensor(chosen, device=device))["raw_score"]
                if values.shape != (len(chosen),) or not bool(torch.isfinite(values).all()):
                    raise ValueError("nonfinite or misaligned joint normal reference scores")
                scores.append(values.float().cpu().numpy())
            frames.append(dict(frame=records[frame]["frame"], eligible=eligible, retained=len(chosen)))
            if (frame + 1) % 50 == 0:
                print(f"calibration 201 {frame + 1}/{len(records)} elapsed={(time.monotonic()-start)/60:.1f}min", flush=True)
    if not scores:
        raise ValueError("STU 201 has no reliable normal reference points")
    if sum(map(len, scores)) != sum(row["retained"] for row in frames):
        raise ValueError("normal reference scores and their sampled point counts disagree")
    quantiles = np.quantile(np.concatenate(scores), CALIBRATION_PROBABILITIES).astype(np.float32)
    if not np.isfinite(quantiles).all() or np.any(np.diff(quantiles) < 0):
        raise ValueError("joint normal reference quantiles must be finite and nondecreasing")
    model.calibration.copy_(torch.from_numpy(quantiles).to(device))
    model.calibrated.fill_(True)
    result = dict(reference="STU 201 reliable normal points", data_identity=identity(records),
                  frames=frames, target_scans=len(records),
                  score_version=NORMAL_SCORE_VERSION, probabilities=CALIBRATION_PROBABILITIES.tolist(),
                  quantiles=quantiles.tolist(), seed=seed, samples_per_frame=2048,
                  eligible_points=sum(row["eligible"] for row in frames), reference_points=sum(map(len, scores)),
                  seconds=time.monotonic() - start, anomaly_labels_used=False,
                  inference="reused from full normal development" if reused else "dedicated normal reference inference",
                  point_scope="actual returns with at least one reliable normal class in the 2.5–50 metre supervision range",
                  sampling="uniform without replacement within each frame's reliable normal points; no class balancing or support filtering",
                  score_definition=f"formal raw_score returned by components for variant={model.variant}; identical to inference",
                  meaning="one monotone normal-reference transform preserves the joint score ordering; not an anomaly probability or anomaly-performance estimate")
    write_json(output / "calibration.json", result)
    return result


def development_indices(records, count, *, by_scene=False):
    """Select normal checkpoints on central source frames or uniformly spaced target frames."""
    if not records or count < 1:
        raise ValueError("normal development selection requires nonempty records")
    if not by_scene:
        return np.linspace(0, len(records) - 1, min(count, len(records)), dtype=int).tolist()
    scenes = defaultdict(list)
    for index, row in enumerate(records):
        scenes[row["scene"]].append(index)
    names = sorted(scenes)
    chosen = np.linspace(0, len(names), min(count, len(names)), endpoint=False, dtype=int)
    return [scenes[names[i]][len(scenes[names[i]]) // 2] for i in chosen]


def learning_rate_factor(update, updates):
    """Use optimizer-update coordinates, independent of model speed or checkpoint selection."""
    if not 1 <= update <= updates:
        raise ValueError("learning-rate update must lie within the complete stage")
    warmup = max(1, math.ceil(.03 * updates))
    if update <= warmup:
        return update / warmup
    phase = (update - warmup) / max(1, updates - warmup)
    return .05 + .95 * (1 + math.cos(math.pi * phase)) / 2


def train_stage(model, records, development, selection, order, config, stage, output,
                device, workers, *, stages, resume=None, checkpoint_every=250):
    from .data import NormalScans
    from .model import NORMAL_VERSION
    optimizer = normal_optimizer(model, stage)
    batch = config["batch"]
    updates = math.ceil(len(order) / batch)
    completed = 0
    best = None
    selected_update = None
    if resume is not None:
        if resume["stage"] != stage or resume["config"] != config:
            raise ValueError("resume changes the model, data or training configuration")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        completed = resume["trained_updates"]
        best = resume["best_selection"]
        selected_update = resume["selected_update"]
        restore_rng(resume["rng"], device)
    if not 0 <= completed <= updates:
        raise ValueError("resume update is outside the stage budget")
    dataset = NormalScans(records, augment=True, queries=config["queries"], seed=config["seed"])
    loader = DataLoader(dataset, batch_size=None, sampler=order[completed * batch:],
                        num_workers=workers, pin_memory=device.type == "cuda",
                        prefetch_factor=1 if workers else None,
                        generator=torch.Generator().manual_seed(config["seed"]))
    iterator = iter(loader)
    frequency = config["eval_every" if stage == "source" else "target_eval_every"]
    log_path = output / "training.jsonl"
    # A resumed run discards reports newer than its last fully saved update.
    if resume is not None and log_path.exists():
        lines = [json.loads(line) for line in log_path.read_text().splitlines()]
        lines = [row for row in lines if row["stage"] != stage or row["update"] <= completed]
        log_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in lines))
    start = time.monotonic()
    initial_update = completed
    model.train()
    for update in range(completed + 1, updates + 1):
        count = min(batch, len(order) - (update - 1) * batch)
        factor = learning_rate_factor(update, updates)
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * factor
        optimizer.zero_grad(set_to_none=True)
        terms = defaultdict(float)
        for _ in range(count):
            sample = to_device(next(iterator), device)
            # Sequential accumulation averages frame losses without mixing point populations.
            loss, detail = model.loss(sample)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite {stage} loss at update {update}")
            (loss / count).backward()
            terms["objective"] += float(loss.detach()) / count
            for key, value in detail.items():
                terms[key] += float(value) / count
        finite = [torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                  if parameter.grad is not None]
        if not finite or not bool(torch.stack(finite).all()):
            raise FloatingPointError(f"nonfinite or absent {stage} gradient at update {update}")
        optimizer.step()
        trained_frames = min(update * batch, len(order))
        row = dict(stage=stage, update=update, frames=trained_frames,
                   learning_rates=[group["lr"] for group in optimizer.param_groups], **terms)
        with log_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        evaluate_now = update % frequency == 0 or update == updates
        if evaluate_now:
            state = rng_state(device)
            measured = normal_development(model, development, selection, device, workers,
                                          queries=config["queries"], seed=config["seed"])
            restore_rng(state, device)
            model.train()
            quality = normal_selection(measured)
            if best is None or quality > tuple(best):
                best, selected_update = quality, update
                atomic_save(output / f"{stage}_best.pt", dict(version=NORMAL_VERSION, config=config, mode="normal_hypothesis",
                    model=model.state_dict(), stage=stage, update=update, development=measured))
                write_json(output / f"{stage}_development.json", measured)
            print(f"{stage} {update}/{updates}: normal mIoU={measured['mean_iou_gt']:.6f}; "
                  f"selected={selected_update}", flush=True)
        stages[stage] = dict(trained_frames=trained_frames, trained_updates=update,
                             planned_frames=len(order), planned_updates=updates,
                             budget_complete=update == updates, selected_update=selected_update)
        if evaluate_now or update % checkpoint_every == 0:
            # Only a completed optimizer update is resumable; no partial batch is published.
            disk_check(500_000_000)
            atomic_save(output / "last.pt", dict(version=NORMAL_VERSION, config=config, mode="normal_hypothesis",
                model=model.state_dict(), optimizer=optimizer.state_dict(), stage=stage,
                trained_updates=update, best_selection=best, selected_update=selected_update,
                stages=stages, rng=rng_state(device)))
            write_json(output / "stages.json", stages)
        if update % 50 == 0 or update == updates:
            elapsed = time.monotonic() - start
            print(f"{stage} {update}/{updates}: objective={terms['objective']:.5f}, "
                  f"{(update-initial_update)/max(elapsed,1e-9):.2f} updates/s", flush=True)
    del iterator, loader
    return stages[stage]


def training_data(source_directory, stu_root):
    from .data import normal_records
    source = normal_records("nuscenes", directory=source_directory)
    source_development = normal_records("nuscenes", development=True, directory=source_directory)
    target = normal_records("206", root=stu_root)
    target_development = normal_records("201", development=True, root=stu_root)
    counts = tuple(map(len, (source, source_development, target, target_development)))
    if counts != (28130, 6019, 449, 682):
        raise ValueError(f"paper training populations require (28130, 6019, 449, 682), got {counts}")
    return source, source_development, target, target_development


def run(args):
    from .data import (NormalScans, NORMAL_CLASSES, NUSCENES_NORMAL_SETS,
                       STU_NORMAL_SEMANTICS)
    from .model import (NormalHypothesis, NORMAL_ARCHITECTURE, NORMAL_SCORE_VERSION,
                        NORMAL_VERSION)
    device = torch.device(args.device)
    resources = runtime_snapshot()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    processors = len(resources["cpu_affinity"])
    # Bounded CPU prefetch avoids replicating full scans across all available cores.
    workers = (min(8, max(0, processors - 2), int(resources["memory_available"] // 1_500_000_000))
               if args.workers is None else args.workers)
    if workers < 0 or workers >= processors:
        raise ValueError("workers must leave a CPU for model execution")
    torch.set_num_threads(min(4, max(1, processors - workers)))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    disk_check(1_500_000_000)
    source, source_dev, target, target_dev = training_data(args.source_directory, args.stu_root)
    pools, present = normal_replay_pools(source, target)
    orders = {stage: normal_order(len(source), len(target), epochs, args.seed, stage, pools)
              for stage, epochs in (("source", 2), ("target", 8))}
    selection = dict(source=development_indices(source_dev, 50, by_scene=True),
                     target=development_indices(target_dev, 68))
    seed_all(args.seed)
    model = NormalHypothesis(args.variant).to(device)
    initial = model.load_pretrained(args.initial)
    config = dict(version=NORMAL_VERSION, architecture=NORMAL_ARCHITECTURE,
                  score_version=NORMAL_SCORE_VERSION, variant=args.variant, seed=args.seed,
                  classes=list(NORMAL_CLASSES), source_mapping=NUSCENES_NORMAL_SETS,
                  target_mapping=STU_NORMAL_SEMANTICS, synthetic_anomalies=False,
                  source_epochs=2, target_epochs=8, batch=2, queries=4096,
                  eval_every=2000, target_eval_every=225, selection=NORMAL_SELECTION,
                  initial_sha256=initial["sha256"], loss=model.loss_weights(),
                  source_replay_fraction=.2,
                  source_replay=dict(rule="one source replay per four target scans; half prioritize absent refined classes",
                                     refinement_classes=sorted(pools), target_training_label_classes=present),
                  data_identity={name: identity(rows) for name, rows in
                                 (("source", source), ("source_validation", source_dev),
                                  ("target", target), ("target_validation", target_dev))},
                  source_validation_indices=selection["source"], target_validation_indices=selection["target"],
                  budget={stage: dict(visits=len(order), updates=math.ceil(len(order)/2),
                                      order_identity=identity(order),
                                      schedule="3% linear warmup times cosine decay to 5%; indexed by optimizer update")
                          for stage, order in orders.items()},
                  precision="float32", calibrated_output=args.calibrate)
    # JSON normalization also makes equality stable after restoring a saved run.
    config = json.loads(json.dumps(config))
    output = args.output
    resume = None
    if args.resume:
        resume = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        NormalHypothesis.validate_checkpoint(resume)
        if resume["config"] != config:
            raise ValueError("resume changes the data, seed, variant or scientific configuration")
        stages = resume["stages"]
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("training output is not empty; use --resume for this exact run")
        output.mkdir(parents=True, exist_ok=True)
        stages = {}
        write_json(output / "config.json", config)
    if args.match_run:
        normal_reference(args.match_run, config)
    write_json(output / "resources.json", dict(**resources, workers=workers, threads=torch.get_num_threads()))
    for stage, records, development in (("source", source, source_dev),
                                        ("target", target + source, target_dev)):
        if resume is not None and resume["stage"] == "target" and stage == "source":
            continue
        if resume is None or resume["stage"] != stage:
            if stage == "target":
                selected = torch.load(output / "source_best.pt", map_location="cpu", weights_only=False)
                model.load_state_dict(selected["model"], strict=True)
        train_stage(model, records, development, selection[stage], orders[stage], config,
                    stage, output, device, workers, stages=stages,
                    resume=resume if resume is not None and resume["stage"] == stage else None,
                    checkpoint_every=args.checkpoint_every)
        resume = None
    selected = torch.load(output / "target_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"], strict=True)
    cssr_reference = None
    if args.variant == "cssr":
        def training_loader(augment):
            return DataLoader(NormalScans(source + target, augment=augment, seed=args.seed),
                              batch_size=None, num_workers=workers, pin_memory=device.type == "cuda",
                              prefetch_factor=1 if workers else None,
                              generator=torch.Generator().manual_seed(args.seed))
        cssr_reference = model.fit_cssr_reference(training_loader(False), training_loader(True), device)
    reference = dict(seed=args.seed) if args.calibrate else None
    measured = normal_development(model, target_dev, list(range(len(target_dev))), device,
                                  workers, seed=args.seed, calibration=reference)
    write_json(output / "normal201.json", measured)
    if args.calibrate:
        normal_calibration(model, target_dev, device, workers, output, seed=args.seed, reference=reference)
    normal201 = dict(mean_iou_gt=measured["mean_iou_gt"], scans=measured["scans"],
                     predictive=measured["predictive"])
    atomic_save(output / "model.pt", dict(version=NORMAL_VERSION, config=config, mode=model.mode, model=model.state_dict(),
                stages=stages, frozen=True, update=selected["update"], cssr_reference=cssr_reference,
                normal201=normal201))
    result = dict(stages=stages, checkpoint="model.pt", budget_complete=True,
                  normal201=normal201,
                  anomaly_evaluated=False, cssr_reference=cssr_reference)
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def suite(args):
    """Run the complete matched study sequentially on the available validation data."""
    from .evaluate import summarize_experiments
    from .model import NormalHypothesis, NORMAL_VARIANTS
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    variants = ("semantic", "joint", "separate", *[name for name in NORMAL_VARIANTS
                                                  if name not in ("semantic", "joint", "separate")])
    status = dict(seeds=[206, 307, 409], variants=list(variants), completed=[],
                  test_status="unavailable: neither hidden test inputs nor an official evaluation service were supplied")
    if (root / "progress.json").exists():
        previous = json.loads((root / "progress.json").read_text())
        if previous["seeds"] != status["seeds"] or previous["variants"] != status["variants"]:
            raise ValueError("the existing study uses a different experiment matrix")
        status["completed"] = previous["completed"]
    common = ["--device", args.device]
    if args.workers is not None:
        common += ["--workers", str(args.workers)]

    def command(module, arguments, task):
        status["active"] = task
        status.pop("failure", None)
        write_json(root / "progress.json", status)
        print(f"\nStudy task: {task}", flush=True)
        try:
            subprocess.run([sys.executable, "-u", "-m", module, *map(str, arguments)], check=True)
        except subprocess.CalledProcessError as error:
            status.update(active=None, failure=dict(task=task, exit_code=error.returncode))
            write_json(root / "progress.json", status)
            raise
        if task not in status["completed"]:
            status["completed"].append(task)
        write_json(root / "progress.json", status)

    for seed in status["seeds"]:
        for variant in variants:
            folder = root / f"{variant}-{seed}"
            checkpoint = folder / "model.pt"
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                NormalHypothesis.validate_checkpoint(saved, require_fitted=True)
                if (saved["config"]["variant"] != variant or saved["config"]["seed"] != seed
                        or not saved.get("frozen") or not all(row["budget_complete"] for row in saved["stages"].values())):
                    raise ValueError(f"existing run does not complete this study task: {folder}")
                normal_reference(folder, saved["config"])
                del saved
            else:
                arguments = ["--variant", variant, "--seed", seed, "--output", folder,
                             "--initial", args.initial, "--source-directory", args.source_directory,
                             "--stu-root", args.stu_root, "--checkpoint-every", args.checkpoint_every, *common]
                if (folder / "last.pt").exists():
                    arguments.append("--resume")
                if variant != "semantic":
                    arguments += ["--match-run", root / f"semantic-{seed}"]
                command("src.train", arguments, f"train {variant} seed {seed}")
            if variant not in ("semantic", "joint", "separate"):
                for readout in (("energy", "softmax") if variant == "standard" else ("joint",)):
                    name = "val_" + readout if variant == "standard" else "val"
                    destination = folder / (name + ".json")
                    if not destination.exists():
                        command("src.evaluate", ["validate", "--checkpoint", checkpoint, "--manifest", args.manifest,
                                "--readout", readout, "--output", destination, *common],
                                f"validate {variant}/{readout} seed {seed}")
        paths = [item for name in ("semantic", "joint", "separate")
                 for item in ("--" + name, root / f"{name}-{seed}" / "model.pt")]
        paired = root / f"paired-{seed}"
        comparison_path = paired / "comparison.json"
        if not comparison_path.exists():
            # Interrupted point exports are deterministic scratch arrays owned by this task.
            if paired.exists():
                expected = {f"{name}.npy" for name in ("semantic", "joint", "separate", "joint_appearance",
                                                     "joint_common_density", "joint_independent_minima", "returns", "labels")}
                files = list(paired.iterdir())
                if any(not path.is_file() or path.name not in expected for path in files):
                    raise ValueError(f"unrecognized incomplete comparison contents in {paired}")
                for path in files:
                    path.unlink()
            command("src.evaluate", ["compare", *paths, "--manifest", args.manifest, "--output", paired,
                    "--fixed-readouts", "--discard-fixed-records", "--normal-fpr", .001, .005, .01, .02, .05, *common],
                    f"paired validation seed {seed}")
        comparison = json.loads(comparison_path.read_text())
        for variant in ("semantic", "joint", "separate"):
            folder = root / f"{variant}-{seed}"
            config = json.loads((folder / "config.json").read_text())
            normal = json.loads((folder / "normal201.json").read_text())
            # These metrics use the same forward pass as the paired decisions.
            write_json(folder / "val.json", dict(variant=variant, seed=seed, readout="joint", split="val",
                complete=True, version=config["version"], architecture=config["architecture"],
                score_version=config["score_version"], manifest_sha256=comparison["manifest_sha256"],
                metrics=comparison["methods"][variant]["metrics"], normal201=normal,
                points=comparison["points"], scans=comparison["scans"], source=str(comparison_path)))
        if not (paired / "normal.json").exists():
            command("src.evaluate", ["compare-normal", *paths, "--data", args.stu_root, "--fixed-readouts",
                    "--output", paired / "normal.json", *common], f"paired normal decisions seed {seed}")
        for variant in ("semantic", "joint"):
            folder = root / f"{variant}-{seed}"
            if not (folder / "runtime.json").exists():
                command("src.evaluate", ["benchmark", "--checkpoint", folder / "model.pt",
                        "--manifest", args.manifest, "--output", folder / "runtime.json",
                        "--device", args.device], f"runtime {variant} seed {seed}")
    write_json(root / "summary.json", summarize_experiments(root))
    subprocess.run([sys.executable, "figures/preview.py", "--comparisons",
                    *[str(root / f"paired-{seed}" / "comparison.json") for seed in status["seeds"]]], check=True)
    status.update(active=None, validation_complete=True, test_complete=False)
    write_json(root / "progress.json", status)


def main():
    from .data import DATA_ROOT
    from .model import NORMAL_VARIANTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=NORMAL_VARIANTS, default="joint")
    parser.add_argument("--seed", type=int, choices=(206, 307, 409), default=206)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", type=Path, default=Path("assets/nuscenes.pth"))
    parser.add_argument("--source-directory", type=Path, default=Path("results/data/background"))
    parser.add_argument("--stu-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--match-run", type=Path, help="completed paired run with the same seed and training budget")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--calibrate", action="store_true", help="fit the optional monotone normal-reference output scale")
    parser.add_argument("--suite", action="store_true", help="run all nine variants and three seeds, validation, paired analyses and runtime")
    parser.add_argument("--manifest", type=Path, default=Path("assets/val.json"))
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("checkpoint interval must be positive")
    if args.suite:
        if args.calibrate or args.match_run or args.resume:
            parser.error("--suite uses raw scores and resumes each completed stage automatically")
        suite(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
