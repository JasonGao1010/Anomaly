"""Normal-only SERVE training, development and calibration utilities.

The repository currently has no end-to-end SERVE training entry point.
"""

from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import file_sha256, identity, write_json
from .evaluate import memory_available
from .model import to_device, LITEPT_COMMIT, WEIGHTS_REVISION, WEIGHTS_SHA256

ROOT = Path(__file__).resolve().parents[1]
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


@torch.no_grad()
def normal_development(model, records, indices, device, workers, *, queries=4096, calibration=None, deadline=None):
    from .data import NormalScans
    model.eval()
    if calibration is not None:
        if (list(indices) != list(range(len(records))) or not records
                or any(row.get("source") != "normal_stu" or row.get("scene") != "201" for row in records)):
            raise ValueError("calibration reuse requires the complete ordered STU 201 development population")
        calibration.update(frames=[], scores=[], data_identity=identity(records), variant=model.variant)
    data = NormalScans(records, queries=queries)
    loader = DataLoader(data, batch_size=None, sampler=indices, num_workers=workers,
                        pin_memory=True, prefetch_factor=1 if workers else None)
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
    return dict(**measured, **summaries["joint"], semantic_only=summaries["semantic"],
                joint_comparison={key: int(totals[key]) for key in ("set_points", "corrected_points", "worsened_points")},
                per_class_comparison=dict(corrected_points=class_changes[:, 0].tolist(),
                                          worsened_points=class_changes[:, 1].tolist(),
                                          population="reliable singleton-label points; class index is ground truth"),
                semantic_definition="formal inference decision for this variant; semantic_only is an internal diagnostic of these same weights, not an independently trained baseline",
                iou_definition="pooled reliable singleton-label points; mean_iou_gt averages the fixed ground-truth-present classes; mean_iou_present averages nonzero unions; set accuracy also includes coarse labels",
                strata=strata, observation_diagnostics={key: value.tolist() for key, value in diagnostics.items()},
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


def normal_baseline_comparison(path, config, measured):
    other, _ = normal_reference(path, config)
    if other.get("variant") != "semantic":
        raise ValueError("an independent normal semantic baseline must use variant=semantic")
    baseline = json.loads((Path(path) / "normal201.json").read_text())
    if baseline["ground_truth_points"] != measured["ground_truth_points"] or baseline["scans"] != measured["scans"]:
        raise ValueError("normal baseline and method development must cover the identical labeled population")
    difference = [None if first is None or second is None else first - second
                  for first, second in zip(measured["iou"], baseline["iou"])]
    return dict(baseline=str(Path(path).resolve()), comparison="independently trained semantic variant; identical normal data, frame order and nominal update schedule",
                mean_iou_gt_difference=measured["mean_iou_gt"] - baseline["mean_iou_gt"],
                set_accuracy_difference=measured["set_accuracy"] - baseline["set_accuracy"],
                per_class_iou_difference=difference, ground_truth_points=measured["ground_truth_points"],
                absent_classes=measured["absent_classes"])


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
        loader = DataLoader(NormalScans(records, queries=1), batch_size=None, num_workers=workers,
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
                  score_definition=f"minimum formal class energy for variant={model.variant}; identical raw_score as inference",
                  meaning="one monotone normal-reference transform preserves the joint score ordering; not an anomaly probability or anomaly-performance estimate")
    write_json(output / "calibration.json", result)
    return result
