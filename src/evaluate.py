"""Official pooled-point validation, raw-slot inference and inference timing."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (Scans, VERSION, PILOT_VERSION, CONTINUATION_VERSION, NATIVE_VERSION, NDP_VERSION, SOURCE_VERSION,
                   load_manifest, make_real_manifest, point_targets, read_scan, read_nuscenes, write_json)
from .model import Segmentor, prepare_scan, scatter_scores, to_device
from .normal import angular_observation, prediction_metrics, SCALES
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


STU_COMMIT = "8f0f09c2ca4bf7b665e0ae5919b4092ddae140a2"


def normal_record(record):
    return (record.get("source") in ("nuscenes", "normal_stu")
            and record.get("group") in ("normal_nuscenes", "normal_stu")
            and not record.get("anomaly", 0) and not record.get("delta") and not record.get("augmented", False))


class PreparedScans(Scans):
    def __init__(self, manifest, *, relations=False, normal=False, voxel=True, normal_reference=True, hypotheses=False):
        super().__init__(manifest)
        self.relations = relations
        self.normal, self.voxel = normal, voxel
        self.normal_reference = normal_reference
        self.hypotheses = hypotheses

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        result = (prepare_scan(sample, relations=self.relations) if self.voxel else
                  {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                   for key, value in sample.items()})
        if self.hypotheses:
            from .normal import hypothesis_observation
            result["observation"] = hypothesis_observation(sample["xyzi"])
        if self.manifest["version"] == SOURCE_VERSION and self.manifest["kind"] == "train":
            record = self.records[index]
            control = np.zeros(len(sample["targets"]), dtype=bool)
            if record.get("group") == "control_nuscenes":
                # Delta slots identify actual inserted returns, not the surrounding
                # normal background or occluded/ignored replacement points.
                with np.load(record["delta"], allow_pickle=False) as delta:
                    inserted = delta["slots"][delta["labels"] == 1]
                control = np.isin(sample["slots"], inserted) & (sample["targets"] == 0)
                if int(control.sum()) != record["inserted_points"]:
                    raise ValueError("decoded normal-control points differ from manifest")
            result["control_mask"] = torch.from_numpy(control)
        if self.normal:
            result["observation"] = angular_observation(sample["xyzi"])
            # Only unmodified real-normal sources anchor the auxiliary task.
            record = self.records[index]
            source_only = self.manifest["version"] == SOURCE_VERSION
            result["normal_training"] = not bool(record.get("delta")) if source_only else normal_record(record)
            original = None
            if self.normal_reference and self.manifest["version"] == NDP_VERSION:
                original = self._source(record["frame"])
            elif self.normal_reference and source_only and record.get("delta"):
                original = read_nuscenes({key: value for key, value in record.items() if key != "delta"},
                                        self.manifest["mapping"])
            if original is not None:
                # The auxiliary target is the unchanged source, never a pasted
                # point relabeled as normal. Original ignored classes stay ignored.
                result["normal_reference"] = dict(normal_training=True,
                    observation=angular_observation(original.xyzi[original.actual]),
                    targets=torch.from_numpy(point_targets(original)[original.actual].copy()))
        return result


@torch.no_grad()
def evaluate_normal(model, manifest, device, workers=2):
    if model.normal is None:
        raise ValueError("this checkpoint has no normal return field")
    indices = [i for i, row in enumerate(manifest["records"])
               if normal_record(row) or manifest["version"] in (NDP_VERSION, SOURCE_VERSION)]
    if not indices:
        raise ValueError("manifest has no reliable unmodified normal scans")
    data = PreparedScans(manifest, normal=True, voxel=False)
    loader = DataLoader(data, batch_size=None, sampler=indices, num_workers=workers,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(0),
        **({"prefetch_factor": 1} if workers else {}))
    totals = {str(size): dict(mae_m=0., coverage90=0., width90_m=0., joint_nll=0.) for size in SCALES}
    model.eval()
    points, start = 0, time.perf_counter()
    for sample in loader:
        sample = sample.get("normal_reference", sample)
        measured = prediction_metrics(model.normal, to_device(sample, device))
        points += measured["points"]
        for size, values in measured["scales"].items():
            for key, value in values.items():
                totals[size][key] += value * (1 if key == "joint_nll" else measured["points"])
    for values in totals.values():
        for key in values:
            values[key] /= len(indices) if key == "joint_nll" else points
    return dict(scales=totals, scans=len(indices), points=points, seconds=time.perf_counter() - start,
                manifest_sha256=manifest["sha256"], role="normal prediction diagnostic; not anomaly detection accuracy")


def precision(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast(device):
    return torch.autocast(device.type, dtype=precision(device), enabled=device.type == "cuda")


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot determine available physical memory")


def better(metrics, previous):
    if not all(np.isfinite(metrics[k]) for k in ("AP", "FPR95", "AUROC")):
        raise ValueError("nonfinite official validation metrics")
    return previous is None or (metrics["AP"], -metrics["FPR95"], metrics["AUROC"]) > (
        previous["AP"], -previous["FPR95"], previous["AUROC"])


def evaluation_indices(manifest):
    """Keep reference selection separate from the official five-point rule."""
    reference = manifest.get("version") == SOURCE_VERSION and manifest.get("evaluation_role") == "reference"
    return [i for i, row in enumerate(manifest["records"])
            if row["eligible"] and (not reference or row.get("role") == "reference")]


@torch.no_grad()
def evaluate(model, manifest, device, workers=4, score_path=None, record_points=False):
    """Call the pinned official implementation once across the complete valid set."""
    if manifest["kind"] not in ("val", "test"):
        raise ValueError("evaluation requires a held-out validation or test manifest")
    indices = evaluation_indices(manifest)
    selection = (dict(evaluation_role="reference")
                 if manifest.get("version") == SOURCE_VERSION and manifest.get("evaluation_role") == "reference" else {})
    count = sum(manifest["records"][i]["normal"] + manifest["records"][i]["anomaly"] for i in indices)
    if not count:
        raise ValueError("no eligible evaluation points")
    # Budget sklearn's exact sorting, targets and cumulative sums; never use swap.
    required = 64 * count + 1_000_000_000
    if memory_available() < required:
        raise RuntimeError(f"official metrics require about {required / 1e9:.1f} GB free RAM")
    dataset = PreparedScans(manifest, relations=getattr(model, "relation", None) is not None,
                            normal=getattr(model, "normal", None) is not None, normal_reference=False,
                            hypotheses=model.mode == "normal_hypothesis")
    loader = DataLoader(dataset, batch_size=None, sampler=indices, num_workers=workers,
                        pin_memory=device.type == "cuda",
                        **({"prefetch_factor": 1} if workers else {}),
                        generator=torch.Generator().manual_seed(0))
    model.eval()
    calculator = PointOODMetricsCalculator()
    # Optional diagnostic export preserves official point order and exact scores.
    scores = (np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(count,))
              if score_path is not None else np.empty(count, np.float32))
    labels = np.empty(count, np.int8)
    raw_scores = identities = None
    frames, raw_cursor = [], 0
    if record_points:
        if score_path is None:
            raise ValueError("point recording needs a persistent evaluation score path")
        score_path = Path(score_path)
        raw_count = sum(manifest["records"][i]["points"] for i in indices)
        raw_scores = np.lib.format.open_memmap(score_path.with_stem(score_path.stem + "_all"), mode="w+",
                                               dtype=np.float32, shape=(raw_count,))
        identity_path = score_path.parent / "val_points.npy"
        identities = np.lib.format.open_memmap(identity_path, mode="r" if identity_path.exists() else "w+",
            dtype=np.dtype([("slot", "<u4"), ("target", "i1")]), shape=(raw_count,))
        if identities.shape != (raw_count,):
            raise ValueError("evaluation point population changed")
    cursor = 0
    start = time.perf_counter()
    for number, sample in enumerate(loader, 1):
        batch = to_device(sample, device)
        with autocast(device):
            prediction = model(batch)
        if not torch.isfinite(prediction).all():
            raise ValueError("nonfinite model prediction")
        prediction = prediction.cpu().numpy()
        # Recover official three-value truth from point targets only for scoring.
        truth = np.where(sample["targets"].numpy() < 0, 0, sample["targets"].numpy() + 1)
        calculator.update(sample["xyzi"][:, :3].numpy(), prediction, truth)
        selected_scores = calculator.all_scores.pop()
        selected_labels = calculator.all_labels.pop()
        stop = cursor + len(selected_scores)
        scores[cursor:stop], labels[cursor:stop] = selected_scores, selected_labels
        if record_points:
            raw_stop = raw_cursor + len(prediction)
            raw_scores[raw_cursor:raw_stop] = prediction
            for key, source in (("slot", "slots"), ("target", "targets")):
                values = sample[source].numpy()
                if identities.flags.writeable:
                    identities[raw_cursor:raw_stop][key] = values
                elif not np.array_equal(identities[raw_cursor:raw_stop][key], values):
                    raise ValueError("evaluation point identity or label changed")
            frames.append(dict(index=int(sample["index"]), start=raw_cursor, stop=raw_stop,
                               metric_start=cursor, metric_stop=stop))
            raw_cursor = raw_stop
        cursor = stop
        if number % 100 == 0 or number == len(indices):
            print(f"validation {number}/{len(indices)} scans, {cursor}/{count} points", flush=True)
    if cursor != count:
        raise ValueError("official evaluation count differs from the fixed manifest")
    if score_path is not None:
        scores.flush()
    if record_points:
        if raw_cursor != raw_count:
            raise ValueError("full-return evaluation recording omitted points")
        raw_scores.flush()
        identities.flush()
        write_json(score_path.parent / "val.json", dict(manifest_sha256=manifest["sha256"], frames=frames,
            points=raw_count, metric_points=count, identities="val_points.npy",
            scope="Each val*.npy is the exact selected metric population; its *_all.npy companion preserves every actual return, including ignored context, in the selected eligible full scans. Source reference selection, when declared, also excludes coverage and control roles.",
            **selection))
        del raw_scores, identities
    del batch, sample, dataset, loader
    gc.collect()
    # Integer 0/1 labels are exact in int8; sklearn still uses its own float64 sums.
    calculator.all_scores, calculator.all_labels = [scores], [labels]
    metrics = {k: float(v) for k, v in calculator.compute_metrics().items()}
    better(metrics, None)
    return dict(metrics=metrics, scans=len(indices), points=count,
                seconds=time.perf_counter() - start, manifest_sha256=manifest["sha256"],
                official_commit=STU_COMMIT, **selection)


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") == "AJAE-normal-hypothesis":
        from .model import NormalHypothesis, SCORE_VERSION
        if bool(saved["model"]["calibrated"]) and saved["config"].get("score_version") != SCORE_VERSION:
            raise ValueError("normal references were fitted for another score; refit with --calibration-only")
        model = NormalHypothesis()
        model.load_state_dict(saved["model"], strict=True)
        return model.to(device).eval(), saved
    if saved.get("version") not in (VERSION, PILOT_VERSION, CONTINUATION_VERSION, NATIVE_VERSION, NDP_VERSION, SOURCE_VERSION):
        raise ValueError("checkpoint does not belong to a supported V4 experiment")
    if saved["mode"] == "field" and "compatibility.tokens.0.weight" in saved["model"]:
        raise ValueError("checkpoint uses the retired kernel readout; use its recorded code revision for historical evaluation")
    model = Segmentor(saved["mode"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


@torch.no_grad()
def infer(model, scan, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    io_timing = {}
    frame = read_scan(scan, io_timing=io_timing)
    read_seconds = io_timing["seconds"]
    if frame.actual.any():
        xyzi = frame.xyzi[frame.actual].copy()
        sample = prepare_scan(dict(xyzi=xyzi, slots=frame.return_slots,
                                   targets=np.full(int(frame.actual.sum()), -1, np.int8),
                                   slot_count=len(frame.xyzi), index=frame.frame_id),
                              relations=getattr(model, "relation", None) is not None)
        if getattr(model, "normal", None) is not None:
            sample["observation"] = angular_observation(xyzi)
        elif model.mode == "normal_hypothesis":
            from .normal import hypothesis_observation
            sample["observation"] = hypothesis_observation(xyzi)
        sample = to_device(sample, device)
        with autocast(device):
            prediction = model(sample)
        prediction = scatter_scores(prediction, sample["slots"], sample["slot_count"])
        if not torch.isfinite(prediction).all():
            raise ValueError("nonfinite inference output")
        prediction = prediction.cpu().numpy()
    else:
        prediction = np.zeros(len(frame.xyzi), np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return prediction, dict(read_seconds=read_seconds, seconds=time.perf_counter() - start - read_seconds,
                            peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                            real_points=int(frame.actual.sum()), slots=len(frame.xyzi))


def summarize(output, split="val"):
    """Report the fixed seed-0 comparison without invented repeat uncertainty."""
    output = Path(output)
    report = {}
    identities = set()
    for method in ("base", "attention", "continue", "fusion"):
        path = output / "0" / method / ("result.json" if split == "val" else "test.json")
        if not path.is_file():
            raise ValueError(f"incomplete seed-0 comparison: {path}")
        row = json.loads(path.read_text())
        if row.get("version") != VERSION or row["seed"] != 0 or row["method"] != method:
            raise ValueError(f"incorrect experiment identity: {path}")
        if not row.get("complete"):
            raise ValueError(f"unfinished experiment: {path}")
        identities.add(row["val_manifest_sha256"] if split == "val" else row["manifest_sha256"])
        metrics = row["best_metrics"] if split == "val" else row["metrics"]
        better(metrics, None)
        epoch = row["best_epoch"] if split == "val" else row["checkpoint_epoch"]
        report[method] = dict(seed=0, epoch=epoch, **metrics)
        if split == "val":
            report[method]["validation_improved_from_epoch0"] = row["validation_improved_from_epoch0"]
    if len(identities) != 1:
        raise ValueError("results use different evaluation sets")
    write_json(output / ("summary.json" if split == "val" else "test_summary.json"),
               dict(version=VERSION, seeds=[0], repeat_uncertainty_estimated=False,
                    split=split, methods=report))


def mining_indices(manifest):
    """Bounded training-source inspection, chosen before seeing model scores."""
    groups = {}
    for index, row in enumerate(manifest["records"]):
        if row.get("subset") != "train":
            raise ValueError("hard-example mining must exclude internal checks and validation")
        key = (row["group"], row.get("scene", row.get("geometry", row.get("world", "206"))))
        groups.setdefault(key, []).append(index)
    indices = []
    for (group, _), rows in sorted(groups.items()):
        count = {"base": 1, "targeted": 3, "normal_nuscenes": 3, "normal_stu": 31}[group]
        key = "anomaly" if group == "targeted" else "frame"
        rows.sort(key=lambda i: (manifest["records"][i][key], manifest["records"][i]["frame"]))
        positions = [len(rows) // 2] if count == 1 else np.linspace(0, len(rows) - 1, min(count, len(rows))).round().astype(int)
        indices.extend(rows[int(p)] for p in positions)
    return sorted(set(indices))


@torch.no_grad()
def mine(model, manifest, checkpoint, output, device, workers=4):
    """Store both ranking tails as candidates; never modify training supervision."""
    from collections import Counter
    if manifest["kind"] != "train":
        raise ValueError("mining requires training sources")
    indices = mining_indices(manifest)
    dataset = PreparedScans(manifest, relations=getattr(model, "relation", None) is not None,
                            normal=getattr(model, "normal", None) is not None)
    loader = DataLoader(dataset, batch_size=None, sampler=indices, num_workers=workers,
                        pin_memory=device.type == "cuda",
                        **({"prefetch_factor": 1} if workers else {}),
                        generator=torch.Generator().manual_seed(0))
    model.eval()
    candidates, scans = [], []
    for number, sample in enumerate(loader, 1):
        index = int(sample["index"])
        row = manifest["records"][index]
        batch = to_device(sample, device)
        with autocast(device):
            prediction = model(batch)
        scores = prediction.float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite mining scores")
        xyz, labels, slots = sample["xyzi"][:, :3].numpy(), sample["targets"].numpy(), sample["slots"].numpy()
        summary = dict(index=index, group=row["group"], frame=row["frame"])
        for label, name in ((0, "high_normal"), (1, "low_anomaly")):
            selected = np.flatnonzero(labels == label)
            if not len(selected):
                continue
            values = scores[selected]
            summary[name] = dict(points=len(selected), quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist())
            center = selected[np.argmax(values) if label == 0 else np.argmin(values)]
            local = selected[np.linalg.norm(xyz[selected] - xyz[center], axis=1) <= .35]
            candidate = dict(index=index, source=row, kind=name, score=float(scores[center]),
                             center_slot=int(slots[center]), sensor_xyz=xyz[center].tolist(),
                             radius_m=.35, slots=slots[local].tolist(), scores=scores[local].tolist(),
                             patch_median=float(np.median(scores[local])))
            if row.get("source") == "nuscenes":
                raw = np.fromfile(row["label"], np.uint8)
                category = manifest["mapping"][int(raw[slots[center]])]
                candidate.update(raw_semantic=category["raw"], semantic_name=category["name"])
            else:
                original = dataset._source(row["frame"])
                candidate.update(raw_semantic=int(original.semantic[slots[center]]) if label == 0 else 2,
                                 world_xyz=(xyz[center] @ original.pose[:3, :3].T + original.pose[:3, 3]).tolist())
            candidates.append(candidate)
        scans.append(summary)
        if number % 50 == 0 or number == len(indices):
            print(f"training-source inspection {number}/{len(indices)} scans", flush=True)
        del batch, prediction
    kept = []
    for kind in ("high_normal", "low_anomaly"):
        for group in ("base", "targeted", "normal_nuscenes", "normal_stu"):
            chosen = []
            pool = sorted((r for r in candidates if r["kind"] == kind and r["source"]["group"] == group),
                          key=lambda r: r["score"], reverse=kind == "high_normal")
            for row in pool:
                source = row["source"]
                if kind == "low_anomaly":
                    same = lambda old: source.get("geometry", source.get("world")) == old["source"].get("geometry", old["source"].get("world"))
                elif source.get("source") == "nuscenes":
                    same = lambda old: source["scene"] == old["source"]["scene"]
                else:
                    same = lambda old: np.linalg.norm(np.array(row["world_xyz"]) - old["world_xyz"]) < 1.
                if any(same(old) for old in chosen):
                    continue
                chosen.append(row)
                if len(chosen) == 8:
                    break
            kept.extend(chosen)
    output = Path(output)
    write_json(output / "candidates.json", dict(checkpoint=str(Path(checkpoint).resolve()),
        train_manifest=manifest["sha256"], inspected_scans=dict(Counter(r["group"] for r in scans)),
        score="raw anomaly logit; larger means more anomalous", scans=scans, candidates=kept,
        scope="Training-source candidates only; no validation threshold, new supervision, or claim of cross-scene anomaly generalization."))
    print(json.dumps(dict(inspected=len(indices), candidates=dict(Counter(r["kind"] for r in kept)))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "test", "infer", "benchmark", "mine", "normal"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        if name == "validate":
            command.add_argument("--manifest", type=Path, default=Path("assets/val.json"))
        if name == "mine":
            command.add_argument("--manifest", type=Path, default=Path("results/data/train.json"))
        if name == "normal":
            command.add_argument("--manifest", type=Path, required=True)
        if name == "test":
            command.add_argument("--data", type=Path, required=True)
        if name in ("validate", "test", "mine", "normal"):
            command.add_argument("--workers", type=int, default=4)
        else:
            command.add_argument("--scans", type=Path, nargs="+", required=True)
            if name == "benchmark":
                command.add_argument("--warmup", type=int, default=5)
                command.add_argument("--repeats", type=int, default=20)
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.action == "summarize":
        summarize(args.output, args.split)
        return
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model, saved = load_model(args.checkpoint, device)
    normal_run = saved.get("version") == "AJAE-normal-hypothesis"
    if normal_run and args.action in ("normal", "mine"):
        parser.error("this action belongs to the earlier supervised field; normal-only development is recorded by src.train --normal")
    if args.action == "normal":
        result = evaluate_normal(model, load_manifest(args.manifest, "train"), device, args.workers)
        write_json(args.output, dict(checkpoint=str(args.checkpoint.resolve()), **result))
    elif args.action == "mine":
        mine(model, load_manifest(args.manifest, "train"), args.checkpoint, args.output, device, args.workers)
    elif args.action in ("validate", "test"):
        if args.action == "test":
            if not (saved.get("frozen") if normal_run else saved.get("selected") and saved.get("complete")):
                raise ValueError("test requires a selected checkpoint from a completed fixed budget")
            manifest = make_real_manifest(args.data, partition="test", workers=args.workers)
            if manifest["directory"] == saved["config"].get("val_directory"):
                raise ValueError("the validation set cannot be presented as final test data")
        else:
            manifest = load_manifest(args.manifest, "val")
        result = evaluate(model, manifest, device, args.workers)
        metadata = (dict(version=saved["version"], complete=saved.get("frozen", False),
                         seed=saved["config"]["seed"], mode=saved["mode"], method=saved["mode"],
                         checkpoint_update=saved.get("stages", {}).get("target", {}).get("selected_update"))
                    if normal_run else dict(version=saved["version"], complete=saved["complete"], seed=saved["seed"],
                                            mode=saved["mode"], method=saved["method"], checkpoint_epoch=saved["epoch"],
                                            checkpoint_update=saved.get("successful_updates")))
        write_json(args.output, dict(**metadata, checkpoint=str(args.checkpoint.resolve()), **result))
    elif args.action == "infer":
        from .train import disk_check
        disk_check(sum(p.stat().st_size // 16 * 24 for p in args.scans))
        args.output.mkdir(parents=True, exist_ok=True)
        for index, scan in enumerate(args.scans):
            prediction, timing = infer(model, scan, device)
            folder = args.output / scan.parent.parent.name
            folder.mkdir(exist_ok=True)
            np.savetxt(folder / (scan.stem + ".txt"), prediction, fmt="%.9g")
            print(dict(scan=str(scan), **timing), flush=True)
            if (index + 1) % 100 == 0:
                disk_check()
    else:
        if args.repeats < 1 or args.warmup < 0:
            parser.error("invalid benchmark warmup or repeat count")
        timings = []
        for step in range(args.warmup + args.repeats):
            _, timing = infer(model, args.scans[step % len(args.scans)], device)
            if step >= args.warmup:
                timings.append(timing)
        latency = np.array([t["seconds"] for t in timings]) * 1000
        write_json(args.output, dict(batch_size=1, parameters=sum(p.numel() for p in model.parameters()),
                                    scans=[str(p) for p in args.scans], repeats=args.repeats,
                                    mean_ms=float(latency.mean()), p95_ms=float(np.percentile(latency, 95)),
                                    mean_read_ms=float(np.mean([t["read_seconds"] for t in timings]) * 1000),
                                    peak_vram_bytes=max(t["peak_vram_bytes"] for t in timings),
                                    hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"))


if __name__ == "__main__":
    main()
