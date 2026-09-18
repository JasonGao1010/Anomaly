"""Official pooled-point validation, raw-slot inference and inference timing."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (Scans, VERSION, load_manifest, make_real_manifest,
                   read_scan, write_json)
from .model import Segmentor, prepare_scan, scatter_scores, to_device
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


STU_COMMIT = "8f0f09c2ca4bf7b665e0ae5919b4092ddae140a2"


class PreparedScans(Scans):
    def __getitem__(self, index):
        return prepare_scan(super().__getitem__(index))


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


@torch.no_grad()
def evaluate(model, manifest, device, workers=4):
    """Call the pinned official implementation once across the complete valid set."""
    if manifest["kind"] not in ("val", "test"):
        raise ValueError("evaluation requires a real held-out STU manifest")
    indices = [i for i, row in enumerate(manifest["records"]) if row["eligible"]]
    count = sum(manifest["records"][i]["normal"] + manifest["records"][i]["anomaly"] for i in indices)
    if not count:
        raise ValueError("no eligible STU evaluation points")
    # Budget sklearn's exact sorting, targets and cumulative sums; never use swap.
    required = 64 * count + 1_000_000_000
    if memory_available() < required:
        raise RuntimeError(f"official metrics require about {required / 1e9:.1f} GB free RAM")
    dataset = PreparedScans(manifest)
    loader = DataLoader(dataset, batch_size=None, sampler=indices, num_workers=workers,
                        pin_memory=device.type == "cuda",
                        **({"prefetch_factor": 1} if workers else {}),
                        generator=torch.Generator().manual_seed(0))
    model.eval()
    calculator = PointOODMetricsCalculator()
    scores = np.empty(count, np.float32)
    labels = np.empty(count, np.int8)
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
        cursor = stop
        if number % 100 == 0 or number == len(indices):
            print(f"validation {number}/{len(indices)} scans, {cursor}/{count} points", flush=True)
    if cursor != count:
        raise ValueError("official evaluation count differs from the fixed manifest")
    del batch, sample, dataset, loader
    gc.collect()
    # Integer 0/1 labels are exact in int8; sklearn still uses its own float64 sums.
    calculator.all_scores, calculator.all_labels = [scores], [labels]
    metrics = {k: float(v) for k, v in calculator.compute_metrics().items()}
    better(metrics, None)
    return dict(metrics=metrics, scans=len(indices), points=count,
                seconds=time.perf_counter() - start, manifest_sha256=manifest["sha256"],
                official_commit=STU_COMMIT)


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") != VERSION:
        raise ValueError("checkpoint does not belong to F240-R2")
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
        sample = prepare_scan(dict(xyzi=frame.xyzi[frame.actual].copy(), slots=frame.return_slots,
                                   targets=np.full(int(frame.actual.sum()), -1, np.int8),
                                   slot_count=len(frame.xyzi), index=frame.frame_id))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "test", "infer", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        if name == "validate":
            command.add_argument("--manifest", type=Path, default=Path("assets/val.json"))
        if name == "test":
            command.add_argument("--data", type=Path, required=True)
        if name in ("validate", "test"):
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
    model, saved = load_model(args.checkpoint, device)
    if args.action in ("validate", "test"):
        if args.action == "test":
            if not saved.get("selected") or not saved.get("complete"):
                raise ValueError("test requires a selected checkpoint from a completed fixed budget")
            manifest = make_real_manifest(args.data, partition="test", workers=args.workers)
            if manifest["directory"] == saved["config"]["val_directory"]:
                raise ValueError("the validation set cannot be presented as final test data")
        else:
            manifest = load_manifest(args.manifest, "val")
        result = evaluate(model, manifest, device, args.workers)
        write_json(args.output, dict(version=VERSION, complete=saved["complete"],
                                    checkpoint=str(args.checkpoint.resolve()), seed=saved["seed"],
                                    mode=saved["mode"], method=saved["method"],
                                    checkpoint_epoch=saved["epoch"], **result))
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
