"""Task losses and explicit-budget AJAE V1 execution; check never updates weights."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from .data import FrozenDataset, FrozenFrame, _atomic_json, host_disk, source_identity
from .model import AJAE, ScanTransform, load_config, to_device
from .protocol import PROJECT_ROOT


def detection_loss(scores, target):
    """Each present class retains its declared half weight, even in zero-anomaly batches."""
    zero = scores.sum() * 0
    normal, anomaly = scores[target == 0], scores[target == 1]
    return .5 * ((F.softplus(normal).mean() if len(normal) else zero)
                 + (F.softplus(-anomaly).mean() if len(anomaly) else zero))


def keep_loss(original, inserted, mode="worst"):
    if original.shape != inserted.shape:
        raise ValueError("normal pairs must have the same physical return identity")
    if not len(original):
        return (original.sum() + inserted.sum()) * 0
    before, after = F.softplus(original), F.softplus(inserted)
    if mode == "worst":
        return torch.maximum(before, after).mean()
    if mode == "mean":
        return .5 * (before.mean() + after.mean())
    raise ValueError("unknown normal-pair objective")


def tail_loss(scores, target, frames, config, generator):
    """Live-batch tail approximation; no stale score queue or metric-unbiased claim."""
    normal, anomaly = torch.where(target == 0)[0], torch.where(target == 1)[0]
    if not len(normal) or not len(anomaly):
        return scores.sum() * 0, dict(pairs=0, cross_frame_pairs=0)
    high = normal[torch.argsort(scores[normal], descending=True, stable=True)
                  [:max(1, math.ceil(len(normal) * config["normal_tail_fraction"]))]]
    low = anomaly[torch.argsort(scores[anomaly], stable=True)
                  [:max(1, math.ceil(len(anomaly) * config["anomaly_tail_fraction"]))]]
    losses, cross = [], 0
    count = config["pairs_per_tail"]
    for apool, npool in ((anomaly, high), (low, normal)):
        a = apool[torch.randint(len(apool), (count,), generator=generator, device=scores.device)]
        n = npool[torch.randint(len(npool), (count,), generator=generator, device=scores.device)]
        losses.append(F.softplus((config["margin"] + scores[n] - scores[a]) / config["temperature"]).mean())
        cross += int((frames[a] != frames[n]).sum())
    return .5 * (losses[0] + losses[1]), dict(pairs=2 * count, cross_frame_pairs=cross)


def auxiliary_fraction(step, config):
    return min(1., max(0., (step - config["warmup_steps"]) / config["ramp_steps"]))


def _take(rows, count, rng):
    return rows if len(rows) <= count else np.sort(rng.choice(rows, count, replace=False))


def query_rows(frozen, original, config, rng):
    """Labels select loss queries only; all neighborhood inputs remain label-blind."""
    slots, target = frozen.source.real_slots, frozen.anomaly_target[frozen.source.real_slots]
    normal = _take(np.flatnonzero(target == 0), config["normal_queries"], rng)
    anomaly = _take(np.flatnonzero(target == 1), config["anomaly_queries"], rng)
    detection = np.sort(np.r_[normal, anomaly])
    unchanged = (~frozen.inserted_mask & ~frozen.occluded_original_mask
                 & ~original.zero_slot_mask & (original.labels.semantic_target != 255))
    candidates = np.flatnonzero(unchanged)
    if not (np.array_equal(original.xyzi[candidates], frozen.source.xyzi[candidates])
            and np.array_equal(original.labels.packed[candidates], frozen.source.labels.packed[candidates])
            and np.all(frozen.anomaly_target[candidates] == 0)):
        raise ValueError("retained-normal pairing changed the original physical return or label")
    # Near changed returns, including an opaque occlusion without an inserted return.
    changed = np.concatenate((frozen.source.xyzi[frozen.inserted_mask, :3],
                             original.xyzi[frozen.occluded_original_mask, :3]))
    near = np.zeros(len(candidates), bool)
    if len(changed) and len(candidates):
        distance = cKDTree(changed).query(original.xyzi[candidates, :3], workers=1)[0]
        near = distance <= config["keep_near_m"]
    count = min(config["keep_queries"], len(candidates))
    nnear = min(int(round(count * config["keep_near_fraction"])), int(near.sum()))
    nfar = min(count - nnear, int((~near).sum()))
    nnear = min(count - nfar, int(near.sum()))
    kept = np.sort(np.r_[_take(candidates[near], nnear, rng), _take(candidates[~near], nfar, rng)])
    kept_rows = np.searchsorted(slots, kept)
    union = np.union1d(detection, kept_rows)
    return dict(query=torch.from_numpy(union),
        detection_index=torch.from_numpy(np.searchsorted(union, detection)),
        target=torch.from_numpy(target[detection].astype(np.int64)),
        keep_index=torch.from_numpy(np.searchsorted(union, kept_rows)),
        original_query=torch.from_numpy(np.searchsorted(original.real_slots, kept)),
        keep_slot=kept, near_pairs=int(np.isin(kept, candidates[near]).sum()))


class TrainingFrames:
    def __init__(self, config, data_root):
        self.config = config
        self.dataset = FrozenDataset(PROJECT_ROOT / "results/synthetic", data_root, "train")
        self.transform = None
        self.originals = OrderedDict()

    def __getitem__(self, request):
        index, draw, need_original = request
        if self.transform is None:
            torch.set_num_threads(1)
            self.transform = ScanTransform(self.config, workers=1)
        path, identity, frame = self.dataset.samples[index]
        original = self.dataset.sequence[frame]
        frozen = FrozenFrame.load(path, original, identity)
        rng = np.random.default_rng(np.random.SeedSequence([self.config["training"]["seed"], 11, draw]))
        queries = query_rows(frozen, original, self.config["training"], rng)
        before = None
        if need_original and len(queries["original_query"]):
            key = source_identity(original)
            if key not in self.originals:
                self.originals[key] = self.transform(original)
                if len(self.originals) > 2:
                    self.originals.popitem(last=False)
            self.originals.move_to_end(key)
            before = self.originals[key]
        unchanged_scan = not (frozen.inserted_mask | frozen.occluded_original_mask).any()
        scan = before if before is not None and unchanged_scan else self.transform(frozen.source)
        return dict(scan=scan, original=before, **queries,
            frame=frame, world=identity, draw=draw, sample=index)


class Requests:
    def __init__(self, probabilities, config, steps, start=0):
        self.cdf = np.cumsum(probabilities)
        self.cdf[-1] = 1.
        self.config, self.steps, self.start = config, steps, start

    def __iter__(self):
        t, loss = self.config["training"], self.config["loss"]
        for step in range(self.start, self.steps):
            need = loss["keep_weight"] > 0 and auxiliary_fraction(step, t) > 0
            for offset in range(t["batch_frames"]):
                draw = step * t["batch_frames"] + offset
                rng = np.random.default_rng(np.random.SeedSequence([t["seed"], 7, draw]))
                yield int(np.searchsorted(self.cdf, rng.random(), side="right")), draw, need

    def __len__(self):
        return (self.steps - self.start) * self.config["training"]["batch_frames"]


def _collate(rows):
    return rows


@contextmanager
def _preserve_buffers(model):
    # Recomputed BatchNorm must not update its running statistics a second time.
    buffers = [(value, value.clone()) for value in model.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for value, saved in buffers:
                value.copy_(saved)


def training_forward(model, scan, query):
    return checkpoint(model, scan, query, use_reentrant=False,
                      context_fn=lambda: (nullcontext(), _preserve_buffers(model)))


def batch_loss(model, rows, config, step, *, full_objective=False):
    device = next(model.parameters()).device
    all_scores, all_targets, all_frames, before, after = [], [], [], [], []
    for row in rows:
        scan = to_device(row["scan"], device)
        scores = training_forward(model, scan, row["query"].to(device))
        all_scores.append(scores[row["detection_index"].to(device)])
        all_targets.append(row["target"].to(device))
        all_frames.append(torch.full_like(all_targets[-1], row["frame"]))
        if row["original"] is not None:
            before.append(training_forward(model, to_device(row["original"], device), row["original_query"].to(device)))
            after.append(scores[row["keep_index"].to(device)])
    scores, target, frames = map(torch.cat, (all_scores, all_targets, all_frames))
    det = detection_loss(scores, target)
    keep = keep_loss(torch.cat(before), torch.cat(after), config["loss"]["keep_mode"]) if before else scores.sum() * 0
    generator = torch.Generator(device=device).manual_seed(config["training"]["seed"] + 31 + step)
    tail, tail_stats = tail_loss(scores, target, frames, config["loss"], generator)
    fraction = 1. if full_objective else auxiliary_fraction(step, config["training"])
    total = det + fraction * (config["loss"]["keep_weight"] * keep + config["loss"]["tail_weight"] * tail)
    stats = dict(total=float(total.detach()), detection=float(det.detach()), keep=float(keep.detach()),
        tail=float(tail.detach()), auxiliary_fraction=fraction, normal_queries=int((target == 0).sum()),
        anomaly_queries=int((target == 1).sum()), retained_normal_pairs=sum(len(x) for x in before),
        **tail_stats)
    return total, stats


def load_checkpoint(path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "ajae-v1-checkpoint":
        raise ValueError("checkpoint is not formal AJAE V1")
    model = AJAE(saved["config"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved


def check(config, data_root, examples):
    """Full real scans, task gradients and identity checks; no optimizer is created."""
    torch.manual_seed(config["training"]["seed"])
    dataset = TrainingFrames(config, data_root)
    rows = [dataset[(index, draw, True)] for draw, index in enumerate(examples)]
    model = AJAE(config).cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss, stats = batch_loss(model, rows, config, 0, full_objective=True)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite real-scan task loss")
    loss.backward()
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    if any(not torch.isfinite(g).all() for g in gradients.values()):
        raise FloatingPointError("nonfinite real-scan task gradient")
    stats.update(parameters=sum(p.numel() for p in model.parameters()),
        parameters_with_gradient=sum(p.numel() for p in model.parameters() if p.grad is not None),
        branch_gradient_norms={name: float(torch.linalg.vector_norm(torch.stack([
            g.float().norm() for key, g in gradients.items() if key.startswith(name + ".")])))
            for name in ("backbone", "point", "relations", "base_head", "relation_head")
            if any(key.startswith(name + ".") for key in gradients)},
        full_input_returns=[len(row["scan"]["xyzi"]) for row in rows],
        source_frames=[row["frame"] for row in rows], examples=examples,
        optimizer_steps=0, checkpoint_saved=False)
    torch.cuda.synchronize()
    stats.update(forward_backward_seconds=time.perf_counter() - started,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated())
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        full = model(to_device(rows[0]["scan"], "cuda"))
        subset = model(to_device(rows[0]["scan"], "cuda"), rows[0]["query"].cuda())
        if not torch.isfinite(full).all():
            raise FloatingPointError("full-scan prediction contains nonfinite values")
        error = float((full[rows[0]["query"].cuda()] - subset).abs().max())
        if not torch.allclose(full[rows[0]["query"].cuda()], subset, rtol=2e-5, atol=2e-5):
            raise ValueError("loss-query restriction changes full-scan scores")
        stats.update(predicted_real_returns=len(full), subset_max_absolute_error=error)
    from .protocol import load_protocol
    from .scene import STUSequence
    validation = FrozenDataset(PROJECT_ROOT / "results/synthetic", data_root, "validation")[161]
    real = STUSequence.open(data_root, protocol=load_protocol(), partition="val",
                           sequence_id=125, label_mode="forbidden")[25]
    transform = ScanTransform(config, workers=8)
    stats["evaluation_inputs"] = []
    for scope, source in (("synthetic_201_161", validation.source), ("unlabelled_val_125_25", real)):
        started = time.perf_counter()
        prediction = model.predict(source, transform)
        stats["evaluation_inputs"].append(dict(scope=scope, input_returns=source.real_count,
            output_scores=len(prediction.anomaly_score), all_finite=bool(np.isfinite(prediction.anomaly_score).all()),
            seconds=time.perf_counter() - started))
    print(json.dumps(stats, indent=2, allow_nan=False), flush=True)
    return stats


def fit(config, data_root, steps, output, resume=None):
    if steps < 1:
        raise ValueError("training needs a positive explicit update budget")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("training output is occupied; resume into an empty output directory")
    dataset = TrainingFrames(config, data_root)
    probabilities = dataset.dataset.sampling_probabilities(PROJECT_ROOT / config["training"]["sampling"])
    torch.manual_seed(config["training"]["seed"])
    model = AJAE(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    identities = [[identity, frame] for _, identity, frame in dataset.dataset.samples]
    start = 0
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (saved.get("format") != "ajae-v1-checkpoint" or saved["config"] != config
                or saved["samples"] != identities or not torch.equal(saved["probabilities"], torch.from_numpy(probabilities))):
            raise ValueError("resume configuration, input order or frame probabilities changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start = saved["step"]
        if start >= steps:
            raise ValueError("explicit budget has no remaining updates")
    # Two atomic checkpoint copies plus optimizer states and bounded logs are reserved.
    disk = host_disk()
    peak = (2 * sum(p.numel() for p in model.parameters()) * 16 + 256_000_000
            + (steps - start) * (512 + 20 * config["training"]["batch_frames"]))
    if peak >= disk["SizeRemaining"] - disk["reserve_bytes"]:
        raise OSError("training checkpoint peak would invade the E: reserve")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "config.json", config)
    workers = config["training"]["workers"]
    loader = DataLoader(dataset, sampler=Requests(probabilities, config, steps, start),
        batch_size=config["training"]["batch_frames"], num_workers=workers,
        collate_fn=_collate, pin_memory=True,
        **(dict(multiprocessing_context="spawn", prefetch_factor=1) if workers else {}))
    model.train()
    with (output / "loss.jsonl").open("a") as log:
        for step, rows in enumerate(loader, start):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, stats = batch_loss(model, rows, config, step)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite task loss at update {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            stats.update(step=step + 1, gradient_norm=float(norm), seconds=time.perf_counter() - started,
                         samples=[r["sample"] for r in rows])
            log.write(json.dumps(stats, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(stats), flush=True)
            if (step + 1) % 100 == 0:
                host_disk()
            if (step + 1) % config["training"]["save_every"] == 0 or step + 1 == steps:
                saved = dict(format="ajae-v1-checkpoint", config=config, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), step=step + 1, samples=identities,
                    probabilities=torch.from_numpy(probabilities), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all())
                temporary = output / "model.tmp"
                try:
                    torch.save(saved, temporary)
                    temporary.replace(output / "model.pt")
                finally:
                    temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "fit"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "protocol/model.json")
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--sample", type=int, action="append", help="fixed training manifest index for check")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        parser.error("the LitePT sparse-convolution implementation requires CUDA")
    if args.command == "check":
        if args.steps is not None or args.output is not None or args.resume is not None:
            parser.error("check has no optimization budget, output directory or resume state")
        check(config, args.data_root, args.sample if args.sample is not None else [161, 162])
    else:
        if args.steps is None or args.output is None or args.sample is not None:
            parser.error("fit requires --steps and --output; fixed check samples are not training input")
        fit(config, args.data_root, args.steps, args.output, args.resume)


if __name__ == "__main__":
    main()
