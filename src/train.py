"""Task losses and explicit-budget AJAE V1 execution; check never updates weights."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from .data import FrozenDataset, FrozenFrame, _atomic_json, host_disk, source_identity
from .model import AJAE, ScanTransform, load_config, to_device, validate_config
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


def tail_loss(scores, target, frames, config, generator, *, selections=None):
    """Live-batch tail approximation; no stale score queue or metric-unbiased claim."""
    normal, anomaly = torch.where(target == 0)[0], torch.where(target == 1)[0]
    if not len(normal) or not len(anomaly):
        if selections is not None:
            selections.update(high_normal=normal[:0], low_anomaly=anomaly[:0], pairs=[])
        return scores.sum() * 0, dict(pairs=0, cross_frame_pairs=0)
    high = normal[torch.argsort(scores[normal], descending=True, stable=True)
                  [:max(1, math.ceil(len(normal) * config["normal_tail_fraction"]))]]
    low = anomaly[torch.argsort(scores[anomaly], stable=True)
                  [:max(1, math.ceil(len(anomaly) * config["anomaly_tail_fraction"]))]]
    losses, cross = [], 0
    if selections is not None:
        selections.update(high_normal=high, low_anomaly=low, pairs=[])
    count = config["pairs_per_tail"]
    for apool, npool in ((anomaly, high), (low, normal)):
        a = apool[torch.randint(len(apool), (count,), generator=generator, device=scores.device)]
        n = npool[torch.randint(len(npool), (count,), generator=generator, device=scores.device)]
        if selections is not None:
            selections["pairs"].append((a, n))
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
    def __init__(self, config, data_root, *, preprocessing=None, cache_bytes=0):
        self.config = config
        self.dataset = FrozenDataset(PROJECT_ROOT / "results/synthetic", data_root, "train")
        self.preprocessing = ScanTransform(config, state=preprocessing).state_dict()
        self.transform = None
        self.originals = OrderedDict()
        self.cache, self.cache_bytes, self.cached_bytes = OrderedDict(), cache_bytes, 0

    def scan(self, source, key):
        # The cache belongs to one frozen dataset, config and calibration instance.
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        result = self.transform(source)
        size = sum(t.numel() * t.element_size() for t in result.values())
        if size <= self.cache_bytes:
            while self.cached_bytes + size > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cached_bytes -= sum(t.numel() * t.element_size() for t in old.values())
            self.cache[key] = result
            self.cached_bytes += size
        return result

    def queries(self, index, draw):
        path, identity, frame = self.dataset.samples[index]
        original = self.dataset.sequence[frame]
        frozen = FrozenFrame.load(path, original, identity)
        rng = np.random.default_rng(np.random.SeedSequence([self.config["training"]["seed"], 11, draw]))
        return frozen, original, query_rows(frozen, original, self.config["training"], rng)

    def __getitem__(self, request):
        index, draw, need_original = request
        if self.transform is None:
            torch.set_num_threads(1)
            self.transform = ScanTransform(self.config, state=self.preprocessing, workers=1)
        frozen, original, queries = self.queries(index, draw)
        identity, frame = frozen.world_identity, frozen.source.frame_id
        before = None
        if need_original and len(queries["original_query"]):
            key = source_identity(original)
            if key not in self.originals:
                self.originals[key] = self.scan(original, ("original", key))
                if len(self.originals) > 2:
                    self.originals.popitem(last=False)
            self.originals.move_to_end(key)
            before = self.originals[key]
        unchanged_scan = not (frozen.inserted_mask | frozen.occluded_original_mask).any()
        scan = before if before is not None and unchanged_scan else self.scan(frozen.source, (identity, frame))
        return dict(scan=scan, original=before, **queries,
            frame=frame, world=identity, draw=draw, sample=index)


class Requests:
    def __init__(self, probabilities, config, steps, start=0, *, samples=None):
        self.samples = None if samples is None else np.asarray(samples, dtype=np.int64)
        if self.samples is not None and (not len(self.samples) or len(set(self.samples)) != len(self.samples)
                or len(self.samples) % config["training"]["batch_frames"]):
            raise ValueError("fixed sampling needs unique complete batches")
        self.cdf = None if samples is not None else np.cumsum(probabilities)
        if self.cdf is not None:
            self.cdf[-1] = 1.
        self.config, self.steps, self.start = config, steps, start

    def __iter__(self):
        t, loss = self.config["training"], self.config["loss"]
        epoch, order = None, None
        for step in range(self.start, self.steps):
            need = loss["keep_weight"] > 0 and auxiliary_fraction(step, t) > 0
            if self.samples is not None:
                current = step * t["batch_frames"] // len(self.samples)
                if current != epoch:
                    epoch = current
                    rng = np.random.default_rng(np.random.SeedSequence([t["seed"], 79, epoch]))
                    order = rng.permutation(self.samples)
            for offset in range(t["batch_frames"]):
                draw = step * t["batch_frames"] + offset
                if order is None:
                    rng = np.random.default_rng(np.random.SeedSequence([t["seed"], 7, draw]))
                    sample = int(np.searchsorted(self.cdf, rng.random(), side="right"))
                else:
                    sample = int(order[draw % len(order)])
                yield sample, draw, need

    def __len__(self):
        return (self.steps - self.start) * self.config["training"]["batch_frames"]


def _collate(rows):
    return rows


def select_samples(dataset, records):
    lookup = {(world, frame): i for i, (_, world, frame) in enumerate(dataset.samples)}
    keys = [(r["identity"], r["frame"]) for r in records]
    if len(set(keys)) != len(keys) or any(k not in lookup for k in keys):
        raise ValueError("fixed sample identities must uniquely belong to this frozen split")
    return [lookup[k] for k in keys]


class _preserve_buffers:
    """Reusable across separate loss gradients through the same checkpoint graph."""
    def __init__(self, model):
        self.model, self.stack = model, []

    def __enter__(self):
        self.stack.append([(value, value.clone()) for value in self.model.buffers()])
        return self

    def __exit__(self, *_):
        # Recomputed BatchNorm must not count the same observation again.
        with torch.no_grad():
            for value, saved in self.stack.pop():
                value.copy_(saved)


def training_forward(model, scan, query, *, return_features=False):
    return checkpoint(model, scan, query, use_reentrant=False,
                      **(dict(return_features=True) if return_features else {}),
                      context_fn=lambda: (nullcontext(), _preserve_buffers(model)))


@contextmanager
def evaluation_state(model):
    mode = model.training
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy, python = np.random.get_state(), random.getstate()
    try:
        with _preserve_buffers(model), torch.no_grad():
            model.eval()
            yield
    finally:
        model.train(mode)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(numpy)
        random.setstate(python)


def batch_loss(model, rows, config, step, *, full_objective=False, details=False):
    device = next(model.parameters()).device
    all_scores, all_targets, all_frames, before, after = [], [], [], [], []
    observed = dict(scores=[], context=[], point=[], score_targets=[])
    def forward(scan, query, target):
        output = training_forward(model, to_device(scan, device), query.to(device), return_features=details)
        if not details:
            return output
        for name in ("context", "point"):
            observed[name].append(output[name])
        observed["scores"].append(output["score"])
        observed["score_targets"].append(target.to(device))
        return output["score"]
    for row in rows:
        query_target = torch.zeros(len(row["query"]), dtype=torch.long)
        query_target[row["detection_index"]] = row["target"]
        scores = forward(row["scan"], row["query"], query_target)
        all_scores.append(scores[row["detection_index"].to(device)])
        all_targets.append(row["target"].to(device))
        all_frames.append(torch.full_like(all_targets[-1], row["frame"]))
        if row["original"] is not None:
            before.append(forward(row["original"], row["original_query"], torch.zeros(len(row["original_query"]), dtype=torch.long)))
            after.append(scores[row["keep_index"].to(device)])
    scores, target, frames = map(torch.cat, (all_scores, all_targets, all_frames))
    det = detection_loss(scores, target)
    keep = keep_loss(torch.cat(before), torch.cat(after), config["loss"]["keep_mode"]) if before else scores.sum() * 0
    generator = torch.Generator(device=device).manual_seed(config["training"]["seed"] + 31 + step)
    selections = {} if details else None
    if config["loss"]["tail_weight"] > 0 or details:
        tail, tail_stats = tail_loss(scores, target, frames, config["loss"], generator, selections=selections)
    else:
        tail, tail_stats = scores.sum() * 0, dict(pairs=0, cross_frame_pairs=0)
    fraction = 1. if full_objective else auxiliary_fraction(step, config["training"])
    total = det + fraction * (config["loss"]["keep_weight"] * keep + config["loss"]["tail_weight"] * tail)
    stats = dict(total=float(total.detach()), detection=float(det.detach()), keep=float(keep.detach()),
        tail=float(tail.detach()), auxiliary_fraction=fraction, normal_queries=int((target == 0).sum()),
        anomaly_queries=int((target == 1).sum()), retained_normal_pairs=sum(len(x) for x in before),
        **tail_stats)
    if details:
        mean = keep_loss(torch.cat(before), torch.cat(after), "mean") if before else scores.sum() * 0
        observed.update(components=dict(detection=det, keep=keep, tail=tail, keep_mean=mean),
                        tail=selections, target=target)
        return total, stats, observed
    return total, stats


def load_checkpoint(path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "ajae-v1-checkpoint" or "preprocessing" not in saved:
        raise ValueError("checkpoint needs formal AJAE V1 weights and saved preprocessing")
    ScanTransform(saved["config"], state=saved["preprocessing"])
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
    transform = ScanTransform(config, state=dataset.preprocessing, workers=8)
    stats["evaluation_inputs"] = []
    for scope, source in (("synthetic_201_161", validation.source), ("unlabelled_val_125_25", real)):
        started = time.perf_counter()
        prediction = model.predict(source, transform)
        stats["evaluation_inputs"].append(dict(scope=scope, input_returns=source.real_count,
            output_scores=len(prediction.anomaly_score), all_finite=bool(np.isfinite(prediction.anomaly_score).all()),
            seconds=time.perf_counter() - started))
    print(json.dumps(stats, indent=2, allow_nan=False), flush=True)
    return stats


def _gradient_comparison(vectors):
    norms = {name: float(value.double().norm()) for name, value in vectors.items()}
    cosine = {}
    for i, left in enumerate(vectors):
        for right in list(vectors)[i + 1:]:
            denominator = norms[left] * norms[right]
            cosine[left + ":" + right] = (float(torch.dot(vectors[left].double(), vectors[right].double()))
                                           / denominator if denominator else None)
    return dict(norm=norms, cosine=cosine)


def gradient_preview(config, data_root, exposure):
    """Fixed content-selected batches at one random initialization; never update parameters."""
    groups = ("near_sparse_1_4", "far_at_least_20", "weak_background_visible",
              "normal_sparse_with_nonextreme_anomaly")
    full_step = config["training"]["warmup_steps"] + config["training"]["ramp_steps"]
    chosen = {name: next((r["step"] for r in exposure["records"]
                         if r["step"] >= full_step and r["flags"][name] is True), None) for name in groups}
    torch.manual_seed(config["training"]["seed"])
    model, dataset = AJAE(config).cuda().train(), TrainingFrames(config, data_root)
    parameters = list(model.named_parameters())
    result = dict(selection_rule="first request step after full ramp containing each declared group; deduplicate steps",
                  selected_steps=chosen, optimizer_steps=0, checkpoint_saved=False,
                  scope="one unchanged random initialization, train-mode batch statistics restored between batches; not learning or transfer evidence",
                  batches=[])
    for step in sorted(set(chosen.values()) - {None}):
        records = [r for r in exposure["records"] if r["step"] == step]
        rows = [dataset[(r["sample"], r["draw"], r["keep_active"])] for r in records]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with _preserve_buffers(model):
            _, stats, observed = batch_loss(model, rows, config, step, details=True)
            tensors = [p for _, p in parameters]
            sizes = {name: len(observed[name]) for name in ("scores", "context", "point")}
            for name in sizes:
                tensors.extend(observed[name])
            vectors = {name: {} for name in ("shared_parameters", "heads", *sizes)}
            score_direction = {}
            for component, loss in observed["components"].items():
                gradients = torch.autograd.grad(loss, tensors, allow_unused=True, retain_graph=True)
                if any(g is not None and not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError(f"nonfinite {component} diagnostic gradient")
                def vector(pairs):
                    return torch.cat([(g.detach().float().cpu() if g is not None else torch.zeros_like(t, device="cpu"))
                                      .reshape(-1) for g, t in pairs])
                for scope, is_head in (("shared_parameters", False), ("heads", True)):
                    vectors[scope][component] = vector([(g, p) for (name, p), g in zip(parameters, gradients)
                        if (name.startswith(("base_head.", "relation_head."))) == is_head])
                offset = len(parameters)
                for scope, count in sizes.items():
                    vectors[scope][component] = vector(zip(gradients[offset:offset + count], tensors[offset:offset + count]))
                    offset += count
                labels, force = torch.cat(observed["score_targets"]).cpu(), vectors["scores"][component]
                score_direction[component] = {name: dict(count=int((labels == value).sum()),
                    norm=float(force[labels == value].double().norm()), signed_sum=float(force[labels == value].double().sum()))
                    for name, value in (("normal", 0), ("anomaly", 1))}
                del gradients
            comparison = {name: _gradient_comparison(values) for name, values in vectors.items()}
            # The mean control shares these exact forward passes, including BatchNorm observations.
            weights = dict(detection=1., keep=config["loss"]["keep_weight"], tail=config["loss"]["tail_weight"],
                           keep_mean=config["loss"]["keep_weight"])
            for value in comparison.values():
                value["weighted_norm"] = {name: norm * weights[name] for name, norm in value["norm"].items()}
            high = observed["tail"]["high_normal"].detach().cpu().numpy()
            low = observed["tail"]["low_anomaly"].detach().cpu().numpy()
            selected_pairs = observed["tail"]["pairs"]
            sampled_a, sampled_n = [np.concatenate([pair[k].detach().cpu().numpy() for pair in selected_pairs])
                                    if selected_pairs else np.empty(0, np.int64) for k in (0, 1)]
            tails, offset = [], 0
            for row, record in zip(rows, records):
                stop = offset + len(row["target"])
                sides = record["sparse_detection_index"]
                witness = None if sides is None else dict(
                    high_normal=int(np.isin(high, np.asarray(sides["normal"], dtype=np.int64) + offset).sum()),
                    low_anomaly=int(np.isin(low, np.asarray(sides["anomaly"], dtype=np.int64) + offset).sum()),
                    sampled_normal=int(np.isin(sampled_n, np.asarray(sides["normal"], dtype=np.int64) + offset).sum()),
                    sampled_anomaly=int(np.isin(sampled_a, np.asarray(sides["anomaly"], dtype=np.int64) + offset).sum()))
                tails.append(dict(sample=record["sample"], draw=record["draw"], flags=record["flags"],
                    normal_queries=int((row["target"] == 0).sum()), anomaly_queries=int((row["target"] == 1).sum()),
                    high_normal=int(((high >= offset) & (high < stop)).sum()),
                    low_anomaly=int(((low >= offset) & (low < stop)).sum()), sparse_witness=witness,
                    sampled_normal=int(((sampled_n >= offset) & (sampled_n < stop)).sum()),
                    sampled_anomaly=int(((sampled_a >= offset) & (sampled_a < stop)).sum())))
                offset = stop
            torch.cuda.synchronize()
            report = dict(step=step, samples=[r["sample"] for r in records],
                losses={k: float(v.detach()) for k, v in observed["components"].items()},
                gradients=comparison, score_gradient_direction=score_direction,
                input_returns=[len(r["scan"]["xyzi"]) for r in rows], tails=tails, **stats,
                seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
            result["batches"].append(report)
            print(json.dumps(dict(event="gradient_preview", step=step, losses=report["losses"], seconds=report["seconds"])), flush=True)
            del tensors, observed, vectors, loss, report
        del rows
    return result


def preview(config, data_root, steps, output):
    from .exposure import preview as exposure_preview
    if steps < 1:
        raise ValueError("preview needs a positive logical sampling budget; it performs no optimization")
    exposure = exposure_preview(config, data_root, steps=steps, workers=8)
    result = dict(format="ajae-training-preview", config=config, exposure=exposure,
                  gradients=gradient_preview(config, data_root, exposure), optimizer_steps=0)
    _atomic_json(output, result)
    return result


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fit(config, data_root, steps, output, resume=None, *, micro=None):
    if steps < 1:
        raise ValueError("training needs a positive explicit update budget")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("training output is occupied; resume into an empty output directory")
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume is not None else None
    if saved is not None and (saved.get("format") != "ajae-v1-checkpoint" or "preprocessing" not in saved):
        raise ValueError("resume requires the saved inference preprocessing state")
    if micro is not None and not 1 <= steps <= micro["maximum_updates"]:
        raise ValueError("micro learning cannot exceed its declared update budget")
    dataset = TrainingFrames(config, data_root, preprocessing=saved["preprocessing"] if saved is not None else None,
                             cache_bytes=2 * 2**30 if micro else 0)
    selected = select_samples(dataset.dataset, micro["selection"]["train"]) if micro else None
    probabilities = (None if micro else
        dataset.dataset.sampling_probabilities(PROJECT_ROOT / config["training"]["sampling"]))
    torch.manual_seed(config["training"]["seed"])
    model = AJAE(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    identities = [[identity, frame] for _, identity, frame in dataset.dataset.samples]
    start = 0
    if resume is not None:
        same_probabilities = (saved["probabilities"] is None if probabilities is None else
                              torch.equal(saved["probabilities"], torch.from_numpy(probabilities)))
        if (saved["config"] != config or saved.get("micro") != micro
                or saved["samples"] != identities or not same_probabilities):
            raise ValueError("resume configuration, input order or frame probabilities changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start = saved["step"]
        if start >= steps:
            raise ValueError("explicit budget has no remaining updates")
    # Include retained evaluation states, an emergency state and atomic output overlap.
    disk = host_disk()
    peak = ((7 if micro else 2) * sum(p.numel() for p in model.parameters()) * 20 + 512_000_000
            + (steps - start) * (512 + 20 * config["training"]["batch_frames"]))
    if peak >= disk["SizeRemaining"] - disk["reserve_bytes"]:
        raise OSError("training checkpoint peak would invade the E: reserve")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "config.json", config)
    run = dict(status="running", requested_updates=steps, start_update=start,
        started_unix=time.time(), host_E_before=disk, estimated_peak_new_bytes=peak,
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
            cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
            preparation_workers=config["training"]["workers"], transform_cache_bytes_per_worker=dataset.cache_bytes))
    _atomic_json(output / "run.json", run)
    torch.cuda.reset_peak_memory_stats()
    prepared = None
    if micro:
        from .evaluate import prepare_fixed, evaluate_fixed
        _atomic_json(output / "selection.json", micro)
        prepared = prepare_fixed(data_root, micro["selection"])
        transform = ScanTransform(config, state=dataset.preprocessing, workers=8)

    def snapshot(step, failure=None):
        payload = dict(format="ajae-v1-checkpoint", config=config, model=model.state_dict(),
            preprocessing=dataset.preprocessing, optimizer=optimizer.state_dict(), step=step, samples=identities,
            probabilities=None if probabilities is None else torch.from_numpy(probabilities),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), micro=micro)
        if failure is not None:
            payload.update(failure=str(failure), gradients={name: p.grad for name, p in model.named_parameters() if p.grad is not None})
        save_checkpoint(output / ("failure.pt" if failure is not None else f"{step}.pt" if micro else "model.pt"), payload)

    def evaluate(step):
        with evaluation_state(model):
            result = evaluate_fixed(model, transform, prepared,
                include_real=step in micro["evaluation"]["real_steps"], directory=output)
        _atomic_json(output / f"{step}.json", dict(step=step, checkpoint=f"{step}.pt", **result))

    workers = config["training"]["workers"]
    loader = DataLoader(dataset, sampler=Requests(probabilities, config, steps, start, samples=selected),
        batch_size=config["training"]["batch_frames"], num_workers=workers,
        collate_fn=_collate, pin_memory=True, persistent_workers=bool(workers),
        generator=torch.Generator().manual_seed(config["training"]["seed"] + 83) if micro else None,
        **(dict(multiprocessing_context="spawn", prefetch_factor=1) if workers else {}))
    model.train()
    completed, rows = start, []
    try:
        if micro:
            snapshot(start)
            evaluate(start)
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
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError(f"nonfinite parameter after update {step + 1}")
                completed = step + 1
                stats.update(step=completed, gradient_norm=float(norm), seconds=time.perf_counter() - started,
                    samples=[r["sample"] for r in rows], draws=[r["draw"] for r in rows],
                    full_input_returns=[len(r["scan"]["xyzi"]) for r in rows])
                if prepared is not None:
                    hits = dict(normal=0, anomaly=0, active_keep=0)
                    for row in rows:
                        measured = prepared["geometry"].get((row["world"], row["frame"]))
                        if measured is None:
                            continue
                        contrast = measured["contrasts"]["sparse"]
                        slots = row["scan"]["source_slot"][row["query"][row["detection_index"]]].numpy()
                        for name, label, field in (("normal", 0, "normal_source_slots"), ("anomaly", 1, "central_anomaly_slots")):
                            hits[name] += int(np.isin(slots[row["target"].numpy() == label], contrast[field]).sum())
                        if row["original"] is not None:
                            hits["active_keep"] += int(np.isin(row["keep_slot"], contrast["normal_source_slots"]).sum())
                    stats["known_sparse_witness_queries"] = hits
                log.write(json.dumps(stats, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(stats), flush=True)
                if completed % 32 == 0:
                    host_disk()
                if micro and completed in micro["evaluation"]["synthetic_steps"]:
                    snapshot(completed)
                    evaluate(completed)
                elif completed % config["training"]["save_every"] == 0 or completed == steps:
                    snapshot(completed)
    except BaseException as error:
        snapshot(completed, failure=error)
        _atomic_json(output / "failure.json", dict(completed_updates=completed, error=repr(error),
                     samples=[r["sample"] for r in rows], automatic_retry=False))
        run.update(status="stopped", completed_updates=completed, error=repr(error))
        raise
    else:
        run.update(status="completed", completed_updates=completed, host_E_after=host_disk())
    finally:
        run.update(seconds=time.time() - run["started_unix"],
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        _atomic_json(output / "run.json", run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "preview", "fit"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "protocol/model.json")
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--micro", type=Path, help="fixed micro-learning selection; formal defaults remain unchanged")
    parser.add_argument("--sample", type=int, action="append", help="fixed training manifest index for check")
    args = parser.parse_args()
    config = load_config(args.config)
    micro = json.loads(args.micro.read_text()) if args.micro is not None else None
    if micro is not None:
        if args.command != "fit" or micro.get("format") != "ajae-micro-learning":
            parser.error("--micro only supports a declared micro-learning fit")
        config = deepcopy(config)
        config["loss"].update(micro["loss_overrides"])
        config["scope"] = "Authorized micro task learning; at most 256 updates; no short training or external weights."
        validate_config(config)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        parser.error("the LitePT sparse-convolution implementation requires CUDA")
    if args.command == "check":
        if args.steps is not None or args.output is not None or args.resume is not None:
            parser.error("check has no optimization budget, output directory or resume state")
        check(config, args.data_root, args.sample if args.sample is not None else [161, 162])
    elif args.command == "preview":
        if args.steps is None or args.output is None or args.sample is not None or args.resume is not None:
            parser.error("preview requires --steps and --output; it has no fixed check samples or resume state")
        preview(config, args.data_root, args.steps, args.output)
    else:
        if args.steps is None or args.output is None or args.sample is not None:
            parser.error("fit requires --steps and --output; fixed check samples are not training input")
        fit(config, args.data_root, args.steps, args.output, args.resume, micro=micro)


if __name__ == "__main__":
    main()
