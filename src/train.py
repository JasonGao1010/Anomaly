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
import signal
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from .data import (ConditionIndex, FrozenDataset, FrozenFrame, _atomic_json, binary_target, binary_normal_groups, host_disk,
                   normal_conditions, retained_normal_slots, source_identity, runtime_resources)
from .model import AJAE, ScanTransform, inherit_backbone, transfer_parent, load_config, to_device, validate_config
from .protocol import PROJECT_ROOT


def detection_loss(scores, target, *, details=False):
    """Each present class retains its declared half weight, even in zero-anomaly batches."""
    zero = scores.sum() * 0
    normal, anomaly = scores[target == 0], scores[target == 1]
    normal_term = F.softplus(normal).mean() if len(normal) else zero
    anomaly_term = F.softplus(-anomaly).mean() if len(anomaly) else zero
    loss = .5 * (normal_term + anomaly_term)
    if details:
        return loss, dict(normal_loss=float(normal_term.detach()) if len(normal) else None,
                          anomaly_loss=float(anomaly_term.detach()) if len(anomaly) else None)
    return loss


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
    if mode == "increase":
        # Keep baseline supervision; detach only the hinge reference, not the baseline term.
        # A half-scaled maximum has the same value but the wrong baseline gradient.
        return .5 * (before + F.relu(after - before.detach())).mean()
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
    if "group_queries" in config:
        return 1.
    return min(1., max(0., (step - config["warmup_steps"]) / config["ramp_steps"]))


def _take(rows, count, rng):
    return rows if len(rows) <= count else np.sort(rng.choice(rows, count, replace=False))


def grouped_queries(base, sparse, changed, budgets, weights, rng):
    """Independent uniform group draws; spare seats and risk coefficients have separate fallbacks."""
    selected = [_take(pool, budget, rng) for pool, budget in zip((sparse, changed), budgets[1:], strict=True)]
    spare = sum(budgets[1:]) - sum(map(len, selected))
    selected.insert(0, _take(base, budgets[0] + spare, rng))
    mass = np.asarray(weights, np.float64).copy()
    for i in (1, 2):
        if not len(selected[i]):
            mass[0] += mass[i]
            mass[i] = 0.
    if not len(base):
        mass[:] = 0.
    return selected, mass.tolist()


def normal_query_weights(pools, budgets, weights, rng):
    """Balance-heuristic correction targets all normals despite overlapping proposal groups."""
    groups, mass = grouped_queries(*pools, budgets, weights, rng)
    slots = np.unique(np.concatenate(groups))
    correction = np.zeros(len(slots), np.float64)
    if len(pools[0]):
        density = sum(alpha * np.isin(slots, pool) / len(pool) for alpha, pool in zip(mass, pools, strict=True) if len(pool))
        ratio = 1 / (len(pools[0]) * density)
        if np.any(ratio > 2. + 1e-12):
            raise ValueError("normal query correction exceeds its bound")
        for group, alpha in zip(groups, mass, strict=True):
            if len(group):
                index = np.searchsorted(slots, group)
                correction[index] += alpha * ratio[index] / len(group)
    return slots, correction, dict(populations=list(map(len, pools)), queries=list(map(len, groups)), mixture=mass)


def binary_queries(frozen, original, config, rng, sparse_slots):
    if sparse_slots is None:
        raise ValueError("V3 requires source-bound native low-support positions")
    post_groups, raw_groups = binary_normal_groups(frozen, original, sparse_slots)
    budgets = config["group_queries"]
    post, post_weight, post_info = normal_query_weights(post_groups, budgets["normal"], budgets["weights"], rng)
    raw, raw_weight, raw_info = normal_query_weights(raw_groups, budgets["raw"], budgets["weights"], rng)
    target = binary_target(frozen.source, frozen.inserted_mask)
    positives = np.flatnonzero(target == 1)
    anomaly = _take(positives, config["anomaly_queries"], rng)
    slots = np.union1d(post, anomaly)
    weight = np.zeros(len(slots), np.float64)
    weight[np.searchsorted(slots, post)] = post_weight
    return dict(query=torch.from_numpy(np.searchsorted(frozen.source.real_slots, slots)),
        original_query=torch.from_numpy(np.searchsorted(original.real_slots, raw)),
        normal_weight=torch.from_numpy(weight), raw_weight=torch.from_numpy(raw_weight),
        anomaly_index=torch.from_numpy(np.searchsorted(slots, anomaly)),
        target=torch.from_numpy(target[slots].astype(np.int64)),
        population_counts=[len(positives), len(post_groups[0]), len(raw_groups[0])],
        query_groups=dict(post=post_info, raw=raw_info),
        normal_queries=len(post), raw_queries=len(raw), anomaly_queries=len(anomaly))


def query_rows(frozen, original, config, rng, *, sparse_slots=None):
    """Labels select loss queries only; all neighborhood inputs remain label-blind."""
    if config.get("binary_view") == "official_range_v3":
        return binary_queries(frozen, original, config, rng, sparse_slots)
    slots, target = frozen.source.real_slots, frozen.anomaly_target[frozen.source.real_slots]
    if "group_queries" in config:
        if sparse_slots is None:
            raise ValueError("V2 queries require the source-bound low-support index")
        kept, sparse, near = normal_conditions(frozen, original, sparse_slots, config["conditions"]["radius_m"])
        groups = config["group_queries"]
        normal_slots = slots[target == 0]
        normal_groups, normal_weights = grouped_queries(normal_slots, sparse, near,
            groups["normal"], groups["weights"], rng)
        keep_groups, keep_weights = grouped_queries(kept, sparse, near,
            groups["keep"], groups["weights"], rng)
        normal = np.searchsorted(slots, np.unique(np.concatenate(normal_groups)))
        kept_slots = np.unique(np.concatenate(keep_groups))
        anomaly = _take(np.flatnonzero(target == 1), config["anomaly_queries"], rng)
        detection = np.sort(np.r_[normal, anomaly])
        union = np.union1d(detection, np.searchsorted(slots, kept_slots))
        normal_indices = [torch.from_numpy(np.searchsorted(union, np.searchsorted(slots, g))) for g in normal_groups]
        keep_indices = [torch.from_numpy(np.searchsorted(kept_slots, g)) for g in keep_groups]
        return dict(query=torch.from_numpy(union),
            detection_index=torch.from_numpy(np.searchsorted(union, detection)),
            target=torch.from_numpy(target[detection].astype(np.int64)),
            keep_index=torch.from_numpy(np.searchsorted(union, np.searchsorted(slots, kept_slots))),
            original_query=torch.from_numpy(np.searchsorted(original.real_slots, kept_slots)),
            keep_slot=kept_slots, near_pairs=int(np.isin(kept_slots, near).sum()),
            normal_groups=normal_indices, normal_weights=normal_weights,
            keep_groups=keep_indices, keep_weights=keep_weights,
            condition_exposure=dict(
                population=dict(normal=len(normal_slots), sparse=len(sparse), changed=len(near), keep=len(kept)),
                normal_group_queries=list(map(len, normal_groups)), keep_group_queries=list(map(len, keep_groups)),
                normal_weights=normal_weights, keep_weights=keep_weights,
                normal_sparse_mask=np.isin(slots[union], sparse), normal_changed_mask=np.isin(slots[union], near),
                keep_sparse_mask=np.isin(kept_slots, sparse), keep_changed_mask=np.isin(kept_slots, near)))
    normal = _take(np.flatnonzero(target == 0), config["normal_queries"], rng)
    anomaly = _take(np.flatnonzero(target == 1), config["anomaly_queries"], rng)
    detection = np.sort(np.r_[normal, anomaly])
    candidates = retained_normal_slots(frozen, original)
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
        self.exposure_context = None
        self.conditions = (ConditionIndex(PROJECT_ROOT / config["training"]["sampling"], self.dataset,
            config["training"]["conditions"]) if "group_queries" in config["training"] else None)

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
        sparse = self.conditions.slots(original) if self.conditions is not None else None
        return frozen, original, query_rows(frozen, original, self.config["training"], rng, sparse_slots=sparse)

    def __getitem__(self, request):
        v3 = isinstance(request, dict)
        index, draw, need_original = (request["sample"], request["draw"], True) if v3 else request
        if self.transform is None:
            torch.set_num_threads(1)
            self.transform = ScanTransform(self.config, state=self.preprocessing, workers=1)
        frozen, original, queries = self.queries(index, draw)
        identity, frame = frozen.world_identity, frozen.source.frame_id
        if v3:
            if queries["population_counts"] != self.conditions.counts[index].tolist():
                raise ValueError("actual binary populations differ from the sampling denominators")
            yaw = float(np.random.default_rng(np.random.SeedSequence([self.config["training"]["seed"], 13, draw])).uniform(0, 2 * np.pi))
            return dict(scan=self.transform(frozen.source, yaw=yaw), original=self.transform(original, yaw=yaw),
                **queries, frame=frame, world=identity, draw=draw, sample=index, request=request, yaw=yaw)
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
        row = dict(scan=scan, original=before, **queries,
                   frame=frame, world=identity, draw=draw, sample=index)
        if self.exposure_context is not None:
            from .exposure import measure_queries
            row["exposure"] = measure_queries(request, frozen, original, queries, *self.exposure_context)
            row["exposure"].pop("sparse_detection_index")
        return row


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


def training_forward(model, scan, query, *, return_features=False, trace=None):
    return checkpoint(model, scan, query, use_reentrant=False,
                      **(dict(return_features=True) if return_features else {}),
                      **(dict(trace=trace) if trace is not None else {}),
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


def grouped_risk(values, groups, weights):
    """Overlapping identities add their group coefficients without another model forward."""
    return sum((weight * values[index.to(values.device)].mean()
                for index, weight in zip(groups, weights, strict=True) if len(index)), values.sum() * 0)


def population_loss(model, rows, trace=None):
    """One request uses its exact mixture probability; no random batch-size renormalization."""
    device = next(model.parameters()).device
    components, counts = [], np.zeros(3, np.int64)
    for row in rows:
        post = training_forward(model, to_device(row["scan"], device), row["query"].to(device), trace=trace)
        raw = training_forward(model, to_device(row["original"], device), row["original_query"].to(device), trace=trace)
        positive = post[row["anomaly_index"].to(device)]
        anomaly = F.softplus(-positive).double().mean() if len(positive) else post.sum().double() * 0
        normal = (F.softplus(post).double() * row["normal_weight"].to(device)).sum()
        original = (F.softplus(raw).double() * row["raw_weight"].to(device)).sum()
        terms = torch.stack((anomaly, normal, original)) * torch.tensor(row["request"]["coefficients"], device=device, dtype=torch.float64)
        components.append(terms)
        counts += [row["anomaly_queries"], row["normal_queries"], row["raw_queries"]]
    risks = torch.stack(components).mean(0)
    loss = risks.sum()
    return loss, dict(total=float(loss.detach()), detection=float(loss.detach()),
        risk_anomaly=float(risks[0].detach()), risk_post_normal=float(risks[1].detach()),
        risk_raw_normal=float(risks[2].detach()), anomaly_queries=int(counts[0]),
        normal_queries=int(counts[1]), raw_normal_queries=int(counts[2]))


def batch_loss(model, rows, config, step, *, full_objective=False, details=False, trace=None):
    if config.get("format") == "ajae-v3":
        return population_loss(model, rows, trace)
    device = next(model.parameters()).device
    all_scores, all_targets, all_frames, before, after = [], [], [], [], []
    grouped = "group_queries" in config["training"]
    point_anomaly = config["loss"].get("anomaly_reduction", "frame") == "point"
    normal_risks, anomaly_risks, keep_risks = [], [], []
    observed = dict(scores=[], context=[], point=[], score_targets=[])
    def forward(scan, query, target):
        output = training_forward(model, to_device(scan, device), query.to(device), return_features=details, trace=trace)
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
        if grouped:
            normal_risks.append(grouped_risk(F.softplus(scores), row["normal_groups"], row["normal_weights"]))
            positive = all_scores[-1][all_targets[-1] == 1]
            if len(positive):
                values = F.softplus(-positive)
                anomaly_risks.append(values if point_anomaly else values.mean())
            keep_risks.append(scores.sum() * 0)
        if row["original"] is not None:
            before.append(forward(row["original"], row["original_query"], torch.zeros(len(row["original_query"]), dtype=torch.long)))
            after.append(scores[row["keep_index"].to(device)])
            if grouped:
                keep_risks[-1] = grouped_risk(.5 * (F.softplus(before[-1]) + F.softplus(after[-1])),
                    row["keep_groups"], row["keep_weights"])
    scores, target, frames = map(torch.cat, (all_scores, all_targets, all_frames))
    if grouped:
        normal = torch.stack(normal_risks).mean()
        if not anomaly_risks:
            anomaly = scores.sum() * 0
        elif point_anomaly:
            # Every queried positive has equal weight across the complete batch.
            anomaly = torch.cat(anomaly_risks).mean()
        else:
            anomaly = torch.stack(anomaly_risks).mean()
        det, keep = .5 * (normal + anomaly), torch.stack(keep_risks).mean()
        class_losses = dict(normal_loss=float(normal.detach()),
                           anomaly_loss=float(anomaly.detach()) if anomaly_risks else None,
                           positive_frames=len(anomaly_risks))
    else:
        det, class_losses = detection_loss(scores, target, details=True)
        keep = keep_loss(torch.cat(before), torch.cat(after), config["loss"]["keep_mode"]) if before else scores.sum() * 0
    generator = torch.Generator(device=device).manual_seed(config["training"]["seed"] + 31 + step)
    selections = {} if details else None
    if config["loss"]["tail_weight"] > 0 or (details and not grouped):
        tail, tail_stats = tail_loss(scores, target, frames, config["loss"], generator, selections=selections)
    else:
        tail, tail_stats = scores.sum() * 0, dict(pairs=0, cross_frame_pairs=0)
    fraction = 1. if full_objective else auxiliary_fraction(step, config["training"])
    total = det + fraction * (config["loss"]["keep_weight"] * keep + config["loss"]["tail_weight"] * tail)
    stats = dict(total=float(total.detach()), detection=float(det.detach()), keep=float(keep.detach()),
        tail=float(tail.detach()), auxiliary_fraction=fraction, normal_queries=int((target == 0).sum()),
        anomaly_queries=int((target == 1).sum()), retained_normal_pairs=sum(len(x) for x in before),
        **tail_stats, **class_losses)
    if details:
        mean = keep if grouped else keep_loss(torch.cat(before), torch.cat(after), "mean") if before else scores.sum() * 0
        observed.update(components=dict(detection=det, keep=keep, tail=tail, keep_mean=mean),
                        tail=selections, target=target)
        return total, stats, observed
    return total, stats


def accumulate_batches(model, batches, config, step):
    """Backpropagate each physical microbatch; the caller clips and updates exactly once."""
    accumulation = config["training"].get("accumulation_steps", 1)
    rows, stats = [], {}
    for _ in range(accumulation):
        batch = next(batches)
        loss, current = batch_loss(model, batch, config, step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite task loss at update {step}")
        (loss / accumulation).backward()
        rows.extend(batch)
        if config.get("format") != "ajae-v3":
            stats = current
        else:
            for key, value in current.items():
                stats[key] = stats.get(key, 0) + (value if key.endswith("queries") else value / accumulation)
    return rows, stats


def load_checkpoint(path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") not in {"ajae-v1-checkpoint", "ajae-v3-checkpoint"} or "preprocessing" not in saved:
        raise ValueError("checkpoint needs formal AJAE weights and saved preprocessing")
    if "failure" in saved:
        raise ValueError("partial failure snapshots cannot be evaluated as completed updates")
    ScanTransform(saved["config"], state=saved["preprocessing"])
    model = AJAE(saved["config"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved


def check(config, data_root, examples, *, experiment=None):
    """Full real scans and task gradients; V2 also checks inherited state without any update."""
    if config.get("format") == "ajae-v3":
        return check_multiscale(config, data_root, experiment)
    torch.manual_seed(config["training"]["seed"])
    saved = (torch.load(PROJECT_ROOT / experiment["warm_start"]["checkpoint"], map_location="cpu", weights_only=True)
             if experiment else None)
    dataset = TrainingFrames(config, data_root, preprocessing=saved["preprocessing"] if saved else None)
    rows = [dataset[(index, draw, True)] for draw, index in enumerate(examples)]
    model = AJAE(config).cuda()
    if experiment:
        if len(rows) != config["training"]["batch_frames"]:
            raise ValueError("V2 check requires exactly one complete two-frame batch")
        identities = [[identity, frame] for _, identity, frame in dataset.dataset.samples]
        optimizer = initialize_stage(model, saved, config, experiment)
        conditions = dataset.conditions.state_dict()
        probabilities = dataset.dataset.sampling_probabilities(PROJECT_ROOT / config["training"]["sampling"])
        local = dict(format="ajae-v1-checkpoint", step=0, config=config, experiment=experiment,
            samples=identities, probabilities=torch.from_numpy(probabilities), optimizer=optimizer.state_dict(),
            scheduler_state=schedule_state(optimizer, config, 0), condition_sources=conditions)
        validate_resume_state(local, config, identities, probabilities, experiment, condition_sources=conditions)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (_preserve_buffers(model) if experiment else nullcontext()):
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
    if experiment:
        equal = all(torch.equal(value.cpu(), saved["model"][name]) for name, value in model.state_dict().items())
        if not equal or optimizer.state:
            raise ValueError("V2 check altered parent weights, buffers or optimizer state")
        stats["warm_start"] = dict(parent_step=saved["step"], local_step=0, all_model_tensors_equal=equal,
            optimizer_state_entries=len(optimizer.state), strict_local_resume=True,
            learning_rates={str(u): learning_rates(config, u) for u in (1, 50, 2048)})
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
    if experiment:
        stats["group_query_seats"] = [dict(normal=[len(g) for g in row["normal_groups"]],
            keep=[len(g) for g in row["keep_groups"]], union=len(row["query"]),
            normal_weights=row["normal_weights"], keep_weights=row["keep_weights"]) for row in rows]
        print(json.dumps(stats, indent=2, allow_nan=False), flush=True)
        return stats
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


def check_multiscale(config, data_root, experiment):
    """Two real scans, full/query equivalence and backward only; no optimizer is created."""
    import csv
    from .protocol import load_protocol
    from .scene import STUSequence
    torch.manual_seed(config["training"]["seed"])
    saved = torch.load(PROJECT_ROOT / experiment["warm_start"]["checkpoint"], map_location="cpu", weights_only=True)
    validate_stage_parent(saved, config, experiment)
    model = AJAE(config).cuda()
    migration = transfer_parent(model, saved)
    original_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    transform = ScanTransform(config, state=saved["preprocessing"], workers=4)
    with (PROJECT_ROOT / "results/profile/tables/frames.csv").open(encoding="utf-8-sig") as stream:
        frames = sorted([r for r in csv.DictReader(stream) if int(r["state"]) == 3],
                        key=lambda r: (int(r["visible"]), int(r["sequence"]), int(r["frame"])))
    selected = [("ordinary", frames[len(frames) // 2]), ("dense", frames[-1])]
    report = dict(optimizer_steps=0, checkpoint_saved=False, migration=migration,
                  host_E_before=host_disk(), resources_before=runtime_resources(), scans=[])
    for scope, record in selected:
        source = STUSequence.open(data_root, protocol=load_protocol(), partition="val",
            sequence_id=int(record["sequence"]), label_mode="required")[int(record["frame"])]
        started = time.perf_counter()
        prepared = transform(source)
        preparation_seconds = time.perf_counter() - started
        scan = to_device(prepared, "cuda")
        query = torch.from_numpy(np.linspace(0, source.real_count - 1, min(10240, source.real_count), dtype=np.int64)).cuda()
        model.zero_grad(set_to_none=True)
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            output = model(scan, return_features=True)
            full_score, full_relation = output["score"][query].cpu(), output["relation"][query].cpu()
            if len(output["score"]) != source.real_count or not torch.isfinite(output["score"]).all():
                raise ValueError("V3 full scan lost a physical return or produced nonfinite scores")
            del output
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        model.train()
        set_trainable(model, config, 129)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = training_forward(model, scan, query, return_features=True)
        torch.testing.assert_close(output["score"].detach().cpu(), full_score, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(output["relation"].detach().cpu(), full_relation, rtol=2e-5, atol=2e-5)
        # An unlabeled numerical probe checks derivatives without training on development labels.
        output["score"].square().mean().backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("V3 real-scan backward produced a nonfinite gradient")
        torch.cuda.synchronize()
        item = dict(scope=scope, sequence=source.sequence_id, frame=source.frame_id,
            source_identity=source_identity(source), returns=source.real_count, queries=len(query),
            support_cells=[len(scan[f"cell_{i}_count"]) for i in range(3)],
            preparation_seconds=preparation_seconds, full_inference_seconds=inference_seconds,
            query_forward_backward_seconds=time.perf_counter() - started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), gradients_finite=True,
            score_max_error=float((output["score"].detach().cpu() - full_score).abs().max()),
            relation_max_error=float((output["relation"].detach().cpu() - full_relation).abs().max()),
            new_head_columns_gradient_norm=float(model.head[0].weight.grad[:, 128:192].norm()))
        if item["new_head_columns_gradient_norm"] == 0:
            raise ValueError("zero-initialized new score columns cannot begin learning")
        report["scans"].append(item)
        print(f"V3核验 {scope} {source.sequence_id}/{source.frame_id} 点={source.real_count} "
              f"前向={inference_seconds:.2f}s 前反向={item['query_forward_backward_seconds']:.2f}s "
              f"显存={item['peak_cuda_bytes']/2**30:.2f}GiB", flush=True)
        del output, scan, prepared
    report["paired_risk"] = check_population(model, config, data_root, saved["preprocessing"])
    if any(not torch.equal(value.cpu(), original_state[name]) for name, value in model.state_dict().items()):
        raise ValueError("no-update V3 verification changed parameters or BN buffers")
    report.update(all_parameters_and_BN_unchanged=True, resources_after=runtime_resources(), host_E_after=host_disk())
    return report


def check_population(model, config, data_root, preprocessing):
    """Exercise two real frozen requests through augmentation and the new risk, without updates."""
    from .coverage import CoverageRequests
    dataset = TrainingFrames(config, data_root, preprocessing=preprocessing)
    requests = CoverageRequests(dataset.conditions.population, dataset.conditions.cells,
        dataset.conditions.counts[:, 0], config["training"]["seed"], 8192)
    rows = [dataset[requests.take()] for _ in range(2)]
    torch.set_num_threads(4)
    model.train()
    set_trainable(model, config, 129)
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss, stats = batch_loss(model, rows, config, 0)
    loss.backward()
    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError("V3 augmented physical-pair risk or gradient is nonfinite")
    torch.cuda.synchronize()
    stats.update(optimizer_steps=0, scans=4, seconds=time.perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), requests=[dict(r["request"],
            world=r["world"], frame=r["frame"], yaw=r["yaw"], population_counts=r["population_counts"],
            query_groups=r["query_groups"]) for r in rows])
    print(f"V3配对风险核验 请求=2 扫描=4 loss={stats['total']:.5f} 前反向={stats['seconds']:.2f}s 更新=0", flush=True)
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


def optimization_log_summary(records):
    """Describe completed updates, without treating correlated batches as independent trials."""
    from scipy.stats import spearmanr
    def summarize(rows):
        if not rows:
            return dict(updates=0)
        count = np.array([r["anomaly_queries"] for r in rows])
        gradient = np.array([r["gradient_norm"] for r in rows])
        return dict(updates=len(rows), zero_anomaly=int((count == 0).sum()),
            anomaly_mean=float(count.mean()),
            anomaly_quantiles=dict(zip(("min", "q25", "median", "q75", "q95", "q99", "max"),
                np.quantile(count, [0, .25, .5, .75, .95, .99, 1]).tolist())),
            gradient_median=float(np.median(gradient)), gradient_p95=float(np.quantile(gradient, .95)),
            gradient_max=float(gradient.max()), clipped=int((gradient > 1).sum()),
            over_100=int((gradient > 100).sum()), over_1000=int((gradient > 1000).sum()),
            count_gradient_spearman=float(spearmanr(count, gradient).statistic)
                if np.ptp(count) > 0 and np.ptp(gradient) > 0 else None)
    active = [r for r in records if r["auxiliary_fraction"] == 1]
    return dict(all=summarize(records), full_keep=summarize(active),
        by_anomaly_count={name: summarize([r for r in active if lo <= r["anomaly_queries"] < hi])
            for name, lo, hi in (("zero", 0, 1), ("1_4", 1, 5), ("5_19", 5, 20),
                                ("20_99", 20, 100), ("100_499", 100, 500), ("500_plus", 500, math.inf))})


def diagnostic_batches(records, checkpoint_step):
    """Choose before new predictions: historical peaks, denominator controls and the next saved-state update."""
    active = [r for r in records if r["step"] <= checkpoint_step and r["auxiliary_fraction"] == 1]
    chosen = {}
    def add(row, reason):
        if row is not None:
            chosen.setdefault(row["step"], dict(record=row, reasons=[]))["reasons"].append(reason)
    def median(rows):
        return sorted(rows, key=lambda r: (r["gradient_norm"], r["step"]))[len(rows) // 2] if rows else None
    for row in sorted(active, key=lambda r: (-r["gradient_norm"], r["step"]))[:3]:
        add(row, "top_three_historical_gradient_inputs_at_new_weights")
    keep = [r for r in active if r["keep"] > r["detection"]]
    add(max(keep, key=lambda r: r["gradient_norm"]) if keep else None, "largest_peak_with_keep_loss_above_detection")
    add(median(active), "median_gradient_control")
    add(median([r for r in active if r["anomaly_queries"] == 0]), "zero_anomaly_control")
    add(median([r for r in active if 1 <= r["anomaly_queries"] <= 4]), "few_anomaly_control")
    add(next((r for r in records if r["step"] == checkpoint_step + 1), None), "next_update_with_saved_preupdate_state")
    return [chosen[k] for k in sorted(chosen)]


class _Numerics:
    """Detached reductions only; do not retain activations or change the forward graph."""
    def __init__(self, model):
        self.model, self.values, self.handles, self.enabled = model, {}, [], True

    def __call__(self, name, value):
        if not self.enabled or not value.numel():
            return
        x = value.detach().double()
        summary = torch.stack((x.new_tensor(x.numel()), x.square().sum(), x.abs().max(), x.min(), x.max()))
        if name in self.values:
            old = self.values[name]
            summary = torch.stack((old[0] + summary[0], old[1] + summary[1],
                torch.maximum(old[2], summary[2]), torch.minimum(old[3], summary[3]), torch.maximum(old[4], summary[4])))
        self.values[name] = summary

    def __enter__(self):
        def normalization(name, module, args):
            if not self.enabled:
                return
            x = args[0].detach().float()
            dims = (0,) if isinstance(module, torch.nn.BatchNorm1d) else tuple(range(-len(module.normalized_shape), 0))
            variance = x.var(dims, unbiased=False)
            self(name + ".input_variance", variance)
            # This scale is a sensitivity clue, not the full normalization Jacobian.
            gain = module.weight.detach().abs() if isinstance(module, torch.nn.BatchNorm1d) else module.weight.detach().abs().max()
            self(name + ".scale_bound", gain / (variance + module.eps).sqrt())
        def output(name, module, args, value):
            if self.enabled:
                self(name + ".output", value if isinstance(value, torch.Tensor) else value.feat)
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                self.handles.append(module.register_forward_pre_hook(lambda m, a, n=name: normalization(n, m, a)))
            if name in ("backbone", "context", "point", "base_head", "relation_head"):
                self.handles.append(module.register_forward_hook(lambda m, a, v, n=name: output(n, m, a, v)))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def summary(self):
        result = {}
        for name, value in self.values.items():
            n, square, absolute, low, high = value.cpu().tolist()
            if not all(math.isfinite(v) for v in (n, square, absolute, low, high)):
                raise FloatingPointError(f"nonfinite diagnostic intermediate: {name}")
            result[name] = dict(elements=int(n), rms=math.sqrt(square / n), absolute_max=absolute, min=low, max=high)
        return result


@contextmanager
def normalization_mode(model, current_scan=False, *, reference=None):
    """Change only BN statistics; restore buffers, modes and RNG even on failure."""
    if current_scan and reference is not None:
        raise ValueError("choose current-scan or reference BN statistics, not both")
    with evaluation_state(model):
        if current_scan:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.train()
        elif reference is not None:
            source = dict(reference.named_modules())
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    other = source.get(name)
                    if not isinstance(other, torch.nn.BatchNorm1d):
                        raise ValueError(f"reference lacks matching BatchNorm: {name}")
                    # Affine parameters remain those of the evaluated model.
                    for key in ("running_mean", "running_var", "num_batches_tracked"):
                        value, replacement = getattr(module, key), getattr(other, key)
                        if value is None or replacement is None or value.shape != replacement.shape:
                            raise ValueError(f"incompatible BatchNorm buffer: {name}.{key}")
                        value.copy_(replacement)
        yield


def efficiency(checkpoint_path, data_root, output, reference=None, *, model_class=AJAE):
    """Compare a disposable complete update; never append to the training trajectory."""
    import hashlib
    import inspect
    output = Path(output)
    if output.exists() or output.with_suffix(".pt").exists():
        raise ValueError("efficiency output already exists")
    disk = host_disk()
    if disk["SizeRemaining"] - disk["reserve_bytes"] < 4 * 2**30:
        raise OSError("reserve space for bounded comparison tensors and atomic writes")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("format") != "ajae-v1-checkpoint" or "failure" in saved:
        raise ValueError("efficiency comparison requires a complete-update checkpoint")
    ScanTransform(saved["config"], state=saved["preprocessing"])
    model = model_class(saved["config"]).cuda()
    model.load_state_dict(saved["model"], strict=True)
    config, step = saved["config"], saved["step"]
    if config["loss"]["keep_mode"] != "mean" or config["loss"]["tail_weight"] != 0:
        raise ValueError("efficiency comparison requires the unchanged mean/no-tail recipe")
    dataset = TrainingFrames(config, data_root, preprocessing=saved["preprocessing"])
    probabilities = dataset.dataset.sampling_probabilities(PROJECT_ROOT / config["training"]["sampling"])
    identities = [[identity, frame] for _, identity, frame in dataset.dataset.samples]
    validate_resume_state(saved, config, identities, probabilities, saved["experiment"])
    requests = list(Requests(probabilities, config, step + 1, start=step))
    rows = [dataset[request] for request in requests]
    if len(rows) != 2 or any(row["original"] is None for row in rows):
        raise ValueError("comparison must contain the declared four full-scan forwards")
    torch.set_num_threads(4)
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(cpu(item) for item in value)
        return value
    def pin(value):
        if isinstance(value, torch.Tensor):
            return value.pin_memory()
        if isinstance(value, dict):
            return {key: pin(item) for key, item in value.items()}
        return value
    scans = [(row[key], row[query]) for row in rows
             for key, query in (("scan", "query"), ("original", "original_query"))]
    dependencies = []
    for scan, query in scans:
        ids = scan["neighbors"][scan["geometry_inverse"][query]].long()
        support = torch.unique(torch.cat((query, ids.clamp_min(0).flatten())), sorted=True)
        second = scan["neighbors"][scan["geometry_inverse"][support]].long()
        inputs = torch.unique(torch.cat((support, second.clamp_min(0).flatten())), sorted=True)
        dependencies.append(dict(S=support, T=inputs, queries=query))
    inputs = cpu(rows)
    rows = [pin(row) for row in rows]
    parameters = list(model.named_parameters())
    optimizer = make_optimizer(model, config)
    def restore():
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(deepcopy(saved["optimizer"]))
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        name, values, position, has_gauss, cached = saved["numpy_rng"]
        np.random.set_state((name, values.numpy().astype(np.uint32), position, has_gauss, cached))
        model.train()
    def update(capture=False):
        restore()
        scores, dependency_trace = [], {}
        def observe(name, value):
            if name.endswith((".input_rows", ".output_rows")):
                records = dependency_trace.setdefault(name, [])
                if len(records) < 4:
                    records.append(cpu(value))
        def score_hook(_module, _args, value):
            if len(scores) < 4:
                scores.append(value.detach().clone())
        handle = model.register_forward_hook(score_hook) if capture else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        set_learning_rates(optimizer, config, step + 1)
        before_forward = forward_state(model)  # Include the formal trainer's pre-forward snapshot cost.
        optimizer.zero_grad(set_to_none=True)
        loss, stats = batch_loss(model, rows, config, step, trace=observe if capture else None)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite disposable task loss")
        loss.backward()
        gradients = {name: cpu(p.grad) for name, p in parameters} if capture else None
        norm = torch.nn.utils.get_total_norm([p.grad for _, p in parameters if p.grad is not None], error_if_nonfinite=True)
        extreme = float(norm) >= saved["experiment"]["optimization_monitor"]["gradient_alert"]
        torch.nn.utils.clip_grads_with_norm_([p for _, p in parameters], config["training"]["gradient_clip"], norm)
        clipped = {name: cpu(p.grad) for name, p in parameters} if capture else None
        optimizer.step()
        if any(not torch.isfinite(p).all() for _, p in parameters):
            raise FloatingPointError("nonfinite disposable parameter update")
        torch.cuda.synchronize()
        elapsed, peak = time.perf_counter() - started, torch.cuda.max_memory_allocated()
        if handle is not None:
            handle.remove()
        result = dict(seconds=elapsed, peak_cuda_bytes=peak, gradient_norm=float(norm), gradient_alert=extreme, stats=stats)
        if capture:
            state = cpu(model.state_dict())
            result["tensors"] = dict(scores=cpu(scores), gradients=gradients, clipped_gradients=clipped,
                model=state, optimizer=cpu(optimizer.state_dict()),
                updates={name: state[name].double() - saved["model"][name].double() for name, _ in parameters},
                rng=cpu(forward_state(model)), dependency_trace=dependency_trace)
        return result
    report = dict(format="ajae-efficiency-comparison", checkpoint=str(Path(checkpoint_path).resolve()),
        checkpoint_step=step, next_update=step + 1, formal_updates=0,
        reference=str(reference) if reference is not None else None,
        requests=requests, tolerance=dict(rtol=2e-5, atol=2e-6),
        source_model_sha256=hashlib.sha256(Path(inspect.getfile(model_class)).read_bytes()).hexdigest(),
        host_E_before=disk, resources_before=runtime_resources(),
        scope="next frozen training batch, four forwards in original order; full input evaluation also checked; no recipe or checkpoint changes",
        timing="two warmups then five isolated reset-state updates; synchronized wall time includes GPU transfer and all in-model dependency/cache construction; preparation, reset and tensor capture excluded equally",
        dependency_sizes=[dict(points=len(scan["xyzi"]), queries=len(query), S=len(d["S"]), T=len(d["T"]))
                          for (scan, query), d in zip(scans, dependencies, strict=True)])
    _atomic_json(output, report)
    for _ in range(2):
        update()
    trials = []
    for repeat in range(5):
        trials.append(update())
        print(json.dumps(dict(event="efficiency_update", repeat=repeat, **trials[-1])), flush=True)
    captured = update(capture=True)
    restore()
    evaluation = []
    with evaluation_state(model):
        for scan, query in scans:
            device_scan = to_device(scan, "cuda")
            full = model(device_scan)
            partial = model(device_scan, query.to("cuda"))
            evaluation.append(dict(full=cpu(full), partial=cpu(partial)))
            del device_scan, full, partial
    bundle = dict(inputs=inputs, dependencies=dependencies, evaluation=evaluation,
                  stats=captured["stats"], gradient_norm=captured["gradient_norm"], **captured["tensors"])
    report.update(trials=trials, median_seconds=float(np.median([r["seconds"] for r in trials])),
        peak_cuda_bytes=max(r["peak_cuda_bytes"] for r in trials),
        disposable_updates=8, captured_stats=captured["stats"],
        dependency_trace_present=bool(bundle.pop("dependency_trace")))
    _atomic_json(output, report)
    trace = captured["tensors"]["dependency_trace"]
    if trace:
        for name, expected in (("relations.0.output_rows", "S"), ("relations.0.input_rows", "T"),
                               ("relations.1.output_rows", "queries"), ("relations.1.input_rows", "S")):
            if len(trace.get(name, [])) != 4 or any(not torch.equal(a,b[expected]) for a,b in zip(trace[name],dependencies,strict=True)):
                raise ValueError(f"actual relationship dependencies differ: {name}")
    if reference is None:
        save_checkpoint(output.with_suffix(".pt"), bundle)
        report["comparison_tensors"] = str(output.with_suffix(".pt"))
    else:
        expected = torch.load(reference, map_location="cpu", weights_only=True, mmap=True)
        comparisons = {}
        def compare(actual, wanted, name):
            if isinstance(wanted, torch.Tensor):
                if actual.shape != wanted.shape or actual.dtype != wanted.dtype:
                    raise ValueError(f"tensor identity changed: {name}")
                exact = name.lstrip(".").startswith(("inputs", "dependencies", "rng")) or not wanted.is_floating_point()
                difference = (actual.double() - wanted.double()).abs()
                limit = torch.zeros_like(difference) if exact else 2e-6 + 2e-5 * wanted.double().abs()
                comparisons[name] = dict(elements=actual.numel(), exact=exact,
                    bitwise_equal=torch.equal(actual,wanted),
                    outside_tolerance=int((~torch.isfinite(difference) | (difference > limit)).sum()),
                    max_absolute=float(difference.max()) if difference.numel() else 0.,
                    relative_l2=float(difference.norm() / wanted.double().norm().clamp_min(1e-30)))
            elif isinstance(wanted, dict):
                if actual.keys()!=wanted.keys(): raise ValueError(f"comparison keys differ: {name}")
                for key in wanted: compare(actual[key],wanted[key],f"{name}.{key}")
            elif isinstance(wanted, (list,tuple)):
                if len(actual)!=len(wanted): raise ValueError(f"comparison length differs: {name}")
                for i,(a,b) in enumerate(zip(actual,wanted,strict=True)): compare(a,b,f"{name}.{i}")
            elif isinstance(wanted,float):
                if actual is None or not math.isfinite(actual): raise ValueError(f"invalid scalar: {name}")
                limit = 0. if name.startswith(".optimizer.param_groups") else 2e-6+2e-5*abs(wanted)
                comparisons[name]=dict(outside_tolerance=int(abs(actual-wanted)>limit),max_absolute=abs(actual-wanted))
            elif actual != wanted:
                raise ValueError(f"comparison identity differs: {name}")
        compare(bundle,expected,"")
        report["comparisons"] = comparisons
        report["outside_tolerance"] = {k:v for k,v in comparisons.items() if v["outside_tolerance"]}
        report["compatible"] = not report["outside_tolerance"]
    restore()
    report.update(status="completed", resources_after=runtime_resources(), host_E_after=host_disk(),
        saved_reference_state_restored=all(torch.equal(v.cpu(),saved["model"][k]) for k,v in model.state_dict().items()))
    _atomic_json(output, report)
    return report


def diagnose(checkpoint_path, data_root, log_path, output):
    """Trained-state localization; disposable optimizer probes never enter the formal trajectory."""
    from .evaluate import prepare_fixed, evaluate_fixed
    output = Path(output)
    if output.exists():
        raise ValueError("diagnostic output already exists")
    records = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    if [r["step"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("diagnostics require a complete logged update prefix")
    model, saved = load_checkpoint(checkpoint_path)
    config, step = saved["config"], saved["step"]
    if config["loss"]["keep_mode"] != "mean" or config["loss"]["tail_weight"] != 0:
        raise ValueError("this diagnostic compares the declared detection-plus-mean objective")
    chosen = diagnostic_batches(records, step)
    selection = dict(train=saved["experiment"]["selection"]["train"], val={})
    report = dict(format="ajae-optimization-diagnostic", checkpoint=str(checkpoint_path), checkpoint_step=step,
        formal_optimizer_updates=0, disposable_optimizer_steps=0, status="running",
        scope="206 only; saved trained state; historical inputs at new weights are not historical peak reproduction; no calibration or recipe changes",
        log_prefix=len(records), log_summary=optimization_log_summary(records),
        evaluated_prefix_summary=optimization_log_summary(records[:step]), selection=selection,
        batch_selection=[dict(step=c["record"]["step"], samples=c["record"]["samples"], draws=c["record"]["draws"],
                              reasons=c["reasons"]) for c in chosen],
        batches=[], normalization={}, host_E_before=host_disk(), resources_before=runtime_resources())
    _atomic_json(output, report)
    dataset = TrainingFrames(config, data_root, preprocessing=saved["preprocessing"])
    parameters = list(model.named_parameters())
    scopes = {name: ".".join(name.split(".")[:2]) if name.startswith("relations.") else name.split(".")[0]
              for name, _ in parameters}
    optimizer = make_optimizer(model, config)
    def restore():
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(deepcopy(saved["optimizer"]))
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for choice in chosen:
            reference = choice["record"]
            prepared_at = time.perf_counter()
            rows = [dataset[(index, draw, True)] for index, draw in zip(reference["samples"], reference["draws"], strict=True)]
            if sum(int((r["target"] == 1).sum()) for r in rows) != reference["anomaly_queries"]:
                raise ValueError("reconstructed queries do not match the recorded anomaly denominator")
            preparation_seconds = time.perf_counter() - prepared_at
            torch.set_num_threads(4)
            restore()
            model.train()
            with _Numerics(model) as trace:
                total, stats, observed = batch_loss(model, rows, config, step, details=True, trace=trace)
                if not torch.isfinite(total):
                    raise FloatingPointError("nonfinite diagnostic objective")
                trace.enabled = False
                scores = torch.cat([observed["scores"][2 * i][r["detection_index"].cuda()] for i, r in enumerate(rows)])
                anomaly = scores[observed["target"] == 1]
                components = dict(detection=observed["components"]["detection"], keep=observed["components"]["keep"],
                    detection_anomaly=.5 * F.softplus(-anomaly).mean() if len(anomaly) else scores.sum() * 0)
                tensors = [p for _, p in parameters] + observed["context"] + observed["point"] + observed["scores"]
                gradients = {}
                for name, loss in components.items():
                    values = torch.autograd.grad(loss, tensors, allow_unused=True, retain_graph=True)
                    gradients[name] = [(g.detach().cpu() if g is not None else torch.zeros_like(t, device="cpu"))
                                       for g, t in zip(values, tensors, strict=True)]
                    if any(not torch.isfinite(g).all() for g in gradients[name]):
                        raise FloatingPointError(f"nonfinite {name} diagnostic gradient")
                    del values
                groups = {scope: [i for i, (name, _) in enumerate(parameters) if scopes[name] == scope]
                          for scope in sorted(set(scopes.values()))}
                groups["all_parameters"] = list(range(len(parameters)))
                offset = len(parameters)
                for name in ("context", "point", "scores"):
                    groups["shared_" + name] = list(range(offset, offset + len(observed[name])))
                    offset += len(observed[name])
                compared = {scope: _gradient_comparison({name: torch.cat([value[i].reshape(-1) for i in indices])
                                for name, value in gradients.items()}) for scope, indices in groups.items()}
                score_l1 = sum(float((gradients["detection"][i] + config["loss"]["keep_weight"] * gradients["keep"][i]).double().abs().sum())
                               for i in groups["shared_scores"])
                total.backward()
                additive, largest = {}, []
                for i, (name, parameter) in enumerate(parameters):
                    actual = parameter.grad.detach().cpu().double() if parameter.grad is not None else torch.zeros_like(parameter, device="cpu", dtype=torch.float64)
                    expected = gradients["detection"][i].double() + config["loss"]["keep_weight"] * gradients["keep"][i].double()
                    value = additive.setdefault(scopes[name], dict(error_squared=0., gradient_squared=0., absolute_max=0.))
                    value["error_squared"] += float((actual - expected).square().sum())
                    value["gradient_squared"] += float(actual.square().sum())
                    value["absolute_max"] = max(value["absolute_max"], float((actual - expected).abs().max()))
                    largest.append(dict(parameter=name, gradient_norm=float(actual.norm())))
                additive = {name: dict(relative_l2_error=math.sqrt(v["error_squared"] / max(v["gradient_squared"], 1e-30)),
                                      absolute_max=v["absolute_max"]) for name, v in additive.items()}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                changes, anomaly_linear_change = {}, 0.
                for i, (name, parameter) in enumerate(parameters):
                    actual = parameter.detach().cpu().double()
                    if not torch.isfinite(actual).all():
                        raise FloatingPointError("nonfinite disposable AdamW update")
                    before = saved["model"][name].double()
                    delta = actual - before
                    values = changes.setdefault(scopes[name], dict(parameter_squared=0., update_squared=0.))
                    values["parameter_squared"] += float(before.square().sum())
                    values["update_squared"] += float(delta.square().sum())
                    anomaly_linear_change += float((delta * gradients["detection_anomaly"][i].double()).sum())
                changes = {name: dict(parameter_norm=math.sqrt(v["parameter_squared"]), update_norm=math.sqrt(v["update_squared"]),
                    relative_update=math.sqrt(v["update_squared"]) / (math.sqrt(v["parameter_squared"]) + 1e-12))
                    for name, v in changes.items()}
                report["batches"].append(dict(input_log_step=reference["step"], parameter_step=step, reasons=choice["reasons"],
                    historical_gradient=reference["gradient_norm"], historical_detection=reference["detection"],
                    historical_keep=reference["keep"], stats=stats, gradients=compared, score_gradient_l1=score_l1,
                    combined_gradient_before_clip=float(norm), updates=changes,
                    gradient_additivity=additive, largest_parameter_gradients=sorted(largest, key=lambda v: -v["gradient_norm"])[:10],
                    anomaly_loss_first_order_update_estimate=anomaly_linear_change, numerics=trace.summary(),
                    preparation_seconds=preparation_seconds, total_seconds=time.perf_counter() - prepared_at))
                report["disposable_optimizer_steps"] += 1
                _atomic_json(output, report)
                print(json.dumps(dict(event="trained_gradient_diagnostic", input_step=reference["step"],
                    parameter_step=step, gradient=float(norm), seconds=report["batches"][-1]["total_seconds"])), flush=True)
                del total, observed, tensors, components, gradients, scores, anomaly, loss, trace
            del rows
        restore()
        prepared = prepare_fixed(data_root, selection, synthetic_splits=["train"])
        transform = ScanTransform(config, state=saved["preprocessing"], workers=4)
        with evaluation_state(model):
            report["normalization"] = evaluate_fixed(model, transform, prepared, directory=output.parent,
                include_normalization=True)["train"]["normalization"]
        report["state_restored"] = all(torch.equal(v.cpu(), saved["model"][k]) for k, v in model.state_dict().items())
        if not report["state_restored"] or any(float(v["step"]) != step for v in saved["optimizer"]["state"].values()):
            raise ValueError("diagnostics changed the saved reference state")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="stopped", error=repr(error), automatic_retry=False)
        raise
    finally:
        report.update(seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(), host_E_after=host_disk())
        _atomic_json(output, report)
    return report


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def make_optimizer(model, config):
    t = config["training"]
    if config.get("format") == "ajae-v3":
        groups = []
        for name, rate in (("backbone", t["backbone_learning_rate"]),
                           ("inherited", t["inherited_learning_rate"]), ("new_modules", t["learning_rate"])):
            for decay in (True, False):
                parameters = [p for key, p in model.named_parameters()
                    if parameter_family(key) == name and (p.ndim >= 2) == decay]
                groups.append(dict(name=name + ("_decay" if decay else "_no_decay"), params=parameters,
                                   lr=rate, weight_decay=t["weight_decay"] if decay else 0.))
        return torch.optim.AdamW(groups, lr=t["learning_rate"], betas=(.9, .999), eps=1e-8)
    parameters = model.parameters()
    if config["initialization"] == "nuscenes_litept_s":
        parameters = [dict(name="backbone", params=list(model.backbone.parameters()), lr=t["backbone_learning_rate"]),
            dict(name="new_modules", params=[p for name, p in model.named_parameters()
                                           if not name.startswith("backbone.")], lr=t["learning_rate"])]
    return torch.optim.AdamW(parameters, lr=t["learning_rate"], weight_decay=t["weight_decay"])


def parameter_family(name):
    return "backbone" if name.startswith("backbone.") else "inherited" if name.startswith(("context.", "point.")) else "new_modules"


def set_trainable(model, config, update):
    if config.get("format") == "ajae-v3":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(update > config["training"]["learning_rate_schedule"]["freeze_updates"]
                                     or parameter_family(name) == "new_modules")


def learning_rate_factor(update, schedule):
    """One-based update: first uses the floor, warmup's last uses the peak, final uses the floor."""
    if not 1 <= update <= schedule["total_updates"]:
        raise ValueError("update lies outside the declared learning-rate cycle")
    warmup = schedule["warmup_updates"]
    if update <= warmup:
        return schedule["start_factor"] + (1 - schedule["start_factor"]) * (update - 1) / (warmup - 1)
    phase = (update - warmup) / (schedule["total_updates"] - warmup)
    return schedule["end_factor"] + (1 - schedule["end_factor"]) * .5 * (1 + math.cos(math.pi * phase))


def learning_rates(config, update):
    t = config["training"]
    if config.get("format") == "ajae-v3":
        schedule = t["learning_rate_schedule"]
        if not 1 <= update <= schedule["total_updates"]:
            raise ValueError("V3 update lies outside the declared1024-update schedule")
        warmup, freeze, decay = (schedule[k] for k in ("warmup_updates", "freeze_updates", "decay_start"))
        if update <= decay:
            new = .1 + .9 * min(update - 1, warmup - 1) / (warmup - 1)
            inherited = 0. if update <= freeze else .1 + .9 * (update - freeze - 1) / (warmup - 1)
        else:
            new = inherited = .1 + .9 * .5 * (1 + math.cos(math.pi * (update - decay) / (schedule["total_updates"] - decay)))
        return [factor * peak for peak, factor in ((t["backbone_learning_rate"], inherited),
            (t["inherited_learning_rate"], inherited), (t["learning_rate"], new)) for _ in range(2)]
    factor = learning_rate_factor(update, t["learning_rate_schedule"]) if "learning_rate_schedule" in t else 1.
    return [factor * t["backbone_learning_rate"], factor * t["learning_rate"]] if "backbone_learning_rate" in t else [factor * t["learning_rate"]]


def set_learning_rates(optimizer, config, update):
    rates = learning_rates(config, update)
    if len(rates) != len(optimizer.param_groups):
        raise ValueError("optimizer groups differ from the declared learning rates")
    for group, rate in zip(optimizer.param_groups, rates, strict=True):
        group["lr"] = rate
    return {group.get("name", "all_parameters"): rate for group, rate in zip(optimizer.param_groups, rates, strict=True)}


def schedule_state(optimizer, config, completed):
    if "learning_rate_schedule" not in config["training"]:
        return None
    # Checkpoints retain the last applied rate; the next update derives its own rate from the global index.
    result = dict(completed_updates=completed, learning_rates=[g["lr"] for g in optimizer.param_groups])
    if config.get("format") == "ajae-v3":
        result.update(phase="adapt" if completed <= 128 else "joint", accumulation_boundary=0,
            parameter_updates={g["name"]: sorted({int(optimizer.state.get(p, {}).get("step", 0)) for p in g["params"]})
                               for g in optimizer.param_groups})
    return result


def parameter_change(model, before):
    """Detached parameter-family L2 changes; no rescaling of the actual optimization step."""
    sums = {}
    for name, parameter in model.named_parameters():
        group = (parameter_family(name) if getattr(model, "config", {}).get("relation_mode") == "multiscale" else
                 "backbone" if name.startswith("backbone.") else "new_modules")
        previous = before[name].double()
        values = torch.stack((previous.square().sum(), (parameter.detach().double() - previous).square().sum()))
        sums[group] = sums.get(group, 0) + values
    result = {}
    for group, values in sums.items():
        magnitude, delta = values.sqrt().cpu().tolist()
        result[group] = dict(parameter_norm=magnitude, update_norm=delta, relative_update=delta / (magnitude + 1e-12))
    return result


def gradient_groups(model):
    sums = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            group = (parameter_family(name) if getattr(model, "config", {}).get("relation_mode") == "multiscale" else
                     "backbone" if name.startswith("backbone.") else "new_modules")
            sums[group] = sums.get(group, 0) + parameter.grad.detach().double().square().sum()
    return {name: float(value.sqrt()) for name, value in sums.items()}


def forward_state(model):
    """Only buffers and RNG change before the optimizer; parameters need no per-step copy."""
    name, values, position, has_gauss, cached = np.random.get_state()
    return dict(buffers={name: value.detach().clone() for name, value in model.named_buffers()},
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        python_rng=random.getstate(),
        numpy_rng=(name, torch.from_numpy(values.astype(np.int64)), position, has_gauss, cached))


def restore_forward_state(model, state):
    """Discard an interrupted accumulation while parameters and optimizer are still unchanged."""
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(state["buffers"][name])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    name, values, position, has_gauss, cached = state["numpy_rng"]
    np.random.set_state((name, values.numpy().astype(np.uint32), position, has_gauss, cached))
    model.zero_grad(set_to_none=True)


def intensity_summary(sequence):
    """Exact raw-return quantiles from the one normal training source, without label conditioning."""
    histogram, frames = {}, []
    for index in range(len(sequence)):
        source = sequence[index]
        values = source.xyzi[source.real_slots, 3]
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite raw training intensity")
        levels, counts = np.unique(values, return_counts=True)
        for level, count in zip(levels, counts):
            histogram[float(level)] = histogram.get(float(level), 0) + int(count)
        frames.append(dict(frame=source.frame_id, returns=len(values), min=float(values.min()),
                           max=float(values.max()), above_1=int((values > 1).sum())))
    levels = np.array(sorted(histogram))
    counts = np.array([histogram[level] for level in levels], dtype=np.int64)
    cumulative, n = counts.cumsum(), int(counts.sum())
    quantiles = {}
    for name, q in (("q01", .01), ("q25", .25), ("q50", .5), ("q75", .75),
                    ("q95", .95), ("q99", .99), ("q999", .999)):
        position = q * (n - 1)
        lo, hi = math.floor(position), math.ceil(position)
        left, right = levels[np.searchsorted(cumulative, [lo, hi], side="right")]
        quantiles[name] = float(left + (position - lo) * (right - left))
    return dict(source="raw_train_206", frames=len(frames), returns=n, min=float(levels[0]), max=float(levels[-1]),
        mean=float(np.dot(levels, counts) / n), quantiles=quantiles,
        above_1=int(counts[levels > 1].sum()), above_255=int(counts[levels > 255].sum()),
        negative=int(counts[levels < 0].sum()), nonfinite=0, per_frame=frames,
        rule="identity_raw_stu_no_clip", physical_cross_sensor_calibration_verified=False)


def initialize(config, data_root, weights, output):
    """Save a new pretrained step zero and inspect real inputs; never call optimizer.step()."""
    if config["initialization"] != "nuscenes_litept_s":
        raise ValueError("initialize requires the explicit pretrained candidate configuration")
    output = Path(output)
    if any((output / name).exists() for name in ("0.pt", "diagnostic.json")):
        raise ValueError("pretrained initialization output already exists")
    disk, resources = host_disk(), runtime_resources()
    if disk["SizeRemaining"] - disk["reserve_bytes"] < 512 * 2**20:
        raise OSError("pretrained initialization would invade the E: reserve")
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = TrainingFrames(config, data_root)
    probabilities = dataset.dataset.sampling_probabilities(PROJECT_ROOT / config["training"]["sampling"])
    model = AJAE(config).cuda()
    fresh = {name: value.cpu().clone() for name, value in model.state_dict().items() if not name.startswith("backbone.")}
    loading = inherit_backbone(model, config, weights)
    if any(not torch.equal(model.state_dict()[name].cpu(), value) for name, value in fresh.items()):
        raise ValueError("pretrained loading changed new AJAE modules")
    optimizer = make_optimizer(model, config)
    selection = json.loads((PROJECT_ROOT / "protocol/micro.json").read_text())["selection"]["train"]
    # Reuse prior fixed identities; select by physical roles before seeing new scores.
    roles = (("sparse_normal", "sparse_normal"), ("near_1_4", "near_1_4"),
             ("far_20_plus", "middle_20_plus"), ("zero_anomaly", "zero_anomaly"))
    chosen, used = [], set()
    for pair in roles:
        batch = []
        for role in pair:
            index = next(i for i, row in enumerate(selection) if row["role"] == role and i not in used)
            used.add(index)
            batch.append(selection[index])
        chosen.append(batch)
    name, values, position, has_gauss, cached = np.random.get_state()
    payload = dict(format="ajae-v1-checkpoint", config=config,
        model={name: value.cpu().clone() for name, value in model.state_dict().items()},
        preprocessing=dataset.preprocessing, optimizer=optimizer.state_dict(), step=0,
        samples=[[identity, frame] for _, identity, frame in dataset.dataset.samples],
        probabilities=torch.from_numpy(probabilities), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
        numpy_rng=(name, torch.from_numpy(values.astype(np.int64)), position, has_gauss, cached),
        experiment=None, initialization_source=deepcopy(config["pretrained"]))
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output / "0.pt", payload)
    bn = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm1d)}
    report = dict(format="ajae-pretrained-initialization", status="running", config=config, checkpoint="0.pt",
        optimizer_steps=0, loading=loading, new_modules_unchanged=True,
        new_module_parameters=sum(p.numel() for name, p in model.named_parameters() if not name.startswith("backbone.")),
        optimizer_groups=[dict(name=g["name"], learning_rate=g["lr"], weight_decay=g["weight_decay"],
                               parameters=sum(p.numel() for p in g["params"])) for g in optimizer.param_groups],
        optimizer_state_entries=len(optimizer.state), trainable_backbone=all(p.requires_grad for p in model.backbone.parameters()),
        batchnorm={name: dict(epsilon=m.eps, momentum=m.momentum, running_variance_min=float(m.running_var.min()),
                             batches_tracked=int(m.num_batches_tracked)) for name, m in bn.items()},
        scope="206 only; fixed existing identities; unchanged pretrained step zero; gradients with full mean weight are prospective diagnostics, not post-warmup evidence; no task or transfer evaluation",
        selection=chosen, batches=[], normalization=[], host_E_before=disk, resources_before=resources,
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            gpu_bytes=torch.cuda.get_device_properties(0).total_memory, cpu_affinity=sorted(os.sched_getaffinity(0)),
            torch_threads=torch.get_num_threads(), preparation_workers=0, transform_threads=1))
    _atomic_json(output / "diagnostic.json", report)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        report["intensity"] = intensity_summary(dataset.dataset.sequence)
        for batch_index, records in enumerate(chosen):
            indices = select_samples(dataset.dataset, records)
            rows = [dataset[(index, batch_index * 2 + offset, True)] for offset, index in enumerate(indices)]
            model.train()
            with _preserve_buffers(model), _Numerics(model) as numerics:
                loss, stats = batch_loss(model, rows, config, 0, full_objective=True, trace=numerics)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite pretrained diagnostic loss")
                numerics.enabled = False
                loss.backward()
                norms = {}
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        raise ValueError(f"task gradient is absent from {name}")
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"nonfinite pretrained diagnostic gradient: {name}")
                    group = name.split(".")[0]
                    norms[group] = norms.get(group, 0.) + float(parameter.grad.double().square().sum())
                report["batches"].append(dict(batch=batch_index, samples=indices,
                    full_input_returns=[len(r["scan"]["xyzi"]) for r in rows], **stats,
                    gradient_norms={k: math.sqrt(v) for k, v in norms.items()},
                    gradient_norm=math.sqrt(sum(norms.values())), numerics=numerics.summary()))
            model.zero_grad(set_to_none=True)
            for row in rows:
                scan = to_device(row["original"], "cuda")
                controls, scores = {}, {}
                for mode, current in (("pretrained_running", False), ("current_scan", True)):
                    with normalization_mode(model, current), _Numerics(model) as numerics:
                        value = model(scan, trace=numerics)
                        numerics("scores", value)
                        scores[mode] = value.cpu()
                        controls[mode] = numerics.summary()
                delta = scores["current_scan"].double() - scores["pretrained_running"].double()
                report["normalization"].append(dict(frame=row["frame"], source_returns=len(delta),
                    input="original_uninserted_206", controls=controls,
                    score_difference=dict(mean=float(delta.mean()), rms=float(delta.square().mean().sqrt()),
                                          absolute_max=float(delta.abs().max()))))
            _atomic_json(output / "diagnostic.json", report)
            print(json.dumps(dict(event="pretrained_no_update_batch", batch=batch_index,
                samples=indices, loss=stats["total"], gradient_norm=report["batches"][-1]["gradient_norm"])), flush=True)
        # This compares all parameters and buffers, not just the optimizer step counter.
        report["state_unchanged"] = all(torch.equal(value.cpu(), payload["model"][name])
                                         for name, value in model.state_dict().items())
        if not report["state_unchanged"] or optimizer.state:
            raise ValueError("no-update diagnostics changed the saved initialization")
        report.update(status="completed", host_E_after=host_disk())
    except BaseException as error:
        report.update(status="stopped", error=repr(error), automatic_retry=False)
        raise
    finally:
        report.update(seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        _atomic_json(output / "diagnostic.json", report)
    return report


def repaired_selection(selection, replacements):
    """Retain every source-frame choice; only explicit old-to-new world identities may change."""
    selected = deepcopy(selection)
    values = list(replacements.values())
    if (len(set(values)) != len(values) or set(replacements) & set(values)
            or any(len(key) != 64 for key in [*replacements, *values])):
        raise ValueError("world replacements must be one-to-one, nonchained identities")
    for split in ("train", "validation"):
        for record in selected[split]:
            if record["identity"] in replacements:
                # Selection-time counts remain historical; evaluation reads the new full scan.
                record["selection_world_identity"] = record["identity"]
                record["identity"] = replacements[record["identity"]]
    return selected


def load_experiment(path, arm=None):
    experiment = json.loads(Path(path).read_text())
    if "control_of" in experiment:
        reference = json.loads((PROJECT_ROOT / experiment["control_of"]).read_text())
        if reference.get("format") != "ajae-v2-learning" or "control_of" in reference:
            raise ValueError("short controls require the original V2 declaration")
        experiment = reference | experiment
    if experiment.get("format") not in ("ajae-micro-learning", "ajae-short-learning", "ajae-staged-learning", "ajae-v2-learning", "ajae-v3-learning"):
        raise ValueError("unknown finite-learning declaration")
    if "selection_from" in experiment:
        experiment["selection"] = json.loads((PROJECT_ROOT / experiment["selection_from"]).read_text())["selection"]
    if "replacements_from" in experiment:
        if experiment["format"] not in {"ajae-v2-learning", "ajae-v3-learning"}:
            raise ValueError("world replacement is an explicit new-stage operation")
        root = json.loads((PROJECT_ROOT / experiment["replacements_from"]).read_text())
        current = {e["world_identity"]: e["path"] for s in root["splits"].values() for e in s["worlds"]}
        replacements = root.get("world_replacements", {})
        if any(current.get(r["identity"]) != r["path"] for r in replacements.values()):
            raise ValueError("world replacement target is absent from the current pool")
        experiment["world_replacements"] = {key: r["identity"] for key, r in replacements.items()}
        experiment["selection"] = repaired_selection(experiment["selection"], experiment["world_replacements"])
    if "arms" in experiment:
        arms = experiment.pop("arms")
        if arm not in arms:
            raise ValueError("choose an explicitly declared experiment arm")
        for section, changes in arms[arm].items():
            if section not in {"model", "loss", "training"}:
                raise ValueError("arm changes must name a configuration section")
            experiment.setdefault(section + "_overrides", {}).update(changes)
        experiment["arm"] = arm
    elif arm is not None:
        raise ValueError("this experiment does not declare multiple arms")
    return experiment


def experiment_config(experiment, path=None):
    config = load_config(path or (PROJECT_ROOT / experiment["base_config"] if experiment else PROJECT_ROOT / "protocol/model.json"))
    if experiment is not None:
        if experiment["format"] == "ajae-v3-learning":
            config["format"] = "ajae-v3"
            for key in ("model", "training", "loss"):
                config[key] = deepcopy(experiment[key])
        else:
            config["model"].update(experiment.get("model_overrides", {}))
            config["loss"].update(experiment["loss_overrides"])
            config["training"].update(experiment.get("training_overrides", {}))
        config["scope"] = experiment.get("scope", config.get("scope", ""))
    return validate_config(config)


def validate_initial_state(saved, config, identities, initial_changes=None):
    # Fresh starts permit only the explicitly declared objective or pretrained LR schedule change.
    # Resume changes require their own explicit validator and an unchanged completed history.
    actual = {k: v for k, v in config.items() if k != "scope"}
    expected = deepcopy({k: v for k, v in saved["config"].items() if k != "scope"})
    if saved["step"] != 0 or saved["optimizer"]["state"]:
        raise ValueError("fresh learning requires an untrained step-zero state")
    if initial_changes is not None:
        if (initial_changes == {"loss.keep_mode": {"from": "mean", "to": "worst"}}
                and expected["loss"]["keep_mode"] == "mean" and actual["loss"]["keep_mode"] == "worst"):
            expected["loss"]["keep_mode"] = "worst"
        elif (actual["initialization"] == expected["initialization"] == "nuscenes_litept_s"
                and "learning_rate_schedule" not in expected["training"]
                and "learning_rate_schedule" in actual["training"]
                and initial_changes == {"training.learning_rate_schedule": {
                    "from": None, "to": actual["training"]["learning_rate_schedule"]}}):
            validate_config(config)
            expected["training"]["learning_rate_schedule"] = deepcopy(actual["training"]["learning_rate_schedule"])
        else:
            raise ValueError("initial exception permits only the declared mean-to-worst change or new pretrained schedule")
    if actual != expected or saved["samples"] != identities:
        raise ValueError("initial model, objective, training definition or input identities changed")


def validate_resume_state(saved, config, identities, probabilities, experiment, *, without_201=False,
                          condition_sources=None, extend_warmup=False):
    if "failure" in saved:
        raise ValueError("a partial failure snapshot is not a completed-update resume state")
    expected = deepcopy(saved.get("experiment", saved.get("micro")))
    expected_config = deepcopy(saved["config"])
    if extend_warmup and (expected_config != config or expected != experiment):
        old = expected_config["training"].get("learning_rate_schedule", {})
        new = config["training"].get("learning_rate_schedule", {})
        if (not expected or not experiment or expected.get("format") != "ajae-v2-learning"
                or experiment.get("format") != "ajae-v2-learning"
                or old.get("kind") != "linear_warmup_cosine"
                or not 0 <= saved["step"] <= old.get("warmup_updates", -1)
                or new.get("total_updates", 0) <= old["total_updates"]
                or expected["maximum_updates"] != old["total_updates"]
                or experiment["maximum_updates"] != new["total_updates"]):
            raise ValueError("budget extension requires a V2 checkpoint within the unchanged warmup")
        # A longer cosine cycle may only replace the future, never the applied LR prefix.
        if any(learning_rates(saved["config"], update) != learning_rates(config, update)
               for update in range(1, max(1, saved["step"]) + 1)):
            raise ValueError("budget extension changed a completed update's learning rates")
        old["total_updates"] = new["total_updates"]
        expected_config["scope"] = config["scope"]
        expected_config["training"]["save_every"] = config["training"]["save_every"]
        expected["training_overrides"]["learning_rate_schedule"]["total_updates"] = new["total_updates"]
        expected["training_overrides"]["save_every"] = config["training"]["save_every"]
        for key in ("scope", "maximum_updates", "checkpoint_steps", "stop"):
            expected[key] = deepcopy(experiment[key])
        for key in ("synthetic_steps", "real_steps", "paired_normal_steps", "full_val19_steps", "primary"):
            expected["evaluation"][key] = deepcopy(experiment["evaluation"][key])
    if without_201:
        if (not expected or not experiment or expected.get("format") != "ajae-staged-learning"
                or experiment.get("format") != "ajae-staged-learning"
                or experiment.get("evaluation", {}).get("synthetic_splits") != ["train"]
                or experiment["evaluation"].get("full_synthetic_steps") != []):
            raise ValueError("201 removal requires a staged declaration with no 201 evaluation")
        # This explicit resume option changes only 201 evaluation, never training or val19.
        expected["evaluation"].update(synthetic_splits=["train"], full_synthetic_steps=[],
                                      synthetic=experiment["evaluation"]["synthetic"])
    same_probabilities = (saved["probabilities"] is None if probabilities is None else
                          torch.equal(saved["probabilities"], torch.from_numpy(probabilities)))
    if (expected_config != config or expected != experiment
            or saved["samples"] != identities or not same_probabilities):
        raise ValueError("resume configuration, input order or frame probabilities changed")
    if "group_queries" in config["training"] and (condition_sources is None
            or saved.get("condition_sources") != condition_sources):
        raise ValueError("resume native-low-support identities or slot sets changed")
    if "learning_rate_schedule" in config["training"]:
        rates = learning_rates(config, max(1, saved["step"]))
        expected_schedule = dict(completed_updates=saved["step"], learning_rates=rates)
        if config.get("format") == "ajae-v3":
            counts = {}
            for group in saved["optimizer"]["param_groups"]:
                values = [saved["optimizer"]["state"].get(p, {}) for p in group["params"]]
                actual = sorted({int(v.get("step", 0)) for v in values})
                expected_count = saved["step"] if group["name"].startswith("new_modules") else max(0, saved["step"] - 128)
                if actual != [expected_count] or any(not torch.isfinite(x).all() for v in values for x in v.values() if isinstance(x, torch.Tensor)):
                    raise ValueError("V3 parameter group has inconsistent actual optimizer updates")
                counts[group["name"]] = actual
            expected_schedule.update(phase="adapt" if saved["step"] <= 128 else "joint",
                                     accumulation_boundary=0, parameter_updates=counts)
            requests = saved["step"] * config["training"]["batch_frames"] * config["training"]["accumulation_steps"]
            expected_streams = {name: dict(seed=config["training"]["seed"], stream=tag, next_draw=requests)
                                for name, tag in (("query", 11), ("augmentation", 13))}
            if saved["request_state"]["consumed_requests"] != requests or saved["streams"] != expected_streams:
                raise ValueError("V3 recovery must bind RNG and coverage to the consumed accumulation boundary")
        if (saved.get("scheduler_state") != expected_schedule
                or [group["lr"] for group in saved["optimizer"]["param_groups"]] != rates):
            raise ValueError("saved learning rates do not match the completed global update")
    return saved["step"]


def validate_branch_state(saved, config, identities, probabilities, experiment):
    """Only the declared mean/increase protection experiment may fork a trained state."""
    branch, evaluation = experiment["branch"], experiment["evaluation"]
    start = validate_resume_state(saved, saved["config"], identities, probabilities, saved["experiment"])
    expected = deepcopy(saved["config"])
    if (experiment["format"] != "ajae-short-learning" or start != branch["step"] or start != 1024
            or experiment["maximum_updates"] != start + 128
            or experiment.get("arm") not in {"mean", "increase"}
            or config["loss"]["keep_mode"] != experiment["arm"]
            or expected["loss"]["keep_mode"] != "mean"
            or experiment["selection"] != saved["experiment"]["selection"]
            or any(evaluation.get(key) != [start + 128]
                   for key in ("synthetic_steps", "real_steps", "paired_normal_steps"))
            or evaluation.get("synthetic_splits") != ["train"]
            or any(evaluation.get(key, []) for key in
                   ("full_val19_steps", "full_synthetic_steps", "normalization_steps"))):
        raise ValueError("protection branch requires the declared1024-to1152 mean/increase control and fixed206/152 evaluation")
    expected["scope"] = config["scope"]
    expected["loss"]["keep_mode"] = experiment["arm"]
    if expected != config:
        raise ValueError("protection branch may change only loss.keep_mode and its description")
    required = ("model", "optimizer", "preprocessing", "torch_rng", "cuda_rng", "python_rng", "numpy_rng")
    if any(key not in saved for key in required) or not saved["optimizer"]["state"]:
        raise ValueError("protection branch requires the complete trained state")
    for state in saved["optimizer"]["state"].values():
        if int(state["step"]) != start or any(not torch.isfinite(value).all()
                for value in state.values() if isinstance(value, torch.Tensor)):
            raise ValueError("source optimizer is nonfinite or has inconsistent update counts")
    if any(not torch.isfinite(value).all() for value in saved["model"].values()):
        raise ValueError("source model contains nonfinite values")
    return start


def validate_stage_parent(saved, config, experiment):
    """Validate historical parent state independently of explicitly repaired diagnostic worlds."""
    if (experiment.get("format") not in {"ajae-v2-learning", "ajae-v3-learning"} or experiment["warm_start"]["step"] != 1152
            or saved.get("step") != 1152 or saved.get("format") != "ajae-v1-checkpoint"
            or "group_queries" not in config["training"] or "preprocessing" not in saved
            or saved["config"]["loss"]["keep_mode"] != "mean"
            or saved["config"]["loss"]["tail_weight"] != 0.
            or repaired_selection(saved.get("experiment", {}).get("selection", {}),
                                  experiment.get("world_replacements", {})) != experiment["selection"]):
        raise ValueError("the new stage must inherit the declared mean1152 model and fixed evaluation selection")
    probabilities = saved["probabilities"].numpy() if saved["probabilities"] is not None else None
    # Validate the parent's own completed history; repaired worlds belong to the new stage.
    validate_resume_state(saved, saved["config"], saved["samples"], probabilities, saved["experiment"])
    if experiment["format"] == "ajae-v3-learning":
        if config != experiment_config(experiment):
            raise ValueError("V3 initialization differs from its complete declaration")
        ScanTransform(config, state=saved["preprocessing"])
        if any(not torch.isfinite(value).all() for value in saved["model"].values()):
            raise ValueError("V3 parent contains nonfinite tensors")
        return
    expected = deepcopy(saved["config"])
    expected["model"].update(experiment.get("model_overrides", {}))
    expected["training"].update(experiment["training_overrides"])
    expected["loss"].update(experiment["loss_overrides"])
    expected["scope"] = experiment["scope"]
    if config != expected:
        raise ValueError("V2 may change only its declared risk, normalization, sampling and optimization settings")
    ScanTransform(config, state=saved["preprocessing"])
    if any(not torch.isfinite(value).all() for value in saved["model"].values()):
        raise ValueError("V2 parent model contains nonfinite tensors")


def initialize_stage(model, saved, config, experiment):
    """Inherit compatible weights and start the declared new-stage optimizer and RNG at zero."""
    validate_stage_parent(saved, config, experiment)
    if config.get("format") == "ajae-v3":
        transfer_parent(model, saved)
    else:
        model.load_state_dict(saved["model"], strict=True)
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    optimizer = make_optimizer(model, config)
    set_learning_rates(optimizer, config, 1)
    return optimizer


def fit(config, data_root, steps, output, resume=None, *, experiment=None, resume_without_201=False,
        extend_warmup=False):
    if steps < 1:
        raise ValueError("training needs a positive explicit update budget")
    output = Path(output)
    v3 = experiment is not None and experiment["format"] == "ajae-v3-learning"
    v2 = experiment is not None and experiment["format"] == "ajae-v2-learning"
    staged = experiment is not None and experiment["format"] in {"ajae-staged-learning", "ajae-v2-learning", "ajae-v3-learning"}
    warm_starting = (v2 or v3) and resume is None
    controlled = experiment is not None and "branch" in experiment
    branch_start = controlled and resume is None
    in_place = (staged or controlled) and resume is not None and Path(resume).resolve().parent == output.resolve()
    if resume_without_201 and not in_place:
        raise ValueError("201 removal requires resuming the same staged output directory")
    if extend_warmup and (not v2 or not in_place):
        raise ValueError("warmup budget extension requires resuming the same V2 output directory")
    if output.exists() and any(output.iterdir()) and not in_place:
        raise ValueError("training output is occupied; resume into an empty output directory")
    initial = PROJECT_ROOT / experiment["initial_checkpoint"] if experiment and "initial_checkpoint" in experiment else None
    if v2 or v3:
        initial = PROJECT_ROOT / experiment["warm_start"]["checkpoint"]
    if controlled:
        if "initial_checkpoint" in experiment or steps != experiment["maximum_updates"]:
            raise ValueError("protection control starts from its trained branch state and ends at the declared1152 update")
        initial = PROJECT_ROOT / experiment["branch"]["checkpoint"]
    state_path = resume if resume is not None else initial
    saved = torch.load(state_path, map_location="cpu", weights_only=True) if state_path is not None else None
    if config["initialization"] == "nuscenes_litept_s" and saved is None:
        raise ValueError("pretrained training must start from its saved initialized state")
    if saved is not None and (saved.get("format") not in {"ajae-v1-checkpoint", "ajae-v3-checkpoint"} or "preprocessing" not in saved):
        raise ValueError("resume requires the saved inference preprocessing state")
    if experiment is not None and not 1 <= steps <= experiment["maximum_updates"]:
        raise ValueError("finite learning cannot exceed its declared update budget")
    if steps > config["training"].get("learning_rate_schedule", {}).get("total_updates", steps):
        raise ValueError("training budget exceeds the declared learning-rate cycle")
    fixed_passes = experiment is not None and experiment["format"] == "ajae-micro-learning"
    dataset = TrainingFrames(config, data_root, preprocessing=saved["preprocessing"] if saved is not None else None,
                             cache_bytes=2 * 2**30 if fixed_passes else 0)
    if v2:
        preparation = json.loads((PROJECT_ROOT / config["training"]["sampling"]).with_suffix(".json").read_text())
        if (preparation["parameters"] != config["training"]["conditions"]
                or not preparation["regions_satisfied"] or not preparation["collision"]["certified"]):
            raise ValueError("V2 frozen pool has unresolved Euclidean placement or region requirements; resolve the affected worlds before training")
    selected = select_samples(dataset.dataset, experiment["selection"]["train"]) if fixed_passes else None
    probabilities = (None if fixed_passes else
        dataset.dataset.sampling_probabilities(PROJECT_ROOT / config["training"]["sampling"]))
    torch.manual_seed(config["training"]["seed"])
    model = AJAE(config).cuda()
    optimizer = make_optimizer(model, config)
    identities = [[identity, frame] for _, identity, frame in dataset.dataset.samples]
    condition_sources = dataset.conditions.state_dict() if dataset.conditions is not None else None
    start = 0
    if resume is not None:
        start = validate_resume_state(saved, config, identities, probabilities, experiment,
            without_201=resume_without_201, condition_sources=condition_sources, extend_warmup=extend_warmup)
        if start > steps or (start == steps and not staged):
            raise ValueError("explicit budget has no remaining updates")
        if staged or controlled:
            if not in_place:
                raise ValueError("continuous training resumes in its existing output directory")
            logged = experiment["branch"]["step"] if controlled else 0
            if (output / "loss.jsonl").exists():
                with (output / "loss.jsonl").open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row["step"] != logged + 1:
                            raise ValueError("training log has an incomplete update prefix")
                        logged = row["step"]
            if logged != start:
                raise ValueError("resume checkpoint and completed update log differ; no automatic replay")
    elif branch_start:
        start = validate_branch_state(saved, config, identities, probabilities, experiment)
    elif warm_starting:
        optimizer = initialize_stage(model, saved, config, experiment)
    elif saved is not None:
        validate_initial_state(saved, config, identities,
                               experiment.get("initial_changes") if experiment else None)
        if (config["initialization"] == "nuscenes_litept_s"
                and (probabilities is None or saved["probabilities"] is None
                     or not torch.equal(saved["probabilities"], torch.from_numpy(probabilities)))):
            raise ValueError("pretrained initialization and current frame probabilities differ")
    if saved is not None and not warm_starting:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if "python_rng" in saved:
            random.setstate(saved["python_rng"])
            name, values, position, has_gauss, cached = saved["numpy_rng"]
            np.random.set_state((name, values.numpy().astype(np.uint32), position, has_gauss, cached))
    if resume is None and not branch_start:
        set_learning_rates(optimizer, config, 1)
    accumulation = config["training"].get("accumulation_steps", 1)
    consumed = None
    if v3:
        from .coverage import CoverageRequests
        request_state = saved["request_state"] if resume else None
        arguments = (dataset.conditions.population, dataset.conditions.cells, dataset.conditions.counts[:, 0],
                     config["training"]["seed"], steps * config["training"]["batch_frames"] * accumulation)
        consumed = CoverageRequests(*arguments, state=request_state, mixture=experiment["coverage_mixture"])
        sampler = CoverageRequests(*arguments, state=request_state, mixture=experiment["coverage_mixture"])
    else:
        sampler = Requests(probabilities, config, steps, start, samples=selected)
    monitor = experiment.get("optimization_monitor", {}) if experiment else {}
    # Include retained evaluation states, an emergency state and atomic output overlap.
    disk = host_disk()
    checkpoint_bound = sum(p.numel() for p in model.parameters()) * 20 + 64 * 2**20
    retained = (len(experiment["checkpoint_steps"]) + 4 if v3 else len(experiment["evaluation"]["synthetic_steps"]) + 4 if staged else 7 if experiment else 2)
    peak = ((retained + monitor.get("retained_alerts", 0)) * checkpoint_bound
            + (2 * 2**30 if staged else 512_000_000)
            + (steps - start) * (8192 if staged else 512 + 20 * config["training"]["batch_frames"]))
    if peak >= disk["SizeRemaining"] - disk["reserve_bytes"]:
        raise OSError("training checkpoint peak would invade the E: reserve")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "config.json", config)
    run = dict(status="running", requested_updates=steps, start_update=start, pid=os.getpid(),
        initial_checkpoint=str(initial) if initial is not None else None,
        started_unix=time.time(), host_E_before=disk, estimated_peak_new_bytes=peak,
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
            cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
            preparation_workers=config["training"]["workers"], transform_cache_bytes_per_worker=dataset.cache_bytes))
    if controlled:
        run["branch"] = dict(experiment["branch"], arm=experiment["arm"],
                             inherited="model, buffers, AdamW moments, preprocessing, all RNG and global update")
    if v2 or v3:
        run["parent"] = dict(experiment["warm_start"],
            inherited="compatible encoders and named base-head columns" if v3 else "model, BatchNorm buffers, preprocessing; new optimizer and local random streams")
    if staged or controlled:
        run["resources_before"] = runtime_resources()
    if in_place:
        previous = json.loads((output / "run.json").read_text())
        run["previous_segments"] = previous.get("previous_segments", []) + [
            {k: previous.get(k) for k in ("start_update", "completed_updates", "seconds", "status", "budget_extension")}]
    if monitor:
        run["optimization_monitor"] = monitor
        run["retained_alerts"] = previous.get("retained_alerts", []) if in_place else []
    if resume_without_201:
        run["evaluation_change"] = dict(step=start, reason="user_cancelled_201_evaluation",
            previous=saved["experiment"]["evaluation"], current=experiment["evaluation"])
    if extend_warmup and saved["experiment"] != experiment:
        run["budget_extension"] = dict(step=start, previous=saved["experiment"], current=experiment,
                                       completed_learning_rates_unchanged=True)
    _atomic_json(output / "run.json", run)
    torch.cuda.reset_peak_memory_stats()
    prepared = None
    if experiment:
        from .evaluate import prepare_fixed, evaluate_fixed
        _atomic_json(output / "selection.json", experiment)
        prepared = prepare_fixed(data_root, experiment["selection"],
            synthetic_splits=experiment["evaluation"].get("synthetic_splits", ["train", "validation"]),
            binary_view=experiment["evaluation"].get("binary_view"))
        transform = ScanTransform(config, state=dataset.preprocessing, workers=8)
        diagnostic_indices = prepared["indices"]["train"]
        exposure = {index: saved.get("diagnostic_exposure", {}).get(index, 0) if resume or branch_start else 0
                    for index in diagnostic_indices}
    stages = json.loads((output / "exposure.json").read_text()).get("stages", {}) if in_place else {}
    if staged and not v3:
        from .exposure import coverage_context, training_summary
        wanted = {(identities[index][0], identities[index][1])
                  for index, _, _ in Requests(probabilities, config, steps, start)}
        dataset.exposure_context = coverage_context(config, wanted)

    def save_exposure():
        if prepared is not None:
            _atomic_json(output / "exposure.json", dict(
                scope="actual inserted-world requests for the fixed 206 diagnostic set; membership does not imply exposure",
                stages=stages,
                records=[dict(record, sample=index, requests=exposure[index])
                    for record, index in zip(experiment["selection"]["train"], diagnostic_indices, strict=True)]))

    def checkpoint_payload(step):
        name, values, position, has_gauss, cached = np.random.get_state()
        extra = {}
        if v3:
            state = consumed.state_dict()
            state["visited"] = torch.from_numpy(state["visited"])
            extra = dict(request_state=state, streams={name: dict(seed=config["training"]["seed"], stream=tag,
                next_draw=consumed.draw) for name, tag in (("query", 11), ("augmentation", 13))})
        return dict(format="ajae-v3-checkpoint" if v3 else "ajae-v1-checkpoint", config=config, model=model.state_dict(),
            preprocessing=dataset.preprocessing, optimizer=optimizer.state_dict(), step=step, samples=identities,
            probabilities=None if probabilities is None else torch.from_numpy(probabilities),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), experiment=experiment,
            python_rng=random.getstate(), numpy_rng=(name, torch.from_numpy(values.astype(np.int64)), position, has_gauss, cached),
            scheduler_state=schedule_state(optimizer, config, step),
            diagnostic_exposure=exposure if prepared is not None else {},
            **(dict(parent=experiment["warm_start"], condition_sources=condition_sources) if v2 or v3 else {}), **extra)

    def snapshot(step, failure=None, *, rolling=False):
        volume = host_disk()
        if checkpoint_bound > volume["SizeRemaining"] - volume["reserve_bytes"]:
            raise OSError("checkpoint write would invade the host E: reserve")
        payload = checkpoint_payload(step)
        if failure is not None:
            payload.update(failure=str(failure), gradients={name: p.grad for name, p in model.named_parameters() if p.grad is not None})
        filename = "failure.pt" if failure is not None else "resume.pt" if rolling else f"{step}.pt" if experiment else "model.pt"
        save_checkpoint(output / filename, payload)
        if failure is None:
            run["last_complete_checkpoint"] = filename
        save_exposure()

    def record_alert(step, norm, state, rows):
        retained = run["retained_alerts"]
        smallest = min(retained, key=lambda item: item["gradient_norm"]) if retained else None
        full = len(retained) >= monitor["retained_alerts"]
        alert = dict(step=step + 1, gradient_norm=norm, saved=False)
        if full and norm <= smallest["gradient_norm"]:
            return alert
        volume = host_disk()
        if checkpoint_bound > volume["SizeRemaining"] - volume["reserve_bytes"]:
            raise OSError("optimizer alert would invade the host E: reserve")
        payload = checkpoint_payload(step)
        # Parameters and optimizer are still pre-update; restore buffers/RNG to pre-forward.
        payload["model"].update(state["buffers"])
        payload.update({key: value for key, value in state.items() if key != "buffers"})
        payload.pop("scheduler_state")
        payload.update(format="ajae-v1-optimizer-alert", phase="before_forward", update=step + 1,
            gradient_norm=norm, batch_samples=[row["sample"] for row in rows],
            batch_draws=[row["draw"] for row in rows], learning_rates_for_update=[g["lr"] for g in optimizer.param_groups])
        filename = f"alert_{step + 1}.pt"
        save_checkpoint(output / filename, payload)
        if full:
            (output / smallest["checkpoint"]).unlink()
            retained.remove(smallest)
        alert.update(saved=True, checkpoint=filename)
        retained.append(alert)
        return alert

    def evaluate(step):
        nonlocal evaluating
        if stopped:
            return
        evaluating = True
        print(f"评价 {step}/{steps} 开始", flush=True)
        with evaluation_state(model):
            real_scores, synthetic_scores = {}, {}
            raw_scores, real_threshold = {}, None
            schedule = experiment["evaluation"]
            if staged:
                from .evaluate import evaluate_validation, evaluate_synthetic
                if v3:
                    from .evaluate import prepare_reference
                    prepare_reference(data_root, experiment)
                if not v3:
                    stages[str(step)] = training_summary(output / "loss.jsonl", config, step)
                else:
                    stages[str(step)] = dict(updates=step, consumed_requests=consumed.draw,
                        unique_world_frames=int(consumed.visited.sum()), binary_view=config["training"]["binary_view"])
                save_exposure()
                for suffix, due, evaluator, captured in (
                    ("val", schedule["full_val19_steps"], evaluate_validation, real_scores),
                    ("synthetic", schedule["full_synthetic_steps"], evaluate_synthetic, synthetic_scores)):
                    path = output / f"{step}_{suffix}.json"
                    if step not in due:
                        continue
                    if path.exists():
                        if v3 and suffix == "val":
                            from .evaluate import captured_scores
                            captured.update(captured_scores(data_root, output / f"{step}_real.npz"))
                        continue
                    if suffix == "val":
                        captured.update({(int(key), frame): None for key, frames in experiment["selection"]["val"].items()
                                         for frame in frames})
                        if "diagnostic_from" in experiment:
                            diagnostic = json.loads((PROJECT_ROOT / experiment["diagnostic_from"] / "selection.json").read_text())
                            captured.update({(r["sequence"], r["frame"]): None for r in diagnostic["frames"]})
                    else:
                        captured.update({(r["identity"], r["frame"]): None for r in experiment["selection"]["validation"]})
                    result = evaluator(data_root, checkpoint_path=output / f"{step}.pt", directory=output, capture=captured,
                                       **(dict(save_capture=True) if v3 and suffix == "val" else {}))
                    _atomic_json(path, dict(step=step, **result))
                    from .evaluate import print_metrics
                    print_metrics("完整val19" if suffix == "val" else "完整合成集", result)
                if "diagnostic_from" in experiment:
                    from .evaluate import evaluate_diagnostic
                    evaluate_diagnostic(data_root, output / f"{step}.pt", PROJECT_ROOT / experiment["diagnostic_from"],
                        output, capture=real_scores, reference=PROJECT_ROOT / experiment["reference_checkpoint"] if "reference_checkpoint" in experiment else None)
                if v3:
                    from .evaluate import evaluate_normal_source, model_selection
                    full = json.loads((output / f"{step}_val.json").read_text())
                    real_threshold = full["official_high_recall"]["threshold"]
                    raw_scores = {(201, r["frame"]): None for r in experiment["selection"]["validation"]}
                    path = output / f"{step}_normal201.json"
                    if not path.exists():
                        result = evaluate_normal_source(model, transform, data_root, real_threshold, capture=raw_scores)
                        _atomic_json(path, dict(step=step, checkpoint=str((output / f"{step}.pt").resolve()), **result))
                    parent = json.loads((PROJECT_ROOT / "results/keep/mean/global.json").read_text())
                    run["model_selection"] = model_selection(output, parent)
            if not staged or not (output / f"{step}.json").exists():
                result = evaluate_fixed(model, transform, prepared,
                    include_real=step in schedule["real_steps"], directory=output,
                    include_pairs=step in schedule.get("paired_normal_steps", []),
                    include_normalization=step in schedule.get("normalization_steps", []),
                    real_scores=real_scores, synthetic_scores=synthetic_scores, raw_scores=raw_scores,
                    real_threshold=real_threshold)
                _atomic_json(output / f"{step}.json", dict(step=step, checkpoint=f"{step}.pt", **result))
        evaluating = False
        print(f"评价 {step}/{steps} 完成", flush=True)

    workers = config["training"]["workers"]
    loader = DataLoader(dataset, sampler=sampler,
        batch_size=config["training"]["batch_frames"], num_workers=workers,
        collate_fn=_collate, pin_memory=True, persistent_workers=bool(workers),
        generator=torch.Generator().manual_seed(config["training"]["seed"] + 83) if experiment else None,
        **(dict(multiprocessing_context="spawn", prefetch_factor=1) if workers else {}))
    model.train()
    completed, rows = start, []
    before_forward, pending_update = None, False
    stopped, evaluating = False, False

    def request_stop(signum, _):
        nonlocal stopped
        stopped = True
        print(f"暂停请求：完成当前更新后保存，已完成 {completed}/{steps}", flush=True)
        if evaluating:
            raise KeyboardInterrupt("user stopped evaluation at a saved update boundary")

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)} if staged or controlled else {}
    try:
        print(f"训练 {start}/{steps} 就绪，目标 {steps} 步及其评价完成后退出", flush=True)
        if experiment:
            if not in_place:
                snapshot(start)
            if (not staged and not controlled) or start in experiment["evaluation"]["synthetic_steps"]:
                evaluate(start)
        with (output / "loss.jsonl").open("a") as log:
            batches = iter(loader)
            for step in range(start, steps):
                if stopped:
                    snapshot(completed, rolling=True)
                    break
                started = time.perf_counter()
                rates = set_learning_rates(optimizer, config, step + 1)
                set_trainable(model, config, step + 1)
                before_forward = forward_state(model) if monitor or v3 else None
                rows, pending_update = [], True
                optimizer.zero_grad(set_to_none=True)
                rows, stats = accumulate_batches(model, batches, config, step)
                parameters = list(model.parameters())
                norm = torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None], error_if_nonfinite=True)
                extreme = bool(monitor) and float(norm) >= monitor["gradient_alert"]
                measured = bool(monitor) and (step == 0 or (step + 1) % monitor["every_updates"] == 0 or extreme)
                if extreme:
                    stats["gradient_alert"] = record_alert(step, float(norm), before_forward, rows)
                if measured:
                    stats["gradient_groups_before_clip"] = gradient_groups(model)
                    before_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
                # This is exactly the second half of PyTorch's clip_grad_norm_, with the same norm.
                torch.nn.utils.clip_grads_with_norm_(parameters, config["training"]["gradient_clip"], norm)
                pending_update = False  # A failed optimizer step may have partially changed parameters.
                optimizer.step()
                if v3:
                    for row in rows:
                        consumed.consume(row["request"])
                if measured:
                    stats["parameter_changes"] = parameter_change(model, before_parameters)
                    del before_parameters
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError(f"nonfinite parameter after update {step + 1}")
                completed = step + 1
                stats.update(step=completed, gradient_norm=float(norm), learning_rates=rates,
                    seconds=time.perf_counter() - started,
                    samples=[r["sample"] for r in rows], draws=[r["draw"] for r in rows],
                    full_input_returns=[len(r["scan"]["xyzi"]) for r in rows])
                if staged and not v3:
                    stats["exposure"] = [r["exposure"] for r in rows]
                if v3:
                    stats.update(accumulation_steps=accumulation, consumed_requests=consumed.draw,
                        phase="adapt" if completed <= 128 else "joint",
                        requests=[dict(r["request"], world=r["world"], frame=r["frame"], yaw=r["yaw"],
                            population_counts=r["population_counts"], query_groups=r["query_groups"],
                            anomaly_queries=r["anomaly_queries"]) for r in rows])
                    for row in rows:
                        if row["sample"] in exposure:
                            exposure[row["sample"]] += 1
                elif prepared is not None:
                    hits = dict(normal=0, anomaly=0, active_keep=0)
                    for row in rows:
                        if row["sample"] in exposure:
                            exposure[row["sample"]] += 1
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
                displayed_rates = [g["lr"] for g in optimizer.param_groups[::2]] if v3 else rates.values()
                rates_text = "/".join(f"{rate:.2e}" for rate in displayed_rates)
                print(f"训练 {completed:4d}/{steps} loss={stats['total']:.5f} grad={float(norm):.3g} "
                      f"lr={rates_text} {stats['seconds']:.1f}s", flush=True)
                run["completed_updates"] = completed
                if completed % 32 == 0:
                    run["host_E_latest"] = host_disk()
                    if run["host_E_latest"]["SizeRemaining"] < disk["reserve_bytes"] + 2 * checkpoint_bound:
                        raise OSError("preserve the last complete update before exhausting checkpoint headroom")
                    if staged or controlled:
                        run["resources_latest"] = runtime_resources()
                    _atomic_json(output / "run.json", run)
                if stopped:
                    snapshot(completed, rolling=True)
                    break
                if experiment and completed in experiment["evaluation"]["synthetic_steps"]:
                    snapshot(completed)
                    evaluate(completed)
                elif experiment and completed in experiment.get("checkpoint_steps", []):
                    snapshot(completed)
                elif ((staged or experiment is None) and completed % config["training"]["save_every"] == 0) or completed == steps:
                    snapshot(completed, rolling=staged)
    except BaseException as error:
        if (staged or controlled) and stopped and isinstance(error, KeyboardInterrupt):
            snapshot(completed, rolling=True)
            run.update(status="interrupted", completed_updates=completed, error=repr(error))
        else:
            try:
                if v3 and pending_update:
                    restore_forward_state(model, before_forward)
                    set_learning_rates(optimizer, config, max(1, completed))
                    snapshot(completed, rolling=True)
                    run["recovery"] = "discarded partial gradients; saved model/RNG and consumed requests at the preceding complete update"
                else:
                    snapshot(completed, failure=error)
            except OSError as storage_error:
                run["emergency_save_error"] = str(storage_error)
            _atomic_json(output / "failure.json", dict(completed_updates=completed, error=repr(error),
                         samples=[r["sample"] for r in rows], automatic_retry=False))
            run.update(status="stopped", completed_updates=completed, error=repr(error))
            raise
    else:
        run.update(status="interrupted" if stopped else "completed", completed_updates=completed, host_E_after=host_disk())
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        run.update(seconds=time.time() - run["started_unix"],
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        _atomic_json(output / "run.json", run)
        label = {"completed": "训练及评价完成", "interrupted": "已暂停"}.get(run["status"], "故障停止")
        print(f"{label} {completed}/{steps}，记录：{output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "preview", "diagnose", "initialize", "fit"))
    parser.add_argument("--config", type=Path, help="defaults to the experiment's base configuration, or model.json")
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="completed trained state for isolated optimization diagnosis")
    parser.add_argument("--log", type=Path, help="completed update log for diagnosis")
    parser.add_argument("--efficiency", action="store_true", help="isolated complete-update numerical and timing comparison")
    parser.add_argument("--reference", type=Path, help="saved comparison tensors for --efficiency")
    parser.add_argument("--reference-code", type=Path, help="original model.py from a Git worktree; compare sequentially in one CUDA process")
    parser.add_argument("--weights", type=Path, help="official source weights for initialize only")
    parser.add_argument("--resume-without-201", action="store_true",
                        help="explicitly remove only 201 evaluation when resuming the staged run")
    parser.add_argument("--extend-warmup", action="store_true",
                        help="extend a V2 budget during unchanged warmup; preserve every completed update's recipe")
    parser.add_argument("--experiment", type=Path, help="declared finite or continuous learning budget and evaluation scope")
    parser.add_argument("--arm", help="arm explicitly listed in the experiment declaration")
    parser.add_argument("--sample", type=int, action="append", help="fixed training manifest index for check")
    args = parser.parse_args()
    if (args.efficiency or args.reference is not None or args.reference_code is not None) and args.command != "diagnose":
        parser.error("efficiency comparison uses the existing diagnose entry")
    if (args.reference is not None or args.reference_code is not None) and not args.efficiency:
        parser.error("reference inputs require --efficiency")
    if args.reference is not None and args.reference_code is not None:
        parser.error("choose saved tensors or a sequential original-code comparison")
    if args.resume_without_201 and (args.command != "fit" or args.resume is None or args.experiment is None):
        parser.error("--resume-without-201 requires a staged --experiment and --resume")
    if args.extend_warmup and (args.command != "fit" or args.resume is None or args.experiment is None):
        parser.error("--extend-warmup requires a V2 --experiment and --resume")
    if args.command != "diagnose" and (args.checkpoint is not None or args.log is not None):
        parser.error("--checkpoint and --log are diagnosis inputs, not training initialization")
    if args.weights is not None and args.command != "initialize":
        parser.error("--weights only initializes a fresh trajectory; it cannot modify a resume state")
    if args.arm is not None and (args.command not in {"fit", "check"} or args.experiment is None):
        parser.error("--arm requires a declared --experiment fit or check")
    experiment = load_experiment(args.experiment, args.arm) if args.experiment is not None else None
    if experiment is not None and args.command != "fit" and not (
            args.command == "check" and experiment["format"] in {"ajae-v2-learning", "ajae-v3-learning"}):
        parser.error("--experiment supports a declared fit or a V2/V3 no-update check")
    config = experiment_config(experiment, args.config)
    if config["initialization"] != "random_no_external_weights" and args.command in ("check", "preview") and not experiment:
        parser.error("use initialize to inspect the actual pretrained state")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        parser.error("the LitePT sparse-convolution implementation requires CUDA")
    if args.command == "initialize":
        if (args.weights is None or args.output is None or args.steps is not None
                or args.sample is not None or args.resume is not None):
            parser.error("initialize requires --weights and --output; no updates or resume")
        initialize(config, args.data_root, args.weights, args.output)
    elif args.command == "diagnose":
        if (args.checkpoint is None or (args.log is None and not args.efficiency) or args.output is None
                or args.steps is not None or args.sample is not None or args.resume is not None):
            parser.error("diagnose requires --checkpoint, --log and --output; no formal updates or resume")
        if args.efficiency:
            try:
                if args.reference_code is None:
                    efficiency(args.checkpoint, args.data_root, args.output, args.reference)
                else:
                    # Both classes use the same loaded CUDA libraries and algorithm caches; no live class is patched.
                    import importlib.util
                    import sys
                    spec = importlib.util.spec_from_file_location("src._efficiency_reference", args.reference_code)
                    original = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = original
                    spec.loader.exec_module(original)
                    reference_path = args.output.with_name(args.output.stem + "_reference.json")
                    control_path = args.output.with_name(args.output.stem + "_control.json")
                    baseline = efficiency(args.checkpoint, args.data_root, reference_path, model_class=original.AJAE)
                    candidate = efficiency(args.checkpoint, args.data_root, args.output, reference_path.with_suffix(".pt"))
                    control = efficiency(args.checkpoint, args.data_root, control_path, reference_path.with_suffix(".pt"), model_class=original.AJAE)
                    candidate.update(reference_median_seconds=baseline["median_seconds"],
                        reference_repeat_median_seconds=control["median_seconds"],
                        reference_repeat_compatible=control["compatible"],
                        comparison_scope="original, candidate, original in one CUDA process; same frozen batch and reset state; no kernel/precision setting changed")
                    _atomic_json(args.output, candidate)
            except BaseException as error:
                if args.output.exists():
                    report = json.loads(args.output.read_text())
                    report.update(status="stopped", error=repr(error), automatic_retry=False)
                    _atomic_json(args.output, report)
                raise
        else:
            diagnose(args.checkpoint, args.data_root, args.log, args.output)
    elif args.command == "check":
        if args.steps is not None or (args.output is not None and config.get("format") != "ajae-v3") or args.resume is not None:
            parser.error("check has no optimization budget, output directory or resume state")
        result = check(config, args.data_root, args.sample if args.sample is not None else [161, 162], experiment=experiment)
        if args.output is not None:
            _atomic_json(args.output, result)
    elif args.command == "preview":
        if args.steps is None or args.output is None or args.sample is not None or args.resume is not None:
            parser.error("preview requires --steps and --output; it has no fixed check samples or resume state")
        preview(config, args.data_root, args.steps, args.output)
    else:
        if args.steps is None or args.output is None or args.sample is not None:
            parser.error("fit requires --steps and --output; fixed check samples are not training input")
        fit(config, args.data_root, args.steps, args.output, args.resume, experiment=experiment,
            resume_without_201=args.resume_without_201, extend_warmup=args.extend_warmup)


if __name__ == "__main__":
    main()
