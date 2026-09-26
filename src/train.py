"""Run a bounded V4 training segment with explicit sampling and validation intervals."""

import argparse
import copy
from collections import Counter, defaultdict, deque
import gc
import heapq
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
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import (VERSION, PILOT_VERSION, CONTINUATION_VERSION, NATIVE_VERSION, NDP_VERSION, SOURCE_VERSION, Scans, file_sha256, identity,
                   load_manifest, write_json)
from .evaluate import PreparedScans, autocast, better, evaluate, evaluation_indices, memory_available, precision
from .model import (POINT_CHUNK, Segmentor, balanced_loss, ranking_loss, to_device,
                    LITEPT_COMMIT, WEIGHTS_REVISION, WEIGHTS_SHA256, RELATION_MODES)
from .normal import NORMAL_MODES, SCALES, HYPOTHESES, KERNELS, RAY_CHUNK, LOWER, UPPER


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


def source_order(records, seed, updates, far_updates=None):
    """Fixed role quotas with complete, hierarchically interleaved pool cycles."""
    pools = {name: [] for name in ("dense", "sparse", "control", "original")}
    for index, row in enumerate(records):
        group, count = row.get("group"), int(row.get("anomaly", 0))
        if group == "anomaly_nuscenes" and count > 0:
            pools["dense" if count >= 5 else "sparse"].append(index)
        elif group == "control_nuscenes" and int(row.get("inserted_points", 0)) > 0:
            pools["control"].append(index)
        elif group == "normal_nuscenes" and row.get("normal", 0) > 0 and not row.get("delta"):
            pools["original"].append(index)
    quotas = dict(dense=3, sparse=1, control=2, original=2)
    for role, quota in quotas.items():
        if len(pools[role]) < quota:
            raise ValueError(f"source sampling requires at least {quota} eligible {role} records")

    def path(index, role):
        row = records[index]
        if role == "original":
            return (row["scene"],)
        count = int(row["anomaly"] if role != "control" else row["inserted_points"])
        if role == "control":
            histogram = row["inserted_point_histogram"]
            if len(histogram) != 5 or sum(histogram) != count:
                raise ValueError("control range strata require all supervised inserted points")
            band = int(np.searchsorted(np.cumsum(histogram), (count + 1) // 2))
        else:
            distance = row.get("point_range_median")
            if distance is None or not np.isfinite(distance) or not LOWER <= distance <= UPPER:
                raise ValueError("anomaly range strata require supervised-point median range")
            band = int(np.searchsorted([10., 20., 30., 40.], distance, side="right"))
        return (band, row["instance"], row["scene"], count.bit_length() - 1,
                row.get("variant", ""), row.get("segment", 0))

    def permutation(role, cycle):
        role_id = tuple(quotas).index(role)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 719, role_id, cycle]))
        tree = {}
        for index in pools[role]:
            node = tree
            for key in path(index, role):
                node = node.setdefault(key, {})
            node.setdefault(None, []).append(index)

        def interleave(node, depth=0):
            if None in node:
                ordered = sorted(node[None], key=lambda i: (records[i].get("timestamp", records[i].get("frame", i)), i))
                intervals, result = deque([(0, len(ordered))]), []
                # Visit separated temporal representatives before adjacent frames.
                while intervals:
                    left, right = intervals.popleft()
                    middle = (left + right) // 2
                    result.append(ordered[middle])
                    children = [(left, middle), (middle + 1, right)]
                    for child in rng.permutation(2):
                        a, b = children[int(child)]
                        if a < b:
                            intervals.append((a, b))
                return result
            keys = list(node)
            queues = [deque(interleave(node[keys[int(i)]], depth + 1)) for i in rng.permutation(len(keys))]
            result = []
            if depth == 0 and role != "original":
                # Merge evenly spaced visits at each band's share of this role pool.
                # Rare distances remain spread across the cycle instead of ending early.
                schedule = [(.5 / len(queue), index, len(queue)) for index, queue in enumerate(queues)]
                heapq.heapify(schedule)
                while schedule:
                    _, index, capacity = heapq.heappop(schedule)
                    queue = queues[index]
                    result.append(queue.popleft())
                    if queue:
                        consumed = capacity - len(queue)
                        heapq.heappush(schedule, ((consumed + .5) / capacity, index, capacity))
                return result
            queues = deque(queues)
            while queues:
                queue = queues.popleft()
                result.append(queue.popleft())
                if queue:
                    queues.append(queue)
            return result

        return interleave(tree)

    cycles, queues = dict.fromkeys(quotas, 0), {role: deque() for role in quotas}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 720]))
    result = []
    for update in range(updates):
        batch = []
        for role, quota in quotas.items():
            for _ in range(quota):
                if not queues[role]:
                    queues[role].extend(permutation(role, cycles[role]))
                    cycles[role] += 1
                # A cycle boundary must not duplicate an observation within a batch.
                while queues[role][0] in batch:
                    queues[role].rotate(-1)
                batch.append(queues[role].popleft())
        # Shuffle role positions without reversing either side of a pool boundary.
        positions = rng.permutation([role for role, quota in quotas.items() for _ in range(quota)])
        offset, parts = 0, {}
        for role, quota in quotas.items():
            parts[role] = deque(batch[offset:offset + quota])
            offset += quota
        result.extend(parts[role].popleft() for role in positions)
        if far_updates is not None and any(records[i].get("anomaly", 0) > 0 and
                records[i]["point_range_median"] >= 30. for i in batch):
            far_updates.append(update)
    return result


def effective_batches(order, rank=0, world_size=1):
    for start in range(0, len(order), BATCH_SIZE):
        yield order[start:start + BATCH_SIZE][rank::world_size]


def progress_line(step, total, loss, seconds, peak_bytes):
    def clock(value):
        minutes, seconds = divmod(max(0, int(value)), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    rate = seconds / max(step, 1)
    return (f"训练 {step}/{total} ({step / total:.1%}) | 损失 {loss:.4f} | "
            f"{rate:.1f}秒/步 | 已用 {clock(seconds)} | 预计剩余 {clock((total - step) * rate)} | "
            f"显存 {peak_bytes / 1e9:.1f}GB")


def material_order(order, manifest, indices, start, stop):
    """Repeat verified materials only in their existing source/type positions."""
    groups = {manifest["records"][i]["group"] for i in indices}
    if len(groups) != 1 or not indices:
        raise ValueError("material comparison requires one source/type group")
    result, cursor = list(order), 0
    for position in range(start * BATCH_SIZE, stop * BATCH_SIZE):
        if manifest["records"][order[position]]["group"] in groups:
            result[position] = indices[cursor % len(indices)]
            cursor += 1
    return result


def pilot_order(manifest, seed, updates, *, sampling=None, segment=0, paired=False, background=None, passes=2,
                far_updates=None):
    """Source quotas stay fixed; a new segment gets its own reproducible permutation."""
    if manifest.get("version") == SOURCE_VERSION:
        if passes is not None or updates < 1 or segment:
            raise ValueError("source role quotas require an explicit update budget and segment zero")
        return source_order(manifest["records"], seed, updates, far_updates)
    if manifest.get("version") in (NATIVE_VERSION, NDP_VERSION):
        order = []
        for epoch in range(passes * segment, passes * segment + passes):
            order.extend(epoch_order(len(manifest["records"]), seed, 1, epoch))
        if updates != math.ceil(len(order) / BATCH_SIZE):
            raise ValueError("native training must finish the configured complete data passes")
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
        if row["group"] == "control_nuscenes":
            counts["inserted_normal"] = counts.get("inserted_normal", 0) + int(row.get("inserted_points", 0))
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


def continuation_factor(step, total):
    """One fixed, un-warmed cosine segment: 10% to 1% of the original peak."""
    if not 1 <= step <= total or total < 2:
        raise ValueError("continuation step outside the fixed execution budget")
    return .01 + .09 * .5 * (1 + math.cos(math.pi * (step - 1) / (total - 1)))


def local_mask(sample, local):
    mask = torch.isin(sample["slots"], torch.as_tensor(local["slots"], device=sample["slots"].device))
    if int(mask.sum()) != len(local["slots"]) or not bool((sample["targets"][mask] == 0).all()):
        raise ValueError("local supervision must address the same verified normal returns")
    return mask


def forward_loss(model, samples, counts, *, rank_weight=0., rank_seed=0, auc_weight=.1, fpr95_weight=.1,
                 local=None, local_count=0, record_points=False):
    """Legacy per-pair objectives; the complete model uses cached_backward."""
    predictions = [model(sample) for sample in samples]
    prediction = torch.cat(predictions)
    targets = torch.cat([sample["targets"] for sample in samples])
    bce = balanced_loss(prediction, targets, counts)
    if rank_weight:
        rank, details = ranking_loss(prediction, targets, rank_seed,
                                     auc_weight=auc_weight, fpr95_weight=fpr95_weight)
    else:
        rank, details = bce * 0, {}
    loss = bce + rank_weight * rank
    if record_points:
        details["point_scores"] = [value.detach() for value in predictions]
    if local is not None:
        chosen = [p[local_mask(s, local)] for s, p in zip(samples, predictions) if int(s["index"]) == local["index"]]
        # Normalize across all occurrences in the effective batch, not per microbatch.
        extra = local["weight"] * F.softplus(torch.cat(chosen).float()).sum() / local_count if chosen else bce * 0
        loss = loss + extra
        details.update(local=extra.detach(), local_scores=[p.detach().cpu().tolist() for p in chosen])
    return loss, dict(bce=bce.detach(), **details)


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


def cached_backward(model, samples, device, *, rank_weight, rank_seed, microbatch=2, scaler=None):
    """Joint effective-batch detection gradient with fixed microbatch forwards.

    Replay uses the exact same inputs/RNG and pre-forward buffers. Running BN
    statistics advance only during the first logical pass, never twice.
    """
    def buffers():
        return {name: value.detach().clone() for name, value in model.named_buffers()}

    def restore_buffers(saved):
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(saved[name])

    cached, scores, labels, controls = [], [], [], []
    for begin in range(0, len(samples), microbatch):
        state = dict(rng=rng_state(device), buffers=buffers(), begin=begin)
        pair = [to_device(sample, device) for sample in samples[begin:begin + microbatch]]
        with torch.no_grad(), autocast(device):
            query_indices = [(sample["targets"] >= 0).nonzero().flatten() for sample in pair]
            prediction = [model(sample, query_indices=indices) for sample, indices in zip(pair, query_indices)]
        scores.extend(prediction)
        labels.extend(sample["targets"][indices] for sample, indices in zip(pair, query_indices))
        controls.extend(sample["control_mask"][indices] for sample, indices in zip(pair, query_indices))
        cached.append(state)
        del pair, prediction
    final_rng, final_buffers = rng_state(device), buffers()
    leaf = torch.cat(scores).detach().requires_grad_()
    targets = torch.cat(labels)
    counts = torch.stack([(targets == label).sum() for label in (0, 1)])
    bce = balanced_loss(leaf, targets, counts, control_mask=torch.cat(controls))
    rank, details = ranking_loss(leaf, targets, rank_seed) if rank_weight else (bce * 0, {})
    detection = bce + rank_weight * rank
    if not torch.isfinite(detection):
        raise FloatingPointError("nonfinite effective-batch detection loss")
    if details.get("recall") is not None and abs(float(details["recall"]) - .95) > 2e-6:
        raise FloatingPointError("effective-batch soft recall did not reach 0.95")
    gradient, = torch.autograd.grad(detection, leaf)
    # Equal weight per eligible normal scan, then scale/block means inside it.
    references = [sample.get("normal_reference", sample) for sample in samples]
    normal_count = sum(bool(sample.get("normal_training", False)) and bool((sample["targets"] == 0).any())
                       for sample in references)
    auxiliary = leaf.new_zeros(())
    offset, replay_error = 0, 0.
    try:
        for saved in cached:
            restore_rng(saved["rng"], device)
            restore_buffers(saved["buffers"])
            begin = saved["begin"]
            pair = [to_device(sample, device) for sample in samples[begin:begin + microbatch]]
            # The backbone already forwards each scan independently. Applying the
            # cached chain rule immediately releases that scan's activations;
            # neither the eight-scan ranking pool nor BN's forward batch changes.
            for sample in pair:
                with autocast(device):
                    query_indices = (sample["targets"] >= 0).nonzero().flatten()
                    prediction, normal = model(sample, normal_loss=True, query_indices=query_indices)
                stop = offset + len(prediction)
                reference = leaf.detach()[offset:stop]
                error = (prediction.detach() - reference).abs().max()
                replay_error = max(replay_error, float(error))
                # Check the bound on every update, not just the implementation fixture.
                if not torch.allclose(prediction.detach(), reference, atol=1e-5, rtol=1e-5):
                    raise FloatingPointError(f"gradient-cache replay changed logits (max error {float(error):.8g})")
                normal = normal / max(normal_count, 1)
                surrogate = (prediction * gradient[offset:stop]).sum() + .1 * normal
                if not torch.isfinite(surrogate):
                    raise FloatingPointError("nonfinite gradient-cache replay loss")
                (scaler.scale(surrogate) if scaler is not None else surrogate).backward()
                auxiliary += normal.detach()
                offset = stop
                del prediction, normal, surrogate
            del pair
    finally:
        restore_buffers(final_buffers)
        restore_rng(final_rng, device)
    return detection.detach() + .1 * auxiliary, dict(bce=bce.detach(), **details,
        normal_nll=auxiliary, normal_scans=normal_count, replay_max_abs=replay_error,
        ranking_scans=len(samples), point_scores=[value.detach() for value in scores])


def point_record(directory, train, order, model, device, updates, resume=False, identity_source=None):
    """Retain exact scores and point identities; large activations remain reproducible at saved checkpoints."""
    input_offsets = np.r_[0, np.cumsum([row["points"] for row in train["records"]])]
    visit_offsets = np.r_[0, np.cumsum([train["records"][i]["points"] for i in order])]
    buffers = list(model.named_buffers())
    buffer_offsets = np.r_[0, np.cumsum([value.numel() for _, value in buffers])]
    random = rng_state(device)
    random_sizes = {key: random[key].numel() for key in ("torch", "cuda") if random[key] is not None}
    definition = dict(train_manifest=train["sha256"], sampling="sampling.json/order",
        inputs="Existing manifest sources are retained without duplicating raw scans; points.npy gives each record's actual-return slots and supervision",
        input_offsets=input_offsets.tolist(), visit_offsets=visit_offsets.tolist(),
        scores="train.npy: float32 actual training-forward logits, ordered by scan visit then original return slot; includes ignored context points",
        buffers=[dict(name=name, shape=list(value.shape), start=int(buffer_offsets[i]), stop=int(buffer_offsets[i+1]))
                 for i, (name, value) in enumerate(buffers)], rng_sizes=random_sizes,
        states="buffers.npy/rng.npy row 0 precedes training; row u follows update u. Full parameter/optimizer states are retained at step*.pt",
        valid_prefix="Use the last resumable checkpoint's planned_updates after interruption; later file entries are not completed evidence",
        feature_scope="All logits and point identities are retained. Full per-point activations at every update are not stored; checkpoints support later fixed-state layer inspection, not bitwise reconstruction of every unsaved intermediate training state.")
    path = directory / "record.json"
    if identity_source is not None:
        previous = json.loads((identity_source / "record.json").read_text())
        if previous["train_manifest"] != train["sha256"] or previous["input_offsets"] != input_offsets.tolist():
            raise ValueError("shared point identities must describe exactly the same training inputs")
        definition["identity_source"] = str(identity_source.resolve())
        if not (directory / "points.npy").exists():
            os.link(identity_source / "points.npy", directory / "points.npy")
    if resume:
        if json.loads(path.read_text()) != definition:
            raise ValueError("recorded point population or state layout changed")
    else:
        write_json(path, definition)
    specifications = dict(points=((int(input_offsets[-1]),), np.dtype([("slot", "<u4"), ("target", "i1")])),
        train=((int(visit_offsets[-1]),), np.float32), buffers=((updates + 1, int(buffer_offsets[-1])), np.float32),
        rng=((updates + 1, sum(random_sizes.values())), np.uint8))
    arrays = {name: np.lib.format.open_memmap(directory / f"{name}.npy",
              mode="r" if name == "points" and identity_source else "r+" if resume else "w+", dtype=dtype, shape=shape)
              for name, (shape, dtype) in specifications.items()}
    if any(arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype)
           for name, (shape, dtype) in specifications.items()):
        raise ValueError("record arrays differ from their declared population")
    return dict(arrays=arrays, input_offsets=input_offsets, visit_offsets=visit_offsets, buffers=buffers, device=device)


def record_state(record, step):
    """Reading buffers and generators must not change model state or consume random draws."""
    record["arrays"]["buffers"][step] = torch.cat([value.detach().reshape(-1).float()
        for _, value in record["buffers"]]).cpu().numpy()
    random = rng_state(record["device"])
    record["arrays"]["rng"][step] = torch.cat([random[key].cpu() for key in ("torch", "cuda")
                                              if random[key] is not None]).numpy()


def record_scan(record, sample, scores, visit):
    """The scan's raw slots identify all supervised and ignored model outputs."""
    index = int(sample["index"])
    begin, end = record["input_offsets"][index:index + 2]
    start, stop = record["visit_offsets"][visit:visit + 2]
    if end - begin != len(scores) or stop - start != len(scores):
        raise ValueError("recorded logits differ from their input point population")
    identities = record["arrays"]["points"][begin:end]
    for key, source in (("slot", "slots"), ("target", "targets")):
        values = sample[source].cpu().numpy()
        if identities.flags.writeable:
            identities[key] = values
        elif not np.array_equal(identities[key], values):
            raise ValueError("continued training changed a raw point identity or label")
    record["arrays"]["train"][start:stop] = scores.float().cpu().numpy()


@torch.no_grad()
def observe_local(model, sample, local, device):
    """Compare parameter updates with fixed inference, without consuming training RNG or updating BN."""
    state = rng_state(device)
    modes = [(module, module.training) for module in model.modules()]
    buffers = [value.clone() for value in model.buffers()]
    try:
        seed_all(0)
        model.eval()
        with autocast(device):
            scores = model(sample)[local_mask(sample, local)].float().cpu()
        if any(not torch.equal(a, b) for a, b in zip(buffers, model.buffers())):
            raise ValueError("diagnostic inference changed a model buffer")
        return dict(scores=scores.tolist(), mean_BCE=float(F.softplus(scores).mean()),
                    wrong_at_zero=int((scores >= 0).sum()))
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(state, device)


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


def normal_numerics_repair(previous, current):
    """Accept only the identified log-CDF backward repair of the source-only run."""
    affected = {
        "src/normal.py": "cb40bb3bd78e52bfae02c9a4c4181214e8d1672eccc7561f28935db45e093cc3",
        "src/train.py": "5561c185c8e4f2b79fd2838fbf9c3d95fb17c744d42a3e3311b838ef136e8974",
    }
    if (previous.get("recipe") != "field" or previous.get("data_version") != SOURCE_VERSION
            or "normal_numerics" in previous
            or current.get("normal_numerics") != "stable_logcdf_backward"):
        return False
    files = current.get("code", {}).get("files", {})
    if any(previous.get("code", {}).get("files", {}).get(name) != digest or name not in files
           for name, digest in affected.items()):
        return False
    repaired = copy.deepcopy(previous)
    repaired["normal_numerics"] = current["normal_numerics"]
    for name in affected:
        repaired["code"]["files"][name] = files[name]
    # The data, optimizer, schedule, dependencies and every other source file stay exact.
    return identity(repaired) == identity(current)


def configuration(train, val, device, world_size, *, updates=None, initial=None,
                  eval_every=None, recipe="mixed", optimizer_state="reset", segment=0, objective="bce", branch=None,
                  hard_pool=None, material_indices=None, baseline_eval=None, passes=None, seed=0):
    if branch:
        parent = torch.load(initial, map_location="cpu", weights_only=False)
        previous = parent["config"]
        mining = branch in ("control", "hard")
        material = branch in ("material-control", "material", "local")
        continuation = mining or material
        start_update = 1000 if continuation else 500
        budget = 40 if material else 500
        if (recipe != "native" or objective != "metrics" or updates != budget or eval_every != budget
                or world_size != 1 or parent["successful_updates"] != start_update or parent["planned_updates"] != start_update
                or parent["next_batch"] != 0 or parent["epoch"] != start_update // 500 or not parent["complete"]
                or previous["train_manifest"] != train["sha256"] or previous["val_manifest"] != val["sha256"]
                or previous["recipe"] != "native" or previous["objective"] != "metrics"
                or {int(s["step"]) for s in parent["optimizer"]["state"].values()} != {start_update}
                or (continuation and (previous.get("branch") != "lr" or previous.get("lr_scale") != .3))):
            raise ValueError("diagnostic branch requires its specified complete parent training state")
        result = copy.deepcopy(previous)
        result.update(code=code_record(), initial=str(initial.resolve()), initial_sha256=file_sha256(initial),
                      parent_configuration=identity(previous), branch=branch, start_update=start_update,
                      updates=start_update+budget, additional_updates=budget, schedule_updates=previous.get("schedule_updates",previous["updates"]),
                      epochs=None, scan_visits=BATCH_SIZE*budget, eval_every=eval_every,
                      optimizer_state="inherit", lr_scale=.3 if continuation or branch == "lr" else 1.,
                      validation=f"inherited update {start_update} plus one full validation at update {start_update+budget}",
                      reference_sampling=str(initial.parent / "sampling.json"))
        if material:
            baseline = json.loads(Path(baseline_eval).read_text())
            if (baseline["checkpoint_sha256"] != result["initial_sha256"]
                    or baseline["manifest_sha256"] != val["sha256"] or not material_indices
                    or baseline["attention_source"] != file_sha256(ROOT / "vendor/litept/model.py")
                    or any(train["records"][i]["group"] != "normal_stu" for i in material_indices)):
                raise ValueError("material diagnosis requires corrected C and verified STU normal materials")
            result.update(material_indices=material_indices, initial_validation=baseline,
                          baseline_eval=str(baseline_eval))
            if branch == "local":
                if material_indices != [6006]:
                    raise ValueError("the local diagnostic is specified only for record 6006")
                data = Scans(train)
                sample = data[6006]
                chosen = (sample["targets"] == 0) & np.all(np.floor(sample["xyzi"][:, :3] / .75) == [13, 2, -3], axis=1)
                slots = sample["slots"][chosen]
                if len(slots) != 24 or not np.all(data._source(train["records"][6006]["frame"]).semantic[slots] == 40):
                    raise ValueError("the preselected 24-point road patch changed")
                result["local"] = dict(index=6006, slots=slots.tolist(), cell=[13, 2, -3], cell_size=.75,
                    weight=1., label=0, normalization="mean over all selected point occurrences in the effective batch; zero when absent",
                    control="results/train/native/transfer/material/0/conditional")
        if mining:
            pool = json.loads(Path(hard_pool).read_text())
            if pool["train_manifest"] != train["sha256"] or pool["checkpoint_sha256"] != result["initial_sha256"]:
                raise ValueError("hard pool must be scored by the same C checkpoint on the current training manifest")
            result.update(hard_pool=str(Path(hard_pool).resolve()),hard_pool_sha256=file_sha256(hard_pool))
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
    if recipe == "field":
        bounded = passes is None and updates is not None and updates > 0 and train["version"] == SOURCE_VERSION
        if train["version"] == SOURCE_VERSION and not bounded:
            raise ValueError("source role quotas require --updates; complete-pass epochs do not define these quotas")
        if (train["version"] not in (NATIVE_VERSION, NDP_VERSION, SOURCE_VERSION) or world_size != 1
                or not (bounded or passes is not None and passes > 0 and updates is None)
                or file_sha256(initial) != WEIGHTS_SHA256 or optimizer_state != "reset"):
            raise ValueError("field training requires fixed data, public weights, an explicit budget and one GPU")
        visits = updates * BATCH_SIZE if bounded else passes * len(train["records"])
        updates = math.ceil(visits / BATCH_SIZE)
        result.update(version=train["version"], model="field", recipe="field", epochs=passes, updates=updates,
            normal_numerics="stable_logcdf_backward", architecture="hypothesis_readout",
            scan_visits=visits, initial=str(initial.resolve()), initial_sha256=WEIGHTS_SHA256,
            optimizer_state="reset", peak_lr=PEAK_LR[0], microbatch=2, objective="metrics", precision="torch.float32",
            eval_every=eval_every or updates, sampling_segment=0,
            sampling="complete shuffled passes; last effective batch may contain fewer than eight scans",
            loss=dict(bce="group means over all effective-batch supervised points",
                group_weights=dict(anomaly=.5, inserted_normal=.25, remaining_normal=.25),
                missing_groups="renormalize present weights; source quotas retain all three groups",
                ranking_scope="effective batch",
                tau=1., ap_weight=1., auc_weight=.1, fpr95_weight=.1, normal_weight=.1,
                positive_anchors=256, positive_references="all", normal_top=512, normal_random=3584,
                normal_weights="top: 1; rest: population / sample; normalize by full normal count",
                threshold_recall=.95, threshold_gradient="implicit", bce_only_fraction=.1, ramp_end_fraction=.2,
                rank_seed="seed * 100000000 + update * 8; independent generator",
                normal_reduction="mean of point-normalized block joint NLLs, then scales, then eligible normal scans"),
            normal_field=dict(scales=SCALES, hypotheses=HYPOTHESES, kernels=KERNELS,
                ray_chunk=RAY_CHUNK, range_m=[LOWER, UPPER], ray_origin="scan reference origin approximation",
                context_projection="once per independent block and layer, then gather the same 24 non-self neighbors",
                compatibility="complete-hypothesis log density and context log prior; shared 2-16-64 map and hypothesis attention",
                precision="FP32 field, ray density, compatibility and output head", normal_pretraining=False),
            model_precision="FP32 including backbone; fixed kernel-offset sparse convolution reduction; at most two full attention patches per chunk, recomputed in backward",
            gradient_cache=dict(microbatch=2, score_atol=1e-5, score_rtol=1e-5,
                forwards="one score pass plus one replay; per-scan activation release; BN advances once",
                queries="supervised points only; full observed point context retained in both passes"),
            validation="explicit development manifest; checkpoint selection permitted; not an independent final test")
        if train["version"] == SOURCE_VERSION:
            selected = pilot_order(train, seed, updates, passes=None)
            paired_visits = sum(bool(train["records"][i].get("delta")) for i in selected)
            result.update(data_recipe=train.get("recipe", {}), normal_source_visits=visits,
                paired_normal_source_visits=paired_visits, shared_normal_source_visits=visits - paired_visits,
                sampling="fixed 3 dense anomaly / 1 sparse anomaly / 2 observed normal control / 2 original; complete role-pool cycles before reuse",
                source_sampling=dict(quota=dict(dense_anomaly=3, sparse_anomaly=1, inserted_normal=2, original=2),
                    dense="at least five supervised anomaly points", sparse="one to four supervised anomaly points",
                    control="at least one inserted normal point within 2.5-50 m",
                    order="distance bands follow role-pool frame proportions throughout each cycle; interleave donor, scene, point-count bin and placement within each band; temporally separated frames first",
                    distance_allocation="merge evenly spaced per-band visits (j+0.5)/band_size; source training capacities only",
                    range="anomaly: supervised-point median; control: median band of inserted-point histogram",
                    repeats="only after all records in the same role pool were visited; never within one effective batch"),
                source_groups=source_counts(train, selected), data_passes_equivalent=visits / len(train["records"]),
                data_roles=dict(training="raw nuScenes scans, synthetic road obstacles and observed normal placement controls",
                    positive="explicit synthetic obstacle returns; original debris and void remain ignored",
                    normal_auxiliary="unchanged nuScenes source; paired original for every modified scan, shared prediction for unmodified scans",
                    development="explicit development manifest; official pooled-point metrics"))
        if train["version"] == NDP_VERSION:
            result.update(data_recipe=train["recipe"], normal_source_visits=visits,
                comparison=dict(reference="NDP-EE, arXiv:2604.09232v2 Table 1",
                    published_val_percent=dict(AP=74.24, FPR95=1.43, AUROC=99.53),
                    shared="206 and fixed public Perlin generation; official STU validation protocol",
                    differences="model/loss, nuScenes versus SemanticKITTI/Panoptic-CUDAL pretraining, learning-rate schedule, no NDP coordinate/instance augmentation or soft-void loss",
                    checkpoint="one selected checkpoint for all three metrics; no hidden-test experiment"))
        return result
    if recipe == "native":
        visits = 2 * len(train["records"])
        if optimizer_state == "inherit":
            parent = torch.load(initial, map_location="cpu", weights_only=False)
            previous = parent["config"]
            inherited = parent["successful_updates"] + previous.get("parent_updates", 0)
            if (not parent["complete"] or parent["overflows"] or previous.get("branch")
                    or previous["train_manifest"] != train["sha256"] or previous["val_manifest"] != val["sha256"]
                    or previous.get("recipe") != "native" or previous["objective"] != objective
                    or updates != math.ceil(visits / BATCH_SIZE) or eval_every != updates
                    or segment != previous["sampling_segment"] + 1 or world_size != 1
                    or {int(s["step"]) for s in parent["optimizer"]["state"].values()} != {inherited}):
                raise ValueError("native continuation requires a completed uniform parent and two full passes")
            # Analysis code may evolve; the data reader, model and numerical kernels must not.
            for name, digest in previous["code"]["files"].items():
                if name in ("src/data.py", "src/model.py") or name.startswith("vendor/"):
                    if result["code"]["files"][name] != digest:
                        raise ValueError(f"continuation changed model or data mathematics: {name}")
            result = copy.deepcopy(previous)
            result.pop("recording", None)
            result.update(code=code_record(), initial=str(initial.resolve()), initial_sha256=file_sha256(initial),
                parent_updates=inherited, parent_configuration=identity(previous), updates=updates,
                additional_updates=updates, scan_visits=visits, sampling_segment=segment,
                data_passes=[2 * segment + 1, 2 * segment + 2], optimizer_state="inherit", eval_every=eval_every,
                continuation_schedule=dict(initial_fraction=.1, final_fraction=.01, warmup=False,
                    definition="cosine over this segment only; multiply each inherited original peak_lr"),
                validation="internal development before/after continuation; val19 only at the fixed endpoint",
                initial_validation=dict(metrics=parent["final_metrics"], manifest_sha256=val["sha256"],
                                        inherited=True, source=str(initial.resolve())))
            return result
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


def hard_order(order, train, pool, start_update, updates):
    """Replace exactly one rotating pair per batch; keep source/type and six positions."""
    rng = np.random.default_rng(0)
    queues, used = {}, defaultdict(int)
    for group,indices in pool["groups"].items():
        if not indices or len(indices)!=len(set(indices)) or any(train["records"][i]["group"]!=group for i in indices):
            raise ValueError("invalid training-only hard pool")
        queues[group] = rng.permutation(indices).tolist()
    result = list(order)
    for step in range(start_update,updates):
        batch = order[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        for offset in ((step-start_update)%4*2, (step-start_update)%4*2+1):
            group = train["records"][batch[offset]]["group"]
            candidates = queues[group]
            # Avoid unchanged replacements and duplicate scans within the same batch.
            for _ in range(len(candidates)):
                chosen = candidates[used[group]%len(candidates)]
                used[group] += 1
                if chosen not in result[step*BATCH_SIZE:(step+1)*BATCH_SIZE]:break
            else:
                raise ValueError("hard pool too small to provide distinct replacements")
            result[step*BATCH_SIZE+offset] = chosen
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


def validate_all(model, val, device, workers, score_path=None, record_points=False):
    rank, _ = rank_info()
    sync_buffers(model)
    result = None
    if rank == 0:
        try:
            result = dict(ok=True, result=evaluate(model, val, device, workers, score_path=score_path,
                                                  **(dict(record_points=True) if record_points else {})))
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
               parent_updates=config.get("parent_updates", 0),
               cumulative_updates=state["successful_updates"] + config.get("parent_updates", 0),
               gradient_steps=state.get("gradient_steps"), clipped_steps=state.get("clipped_steps"),
               gradient_norm_sum=state.get("gradient_norm_sum"), gradient_norm_max=state.get("gradient_norm_max"),
               training_seconds=state.get("training_seconds"), validation_seconds=state.get("validation_seconds"),
               planned_updates=state["planned_updates"], successful_updates=state["successful_updates"],
               overflows=state["overflows"], parameters=state["parameters"],
               train_manifest_sha256=config.get("train_manifest"), val_manifest_sha256=config.get("val_manifest"),
               validation_improved_from_epoch0=(state["best_epoch"] > 0) if state["stage"] == 2 else None))


def train_stage(args, train, val, seed, method, device, config):
    global STOP
    if method == "field" and config.get("recording"):
        raise ValueError("field supervised-query training cannot record full-point scores; ignored scores are not computed")
    if seed != 0 and not config.get("decoder_comparison") and method != "field":
        raise ValueError("F240-R2 fixes the sole experiment seed to 0")
    rank, world_size = rank_info()
    native = method == "conditional" or method in (*RELATION_MODES, *NORMAL_MODES)
    continuation = bool(config.get("continuation_schedule"))
    branch = config.get("branch")
    pilot = method == "pilot" or native
    stage = 1 if method == "base" or native else 2
    parent = torch.load(args.initial, map_location="cpu", weights_only=False) if (pilot and not native) or branch or continuation else None
    mode = method if native else parent["mode"] if pilot else "base" if method in ("base", "continue") else method
    directory = args.output / str(seed) / method
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    last, best_path = directory / "last.pt", directory / "best.pt"
    seed_all(seed)
    model = Segmentor(mode, **({"recompute": False} if config.get("retain_activations") else {}))
    load_record = None
    numerical_repair = None
    resume = last.exists()
    if resume:
        if not args.resume:
            raise ValueError(f"{last} exists; use --resume to continue its exact state")
        saved = torch.load(last, map_location="cpu", weights_only=False)
        exact = identity(saved["config"]) == identity(config)
        repaired = (not exact and not saved["complete"] and method == "field"
                    and normal_numerics_repair(saved["config"], config))
        if not (exact or repaired) or saved["seed"] != seed or saved["method"] != method:
            raise ValueError("resume configuration, code, dependencies or data differ")
        if repaired:
            numerical_repair = dict(reason="stable_logcdf_backward", resume_step=saved["planned_updates"],
                previous_configuration=identity(saved["config"]), previous_code=saved["config"]["code"],
                current_code=config["code"])
        if saved["complete"]:
            if rank == 0:
                write_result(directory, saved, config)
                print(f"completed: seed={seed} method={method}", flush=True)
            return True
        model.load_state_dict(saved["model"], strict=True)
    elif branch or continuation:
        model.load_state_dict(parent["model"], strict=True)
        load_record = dict(parent=str(args.initial), parent_sha256=config["initial_sha256"],
                           inherited_full_validation=config.get("initial_validation") or parent["validation"],
                           inherited_optimizer_updates=config.get("parent_updates", config.get("start_update")),
                           inherited_rng=True, inherited_sampling_offset=config.get("start_update", 0)*BATCH_SIZE)
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
    scaler = torch.amp.GradScaler("cuda", enabled=method != "field" and precision(device) == torch.float16)
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
        if numerical_repair and rank == 0:
            record = json.loads((directory / "config.json").read_text())
            record.update(configuration=config, numerical_repair=numerical_repair)
            write_json(directory / "config.json", record)
            print(f"数值修复续训：已恢复第 {state['planned_updates']} 步的模型、优化器和随机状态；"
                  "后续使用稳定的正态分布对数累积概率梯度。", flush=True)
        del saved
    else:
        seed_all(seed + stage * 1000 + rank * 10000)
        if branch:
            optimizer.load_state_dict(parent["optimizer"])
            scaler.load_state_dict(parent["scaler"])
            start_update = config["start_update"]
            initial_validation = config.get("initial_validation", parent["validation"])
            state.update(epoch=start_update//steps_per_epoch, planned_updates=start_update, successful_updates=start_update,
                         best_epoch=start_update//steps_per_epoch, best_metrics=initial_validation["metrics"])
            restore_rng(parent["rng"][rank], device)
            saved = capture(model, optimizer, scaler, state, config, device,
                            selected=True, validation=initial_validation)
            if rank == 0:
                atomic_save(best_path, saved)
                write_json(directory / f"epoch{state['epoch']}.json", dict(inherited=True, **initial_validation))
            del saved
        elif pilot and config.get("optimizer_state") == "inherit":
            if (parent["config"]["world_size"] != world_size or parent["config"]["sampling"] != config["sampling"]
                    or parent["config"]["train_manifest"] != train["sha256"]):
                raise ValueError("optimizer continuation requires the same data, recipe and world size")
            optimizer.load_state_dict(parent["optimizer"])
            scaler.load_state_dict(parent["scaler"])
            restore_rng(parent["rng"][rank], device)
            load_record["inherited_optimizer_updates"] = config.get("parent_updates", parent["successful_updates"])
            if continuation:
                state.update(best_metrics=config["initial_validation"]["metrics"], best_epoch=0)
                atomic_save(best_path, capture(model, optimizer, scaler, state, config, device,
                                              selected=True, validation=config["initial_validation"]))
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
    dataset = PreparedScans(train, relations=method in RELATION_MODES, normal=method in NORMAL_MODES)
    if pilot:
        del parent
        far_updates = []
        full_order = pilot_order(train, seed, schedule_total, sampling=config.get("sampling"),
                                 segment=config.get("sampling_segment", 0), paired=config.get("recipe") == "paired",
                                 background=config.get("background_reference"), passes=config.get("epochs", 2) if method == "field" else 2,
                                 far_updates=far_updates)
        if branch and full_order != json.loads(Path(config["reference_sampling"]).read_text())["order"]:
            raise ValueError("branch scan order differs from the original recorded stream")
        reference_order = full_order
        if branch == "hard":
            if file_sha256(config["hard_pool"]) != config["hard_pool_sha256"]:
                raise ValueError("hard pool changed after configuration")
            full_order = hard_order(full_order,train,json.loads(Path(config["hard_pool"]).read_text()),
                                    config["start_update"],total)
        elif branch in ("material", "local"):
            full_order = material_order(full_order, train, config["material_indices"], config["start_update"], total)
        if branch == "local" and full_order != json.loads((Path(config["local"]["control"]) / "sampling.json").read_text())["order"]:
            raise ValueError("local weighting must reuse every material-control input position")
        if native and rank == 0:
            executed = full_order[config.get("start_update", 0) * BATCH_SIZE:total * BATCH_SIZE]
            distant = (dict(far_positive_updates=far_updates,
                            far_positive_fraction=len(far_updates) / schedule_total,
                            sampling=config.get("source_sampling"),
                            dense_anomaly_visits=sum(train["records"][i]["anomaly"] >= 5 for i in executed),
                            sparse_anomaly_visits=sum(0 < train["records"][i]["anomaly"] < 5 for i in executed))
                       if train["version"] == SOURCE_VERSION else {})
            write_json(directory / "sampling.json", dict(train_manifest=train["sha256"], order=full_order,
                sources=source_counts(train, executed), distinct_records=len(set(executed)),
                executed_order=executed, start_update=config.get("start_update", 0),
                replaced_visits=sum(a!=b for a,b in zip(reference_order,full_order)),
                passes=None if branch else config.get("epochs", 2),
                scans_per_pass=None if train["version"] == SOURCE_VERSION else len(train["records"]), visits=len(executed),
                **distant))
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
    local = config.get("local")
    recording = None
    if config.get("recording"):
        source = args.initial.parent if continuation else None
        recording = point_record(directory, train, full_order, model, device, total, resume, identity_source=source)
        if source is not None and not (directory / "val_points.npy").exists():
            os.link(source / "val_points.npy", directory / "val_points.npy")
        if not resume:
            record_state(recording, 0)
            atomic_save(directory / "step0.pt", capture(model, optimizer, scaler, state, config, device, selected=False))
    if local:
        local_sample = to_device(dataset[local["index"]], device)
        trace_path = directory / "local.json"
        trace = json.loads(trace_path.read_text()) if resume and trace_path.exists() else dict(
            definition=local, initial=observe_local(model, local_sample, local, device), updates=[],
            interpretation="Before/after optimizer scores share the post-forward BN buffers; changes between consecutive updates also include training BN updates. Training-mode scores are recorded separately.")
        write_json(trace_path, trace)
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
                factor = continuation_factor(step, total) if continuation else lr_factor(step, schedule_total)
                group["lr"] = group["peak_lr"] * factor * config.get("lr_scale", 1.)
            sync_buffers(model)
            loss_sum = torch.zeros((), device=device)
            components, pair_details = dict(bce=0., ap=0., auc=0., fpr95=0.), []
            if local:
                components["local"] = 0.
            local_count = len(local["slots"]) * global_indices.count(local["index"]) if local else 0
            pair_size = config["microbatch"] if native else 1
            pair_count = math.ceil(len(samples) / pair_size)
            ramp = (1. if continuation else ranking_weight(step, schedule_total)) if config.get("objective") == "metrics" else 0.
            if method == "field":
                loss_sum, details = cached_backward(model, samples, device, rank_weight=ramp,
                    rank_seed=seed * 100000000 + step * 8, microbatch=pair_size, scaler=scaler)
                if recording is not None:
                    for offset, (sample, scores) in enumerate(zip(samples, details["point_scores"])):
                        record_scan(recording, sample, scores, (step - 1) * BATCH_SIZE + offset)
                details.pop("point_scores")
                components.update({key: float(details.get(key, 0.)) for key in (*components, "normal_nll")})
                pair_details.append({key: float(value) if isinstance(value, torch.Tensor) else value
                                     for key, value in details.items() if key not in components})
            else:
                for pair_index, begin in enumerate(range(0, len(samples), pair_size)):
                    pair = [to_device(sample, device) for sample in samples[begin:begin + pair_size]]
                    with autocast(device):
                        loss, details = forward_loss(model, pair, counts, rank_weight=ramp / pair_count,
                                                    rank_seed=seed * 100000000 + (step + config.get("parent_updates", 0)) * 8 + pair_index,
                                                    auc_weight=config.get("loss", {}).get("auc_weight", .1),
                                                    fpr95_weight=config.get("loss", {}).get("fpr95_weight", .1),
                                                    local=local, local_count=local_count, record_points=recording is not None)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite segmentation loss")
                    if details.get("recall") is not None and abs(float(details["recall"]) - .95) > 2e-6:
                        raise FloatingPointError("smooth recall threshold did not reach 0.95")
                    scaler.scale(loss).backward()
                    if recording is not None:
                        for offset, (sample, scores) in enumerate(zip(pair, details.pop("point_scores"))):
                            record_scan(recording, sample, scores, (step - 1) * BATCH_SIZE + begin + offset)
                    loss_sum += loss.detach()
                    for key in components:
                        components[key] += float(details.get(key, 0.)) / (1 if key in ("bce", "local") else pair_count)
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
                raise FloatingPointError("nonfinite gradient; no silent skipped update")
            before = scaler.get_scale()
            if local:
                before_update = observe_local(model, local_sample, local, device)
                parameters_before = [p.detach().clone() for p in model.parameters()]
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.is_enabled() and scaler.get_scale() < before
            if local:
                after_update = observe_local(model, local_sample, local, device)
                delta = sum((p.detach() - previous).double().square().sum()
                            for p, previous in zip(model.parameters(), parameters_before)).sqrt()
                del parameters_before
                trace["updates"].append(dict(step=step, indices=global_indices, local_point_visits=local_count,
                    before_optimizer=before_update, after_optimizer=after_update, parameter_delta_l2=float(delta),
                    gradient_norm=float(norm), overflow=bool(overflow), local_loss=components["local"],
                    training_scores=[s for pair in pair_details for s in pair.get("local_scores", [])]))
                write_json(trace_path, trace)
            state["planned_updates"] += 1
            state["successful_updates"] += int(not overflow)
            state["overflows"] += int(overflow)
            state["epoch_loss"] += loss_sum.item()
            state["epoch_points"] = [a + b for a, b in zip(state["epoch_points"], counts.tolist())]
            state["epoch_frames"] += len(global_indices)
            state["next_batch"] = batch_number + 1
            if recording is not None:
                record_state(recording, step)
            state["training_seconds"] += time.perf_counter() - update_start
            need_stop = torch.tensor(int(STOP), device=device)
            if world_size > 1:
                dist.all_reduce(need_stop, op=dist.ReduceOp.MAX)
            check_disk = (method == "field" and step == 1) or step % args.save_every == 0
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
                if recording is not None:
                    for array in recording["arrays"].values():
                        array.flush()
                saved = capture(model, optimizer, scaler, state, config, device, selected=False)
                if rank == 0:
                    atomic_save(last, saved)
                    if recording is not None and step in config["recording"]["checkpoint_updates"]:
                        atomic_save(directory / f"step{step}.pt", saved)
                del saved
            if rank == 0 and (recording is not None or step == 1 or step % 25 == 0 or overflow or need_stop):
                row = dict(event="update", seed=seed, method=method, epoch=epoch + 1,
                           batch=batch_number + 1, batches=steps_per_epoch, loss=loss_sum.item(),
                           normal=int(counts[0]), anomaly=int(counts[1]), planned=step,
                           successful=state["successful_updates"], overflow=bool(overflow),
                           cumulative_update=step + config.get("parent_updates", 0),
                           lr=[g["lr"] for g in optimizer.param_groups],
                           elapsed_seconds=time.perf_counter() - start,
                           objective=config.get("objective", "bce"), ranking_weight=ramp,
                           loss_components=components, pairs=pair_details,
                           gradient_norm=float(norm), clipped_steps=state["clipped_steps"],
                           peak_vram_bytes=torch.cuda.max_memory_allocated(device))
                with (directory / "log.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                if step == 1 or step % 25 == 0 or overflow or need_stop:
                    if not getattr(args, "progress", False):
                        print(json.dumps(row, allow_nan=False), flush=True)
            if rank == 0 and getattr(args, "progress", False):
                print("\r" + progress_line(step, total, loss_sum.item(), state["training_seconds"],
                      torch.cuda.max_memory_allocated(device)), end="\n" if need_stop or step == total else "", flush=True)
            if need_stop:
                if rank == 0:
                    print(f"已停止，恢复检查点：{last}", flush=True)
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
        if rank == 0 and getattr(args, "progress", False):
            print("\n开始完整 nuScenes 验证；上面的剩余时间仅估算训练部分。", flush=True)
        evaluation_scores = directory / f"val{state['planned_updates']}.npy" if recording is not None else (
            args.score_path if (epoch + 1) * steps_per_epoch >= total else None)
        if config.get("decoder_comparison"):
            evaluation_scores = directory / "val.npy"
        result = validate_all(model, val, device, args.workers,
                              score_path=evaluation_scores, **(dict(record_points=True) if recording is not None else {}))
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
            if recording is not None:
                for array in recording["arrays"].values():
                    array.flush()
                atomic_save(directory / f"step{state['planned_updates']}.pt", saved)
            write_json(directory / f"epoch{epoch + 1}.json", report)
            if getattr(args, "progress", False):
                print(f"验证完成：{json.dumps(result['metrics'])}；检查点：{last}", flush=True)
            else:
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
    del saved, model, optimizer, dataset, recording
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
    return True


def preflight(args, train, val, device, config, resources, method="field"):
    """Measure the actual mixed update without changing the initial checkpoint."""
    seed = getattr(args, "seed", 0)
    seed_all(seed)
    dataset = PreparedScans(train, relations=method in RELATION_MODES, normal=method in NORMAL_MODES)
    order = pilot_order(train, seed, args.updates, sampling=config["sampling"],
                          segment=config["sampling_segment"], paired=config.get("recipe") == "paired",
                          background=config.get("background_reference"), passes=config.get("epochs", 2) if method == "field" else 2)
    indices = order[:BATCH_SIZE]
    if config.get("recipe") in ("native", "field"):
        model = Segmentor(method, **({"recompute": False} if config.get("retain_activations") else {})).to(device)
        model.load_pretrained(args.initial)
    else:
        parent = torch.load(args.initial, map_location="cpu", weights_only=False)
        model = Segmentor(parent["mode"]).to(device)
        model.load_state_dict(parent["model"], strict=True)
        del parent
    model.train()
    if method == "field":
        samples = [dataset[index] for index in indices]
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        loss, details = cached_backward(model, samples, device, rank_weight=1., rank_seed=seed * 100000000 + 8)
        details.pop("point_scores")
        torch.cuda.synchronize(device)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite full-model effective-batch gradient")
        result = dict(configuration=config, resources=resources, indices=indices, parameter_updates=0,
            points=[len(sample["xyzi"]) for sample in samples], loss=float(loss),
            details={key: float(value) if isinstance(value, torch.Tensor) else value for key, value in details.items()},
            seconds=time.perf_counter() - start, peak_vram_bytes=torch.cuda.max_memory_allocated(device))
        write_json(args.output / "preflight.json", result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("configuration", "resources")}))
        return
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
                                        rank_seed=8 + pair_index, record_points=bool(config.get("recording")))
        loss.backward()
        if config.get("recording"):
            scores = details.pop("point_scores")
            if any(len(value) != len(sample["xyzi"]) or not torch.isfinite(value).all()
                   for sample, value in zip(pair, scores)):
                raise ValueError("recording did not retain every finite point logit")
            details["recorded_points"] = sum(len(value.float().cpu().numpy()) for value in scores)
            del scores
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
    write_json(args.output / (f"preflight_{method}.json" if config.get("decoder_comparison") else "preflight.json"), result)
    print(json.dumps({key: result[key] for key in
                     ("mixed_batch_seconds", "peak_vram_bytes", "parameter_updates", "scans")}))


NORMAL_SELECTION = "maximum mIoU over the fixed ground-truth-present normal development classes; minimum normal objective breaks exact ties; no anomaly labels"


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
            raise TimeoutError("deadline reached before complete normal development; selected weights are saved; retry with --calibration-only and a new deadline")
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
        raise TimeoutError("deadline reached before normal calibration; selected weights are saved; retry with --calibration-only and a new deadline")
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
                raise TimeoutError("deadline reached before complete normal calibration; selected weights are saved; retry with --calibration-only and a new deadline")
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


class SupportScans:
    """Single-view frozen features; labels select training queries, never inputs."""

    def __init__(self, records, *, development=False, full=False):
        self.records, self.development = records, development
        self.full = full

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from .data import read_normal_record
        from .model import voxelize
        from .normal import support_conditions
        row = self.records[index]
        raw = read_normal_record(row)
        allowed = raw["allowed"]
        valid = np.flatnonzero(allowed.any(1))
        singleton = np.flatnonzero(allowed.sum(1) == 1)
        budget = 2048 if self.development or row["source"] == "nuscenes" else 4096
        rng = np.random.default_rng(np.random.SeedSequence([206, index, int(self.development)]))
        if self.full:
            chosen = valid
        elif self.development:
            chosen = rng.choice(valid, min(len(valid), budget), replace=False)
        else:
            # Reserve fine-label coverage; coarse normal sets remain eligible for
            # the remaining budget without inventing a fine semantic category.
            distance = np.linalg.norm(raw["xyzi"][:, :3], axis=1)
            bands = np.searchsorted([10., 20., 35.], distance, side="right")
            chosen = []
            for category in range(19):
                for band in range(4):
                    candidates = singleton[allowed[singleton, category] & (bands[singleton] == band)]
                    if len(candidates):
                        chosen.extend(rng.choice(candidates, min(len(candidates), budget // 76), replace=False))
            chosen = np.unique(np.asarray(chosen, dtype=np.int64))
            remaining = np.setdiff1d(valid, chosen, assume_unique=True)
            chosen = np.r_[chosen, rng.choice(remaining, min(len(remaining), budget-len(chosen)), replace=False)]
        chosen = np.sort(chosen).astype(np.int64)
        sample = voxelize(raw["xyzi"], official=True)
        sample.update(queries=torch.from_numpy(chosen),
            conditions=torch.from_numpy(support_conditions(raw["xyzi"], chosen, range_only=self.full)),
            allowed=torch.from_numpy(allowed[chosen]),
            semantic=torch.from_numpy(np.where(allowed[chosen].sum(1) == 1,
                allowed[chosen].argmax(1), -1).astype(np.int16)),
            slots=torch.from_numpy(raw["slots"][chosen].astype(np.int64)),
            index=index, source=int(row["source"] != "nuscenes"))
        return sample


def auxiliary_records(path, split="train", samples=1920):
    """Choose frames only by anomaly-count strata within the authorized source split."""
    if split not in ("train", "validation") or type(samples) is not int or samples < 1:
        raise ValueError("auxiliary selection requires train/validation and a positive sample count")
    sequence = 206 if split == "train" else 201
    path = Path(path).resolve()
    pool = json.loads(path.read_text())
    subset = pool.get("splits", {}).get(split, {})
    if pool.get("format") != "stu-frozen-dataset" or subset.get("source_sequence") != sequence:
        raise ValueError(f"auxiliary {split} input must use STU{sequence}")
    buckets = [[] for _ in range(4)]
    total_frames = 0
    for entry in subset["worlds"]:
        folder = (path.parent / entry["path"]).resolve()
        folder.relative_to(path.parent / split)
        world = json.loads((folder / "manifest.json").read_text())
        if world["world_identity"] != entry["world_identity"] or world["source_sequence"] != sequence:
            raise ValueError("auxiliary world and source identities disagree")
        total_frames += len(world["frames"])
        for row in world["frames"]:
            if row["in_range"] < 0:
                raise ValueError("auxiliary anomaly counts must be nonnegative")
            if row["in_range"] == 0:
                continue
            band = int(np.searchsorted([4, 20, 100], row["in_range"]))
            buckets[band].append(dict(world=entry["world_identity"], frame=int(row["frame"]),
                delta=str(folder / "frames" / f"{row['frame']:06d}.npz"),
                recorded_anomalies=int(row["in_range"]), recorded_range=float(row["range"])))
    available = np.array([len(rows) for rows in buckets], dtype=np.int64)
    positive = int(available.sum())
    if not positive:
        raise ValueError("the supplied pool has no in-range anomaly observations")
    if total_frames != subset["samples"]:
        raise ValueError("auxiliary world frame counts disagree with the source pool")
    budget = min(samples, positive)
    quota = np.minimum(available, budget // 4)
    remaining = budget - int(quota.sum())
    # Redistribute unavailable stratum slots, retaining distinct source frames.
    while remaining:
        active = np.flatnonzero(quota < available)
        share = max(1, remaining // len(active))
        for band in active:
            add = min(share, int(available[band]-quota[band]), remaining)
            quota[band] += add
            remaining -= add
    rng = np.random.default_rng(sequence)
    records = [rows[index] for rows, count in zip(buckets, quota)
               for index in rng.choice(len(rows), int(count), replace=False)]
    records.sort(key=lambda r: (r["frame"], r["world"]))
    for row in records:
        row["delta_sha256"] = file_sha256(row["delta"])
    result = dict(pool=str(path), pool_sha256=file_sha256(path), worlds=len(subset["worlds"]),
        split=split, source_sequence=sequence, available_frames=total_frames, records=records,
        requested_frames=samples, selected_frames=len(records), seed=sequence,
        positive_frames=positive, zero_anomaly_frames_excluded=total_frames-positive,
        count_strata=[dict(anomaly_points=name, available=int(n), selected=int(k))
                      for name, n, k in zip(("1-4", "5-20", "21-100", ">100"), available, quota)],
        selection="equal frame quotas for anomaly counts 1-4, 5-20, 21-100 and >100; redistribute shortages; seeded sampling without replacement; no shape or distance balancing; zero-anomaly frames and val19 excluded")
    result["sha256"] = identity(result)
    return result


class AuxiliaryScans:
    """Keep every synthetic anomaly and original semantic labels on normal returns."""

    def __init__(self, manifest, full=False):
        from .data import STUSequence, DATA_ROOT, normal_records
        self.sequence_id = manifest["source_sequence"]
        if self.sequence_id not in (206, 201) or manifest["split"] != ("train" if self.sequence_id == 206 else "validation"):
            raise ValueError("auxiliary source must be train206 or validation201")
        self.records = manifest["records"]
        self.full = full
        self.sequence = (STUSequence(DATA_ROOT) if self.sequence_id == 206 else
                         {row["frame"]: row for row in normal_records("201", development=True)})
        self.previous = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from .data import Frame, STU_NORMAL_SEMANTICS, restore_delta, point_targets
        from .model import voxelize
        from .normal import support_conditions
        row = self.records[index]
        if self.previous is None or self.previous.frame_id != row["frame"]:
            original = self.sequence[row["frame"]]
            self.previous = (original if self.sequence_id == 206 else
                Frame(row["frame"], np.fromfile(original["scan"], dtype="<f4").reshape(-1, 4),
                      np.asarray(original["pose"], dtype=np.float64),
                      np.fromfile(original["label"], dtype="<u4"), sequence_id=201, partition="train"))
        if file_sha256(row["delta"]) != row["delta_sha256"]:
            raise ValueError("supplied auxiliary observation changed after selection")
        restored = restore_delta(row["delta"], self.previous, row["world"])
        slots = restored.return_slots
        raw = dict(xyzi=restored.xyzi[slots], slots=slots, targets=point_targets(restored)[slots])
        labels = restored.semantic[slots]
        semantic = np.full(len(labels), -1, np.int64)
        for label, category in STU_NORMAL_SEMANTICS.items():
            semantic[labels == label] = category
        normal = np.flatnonzero((raw["targets"] == 0) & (semantic >= 0))
        anomaly = np.flatnonzero(raw["targets"] == 1)
        if not len(anomaly) or len(anomaly) != row["recorded_anomalies"]:
            raise ValueError("restored auxiliary anomaly count disagrees with its source record")
        if self.full:
            chosen = np.flatnonzero(raw["targets"] >= 0)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.sequence_id, index, 314]))
            normal = rng.choice(normal, min(512, len(normal)), replace=False)
            chosen = np.sort(np.r_[normal, anomaly]).astype(np.int64)
        semantic[raw["targets"] != 0] = -1
        sample = voxelize(raw["xyzi"], official=True)
        sample.update(queries=torch.from_numpy(chosen), targets=torch.from_numpy(raw["targets"][chosen]),
                      conditions=torch.from_numpy(support_conditions(raw["xyzi"], chosen, range_only=True)),
                      semantic=torch.from_numpy(semantic[chosen]),
                      slots=torch.from_numpy(raw["slots"][chosen].astype(np.int64)),
                      index=index, frame=int(self.records[index]["frame"]))
        return sample


def auxiliary_cache(manifest, output, device, workers):
    """Encode training queries or complete development observations once."""
    from .model import FrozenPerception
    development = manifest.get("source_sequence") == 201
    if manifest.get("source_sequence") not in (206, 201):
        raise ValueError("auxiliary data must use STU206 training or STU201 development")
    output.mkdir(parents=True, exist_ok=True)
    expected = sum(row["recorded_anomalies"] for row in manifest["records"])
    capacity = (350_000 if development else 512) * len(manifest["records"]) + expected
    disk_check(capacity * 1100 + 200_000_000)
    perception = FrozenPerception().to(device).eval()
    loader = DataLoader(AuxiliaryScans(manifest, full=development), batch_size=None,
        num_workers=workers, pin_memory=True, generator=torch.Generator().manual_seed(206),
        **({"prefetch_factor": 1} if workers else {}))
    values = {key: [] for key in ("features", "targets", "semantic", "frame", "slot", "view")}
    paths = []
    anomalies = normals = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for number, sample in enumerate(loader, 1):
            batch = to_device(sample, device)
            encoded = perception.encode(batch, indices=batch["queries"])["features"]
            if not bool(torch.isfinite(encoded).all()):
                raise ValueError("nonfinite auxiliary frozen features")
            anomalies += int((sample["targets"] == 1).sum())
            normals += int((sample["targets"] == 0).sum())
            if development:
                path = output / f"{number-1:04d}.npz"
                np.savez(path, features=encoded.cpu().numpy(), targets=sample["targets"].numpy(),
                    semantic=sample["semantic"].numpy(), slot=sample["slots"].numpy(),
                    conditions=sample["conditions"].numpy(),
                    predicted=perception.seg_head(encoded[:, 180:]).argmax(-1).cpu().numpy())
                paths.append(str(path))
                if number % 20 == 0:
                    disk_check(200_000_000)
            else:
                values["features"].append(encoded.cpu().numpy())
                for key in ("targets", "semantic"):
                    values[key].append(sample[key].numpy())
                values["frame"].append(np.full(len(encoded), sample["frame"], np.int64))
                values["slot"].append(sample["slots"].numpy())
                values["view"].append(np.full(len(encoded), number-1, np.int64))
            if number == 1 or number % 25 == 0 or number == len(manifest["records"]):
                elapsed = time.perf_counter() - started
                print(f"auxiliary features {number}/{len(manifest['records'])}: "
                      f"remaining {(len(manifest['records'])-number)*elapsed/number:.0f}s", flush=True)
            del batch, encoded
    if anomalies != expected:
        raise ValueError("restored auxiliary anomaly counts disagree with the selected pool")
    if development:
        result = dict(frames=paths, manifest=manifest["sha256"], anomalies=anomalies, normals=normals,
            seconds=time.perf_counter()-started, source_sequence=201,
            scope="all valid normal and anomaly returns in selected synthetic development scans, including 1-4-anomaly-point scans; not val19")
        write_json(output / "features.json", result)
        del perception, loader, sample
        torch.cuda.empty_cache()
        return result
    values = {key: np.concatenate(rows) for key, rows in values.items()}
    actual_anomalies = int((values["targets"] == 1).sum())
    if (not actual_anomalies or np.any(values["semantic"][values["targets"] == 1] != -1)
            or np.any(values["semantic"][values["targets"] == 0] < 0)):
        raise ValueError("auxiliary anomaly and normal semantic identities disagree")
    np.savez(output / "auxiliary.npz", **values)
    write_json(output / "auxiliary.json", dict(manifest=manifest["sha256"], frames=len(manifest["records"]),
        anomalies=actual_anomalies, source_recorded_anomalies=expected,
        normals=int((values["targets"] == 0).sum()),
        initial_sha256=WEIGHTS_SHA256, seconds=time.perf_counter()-started,
        role="all actual in-range synthetic anomalies in selected supplied STU206 views plus at most 512 true-semantic normal returns per modified scan"))
    del perception, loader, sample
    torch.cuda.empty_cache()
    return {key: torch.as_tensor(value, device=device) for key, value in values.items()}


@torch.no_grad()
def auxiliary_development(scorer, cache, device):
    """Exact pooled scores over every valid point in the fixed synthetic dev scans."""
    from .evaluate import rank_metrics
    scorer.eval()
    normal, anomaly, eligible_normal, eligible_anomaly = [], [], [], []
    for path in cache["frames"]:
        with np.load(path) as values:
            features = values["features"]
            conditions, predicted, targets = values["conditions"], values["predicted"], values["targets"]
            scores = []
            for start in range(0, len(features), 4096):
                stop = start + 4096
                scores.append(scorer(torch.as_tensor(features[start:stop], device=device),
                    torch.as_tensor(conditions[start:stop], device=device),
                    torch.as_tensor(predicted[start:stop], device=device)).cpu().numpy())
            scores = np.concatenate(scores)
            normal.append(scores[targets == 0])
            anomaly.append(scores[targets == 1])
            if len(anomaly[-1]) >= 5:
                eligible_normal.append(normal[-1])
                eligible_anomaly.append(anomaly[-1])
    return dict(metrics=rank_metrics(np.concatenate(eligible_normal), np.concatenate(eligible_anomaly)),
        all_positive_frame_metrics=rank_metrics(np.concatenate(normal), np.concatenate(anomaly)),
        frames=len(cache["frames"]), normal_points=sum(map(len, normal)),
        anomaly_points=sum(map(len, anomaly)), eligible_frames=len(eligible_normal),
        scope="synthetic STU201; primary metrics require at least five valid anomalies per frame; all_positive_frame_metrics also retain 1-4-point observations")


def auxiliary_loss(scorer, data, positive, negative):
    """The deployed normal-support distances learn rejection and normal semantics together."""
    indices = torch.cat((negative, positive))
    energy = scorer.class_energy(data["features"][indices],
        source=torch.ones_like(indices), frame=data["frame"][indices])
    score = energy.amin(-1) / scorer.temperature
    n = len(negative)
    if len(positive) != n or not n or not bool(torch.isfinite(score).all()):
        raise ValueError("auxiliary ranking requires equal nonempty pairs with finite normal support")
    ranking = F.softplus(1. + score[:n] - score[n:]).mean()
    labels = data["semantic"][negative]
    if bool((labels < 0).any()) or bool((labels >= scorer.classes).any()):
        raise ValueError("auxiliary background requires genuine normal semantic labels")
    # A rare class may have no independent anchor after temporal exclusion.
    valid = torch.isfinite(energy[:n].gather(1, labels[:, None]).squeeze(1))
    normal = F.cross_entropy(-energy[:n][valid] / scorer.temperature, labels[valid]) if bool(valid.any()) else score[:n].sum() * 0
    return .5 * (ranking + normal), ranking, normal, int((~valid).sum())


def support_records():
    """Use normal training sources and official normal development sources only."""
    from .data import normal_records, attach_normal_annotations, NORMAL_ANNOTATIONS
    reviewed = {row["token"] for row in json.loads(NORMAL_ANNOTATIONS.read_text())["records"]}
    sources = []
    for development in (False, True):
        path = Path("results/data/background") / ("val.json" if development else "train.json")
        manifest = json.loads(path.read_text())
        scenes = defaultdict(list)
        for row in manifest["records"]:
            if row.get("source") != "nuscenes" or row.get("delta") or row.get("anomaly", 0):
                raise ValueError("only original normal-supervised source observations are permitted")
            scenes[row["scene"]].append(row)
        selected = {}
        for rows in scenes.values():
            count = 1 if development else 2
            for fraction in np.linspace(0, 1, count+2)[1:-1]:
                row = rows[min(len(rows)-1, int(fraction*len(rows)))]
                selected[row["token"]] = copy.deepcopy(row)
            for row in rows:
                if row["token"] in reviewed or row.get("normal_slots"):
                    selected[row["token"]] = copy.deepcopy(row)
        sources.append(attach_normal_annotations(list(selected.values()), manifest["root"]))
    return (normal_records("206") + sources[0],
            normal_records("201", development=True) + sources[1])


def support_cache(model, records, output, device, workers, *, development=False, deadline=None):
    """Run the immutable backbone once; all density candidates reuse these values."""
    prefix = "development" if development else "training"
    capacities = [2048 if development or row["source"] == "nuscenes" else 4096 for row in records]
    capacity = sum(capacities)
    paths = {key: output / (prefix + "_" + key + ".npy") for key in
             ("features", "conditions", "allowed", "semantic", "source", "frame", "slot")}
    shapes = dict(features=(capacity, 252), conditions=(capacity, 2), allowed=(capacity, 19), semantic=(capacity,),
                  source=(capacity,), frame=(capacity,), slot=(capacity,))
    types = dict(features=np.float32, conditions=np.float32, allowed=np.bool_, semantic=np.int16,
                 source=np.uint8, frame=np.int32, slot=np.uint32)
    if any(path.exists() for path in paths.values()):
        raise ValueError("feature extraction will not overwrite an existing cache")
    arrays = {key: np.lib.format.open_memmap(path, mode="w+", dtype=types[key], shape=shapes[key])
              for key, path in paths.items()}
    loader = DataLoader(SupportScans(records, development=development), batch_size=None,
        num_workers=workers, pin_memory=True, prefetch_factor=1 if workers else None,
        generator=torch.Generator().manual_seed(206))
    cursor, frames = 0, []
    started = time.perf_counter()
    for number, sample in enumerate(loader, 1):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError("normal feature extraction exceeded its reserved budget")
        batch = to_device(sample, device)
        with torch.inference_mode():
            encoded = model.perception.encode(batch, indices=batch["queries"])
        count = len(sample["queries"])
        if not torch.isfinite(encoded["features"]).all():
            raise ValueError("nonfinite frozen features")
        stop = cursor + count
        arrays["features"][cursor:stop] = encoded["features"].cpu().numpy()
        for key in ("conditions", "allowed", "semantic"):
            arrays[key][cursor:stop] = sample[key].numpy()
        arrays["source"][cursor:stop] = sample["source"]
        arrays["frame"][cursor:stop] = sample["index"]
        arrays["slot"][cursor:stop] = sample["slots"].numpy()
        frames.append(dict(index=int(sample["index"]), begin=cursor, end=stop,
                           source=records[int(sample["index"])]["source"],
                           scene=records[int(sample["index"])]["scene"]))
        cursor = stop
        if number % 50 == 0 or number == len(records):
            elapsed = time.perf_counter() - started
            print(f"normal {prefix}: {number}/{len(records)} frames, {cursor:,} points, "
                  f"{elapsed/60:.1f} min, remaining {(len(records)-number)*elapsed/number/60:.1f} min", flush=True)
        if number % 250 == 0:
            disk_check()
    for value in arrays.values():
        value.flush()
    info = dict(count=cursor, capacity=capacity, paths={key:str(path) for key,path in paths.items()},
                frames=frames, seconds=time.perf_counter()-started, labels="normal_candidate_sets_v1")
    write_json(output / (prefix + "_features.json" if development else "training.json"), info)
    del arrays, batch, sample, encoded, loader
    gc.collect()
    torch.cuda.empty_cache()
    return info


def support_indices(cache, *, training, include_coarse=False):
    """Bound fitting memory and preserve target-domain support in each class."""
    labels = np.load(cache["paths"]["semantic"], mmap_mode="r")[:cache["count"]]
    source = np.load(cache["paths"]["source"], mmap_mode="r")[:cache["count"]]
    selected, counts = [], []
    for category in range(19):
        groups = [np.flatnonzero((labels == category) & (source == domain)) for domain in (0, 1)]
        rng = np.random.default_rng(np.random.SeedSequence([206, category, int(training)]))
        if training:
            target_count = min(len(groups[1]), 24000)
            source_count = min(len(groups[0]), 30000 if not target_count else max(512, target_count//4))
        else:
            target_count = min(len(groups[1]), 8192)
            # Source development supplies only classes absent in normal 201.
            source_count = min(len(groups[0]), 8192) if not target_count else 0
        chosen = np.r_[rng.choice(groups[1], target_count, replace=False),
                       rng.choice(groups[0], source_count, replace=False)].astype(np.int64)
        selected.append(np.sort(chosen))
        counts.append(dict(category=category, target_available=len(groups[1]), source_available=len(groups[0]),
                           target_selected=target_count, source_selected=source_count))
    if include_coarse:
        allowed = np.load(cache["paths"]["allowed"], mmap_mode="r")[:cache["count"]]
        coarse = np.flatnonzero(labels < 0)
        codes = allowed[coarse].astype(np.int64) @ (1 << np.arange(19, dtype=np.int64))
        for code in np.unique(codes):
            eligible = coarse[codes == code]
            rng = np.random.default_rng(np.random.SeedSequence([206, int(code), int(training)]))
            chosen = np.sort(rng.choice(eligible, min(len(eligible), 12000), replace=False))
            selected.append(chosen)
            counts.append(dict(allowed_classes=np.flatnonzero(allowed[eligible[0]]).tolist(),
                               available=len(eligible), selected=len(chosen)))
    return selected, counts


def support_standardization(cache, indices):
    """Fixed full-dimensional affine coordinates, computed on training only."""
    features = np.load(cache["paths"]["features"], mmap_mode="r")
    conditions = np.load(cache["paths"]["conditions"], mmap_mode="r")
    count, total, square, ranges, range_square = 0, np.zeros(253), np.zeros(253), 0., 0.
    for group in indices:
        for begin in range(0, len(group), 8192):
            chosen = group[begin:begin+8192]
            x = np.column_stack((features[chosen], conditions[chosen, 1])).astype(np.float64)
            r = conditions[chosen, 0].astype(np.float64)
            count += len(x)
            total += x.sum(0)
            square += np.square(x).sum(0)
            ranges += r.sum()
            range_square += np.square(r).sum()
    if not count:
        raise ValueError("no trustworthy normal supervision")
    mean = total/count
    return dict(location=mean, scale=np.sqrt(np.maximum(square/count-mean*mean, 1e-6)),
                range_location=ranges/count,
                range_scale=float(np.sqrt(max(range_square/count-(ranges/count)**2, 1e-6))))


def fit_support_class(task):
    """Fit each normal class independently, with bounded deterministic CPU work."""
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from threadpoolctl import threadpool_limits
    import warnings
    category, chosen, cache, standard, candidates = task
    threadpool_limits(limits=2)
    torch.set_num_threads(2)
    if len(chosen) < 2:
        return category, [None] * len(candidates)
    features = np.load(cache["paths"]["features"], mmap_mode="r")
    conditions = np.load(cache["paths"]["conditions"], mmap_mode="r")
    values = np.column_stack((features[chosen], conditions[chosen, 1])).astype(np.float64)
    values = (values-standard["location"])/standard["scale"]
    distance = (conditions[chosen, 0].astype(np.float64)-standard["range_location"])/standard["range_scale"]
    basis = np.column_stack((np.ones(len(chosen)), distance, distance**2))
    penalty = np.diag([0., .01, .01]) * len(values)
    coefficients = np.linalg.solve(basis.T@basis+penalty, basis.T@values)
    unconditional = np.zeros_like(coefficients)
    unconditional[0] = values.mean(0)
    fits = []
    for spec in candidates:
        started = time.perf_counter()
        coefficient = coefficients if spec["conditioned"] else unconditional
        residual = values-basis@coefficient
        modes = min(spec["modes"], max(1, len(values)//32))
        mixture = GaussianMixture(n_components=modes, covariance_type="tied",
            reg_covar=spec["regularization"], max_iter=60, tol=.002,
            n_init=1, init_params="k-means++", random_state=206+category)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mixture.fit(residual)
        white = residual @ mixture.precisions_cholesky_
        centers = mixture.means_ @ mixture.precisions_cholesky_
        square = np.maximum(0., np.square(white).sum(1)[:, None] + np.square(centers).sum(1)[None]
                            - 2 * white @ centers.T)
        log_weights = np.log(mixture.weights_)
        def variance_objective(beta):
            # Proper conditional volume penalty prevents variance-only escape.
            bounded = np.tanh(basis @ beta / 4)
            log_variance = 4 * bounded
            scaled = .5 * square * np.exp(-log_variance[:, None])
            component = log_weights - scaled
            log_sum = logsumexp(component, axis=1)
            value = np.mean(.5 * values.shape[1] * log_variance - log_sum) + .005 * (beta @ beta)
            derivative = (.5 * values.shape[1] - (np.exp(component-log_sum[:, None])*scaled).sum(1))
            gradient = basis.T @ (derivative*(1-bounded**2)) / len(values) + .01 * beta
            return value, gradient
        variance_fit = (minimize(variance_objective, np.zeros(3), jac=True, method="L-BFGS-B",
                                options=dict(maxiter=80, ftol=1e-10)) if spec["conditioned"] else None)
        variance = np.zeros(3) if variance_fit is None else variance_fit.x
        if not np.isfinite(variance).all():
            raise ValueError("nonfinite conditional variance fit")
        fits.append(dict(coefficients=coefficient.astype(np.float32),
            log_variance_coefficients=variance.astype(np.float32),
            centers=mixture.means_.astype(np.float32),
            precision_cholesky=mixture.precisions_cholesky_.astype(np.float32),
            log_volume=float(np.log(np.diag(np.linalg.cholesky(mixture.covariances_))).sum()),
            log_weights=np.log(mixture.weights_).astype(np.float32),
            observations=len(values), iterations=int(mixture.n_iter_), converged=bool(mixture.converged_),
            warnings=[str(w.message) for w in caught], seconds=time.perf_counter()-started,
            variance_converged=True if variance_fit is None else bool(variance_fit.success),
            training_nll=float(variance_objective(variance)[0] - .005*(variance@variance)
                + np.log(np.diag(np.linalg.cholesky(mixture.covariances_))).sum()
                + .5*values.shape[1]*math.log(2*math.pi))))
    return category, fits


def support_development(scorer, cache, indices, device):
    """Select density capacity using held-out normal likelihood, never anomaly AP."""
    features = np.load(cache["paths"]["features"], mmap_mode="r")
    conditions = np.load(cache["paths"]["conditions"], mmap_mode="r")
    rows = []
    with torch.inference_mode():
        for category, chosen in enumerate(indices):
            if not len(chosen) or not bool(scorer.present[category]):
                continue
            losses, scores, correct = [], [], 0
            for begin in range(0, len(chosen), 8192):
                at = chosen[begin:begin+8192]
                energy = scorer.class_energy(torch.tensor(features[at], device=device),
                                             torch.tensor(conditions[at], device=device))
                losses.append(energy[:, category].cpu().numpy())
                scores.append(energy.amin(1).cpu().numpy())
                correct += int((energy.argmin(1) == category).sum())
            loss, score = np.concatenate(losses), np.concatenate(scores)
            if not np.isfinite(loss).all() or not np.isfinite(score).all():
                raise ValueError("nonfinite held-out normal density")
            bands = np.searchsorted(np.log([10.,20.,35.]), conditions[chosen,0], side="right")
            by_range = []
            for band in range(4):
                selected = bands == band
                by_range.append(dict(band=band, points=int(selected.sum()),
                    score_quantiles=np.quantile(score[selected], [.5,.95,.99]).tolist() if selected.any() else None))
            rows.append(dict(category=category, points=len(chosen), mean_nll=float(loss.mean(dtype=float)),
                support_class_accuracy=correct/len(chosen),
                score_quantiles=np.quantile(score, [.1, .5, .9, .95, .99, .999]).tolist(), ranges=by_range))
    if not rows:
        raise ValueError("normal development has no supported classes")
    return dict(class_mean_nll=float(np.mean([row["mean_nll"] for row in rows])), classes=rows,
                scope="Sampled singleton normal points; each represented class has equal model-selection weight")


def target_normal_cache(cache, sequence):
    """View only the STU prefix of a frozen cache; source points are never loaded."""
    rows = [row for row in cache["frames"] if row["source"] == "normal_stu"]
    cursor = 0
    for index, row in enumerate(rows):
        if (str(row["scene"]) != str(sequence) or row["index"] != index
                or row["begin"] != cursor or row["end"] <= cursor):
            raise ValueError("target cache must preserve the complete original STU prefix")
        cursor = row["end"]
    if not cursor or rows != cache["frames"][:len(rows)]:
        raise ValueError("target cache cannot relabel or interleave source samples")
    source = np.load(cache["paths"]["source"], mmap_mode="r")[:cursor]
    if not np.all(source == 1):
        raise ValueError("non-STU point entered the target-only cache view")
    return dict(cache, count=cursor, capacity=cursor, frames=rows)


def normal_cache(cache, device):
    """Read each immutable normal feature array once, excluding unused capacity."""
    values = {}
    for key in ("features", "conditions", "semantic", "source", "frame", "allowed"):
        raw = np.load(cache["paths"][key], mmap_mode="r")[:cache["count"]]
        dtype = torch.float32 if key in ("features", "conditions") else torch.bool if key == "allowed" else torch.long
        values[key] = torch.tensor(raw, device=device, dtype=dtype)
    return values


def instance_memory(cache, size):
    """Keep real observations across normal label sets, ranges and acquisition frames."""
    from scipy.linalg import solve_triangular
    arrays = {k: np.load(v, mmap_mode="r")[:cache["count"]] for k,v in cache["paths"].items()}
    source, frame = arrays["source"], arrays["frame"]
    if not np.all(source == 1):
        raise ValueError("current normal instance training admits STU206 only")
    allowed = arrays["allowed"]
    codes = allowed.astype(np.int64) @ (1 << np.arange(19,dtype=np.int64))
    if not np.all(codes>0):
        raise ValueError("instance memory must contain verified normal observations only")
    bands = np.searchsorted(np.log([10.,20.,35.]), arrays["conditions"][:,0], side="right")
    rng = np.random.default_rng(206)
    chosen, report = [], []
    for domain, mass in ((1,1.),):
        population = np.flatnonzero(source==domain)
        groups = defaultdict(list)
        for code in np.unique(codes[population]):
            for band in range(4):
                ids = population[(codes[population]==code)&(bands[population]==band)]
                if len(ids):
                    groups[(int(code),band)] = ids
        budget = min(len(population), round(size*mass))
        quota = {key:0 for key in groups}
        remaining=budget
        while remaining:
            active=[key for key,ids in groups.items() if quota[key]<len(ids)]
            if not active:
                break
            share=max(1,remaining//len(active))
            for key in active:
                add=min(share,len(groups[key])-quota[key],remaining)
                quota[key]+=add;remaining-=add
                if not remaining:
                    break
        for (code,band),ids in groups.items():
            number=quota[(code,band)]
            # One point per acquisition frame precedes repeated points from that frame.
            shuffled=rng.permutation(ids)
            _, first=np.unique(frame[shuffled],return_index=True)
            primary=shuffled[np.sort(first)]
            primary=rng.permutation(primary)
            rest=np.setdiff1d(shuffled,primary,assume_unique=True)
            selected=np.r_[primary[:number],rng.choice(rest,max(0,number-len(primary)),replace=False)]
            chosen.extend(selected.tolist())
            report.append(dict(source=domain,allowed_code=code,range_band=band,
                               available=len(ids),selected=len(selected),frames=len(np.unique(frame[selected]))))
    chosen=np.sort(np.asarray(chosen,dtype=np.int64))
    if len(chosen)!=len(np.unique(chosen)) or not len(chosen):
        raise ValueError("normal memory must retain distinct real point identities")
    # Target sensor defines all coordinates; auxiliary-source statistics cannot move them.
    total=np.zeros(252);square=np.zeros((252,252));count=0
    for row in cache["frames"]:
        if row["source"]!="normal_stu":
            continue
        if str(row["scene"])!="206":
            raise ValueError("target training must use normal STU206 only")
        x=np.asarray(arrays["features"][row["begin"]:row["end"]],dtype=np.float64)
        total+=x.sum(0);square+=x.T@x;count+=len(x)
    mean=total/count
    covariance=square/count-np.outer(mean,mean)
    scale=np.sqrt(np.maximum(np.diag(covariance),1e-6))
    correlation=covariance/scale[:,None]/scale[None,:]
    chol=np.linalg.cholesky((correlation+correlation.T)*.5+.01*np.eye(252))
    whitener=solve_triangular(chol,np.eye(252),lower=True).T/scale[:,None]
    group=np.asarray(frame,dtype=np.int64).copy()
    return chosen, group, mean, whitener, dict(points=len(chosen),target_statistics_points=count,
        source_points=int((source[chosen]==0).sum()),target_points=int((source[chosen]==1).sum()),
        selection=report,coordinates="STU206 only; all 252 directions retained; correlation ridge .01",
        exclusion="STU206 frames less than 16 apart; no auxiliary-source instances")


@torch.no_grad()
def instance_development(scorer, data, indices, *, retain=False, exclude_neighbors=False):
    """Measure actual semantic support and normal scores on a specified normal population."""
    scorer.eval()
    from .normal import normal_semantic_metrics
    totals=torch.zeros(19,7,device=data["features"].device,dtype=torch.float64)
    confusion=torch.zeros(19,19,device=data["features"].device,dtype=torch.long)
    scores=[]
    for start in range(0,len(indices),4096):
        at=indices[start:start+4096]
        f,g,y=data["features"][at],data["conditions"][at],data["semantic"][at]
        if bool((y<0).any()):
            raise ValueError("normal class development requires actual fine labels")
        identity = (dict(source=data["source"][at],frame=data.get("group",data["frame"])[at])
                    if exclude_neighbors else {})
        energy=scorer.class_energy(f,**identity)
        prediction=energy.argmin(1)
        correct=prediction==y
        confusion+=torch.bincount(y*19+prediction,minlength=361).reshape(19,19)
        valid=torch.isfinite(energy.gather(1,y[:,None]).squeeze(1))
        # An unseen normal fine class remains in recall and anomaly support.
        # Its unavailable class likelihood is reported, never invented or dropped.
        ce=torch.zeros_like(y,dtype=torch.float32)
        if bool(valid.any()):
            ce[valid]=F.cross_entropy(-energy[valid]/scorer.temperature,y[valid],reduction="none")
        raw=scorer.raw_score(f,g,**identity)
        far=g[:,0]>=math.log(35.)
        values=torch.stack((torch.ones_like(raw),correct.float(),ce,far.float(),
                            (far&correct).float(),raw,valid.float()),-1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("nonfinite normal instance development")
        totals.index_add_(0,y,values.double())
        if retain:
            scores.append(raw.cpu())
    rows=[dict(category=c,points=int(v[0]),recall=float(v[1]/v[0]),ce=float(v[2]/v[6]) if v[6] else None,
               class_reference_available=bool(v[6]),
               far_points=int(v[3]),far_recall=float(v[4]/v[3]) if v[3] else None,
               mean_score=float(v[5]/v[0])) for c,v in enumerate(totals.cpu().tolist()) if v[0]]
    reliable=[r for r in rows if r["points"]>=128]
    result=dict(points=int(totals[:,0].sum()),classes=rows,
        class_recall=float(np.mean([r["recall"] for r in reliable])),
        class_ce=float(np.mean([r["ce"] for r in reliable if r["ce"] is not None])),
        points_without_fine_reference=int(totals[:,0].sum()-totals[:,6].sum()),
        point_recall=float(totals[:,1].sum()/totals[:,0].sum()),
        far_points=int(totals[:,3].sum()),far_recall=float(totals[:,4].sum()/totals[:,3].sum()),
        far_class_recall=float(np.mean([r["far_recall"] for r in rows if r["far_points"]])),
        semantics=normal_semantic_metrics(confusion),
        role="normal instance-support development; no anomaly labels or density-likelihood surrogate")
    return (result,torch.cat(scores)) if retain else result


def normal_main():
    """Learn a bounded multilevel metric after the frozen official perception network."""
    from threadpoolctl import threadpool_limits
    from .model import FrozenSupport
    from .normal import InstanceSupport, ScoreCalibration, select_score_calibration, INSTANCE_VERSION
    parser=argparse.ArgumentParser(description=normal_main.__doc__)
    parser.add_argument("--normal",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--features",type=Path,default=Path("results/train/normal/features"))
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--workers",type=int,default=16)
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--deadline",type=float,required=True)
    parser.add_argument("--reserve-seconds",type=int,default=2700)
    parser.add_argument("--auxiliary-manifest",type=Path)
    parser.add_argument("--initial-module",type=Path)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("use an empty output directory")
    # Measured full inference is about 22 minutes; retain 45 minutes for
    # normal checks, any authorized final inference, and independent arithmetic.
    reserve_seconds=args.reserve_seconds
    if reserve_seconds < 0:
        raise ValueError("time reserve must be nonnegative")
    if time.time()>=args.deadline-reserve_seconds:
        raise TimeoutError("insufficient time after the declared evaluation reserve")
    threadpool_limits(limits=args.threads);torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    seed_all(206)
    resources=runtime_snapshot();disk_check(500_000_000)
    cache_config=json.loads((args.features/"config.json").read_text())
    if cache_config["initial_sha256"]!=WEIGHTS_SHA256 or cache_config["trained_backbone_parameters"]!=0:
        raise ValueError("reuse only identified official frozen features")
    cache=json.loads((args.features/"training.json").read_text())
    dev_cache=json.loads((args.features/"development_features.json").read_text())
    if any(row.get("labels")!="normal_candidate_sets_v1" for row in (cache,dev_cache)):
        raise ValueError("normal cache must retain actual partial label sets")
    cache=target_normal_cache(cache,"206")
    dev_cache=target_normal_cache(dev_cache,"201")
    args.output.mkdir(parents=True)
    config=dict(version=INSTANCE_VERSION,architecture="frozen_litept_normal_instance_support",
        score_version="bounded_full_rank_multilevel_nearest_real_normal",features=str(args.features),
        initial_sha256=WEIGHTS_SHA256,seed=206,training_points=cache["count"],development_points=dev_cache["count"],
        memory_size=16384,batch_size=1024,epochs=args.epochs,learning_rate=.0002 if args.initial_module else .001,
        coordinate_source="STU206 only",data_scope="STU206_STU201",
        training="STU206 only; equal normal-class loss mass; no nuScenes training, reference or development points",
        selection="STU201 odd contiguous 64-frame blocks only; mean IoU over ground-truth-present normal classes, then available-class cross-entropy; absent training classes retain zero recall and IoU",
        calibration="STU201 even blocks only; existing normal class/range tail rule",
        calibration_bandwidths=[0.,.25,.5,1.],backbone_frozen=True,
        no_synthetic_anomalies=args.auxiliary_manifest is None,
        val19_used_for_selection=False,evaluation="repeated val19 evaluation after development selection",
        deadline=args.deadline,reserve_seconds=reserve_seconds)
    write_json(args.output/"config.json",config);write_json(args.output/"resources.json",resources)
    started=time.perf_counter()
    auxiliary = synthetic_dev = None
    if args.auxiliary_manifest is not None:
        manifest = auxiliary_records(args.auxiliary_manifest)
        write_json(args.output/"auxiliary_views.json", manifest)
        config.update(auxiliary_manifest=str(args.auxiliary_manifest.resolve()),
            auxiliary_identity=manifest["sha256"],
            auxiliary_objective="normal class CE plus 0.5 times synthetic-background class CE and pairwise softplus(1 + normal distance - anomaly distance); distances divided by normal-only temperature",
            auxiliary_pairs_per_batch=128,
            auxiliary_sampling="four anomaly-count strata select frames; uniform selected views and uniform anomaly/normal points within the same view",
            selection="synthetic STU201 AP, then AUROC, then negative FPR95, subject to no decline in real normal STU201 mean IoU from initialization; val19 unused")
        write_json(args.output/"config.json", config)
        auxiliary = auxiliary_cache(manifest, args.output, torch.device("cuda"), args.workers)
        dev_manifest = auxiliary_records(args.auxiliary_manifest, split="validation", samples=80)
        write_json(args.output/"synthetic_development.json", dev_manifest)
        synthetic_dev = auxiliary_cache(dev_manifest, args.output/"synthetic_development", torch.device("cuda"), args.workers)
    chosen,groups,location,whitener,initialization=instance_memory(cache,config["memory_size"])
    config["memory_size"]=len(chosen)
    device=torch.device("cuda")
    data,dev=normal_cache(cache,device),normal_cache(dev_cache,device)
    scorer=InstanceSupport(memory_size=len(chosen)).to(device)
    scorer.location.copy_(torch.tensor(location,device=device,dtype=torch.float32))
    scorer.whitener.copy_(torch.tensor(whitener,device=device,dtype=torch.float32))
    at=torch.tensor(chosen,device=device)
    data["group"]=torch.tensor(groups,device=device)
    for key,value in (("memory",data["features"][at]),("memory_allowed",data["allowed"][at]),
                      ("memory_source",data["source"][at]),("memory_frame",data["group"][at])):
        getattr(scorer,key).copy_(value)
    del groups,chosen,at
    count=len(data["semantic"])
    if auxiliary is not None:
        positives = (auxiliary["targets"] == 1).nonzero().flatten()
        negatives = (auxiliary["targets"] == 0).nonzero().flatten()
        if not len(positives) or not len(negatives):
            raise ValueError("auxiliary learning requires genuine anomaly and normal supervision")
        view_count = len(manifest["records"])
        positive_counts = torch.bincount(auxiliary["view"][positives], minlength=view_count)
        negative_counts = torch.bincount(auxiliary["view"][negatives], minlength=view_count)
        if bool((positive_counts == 0).any()) or bool((negative_counts == 0).any()):
            raise ValueError("each auxiliary training view must contain anomaly and normal supervision")
        positive_offsets = positive_counts.cumsum(0) - positive_counts
        negative_offsets = negative_counts.cumsum(0) - negative_counts
    weights=torch.zeros(count,device=device)
    codes=(data["allowed"].long()*(1<<torch.arange(19,device=device))).sum(1)
    mass_rows=[]
    for domain,mass in ((1,1.),):
        mask=data["source"]==domain
        labels,which,frequency=torch.unique(codes[mask],return_inverse=True,return_counts=True)
        value=frequency[which].float().reciprocal()
        weights[mask]=value/value.sum()*(count*mass)
        mass_rows.extend(dict(source=domain,allowed_code=int(c),points=int(n),loss_mass=mass/len(labels))
                         for c,n in zip(labels.cpu(),frequency.cpu()))
    initialization["supervision_mass"]=mass_rows
    # Freeze the distance temperature on independent real training neighbors only.
    probe=torch.nonzero(data["semantic"]>=0).flatten()
    probe=probe[torch.linspace(0,len(probe)-1,min(8192,len(probe)),device=device).long()]
    nearest=[]
    scorer.eval()
    with torch.no_grad():
        for start in range(0,len(probe),1024):
            at=probe[start:start+1024]
            e=scorer.class_energy(data["features"][at],source=data["source"][at],frame=data["group"][at])
            v=e.gather(1,data["semantic"][at,None]).squeeze(1)
            nearest.append(v[torch.isfinite(v)&(v>0)])
        scorer.temperature.copy_(torch.cat(nearest).median().clamp_min(1e-6))
    initialization["temperature"]=float(scorer.temperature)
    if args.initial_module is not None:
        initial = torch.load(args.initial_module, map_location=device, weights_only=False)
        if initial["config"]["initial_sha256"] != WEIGHTS_SHA256 or initial["config"].get("data_scope") != "STU206_STU201":
            raise ValueError("module initialization must use the current official-feature STU-only model")
        for key in ("location", "whitener", "memory", "memory_allowed", "memory_source", "memory_frame"):
            if not torch.equal(getattr(scorer,key), initial["model"]["scorer."+key]):
                raise ValueError("module initialization has different normal references or coordinates")
        with torch.no_grad():
            scorer.transform.copy_(initial["model"]["scorer.transform"])
            scorer.temperature.copy_(initial["model"]["scorer.temperature"])
        initialization["module_initialization"] = dict(path=str(args.initial_module), sha256=file_sha256(args.initial_module),
            role="reuse only the previously learned STU-normal distance transform and its fixed temperature")
        config["module_initialization"] = initialization["module_initialization"]
        initialization["temperature"] = float(scorer.temperature)
        del initial
    scorer.train()
    probe=probe[:config["batch_size"]]
    tick=time.perf_counter()
    energy=scorer.class_energy(data["features"][probe],source=data["source"][probe],frame=data["group"][probe])
    allowed=data["allowed"][probe]
    valid=(torch.isfinite(energy)|~allowed).all(1)
    logits=-energy[valid]/scorer.temperature
    loss=(torch.logsumexp(logits,-1)-torch.logsumexp(logits.masked_fill(~allowed[valid],-torch.inf),-1)).mean()
    if auxiliary is not None:
        views = torch.linspace(0, view_count-1, 128, device=device).long()
        extra, ranking, _, _ = auxiliary_loss(scorer, auxiliary,
            positives[positive_offsets[views]], negatives[negative_offsets[views]])
        auxiliary_gradient = torch.autograd.grad(ranking, scorer.transform, retain_graph=True)[0]
        if not bool(torch.isfinite(auxiliary_gradient).all()) or not bool(auxiliary_gradient.abs().sum() > 0):
            raise ValueError("real auxiliary labels must update the deployed metric")
        initialization["anomaly_ranking_gradient_norm"] = float(auxiliary_gradient.norm())
        loss = loss + extra
    loss.backward()
    if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in scorer.parameters()):
        raise ValueError("real normal metric preflight has nonfinite loss or gradient")
    scorer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    initialization["real_preflight"]=dict(points=len(probe),valid=int(valid.sum()),loss=float(loss),
        forward_backward_seconds=time.perf_counter()-tick,gpu_peak_bytes=torch.cuda.max_memory_allocated())
    print("normal real-data preflight "+json.dumps(initialization["real_preflight"]),flush=True)
    write_json(args.output/"initialization.json",initialization)
    selected=((dev["source"]==1)&((dev["frame"]//64)%2==1)).nonzero().flatten()
    calibration=((dev["source"]==1)&((dev["frame"]//64)%2==0)).nonzero().flatten()
    baseline=instance_development(scorer,dev,selected)
    best=dict(epoch=0,development=baseline,trained_metric=False)
    model=FrozenSupport(scorer=scorer).to(device).eval()
    with torch.no_grad():
        predicted=model.perception.seg_head(dev["features"][:,180:]).argmax(-1)
    def fit_current_calibration():
        holdout,hold_scores=instance_development(scorer,dev,selected,retain=True)
        reference,ref_scores=instance_development(scorer,dev,calibration,retain=True)
        scorer.calibration, report = select_score_calibration(
            (ref_scores,predicted[calibration].cpu(),dev["conditions"][calibration].cpu()),
            (hold_scores,predicted[selected].cpu(),dev["conditions"][selected].cpu()),
            config["calibration_bandwidths"],device)
        return dict(report,reference=reference,holdout=holdout)
    if synthetic_dev is not None:
        best["calibration"] = fit_current_calibration()
        best["synthetic"] = auxiliary_development(scorer, synthetic_dev, device)
        print("synthetic development initial "+json.dumps(best["synthetic"]),flush=True)
    atomic_save(args.output/"selected.pt",dict(scorer=scorer.state_dict(),selection=best))
    write_json(args.output/"development.json",dict(initial=baseline,selected=best))
    print(f"initial normal metric: class recall={baseline['class_recall']:.4f}, CE={baseline['class_ce']:.4f}",flush=True)
    optimizer=torch.optim.AdamW(scorer.parameters(),lr=config["learning_rate"],weight_decay=0.)
    stale=0
    for epoch in range(1,args.epochs+1):
        if time.time()>=args.deadline-reserve_seconds:
            break
        epoch_start=time.perf_counter();scorer.train()
        generator=torch.Generator(device=device).manual_seed(206+epoch)
        order=torch.randperm(count,generator=generator,device=device)
        for group in optimizer.param_groups:
            group["lr"]=config["learning_rate"]*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs)))
        total_loss=0.;skipped=0;processed=0
        auxiliary_ranking=0.;auxiliary_ce=0.;auxiliary_skipped=0;updates=0
        for start in range(0,count,config["batch_size"]):
            at=order[start:start+config["batch_size"]]
            energy=scorer.class_energy(data["features"][at],source=data["source"][at],frame=data["group"][at])
            allowed=data["allowed"][at]
            valid=(torch.isfinite(energy)|~allowed).all(1)
            skipped+=int((~valid).sum())
            if not bool(valid.any()):
                continue
            logits=-energy[valid]/scorer.temperature
            supported=torch.logsumexp(logits.masked_fill(~allowed[valid],-torch.inf),-1)
            loss=((torch.logsumexp(logits,-1)-supported)*weights[at][valid]).sum()/len(at)
            if auxiliary is not None:
                # Equal view mass prevents dense nearby objects dominating the anomaly gradient.
                views = torch.randint(view_count, (128,), generator=generator, device=device)
                offsets = (torch.rand(128, generator=generator, device=device)*positive_counts[views]).long()
                p = positives[positive_offsets[views]+offsets]
                offsets = (torch.rand(128, generator=generator, device=device)*negative_counts[views]).long()
                q = negatives[negative_offsets[views]+offsets]
                extra, ranking, augmented_ce, missing = auxiliary_loss(scorer, auxiliary, p, q)
                loss = loss + extra
                auxiliary_ranking += float(ranking.detach())
                auxiliary_ce += float(augmented_ce.detach())
                auxiliary_skipped += missing
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(scorer.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();scorer.project_metric()
            total_loss+=float(loss.detach())*len(at)
            processed+=len(at)
            updates+=1
            if processed//config["batch_size"]%500==0:
                elapsed=time.perf_counter()-epoch_start
                print(f"normal metric epoch {epoch}/{args.epochs}: {processed}/{count} points, "
                      f"loss={total_loss/max(processed,1):.4f}, remaining {(count-processed)*elapsed/max(processed,1):.0f}s",flush=True)
                if time.time()>=args.deadline-reserve_seconds:
                    break
        measured=instance_development(scorer,dev,selected)
        quality=(measured["semantics"]["mean_iou_gt"],-measured["class_ce"])
        improved=quality>(best["development"]["semantics"]["mean_iou_gt"],-best["development"]["class_ce"])
        synthetic = calibration_report = None
        if synthetic_dev is not None:
            calibration_report = fit_current_calibration()
            synthetic = auxiliary_development(scorer, synthetic_dev, device)
            metrics, previous = synthetic["metrics"], best["synthetic"]["metrics"]
            quality = (metrics["AP"], metrics["AUROC"], -metrics["FPR95"])
            improved = (measured["semantics"]["mean_iou_gt"] >= baseline["semantics"]["mean_iou_gt"]
                and quality > (previous["AP"], previous["AUROC"], -previous["FPR95"]))
            print(f"synthetic development epoch {epoch}: "+json.dumps(synthetic),flush=True)
        row=dict(epoch=epoch,development=measured,training_loss=total_loss/max(processed,1),
                 processed_points=processed,skipped_without_independent_allowed_anchors=skipped,seconds=time.perf_counter()-epoch_start)
        if auxiliary is not None:
            row["auxiliary"] = dict(pairs=128*updates, ranking_loss=auxiliary_ranking/max(updates,1),
                normal_semantic_loss=auxiliary_ce/max(updates,1),
                normal_without_independent_class_anchor=auxiliary_skipped)
            row["synthetic"] = synthetic
        with (args.output/"training.jsonl").open("a") as handle:
            handle.write(json.dumps(row)+"\n")
        print(f"normal metric epoch {epoch}/{args.epochs}: mIoU={measured['semantics']['mean_iou_gt']:.4f}, class recall={measured['class_recall']:.4f}, "
              f"CE={measured['class_ce']:.4f}, far recall={measured['far_recall']:.4f}, "
              f"skipped={skipped}, {row['seconds']:.1f}s",flush=True)
        if improved:
            best=dict(epoch=epoch,development=measured,trained_metric=True)
            if synthetic is not None:
                best.update(synthetic=synthetic, calibration=calibration_report)
            atomic_save(args.output/"selected.pt",dict(scorer=scorer.state_dict(),selection=best));stale=0
        else:
            stale+=1
        write_json(args.output/"development.json",dict(initial=baseline,selected=best,last=row))
        if epoch>=4 and stale>=3:
            break
    saved=torch.load(args.output/"selected.pt",map_location=device,weights_only=False)
    # The selected and final epochs may use different calibration buffer schemas.
    bandwidth=float(saved["scorer"].get("calibration.range_bandwidth",0.))
    scorer.calibration=ScoreCalibration(bandwidth).to(device)
    scorer.load_state_dict(saved["scorer"]);scorer.eval()
    training=instance_development(scorer,data,torch.arange(count,device=device),exclude_neighbors=True)
    write_json(args.output/"normal206.json",dict(training,scope="all cached STU206 normal points; adjacent reference frames excluded"))
    calibration_report = best.get("calibration") or fit_current_calibration()
    write_json(args.output/"calibration.json",calibration_report)
    config.update(calibration_bandwidth=calibration_report["range_bandwidth"],calibrated=bool(scorer.calibration.enabled))
    write_json(args.output/"config.json",config)
    atomic_save(args.output/"frozen.pt",dict(version=INSTANCE_VERSION,mode="frozen_support",model=model.state_dict(),
        config=config,frozen=True,selected=True,complete=True,selection=best,final_val19_evaluated=False))
    (args.output/"selected.pt").unlink()
    result=dict(version=INSTANCE_VERSION,complete=True,selected=best,initial=baseline,
        trainable_parameters=sum(p.numel() for p in scorer.parameters()),training_seconds=time.perf_counter()-started,
        gpu_peak_bytes=torch.cuda.max_memory_allocated(),val19_used_for_selection=False,
        no_synthetic_anomalies=args.auxiliary_manifest is None,backbone_frozen=True,
        evaluation_status="not yet repeated")
    write_json(args.output/"result.json",result);print(json.dumps(result),flush=True)


def main():
    if "--normal" in sys.argv:
        normal_main()
        return
    parser = argparse.ArgumentParser(description="Train the complete observation-constrained normal-field segmentor.")
    parser.add_argument("--train-manifest", type=Path, required=True, help="training data for this run")
    parser.add_argument("--val-manifest", type=Path, required=True, help="development data used for model selection")
    parser.add_argument("--output", type=Path, required=True, help="experiment output directory")
    parser.add_argument("--initial", type=Path, default=Path("assets/nuscenes.pth"))
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--epochs", type=int, help="explicit complete training-data passes")
    budget.add_argument("--updates", type=int, help="fixed source-only update budget with 4 anomaly / 2 normal control / 2 original scans")
    parser.add_argument("--retain-activations", action="store_true", help="retain model/normal activations instead of recomputing; uses more VRAM")
    parser.add_argument("--progress", action="store_true", help="compact live progress; detailed JSON remains in log.jsonl")
    parser.add_argument("--eval-every", type=int, help="updates between development evaluations; default: endpoint only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--score-path", type=Path, help="optional endpoint scores in official point order")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true", help="one complete effective-batch backward; no parameter update")
    args = parser.parse_args()
    if (args.epochs is not None and args.epochs < 1 or args.updates is not None and args.updates < 1
            or args.seed < 0 or args.workers < 0 or args.threads < 1 or args.save_every < 1
            or args.eval_every is not None and args.eval_every < 1):
        parser.error("invalid training budget or runtime settings")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        parser.error("the full effective-batch gradient cache currently supports one GPU")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        parser.error("LitePT sparse convolution/FlashAttention requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    train, val = load_manifest(args.train_manifest, "train"), load_manifest(args.val_manifest, "val")
    config = configuration(train, val, device, 1, initial=args.initial, recipe="field",
                           eval_every=args.eval_every, passes=args.epochs, updates=args.updates, seed=args.seed)
    config["seeds"] = [args.seed]
    if args.retain_activations:
        config["retain_activations"] = True
        config["gradient_cache"]["activations"] = "retain model and normal-field intermediates; backbone attention still recomputes"
    args.updates = config["updates"]
    resources = runtime_snapshot()
    if args.workers + args.threads > len(os.sched_getaffinity(0)):
        parser.error("worker and numerical-library threads exceed CPU affinity")
    running = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                             check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()
    other = [int(pid.strip()) for pid in running if pid.strip().isdigit() and int(pid.strip()) != os.getpid()]
    if other:
        raise RuntimeError(f"other CUDA processes must finish before this run: {other}")
    # Best/last optimizer states, atomic replacement, bounded logs and optional scores.
    peak = 100_000_000 if args.check else 2_000_000_000 + (
        4 * sum(val["records"][i]["normal"] + val["records"][i]["anomaly"] for i in evaluation_indices(val))
        if args.score_path else 0)
    disk_check(peak)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "resources.json", resources)
    if args.check:
        preflight(args, train, val, device, config, resources, "field")
        return
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    print(json.dumps(dict(model="field", seed=args.seed, samples=config["samples"],
                          passes=args.epochs, updates=config["updates"], eval_every=config["eval_every"])), flush=True)
    train_stage(args, train, val, args.seed, "field", device, config)


if __name__ == "__main__":
    main()
