"""Run a bounded V4 training segment with explicit sampling and validation intervals."""

import argparse
import copy
from collections import Counter, defaultdict, deque
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

        def interleave(node):
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
            queues = deque(deque(interleave(node[keys[int(i)]])) for i in rng.permutation(len(keys)))
            result = []
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
                    order="interleave distance band, donor, scene, point-count bin and placement; temporally separated frames first",
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


def main():
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
