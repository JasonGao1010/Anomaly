"""Train V3 with source-uniform paired requests and fixed development panels."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import psutil
import torch
from torch import nn

from .data import DEFAULT_SAMPLES, FrozenDataset, detection_targets
from .evaluate import (
    EXPOSURE_GROUPS,
    coverage_status,
    disappeared,
    evaluate,
    evaluation_kind,
    instance_rows,
    write_json,
)
from .model import (
    V3,
    SEED,
    METHOD,
    LOSS_TERMS,
    configure_runtime,
    compatible_parameter,
    lr_factor,
    optimizer,
    paired_loss,
    prepare_scan,
    transfer,
    tail_weights,
)


@dataclass
class PairInput:
    scans: tuple
    positive_target: np.ndarray
    negative_target: np.ndarray
    instance: np.ndarray
    frame: int
    world: str
    groups: list
    raw_labels: tuple
    neighborhoods: tuple


def pair_rows(dataset, index, original, rendered, target):
    world = dataset.worlds[index // len(dataset.sequence)]
    metadata = {
        "60001": dict(
            height_m=world["height_m"],
            height_source=world["height_source"],
            object_id=world["identity"],
        )
    }
    rows = instance_rows(rendered.source, target, metadata)
    # The frozen pool has one verified object; only then can frame D be assigned to it.
    count = disappeared(original, rendered)
    for row in rows:
        row["disappeared"] = count
    return rows


def prepare_pair(dataset, index):
    original, rendered = dataset.pair(int(index))
    positive = rendered.source
    target = detection_targets(positive, inserted=rendered.inserted_mask)
    negative_target = detection_targets(original)
    scans = (
        prepare_scan(original.xyzi[original.observation_slots]),
        prepare_scan(positive.xyzi[positive.observation_slots]),
    )
    return PairInput(
        scans,
        target,
        negative_target,
        positive.labels.instance[positive.observation_slots],
        original.frame_id,
        rendered.world_identity,
        pair_rows(dataset, index, original, rendered, target),
        (
            original.labels.semantic[original.observation_slots],
            positive.labels.semantic[positive.observation_slots],
        ),
        tuple(
            neighborhood_summary(scan, labels)
            for scan, labels in zip(scans, (negative_target, target))
        ),
    )


def resources(expected_disk_bytes=0):
    """The Windows host free space, rather than ext4 capacity, constrains writes."""
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "Get-Volume -DriveLetter E | Select-Object Size,SizeRemaining | ConvertTo-Json -Compress",
    ]
    volume = json.loads(subprocess.check_output(command, text=True, timeout=20))
    if volume["SizeRemaining"] - expected_disk_bytes < 10_000_000_000:
        raise RuntimeError("expected peak writes would consume the E: 10 GB reserve")
    memory = psutil.virtual_memory()
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    ).strip()
    return dict(
        logical_cpus=len(os.sched_getaffinity(0)),
        physical_cpus=psutil.cpu_count(logical=False),
        memory_available=memory.available,
        swap_used=psutil.swap_memory().used,
        host_volume=volume,
        gpu=gpu,
        gpu_processes=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip(),
    )


def request_at(seed, update, count, worlds, frames):
    # Each update owns an independent stream: extending 128 to 256 preserves its prefix.
    rng = np.random.default_rng(np.random.SeedSequence([seed, update]))
    source = rng.integers(frames, size=count)
    world = rng.integers(worlds, size=count)
    return world * frames + source


def condition_flags(row):
    return dict(
        few_returns=1 <= row["observed_returns"] <= 4,
        low_height=None if row["height_m"] is None else row["height_m"] <= 0.30,
        far_range=35 <= row["distance_m"] <= 50,
        weak_disappearance=None
        if row.get("disappeared") is None
        else row["disappeared"] <= 4,
    )


def record_conditions(exposure, frame, world, rows):
    groups = exposure.setdefault("conditions", {})
    unknown = exposure.setdefault("unknown", {name: 0 for name in EXPOSURE_GROUPS})
    intersections = exposure.setdefault("intersections", {})
    for row in rows:
        flags = condition_flags(row)
        tags = []
        for name, value in flags.items():
            group = groups.setdefault(name, dict(requests=[], sources=[], objects=[]))
            if value is None:
                unknown[name] += 1
            elif value:
                tags.append(name)
                group["requests"] = sorted(
                    {tuple(v) for v in group["requests"]} | {(world, frame)}
                )
                group["sources"] = sorted(set(group["sources"]) | {frame})
                if row["object_id"] is not None:
                    group["objects"] = sorted(
                        {tuple(v) for v in group["objects"]}
                        | {(world, row["instance"])}
                    )
        if tags:
            key = "+".join(tags)
            intersections[key] = intersections.get(key, 0) + 1


def _coverage_metadata(request):
    samples, data_root, indices = request
    dataset = FrozenDataset(samples, data_root, "train")
    rows = {}
    for index in indices:
        original, rendered = dataset.pair(index)
        target = detection_targets(rendered.source, inserted=rendered.inserted_mask)
        rows[index] = pair_rows(dataset, index, original, rendered, target)
    return rows


def prepare_coverage(dataset, data_root, seed, output, workers):
    """Inspect the unchanged 1024-step request prefix; no model or resampling is used."""
    requests = [
        request_at(seed, step, 8, len(dataset.worlds), len(dataset.sequence)).tolist()
        for step in range(1, 1025)
    ]
    indices = sorted(set(sum(requests, [])))
    # Bind reused metadata to the exact frozen inputs and source file revisions.
    files = [dataset.samples[i][0] for i in indices]
    files += sorted(dataset.sequence.sequence_dir.rglob("*.bin"))
    files += sorted(dataset.sequence.sequence_dir.rglob("*.label"))
    files += [
        dataset.sequence.sequence_dir / name for name in ("poses.txt", "calib.txt")
    ]
    files += [world["path"] / "manifest.json" for world in dataset.worlds]
    identity = dict(
        worlds=[w["identity"] for w in dataset.worlds],
        files=[
            [str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in files
        ],
    )
    output = Path(output)
    if output.exists():
        saved = json.loads(output.read_text())
        if (
            saved.get("method") != METHOD
            or saved["seed"] != seed
            or saved["inputs"] != identity
        ):
            raise ValueError(
                "coverage metadata belongs to different inputs or seed; use a separate coverage path"
            )
        return saved
    start = time.perf_counter()
    # Keep all requests for one source in a worker to reuse the existing one-frame cache.
    buckets = [[] for _ in range(workers)]
    for frame in dataset.sequence.frame_ids:
        buckets[frame % workers].extend(
            i for i in indices if i % len(dataset.sequence) == frame
        )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = {}
        for part in pool.map(
            _coverage_metadata, [(dataset.directory, data_root, b) for b in buckets]
        ):
            records.update(part)
    exposure, groups = (
        {},
        {name: dict(first_update=None, node=None) for name in EXPOSURE_GROUPS},
    )
    snapshots = {}
    for step, selected in enumerate(requests, 1):
        for index in selected:
            world = dataset.worlds[index // len(dataset.sequence)]["identity"]
            record_conditions(
                exposure, index % len(dataset.sequence), world, records[index]
            )
        for name, entry in groups.items():
            count = exposure.get("conditions", {}).get(name, {})
            if entry["first_update"] is None and all(
                len(count.get(key, [])) >= minimum
                for key, minimum in (("requests", 20), ("sources", 5), ("objects", 5))
            ):
                entry.update(first_update=step, node=128 * ((step + 127) // 128))
        if step % 128 == 0:
            snapshots[str(step)] = coverage_status(exposure, dict(groups=groups), step)
    nodes = [group["node"] for group in groups.values()]
    result = dict(
        method=METHOD,
        seed=seed,
        requests_per_update=8,
        max_updates=1024,
        inputs=identity,
        groups=groups,
        common_node=max(nodes) if all(n is not None for n in nodes) else None,
        snapshots=snapshots,
        unknown=exposure.get("unknown", {}),
        intersections=exposure.get("intersections", {}),
        scope="observed request prefix only; insufficient prefix support never asserts absence from the pool",
        seconds=time.perf_counter() - start,
    )
    write_json(output, result)
    return result


def neighborhood_summary(scan, target):
    distances, same, mixed, labeled = [], 0, 0, 0
    edges = 0
    for start in range(0, len(target), 8192):
        neighbor = scan.neighbors[start : start + 8192, 1:]
        center = scan.features[start : start + len(neighbor), :3]
        distance = np.linalg.norm(
            scan.features[neighbor, :3] - center[:, None, :], axis=-1
        )
        if distance.shape[1]:
            distances.append(distance[:, -1])
        same += int(
            (
                scan.inverse[neighbor]
                == scan.inverse[start : start + len(neighbor), None]
            ).sum()
        )
        query = target[start : start + len(neighbor), None]
        valid = (query >= 0) & (target[neighbor] >= 0)
        mixed += int((valid & (query != target[neighbor])).sum())
        labeled += int(valid.sum())
        edges += neighbor.size
    return dict(
        other_edges=edges,
        same_voxel_fraction=same / edges if edges else None,
        labeled_edges=labeled,
        anomaly_normal_edge_fraction=mixed / labeled if labeled else None,
        farthest_neighbor_m_quantiles=np.quantile(
            np.concatenate(distances), [0, 0.5, 0.95, 1]
        ).tolist()
        if distances
        else [],
    )


def bn_state(model):
    return {
        name: torch.cat(
            (module.running_mean.detach(), module.running_var.detach())
        ).clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm1d)
    }


def run_update(
    model, opt, pairs, update, halvings=(), balance="instance", tail_weight=0.5
):
    if len(pairs) != 8:
        raise ValueError("one V3 update is exactly 8 requests, in 4 batches of 2 pairs")
    model.train()
    opt.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    previous = [
        torch.cat([p.detach().flatten() for p in group["params"]]).clone()
        for group in opt.param_groups
    ]
    bn_before = bn_state(model)
    terms, scores, grouped, tail_concentration, neighborhoods = [], [], {}, [], []
    relation_before = torch.cat(
        [p.detach().flatten() for p in model.decoder.parameters()]
    ).clone()
    for group in opt.param_groups:
        group["lr"] = group["peak_lr"] * lr_factor(update, halvings)
    for offset in range(0, 8, 2):
        batch = pairs[offset : offset + 2]
        predictions = model([scan for pair in batch for scan in pair.scans])
        losses = []
        for pair, negative, positive in zip(batch, predictions[::2], predictions[1::2]):
            pt = torch.as_tensor(pair.positive_target, device=device)
            nt = torch.as_tensor(pair.negative_target, device=device)
            identity = torch.as_tensor(pair.instance.astype(np.int64), device=device)
            loss, parts = paired_loss(
                positive, negative, pt, nt, identity, balance, tail_weight
            )
            losses.append(loss)
            terms.append(parts.detach())
            for role, values, target in (
                ("inserted", positive, pt),
                ("original", negative, nt),
            ):
                for label in (0, 1):
                    chosen = values[target == label].detach()
                    if len(chosen):
                        scores.append(
                            (
                                f"{role}/{label}",
                                len(chosen),
                                chosen.mean(),
                                chosen.square().mean(),
                                chosen.min(),
                                chosen.max(),
                            )
                        )
            for view, values, target, scan, raw, neighborhood in zip(
                ("original", "inserted"),
                (negative, positive),
                (nt, pt),
                pair.scans,
                pair.raw_labels,
                pair.neighborhoods,
            ):
                with torch.no_grad():
                    normal = target == 0
                    weights = (
                        tail_weights(nn.functional.softplus(values[normal]))
                        .cpu()
                        .numpy()
                    )
                normal_mask = normal.cpu().numpy()
                semantic = raw[normal_mask]
                radius = np.linalg.norm(scan.features[normal_mask, :3], axis=1)
                tail_concentration.append(
                    dict(
                        world=pair.world,
                        frame=pair.frame,
                        view=view,
                        normal_points=len(weights),
                        tail_points=int((weights > 0).sum()),
                        semantic_mass={
                            str(int(label)): float(weights[semantic == label].sum())
                            for label in np.unique(semantic)
                        },
                        far_mass=float(weights[radius >= 35].sum()),
                    )
                )
                neighborhoods.append(
                    dict(
                        world=pair.world,
                        frame=pair.frame,
                        view=view,
                        **neighborhood,
                    )
                )
            for row in pair.groups:
                mask = (pt == 1) & (identity == row["instance"])
                value = nn.functional.softplus(-positive[mask]).detach().mean()
                for kind in ("returns", "distance", "height"):
                    if row[kind] is not None:
                        grouped.setdefault(f"{kind}/{row[kind]}", []).append(value)
        # Four micro-batches sum to the arithmetic mean of the eight requests.
        (torch.stack(losses).sum() / 8).backward()
    grad_groups = []
    for group in opt.param_groups:
        norm = (
            torch.stack(
                [
                    p.grad.detach().square().sum()
                    for p in group["params"]
                    if p.grad is not None
                ]
            )
            .sum()
            .sqrt()
        )
        grad_groups.append(float(norm))
    relation_gradient = float(
        torch.stack(
            [
                p.grad.square().sum()
                for p in model.decoder.parameters()
                if p.grad is not None
            ]
        )
        .sum()
        .sqrt()
    )
    gradient = nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    opt.step()
    parameter_groups = []
    for old, grad, group in zip(previous, grad_groups, opt.param_groups):
        current = torch.cat([p.detach().flatten() for p in group["params"]])
        parameter_groups.append(
            dict(
                role=group["role"],
                weight_decay=group["weight_decay"],
                lr=group["lr"],
                gradient_norm=grad,
                relative_update=float(
                    (current - old).norm() / old.norm().clamp_min(1e-12)
                ),
            )
        )
    score_summary = {}
    for role in {row[0] for row in scores}:
        rows = [row for row in scores if row[0] == role]
        count = sum(row[1] for row in rows)
        mean = sum(row[1] * float(row[2]) for row in rows) / count
        second = sum(row[1] * float(row[3]) for row in rows) / count
        score_summary[role] = dict(
            points=count,
            mean=mean,
            std=max(0.0, second - mean**2) ** 0.5,
            minimum=min(float(row[4]) for row in rows),
            maximum=max(float(row[5]) for row in rows),
        )
    torch.cuda.synchronize()
    loss_terms = torch.stack(terms).cpu().numpy()
    return dict(
        update=update,
        scan_views=update * 16,
        seconds=time.perf_counter() - start,
        loss_terms=dict(
            zip(
                LOSS_TERMS,
                loss_terms.mean(0).tolist(),
            )
        ),
        loss_term_std=dict(
            zip(
                LOSS_TERMS,
                loss_terms.std(0).tolist(),
            )
        ),
        group_anomaly_losses={
            key: float(torch.stack(values).mean()) for key, values in grouped.items()
        },
        zero_anomaly_requests=sum(not np.any(p.positive_target == 1) for p in pairs),
        few_return_instances=sum(r["points"] < 5 for p in pairs for r in p.groups),
        visible_instances=sum(len(p.groups) for p in pairs),
        parameter_groups=parameter_groups,
        gradient_norm=float(gradient),
        clipped=bool(gradient > 1),
        relation_update=dict(
            gradient_norm=relation_gradient,
            relative_update=float(
                (
                    torch.cat(
                        [p.detach().flatten() for p in model.decoder.parameters()]
                    )
                    - relation_before
                ).norm()
                / relation_before.norm().clamp_min(1e-12)
            ),
        ),
        tail_concentration=tail_concentration,
        neighborhoods=neighborhoods,
        bn_change={
            name: float((value - bn_before[name]).norm())
            for name, value in bn_state(model).items()
        },
        score_distribution=score_summary,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_bytes=psutil.Process().memory_info().rss,
    )


def save_checkpoint(path, model, opt, step, metadata, exposure):
    # Write atomically so an interrupted save cannot destroy the resumable state.
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    payload = dict(
        format=METHOD,
        mechanism=model.mechanism,
        seed=model.seed,
        model=model.state_dict(),
        optimizer=opt.state_dict(),
        step=step,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
        exposure=exposure,
        **metadata,
    )
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_continuation_recipe(old, new, step, budget):
    for key in ("balance", "tail_weight", "backbone_lr", "new_lr"):
        if old[key] != new[key]:
            raise ValueError(
                "changing the training risk or base rates requires a new route"
            )
    previous, current = old["fixed_recipe"], new["fixed_recipe"]
    if (previous is None) != (current is None):
        raise ValueError("continuation must retain the shared recipe")
    if previous is not None:
        # Only future budget and decay decisions may change; the request stream is
        # indexed by seed and update, independently of the planned final budget.
        def immutable(recipe):
            return {k: v for k, v in recipe.items() if k not in {"steps", "halve_at"}}

        if immutable(previous) != immutable(current):
            raise ValueError(
                "continuation cannot change the shared scientific settings"
            )
        if current["steps"] < previous["steps"]:
            raise ValueError("continuation cannot reduce the shared budget")
    if [v for v in new["halve_at"] if v <= step] != [
        v for v in old["halve_at"] if v <= step
    ]:
        raise ValueError("a continuation cannot rewrite past learning rates")
    if budget <= step:
        raise ValueError("continuation must extend beyond the saved update")


def train(args):
    panels = json.loads(args.panels.read_text())
    if panels.get("format") != "ajae-v3-panels":
        raise ValueError("prepare fixed V3 panels before training")
    if args.steps < 1 or len(set(args.halve_at)) != len(args.halve_at):
        raise ValueError("invalid update budget or repeated decay node")
    if args.halve_at != sorted(args.halve_at) or any(
        right - left < 128 for left, right in zip(args.halve_at, args.halve_at[1:])
    ):
        raise ValueError(
            "observe at least 128 updates between learning-rate reductions"
        )
    fixed = json.loads(args.fixed_recipe.read_text()) if args.fixed_recipe else None
    if fixed is not None:
        expected = dict(
            steps=args.steps,
            seed=args.seed,
            route=args.route,
            balance=args.balance,
            backbone_lr=args.backbone_lr,
            new_lr=args.new_lr,
            halve_at=args.halve_at,
        )
        if fixed != expected:
            raise ValueError(
                "D0-D3 and risk controls must use exactly the shared recipe's seed, initialization, risk, rates, and budget"
            )
    if args.halve_at and (
        not args.reason or args.mechanism != "D3" and not args.fixed_recipe
    ):
        raise ValueError(
            "record the decay reason; mechanism controls require a common fixed recipe"
        )
    if args.mechanism != "D3" and args.tail_weight != 0.5:
        raise ValueError(
            "structure controls keep lambda=0.5; vary normal risk only on D3"
        )
    if (args.mechanism != "D3" or args.tail_weight != 0.5) and not args.fixed_recipe:
        raise ValueError(
            "D0-D3 controls require the same predeclared budget and learning-rate trajectory"
        )
    lr_factor(1, args.halve_at)
    checkpoints = sum(evaluation_kind(i) is not None for i in range(args.steps + 1)) + 1
    snapshot = resources(expected_disk_bytes=(checkpoints + 1) * 200_000_000)
    configure_runtime()
    torch.manual_seed(args.seed)
    dataset = FrozenDataset(args.samples, args.data_root, "train")
    workers = min(
        8,
        max(1, snapshot["logical_cpus"] // 2),
        max(1, snapshot["memory_available"] // 500_000_000),
    )
    coverage_plan = prepare_coverage(
        dataset, args.data_root, args.seed, args.coverage, workers
    )
    print(
        json.dumps(
            dict(
                coverage_nodes=coverage_plan["groups"],
                common_node=coverage_plan["common_node"],
                unresolved=coverage_plan["snapshots"]["1024"]["reasons"],
            )
        ),
        flush=True,
    )
    model = V3(args.mechanism, args.seed)
    recipe = dict(
        balance=args.balance,
        tail_weight=args.tail_weight,
        backbone_lr=args.backbone_lr,
        new_lr=args.new_lr,
        halve_at=args.halve_at,
        fixed_recipe=fixed,
    )
    exposure = dict(
        sources=[],
        worlds=[],
        world_frames=[],
        instance_observations=[],
        requests=0,
        zero_anomaly_requests=0,
        few_return_instances=0,
        visible_instances=0,
        clipped_updates=0,
        training_seconds=0.0,
    )
    metadata = dict(
        method=METHOD,
        route=args.route,
        parameter_counts=dict(
            total=sum(p.numel() for p in model.parameters()),
            transferred_scope=sum(
                p.numel()
                for name, p in model.named_parameters()
                if compatible_parameter(name)
            ),
            decoder=sum(p.numel() for p in model.decoder.parameters()),
        ),
        coverage_plan=coverage_plan,
        panels=panels,
        recipe=recipe,
        initialization=None,
        decisions=[
            dict(
                from_step=0,
                to_step=args.steps,
                reason=args.reason or "first V3 exploration budget",
                halve_at=args.halve_at,
            )
        ],
        resources=snapshot,
        numerics=dict(
            dtype="float32",
            tf32=False,
            activation_recomputation="BN-free backbone blocks and complete-scan-key decoder query chunks",
            neighbor_count=32,
            decoder_chunk_size=model.decoder.chunk_size,
            torch=str(torch.__version__),
        ),
    )
    step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("format") != METHOD:
            raise ValueError(
                "cannot resume an earlier method as the refined original-point model"
            )
        if any(
            saved[key] != value
            for key, value in (
                ("mechanism", args.mechanism),
                ("seed", args.seed),
                ("route", args.route),
                ("panels", panels),
                ("coverage_plan", coverage_plan),
            )
        ):
            raise ValueError(
                "resume would change the scientific route, seed, or fixed panels"
            )
        validate_continuation_recipe(saved["recipe"], recipe, saved["step"], args.steps)
        if not args.reason:
            raise ValueError(
                "record the scientific reason for extending or resuming training"
            )
        model.load_state_dict(saved["model"], strict=True)
        metadata["initialization"] = saved["initialization"]
        metadata["decisions"] = saved["decisions"] + [
            dict(
                from_step=saved["step"],
                previous_budget=saved["decisions"][-1]["to_step"],
                to_step=args.steps,
                reason=args.reason,
                halve_at=args.halve_at,
            )
        ]
        exposure, step = saved["exposure"], saved["step"]
    elif args.route in {"P", "T"}:
        if args.weights is None:
            raise ValueError(
                "P/T require an explicit weight file; random fallback is forbidden"
            )
        metadata["initialization"] = transfer(model, args.weights, args.route)
    elif args.weights:
        raise ValueError("R cannot inherit external weights")
    model.cuda()
    opt = optimizer(model, args.backbone_lr, args.new_lr)
    if args.resume:
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise FileExistsError("choose an empty candidate directory")
    write_json(
        output / "run.json", dict(**metadata, mechanism=args.mechanism, seed=args.seed)
    )
    sources, worlds = set(exposure["sources"]), set(exposure["worlds"])
    world_frames = {tuple(value) for value in exposure["world_frames"]}
    instances = {tuple(value) for value in exposure["instance_observations"]}
    start = time.perf_counter()

    def report(update, result):
        return dict(
            update=update,
            scan_views=update * 16,
            candidate=dict(
                route=args.route,
                mechanism=args.mechanism,
                seed=args.seed,
                update=update,
                tail_weight=args.tail_weight,
            ),
            exposure=exposure,
            coverage_plan=coverage_plan,
            training_seconds=exposure.get("training_seconds"),
            **result,
        )

    if not args.resume:
        save_checkpoint(output / "0.pt", model, opt, 0, metadata, exposure)
        result = evaluate(model, panels, args.data_root, args.samples, "micro")
        write_json(output / "0.json", report(0, result))
    # Loading one request ahead overlaps CPU voxelization with GPU work. Shared source
    # cache access stays in one loader thread; memory remains bounded to two updates.
    with (
        ThreadPoolExecutor(max_workers=1) as loader,
        (output / "train.jsonl").open("a") as log,
    ):

        def load_update(number):
            indices = request_at(
                args.seed, number, 8, len(dataset.worlds), len(dataset.sequence)
            )
            return [prepare_pair(dataset, int(index)) for index in indices]

        pending = loader.submit(load_update, step + 1)
        for update in range(step + 1, args.steps + 1):
            pairs = pending.result()
            if update < args.steps:
                pending = loader.submit(load_update, update + 1)
            stats = run_update(
                model, opt, pairs, update, args.halve_at, args.balance, args.tail_weight
            )
            for pair in pairs:
                record_conditions(exposure, pair.frame, pair.world, pair.groups)
                sources.add(pair.frame)
                worlds.add(pair.world)
                world_frames.add((pair.world, pair.frame))
                instances.update(
                    (pair.world, pair.frame, row["instance"]) for row in pair.groups
                )
            for key in (
                "zero_anomaly_requests",
                "few_return_instances",
                "visible_instances",
            ):
                exposure[key] += stats[key]
            exposure.update(
                sources=sorted(sources),
                worlds=sorted(worlds),
                world_frames=sorted(world_frames),
                instance_observations=sorted(instances),
                requests=update * 8,
                clipped_updates=exposure["clipped_updates"] + stats["clipped"],
                training_seconds=exposure.get("training_seconds", 0.0)
                + stats["seconds"],
            )
            stats["coverage"] = dict(
                sources=len(sources),
                source_fraction=len(sources) / len(dataset.sequence),
                worlds=len(worlds),
                world_fraction=len(worlds) / len(dataset.worlds),
                instance_observations=len(instances),
                world_frames=len(world_frames),
                world_frame_fraction=len(world_frames) / len(dataset),
                nonempty_request_fraction=1
                - exposure["zero_anomaly_requests"] / (update * 8),
            )
            stats["target_exposure"] = coverage_status(exposure, coverage_plan, update)
            stats["elapsed_seconds"] = time.perf_counter() - start
            stats["elapsed_scope"] = (
                "current invocation, including loading and evaluation"
            )
            stats["cumulative_training_seconds"] = exposure["training_seconds"]
            log.write(json.dumps(stats, allow_nan=False) + "\n")
            log.flush()
            print(
                json.dumps(
                    dict(
                        update=update,
                        losses=stats["loss_terms"],
                        seconds=stats["seconds"],
                    )
                ),
                flush=True,
            )
            kind = evaluation_kind(
                update,
                final=args.final and update == args.steps,
                exposure_node=coverage_plan["common_node"],
            )
            if kind or update == args.steps:
                save_checkpoint(
                    output / f"{update}.pt", model, opt, update, metadata, exposure
                )
            if kind:
                result = evaluate(model, panels, args.data_root, args.samples, kind)
                write_json(
                    output / f"{update}.json",
                    report(update, result),
                )
            if update % 16 == 0:
                # Stop before further writes if the host reserve is being consumed.
                snapshot = resources(expected_disk_bytes=400_000_000)
                log.write(json.dumps(dict(update=update, resources=snapshot)) + "\n")
                log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/jasongao/Data/STU")
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--panels", type=Path, default=Path("results/panels.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("P", "T", "R"), default="P")
    parser.add_argument("--mechanism", choices=("D0", "D1", "D2", "D3"), default="D3")
    parser.add_argument(
        "--tail-weight", type=float, choices=(0.0, 0.25, 0.5), default=0.5
    )
    parser.add_argument("--coverage", type=Path, default=Path("results/coverage.json"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--balance", choices=("instance", "frame"), default="instance")
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--new-lr", type=float, default=1e-4)
    parser.add_argument("--halve-at", type=int, nargs="*", default=[])
    parser.add_argument("--reason", default="")
    parser.add_argument(
        "--fixed-recipe",
        type=Path,
        help="shared D0-D3/risk-control recipe JSON, including steps and LR trajectory",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="evaluate selected final weights on the complete real scope",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
