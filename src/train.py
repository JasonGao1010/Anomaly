"""Train V3 with source-uniform paired requests and fixed development panels."""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from .evaluate import evaluate, evaluation_kind, instance_rows, write_json
from .model import (
    V3,
    SEED,
    configure_runtime,
    lr_factor,
    optimizer,
    paired_loss,
    prepare_scan,
    transfer,
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


def prepare_pair(dataset, index):
    original, rendered = dataset.pair(int(index))
    positive = rendered.source
    target = detection_targets(positive, inserted=rendered.inserted_mask)
    world = dataset.worlds[index // len(dataset.sequence)]
    metadata = {
        "60001": dict(height_m=world["height_m"], height_source=world["height_source"])
    }
    return PairInput(
        (
            prepare_scan(original.xyzi[original.observation_slots]),
            prepare_scan(positive.xyzi[positive.observation_slots]),
        ),
        target,
        detection_targets(original),
        positive.labels.instance[positive.observation_slots],
        original.frame_id,
        rendered.world_identity,
        instance_rows(positive, target, metadata),
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


def bn_state(model):
    return {
        name: torch.cat(
            (module.running_mean.detach(), module.running_var.detach())
        ).clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm1d)
    }


def run_update(model, opt, pairs, update, halvings=(), balance="instance"):
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
    terms, scores, grouped = [], [], {}
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
            loss, parts = paired_loss(positive, negative, pt, nt, identity, balance)
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
    gradient = nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    conditions = {}
    for name, module in model.named_modules():
        if hasattr(module, "condition") and module.condition is not None:
            conditions[name] = dict(
                delta_rms=float(module.diagnostics["delta_rms"]),
                gradient_norm=float(
                    torch.stack(
                        [p.grad.square().sum() for p in module.condition.parameters()]
                    )
                    .sum()
                    .sqrt()
                ),
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
                ("anomaly", "inserted_normal", "original_normal"),
                loss_terms.mean(0).tolist(),
            )
        ),
        loss_term_std=dict(
            zip(
                ("anomaly", "inserted_normal", "original_normal"),
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
        conditions=conditions,
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
        format="ajae-v3",
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
                "A/B/C must use exactly the shared recipe's seed, initialization, risk, rates, and budget"
            )
    if args.halve_at and (
        not args.reason or args.mechanism != "A" and not args.fixed_recipe
    ):
        raise ValueError(
            "record the decay reason; mechanism controls require a common fixed recipe"
        )
    if args.mechanism != "A" and not args.fixed_recipe:
        raise ValueError(
            "A/B/C controls require the same predeclared budget and learning-rate trajectory"
        )
    lr_factor(1, args.halve_at)
    checkpoints = sum(evaluation_kind(i) is not None for i in range(args.steps + 1)) + 1
    snapshot = resources(expected_disk_bytes=(checkpoints + 1) * 200_000_000)
    configure_runtime()
    torch.manual_seed(args.seed)
    dataset = FrozenDataset(args.samples, args.data_root, "train")
    model = V3(args.mechanism, args.seed)
    recipe = dict(
        balance=args.balance,
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
    )
    metadata = dict(
        route=args.route,
        panels=panels,
        recipe=recipe,
        initialization=None,
        decisions=[
            dict(
                from_step=0,
                to_step=args.steps,
                reason=args.reason or "first V3 exploration budget",
            )
        ],
        resources=snapshot,
        numerics=dict(
            dtype="float32",
            tf32=False,
            activation_recomputation="BN-free blocks",
            torch=str(torch.__version__),
        ),
    )
    step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if any(
            saved[key] != value
            for key, value in (
                ("mechanism", args.mechanism),
                ("seed", args.seed),
                ("route", args.route),
                ("panels", panels),
            )
        ):
            raise ValueError(
                "resume would change the scientific route, seed, or fixed panels"
            )
        old_recipe = saved["recipe"]
        if any(
            old_recipe[key] != recipe[key]
            for key in ("balance", "backbone_lr", "new_lr", "fixed_recipe")
        ):
            raise ValueError(
                "changing the training risk or base rates requires a new route"
            )
        if [v for v in args.halve_at if v <= saved["step"]] != [
            v for v in old_recipe["halve_at"] if v <= saved["step"]
        ]:
            raise ValueError("a continuation cannot rewrite past learning rates")
        if not args.reason:
            raise ValueError(
                "record the scientific reason for extending or resuming training"
            )
        model.load_state_dict(saved["model"], strict=True)
        metadata["initialization"] = saved["initialization"]
        metadata["decisions"] = saved["decisions"] + [
            dict(from_step=saved["step"], to_step=args.steps, reason=args.reason)
        ]
        exposure, step = saved["exposure"], saved["step"]
        if args.steps <= step:
            raise ValueError("continuation must extend beyond the saved update")
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
            ),
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
            stats = run_update(model, opt, pairs, update, args.halve_at, args.balance)
            for pair in pairs:
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
            stats["elapsed_seconds"] = time.perf_counter() - start
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
            kind = evaluation_kind(update, final=args.final and update == args.steps)
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
    parser.add_argument("--mechanism", choices=("A", "B", "C"), default="A")
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
        help="shared A/B/C recipe JSON, including steps and LR trajectory",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="evaluate selected final weights on the complete real scope",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
