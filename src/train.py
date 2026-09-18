"""Run the fixed F240-R1 budget. --check validates without updating any parameter."""

import argparse
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
from torch.utils.data import DataLoader

from .data import VERSION, file_sha256, identity, load_manifest, read_scan, write_json
from .evaluate import (PreparedScans, autocast, better, evaluate, infer, memory_available,
                       precision, summarize)
from .model import (POINT_CHUNK, Segmentor, balanced_loss, scatter_scores, to_device,
                    LITEPT_COMMIT, WEIGHTS_REVISION, WEIGHTS_SHA256)
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


EPOCHS = (50, 20)
BATCH_SIZE = 8
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


def effective_batches(order, rank=0, world_size=1):
    for start in range(0, len(order), BATCH_SIZE):
        yield order[start:start + BATCH_SIZE][rank::world_size]


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


def optimizer_for(model, stage):
    groups = {}
    normalizations = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    no_decay = {id(p) for m in model.modules() if isinstance(m, normalizations)
                for p in m.parameters(recurse=False)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"all F240-R1 model parameters must be trainable: {name}")
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


def configuration(train, val, device, world_size):
    return dict(version=VERSION, train_manifest=train["sha256"], val_manifest=val["sha256"],
                val_directory=val["directory"], samples=len(train["records"]), epochs=EPOCHS,
                batch_size=BATCH_SIZE, microbatch=1, peak_lr=PEAK_LR, weight_decay=.005,
                adam_betas=(.9, .999), adam_eps=1e-8, gradient_clip=1.,
                warmup_fraction=.05, initial_lr_fraction=.1, final_lr_fraction=.01,
                augmentation=False, point_chunk=POINT_CHUNK, grid_size=.05,
                precision=str(precision(device)), sparse_precision="float32", world_size=world_size,
                code=code_record())


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
    return dict(version=VERSION, mode=model.mode, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                rng=states, config=config, **state, **extra)


def validate_all(model, val, device, workers):
    rank, _ = rank_info()
    sync_buffers(model)
    result = None
    if rank == 0:
        try:
            result = dict(ok=True, result=evaluate(model, val, device, workers))
        except Exception as error:
            result = dict(ok=False, error=f"{type(error).__name__}: {error}")
    result = broadcast_object(result)
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result["result"]


def write_result(directory, state, config):
    write_json(directory / "result.json", dict(version=VERSION, seed=state["seed"], method=state["method"],
               best_epoch=state["best_epoch"], best_metrics=state["best_metrics"],
               planned_updates=state["planned_updates"], successful_updates=state["successful_updates"],
               overflows=state["overflows"], parameters=state["parameters"],
               train_manifest_sha256=config.get("train_manifest"), val_manifest_sha256=config.get("val_manifest"),
               validation_improved_from_epoch0=(state["best_epoch"] > 0) if state["stage"] == 2 else None))


def train_stage(args, train, val, seed, method, device, config):
    global STOP
    rank, world_size = rank_info()
    stage = 1 if method == "base" else 2
    mode = "base" if method in ("base", "continue") else method
    directory = args.output / str(seed) / method
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    last, best_path = directory / "last.pt", directory / "best.pt"
    seed_all(seed)
    model = Segmentor(mode)
    load_record = None
    resume = last.exists()
    if resume:
        if not args.resume:
            raise ValueError(f"{last} exists; use --resume to continue its exact state")
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if identity(saved["config"]) != identity(config) or saved["seed"] != seed or saved["method"] != method:
            raise ValueError("resume configuration, code, dependencies or data differ")
        if saved["complete"]:
            if rank == 0:
                write_result(directory, saved, config)
                print(f"completed: seed={seed} method={method}", flush=True)
            return True
        model.load_state_dict(saved["model"], strict=True)
    elif stage == 1:
        load_record = model.load_pretrained(args.weights)
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
    scaler = torch.amp.GradScaler("cuda", enabled=precision(device) == torch.float16)
    steps_per_epoch = math.ceil(len(train["records"]) / BATCH_SIZE)
    total = EPOCHS[stage - 1] * steps_per_epoch
    state = dict(seed=seed, method=method, stage=stage, epoch=0, next_batch=0, planned_updates=0,
                 successful_updates=0, overflows=0, best_epoch=None, best_metrics=None,
                 epoch_loss=0., epoch_points=[0, 0], epoch_frames=0, complete=False,
                 parameters=sum(p.numel() for p in model.parameters()))
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        for key in state:
            state[key] = saved[key]
        restore_rng(saved["rng"][rank], device)
        del saved
    else:
        seed_all(seed + stage * 1000 + rank * 10000)
        if rank == 0:
            write_json(directory / "config.json", dict(configuration=config, load=load_record,
                                                       seed=seed, method=method, parameters=state["parameters"]))
        if stage == 2:
            result = validate_all(model, val, device, args.workers)
            state.update(best_metrics=result["metrics"], best_epoch=0)
            saved = capture(model, optimizer, scaler, state, config, device, selected=True, validation=result)
            if rank == 0:
                atomic_save(best_path, saved)
                atomic_save(last, dict(saved, selected=False))
                write_json(directory / "epoch0.json", result)
            del saved
    dataset = PreparedScans(train)
    for epoch in range(state["epoch"], EPOCHS[stage - 1]):
        start = time.perf_counter()
        order = epoch_order(len(dataset), seed, stage, epoch)
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
                group["lr"] = group["peak_lr"] * lr_factor(step, total)
            sync_buffers(model)
            loss_sum = torch.zeros((), device=device)
            for sample in samples:
                batch = to_device(sample, device)
                with autocast(device):
                    prediction = model(batch)
                    loss = balanced_loss(prediction, batch["targets"], counts)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite segmentation loss")
                scaler.scale(loss).backward()
                loss_sum += loss.detach()
                del batch, prediction, loss
            sync_gradients(model)
            if world_size > 1:
                dist.all_reduce(loss_sum)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            finite = bool(torch.isfinite(norm))
            if not scaler.is_enabled() and not finite:
                raise FloatingPointError("nonfinite BF16 gradient; no silent skipped update")
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.is_enabled() and scaler.get_scale() < before
            state["planned_updates"] += 1
            state["successful_updates"] += int(not overflow)
            state["overflows"] += int(overflow)
            state["epoch_loss"] += loss_sum.item()
            state["epoch_points"] = [a + b for a, b in zip(state["epoch_points"], counts.tolist())]
            state["epoch_frames"] += len(global_indices)
            state["next_batch"] = batch_number + 1
            need_stop = torch.tensor(int(STOP), device=device)
            if world_size > 1:
                dist.all_reduce(need_stop, op=dist.ReduceOp.MAX)
            check_disk = step % args.save_every == 0
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
                saved = capture(model, optimizer, scaler, state, config, device, selected=False)
                if rank == 0:
                    atomic_save(last, saved)
                del saved
            if rank == 0 and (step % 25 == 0 or overflow or need_stop):
                row = dict(event="update", seed=seed, method=method, epoch=epoch + 1,
                           batch=batch_number + 1, batches=steps_per_epoch, loss=loss_sum.item(),
                           normal=int(counts[0]), anomaly=int(counts[1]), planned=step,
                           successful=state["successful_updates"], overflow=bool(overflow),
                           lr=[g["lr"] for g in optimizer.param_groups],
                           elapsed_seconds=time.perf_counter() - start,
                           peak_vram_bytes=torch.cuda.max_memory_allocated(device))
                with (directory / "log.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
            if need_stop:
                return False
        del iterator, loader
        if state["epoch_frames"] != len(dataset):
            raise ValueError("epoch did not visit every fixed sample exactly once")
        # Save completed training before validation so an evaluation failure is resumable.
        state["next_batch"] = steps_per_epoch
        saved = capture(model, optimizer, scaler, state, config, device, selected=False)
        if rank == 0:
            atomic_save(last, saved)
        del saved
        result = validate_all(model, val, device, args.workers)
        selected = better(result["metrics"], state["best_metrics"])
        if selected:
            state.update(best_metrics=result["metrics"], best_epoch=epoch + 1)
        report = dict(epoch=epoch + 1, mean_batch_loss=state["epoch_loss"] / steps_per_epoch,
                      frames=state["epoch_frames"], class_points=state["epoch_points"],
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
            write_json(directory / f"epoch{epoch + 1}.json", report)
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
    del saved, model, optimizer, dataset
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
    return True


def preflight(args, train, val, device, config, resources):
    """Real-data forward/backward acceptance; no optimizer or parameter updates."""
    seed_all(0)
    dataset = PreparedScans(train)
    index = max(range(len(dataset)), key=lambda i: train["records"][i]["points"])
    sample = to_device(dataset[index], device)
    model = Segmentor().to(device).eval()
    loaded = model.load_pretrained(args.weights)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), autocast(device):
        original = model(sample)
        stored = scatter_scores(original, sample["slots"], sample["slot_count"])
    if len(original) != train["records"][index]["points"] or not torch.isfinite(original).all():
        raise ValueError("missing or nonfinite full-scan predictions")
    empty = torch.ones(sample["slot_count"], dtype=torch.bool, device=device)
    empty[sample["slots"]] = False
    if torch.count_nonzero(stored[empty]):
        raise ValueError("empty-slot predictions must equal zero")
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    records = {}
    for mode in ("base", "attention", "fusion"):
        if mode != "base":
            del model
            model = Segmentor(mode).to(device)
            current = model.state_dict()
            current.update(base_state)
            model.load_state_dict(current, strict=True)
        model.eval()
        with torch.no_grad(), autocast(device):
            prediction = model(sample)
        delta = float((prediction - original).abs().max())
        if not torch.equal(prediction, original):
            raise ValueError(f"zero-initialized stage switch changed inference: {mode}, max={delta}")
        model.train()
        model.zero_grad(set_to_none=True)
        counts = torch.stack([(sample["targets"] == label).sum() for label in (0, 1)])
        with autocast(device):
            prediction = model(sample)
            loss = balanced_loss(prediction, sample["targets"], counts)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                          for p in model.parameters()):
            raise ValueError(f"nonfinite full-scan backward: {mode}")
        if model.detail[0].weight.grad is None or model.backbone.embedding.stem.conv.weight.grad is None:
            raise ValueError("the detail branch or backbone is detached")
        records[mode] = dict(loss=float(loss.detach()), stage_switch_max_error=delta,
                             parameters=sum(p.numel() for p in model.parameters()),
                             peak_vram_bytes=torch.cuda.max_memory_allocated(device))
        del prediction, loss
    model.zero_grad(set_to_none=True)
    del sample
    eligible = [r for r in val["records"] if r["eligible"]]
    chosen = [eligible[0], eligible[len(eligible) // 2], eligible[-1]]
    subset = dict(val, records=chosen)
    subset.pop("sha256")
    subset["sha256"] = identity(subset)
    # This subset checks the evaluator, not validation performance or model selection.
    observed = evaluate(model, subset, device, args.workers)
    official = PointOODMetricsCalculator()
    for row in chosen:
        frame = read_scan(row["scan"], row["label"])
        prediction, _ = infer(model, row["scan"], device)
        official.update(frame.xyzi[:, :3], prediction, frame.semantic)
    reference = {k: float(v) for k, v in official.compute_metrics().items()}
    if observed["metrics"] != reference:
        raise ValueError("pooled evaluation differs from independent official raw-slot evaluation")
    write_json(args.output / "preflight.json", dict(version=VERSION, configuration=config,
               resources=resources, sample=dict(index=index, world=train["records"][index]["world"],
                                                frame=train["records"][index]["frame"]),
               loaded=loaded, checks=records, parameter_updates=0,
               evaluation_check=dict(diagnostic_only=True, scans=len(chosen), points=observed["points"],
                    official_metrics_exact_match=True, full_val_manifest=val["sha256"],
                    identities=[dict(sequence=r["sequence"], frame=r["frame"]) for r in chosen])))
    print(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("assets/train.json"))
    parser.add_argument("--val-manifest", type=Path, default=Path("assets/val.json"))
    parser.add_argument("--weights", type=Path, default=Path("/home/jasongao/Study/AJAE/results/pretrain/nuscenes.pth"))
    parser.add_argument("--output", type=Path, default=Path("results/train"))
    parser.add_argument("--seeds", type=int, nargs="+", choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.workers < 0 or args.threads < 1 or args.save_every < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("invalid runtime resource settings or duplicate seeds")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        parser.error("LitePT sparse convolution/FlashAttention requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
    rank, world_size = rank_info()
    if world_size > BATCH_SIZE:
        parser.error("at most eight GPUs for effective batch eight")
    train, val = load_manifest(args.train_manifest, "train"), load_manifest(args.val_manifest, "val")
    if len(train["worlds"]) != 240:
        parser.error("F240-R1 requires all 240 training worlds")
    config = configuration(train, val, device, world_size)
    resources = runtime_snapshot() if rank == 0 else None
    if args.workers * world_size + args.threads * world_size > len(os.sched_getaffinity(0)):
        parser.error("worker and BLAS thread counts exceed the available CPU affinity")
    processes = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(processes, os.getpid())
    else:
        processes = [os.getpid()]
    gpu_processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()
    other = [int(p.strip()) for p in gpu_processes if p.strip().isdigit() and int(p.strip()) not in processes]
    if other:
        raise RuntimeError(f"other CUDA processes must finish before this run: {other}")
    # All twelve runs retain best/last optimizer states; include atomic replacement.
    disk_check(6_000_000_000 if not args.check else 100_000_000)
    free, _ = torch.cuda.mem_get_info(device)
    if free < 7_000_000_000:
        raise RuntimeError(f"full-scan training verification needs a free GPU; only {free / 1e9:.1f} GB available")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "resources.json", resources)
    if args.check:
        if world_size != 1:
            parser.error("--check uses one GPU; accumulation equivalence is tested separately")
        preflight(args, train, val, device, config, resources)
        return
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    for seed in args.seeds:
        for method in ("base", "attention", "continue", "fusion"):
            if not train_stage(args, train, val, seed, method, device, config):
                return
    if rank == 0 and set(args.seeds) == {0, 1, 2}:
        summarize(args.output)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
