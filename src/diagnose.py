"""Paired checkpoint errors, exact rank contributions and fixed-batch gradients."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Scans, file_sha256, identity, load_manifest, write_json
from .evaluate import PreparedScans, autocast, evaluate, load_model
from .model import balanced_loss, rank_sample, ranking_loss, to_device
from .train import (disk_check, forward_loss, lr_factor, optimizer_for, restore_rng,
                    rng_state, runtime_snapshot, seed_all)


ROOT = Path("results/train/native")
OUTPUT = ROOT / "diagnostics"
CHECKPOINTS = {"m500": ROOT / "metrics/0/conditional/best.pt",
               "m1547": ROOT / "metrics/0/conditional/last.pt",
               "b1547": ROOT / "bce/0/conditional/last.pt"}
RECALLS = (.5, .75, .9, .95, .99)
RECALL_BINS = (0., .25, .5, .75, .9, .95, .99, 1.)
META = np.dtype([("target", "u1"), ("slot", "u4"), ("semantic", "u2"),
                 ("instance", "u2"), ("range", "f4")])
BEST_C = ROOT / "branches/lr/0/conditional/best.pt"


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_sets(output):
    """Choose by source identity before seeing scores, never by observed errors."""
    train = load_manifest("results/data/native/train.json", "train")
    groups = defaultdict(list)
    for i, row in enumerate(train["records"]):
        groups[row["group"]].append(i)
    indices = []
    for group, rows in sorted(groups.items()):
        # Spread 32 observations across each source's stable record order.
        indices.extend(rows[i] for i in np.linspace(0, len(rows) - 1, 32).round().astype(int))
    fixed = dict(train, records=[train["records"][i] for i in sorted(indices)])
    fixed.pop("sha256")
    fixed.update(parent_manifest=train["sha256"], parent_indices=sorted(indices),
                 diagnostic_selection="32 evenly spaced records per source; fixed before scores")
    fixed["sha256"] = identity(fixed)
    normal = load_manifest("results/data/normal.json", "normal")
    check = dict(train, records=[r for r in normal["records"] if r["subset"] == "check"])
    worlds = json.loads(Path("results/data/targeted.json").read_text())["worlds"]
    for world in worlds:
        if world["accepted"] and world["subset"] == "check":
            check["records"].extend(json.loads(Path(world["path"]).read_text())["frames"])
    if any(r["subset"] != "check" for r in check["records"]):
        raise ValueError("internal checks contain training records")
    check.pop("sha256")
    check.update(diagnostic_selection="all reserved nuScenes logs and all three reserved 206 geometries",
                 limitations="normal-background and same-206 new-geometry checks, not cross-scene anomaly validation")
    check["sha256"] = identity(check)
    for name, manifest in (("train", fixed), ("check", check)):
        write_json(output / f"{name}.json", manifest)
    return dict(val=load_manifest("assets/val.json", "val"), train=fixed, check=check)


def frame_metadata(sample, row, split):
    mask = np.asarray(sample["targets"]) >= 0
    out = np.empty(int(mask.sum()), META)
    out["target"] = np.asarray(sample["targets"])[mask]
    out["slot"] = np.asarray(sample["slots"])[mask]
    out["range"] = np.linalg.norm(np.asarray(sample["xyzi"])[mask, :3], axis=1)
    out["semantic"], out["instance"] = 0, 0
    if split == "val":
        packed = np.fromfile(row["label"], dtype="<u4")[out["slot"]]
        out["semantic"], out["instance"] = packed & 65535, packed >> 16
        if not np.array_equal(out["target"], (out["semantic"] == 2).astype(np.uint8)):
            raise ValueError("diagnostic metadata does not match official labels")
    return out


def metadata(manifest, split, output, workers):
    indices = [i for i, r in enumerate(manifest["records"]) if split != "val" or r["eligible"]]
    count = sum(manifest["records"][i]["normal"] + manifest["records"][i]["anomaly"] for i in indices)
    path = output / f"{split}_points.npy"
    offsets_path = output / f"{split}_offsets.json"
    if path.exists() and offsets_path.exists():
        saved = json.loads(offsets_path.read_text())
        if saved["manifest"] != manifest["sha256"]:
            raise ValueError("diagnostic point identity changed")
        return
    result = np.lib.format.open_memmap(path, mode="w+", dtype=META, shape=(count,))
    loader = DataLoader(Scans(manifest), batch_size=None, sampler=indices, num_workers=workers,
                        **({"prefetch_factor": 1} if workers else {}),
                        generator=torch.Generator().manual_seed(0))
    cursor, rows = 0, []
    for sample in loader:
        index = int(sample["index"])
        row = manifest["records"][index]
        values = frame_metadata(sample, row, split)
        stop = cursor + len(values)
        result[cursor:stop] = values
        rows.append(dict(index=index, start=cursor, stop=stop, sequence=row.get("sequence"),
                         frame=row["frame"], group=row.get("group", "STU"),
                         scene=row.get("scene"), geometry=row.get("geometry"), world=row.get("world")))
        cursor = stop
    if cursor != count:
        raise ValueError("diagnostic point count changed")
    result.flush()
    write_json(offsets_path, dict(manifest=manifest["sha256"], points=count, rows=rows,
        identity="offset row -> manifest record; slot -> original return record",
        instance_scope="sequence plus original annotation instance; not a verified physical track"))


@torch.no_grad()
def predict_fixed(model, manifest, path, device, workers):
    count = sum(r["normal"] + r["anomaly"] for r in manifest["records"])
    scores = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(count,))
    loader = DataLoader(PreparedScans(manifest), batch_size=None, num_workers=workers,
                        pin_memory=True, **({"prefetch_factor": 1} if workers else {}),
                        generator=torch.Generator().manual_seed(0))
    model.eval()
    start, cursor = time.perf_counter(), 0
    for sample in loader:
        with autocast(device):
            prediction = model(to_device(sample, device))
        selected = prediction.cpu().numpy()[sample["targets"].numpy() >= 0]
        scores[cursor:cursor + len(selected)] = selected
        cursor += len(selected)
    if cursor != count or not np.isfinite(scores).all():
        raise ValueError("incomplete or nonfinite diagnostic predictions")
    scores.flush()
    return dict(scans=len(manifest["records"]), points=count, seconds=time.perf_counter() - start,
                manifest_sha256=manifest["sha256"],
                scope="diagnostic supervised points, including trusted normal frames; not official validation")


def collect(output, workers, checkpoints=CHECKPOINTS):
    output.mkdir(parents=True, exist_ok=True)
    disk_check(12_000_000_000)
    write_json(output / "resources.json", runtime_snapshot())
    manifests = diagnostic_sets(output)
    for split, manifest in manifests.items():
        metadata(manifest, split, output, workers)
    device = torch.device("cuda")
    for name, path in checkpoints.items():
        model, saved = load_model(path, device)
        record = dict(checkpoint=str(path), checkpoint_sha256=file_sha256(path),
                      update=saved["successful_updates"], splits={})
        for split, manifest in manifests.items():
            destination = output / f"{name}_{split}.npy"
            print(f"diagnostic inference {name} {split}", flush=True)
            if split == "val":
                result = evaluate(model, manifest, device, workers, score_path=destination)
                expected = saved["final_metrics"]
                result["saved_metrics"] = expected
                result["metric_differences"] = {k: result["metrics"][k] - expected[k]
                                                 for k in ("AP", "FPR95", "AUROC")}
            else:
                result = predict_fixed(model, manifest, destination, device, workers)
            record["splits"][split] = result
            write_json(output / f"{name}.json", record)
        del model, saved
        gc.collect()
        torch.cuda.empty_cache()


def score_curve(scores, labels):
    """Exact equal-score groups, not bins: retain ties and every PR step."""
    values, positives, negatives = [], [], []
    for start in range(0, len(scores), 1_000_000):
        stop = min(start + 1_000_000, len(scores))
        unique, inverse = np.unique(scores[start:stop], return_inverse=True)
        pos = np.bincount(inverse[np.asarray(labels[start:stop]) == 1], minlength=len(unique))
        neg = np.bincount(inverse[np.asarray(labels[start:stop]) == 0], minlength=len(unique))
        values.append(unique)
        positives.append(pos)
        negatives.append(neg)
    unique, inverse = np.unique(np.concatenate(values), return_inverse=True)
    positive = np.zeros(len(unique), np.int64)
    negative = np.zeros(len(unique), np.int64)
    np.add.at(positive, inverse, np.concatenate(positives))
    np.add.at(negative, inverse, np.concatenate(negatives))
    tp, fp = np.cumsum(positive[::-1]), np.cumsum(negative[::-1])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1] if tp[-1] else np.zeros_like(precision)
    previous = np.r_[0., recall[:-1]]
    ap = float(np.sum((recall - previous) * precision)) if tp[-1] else None
    fpr = fp / fp[-1] if fp[-1] else np.zeros_like(precision)
    auc = float(np.trapezoid(np.r_[0., recall], np.r_[0., fpr])) if tp[-1] and fp[-1] else None
    operating = []
    if tp[-1]:
        for target in RECALLS:
            index = int(np.searchsorted(recall, target, side="right"))
            operating.append(dict(target=target, recall=float(recall[index]), threshold=float(unique[::-1][index]),
                                  tp=int(tp[index]), fp=int(fp[index]), fn=int(tp[-1] - tp[index]),
                                  fpr=float(fpr[index]), precision=float(precision[index])))
    bands = [float(np.sum(np.maximum(0., np.minimum(recall, high) - np.maximum(previous, low)) * precision))
             for low, high in zip(RECALL_BINS[:-1], RECALL_BINS[1:])]
    return dict(values=unique, positive=positive, negative=negative, precision=precision, recall=recall,
                tp=tp, fp=fp, AP=None if ap is None else 100 * ap,
                AUROC=None if auc is None else 100 * auc,
                FPR95=100 * operating[3]["fpr"] if operating else None,
                operating=operating, AP_bands=[100 * x for x in bands])


def curve_summary(curve):
    return {key: curve[key] for key in ("AP", "AUROC", "FPR95", "operating", "AP_bands")}


def analyze(output, names):
    summaries, curves = {}, {}
    for split in ("val", "train", "check"):
        points = np.load(output / f"{split}_points.npy", mmap_mode="r")
        offsets = json.loads((output / f"{split}_offsets.json").read_text())["rows"]
        for name in names:
            score_path = output / f"{name}_{split}.npy"
            if split != "val" and not score_path.exists():
                continue
            scores = np.load(score_path, mmap_mode="r")
            curve = score_curve(scores, points["target"])
            curves[name, split] = curve
            summaries.setdefault(name, {})[split] = curve_summary(curve)
            print(name, split, {k: curve[k] for k in ("AP", "FPR95", "AUROC")}, flush=True)
            if split == "val":
                if name in ("replay", "ap", "lr", "control", "hard"):
                    mining = name in ("control","hard")
                    directory = ROOT / ("mining" if mining else "branches") / name / "0/conditional"
                    result = json.loads((directory / "result.json").read_text())
                    update = 1500 if mining else 1000
                    if not result["complete"] or result["successful_updates"] != update:
                        raise ValueError("short branch is unfinished")
                    validation = json.loads((directory / f"epoch{update//500}.json").read_text())["validation"]
                    write_json(output / f"{name}.json", dict(checkpoint=str(directory / "last.pt"),
                        checkpoint_sha256=file_sha256(directory / "last.pt"), update=update,
                        splits=dict(val=validation), training=result))
                official = json.loads((output / f"{name}.json").read_text())["splits"]["val"]["metrics"]
                if any(abs(curve[k] - official[k]) > 1e-9 for k in ("AP", "FPR95", "AUROC")):
                    raise ValueError("exact grouped-score analysis differs from official metrics")
                rows = [dict(threshold=float(s), recall=float(r), precision=float(p), tp=int(t), fp=int(f))
                        for s, r, p, t, f in zip(curve["values"][::-1], curve["recall"],
                            curve["precision"], curve["tp"], curve["fp"])]
                write_csv(output / f"{name}_pr.csv", rows)
            else:
                sources = {}
                for group in sorted({r["group"] for r in offsets}):
                    selected = [r for r in offsets if r["group"] == group]
                    s = np.concatenate([scores[r["start"]:r["stop"]] for r in selected])
                    y = np.concatenate([points["target"][r["start"]:r["stop"]] for r in selected])
                    item = curve_summary(score_curve(s, y))
                    item.update(scans=len(selected), normal=int((y == 0).sum()), anomaly=int((y == 1).sum()))
                    threshold = curves[name, "val"]["operating"][3]["threshold"]
                    item["at_STU95"] = dict(threshold=threshold,
                        fp=int(((s >= threshold) & (y == 0)).sum()), fn=int(((s < threshold) & (y == 1)).sum()))
                    item["score_quantiles"] = {str(label): np.quantile(s[y == label], [.01, .05, .5, .95, .99]).tolist()
                                               for label in (0, 1) if (y == label).any()}
                    sources[group] = item
                summaries[name][split]["sources"] = sources
                if split == "check":
                    geometries = {}
                    for geometry in sorted({r["geometry"] for r in offsets if r["geometry"] is not None}):
                        selected = [r for r in offsets if r["geometry"] == geometry]
                        s = np.concatenate([scores[r["start"]:r["stop"]] for r in selected])
                        y = np.concatenate([points["target"][r["start"]:r["stop"]] for r in selected])
                        geometries[geometry] = dict(scans=len(selected), anomaly=int(y.sum()),
                                                   **curve_summary(score_curve(s, y)))
                    summaries[name][split]["geometries"] = geometries
    write_json(output / "curves.json", dict(models=summaries, recall_bins=RECALL_BINS,
        threshold_rule="score >= threshold at the first complete score tie with global recall strictly above target",
        AP_contribution="step precision times recall increment, split exactly at recall-bin boundaries",
        group_definitions=dict(distance=["[2.5,5)","[5,10)","[10,20)","[20,30)","[30,50]"],
            object_returns=["1-4","5-9","10-19","20-49","50-99","100-199","200+"],
            object="sequence:annotation instance; instance zero is unassigned; counts per scan within evaluation range",
            pooled_AP_credit="sum of global precision at each positive's score / all validation positives; additive, not subgroup AP")))
    grouped_errors(output, names, curves)
    plot_curves(output, names, curves)


def grouped_errors(output, names, curves):
    points = np.load(output / "val_points.npy", mmap_mode="r")
    offsets = json.loads((output / "val_offsets.json").read_text())["rows"]
    scores = {name: np.load(output / f"{name}_val.npy", mmap_mode="r") for name in names}
    totals, transitions = defaultdict(lambda: np.zeros(14)), defaultdict(lambda: np.zeros(4, np.int64))
    examples = []
    pairs = [(names[0], name) for name in names[1:]]
    if "replay" in names:
        pairs.extend(("replay", name) for name in ("ap", "lr") if name in names)
    pairs.extend((a,b) for a,b in (("lr","control"),("lr","hard"),("control","hard")) if a in names and b in names)
    for row in offsets:
        start, stop, seq = row["start"], row["stop"], row["sequence"]
        meta = points[start:stop]
        y = meta["target"].astype(bool)
        groups = [("sequence", str(seq), np.ones(len(y), bool))]
        distance = np.digitize(meta["range"], [5., 10., 20., 30.])
        groups.extend(("distance", str(i), distance == i) for i in np.unique(distance))
        for semantic in np.unique(meta["semantic"][~y]):
            groups.append(("normal_semantic", str(int(semantic)), meta["semantic"] == semantic))
        for inst in np.unique(meta["instance"][y]):
            mask = y & (meta["instance"] == inst)
            count = int(mask.sum())
            groups.append(("object" if inst else "unassigned_anomaly", f"{seq}:{inst}", mask))
            if inst:
                groups.append(("object_returns", str(int(np.digitize(count, [5, 10, 20, 50, 100, 200]))), mask))
        predictions = {}
        for name in names:
            curve = curves[name, "val"]
            s = scores[name][start:stop]
            predictions[name] = np.stack([s >= op["threshold"] for op in curve["operating"]])
            # Each positive's precision credit sums to pooled AP, including ties.
            credit = np.zeros(len(s))
            credit[y] = curve["precision"][::-1][np.searchsorted(curve["values"], s[y])] / curve["tp"][-1] * 100
            for kind, key, mask in groups:
                values = totals[name, kind, key]
                values[:4] += [int((mask & ~y).sum()), int((mask & y).sum()), credit[mask].sum(), 1]
                for k, p in enumerate(predictions[name]):
                    values[4 + k * 2:6 + k * 2] += [int((mask & ~y & p).sum()), int((mask & y & ~p).sum())]
        for first, second in pairs:
            for k, target in enumerate(RECALLS):
                a, b = predictions[first][k], predictions[second][k]
                changes = (~y & ~a & b, ~y & a & ~b, y & a & ~b, y & ~a & b)
                for kind, key, mask in groups:
                    transitions[first, second, target, kind, key] += [int((mask & c).sum()) for c in changes]
                if target in (.5, .95):
                    for label, mask in zip(("new_fp", "fixed_fp", "new_fn", "fixed_fn"), changes):
                        selected = np.flatnonzero(mask)
                        # A deterministic middle-slot representative per error direction.
                        if len(selected):
                            j = int(selected[len(selected) // 2])
                            examples.append(dict(first=first, second=second, target_recall=target, change=label,
                                sequence=seq, frame=row["frame"], slot=int(meta["slot"][j]),
                                semantic=int(meta["semantic"][j]), instance=int(meta["instance"][j]),
                                range_m=float(meta["range"][j]), count_in_scan=len(selected),
                                first_score=float(scores[first][start+j]), second_score=float(scores[second][start+j])))
    rows = []
    for (model, kind, key), v in sorted(totals.items()):
        row = dict(model=model, kind=kind, key=key, normal=int(v[0]), anomaly=int(v[1]),
                   pooled_AP_credit=float(v[2]), observations=int(v[3]))
        for k, recall in enumerate(RECALLS):
            row[f"fp{int(100*recall)}"], row[f"fn{int(100*recall)}"] = map(int, v[4+k*2:6+k*2])
        rows.append(row)
    write_csv(output / "groups.csv", rows)
    write_csv(output / "changes.csv", [dict(first=a, second=b, target_recall=q, kind=kind, key=key,
        new_fp=int(v[0]), fixed_fp=int(v[1]), new_fn=int(v[2]), fixed_fn=int(v[3]))
        for (a,b,q,kind,key),v in sorted(transitions.items())])
    write_csv(output / "points.csv", examples)


def plot_curves(output, names, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager, findfont
    from matplotlib.ft2font import FT2Font
    path = "/mnt/c/Windows/Fonts/times.ttf"
    fontManager.addfont(path)
    plt.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in names:
        c = curves[name, "val"]
        axes[0].step(np.r_[0., c["recall"]], np.r_[1., c["precision"]], where="pre", label=f"{name}: AP {c['AP']:.2f}%")
        axes[1].plot(c["recall"], c["fp"], label=name)
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0,1), ylim=(0,1.02))
    axes[1].set(xlabel="Recall", ylabel="False-positive points", xlim=(.4,1), yscale="log", ylim=(1,2e6))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_text() and FT2Font(findfont(item.get_fontproperties(), fallback_to_default=False)).family_name != "Times New Roman":
            raise ValueError("unexpected rendered font")
    fig.savefig(output / "pr.pdf")
    fig.savefig(output / "pr.png", dpi=180)
    plt.close(fig)


def gradient_geometry(vectors):
    result = dict(norms={k: float(torch.linalg.vector_norm(v)) for k,v in vectors.items()}, cosines={})
    for a, b in (("ap", "auc"), ("ap", "fpr95"), ("ap", "bce")):
        denominator = torch.linalg.vector_norm(vectors[a]) * torch.linalg.vector_norm(vectors[b])
        result["cosines"][f"{a}:{b}"] = float(torch.dot(vectors[a], vectors[b]) / denominator) if denominator > 0 else None
    return result


def gradients(output):
    """Probe score and post-interaction feature derivatives without parameter updates."""
    device = torch.device("cuda")
    train = load_manifest("results/data/native/train.json", "train")
    dataset = PreparedScans(train)
    order = json.loads((ROOT / "metrics/0/conditional/sampling.json").read_text())["order"]
    initial = torch.load(CHECKPOINTS["m500"], map_location="cpu", weights_only=False)
    common_rng = initial["rng"][0]
    del initial
    steps = (501, 626, 751, 876)
    report = dict(steps=steps, definitions=dict(shared="64-dimensional voxel state after conditional interaction",
        weights="BCE uses effective-eight-scan counts; AP/AUC/FPR95 weights are 1/4, 0.1/4, 0.1/4 per pair",
        saturation="sigmoid <= 0.001 or >= 0.999; inverse-probability weighted normal comparisons",
        buffers="restore checkpoint model and common RNG before every fixed batch; no optimizer updates"), models={})
    records = train["records"]
    active = [(i, order[i:i+2]) for i in range(0,len(order),2) if i//8 + 1 >= 155]
    empty = sum(not any(records[j]["anomaly"] for j in pair) for _,pair in active)
    report["sampling"] = dict(ranking_enabled_pairs=len(active), no_anomaly_pairs=empty,
                              no_anomaly_fraction=empty/len(active))
    for name, path in CHECKPOINTS.items():
        model, saved = load_model(path, device)
        model.train()
        batches = []
        for step in steps:
            model.load_state_dict(saved["model"], strict=True)
            restore_rng(common_rng, device)
            model.zero_grad(set_to_none=True)
            indices = order[(step-1)*8:step*8]
            counts = torch.tensor([sum(records[i][key] for i in indices) for key in ("normal", "anomaly")], device=device)
            batch = dict(step=step, indices=indices, pairs=[])
            for k in range(4):
                pair_ids = indices[k*2:k*2+2]
                samples = [to_device(dataset[i],device) for i in pair_ids]
                shared = []
                hook = model.conditional.register_forward_hook(lambda module, args, result: shared.append(result))
                with autocast(device):
                    prediction = torch.cat([model(sample) for sample in samples])
                hook.remove()
                targets = torch.cat([s["targets"] for s in samples])
                bce = balanced_loss(prediction, targets, counts)
                rank, details = ranking_loss(prediction, targets, step*8+k, return_terms=True)
                terms = dict(bce=bce, **details.pop("terms"))
                weights = dict(bce=1., ap=.25, auc=.025, fpr95=.025)
                score_grads, feature_grads = {}, {}
                for term, loss in terms.items():
                    grad = torch.autograd.grad(loss * weights[term], (prediction, *shared), retain_graph=True)
                    score_grads[term] = grad[0].float()
                    feature_grads[term] = torch.cat([v.flatten().float() for v in grad[1:]])
                entry = dict(pair=k, indices=pair_ids, groups=[records[i]["group"] for i in pair_ids],
                    positive=int((targets==1).sum()), negative=int((targets==0).sum()),
                    loss={t: float(v.detach()) for t,v in terms.items()}, weights=weights,
                    scores=gradient_geometry(score_grads), features=gradient_geometry(feature_grads))
                if entry["positive"] and entry["negative"]:
                    with torch.no_grad():
                        p, n = prediction[targets==1].float(), prediction[targets==0].float()
                        anchors, sampled, w, top = rank_sample(p,n,step*8+k)
                        prob = torch.sigmoid(sampled[None,:]-anchors[:,None])
                        saturated = (prob<=.001) | (prob>=.999)
                        entry["saturation"] = dict(weighted_negative_pairs=float((saturated*w).sum()/len(anchors)/len(n)),
                            top_negative_pairs=float(saturated[:,:top].float().mean()),
                            positive_pairs=float(((torch.sigmoid(p[None,:]-anchors[:,None])<=.001) |
                                (torch.sigmoid(p[None,:]-anchors[:,None])>=.999)).float().mean()))
                source_grad = []
                cursor = 0
                for sample,index in zip(samples,pair_ids):
                    stop = cursor + len(sample["targets"])
                    source_grad.append(dict(group=records[index]["group"], index=index,
                        norms={term: {str(label):float(torch.linalg.vector_norm(g[cursor:stop][sample["targets"]==label]))
                                      for label in (0,1)} for term,g in score_grads.items()}))
                    cursor = stop
                entry["source_score_gradients"] = source_grad
                (bce + rank/4).backward()
                batch["pairs"].append(entry)
                del samples,prediction,targets,terms,score_grads,feature_grads,shared,grad,bce,rank,loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            batch.update(full_combined_gradient_norm=float(norm), would_clip=bool(norm>1))
            batches.append(batch)
            print(f"gradient diagnosis {name} fixed update {step}, norm {float(norm):.5f}",flush=True)
        report["models"][name] = batches
        write_json(output / "gradients.json", report)
        del model,saved
        gc.collect()
        torch.cuda.empty_cache()


def replay(output):
    """Two eight-update numerical probes from the identical complete state; save no models."""
    device = torch.device("cuda")
    train = load_manifest("results/data/native/train.json", "train")
    dataset = PreparedScans(train)
    order = json.loads((ROOT / "metrics/0/conditional/sampling.json").read_text())["order"]
    passes, states, parameters, predictions = [], [], [], []
    for repeat in range(2):
        seed_all(0)
        model,saved = load_model(CHECKPOINTS["m500"],device)
        optimizer = optimizer_for(model,1)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"][0],device)
        model.train()
        rows = []
        for step in range(501,509):
            indices = order[(step-1)*8:step*8]
            counts = torch.tensor([sum(train["records"][i][key] for i in indices) for key in ("normal","anomaly")],device=device)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * lr_factor(step,1547)
            total = 0.
            for k in range(4):
                samples=[to_device(dataset[i],device) for i in indices[k*2:k*2+2]]
                with autocast(device):
                    loss,details=forward_loss(model,samples,counts,rank_weight=.25,rank_seed=step*8+k)
                loss.backward()
                total+=float(loss.detach())
                del samples,loss,details
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            rows.append(dict(update=step,loss=total,gradient_norm=float(norm)))
        states.append(rng_state(device))
        parameters.append(torch.cat([p.detach().cpu().flatten() for p in model.parameters()]))
        model.eval()
        with torch.no_grad(),autocast(device):
            predictions.append(torch.cat([model(to_device(dataset[i],device)).cpu() for i in order[4000:4008]]))
        passes.append(rows)
        del model,optimizer,saved
        gc.collect()
        torch.cuda.empty_cache()
    delta=(parameters[0]-parameters[1]).abs()
    score_delta=(predictions[0]-predictions[1]).abs()
    write_json(output/"numeric.json",dict(updates_per_probe=8,scan_visits_per_probe=64,probes=passes,
        same_final_torch_rng=torch.equal(states[0]["torch"],states[1]["torch"]),
        same_final_cuda_rng=torch.equal(states[0]["cuda"],states[1]["cuda"]),
        parameter_max_abs=float(delta.max()),parameter_rms=float(delta.square().mean().sqrt()),
        parameter_unequal=int((delta>0).sum()),prediction_max_abs=float(score_delta.max()),
        prediction_mean_abs=float(score_delta.mean()),prediction_unequal=int((score_delta>0).sum()),
        interpretation="Eight-update numerical sensitivity only; does not estimate full-validation AP variance"))
    print(json.loads((output/"numeric.json").read_text()),flush=True)


def observation(xyzi, targets, scores, slots, chosen, **identity_fields):
    """Describe measured returns, not the unseen physical shape or material."""
    from scipy.spatial import cKDTree
    from .render import observation_features
    cloud = xyzi[chosen]
    if len(cloud) > 1:
        features = observation_features(cloud)
    else:
        features = dict(count=1, range_m=float(np.linalg.norm(cloud[0,:3])),
                        spread_m=[0.,0.,0.], linearity=0., planarity=0., intensity=[float(cloud[0,3])]*3)
    center = np.median(cloud[:,:3], axis=0)
    near = np.linalg.norm(xyzi[:,:3]-center,axis=1) <= 2.
    near[chosen] = False
    normal = xyzi[near & (targets==0),:3]
    other = xyzi[near,:3]
    features.update(z_span=float(np.ptp(cloud[:,2])), surrounding_normal=len(normal),
                    relative_z=None if not len(normal) else float(center[2]-np.quantile(normal[:,2],.1)),
                    nearest_other_m=None if not len(other) else float(cKDTree(other).query(cloud[:,:3])[0].min()))
    s = scores[chosen]
    label = int(targets[chosen[0]])
    return dict(**identity_fields, label=label, slots=slots[chosen].astype(int).tolist(),
                center=center.tolist(), features=features,
                score_quantiles=np.quantile(s,[.1,.5,.9]).tolist(),
                bce=float(np.logaddexp(0., s if label==0 else -s).mean()))


def normal_cells(xyzi, targets, scores, *, low=None, high=None):
    """Fixed 0.75 m cells are inspection units, never training crops or labels."""
    eligible = np.flatnonzero(targets==0)
    if low is not None:
        eligible = eligible[(scores[eligible]>=low)&(scores[eligible]<high)]
    elif len(eligible)>512:
        eligible = eligible[np.argpartition(scores[eligible],-512)[-512:]]
    if not len(eligible):
        return []
    cells, counts = np.unique(np.floor(xyzi[eligible,:3]/.75).astype(np.int32),axis=0,return_counts=True)
    grid = np.floor(xyzi[:,:3]/.75).astype(np.int32)
    return [np.flatnonzero((targets==0)&np.all(grid==cell,axis=1))
            for cell in cells[np.argsort(-counts,kind="stable")[:2]]]


def validation_arrays(row,record):
    scored = np.load(OUTPUT / "lr_val.npy",mmap_mode="r")[row["start"]:row["stop"]]
    evaluated = np.load(OUTPUT / "val_points.npy",mmap_mode="r")[row["start"]:row["stop"]]
    raw = np.fromfile(record["scan"],dtype="<f4").reshape(-1,4)
    slots = np.flatnonzero(np.any(raw[:,:3]!=0,axis=1))
    xyzi = raw[slots]
    positions = np.searchsorted(slots,evaluated["slot"])
    target,scores = np.full(len(slots),-1,np.int8),np.full(len(slots),np.nan,np.float32)
    target[positions],scores[positions] = evaluated["target"],scored
    meta = {"slot":slots,"instance":np.zeros(len(slots),np.uint16),"semantic":np.zeros(len(slots),np.uint16)}
    meta["instance"][positions],meta["semantic"][positions] = evaluated["instance"],evaluated["semantic"]
    return xyzi,target,scores,meta


def _validation_cases(task):
    row, record, thresholds = task
    xyzi,target,scores,meta = validation_arrays(row,record)
    result = []
    common = dict(index=row["index"], sequence=row["sequence"], frame=row["frame"], group="STU_validation")
    for instance in np.unique(meta["instance"][target==1]):
        selected = np.flatnonzero((target==1)&(meta["instance"]==instance))
        item = observation(xyzi,target,scores,meta["slot"],selected,**common,instance=int(instance),
                           unit=f"{row['sequence']}:{instance}",kind="anomaly")
        item["missed"] = {str(q):int((scores[selected]<thresholds[str(q)]).sum()) for q in (50,75,90)}
        result.append(item)
    for chosen in normal_cells(xyzi,target,scores,low=thresholds["90"],high=thresholds["50"]):
        item = observation(xyzi,target,scores,meta["slot"],chosen,**common,instance=0,
                           unit=str(row["sequence"]),kind="normal")
        item["semantic_counts"] = {str(v):int((meta["semantic"][chosen]==v).sum()) for v in np.unique(meta["semantic"][chosen])}
        item["band_points"] = int(((scores[chosen]>=thresholds["90"])&(scores[chosen]<thresholds["50"])).sum())
        item["false_positive"] = {str(q):int((scores[chosen]>=thresholds[str(q)]).sum()) for q in (50,75,90)}
        result.append(item)
    return result


def cases(output, workers):
    """Locate C's remaining errors using its already frozen official predictions."""
    output.mkdir(parents=True,exist_ok=True)
    saved = json.loads((OUTPUT/"lr.json").read_text())
    if saved["checkpoint_sha256"] != file_sha256(ROOT/"branches/lr/0/conditional/last.pt"):
        raise ValueError("C prediction identity changed")
    curves = json.loads((OUTPUT/"curves.json").read_text())["models"]["lr"]["val"]
    thresholds = {str(int(100*r["target"])):r["threshold"] for r in curves["operating"]}
    manifest = load_manifest("assets/val.json","val")
    offsets = json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]
    tasks = [(r,manifest["records"][r["index"]],thresholds) for r in offsets]
    observations = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i,result in enumerate(executor.map(_validation_cases,tasks,chunksize=8),1):
            observations.extend(result)
            if i%400==0:
                print(f"C validation observations {i}/{len(tasks)}",flush=True)
    queries = []
    for sequence in (125,169,143):
        rows = [r for r in observations if r["kind"]=="anomaly" and r["sequence"]==sequence]
        # Prioritize missed points, but retain successful observations of the same annotation.
        hard = sorted(rows,key=lambda r:(-r["missed"]["75"],r["frame"]))[:3]
        for row in hard:
            queries.append(dict(row,role="miss"))
        units = {r["unit"] for r in hard}
        good = sorted([r for r in rows if r["unit"] in units],
                      key=lambda r:(r["missed"]["75"]/r["features"]["count"],-r["features"]["count"]))[:2]
        for row in good:
            queries.append(dict(row,role="success_reference"))
    normal = sorted([r for r in observations if r["kind"]=="normal"],key=lambda r:-r["band_points"])
    sequences = set()
    for row in normal:
        if row["sequence"] not in sequences:
            queries.append(dict(row,role="false_positive"))
            sequences.add(row["sequence"])
        if len(sequences)==6:
            break
    from .render import candidate_distance
    offsets_by_index = {r["index"]:r for r in offsets}
    for query in list(queries):
        if query["kind"]!="normal":continue
        xyzi,target,scores,meta = validation_arrays(offsets_by_index[query["index"]],manifest["records"][query["index"]])
        radius = np.linalg.norm(xyzi[:,:3],axis=1)
        eligible = np.flatnonzero((target==0)&(scores<thresholds["90"])&
            (np.abs(radius-query["features"]["range_m"])<2.))
        grid = np.floor(xyzi[:,:3]/.75).astype(np.int32)
        candidates=[]
        for j in np.linspace(0,len(eligible)-1,min(24,len(eligible))).round().astype(int):
            chosen=np.flatnonzero((target==0)&np.all(grid==grid[eligible[j]],axis=1))
            if len(chosen)<5 or np.any(scores[chosen]>=thresholds["90"]):continue
            candidates.append(observation(xyzi,target,scores,meta["slot"],chosen,
                **{k:query[k] for k in ("index","sequence","frame","group","instance","unit","kind")},
                role="normal_success_reference"))
        if candidates:
            queries.append(min(candidates,key=lambda r:candidate_distance(query["features"],r["features"])))
    groups=list(csv.DictReader((OUTPUT/"groups.csv").open(encoding="utf-8-sig")))
    positives=curves["operating"][0]["tp"]+curves["operating"][0]["fn"]
    deficit=[dict(kind=r["kind"],key=r["key"],anomaly=int(r["anomaly"]),
                  AP_deficit=100*int(r["anomaly"])/positives-float(r["pooled_AP_credit"]))
             for r in groups if r["model"]=="lr" and r["kind"] in ("sequence","object","unassigned_anomaly")]
    # Unassigned instance zero still participates in official point evaluation.
    for partition in (("sequence",),("object","unassigned_anomaly")):
        if abs(sum(r["AP_deficit"] for r in deficit if r["kind"] in partition)-(100-curves["AP"]))>1e-7:
            raise ValueError("anomaly AP deficit decomposition is incomplete")
    pr=list(csv.DictReader((OUTPUT/"lr_pr.csv").open(encoding="utf-8-sig")))
    values=np.asarray([float(r["threshold"]) for r in pr])[::-1]
    tp=np.asarray([int(r["tp"]) for r in pr]);fp=np.asarray([int(r["fp"]) for r in pr])
    # Each normal point contributes to every positive ranked below it (ties retained).
    penalty=np.cumsum(np.diff(np.r_[0,tp])[::-1]/(tp+fp)[::-1])*100/positives
    scores=np.load(OUTPUT/"lr_val.npy",mmap_mode="r")
    points=np.load(OUTPUT/"val_points.npy",mmap_mode="r")
    semantic=np.zeros(65536);distance=np.zeros(5)
    for start in range(0,len(scores),1000000):
        part=points[start:start+1000000];normal=part["target"]==0
        weight=penalty[np.searchsorted(values,scores[start:start+1000000][normal])]
        semantic+=np.bincount(part["semantic"][normal],weights=weight,minlength=65536)
        distance+=np.bincount(np.digitize(part["range"][normal],[5,10,20,30]),weights=weight,minlength=5)
    if abs(distance.sum()-(100-curves["AP"]))>1e-7:
        raise ValueError("normal AP deficit decomposition does not sum to 100-AP")
    write_json(output/"cases.json",dict(checkpoint=str(BEST_C),checkpoint_sha256=file_sha256(BEST_C),
        metrics=curves,thresholds=thresholds,queries=queries,observations=observations,
        AP_deficit=sorted(deficit,key=lambda r:-r["AP_deficit"]),
        normal_AP_deficit=dict(semantic={str(i):float(semantic[i]) for i in np.flatnonzero(semantic)},
                              distance=distance.tolist(),
                              definition="100/P sum over positives of FP_group_at_positive_score/(TP+FP); descriptive additive partition of 100-AP, not causal or attainable improvement"),
        limitations="Point/annotation observations; identities are not certified tracks. Descriptor proximity does not certify geometry/material coverage."))
    write_csv(output/"cases.csv",[dict(role=r["role"],sequence=r["sequence"],frame=r["frame"],instance=r["instance"],
        val_index=r["index"],label=r["label"],points=r["features"]["count"],range_m=r["features"]["range_m"],
        score_p10=r["score_quantiles"][0],score_median=r["score_quantiles"][1],score_p90=r["score_quantiles"][2],
        **{f"errors{q}":r.get("missed",r.get("false_positive",{})).get(str(q),0) for q in (50,75,90)},
        intensity_p10=r["features"]["intensity"][0],intensity_median=r["features"]["intensity"][1],
        intensity_p90=r["features"]["intensity"][2],surrounding_normal=r["features"]["surrounding_normal"],
        nearest_other_m=r["features"]["nearest_other_m"]) for r in queries])
    print(f"C cases: {len(observations)} observations, {len(queries)} inspection queries",flush=True)


class MiningScans(PreparedScans):
    def __getitem__(self,index):
        sample = super().__getitem__(index)
        row = self.records[index]
        ids = np.zeros(sample["slot_count"],np.int32)
        if row["anomaly"]:
            with np.load(row["delta"],allow_pickle=False) as delta:
                if row.get("source")=="nuscenes":
                    ids[delta["slots"]] = delta["object_ids"]
                else:
                    ids[delta["source_slot"]] = (delta["packed_labels"]>>16).astype(np.int32)
            if np.any(ids[sample["slots"].numpy()][sample["targets"].numpy()==1]==0):
                raise ValueError("missing synthetic object identity")
        sample["object_ids"] = torch.from_numpy(ids[sample["slots"].numpy()])
        return sample


@torch.no_grad()
def mine(output,workers):
    """Score the entire training pool with C; hard-pool selection uses training only."""
    output.mkdir(parents=True,exist_ok=True)
    disk_check(3_000_000_000)
    write_json(output/"resources.json",runtime_snapshot())
    manifest = load_manifest("results/data/native/train.json","train")
    count = sum(r["normal"]+r["anomaly"] for r in manifest["records"])
    scores_file = np.lib.format.open_memmap(output/"train_scores.npy",mode="w+",dtype=np.float32,shape=(count,))
    labels_file = np.lib.format.open_memmap(output/"train_labels.npy",mode="w+",dtype=np.int8,shape=(count,))
    device = torch.device("cuda")
    model,saved = load_model(BEST_C,device)
    del saved
    loader = DataLoader(MiningScans(manifest),batch_size=None,num_workers=workers,pin_memory=True,
                        prefetch_factor=1,generator=torch.Generator().manual_seed(0))
    frames,observations,cursor = [],[],0
    start = time.perf_counter()
    original_order = json.loads((ROOT/"branches/lr/0/conditional/sampling.json").read_text())["order"]
    visits = np.bincount(original_order[:8000],minlength=len(manifest["records"]))
    for sample in loader:
        index = int(sample["index"])
        row = manifest["records"][index]
        ids = sample.pop("object_ids").numpy()
        with autocast(device):
            prediction = model(to_device(sample,device))
        scores = prediction.cpu().numpy()
        xyzi,targets,slots = [sample[k].numpy() for k in ("xyzi","targets","slots")]
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite C mining scores")
        valid = targets>=0
        stop = cursor+int(valid.sum())
        scores_file[cursor:stop],labels_file[cursor:stop] = scores[valid],targets[valid]
        normal,positive = scores[targets==0],scores[targets==1]
        top = np.partition(normal,max(0,len(normal)-512))[-512:]
        unit = row.get("scene",row.get("world","206"))
        common = dict(index=index,frame=row["frame"],group=row["group"],unit=unit,sequence=206 if "stu" in row["group"] else 0)
        frames.append(dict(**common,start=cursor,stop=stop,normal=len(normal),anomaly=len(positive),
            normal_tail_bce=float(np.logaddexp(0.,top).mean()),
            anomaly_bce=None if not len(positive) else float(np.logaddexp(0.,-positive).mean()),
            normal_quantiles=np.quantile(normal,[.1,.5,.9,.99]).tolist(),
            anomaly_quantiles=None if not len(positive) else np.quantile(positive,[.1,.5,.9]).tolist(),
            visits_before_C=int(visits[index])))
        for instance in np.unique(ids[targets==1]):
            chosen = np.flatnonzero((targets==1)&(ids==instance))
            observations.append(observation(xyzi,targets,scores,slots,chosen,**common,instance=int(instance),kind="anomaly"))
        for chosen in normal_cells(xyzi,targets,scores):
            observations.append(observation(xyzi,targets,scores,slots,chosen,**common,instance=0,kind="normal"))
        cursor = stop
        if (index+1)%200==0 or index+1==len(manifest["records"]):
            print(f"C training mining {index+1}/{len(manifest['records'])}, {time.perf_counter()-start:.1f}s",flush=True)
    assert cursor==count
    scores_file.flush()
    labels_file.flush()
    del model,loader,prediction
    torch.cuda.empty_cache()
    # Rank within each source/type to prevent a sensor-domain quota change.
    from scipy.stats import rankdata
    pools = {}
    for group in sorted({r["group"] for r in frames}):
        rows = [r for r in frames if r["group"]==group]
        difficulty = rankdata([r["normal_tail_bce"] for r in rows],method="average")/len(rows)
        if rows[0]["anomaly"]:
            difficulty = (difficulty+rankdata([r["anomaly_bce"] for r in rows],method="average")/len(rows))/2
        for row,value in zip(rows,difficulty):
            row["difficulty"] = float(value)
        pools[group] = [r["index"] for r in sorted(rows,key=lambda r:(-r["difficulty"],r["index"]))[:int(np.ceil(.2*len(rows)))]]
    curve = score_curve(scores_file,labels_file)
    report = dict(checkpoint=str(BEST_C),checkpoint_sha256=file_sha256(BEST_C),
        train_manifest=manifest["sha256"],frames=frames,observations=observations,
        pooled_training=curve_summary(curve),seconds=time.perf_counter()-start,
        selection="Top 20% in each source/type by percentile of mean top-512 normal BCE; anomaly sources average that percentile with all-positive BCE percentile. No validation scores, labels or queries enter selection.",
        groups=pools)
    write_json(output/"train.json",report)
    write_json(output/"pool.json",{k:report[k] for k in ("checkpoint","checkpoint_sha256","train_manifest","selection","groups")})
    print({"mining_seconds":report["seconds"],"training_AP":curve["AP"],"pool_sizes":{k:len(v) for k,v in pools.items()}},flush=True)


def match_cases(output):
    """Retrieve measured analogues; keep absence and representation claims provisional."""
    from scipy.spatial import cKDTree
    data = json.loads((output/"cases.json").read_text())
    train = json.loads((output/"train.json").read_text())
    fields = ["log_count","log_range","log_spread0","log_spread1","log_spread2",
              "linearity","planarity","intensity10","intensity50","intensity90",
              "log_z_span","log_surrounding_normal","relative_z"]
    def vector(row):
        f = row["features"]
        return np.r_[np.log1p([f["count"],f["range_m"],*f["spread_m"]]),f["linearity"],f["planarity"],
                     f["intensity"],np.log1p([f["z_span"],f["surrounding_normal"]]),f["relative_z"] or 0.]
    matches=[]
    for kind in ("anomaly","normal"):
        rows = [r for r in train["observations"] if r["kind"]==kind]
        if kind=="normal":
            rows += json.loads((output/"coverage.json").read_text())["observations"] if (output/"coverage.json").exists() else []
            rows=list({(r["index"],tuple(r["slots"])):r for r in rows}.values())
        # Every STU normal patch still shares background 206, including synthetic worlds.
        units = ["206" if kind=="normal" and "stu" in r["group"] else r["unit"] for r in rows]
        unit_ids=np.unique(units,return_inverse=True)[1]
        values = np.stack([vector(r) for r in rows])
        center = np.median(values,axis=0)
        scale = np.maximum(np.quantile(values,.75,axis=0)-np.quantile(values,.25,axis=0),.05)
        transformed = (values-center)/scale
        tree = cKDTree(transformed)
        # Calibrate distances against other training worlds/scenes, never validation.
        reference=np.full(len(rows),np.nan)
        k=8
        while np.isnan(reference).any():
            missing=np.flatnonzero(np.isnan(reference))
            distances,neighbors=tree.query(transformed[missing],k=min(k,len(rows)),workers=4)
            different=unit_ids[neighbors]!=unit_ids[missing,None]
            found=different.any(axis=1)
            reference[missing[found]]=distances[found,different[found].argmax(axis=1)]
            if k>=len(rows):break
            k*=4
        reference=np.sort(reference[np.isfinite(reference)])
        for query in [r for r in data["queries"] if r["kind"]==kind]:
            delta = np.linalg.norm(transformed-(vector(query)-center)/scale,axis=1)
            selected=[]
            for domain in ("stu","nuscenes"):
                seen=set()
                for i in np.argsort(delta,kind="stable"):
                    if domain not in rows[i]["group"] or units[i] in seen:continue
                    selected.append(dict(rows[i],distance=float(delta[i]),
                        training_neighbor_percentile=float(np.searchsorted(reference,delta[i],side="right")/len(reference)) if len(reference) else None,
                        visits_before_C=train["frames"][rows[i]["index"]]["visits_before_C"]))
                    seen.add(units[i])
                    if len(seen)==3:break
            matches.append(dict(query=query,neighbors=selected))
    write_json(output/"matches.json",dict(matches=matches,features=fields,
        distance="Euclidean distance after training-only median/IQR scaling, scale floor 0.05; heuristic measured-observation retrieval, not proof of full shape, reflectance or semantic equivalence.",
        limitations="Normal bank: two high-score cells per training scan plus observed cells at query sensor locations in original normal scans; not all normal structures. All STU normal worlds count as one background. Cross-sensor intensity is in native normalized units, not radiometric calibration."))
    print(f"Retrieved training analogues for {len(matches)} C cases",flush=True)
    plot_cases(output)


def _coverage_init(manifest,frames,cells,output):
    global _coverage_scans,_coverage_frames,_coverage_cells,_coverage_scores
    _coverage_scans=Scans(manifest)
    _coverage_frames=frames
    _coverage_cells=cells
    _coverage_scores=np.load(Path(output)/"train_scores.npy",mmap_mode="r")


def _coverage_frame(index):
    sample=_coverage_scans[index]
    row=_coverage_scans.records[index]
    frame=_coverage_frames[index]
    xyzi,target,slots=[sample[k] for k in ("xyzi","targets","slots")]
    score=np.full(len(target),np.nan,np.float32)
    score[target>=0]=_coverage_scores[frame["start"]:frame["stop"]]
    grid=np.floor(xyzi[:,:3]/.75).astype(np.int32)
    inside=(np.linalg.norm(xyzi[:,:3],axis=1)>=2.5)&(np.linalg.norm(xyzi[:,:3],axis=1)<=50)
    counts,observations=[],[]
    for cell in _coverage_cells:
        mask=np.all(grid==cell,axis=1)&inside
        normal=np.flatnonzero(mask&(target==0))
        counts.append(dict(index=index,group=row["group"],cell=cell,actual=int(mask.sum()),normal=len(normal),
                           ignored=int((mask&(target<0)).sum())))
        if len(normal)<5:continue
        observations.append(observation(xyzi,target,score,slots,normal,index=index,frame=row["frame"],
            group=row["group"],unit=row.get("scene","206"),sequence=206 if "stu" in row["group"] else 0,
            instance=0,kind="normal"))
    return counts,observations


def normal_coverage(output,workers):
    """Check near-sensor support without assuming the same cell is the same object."""
    cases=json.loads((output/"cases.json").read_text())
    manifest=load_manifest("results/data/native/train.json","train")
    frames=json.loads((output/"train.json").read_text())["frames"]
    cells=sorted({tuple(np.floor(np.asarray(r["center"])/.75).astype(int).tolist())
                  for r in cases["queries"] if r["role"]=="false_positive"})
    indices=[i for i,r in enumerate(manifest["records"]) if r["group"] in ("normal_nuscenes","normal_stu")]
    counts,observations=[],[]
    with ProcessPoolExecutor(max_workers=workers,initializer=_coverage_init,
                             initargs=(manifest,frames,cells,str(output))) as executor:
        for c,o in executor.map(_coverage_frame,indices,chunksize=8):
            counts.extend(c);observations.extend(o)
    summary=[]
    for group in ("normal_nuscenes","normal_stu"):
        for cell in cells:
            selected=[r for r in counts if r["group"]==group and tuple(r["cell"])==cell]
            summary.append(dict(group=group,cell=cell,scans=len(selected),
                frames_with_returns=sum(r["actual"]>0 for r in selected),
                frames_with_normal=sum(r["normal"]>0 for r in selected),
                normal=sum(r["normal"] for r in selected),ignored=sum(r["ignored"] for r in selected)))
    write_json(output/"coverage.json",dict(summary=summary,counts=counts,observations=observations,
        meaning="Same native-sensor 0.75 m cells, official 2.5-50 m supervision range, original normal scans only. Spatial support is not identity or semantic equivalence. No training labels or hard-pool membership changed."))
    print(summary,flush=True)


def plot_cases(output):
    """Inspect full-scan predictions in local views; no cropped network inference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,findfont
    from matplotlib.ft2font import FT2Font
    fontManager.addfont("/mnt/c/Windows/Fonts/times.ttf")
    plt.rcParams.update({"font.family":"Times New Roman","pdf.fonttype":42,"font.size":10})
    data=json.loads((output/"cases.json").read_text())
    matches=json.loads((output/"matches.json").read_text())["matches"]
    val=load_manifest("assets/val.json","val")
    offsets={r["index"]:r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]}
    training=json.loads((output/"train.json").read_text())
    train_data=Scans(load_manifest("results/data/native/train.json","train"))
    train_scores=np.load(output/"train_scores.npy",mmap_mode="r")
    rows=[]
    for seq in (125,169,143):
        bad=next(r for r in data["queries"] if r["sequence"]==seq and r["role"]=="miss")
        good=next(r for r in data["queries"] if r["sequence"]==seq and r["role"]=="success_reference")
        matched=next(r for r in matches if r["query"]==bad)
        rows.append((bad,good,min(matched["neighbors"],key=lambda x:x["distance"])))
    for bad in [r for r in data["queries"] if r["role"]=="false_positive"][:2]:
        good=next((r for r in data["queries"] if r["role"]=="normal_success_reference" and r["index"]==bad["index"]),None)
        if good is None:continue
        matched=next(r for r in matches if r["query"]==bad)
        rows.append((bad,good,min(matched["neighbors"],key=lambda x:x["distance"])))
    fig=plt.figure(figsize=(13,3.1*len(rows)))
    for row_number,row in enumerate(rows):
        recall=90 if row[0]["kind"]=="normal" and row[0]["false_positive"]["75"]==0 else 75
        for col,item in enumerate(row):
            ax=fig.add_subplot(len(rows),3,row_number*3+col+1,projection="3d")
            if col<2:
                xyzi,target,scores,meta=validation_arrays(offsets[item["index"]],val["records"][item["index"]])
                slots=meta["slot"]
                title=f"STU {item['sequence']}/{item['frame']:06d}"
            else:
                sample=train_data[item["index"]]
                xyzi,target,slots=[sample[k] for k in ("xyzi","targets","slots")]
                scores=np.full(len(target),np.nan,np.float32)
                frame=training["frames"][item["index"]]
                scores[target>=0]=train_scores[frame["start"]:frame["stop"]]
                domain="STU" if "stu" in item["group"] else "nuScenes"
                title=f"Train {item['kind']}: {domain} #{item['index']}"
            chosen=np.searchsorted(slots,np.asarray(item["slots"]))
            assert np.array_equal(slots[chosen],item["slots"])
            xyz=xyzi[:,:3]-np.asarray(item["center"])
            near=np.flatnonzero(np.linalg.norm(xyz,axis=1)<=1.8)
            background=near[np.linspace(0,len(near)-1,min(2500,len(near))).round().astype(int)]
            ax.scatter(*xyz[background].T,s=.5,c="0.75",alpha=.4,rasterized=True)
            wrong=(scores[chosen]>=data["thresholds"][str(recall)])!=(target[chosen]==1)
            ax.scatter(*xyz[chosen].T,s=5,c=np.where(wrong,"#d62728","#1f77b4"),rasterized=True)
            ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),zlim=(-1.,1.2),xlabel="x (m)",ylabel="y (m)",zlabel="z (m)")
            ax.view_init(elev=25,azim=-55)
            ax.set_title(f"{title}\nR{recall}, n={len(chosen)}, median score={np.median(scores[chosen]):.2f}")
    fig.suptitle("C: selected error | successful reference | retrieved training observation\nRed: error at the row's C recall threshold; blue: correct. Training panels are diagnostic only.",y=.995)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.canvas.draw()
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_text() and FT2Font(findfont(item.get_fontproperties(),fallback_to_default=False)).family_name!="Times New Roman":
            raise ValueError("unexpected case-figure font")
    fig.savefig(output/"cases.pdf")
    fig.savefig(output/"cases.png",dpi=180)
    plt.close(fig)


def mining_result(output):
    """Compare the two authorized endpoints and the same raw-slot case identities."""
    report = dict(parent=str(BEST_C),parent_sha256=file_sha256(BEST_C),
                  updates_per_branch=500,scan_visits_per_branch=4000,branches={})
    orders,configs = {},{}
    for name in ("control","hard"):
        directory=ROOT/"mining"/name/"0/conditional"
        result=json.loads((directory/"result.json").read_text())
        if not result["complete"] or result["successful_updates"]!=1500:
            raise ValueError("mining branch is not complete")
        configs[name]=json.loads((directory/"config.json").read_text())["configuration"]
        if configs[name]["initial_sha256"]!=report["parent_sha256"]:
            raise ValueError("C parent checkpoint changed")
        orders[name]=json.loads((directory/"sampling.json").read_text())["order"]
        report["branches"][name]=result
    a,b=np.asarray(orders["control"]),np.asarray(orders["hard"])
    changed=a!=b
    if changed[:8000].any() or changed[12000:].any() or not (changed[8000:12000].reshape(500,8).sum(1)==2).all():
        raise ValueError("paired replacement is not exactly two positions per update")
    records=load_manifest("results/data/native/train.json","train")["records"]
    legacy=load_manifest("assets/train.json","train")
    selected={(r["world"],r["frame"],r["delta_sha256"]) for r in records if r["group"]=="anomaly_stu"}
    available={(r["world"],r["frame"],r["delta_sha256"]) for r in legacy["records"]}
    if not selected<=available:
        raise ValueError("selected STU observations do not belong to the existing source pool")
    unused=[r for r in legacy["records"] if (r["world"],r["frame"],r["delta_sha256"]) not in selected]
    report["existing_STU_observations"]=dict(manifest=legacy["sha256"],available=len(available),
        selected=len(selected),unused=len(unused),unused_with_at_least_100_anomaly_points=sum(r["anomaly"]>=100 for r in unused),
        scope="Manifest inventory only. Unselected observations were not scored or used by these branches; all share background 206.")
    if any(records[x]["group"]!=records[y]["group"] for x,y in zip(a,b)):
        raise ValueError("sampling changed a source/type position")
    report.update(retained_scan_visits=3000,replaced_scan_visits=1000,
                  config_differences=[k for k in configs["control"] if configs["control"][k]!=configs["hard"].get(k)])
    report["pairs"]={name:dict(total=2000,no_anomaly=sum(not any(records[j]["anomaly"] for j in order[i:i+2])
                                  for i in range(8000,12000,2))) for name,order in orders.items()}
    pool=json.loads((output/"pool.json").read_text())
    hard_indices={i for group in pool["groups"].values() for i in group}
    for name,order in orders.items():
        pairs=[order[i:i+2] for i in range(8000,12000,2)]
        normal=[(i,any(records[j]["anomaly"] for j in pair)) for pair in pairs
                for i in pair if i in hard_indices and not records[i]["anomaly"]]
        report["pairs"][name].update(hard_visits=sum(i in hard_indices for i in order[8000:12000]),
            hard_normal_visits=len(normal),hard_normal_in_positive_pairs=sum(present for _,present in normal))
    report["retrieval_exposure"]=[dict(query={k:r["query"][k] for k in ("sequence","frame","instance","role")},
        neighbors=[dict(index=n["index"],group=n["group"],in_hard_pool=n["index"] in hard_indices,
            visits={name:order[8000:12000].count(n["index"]) for name,order in orders.items()}) for n in r["neighbors"]])
        for r in json.loads((output/"matches.json").read_text())["matches"]]
    training=json.loads((output/"train.json").read_text())
    score_file=np.load(output/"train_scores.npy",mmap_mode="r")
    label_file=np.load(output/"train_labels.npy",mmap_mode="r")
    report["C_training_positive_scores"]={}
    for group in ("anomaly_nuscenes","anomaly_stu"):
        positive=np.concatenate([score_file[r["start"]:r["stop"]][label_file[r["start"]:r["stop"]]==1]
                                 for r in training["frames"] if r["group"]==group])
        report["C_training_positive_scores"][group]=dict(count=len(positive),
            quantiles=np.quantile(positive,[.01,.1,.5,.9,.99]).tolist())
    curves=json.loads((OUTPUT/"curves.json").read_text())["models"]
    cases=json.loads((output/"cases.json").read_text())
    meta=np.load(OUTPUT/"val_points.npy",mmap_mode="r")
    offsets={r["index"]:r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]}
    score={name:np.load(OUTPUT/f"{name}_val.npy",mmap_mode="r") for name in ("lr","control","hard")}
    effects=[]
    for query in cases["queries"]:
        offset=offsets[query["index"]]
        points=meta[offset["start"]:offset["stop"]]
        chosen=np.searchsorted(points["slot"],query["slots"])
        if not np.array_equal(points["slot"][chosen],query["slots"]):
            raise ValueError("case point correspondence changed")
        item={k:query[k] for k in ("index","sequence","frame","instance","role","kind")}
        item["count"]=len(chosen)
        item["models"]={}
        for name in score:
            s=score[name][offset["start"]:offset["stop"]][chosen]
            item["models"][name]={str(int(r["target"]*100)):int(((s<r["threshold"]) if query["label"] else (s>=r["threshold"])).sum())
                                    for r in curves[name]["val"]["operating"]}
        effects.append(item)
    report["cases"]=effects
    report["interpretation"]="Single-seed C continuations; test one training-only sampling rule. Endpoints and inherited-start best are separate. C-domain numerical sensitivity remains a quality limitation, not a gate."
    write_json(ROOT/"mining/comparison.json",report)
    print({name:r["final_metrics"] for name,r in report["branches"].items()},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "analyze", "gradients", "replay", "cases", "mine", "match", "coverage", "mining-result"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--names", nargs="+", default=list(CHECKPOINTS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.action == "collect":
        collect(args.output, args.workers)
    elif args.action == "analyze":
        analyze(args.output, args.names)
    elif args.action == "gradients":
        gradients(args.output)
    elif args.action == "cases":
        cases(args.output,args.workers)
    elif args.action == "mine":
        mine(args.output,args.workers)
    elif args.action == "match":
        match_cases(args.output)
    elif args.action == "coverage":
        normal_coverage(args.output,args.workers)
    elif args.action == "mining-result":
        mining_result(args.output)
    else:
        replay(args.output)


if __name__ == "__main__":
    main()
