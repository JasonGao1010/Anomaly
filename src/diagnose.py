"""Paired checkpoint errors, exact rank contributions and fixed-batch gradients."""

import argparse
from collections import defaultdict
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


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
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
                if name in ("replay", "ap", "lr"):
                    directory = ROOT / "branches" / name / "0/conditional"
                    result = json.loads((directory / "result.json").read_text())
                    if not result["complete"] or result["successful_updates"] != 1000:
                        raise ValueError("short branch is unfinished")
                    validation = json.loads((directory / "epoch2.json").read_text())["validation"]
                    write_json(output / f"{name}.json", dict(checkpoint=str(directory / "last.pt"),
                        checkpoint_sha256=file_sha256(directory / "last.pt"), update=1000,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "analyze", "gradients", "replay"))
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
    else:
        replay(args.output)


if __name__ == "__main__":
    main()
