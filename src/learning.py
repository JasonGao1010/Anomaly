"""Static, point-identified learning evidence from a completed recorded run.

Training visits use different weights and stochastic states. These observations
cannot isolate train/eval effects or establish a causal data-coverage diagnosis.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from .attribute import ROOT_ERRORS, cell_keys, poses_for, rank_data
from .data import Scans, load_manifest, write_json
from .diagnose import OUTPUT, write_csv
from .train import runtime_snapshot


def continuation_summary(run, output, workers):
    """Compare complete passes and fixed-state development endpoints, without selecting cases."""
    run, output = Path(run), Path(output)
    result = json.loads((run / "result.json").read_text())
    config = json.loads((run / "config.json").read_text())["configuration"]
    if not result["complete"] or not config.get("continuation_schedule"):
        raise ValueError("summary requires the completed uniform continuation")
    train = load_manifest("results/data/native/train.json", "train")
    if train["sha256"] != config["train_manifest"]:
        raise ValueError("continuation training population changed")
    record = json.loads((run / "record.json").read_text())
    order = json.loads((run / "sampling.json").read_text())["executed_order"]
    visits = defaultdict(list)
    for visit, index in enumerate(order):
        visits[index].append(visit)
    if len(visits) != len(train["records"]) or any(len(v) != 2 for v in visits.values()):
        raise ValueError("continuation did not contain two complete passes")
    scores, points = [np.load(run / f"{name}.npy", mmap_mode="r") for name in ("train", "points")]
    def scan(index):
        row = train["records"][index]
        lo, hi = record["input_offsets"][index:index+2]
        labels = points["target"][lo:hi]
        observed = []
        for number, visit in enumerate(visits[index]):
            values = scores[slice(*record["visit_offsets"][visit:visit+2])]
            if len(values) != len(labels) or not np.isfinite(values).all():
                raise ValueError("missing or nonfinite completed training scores")
            for label, key in ((0, "normal"), (1, "anomaly")):
                selected = values[labels == label].astype(np.float64)
                if len(selected) != row[key]:
                    raise ValueError("recorded training supervision differs from manifest")
                if len(selected):
                    observed.append((row["group"], number, label, len(selected),
                        float(np.logaddexp(0., selected * (1-2*label)).sum()),
                        int(((selected >= 0) if label == 0 else (selected < 0)).sum())))
        return observed
    totals = defaultdict(lambda: [0, 0., 0])
    started = time.perf_counter()
    # NumPy releases the GIL; shared read-only maps avoid per-process copies.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for observed in pool.map(scan, range(len(train["records"]))):
            for group, number, label, count, loss, wrong in observed:
                total = totals[group, number, label]
                total[0] += count
                total[1] += loss
                total[2] += wrong
    passes = [dict(group=group, data_pass=config["data_passes"][number], label=label,
        points=count, bce=loss/count, zero_threshold_errors=wrong)
        for (group, number, label), (count, loss, wrong) in sorted(totals.items())]
    dev = {name:json.loads((output / "development" / f"{name}.json").read_text()) for name in ("start", "end")}
    changes = []
    for split in ("train", "check"):
        for group, before in dev["start"]["splits"][split]["groups"].items():
            # Source, reserved geometry and independent normal scene summaries suffice here.
            if group.startswith("geometry:") and split == "train":
                continue
            after = dev["end"]["splits"][split]["groups"][group]
            changes.append(dict(split=split, group=group, scans=before["scans"],
                normal=before["normal"], anomaly=before["anomaly"],
                metrics={key:dict(before=before.get(key), after=after.get(key),
                    difference=after[key]-before[key] if before.get(key) is not None else None)
                    for key in ("AP", "FPR95", "AUROC", "normal_bce", "anomaly_bce", "FPR_at_geometry_recall95")
                    if key in before}))
    normalization = []
    for name, endpoint in dev.items():
        for group in sorted({r["group"] for r in endpoint["normalization"]["scans"]}):
            rows = [r for r in endpoint["normalization"]["scans"] if r["group"] == group]
            item = dict(endpoint=name, group=group, scans=len(rows),
                mean_abs_logit_difference=float(np.mean([r["mean_abs_change"] for r in rows])),
                normal_loss_improved_scans=sum(r["current_bce_0"] < r["running_bce_0"] for r in rows))
            for label, key in ((0, "normal"), (1, "anomaly")):
                count = sum(train["records"][r["index"]][key] for r in rows)
                if count:
                    for state in ("running", "current"):
                        item[f"{state}_{key}_bce"] = sum((r[f"{state}_bce_{label}"] or 0.) *
                            train["records"][r["index"]][key] for r in rows) / count
            normalization.append(item)
    parent = json.loads((Path(config["initial"]).parent / "result.json").read_text())
    validation = {key:dict(before=parent["final_metrics"][key], after=result["final_metrics"][key],
        difference=result["final_metrics"][key]-parent["final_metrics"][key]) for key in ("AP", "FPR95", "AUROC")}
    write_json(output / "comparison.json", dict(training=result, validation=validation,
        training_passes=passes, development=changes, normalization=normalization,
        training_summary_seconds=time.perf_counter()-started,
        scope="Training-pass logits use changing weights and stochastic states. Development comparisons use fixed checkpoints. BN-only comparisons share weights/inputs and disable stochastic layers. Dataset AP levels are not matched for prevalence or observation difficulty. No independent test or causal coverage claim."))


def point_order(scores, targets):
    """Return opposite-class error and within-scan AP, retaining complete ties."""
    result = np.full(len(scores), np.nan)
    ordered = [np.sort(scores[targets == label]) for label in (0, 1)]
    for label in (0, 1):
        at = np.flatnonzero(targets == label)
        other = ordered[1-label]
        if len(other):
            below = .5*(np.searchsorted(other, scores[at], side="left") +
                        np.searchsorted(other, scores[at], side="right"))
            result[at] = (len(other)-below if label else below)/len(other)
    positive, negative = ordered[1], ordered[0]
    if not len(positive):
        return result, None
    tp = len(positive)-np.searchsorted(positive, positive, side="left")
    fp = len(negative)-np.searchsorted(negative, positive, side="left")
    return result, float(100*np.mean(tp/(tp+fp)))


def two_visits(scores, targets, slots):
    """Exact pointwise score/loss/rank changes; no arbitrary 'learned' threshold."""
    bce = [np.logaddexp(0., s.astype(float)*np.where(targets == 1, -1., 1.)) for s in scores]
    ranked = [point_order(s, targets) for s in scores]
    wrong = [(s < 0) & (targets == 1) | (s >= 0) & (targets == 0) for s in scores]
    rows = []
    for label in (0, 1):
        at = np.flatnonzero(targets == label)
        if not len(at):
            continue
        row = dict(label=label, points=len(at),
                   wrong_both=int((wrong[0][at] & wrong[1][at]).sum()),
                   corrected=int((wrong[0][at] & ~wrong[1][at]).sum()),
                   newly_wrong=int((~wrong[0][at] & wrong[1][at]).sum()),
                   correct_both=int((~wrong[0][at] & ~wrong[1][at]).sum()),
                   bce_improved=int((bce[1][at] < bce[0][at]).sum()),
                   score_change_mean=float((scores[1][at]-scores[0][at]).mean(dtype=float)))
        for v in range(2):
            errors = ranked[v][0][at]
            row.update({f"bce{v+1}":float(bce[v][at].mean()),
                        f"score{v+1}":float(scores[v][at].mean(dtype=float)),
                        f"wrong{v+1}":int(wrong[v][at].sum()),
                        f"rank_error{v+1}":float(errors.mean()) if np.isfinite(errors).all() else None,
                        f"AP{v+1}":ranked[v][1]})
        first,second=ranked[0][0][at],ranked[1][0][at]
        comparable=np.isfinite(first).all() and np.isfinite(second).all()
        row.update(rank_inverted_both=int(((first>0)&(second>0)).sum()) if comparable else None,
                   rank_corrected=int(((first>0)&(second==0)).sum()) if comparable else None,
                   rank_newly_inverted=int(((first==0)&(second>0)).sum()) if comparable else None,
                   rank_improved=int((second<first).sum()) if comparable else None,
                   rank_worsened=int((second>first).sum()) if comparable else None)
        persistent = at[wrong[0][at] & wrong[1][at]]
        candidates = persistent if len(persistent) else at
        selected = candidates[np.argsort(-bce[1][candidates], kind="stable")[:3]]
        row["point_examples"] = [dict(slot=int(slots[i]), score1=float(scores[0][i]),score2=float(scores[1][i]),
            bce1=float(bce[0][i]),bce2=float(bce[1][i]),
            rank1=None if not np.isfinite(ranked[0][0][i]) else float(ranked[0][0][i]),
            rank2=None if not np.isfinite(ranked[1][0][i]) else float(ranked[1][0][i])) for i in selected]
        rows.append(row)
    return rows, bce, ranked, wrong


def _train_init(run, profiles):
    global _run, _record, _scores, _points, _visits, _data, _profiles, _worlds
    _run = Path(run)
    _record = json.loads((_run/"record.json").read_text())
    _scores = np.load(_run/"train.npy", mmap_mode="r")
    _points = np.load(_run/"points.npy", mmap_mode="r")
    order = json.loads((_run/"sampling.json").read_text())["executed_order"]
    _visits = defaultdict(list)
    for visit, index in enumerate(order):
        _visits[index].append(visit)
    _data = Scans(load_manifest("results/data/native/train.json", "train"), cache_size=2)
    _worlds={r["id"]:r["paths"] for r in _data.manifest["worlds"]}
    _profiles = profiles


def _train_scan(index):
    row, visits = _data.records[index], _visits[index]
    if len(visits) != 2:
        raise ValueError("the pointwise comparison requires exactly two complete visits")
    lo, hi = _record["input_offsets"][index:index+2]
    points = _points[lo:hi]
    scores = [_scores[slice(*_record["visit_offsets"][v:v+2])] for v in visits]
    if any(len(s)!=len(points) or not np.isfinite(s).all() for s in scores):
        raise ValueError("training visits have missing or nonfinite point scores")
    rows, bce, ranks, wrong = two_visits(scores, points["target"], points["slot"])
    common = dict(index=index, group=row["group"], unit=row.get("scene",row.get("world","206")), frame=row["frame"],
                  geometry_sources=_worlds.get(row.get("world"),row.get("geometry",[])),
                  visit1=visits[0],visit2=visits[1],update1=visits[0]//8+1,update2=visits[1]//8+1)
    rows = [dict(**common,**r) for r in rows]
    # Decode geometry once per scan; it identifies existing profile selectors.
    sample = _data[index]
    if not np.array_equal(sample["slots"],points["slot"]) or not np.array_equal(sample["targets"],points["target"]):
        raise ValueError("material inputs differ from the recorded point identities")
    source_labels=(np.fromfile(row["label"],dtype=np.uint8) if row.get("source")=="nuscenes" else
                   np.fromfile(_data.sources[row["frame"]]["label"],dtype="<u4")&65535)[sample["slots"]]
    for item in rows:
        persistent=(sample["targets"]==item["label"])&wrong[0]&wrong[1]
        distance=np.linalg.norm(sample["xyzi"][persistent,:3],axis=1)
        item["persistent_ranges"]=np.histogram(distance,bins=[2.5,5,10,20,50.00001])[0].tolist()
        classes,counts=np.unique(source_labels[persistent],return_counts=True)
        item["persistent_source_classes"]={str(k):int(v) for k,v in zip(classes,counts)} if item["label"]==0 else {}
    grid = np.floor(sample["xyzi"][:,:3]/.75).astype(int)
    normals, anomalies = points["target"]==0, points["target"]==1
    output = []
    for pi, profile in _profiles.get(index, []):
        selector = profile["selector"]
        if profile["kind"] == "normal":
            chosen = np.flatnonzero(normals & (grid==selector["cell"]).all(1))
        else:
            # Existing synthetic material profiles store the exact raw slots.
            chosen = np.searchsorted(points["slot"], selector["slots"])
            if not np.array_equal(points["slot"][chosen], selector["slots"]) or not anomalies[chosen].all():
                raise ValueError("anomaly material slots changed")
        if len(chosen)!=profile["points"]:
            raise ValueError("material selector does not recover its observed points")
        item = dict(profile=pi,**common,kind=profile["kind"],points=len(chosen),
                    selector={k:v for k,v in selector.items() if k!="slots"},
                    wrong_both=int((wrong[0][chosen]&wrong[1][chosen]).sum()))
        for v in range(2):
            errors = ranks[v][0][chosen]
            worst = chosen[np.argmax(bce[v][chosen])]
            item.update({f"bce{v+1}":float(bce[v][chosen].mean()),f"wrong{v+1}":int(wrong[v][chosen].sum()),
                f"score{v+1}":np.quantile(scores[v][chosen],[.1,.5,.9]).tolist(),
                f"rank{v+1}":float(errors.mean()) if np.isfinite(errors).all() else None,
                f"worst_slot{v+1}":int(points["slot"][worst])})
        output.append(item)
    return rows, output


def _compact_csv(path, rows):
    """JSON cells preserve nested point identities without Python repr parsing."""
    write_csv(path, [{k:json.dumps(v,ensure_ascii=False,separators=(",",":")) if isinstance(v,(list,dict)) else v
                      for k,v in row.items()} for row in rows])


def learning_records(run, output, workers):
    train = load_manifest("results/data/native/train.json", "train")
    recording = json.loads((run/"record.json").read_text())
    old = json.loads((ROOT_ERRORS/"train_profiles.json").read_text())
    if old["source_manifest"] != train["sha256"] or recording["train_manifest"]!=train["sha256"]:
        raise ValueError("material identity and training record manifests differ")
    observations = json.loads((ROOT_ERRORS/"train.json").read_text())["observations"]
    anomaly_slots = {(r["index"],r["instance"]):r["slots"] for r in observations if r["kind"]=="anomaly"}
    queries = defaultdict(list)
    for i,r in enumerate(old["records"]):
        selected = dict(r,selector=dict(r["selector"]))
        if r["kind"]=="anomaly":
            selected["selector"]["slots"] = anomaly_slots[r["index"],r["selector"]["instance"]]
        queries[r["index"]].append((i,selected))
    scans, profiles = [], []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=_train_init,initargs=(str(run),queries)) as executor:
        for i,(rows,items) in enumerate(executor.map(_train_scan,range(len(train["records"])),chunksize=8),1):
            scans.extend(rows)
            profiles.extend(items)
            if i%800==0:
                print(f"Learning records {i}/{len(train['records'])}, {time.perf_counter()-start:.1f}s",flush=True)
    profiles.sort(key=lambda r:r["profile"])
    if [r["profile"] for r in profiles]!=list(range(len(old["records"]))):
        raise ValueError("material observations are missing or duplicated")
    _compact_csv(output/"visits.csv", scans)
    _compact_csv(output/"materials.csv", profiles)
    groups = []
    for group in sorted({r["group"] for r in scans}):
        for label in (0,1):
            selected = [r for r in scans if r["group"]==group and r["label"]==label]
            if not selected:
                continue
            n = sum(r["points"] for r in selected)
            item = dict(group=group,label=label,scans=len(selected),points=n)
            for k in ("wrong_both","corrected","newly_wrong","correct_both","bce_improved","wrong1","wrong2"):
                item[k] = sum(r[k] for r in selected)
            for k in ("bce1","bce2","score1","score2","score_change_mean"):
                item[k] = sum(r[k]*r["points"] for r in selected)/n
            for k in ("rank_error1","rank_error2","AP1","AP2"):
                valid = [r for r in selected if r[k] is not None]
                item[k] = sum(r[k]*r["points"] for r in valid)/sum(r["points"] for r in valid) if valid else None
            for k in ("rank_inverted_both","rank_corrected","rank_newly_inverted","rank_improved","rank_worsened"):
                valid=[r for r in selected if r[k] is not None]
                item[k]=sum(r[k] for r in valid) if valid else None
            item["persistent_ranges"]=np.sum([r["persistent_ranges"] for r in selected],axis=0).tolist()
            classes=defaultdict(int)
            for r in selected:
                for k,v in r["persistent_source_classes"].items():
                    classes[k]+=v
            item["persistent_source_classes"]=dict(classes)
            groups.append(item)
    hard = sorted(scans,key=lambda r:-r["wrong_both"])
    summary = dict(groups=groups,seconds=time.perf_counter()-start,scans=len(train["records"]),
        supervised_points=sum(r["points"] for r in scans),matched_materials=len(profiles),
        persistent_point_examples=hard[:20],
        persistent_anomaly_examples=sorted([r for r in scans if r["label"]==1],key=lambda r:-r["wrong_both"])[:20],
        range_edges_m=[2.5,5,10,20,50.00001],
        definitions=dict(visits="Actual training-mode logits at two different updates, not frozen-model inference or a train/eval contrast",
            wrong="anomaly logit < 0 or normal logit >= 0; correct classification does not establish correct AP ranking",
            rank="Opposite-class inversion fraction within this full scan, half credit for ties; null if the other class is absent",
            material_selection="Existing measured geometry selectors only; old scores and old visit counts are discarded. These profiles do not exhaust all training normal patches."))
    write_json(output/"learning.json",summary)
    return scans, profiles


def state_records(run, output, scans):
    """Source association is measured at saved update resolution, never causation."""
    train = load_manifest("results/data/native/train.json", "train")
    order = json.loads((run/"sampling.json").read_text())["executed_order"]
    logs = [json.loads(line) for line in (run/"log.jsonl").read_text().splitlines()]
    states = np.load(run/"buffers.npy",mmap_mode="r")
    fields = {r["name"]:r for r in json.loads((run/"record.json").read_text())["buffers"]}
    domain = np.array([train["records"][i]["group"].endswith("stu") for i in order])
    switches = np.r_[False,domain[1:]!=domain[:-1]]
    recent, history = 0., []
    for x in domain:
        recent = .99*recent+.01*float(x)
        history.append(recent)
    changes, layer_rows = [], []
    for name, field in fields.items():
        if not name.endswith("running_mean"):
            continue
        varfield = fields[name.replace("running_mean","running_var")]
        mean = np.asarray(states[:,field["start"]:field["stop"]],float)
        var = np.asarray(states[:,varfield["start"]:varfield["stop"]],float)
        delta = np.sqrt(np.mean(np.diff(mean,axis=0)**2/(var[:-1]+.001),axis=1))
        variance_delta = np.sqrt(np.mean(np.diff(np.log(var+.001),axis=0)**2,axis=1))
        changes.append(delta)
        counts = fields[name.replace("running_mean","num_batches_tracked")]
        observed = states[:,counts["start"]]-states[0,counts["start"]]
        if not np.array_equal(observed,np.minimum(np.arange(len(states))*8,len(order))):
            raise ValueError("BN counters do not match actual recorded training visits")
        # Remove a smooth training-time trend before correlating source composition.
        t = np.linspace(-1,1,len(delta))
        design = np.stack([np.ones(len(t)),t,t*t,t*t*t],axis=1)
        late = np.arange(len(delta))>=len(delta)//2
        source = np.array([history[min((u+1)*8,len(order))-1] for u in range(len(delta))])
        def partial(values):
            a = values-design@np.linalg.lstsq(design,values,rcond=None)[0]
            b = source-design@np.linalg.lstsq(design,source,rcond=None)[0]
            return float(np.corrcoef(a[late],b[late])[0,1])
        channel_association = [partial(mean[1:,j]) for j in range(mean.shape[1])]
        layer_rows.append(dict(layer=name.removesuffix(".running_mean"),
            mean_step_shift=float(delta.mean()),max_step_shift=float(delta.max()),max_shift_update=int(delta.argmax()+1),
            variance_step_log_shift=float(variance_delta.mean()),
            late_source_partial_abs_correlation_median=float(np.median(np.abs(channel_association))),
            late_source_partial_abs_correlation_max=float(np.max(np.abs(channel_association)))))
    delta = np.stack(changes).mean(0)
    training = {(r["index"],r["label"]):r for r in scans}
    updates = []
    pair_summary = defaultdict(lambda:dict(groups=0,ranking=0,empty=0,positive=0,negative=0))
    for u,log in enumerate(logs):
        begin, end = u*8,min((u+1)*8,len(order))
        nloss,ploss,ncount,pcount = 0.,0.,0,0
        for visit in range(begin,end):
            index = order[visit]
            for label in (0,1):
                r = training.get((index,label))
                if r is None:
                    continue
                v = 1 if visit==r["visit1"] else 2
                if label:
                    ploss += r[f"bce{v}"]*r["points"]; pcount += r["points"]
                else:
                    nloss += r[f"bce{v}"]*r["points"]; ncount += r["points"]
        bce = (nloss/ncount+(ploss/pcount if pcount else 0))/(2 if pcount else 1)
        if abs(bce-log["loss_components"]["bce"])>2e-6:
            raise ValueError("recorded point losses do not reproduce the actual update BCE")
        for pair,begin_pair in enumerate(range(begin,end,2)):
            indices = order[begin_pair:min(begin_pair+2,end)]
            composition = "+".join(sorted(train["records"][i]["group"] for i in indices))
            item = pair_summary[composition]
            item["groups"] += 1
            if log["ranking_weight"]>0:
                detail = log["pairs"][pair]
                pos=sum(train["records"][i]["anomaly"] for i in indices)
                neg=sum(train["records"][i]["normal"] for i in indices)
                if detail["positive"]!=pos or detail["negative"]!=neg:
                    raise ValueError("rank-pair identities do not match the actual loss log")
                item["ranking"] += 1; item["empty"] += int(pos==0)
                item["positive"] += pos; item["negative"] += neg
        updates.append(dict(update=u+1,stu_fraction=float(domain[begin:end].mean()),
            source_switches=int(switches[begin:end].sum()),recent_stu=float(history[end-1]),
            bn_shift=float(delta[u]),gradient_norm=log["gradient_norm"],clipped=log["gradient_norm"]>1,
            bce=bce,normal_bce=nloss/ncount,anomaly_bce=ploss/pcount if pcount else None,
            lr=log["lr"][0],ranking_weight=log["ranking_weight"]))
    _compact_csv(output/"states.csv",updates)
    # Condition on current source and training quarter; preceding source is still not randomized evidence.
    transition = defaultdict(list)
    for r in scans:
        for v in (1,2):
            visit = r[f"visit{v}"]
            if visit==0:
                continue
            key=(r["group"],r["label"],min(3,visit*4//len(order)),bool(switches[visit]))
            transition[key].append((r[f"bce{v}"],r["points"]))
    transitions = [dict(group=g,label=label,quarter=q,source_switched=s,scans=len(a),
        point_mean_bce=sum(x*n for x,n in a)/sum(n for _,n in a),scan_mean_bce=float(np.mean([x for x,_ in a])))
        for (g,label,q,s),a in sorted(transition.items())]
    associations = []
    for low,high in ((1,387),(388,774),(775,1161),(1162,len(updates))):
        a=updates[low-1:high]
        for x,y in (("stu_fraction","bn_shift"),("source_switches","bn_shift"),("bn_shift","gradient_norm"),("stu_fraction","bce")):
            associations.append(dict(first=low,last=high,x=x,y=y,spearman=float(spearmanr([r[x] for r in a],[r[y] for r in a]).statistic)))
    result=dict(layers=layer_rows,associations=associations,transitions=transitions,
        pairs=[dict(composition=k,**v) for k,v in sorted(pair_summary.items())],
        updates=len(logs),clipped=sum(r["clipped"] for r in updates),
        top_gradient_updates=sorted(updates,key=lambda r:-r["gradient_norm"])[:12],
        constraints=["BN rows are after an entire update (usually eight scans); no per-scan intermediate buffers were recorded",
            "In train mode BatchNorm uses current scan statistics, not its running means; preceding-source buffer changes alone cannot cause training-forward score drift",
            "Weights, source geometry, sampling context and stochastic depth also vary. Associations do not identify a BN inference mismatch",
            "Only whole-network gradient norms are retained; no layerwise or losswise gradient directions were recorded",
            "No same-input same-weight train/eval comparison is contained in these two visits"])
    write_json(output/"states.json",result)
    return result


def _val_init(run, output, queries):
    global _val_queries, _val_manifest, _val_scores, _val_points, _val_offsets, _val_rank, _val_geometry
    _val_queries = queries
    _val_manifest = load_manifest("assets/val.json","val")
    result = json.loads((Path(run)/"result.json").read_text())
    _val_scores = np.load(Path(run)/f"val{result['successful_updates']}.npy",mmap_mode="r")
    _val_points = np.load(OUTPUT/"val_points.npy",mmap_mode="r")
    _val_offsets = {r["index"]:r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]}
    _val_rank = rank_data(Path(output)/"curve.csv")
    _val_geometry = {}


def _val_scan(index):
    record,offset = _val_manifest["records"][index],_val_offsets[index]
    score = _val_scores[offset["start"]:offset["stop"]]
    meta = _val_points[offset["start"]:offset["stop"]]
    xyzi = np.fromfile(record["scan"],dtype="<f4").reshape(-1,4)[meta["slot"]]
    normal = np.flatnonzero(meta["target"]==0)
    sequence = record["sequence"]
    if sequence not in _val_geometry:
        with np.load(ROOT_ERRORS/f"surface_{sequence}.npz") as saved:
            _val_geometry[sequence] = (saved["keys"],saved["component"],poses_for(Path(record["scan"]).parents[1]))
    keys,components,poses = _val_geometry[sequence]
    pose = poses[record["frame"]]
    world = xyzi[:,:3].astype(float)@pose[:3,:3].T+pose[:3,3]
    component = np.full(len(xyzi),-1)
    component[normal] = components[np.searchsorted(keys,cell_keys(world[normal],meta["semantic"][normal]))]
    grid = np.floor(xyzi[:,:3]/.75).astype(int)
    world_grid = np.floor(world/.75).astype(int)
    values,positive,negative,denominator,ploss,nloss = _val_rank
    rank = np.searchsorted(values,score)
    within,_ = point_order(score,meta["target"])
    global_fp = np.cumsum(negative[::-1])[::-1]
    results=[]
    represented=np.zeros(len(meta),dtype=bool)
    for qi,query in _val_queries[index]:
        selector=query["selector"]
        label = int(query["kind"]=="anomaly")
        if label:
            chosen = np.flatnonzero((meta["target"]==1)&(meta["instance"]==selector["instance"]))
        else:
            chosen = (meta["target"]==0)&(component==selector["component"])
            chosen &= (world_grid==selector["world_cell"]).all(1) if "world_cell" in selector else (grid==selector["cell"]).all(1)
            chosen = np.flatnonzero(chosen)
        if len(chosen)!=query["points"]:
            raise ValueError("validation material selectors do not recover the original point population")
        if represented[chosen].any():
            raise ValueError("local material queries repeat the same validation point")
        represented[chosen]=True
        s = score[chosen]
        loss = (ploss if label else nloss)[rank[chosen]]
        examples = chosen[np.argsort(-loss,kind="stable")[:3]]
        results.append(dict(query=qi,case=query["case"],index=index,sequence=sequence,frame=record["frame"],
            kind=query["kind"],selector=selector,points=len(chosen),AP_loss=float(loss.sum()),
            bce=float(np.logaddexp(0.,s.astype(float)*(1-2*label)).mean()),
            wrong=int(((s<0) if label else (s>=0)).sum()),score=np.quantile(s,[.1,.5,.9]).tolist(),
            within_rank=float(within[chosen].mean()),
            global_pair_error=float(((global_fp[rank[chosen]]-.5*negative[rank[chosen]])/negative.sum()).mean()) if label else None,
            point_examples=[dict(slot=int(meta["slot"][j]),score=float(score[j]),
                AP_loss=float((ploss if label else nloss)[rank[j]]),
                within_rank=float(within[j]),semantic=int(meta["semantic"][j])) for j in examples]))
    return results


def validation_materials(run,output,workers):
    saved = json.loads((ROOT_ERRORS/"val_profiles.json").read_text())
    if saved["source_manifest"] != load_manifest("assets/val.json","val")["sha256"]:
        raise ValueError("validation material identity has changed")
    queries=defaultdict(list)
    for qi,row in enumerate(saved["records"]):
        queries[row["index"]].append((qi,row))
    result=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=_val_init,initargs=(str(run),str(output),queries)) as executor:
        for rows in executor.map(_val_scan,sorted(queries),chunksize=8):
            result.extend(rows)
    result.sort(key=lambda r:r["query"])
    return result


def _physical(vector):
    return dict(extent_m=np.expm1(vector[:3]).tolist(),points=int(round(np.expm1(vector[17]))),
                range_m=np.expm1(vector[18:21]).tolist(),intensity=vector[24:27].tolist(),
                local_residual=vector[14:17].tolist(),normal_proximity_m=np.expm1(vector[31:34]).tolist())


def boundary_points(xyzi, targets, chosen, wrong):
    """Describe mixed input voxels, without treating their presence as a bug."""
    from scipy.spatial import cKDTree
    from .model import voxelize
    inverse=voxelize(np.asarray(xyzi,np.float32))["inverse"].numpy()
    positive=targets==1
    count=np.bincount(inverse[positive],minlength=int(inverse.max())+1)
    mixed=count[inverse[chosen]]>0
    distance=cKDTree(xyzi[positive,:3]).query(xyzi[chosen,:3])[0] if positive.any() else None
    return dict(points=len(chosen),mixed_voxel_points=int(mixed.sum()),
        wrong_outside_mixed_voxels=int((wrong&~mixed).sum()),
        nearest_anomaly_m=np.quantile(distance,[0,.5,.9]).tolist() if distance is not None else None)


def link_evidence(output, materials, queries):
    ledger=json.loads((output/"ledger.json").read_text())
    links=json.loads((ROOT_ERRORS/"material_links.json").read_text())
    if links["validation_manifest"]!=ledger["validation_manifest"]:
        raise ValueError("retrieved materials use different validation identities")
    train_vectors=np.load(ROOT_ERRORS/"train_profiles.npy",mmap_mode="r")
    val_vectors=np.load(ROOT_ERRORS/"val_profiles.npy",mmap_mode="r")
    by_case=defaultdict(list)
    for q in queries:
        relation=links["records"][q["query"]]
        if relation["query"]!=q["query"] or relation["case"]!=q["case"]:
            raise ValueError("material query identities changed")
        same=[n for n in relation["neighbors"] if materials[n["profile"]]["group"].endswith("stu")]
        nearest=min(same,key=lambda n:n["distance"])
        m=materials[nearest["profile"]]
        q.update(material=nearest["profile"],material_index=m["index"],distance=nearest["distance"],
                 blocks=nearest["blocks"],material_bce1=m["bce1"],material_bce2=m["bce2"],
                 material_wrong1=m["wrong1"],material_wrong2=m["wrong2"],material_points=m["points"],
                 material_rank1=m["rank1"],material_rank2=m["rank2"],
                 material_update1=m["update1"],material_update2=m["update2"],
                 candidate_profiles=[n["profile"] for n in relation["neighbors"]])
        by_case[q["case"]].append(q)
    _compact_csv(output/"queries.csv",queries)
    rows, details = [], []
    for kind,collection in (("anomaly",ledger["objects"]),("normal",ledger["surfaces"])):
        for case in collection:
            selected=by_case[case["id"]]
            covered=sum(q["AP_loss"] for q in selected)
            if covered>case["AP_loss"]+1e-8:
                raise ValueError("material profiles duplicate a case's AP contribution")
            failure=max(selected,key=lambda q:q["AP_loss"]) if selected else None
            success=min(selected,key=lambda q:q["AP_loss"]/q["points"]) if selected else None
            if success and (success["query"]==failure["query"] or
                            success["AP_loss"]/success["points"]>=failure["AP_loss"]/failure["points"]):
                success=None
            ref=materials[failure["material"]] if failure else None
            if case["AP_loss"]<=0:
                support,ruled,missing,change="当前排序无失分","无","无","保留成功参照"
            elif not selected or covered<=0:
                support="完整排序已定位失分点；历史素材局部未覆盖本轮错误"
                ruled="不能用旧代表局部解释当前整个结构"
                missing="当前高影响局部与训练素材的实际观测核验"
                change="先核验本轮最高失分点的局部形态、响应与周围结构；暂不改训练"
            elif ref["wrong_both"]:
                support="相关候选存在两次访问均分类错误的点；素材关联仍待核验"
                ruled="排除该候选未进入训练；不能认定预算或模型容量不足"
                missing="近邻是否对应实际失败机制；同权重推理及有效梯度方向"
                change="先核验候选观测；确认关联后检查固定状态的局部计算与监督，暂不重复全量训练"
            elif ref["wrong2"]:
                support="相关候选第二次访问仍有分类错误；不能直接称为没学够"
                ruled="排除该候选未获正常访问；两次权重不同，不能认定模式差异"
                missing="候选相关性；时间、随机层及表示的区分证据"
                change="核对两次困难点位置和观测，必要时对保存状态做小量固定输入前向"
            else:
                support="相关候选第二次训练访问分类正确；验证错误尚不能区分观测泛化与运行状态差异"
                ruled="该候选不能支持普遍未训练或分类完全没学会；不证明素材覆盖等价"
                missing="形态、响应、背景关系的等价核验；当前同权重素材推理"
                change="优先核验泛化与观测差异；不据此增加该候选重复次数或指定点权重"
            rows.append(dict(case=case["id"],kind=kind,AP_loss=case["AP_loss"],gap_share=case["AP_loss"]/ledger["AP_loss"],
                points=case["points"],matched_local_AP=covered,unmatched_local_AP=max(0.,case["AP_loss"]-covered),
                query_count=len(selected),distinct_nearest_materials=len({q["material"] for q in selected}),
                failure_query=failure["query"] if failure else None,success_query=success["query"] if success else None,
                actual_error_points=(dict(index=failure["index"],frame=failure["frame"],points=failure["point_examples"]) if failure else case.get("point_examples",[])),
                material=ref["profile"] if ref else None,material_index=ref["index"] if ref else None,
                material_updates=[ref["update1"],ref["update2"]] if ref else [],
                material_bce=[ref["bce1"],ref["bce2"]] if ref else [],
                material_wrong=[ref["wrong1"],ref["wrong2"]] if ref else [],
                material_rank=[ref["rank1"],ref["rank2"]] if ref else [],
                within_rank=case.get("within_scan_rank_error"),cross_rank=case.get("cross_scan_rank_error"),
                cross_only_AP=case.get("cross_only_AP_loss"),supported=support,excluded=ruled,missing=missing,change=change,
                causal_status="未完成因果区分" if case["AP_loss"]>0 else "无当前失分"))
            # Detailed high-impact examples retain real observations, not only descriptor distances.
            if failure and (kind=="anomaly" or len([d for d in details if d["kind"]=="normal"])<20):
                candidate_rows=[]
                for pi in failure["candidate_profiles"]:
                    candidate_rows.append(dict(**materials[pi],physical=_physical(train_vectors[pi])))
                details.append(dict(case=case["id"],kind=kind,AP_loss=case["AP_loss"],matched_local_AP=covered,
                    failure=dict(**failure,physical=_physical(val_vectors[failure["query"]])),
                    success=dict(**success,physical=_physical(val_vectors[success["query"]])) if success else None,
                    success_scope="Lowest loss per point among existing same-case profiles; not a matched causal control. Null means no better reference is present.",
                    materials=candidate_rows,decision=dict(supported=support,excluded=ruled,missing=missing,change=change)))
    summary=dict(metrics=ledger["metrics"],AP_gap=ledger["AP_loss"],rank_comparison=ledger["rank_comparison"],
        accounting_scope="All official points and all score ties; objects and normal structures are two views of the SAME gap and cannot be added",
        existing_material_scope="Fixed historical geometry selectors, recomputed using current endpoint validation logits and actual current training visits; no old model score is reused",
        query_count=len(queries),material_count=len(materials),
        coverage={kind:dict(cases=sum(r["kind"]==kind for r in rows),
            positive_loss_cases=sum(r["kind"]==kind and r["AP_loss"]>0 for r in rows),
            matched_local_AP=sum(r["matched_local_AP"] for r in rows if r["kind"]==kind),
            unmatched_local_AP=sum(r["unmatched_local_AP"] for r in rows if r["kind"]==kind),
            causally_explained_AP=0.) for kind in ("anomaly","normal")},
        major_cases=details,
        boundaries=["No new training or model forward pass was performed",
            "Material distances are observed descriptors, not learned activations, certified equivalence or a coverage verdict",
            "Two visits use different weights and stochastic states; final validation is another state and another population",
            "Zero-threshold correctness and within-scan AP do not prove generalization or global calibration",
            "Causal explanation of the remaining gap remains unresolved; exact descriptive accounting is complete"])
    return summary,rows


def score_gradients(run,output,materials,evidence):
    """Differentiate saved scores only; never run a model or update a parameter."""
    import torch
    from .model import balanced_loss, ranking_loss
    train=load_manifest("results/data/native/train.json","train")
    order=json.loads((run/"sampling.json").read_text())["executed_order"]
    record=json.loads((run/"record.json").read_text())
    logs=[json.loads(line) for line in (run/"log.jsonl").read_text().splitlines()]
    config=json.loads((run/"config.json").read_text())
    scores=np.load(run/"train.npy",mmap_mode="r")
    points=np.load(run/"points.npy",mmap_mode="r")
    selected={}
    for kind in ("anomaly","normal"):
        for case in [r for r in evidence["major_cases"] if r["kind"]==kind][:5]:
            pi=case["failure"]["material"]
            selected.setdefault(pi,[]).append(case["case"])
    data=Scans(train,cache_size=2)
    old=json.loads((ROOT_ERRORS/"train.json").read_text())
    anomaly_slots={(r["index"],r["instance"]):r["slots"] for r in old["observations"] if r["kind"]=="anomaly"}
    requests=defaultdict(list)
    boundaries=[]
    for pi,cases in selected.items():
        material=materials[pi]; sample=data[material["index"]]
        if material["kind"]=="normal":
            chosen=np.flatnonzero((sample["targets"]==0)&(np.floor(sample["xyzi"][:,:3]/.75)==material["selector"]["cell"]).all(1))
        else:
            chosen=np.searchsorted(sample["slots"],anomaly_slots[material["index"],material["selector"]["instance"]])
        for visit in (material["visit1"],material["visit2"]):
            requests[visit//8+1].append((pi,cases,visit,chosen,sample["slots"][chosen]))
        if material["kind"]=="normal":
            s=scores[slice(*record["visit_offsets"][material["visit2"]:material["visit2"]+2])]
            boundaries.append(dict(profile=pi,index=material["index"],cases=cases,
                **boundary_points(sample["xyzi"],sample["targets"],chosen,s[chosen]>=0)))
    # CUDA is required only to reproduce the original independent sampling RNG.
    device=torch.device("cuda")
    output_rows=[]; residuals=[]
    for update,items in sorted(requests.items()):
        log=logs[update-1]; indices=order[(update-1)*8:update*8]
        counts=torch.tensor([sum(train["records"][i][k] for i in indices) for k in ("normal","anomaly")],device=device,dtype=torch.int64)
        pair_count=(len(indices)+1)//2
        totals=dict(bce=0.,ap=0.,auc=0.,fpr95=0.)
        for pair,begin in enumerate(range(0,len(indices),2)):
            visits=list(range((update-1)*8+begin,min((update-1)*8+begin+2,len(order))))
            arrays=[scores[slice(*record["visit_offsets"][v:v+2])] for v in visits]
            labels=[points[slice(*record["input_offsets"][order[v]:order[v]+2])]["target"] for v in visits]
            logits=torch.tensor(np.concatenate(arrays),device=device,requires_grad=True)
            targets=torch.tensor(np.concatenate(labels),device=device)
            bce=balanced_loss(logits,targets,counts)
            terms=dict(bce=bce)
            totals["bce"]+=float(bce.detach())
            if log["ranking_weight"]>0:
                _,detail=ranking_loss(logits,targets,config["seed"]*100000000+update*8+pair,return_terms=True)
                for k in ("ap","auc","fpr95"):
                    terms[k]=detail["terms"][k]*log["ranking_weight"]*(1 if k=="ap" else .1)/pair_count
                    totals[k]+=float(detail[k])/pair_count
                for k in ("positive","negative","anchors","negatives"):
                    if detail[k]!=log["pairs"][pair][k]:
                        raise ValueError("saved score loss sampling differs from training")
            chosen_items=[item for item in items if item[2] in visits]
            if not chosen_items:
                continue
            gradients={k:torch.autograd.grad(term,logits,retain_graph=True)[0].detach().cpu().numpy() for k,term in terms.items()}
            offsets=np.r_[0,np.cumsum([len(a) for a in arrays])]
            for pi,cases,visit,chosen,slots in chosen_items:
                at=offsets[visits.index(visit)]+chosen
                label=materials[pi]["kind"]=="anomaly"
                signed_scores=(1-2*int(label))*arrays[visits.index(visit)][chosen]
                total=sum(g[at] for g in gradients.values())
                row=dict(profile=pi,index=order[visit],cases=cases,update=update,visit=visit,points=len(chosen),
                    wrong=int((signed_scores>=0).sum()) if not label else int((signed_scores>0).sum()),
                    ranking_weight=log["ranking_weight"],pair_positive=int((targets==1).sum()),
                    pair_negative=int((targets==0).sum()),effective_batch_normal=int(counts[0]),
                    recorded_parameter_gradient_norm=log["gradient_norm"],
                    loss_gradients={k:dict(sum=float(g[at].sum(dtype=float)),max_abs=float(np.abs(g[at]).max()),
                        nonzero=int(np.count_nonzero(g[at]))) for k,g in gradients.items()},
                    normal_push_up=int((total<0).sum()) if not label else None,
                    anomaly_push_down=int((total>0).sum()) if label else None,
                    point_examples=[dict(slot=int(slots[j]),score=float(arrays[visits.index(visit)][chosen[j]]),
                        gradients={k:float(g[at[j]]) for k,g in gradients.items()})
                        for j in np.argsort(-signed_scores,kind="stable")[:3]])
                output_rows.append(row)
        residual={k:totals[k]-log["loss_components"][k] for k in totals}
        if any(abs(v)>2e-6 for v in residual.values()):
            raise ValueError(f"saved score loss replay differs from actual log: {update}, {residual}")
        residuals.append(dict(update=update,errors=residual))
    val=load_manifest("assets/val.json","val")
    val_offsets={r["index"]:r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]}
    val_meta=np.load(OUTPUT/"val_points.npy",mmap_mode="r")
    step=json.loads((run/"result.json").read_text())["successful_updates"]
    val_scores=np.load(run/f"val{step}.npy",mmap_mode="r")
    val_boundaries=[]
    for case in [r for r in evidence["major_cases"] if r["kind"]=="normal"][:5]:
        query=case["failure"]; row=val["records"][query["index"]]; offset=val_offsets[query["index"]]
        meta=val_meta[offset["start"]:offset["stop"]]
        xyz=np.fromfile(row["scan"],dtype="<f4").reshape(-1,4)[meta["slot"]]
        selector=query["selector"]
        pose=poses_for(Path(row["scan"]).parents[1])[row["frame"]]
        world=xyz[:,:3].astype(float)@pose[:3,:3].T+pose[:3,3]
        normal=meta["target"]==0
        saved=np.load(ROOT_ERRORS/f"surface_{row['sequence']}.npz")
        comp=np.full(len(meta),-1)
        comp[normal]=saved["component"][np.searchsorted(saved["keys"],cell_keys(world[normal],meta["semantic"][normal]))]
        grid=np.floor((world if "world_cell" in selector else xyz[:,:3])/.75)
        chosen=np.flatnonzero(normal&(comp==selector["component"])&(grid==selector.get("world_cell",selector.get("cell"))).all(1))
        s=val_scores[offset["start"]:offset["stop"]]
        if len(chosen)!=query["points"]:
            raise ValueError("voxel boundary check changed the material point population")
        val_boundaries.append(dict(case=case["case"],query=query["query"],index=query["index"],
            **boundary_points(xyz,meta["target"],chosen,s[chosen]>=0)))
    result=dict(rows=output_rows,log_residuals=residuals,model_forwards=0,parameter_updates=0,
        training_boundaries=boundaries,validation_boundaries=val_boundaries,
        selection="Nearest same-domain material for the five largest anomaly and five largest normal cases",
        interpretation="FP32 loss derivatives at recorded logits with original CUDA sampling seeds, before cast to backbone output dtype and before parameter-gradient clipping. These are not shared-feature or parameter gradients, nor evidence that a parameter update must improve a point. Nonzero rank derivative does not equal sample inclusion when smooth comparisons saturate.")
    write_json(output/"score_gradients.json",result)
    return result


def analyze_records(run,output,workers):
    output.mkdir(parents=True,exist_ok=True)
    result=json.loads((run/"result.json").read_text())
    if not result["complete"]:
        raise ValueError("static full-run analysis requires completed records")
    ledger=json.loads((output/"ledger.json").read_text())
    if Path(ledger["predictions"])!=run/f"val{result['successful_updates']}.npy":
        raise ValueError("AP ledger belongs to a different run")
    start=time.perf_counter()
    resources=runtime_snapshot()
    scans,materials=learning_records(run,output,workers)
    state_records(run,output,scans)
    queries=validation_materials(run,output,workers)
    evidence,case_rows=link_evidence(output,materials,queries)
    gradients=score_gradients(run,output,materials,evidence)
    for row in case_rows:
        derivatives=[r for r in gradients["rows"] if row["case"] in r["cases"]]
        training_boundary=next((r for r in gradients["training_boundaries"] if row["case"] in r["cases"]),None)
        validation_boundary=next((r for r in gradients["validation_boundaries"] if row["case"]==r["case"]),None)
        row.update(score_derivative_updates=[r["update"] for r in derivatives],
                   training_boundary=training_boundary,validation_boundary=validation_boundary)
        if derivatives and row["kind"]=="normal":
            if all(r["normal_push_up"]==0 for r in derivatives):
                row["supported"]+="；已存分数复算的总损失导数均未把该正常局部推向高分"
                row["excluded"]+="；排除该局部完全漏监督或总分数导数方向反转"
            if all(r["recorded_parameter_gradient_norm"]<=1 for r in derivatives):
                row["excluded"]+="；两次访问所在更新均未触发梯度裁剪"
            if training_boundary["wrong_outside_mixed_voxels"]:
                row["excluded"]+=f"；训练局部有{training_boundary['wrong_outside_mixed_voxels']}个错点不在正常异常混合体素中"
        case=next((r for r in evidence["major_cases"] if r["case"]==row["case"]),None)
        if case:
            case["decision"]={k:row[k] for k in ("supported","excluded","missing","change")}
            case["score_derivative_updates"]=row["score_derivative_updates"]
            case["boundaries"]=dict(training=training_boundary,validation=validation_boundary)
    evidence["score_derivatives"]="score_gradients.json: original-score loss calculations only; no model forward or parameter update"
    evidence["normal_semantic_loss"]={str(k):sum(r["AP_loss"] for r in ledger["surfaces"] if r["semantic"]==k)
                                      for k in sorted({r["semantic"] for r in ledger["surfaces"]})}
    _compact_csv(output/"cases.csv",case_rows)
    write_json(output/"evidence.json",evidence)
    write_json(output/"analysis.json",dict(run=str(run),seconds=time.perf_counter()-start,workers=workers,
        resources=resources,training_forwards=0,evaluation_forwards=0,parameter_updates=0,
        matched_training_scans=len({r['index'] for r in scans}),matched_validation_queries=len(queries),
        score_derivative_records=len(gradients['rows']),coverage=evidence['coverage']))
    print(json.dumps(dict(seconds=time.perf_counter()-start,coverage=evidence['coverage']),ensure_ascii=False),flush=True)
