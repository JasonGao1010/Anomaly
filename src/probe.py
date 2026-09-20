"""Case-specific diagnostics of fixed C, with explicit intervention boundaries."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from .attribute import cell_keys, poses_for, rank_data, signature
from .data import Scans, file_sha256, load_manifest, write_json
from .diagnose import BEST_C, OUTPUT, validation_arrays, write_csv
from .train import disk_check, runtime_snapshot
from .model import prepare_scan


PATCH = .75
PARENTS = ("N141:75", "N125:30")


def _fragments(task):
    output, case = task
    output = Path(output)
    manifest = load_manifest("assets/val.json", "val")
    offsets = {r["index"]:r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]}
    saved = np.load(output/f"surface_{case['sequence']}.npz")
    first = manifest["records"][case["observations"][0][0]]
    poses = poses_for(Path(first["scan"]).parents[1])
    values, _, _, _, _, loss = rank_data()
    episodes, last = [], {}
    all_points, zero_loss_points = 0, 0
    for index, frame, expected_count, expected_loss in case["observations"]:
        xyzi, target, scores, meta = validation_arrays(offsets[index],manifest["records"][index])
        normal = np.flatnonzero(target==0)
        world = xyzi[normal,:3].astype(float)@poses[frame,:3,:3].T+poses[frame,:3,3]
        keys = cell_keys(world,meta["semantic"][normal])
        component = saved["component"][np.searchsorted(saved["keys"],keys)]
        selected = component==case["component"]
        normal, world = normal[selected], world[selected]
        weights = loss[np.searchsorted(values,scores[normal])]
        if len(normal)!=expected_count or abs(weights.sum()-expected_loss)>1e-8:
            raise ValueError("normal fragment points differ from the complete AP ledger")
        all_points += len(normal)
        zero_loss_points += int((weights==0).sum())
        cells, inverse, counts = np.unique(np.floor(world/PATCH).astype(int),axis=0,return_inverse=True,return_counts=True)
        masses = np.bincount(inverse,weights=weights)
        for j in np.flatnonzero(masses>0):
            cell = tuple(int(v) for v in cells[j])
            row = [index,frame,int(counts[j]),float(masses[j])]
            if cell not in last or last[cell]["observations"][-1][1]!=frame-1:
                episode = dict(id=f"{case['id']}/{'_'.join(map(str,cell))}/{frame}",
                    parent=case["id"],world_cell=list(cell),observations=[],AP_loss=0.,points=0)
                episodes.append(episode)
                last[cell] = episode
            episode = last[cell]
            episode["observations"].append(row)
            episode["AP_loss"] += row[3]
            episode["points"] += row[2]
    episodes.sort(key=lambda r:-r["AP_loss"])
    if abs(sum(r["AP_loss"] for r in episodes)-case["AP_loss"])>1e-8 or all_points!=case["points"]:
        raise ValueError("fragment episodes omit normal AP loss")
    cumulative, queries = 0., []
    for episode in episodes:
        episode["selected"] = cumulative<.9*case["AP_loss"]
        cumulative += episode["AP_loss"]
        if episode["selected"]:
            for index, frame, count, mass in episode["observations"]:
                queries.append(dict(index=index,case=case["id"],component=case["component"],semantic=case["semantic"],
                    world_cell=episode["world_cell"],fragment=episode["id"],expected_points=count,expected_AP=mass))
    return dict(parent=case["id"],AP_loss=case["AP_loss"],points=all_points,zero_loss_points=zero_loss_points,
        episodes=episodes,profile_queries=queries,selected_AP=sum(r["expected_AP"] for r in queries),
        scope="Consecutive positive-loss observations of one 0.75 m world cell. All points in each selected cell are retained. Physical identity still depends on the original pose estimate.")


def focus(output, workers):
    """Specify tests before inspecting their new outputs; then refine fixed cases."""
    disk_check(1_000_000_000)
    ledger = json.loads((output/"ledger.json").read_text())
    training = json.loads((output/"train_profiles.json").read_text())
    validation = json.loads((output/"val_profiles.json").read_text())
    links = json.loads((output/"material_links.json").read_text())["records"]
    current = load_manifest("results/data/native/train.json","train")
    old = load_manifest("assets/train.json","train")
    plan = dict(checkpoint=str(BEST_C),checkpoint_sha256=file_sha256(BEST_C),resources=runtime_snapshot(),
        case_tests={
            "P125:1":dict(fixed="C, existing geometries and placements, original rays, labels, full scans, and training-only descriptor scales",
                changed="Candidate observations: current representatives versus all existing observations of the top three STU candidate worlds",
                selection="Rank STU worlds by summed AP mass of queries retrieving them; inspect every available scan in the top three, without selecting by new predictions",
                support="Closer unselected observations support a gap in representative selection. Reliable related observations that C already ranks well support a generalization gap. Neither proves the cause of real-case AP loss without confirming geometry/response/context relevance.",
                decision="If new views improve relevant coverage, prioritize representative selection; if matched observations already rank well, examine transfer/representation before adding repetitions"),
            "P141:4":dict(fixed="C, all 285 real object observations, current training candidates, descriptor scales",
                changed="Retrieval description: full profile, geometry only, or omitting sampling block; no model update",
                support="5080 recurring across unrelated cases or losing its relevance under geometry checks weakens its use as evidence of insufficient learning. Confirm relevance before any fitting test.",
                fitting_if_relevant=dict(fixed="C initialization; same verified training observations and class-balanced BCE; independent same-world views and reserved geometry/log observations held out",
                    changed="At most 40 successful updates on four verified training scans, evaluated at start and endpoint; no other model/data change",
                    support="Training-only improvement suggests memorization; improvement on independent related observations supports insufficient use of those training observations. Neither automatically resolves real 141 or full STU AP.",
                    status="Not started: material relevance is a prerequisite")),
            "normal":dict(fixed="C and complete global score ties; N141:75 and N125:30 point identities",
                changed="Replace one representative with 0.75 m world cells and consecutive positive-loss observation episodes",
                selection="Retain every episode in the loss ledger; profile all observations in the smallest descending prefix accounting for at least 90% of each parent's AP loss",
                support="Inspect actual high-score patches and corresponding training normals before claiming a coverage gap",
                feature_test=dict(fixed="Frozen C; complete input scans; same sampled points and labels; two 64-dimensional representations; identical balanced linear readout and training-only scaling",
                    changed="Readout input: state before conditional context fusion versus actual state entering the C scoring head",
                    training="24 evenly spaced records per current source/type group, plus up to 16 most-supported normal training matches; no validation labels used for fitting",
                    readout="Training-only per-coordinate standardization; balanced logistic loss plus 0.5e-3 times squared weights; zero initialization, L-BFGS, at most 500 iterations, gradient tolerance 1e-7; no hyperparameter selection",
                    checks="Up to eight evenly spaced scans per reserved source unit; representative focused normal episodes and real 125/141 observations",
                    support="Better pre-fusion readout on independent observations suggests information may be less accessible after fusion; both poor suggests representation or coverage limits. This is a diagnostic, not an architecture ablation or causal proof."))},
        scope="Development diagnostics only. Preserve C and official evaluation. No production sampling or full training change.")
    write_json(output/"focus.json",plan)
    selected = [next(r for r in ledger["surfaces"] if r["id"]==name) for name in PARENTS]
    with ProcessPoolExecutor(max_workers=min(2,workers)) as executor:
        normal = list(executor.map(_fragments,[(str(output),r) for r in selected]))
    plan["normal"] = normal
    write_csv(output/"fragments.csv",[dict(parent=s["parent"],fragment=r["id"],AP_loss=r["AP_loss"],
        points=r["points"],world_cell=r["world_cell"],first_frame=r["observations"][0][1],
        last_frame=r["observations"][-1][1],observations=len(r["observations"]),profiled=r["selected"])
        for s in normal for r in s["episodes"]])
    masses = defaultdict(float)
    for link in links:
        q = validation["records"][link["query"]]
        if q["case"]=="P125:1":
            for n in link["neighbors"]:
                t = training["records"][n["profile"]]
                if t["domain"]=="stu":
                    masses[t["unit"]] += q["represented_AP_loss"]
    worlds = sorted(masses,key=lambda k:-masses[k])[:3]
    selected_ids = {(r.get("world"),r["frame"]):i for i,r in enumerate(current["records"]) if r.get("world")}
    plan["observation_worlds"] = [dict(world=w,candidate_query_AP=masses[w],
        scans=[dict(old_index=i,frame=r["frame"],selected_index=selected_ids.get((w,r["frame"])))
            for i,r in enumerate(old["records"]) if r["world"]==w]) for w in worlds]
    # Stability checks change the descriptor, never the trained model or labels.
    x = np.load(output/"train_profiles.npy").astype(float)
    q = np.load(output/"val_profiles.npy").astype(float)
    scale = np.asarray(next(r["scale"] for r in json.loads((output/"material_links.json").read_text())["normalizations"]
        if r["kind"]=="anomaly" and r["domain"]=="stu"))
    indices = np.array([i for i,r in enumerate(training["records"]) if r["kind"]=="anomaly" and r["domain"]=="stu"])
    variants = dict(full=np.arange(36),geometry=np.arange(17),without_sampling=np.r_[0:17,24:36])
    audit = []
    for qi,row in enumerate(validation["records"]):
        if row["kind"]!="anomaly":continue
        neighbors = {}
        for name,dims in variants.items():
            distance = np.linalg.norm((x[indices][:,dims]-q[qi,dims])/scale[dims],axis=1)
            ti = int(indices[int(np.argmin(distance))])
            neighbors[name] = dict(profile=ti,index=training["records"][ti]["index"],distance=float(distance.min()),
                BCE=training["records"][ti]["mean_BCE"],pair_error=training["records"][ti]["reference_pair_error"])
        audit.append(dict(case=row["case"],index=row["index"],frame=row["frame"],AP_loss=row["represented_AP_loss"],neighbors=neighbors))
    plan["candidate_audit"] = audit
    plan["unexplained_AP"] = ledger["AP_loss"]
    write_json(output/"focus.json",plan,indent=None)
    print(dict(normal=[dict(parent=r["parent"],episodes=len(r["episodes"]),profiled_observations=len(r["profile_queries"]),
        selected_AP=r["selected_AP"]) for r in normal],worlds=[(r["world"],len(r["scans"])) for r in plan["observation_worlds"]]),flush=True)


class ObservationScans(Scans):
    def __getitem__(self,index):
        sample = super().__getitem__(index)
        chosen = np.flatnonzero(sample["targets"]==1)
        vector = signature(sample["xyzi"],sample["targets"],chosen)
        result = prepare_scan(sample)
        result["profile"] = vector
        return result


def observe(output,workers):
    """Score unselected real ray observations of the same three fixed worlds."""
    import torch
    from torch.utils.data import DataLoader
    from .evaluate import autocast,load_model
    from .model import to_device
    from .diagnose import score_curve
    disk_check(500_000_000)
    plan = json.loads((output/"focus.json").read_text())
    if plan["checkpoint_sha256"]!=file_sha256(BEST_C):raise ValueError("C changed")
    old = load_manifest("assets/train.json","train")
    training = json.loads((output/"train_profiles.json").read_text())
    vectors = np.load(output/"train_profiles.npy")
    queries = json.loads((output/"val_profiles.json").read_text())
    qvectors = np.load(output/"val_profiles.npy")
    scale = np.asarray(next(r["scale"] for r in json.loads((output/"material_links.json").read_text())["normalizations"]
        if r["kind"]=="anomaly" and r["domain"]=="stu"))
    report = dict(checkpoint_sha256=plan["checkpoint_sha256"],resources=runtime_snapshot(),
        fixed_plan=plan["case_tests"]["P125:1"],worlds=[],records=[])
    for world in plan["observation_worlds"]:
        original = old["records"][world["scans"][0]["old_index"]]
        path = Path(original["delta"]).parents[1]/"world.json"
        world_data = json.loads(path.read_text())["world"]
        if len(world_data["objects"])!=1:raise ValueError("observation probe expects one fixed object per world")
        report["worlds"].append(dict(world=world["world"],definition=str(path),object=world_data["objects"][0]))
    write_json(output/"observations.json",report)
    curve = score_curve(np.load(output/"train_scores.npy",mmap_mode="r"),np.load(output/"train_labels.npy",mmap_mode="r"))
    negative_prefix = np.r_[0,np.cumsum(curve["negative"])]
    lookup = {r["old_index"]:r for w in plan["observation_worlds"] for r in w["scans"]}
    records, features = [], []
    for index,row in lookup.items():
        if row["selected_index"] is None:continue
        ti = next(i for i,r in enumerate(training["records"]) if r["index"]==row["selected_index"] and r["kind"]=="anomaly")
        material = training["records"][ti]
        records.append(dict(world=old["records"][index]["world"],**row,
            points=material["points"],score_quantiles=material["score_quantiles"],mean_BCE=material["mean_BCE"],
            within_scan_pair_error=material["within_scan_pair_error"],reference_pair_error=material["reference_pair_error"]))
        features.append(vectors[ti])
    indices = [i for i,r in lookup.items() if r["selected_index"] is None]
    device = torch.device("cuda")
    model,_ = load_model(BEST_C,device)
    loader = DataLoader(ObservationScans(old),batch_size=None,sampler=indices,num_workers=workers,
        prefetch_factor=1,pin_memory=True,generator=torch.Generator().manual_seed(0))
    start = time.perf_counter()
    with torch.no_grad():
        for number,sample in enumerate(loader,1):
            index = int(sample["index"])
            with autocast(device):scores = model(to_device(sample,device)).cpu().numpy()
            target = sample["targets"].numpy()
            positive,normal = scores[target==1],np.sort(scores[target==0])
            below = .5*(np.searchsorted(normal,positive,side="left")+np.searchsorted(normal,positive,side="right"))
            left,right = (np.searchsorted(curve["values"],positive,side=side) for side in ("left","right"))
            error = 1-(negative_prefix[left]+negative_prefix[right])/(2*negative_prefix[-1])
            records.append(dict(world=old["records"][index]["world"],**lookup[index],
                points=len(positive),score_quantiles=np.quantile(positive,[.1,.5,.9]).tolist(),
                mean_BCE=float(np.logaddexp(0.,-positive).mean()),
                within_scan_pair_error=float((1-below/len(normal)).mean()),reference_pair_error=float(error.mean())))
            features.append(sample["profile"].numpy())
            if number%50==0 or number==len(indices):
                print(f"same-world C inference {number}/{len(indices)}, {time.perf_counter()-start:.1f}s",flush=True)
    features = np.stack(features)
    report.update(records=records,seconds=time.perf_counter()-start,new_forward_scans=len(indices),reused_scans=len(lookup)-len(indices))
    matches=[]
    for qi,q in enumerate(queries["records"]):
        if q["case"]!="P125:1":continue
        choices={}
        for name,selected in (("selected",True),("unselected",False)):
            ids=np.array([i for i,r in enumerate(records) if (r["selected_index"] is not None)==selected])
            distances=np.linalg.norm((features[ids]-qvectors[qi])/scale,axis=1)
            ri=int(ids[np.argmin(distances)])
            delta=(features[ri]-qvectors[qi])/scale
            choices[name]=dict(record=ri,old_index=records[ri]["old_index"],distance=float(distances.min()),
                blocks={k:float(np.linalg.norm(delta[lo:hi])) for k,(lo,hi) in training["feature_blocks"].items()})
        matches.append(dict(index=q["index"],frame=q["frame"],AP_loss=q["represented_AP_loss"],matches=choices))
    report["matches"]=matches
    report["interpretation"]="Only observations change within each fixed geometry world. A larger candidate pool mechanically lowers nearest distances; relevance still requires shape, response and context checks. Unselected scans remain excluded from model fitting."
    np.save(output/"observation_profiles.npy",features)
    write_json(output/"observations.json",report,indent=None)
    print(dict(scans=len(records),new_forward_scans=len(indices),seconds=report["seconds"]),flush=True)


def _even(values,count):
    return [values[i] for i in np.unique(np.linspace(0,len(values)-1,min(count,len(values))).round().astype(int))]


def transfer(output, workers):
    """Test material learning and transfer under corrected C without changing truth."""
    import torch
    from torch.utils.data import DataLoader
    from .attribute import loss_weights, pair_errors
    from .diagnose import curve_summary, score_curve
    from .evaluate import PreparedScans, autocast, load_model
    from .model import to_device
    from .train import material_order

    path = output / "transfer.json"
    train = load_manifest("results/data/native/train.json", "train")
    old = load_manifest("assets/train.json", "train")
    val = load_manifest("assets/val.json", "val")
    baseline = json.loads((output / "precision_eval.json").read_text())
    focus_plan = json.loads((output / "focus.json").read_text())
    original = json.loads((BEST_C.parent / "sampling.json").read_text())["order"]
    modified = material_order(original, train, [6006], 1000, 1040)
    train_data = Scans(train)
    seed_sample = train_data[6006]
    seed_cell = [13, 2, -3]
    seed_indices = np.flatnonzero((seed_sample["targets"] == 0) &
        np.all(np.floor(seed_sample["xyzi"][:, :3] / PATCH).astype(int) == seed_cell, axis=1))
    pose = train_data.sequence[270].pose
    track_center = (seed_sample["xyzi"][seed_indices, :3].astype(float) @ pose[:3, :3].T + pose[:3, 3]).mean(0)
    used_frames = {train["records"][i]["frame"] for i in set(original[8000:8320] + modified[8000:8320])
                   if train["records"][i]["group"].endswith("_stu")}
    unseen = [i for i, r in enumerate(train["records"]) if r["group"] == "normal_stu"
              and abs(r["frame"] - 270) <= 40 and r["frame"] not in used_frames]
    unseen = _even(unseen, 12)
    worlds = focus_plan["observation_worlds"]
    observations = [r for world in worlds for r in world["scans"]]
    representative = {r["old_index"] for world in worlds for r in world["scans"] if r["selected_index"] is not None}
    for world in worlds:
        representative.update(r["old_index"] for r in _even([r for r in world["scans"] if r["selected_index"] is None], 12))
    plan = dict(checkpoint=str(BEST_C), checkpoint_sha256=file_sha256(BEST_C),
        attention_source=file_sha256(Path("vendor/litept/model.py")), train_manifest=train["sha256"],
        baseline=str(output / "precision_eval.json"), cases=["P125:1", "N125:30"],
        fixed="C state including optimizer and RNG; updates 1001-1040 of the original 1547-update schedule times 0.3; full scans, labels, loss, 5 cm voxels and corrected FP32 rotary arithmetic",
        changed="Only the identity of existing normal_stu input positions: original order versus record 6006",
        updates_per_arm=40, scans_per_arm=320, replaced_visits=sum(a != b for a, b in zip(original, modified)),
        material_index=6006, material_cell=seed_cell, material_points=len(seed_indices),
        track_center_world=track_center.tolist(), track_radius_m=.5, unfitted_view_indices=unseen,
        unfitted_scope="Existing 206 observations excluded by source frame from both new training segments; these are not new geometries or backgrounds and may have been seen before C",
        other_materials={"5757": "Semantic relevance audit only; not selected for extra use", "P125": "All 543 observations of the three previously chosen training worlds at baseline; fixed 45-observation subset at both endpoints"},
        interpretation=dict(
            learning="Lower error on the verified training road patch than control supports learnability and insufficient fitting of this patch at C",
            transfer="Improvement on unfitted observations AND N125/full AP beyond control supports usefulness of repeating this material; training-only improvement does not",
            stop="If training improves but real-case/full AP does not, do not increase repetitions of this material; coverage and representation/generalization remain unresolved",
            attribution="AP allocations are descriptive and their positive/normal margins cannot be added; no exclusive cause percentage is assigned"))
    if baseline["checkpoint_sha256"] != plan["checkpoint_sha256"] or baseline["attention_source"] != plan["attention_source"]:
        raise ValueError("transfer diagnosis must use the corrected C baseline")
    report = json.loads(path.read_text()) if path.exists() else dict(plan=plan, resources=runtime_snapshot(), arms={})
    if report["plan"] != plan:
        raise ValueError("the predeclared transfer comparison changed")
    checks = load_manifest(OUTPUT / "check.json", "train")
    check_logs = {checks["split"]["logs"][r["scene"]] for r in checks["records"] if "scene" in r}
    train_logs = {r["log_token"] for r in train["records"] if "log_token" in r}
    prior_checks = [str(OUTPUT / f"{name}.json") for name in ("m500", "m1547", "b1547")
                    if "check" in json.loads((OUTPUT / f"{name}.json").read_text())["splits"]]
    report["evidence_scope"] = dict(
        val19="Repeatedly used for development, model selection and case-guided material selection; not an independent generalization test, despite never entering gradients",
        local_weighting="Diagnostic intervention only; fixed record 6006 and its 24 slots are not a proposed general training rule",
        internal_development=dict(manifest=str(OUTPUT / "check.json"), records=len(checks["records"]),
            normal_scans=sum(r["group"] == "normal_nuscenes" for r in checks["records"]),
            synthetic_scans=sum(r["group"] == "targeted" for r in checks["records"]),
            logs=sorted(check_logs), anomaly_training_log_overlap=sorted(check_logs & train_logs),
            geometries=sorted({r["geometry"] for r in checks["records"] if "geometry" in r}),
            already_evaluated_in=prior_checks,
            limits="Normal-only nuScenes checks and new geometries on the same STU 206 background; cannot jointly establish unseen-background anomaly recognition"),
        pretraining=dict(config="https://github.com/prs-eth/LitePT/blob/436d04801c8151faebe66a1b2d368a9711e7e6aa/configs/nuscenes/semseg-litept-small-v1m1.py",
            dataset="https://github.com/prs-eth/LitePT/blob/436d04801c8151faebe66a1b2d368a9711e7e6aa/datasets/nuscenes.py",
            declared_split="train", exposure="The four check scenes belong to official nuScenes train; treat their backgrounds as pretraining-exposed under the published recipe. Author per-scan visit records are unavailable."),
        final_holdout=dict(established=False, reason="Current internal checks and val19 have already informed development; repartitioning them does not undo that exposure"),
        adoption_requires=["task-based sample and sensor rationale", "one uniform mechanism across structures",
                           "log/source-geometry-isolated controls with explicit pretraining exposure", "bounded search, failed results and seed variability records"],
        next_step="Define development and final-holdout roles before new training; use development for daily selection, val19 only at predeclared milestones, and independent testing after the final recipe is fixed")
    write_json(path, report)
    meta = np.load(OUTPUT / "val_points.npy", mmap_mode="r")
    offsets = json.loads((OUTPUT / "val_offsets.json").read_text())["rows"]
    surface = np.load(output / "surface_125.npz")
    val_poses = poses_for(Path(val["records"][0]["scan"]).parents[1])
    episodes = next(r for r in focus_plan["normal"] if r["parent"] == "N125:30")["episodes"][:2]
    sources = dict(baseline=(BEST_C, output / "precision_deployed.npy", baseline))
    for name in ("material-control", "material", "local"):
        directory = output.parent / "transfer" / name / "0/conditional"
        if (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_text())
            sources[name] = (directory / "last.pt", directory / "scores.npy", result)

    def stats(xyzi, target, scores, chosen, label, curve):
        if not len(chosen):
            return dict(points=0)
        cloud, values = xyzi[chosen], scores[chosen]
        # Readouts use each model's own complete-ranking thresholds and reference classes.
        rank = np.searchsorted(curve["values"], values)
        if np.any(rank == len(curve["values"])) or not np.array_equal(curve["values"][rank], values):
            # Training values need not be present in the validation score alphabet.
            population = curve["negative" if label else "positive"]
            cumulative = np.r_[0, np.cumsum(population)]
            lo = np.searchsorted(curve["values"], values, side="left")
            hi = np.searchsorted(curve["values"], values, side="right")
            below = (cumulative[lo] + cumulative[hi]) / (2 * cumulative[-1])
            error = 1 - below if label else below
        else:
            error = pair_errors(curve["positive"], curve["negative"])[0 if label else 1][rank]
        eigen, axes = np.linalg.eigh(np.cov(cloud[:, :3].astype(float).T)) if len(cloud) > 2 else (np.zeros(3), np.eye(3))
        reference = np.sort(scores[target == (0 if label else 1)])
        within = None
        if len(reference):
            below = (np.searchsorted(reference, values, side="left") + np.searchsorted(reference, values, side="right")) / (2 * len(reference))
            within = float((1 - below if label else below).mean())
        return dict(points=len(chosen), score_quantiles=np.quantile(values, [.1, .5, .9]).tolist(),
            mean_BCE=float(np.logaddexp(0., -values if label else values).mean()),
            fraction_wrong_at_zero=float(((values < 0) if label else (values >= 0)).mean()),
            reference_pair_error=float(error.mean()), within_scan_pair_error=within,
            errors_at_recall={str(int(r["target"] * 100)): int(((values < r["threshold"]) if label else (values >= r["threshold"])).sum()) for r in curve["operating"]},
            xyz_span=np.ptp(cloud[:, :3], axis=0).tolist(), intensity_quantiles=np.quantile(cloud[:, 3], [.1, .5, .9]).tolist(),
            plane_residual_m=float(np.sqrt(max(0., eigen[0]))), plane_normal_vertical=float(abs(axes[2, 0])))

    for name, (checkpoint, score_path, result) in sources.items():
        if name in report["arms"]:
            continue
        started = time.perf_counter()
        scores = np.load(score_path, mmap_mode="r")
        curve = score_curve(scores, meta["target"])
        expected = result["metrics"] if name == "baseline" else result["final_metrics"]
        if any(abs(curve[k] - expected[k]) > 1e-8 for k in ("AP", "FPR95", "AUROC")):
            raise ValueError("material endpoint disagrees with the complete official evaluation")
        _, positive_loss, normal_loss = loss_weights(curve["positive"], curve["negative"])
        arm = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint), metrics=curve_summary(curve),
                   cases={key: dict(points=0, AP_loss=0., observations=[]) for key in plan["cases"]},
                   fragments={r["id"]: dict(points=0, AP_loss=0., observations=[]) for r in episodes}, materials=[], observations=[])
        for offset in offsets:
            if offset["sequence"] != 125:
                continue
            record = val["records"][offset["index"]]
            points = meta[offset["start"]:offset["stop"]]
            s = scores[offset["start"]:offset["stop"]]
            x = np.fromfile(record["scan"], dtype="<f4").reshape(-1, 4)[points["slot"]]
            y = points["target"]
            pose = val_poses[record["frame"]]
            world = x[:, :3].astype(float) @ pose[:3, :3].T + pose[:3, 3]
            normal = np.flatnonzero(y == 0)
            keys = cell_keys(world[normal], points["semantic"][normal])
            positions = np.searchsorted(surface["keys"], keys)
            if not np.array_equal(surface["keys"][positions], keys):
                raise ValueError("normal case identity changed")
            component = surface["component"][positions]
            chosen = dict(P125=np.flatnonzero((y == 1) & (points["instance"] == 1)), N125=normal[component == 30])
            for key, indices in chosen.items():
                case = arm["cases"]["P125:1" if key == "P125" else "N125:30"]
                label = key == "P125"
                row = stats(x, y, s, indices, label, curve)
                mass = float((positive_loss if label else normal_loss)[np.searchsorted(curve["values"], s[indices])].sum())
                case["points"] += len(indices)
                case["AP_loss"] += mass
                case["observations"].append(dict(index=offset["index"], frame=record["frame"], AP_loss=mass, **row))
            for episode in episodes:
                if record["frame"] not in {r[1] for r in episode["observations"]}:
                    continue
                indices = chosen["N125"][np.all(np.floor(world[chosen["N125"]] / PATCH).astype(int) == episode["world_cell"], axis=1)]
                fragment = arm["fragments"][episode["id"]]
                mass = float(normal_loss[np.searchsorted(curve["values"], s[indices])].sum())
                fragment["points"] += len(indices)
                fragment["AP_loss"] += mass
                fragment["observations"].append(dict(index=offset["index"], frame=record["frame"], AP_loss=mass, **stats(x, y, s, indices, False, curve)))
        device = torch.device("cuda")
        model, _ = load_model(checkpoint, device)
        # All old views are rescored after the arithmetic fix, including the nine training representatives.
        selections = dict(train=sorted(set([5757, 6006] + unseen)),
                          old=[r["old_index"] for r in observations if name == "baseline" or r["old_index"] in representative])
        for split, indices in selections.items():
            manifest = train if split == "train" else old
            loader = DataLoader(PreparedScans(manifest), batch_size=None, sampler=indices, num_workers=workers,
                prefetch_factor=1, pin_memory=True, generator=torch.Generator().manual_seed(0))
            with torch.no_grad():
                for number, sample in enumerate(loader, 1):
                    index = int(sample["index"])
                    with autocast(device):
                        prediction = model(to_device(sample, device)).cpu().numpy()
                    x, y = sample["xyzi"].numpy(), sample["targets"].numpy()
                    row = manifest["records"][index]
                    if split == "old":
                        identity_row = next(r for r in observations if r["old_index"] == index)
                        arm["observations"].append(dict(world=row["world"], **identity_row,
                            **stats(x, y, prediction, np.flatnonzero(y == 1), True, curve)))
                    else:
                        source = train_data._source(row["frame"])
                        semantic = source.semantic[sample["slots"].numpy()]
                        if index in (5757, 6006):
                            cell = [4, 10, -3] if index == 5757 else seed_cell
                            chosen = np.flatnonzero((y == 0) & np.all(np.floor(x[:, :3] / PATCH).astype(int) == cell, axis=1))
                            labels, counts = np.unique(semantic[chosen], return_counts=True)
                            arm["materials"].append(dict(index=index, frame=row["frame"], selector=dict(cell=cell), role="retrieved_training_patch",
                                semantics={str(int(k)): int(v) for k, v in zip(labels, counts)},
                                **stats(x, y, prediction, chosen, False, curve)))
                        world = x[:, :3].astype(float) @ source.pose[:3, :3].T + source.pose[:3, 3]
                        chosen = np.flatnonzero((y == 0) & (semantic == 40) & (np.linalg.norm(world - track_center, axis=1) <= .5))
                        arm["materials"].append(dict(index=index, frame=row["frame"], selector=dict(world_center=track_center.tolist(), radius=.5),
                            role="fitted_track" if index == 6006 else "unfitted_track", **stats(x, y, prediction, chosen, False, curve)))
                    if number % 100 == 0:
                        print(f"material learning {name} {split} {number}/{len(indices)}", flush=True)
        arm["seconds"] = time.perf_counter() - started
        report["arms"][name] = arm
        write_json(path, report)
        del model
        torch.cuda.empty_cache()
        print(dict(arm=name, AP=curve["AP"], cases={k: v["AP_loss"] for k, v in arm["cases"].items()}, seconds=arm["seconds"]), flush=True)

    if "supervision" not in report:
        from .model import balanced_loss, ranking_loss, rank_sample
        device = torch.device("cuda")
        model, _ = load_model(BEST_C, device)
        prepared = PreparedScans(train)
        cache, rows = {}, []
        for begin in range(8000, 8320, 8):
            batch = modified[begin:begin + 8]
            counts = torch.tensor([sum(train["records"][i][key] for i in batch) for key in ("normal", "anomaly")], device=device)
            for pair_index in range(4):
                pair = batch[pair_index * 2:pair_index * 2 + 2]
                if 6006 not in pair:
                    continue
                for index in pair:
                    if index not in cache:
                        sample = prepared[index]
                        with torch.no_grad(), autocast(device):
                            values = model(to_device(sample, device)).cpu()
                        cache[index] = values, sample["targets"]
                scores = torch.cat([cache[i][0] for i in pair]).to(device).requires_grad_()
                targets = torch.cat([cache[i][1] for i in pair]).to(device)
                selected, cursor = [], 0
                for index in pair:
                    if index == 6006:
                        selected.extend((seed_indices + cursor).tolist())
                    cursor += len(cache[index][0])
                selected = torch.tensor(selected, device=device)
                if not torch.all(targets[selected] == 0):
                    raise ValueError("verified road points are not receiving normal supervision")
                seed = (begin // 8 + 1) * 8 + pair_index
                bce = balanced_loss(scores, targets, counts)
                _, details = ranking_loss(scores, targets, seed, return_terms=True)
                losses = dict(bce=bce, ap=details["terms"]["ap"] / 4,
                              auc=details["terms"]["auc"] * .1 / 4, fpr95=details["terms"]["fpr95"] * .1 / 4)
                gradients = {}
                for term, loss in losses.items():
                    gradient = torch.autograd.grad(loss, scores, retain_graph=True)[0]
                    local = gradient[selected]
                    if not torch.isfinite(gradient).all() or torch.any(local < 0) or torch.any(gradient[targets < 0] != 0):
                        raise ValueError("invalid score-gradient or ignored-point supervision")
                    gradients[term] = dict(sum=float(local.sum()), nonzero=int((local != 0).sum()),
                        all_points_abs_sum=float(gradient.abs().sum()))
                included = 0
                if details["positive"]:
                    _, sampled, _, _ = rank_sample(scores[targets == 1], scores[targets == 0], seed)
                    membership = torch.autograd.grad(sampled.sum(), scores)[0]
                    included = int((membership[selected] != 0).sum())
                rows.append(dict(update=begin // 8 + 1, pair_index=pair_index, indices=pair, seed=seed,
                    patch_point_visits=len(selected), included_in_ranking=included,
                    positive=details["positive"], positive_score_quantiles=torch.quantile(scores[targets == 1].detach(),
                        torch.tensor([.1, .5, .9], device=device)).tolist() if details["positive"] else None,
                    gradients=gradients))
        report["supervision"] = dict(scope="Fixed corrected C scores at all prescribed pair positions; a loss/gradient implementation check, not the gradients observed during the 40 updates or a causal optimization diagnosis",
            parameter_updates=0, records=rows, point_visits=sum(r["patch_point_visits"] for r in rows),
            included_in_ranking=sum(r["included_in_ranking"] for r in rows),
            point_visits_without_anomaly_partner=sum(r["patch_point_visits"] for r in rows if not r["positive"]),
            gradients={k: sum(r["gradients"][k]["sum"] for r in rows) for k in ("bce", "ap", "auc", "fpr95")},
            interpretation="All patch points have valid normal targets and correctly directed BCE gradients; ranking gradients depend on anomaly partners and score separation. This does not distinguish small effective supervision from interference through shared parameters.")
        write_json(path, report)

    if all(name in report["arms"] for name in ("baseline", "material-control", "material")):
        readings = {}
        for name, arm in report["arms"].items():
            patch = next(r for r in arm["materials"] if r["index"] == 6006 and r["role"] == "retrieved_training_patch")
            views = [r for r in arm["materials"] if r["role"] == "unfitted_track" and r["points"]]
            points = sum(r["points"] for r in views)
            fragments = arm["fragments"].values()
            readings[name] = dict(training_patch_BCE=patch["mean_BCE"],
                training_patch_wrong_at_zero=round(patch["points"] * patch["fraction_wrong_at_zero"]),
                unfitted_views=len(views), unfitted_points=points,
                unfitted_BCE=sum(r["points"] * r["mean_BCE"] for r in views) / points,
                unfitted_FP95=sum(r["errors_at_recall"]["95"] for r in views),
                focused_normal_AP_loss=sum(r["AP_loss"] for r in fragments),
                focused_normal_FP75=sum(o["errors_at_recall"]["75"] for r in fragments for o in r["observations"]),
                case_errors75={k: sum(r["errors_at_recall"]["75"] for r in v["observations"]) for k, v in arm["cases"].items()})
        root = output.parent / "transfer"
        configurations = [json.loads((root / name / "0/conditional/config.json").read_text())["configuration"]
                          for name in ("material-control", "material")]
        differences = [k for k in configurations[0] if configurations[0][k] != configurations[1][k]]
        if differences != ["branch"]:
            raise ValueError("the material comparison changed more than its input replacement")
        report["comparison"] = dict(readings=readings, configuration_differences=differences,
            AP_gain_over_control=report["arms"]["material"]["metrics"]["AP"] - report["arms"]["material-control"]["metrics"]["AP"],
            scope="One paired seed; identical initial RNG and per-pair rank seeds. Changing scan sizes can change subsequent stochastic-depth draws, so repeat uncertainty is not estimated.")
        b, c, m = (readings[k] for k in ("baseline", "material-control", "material"))
        learning = []
        for selected in (True, False):
            views = [r for r in report["arms"]["baseline"]["observations"] if (r["selected_index"] is not None) == selected]
            count = sum(r["points"] for r in views)
            learning.append(dict(scans=len(views), points=count, BCE=sum(r["points"] * r["mean_BCE"] for r in views) / count))
        control_ap = report["arms"]["material-control"]["metrics"]["AP"]
        material_ap = report["arms"]["material"]["metrics"]["AP"]
        report["findings"] = {
            "P125:1": dict(
                supported=(f"修正后重算三个既有训练世界的{sum(r['scans'] for r in learning)}个观测，"
                    f"{learning[0]['scans']}个已选观测{learning[0]['points']}点的BCE为{learning[0]['BCE']:.6f}，"
                    f"未选观测{learning[1]['points']}点的BCE为{learning[1]['BCE']:.6f}；这些候选大多已识别，真实125仍持续失分。"),
                excluded="当前结果不支持优先增加这三个世界的重复观测；尚不能排除其他几何、响应或背景关系覆盖不足。",
                missing="真实物体与候选的形态、表面响应和背景关系仍未证实等价；AP分摊下降不能直接等同于本物体各召回点的漏检减少。",
                modification="保留精度修正和已验证候选，不据此扩产这三个几何，也不更换交互模块。",
                next_test="固定现有异常候选的几何、位姿和射线，逐项核验其表面响应及邻接关系与失败/成功观测的差异；只在训练来源构造单因素对照，真实125仅评价。"),
            "N125:30": dict(
                supported=(f"记录6006为可信道路局部，但重复31次后BCE为{m['training_patch_BCE']:.6f}，原顺序对照为{c['training_patch_BCE']:.6f}，"
                    f"均未低于修正C的{b['training_patch_BCE']:.6f}。两目标连续片段失分为{c['focused_normal_AP_loss']:.6f}→{m['focused_normal_AP_loss']:.6f}，"
                    f"75%召回误报为{c['focused_normal_FP75']}→{m['focused_normal_FP75']}。整体AP增益未验证预设的局部学习与迁移解释。"),
                excluded="这24点均受到正常监督，损失对分数的梯度方向正确；未发现忽略标签或损失到分数这一段的梯度中断。该检查未追踪实际更新中分数到共享参数的梯度。全局AP提高不等于这两个道路片段学好了。5757局部为45点地形和5点人行道，不能直接充当同类道路证据。",
                missing=(f"仍未区分局部监督强度不足与共享参数更新的影响；不能确认真实N125主要属于没学够，亦不能确认它缺样本或学错关联。"
                    f"单次{material_ap-control_ap:.5f}个百分点额外AP收益的重复性未检验。"),
                modification=(f"保存{material_ap:.5f}%候选和{control_ap:.5f}%对照，保留{baseline['metrics']['AP']:.5f}%基线；"
                    "不将6006重复配方直接推广，也不由这次AP提高宣布N125归因完成。"),
                next_test="下一项可固定本轮6006重复输入、C完整起点和40步预算，仅比较原组合损失与原损失加1.0倍这24个训练正常点的局部均值BCE。先判断目标局部能否拟合，再看未参与更新观测、两个真实道路片段及完整AP；仅训练局部改善不能解释真实失分。该改变损失权重的实验本轮未执行。")}
        report["exclusive_additional_cause_AP"] = 0.
        report["scope"] = "Complete official ranking for all endpoints; focused material intervention, not a completed attribution of all remaining AP loss. Positive and normal AP allocations are separate margins."
        if "local" in report["arms"]:
            directory = root / "local/0/conditional"
            config = json.loads((directory / "config.json").read_text())["configuration"]
            trace = json.loads((directory / "local.json").read_text())
            control = configurations[1]
            differences = [k for k in set(config) | set(control) if config.get(k) != control.get(k)]
            if set(differences) != {"code", "branch", "local"}:
                raise ValueError("local diagnostic changed an additional training setting")
            streams = [json.loads((root / name / "0/conditional/sampling.json").read_text())["order"] for name in ("material", "local")]
            if streams[0] != streams[1] or [r["step"] for r in trace["updates"]] != list(range(1001, 1041)):
                raise ValueError("local diagnostic did not preserve all inputs and forty updates")
            if sum(r["local_point_visits"] for r in trace["updates"]) != 24 * 31 or any(
                    r["overflow"] or r["parameter_delta_l2"] <= 0 for r in trace["updates"]):
                raise ValueError("local diagnostic did not execute every planned update")
            checkpoints = [torch.load(root / name / "0/conditional/last.pt", map_location="cpu", weights_only=False)
                           for name in ("material", "local")]
            randoms = [r["rng"][0] for r in checkpoints]
            same_rng = all(torch.equal(randoms[0][k], randoms[1][k]) for k in ("torch", "cuda"))
            same_rng &= randoms[0]["python"] == randoms[1]["python"]
            same_rng &= (randoms[0]["numpy"][0] == randoms[1]["numpy"][0]
                         and np.array_equal(randoms[0]["numpy"][1], randoms[1]["numpy"][1])
                         and randoms[0]["numpy"][2:] == randoms[1]["numpy"][2:])
            del checkpoints
            w = readings["local"]
            steps = trace["updates"]
            effects = np.array([r["after_optimizer"]["mean_BCE"] - r["before_optimizer"]["mean_BCE"] for r in steps])
            local_ap = report["arms"]["local"]["metrics"]["AP"]
            report["local_comparison"] = dict(
                plan=dict(**config["local"], start="C update 1000 full state", updates=40,
                    fixed="material arm's exact 320 scan visits, remaining schedule, optimizer/RNG, model and original composite loss",
                    changed="add weight 1.0 normal BCE averaged over the 24 selected points and their occurrences within each effective batch",
                    decisions=dict(all_improve="supports local-supervision transfer; assess full AP before adoption",
                        training_only="fitting is possible; examine observation coverage and generalization",
                        no_fitting="inspect actual gradients, shared updates and optimization; no capacity conclusion")),
                configuration_differences=sorted(differences), identical_input_order=True, identical_final_rng=bool(same_rng),
                trace=str(directory / "local.json"), initial=trace["initial"], final=steps[-1]["after_optimizer"],
                actual_updates=dict(count=len(steps), improved=int((effects < 0).sum()), worsened=int((effects > 0).sum()),
                    unchanged=int((effects == 0).sum()), with_local_supervision=sum(r["local_point_visits"] > 0 for r in steps),
                    clipped=sum(r["gradient_norm"] > 1 for r in steps), parameter_delta_min=min(r["parameter_delta_l2"] for r in steps)),
                baseline_readout_note=(f"The cached C material readout has BCE {b['training_patch_BCE']:.6f}; this run starts at {trace['initial']['mean_BCE']:.6f}. "
                    "Repeated fixed-state monitor inference was identical. These separate executions are retained separately, not presented as one bitwise-identical prediction stream."),
                change_from_material={k: w[k] - m[k] for k in ("training_patch_BCE", "training_patch_wrong_at_zero", "unfitted_BCE", "unfitted_FP95", "focused_normal_AP_loss", "focused_normal_FP75")},
                AP_gain_over_material=local_ap-material_ap,
                scope="Single paired diagnostic on repeatedly used val19. Case-guided selection is development even without validation gradients; observed transfer is not independent cross-scene or cross-object generalization evidence.")
            finding = report["findings"]["N125:30"]
            fitted = w["training_patch_BCE"] < min(b["training_patch_BCE"], m["training_patch_BCE"])
            transferred = w["focused_normal_AP_loss"] < m["focused_normal_AP_loss"] and w["focused_normal_FP75"] < m["focused_normal_FP75"]
            finding["supported"] += (f" 局部权重1.0的40步诊断中，实际逐步记录的训练24点BCE由{trace['initial']['mean_BCE']:.6f}变为{steps[-1]['after_optimizer']['mean_BCE']:.6f}，"
                f"零分错误由{b['training_patch_wrong_at_zero']}变为{w['training_patch_wrong_at_zero']}；其他观测BCE为{w['unfitted_BCE']:.6f}。"
                f"两道路片段失分为{w['focused_normal_AP_loss']:.6f}，75%召回误报为{w['focused_normal_FP75']}；完整AP为{local_ap:.5f}%，"
                f"相对未加权素材组变化{local_ap-material_ap:+.5f}个百分点。")
            if fitted:
                finding["excluded"] += " 加强局部监督后该训练局部能够改善，因此不支持该局部在当前表示下完全不可拟合。"
                if transferred and w["unfitted_BCE"] < m["unfitted_BCE"]:
                    finding["supported"] += " 训练局部、其他观测与预选真实片段均改善，支持这处局部监督的迁移价值；不只是重复增加扫描次数。"
                    finding["missing"] = "val19已反复参与开发与素材选择，本次迁移不是独立泛化证据；尚缺统一局部规则在隔离背景和源几何上的对照、随机种子复核及极高召回误报变化解释。"
                    finding["next_test"] = ("先明确内部开发与最终留出来源，核对异常任务和预训练暴露；已有350条检查数据只作内部开发。"
                        "再预先固定只依赖训练来源标签与误报的统一局部规则，在多种结构上做同输入同预算对照和种子复核，日常选择只依靠内部开发，val19仅在约定节点评价。")
                else:
                    finding["missing"] = "训练局部可拟合不等于真实道路覆盖充分；仍缺少形态、响应、背景关系的受控迁移证据和重复性估计。"
                    finding["next_test"] = ("固定几何与背景，先在训练来源对目标道路的表面响应和观测条件作单因素变化，"
                        "保留未用于拟合的观测检验；对真实两道路片段只评价，判断已有素材与失败观测的差异是否解释迁移不足。")
            else:
                finding["missing"] = "局部拟合仍不足，需要区分梯度经过共享参数后的影响、优化器历史与训练/推理统计量的变化；不能据此归因于模型容量。"
                finding["next_test"] = "依据逐步前后分数，选择一次局部损失存在却未改善的实际更新，分解局部项与其余项的共享参数梯度及Adam更新方向，固定统计量检查其影响。"
            finding["modification"] = (f"分别保留局部加权{local_ap:.5f}%与未加权{material_ap:.5f}%候选，固定{baseline['metrics']['AP']:.5f}%诊断起点；"
                + ("局部加权仅作为有迁移迹象的诊断候选，尚不固化正式规则。" if transferred
                   else "本次加权未同时改善两道路片段的排序失分与误报，不采用为正式训练规则；后续由观测覆盖与泛化检验决定样本修改。"))
            report["findings"]["P125:1"]["supported"] += (
                f" 本次仅改变正常局部监督，P125失分由未加权组的{report['arms']['material']['cases']['P125:1']['AP_loss']:.6f}"
                f"变为{report['arms']['local']['cases']['P125:1']['AP_loss']:.6f}，75%全局召回下漏检由{m['case_errors75']['P125:1']}"
                f"变为{w['case_errors75']['P125:1']}。该变化包含共享参数与全局排序的影响，不能据此解释为异常几何覆盖已经补齐。")
        write_json(path, report)


def features(output,workers):
    """Measure accessible information with matched readouts of frozen C states."""
    import torch
    from torch.utils.data import DataLoader
    from scipy.optimize import minimize
    from scipy.special import expit
    from threadpoolctl import threadpool_limits
    from .evaluate import PreparedScans,autocast,load_model
    from .model import to_device
    from .diagnose import score_curve,curve_summary
    disk_check(2_000_000_000)
    plan=json.loads((output/"focus.json").read_text())
    training=json.loads((output/"train_profiles.json").read_text())["records"]
    queries=json.loads((output/"val_profiles.json").read_text())["records"]
    links=json.loads((output/"material_links.json").read_text())["records"]
    manifests=dict(train=load_manifest("results/data/native/train.json","train"),
        check=load_manifest("results/train/native/diagnostics/check.json","train"),val=load_manifest("assets/val.json","val"))
    chosen_train=[]
    for group in sorted({r["group"] for r in manifests["train"]["records"]}):
        chosen_train.extend(_even([i for i,r in enumerate(manifests["train"]["records"]) if r["group"]==group],24))
    mass=defaultdict(float)
    for link in links:
        q=queries[link["query"]]
        if q["case"] in PARENTS:
            for n in link["neighbors"]:mass[n["profile"]]+=q["represented_AP_loss"]
    matched=[]
    for profile in sorted(mass,key=lambda i:-mass[i]):
        index=training[profile]["index"]
        if index not in [training[i]["index"] for i in matched]:matched.append(profile)
        if len(matched)==16:break
    chosen_train=sorted(set(chosen_train+[training[i]["index"] for i in matched]))
    train_patches=defaultdict(list)
    for i in matched:train_patches[training[i]["index"]].append(training[i]["selector"]["cell"])
    units=defaultdict(list)
    for i,r in enumerate(manifests["check"]["records"]):units[r.get("scene",r.get("geometry"))].append(i)
    chosen_check=sorted(i for unit in units.values() for i in _even(unit,8))
    ledger=json.loads((output/"ledger.json").read_text())
    chosen_val=set()
    focused_patches=defaultdict(list)
    for case in plan["normal"]:
        for episode in case["episodes"][:8]:
            obs=max(episode["observations"],key=lambda r:r[3])
            chosen_val.add(obs[0])
            focused_patches[obs[0]].append((case["parent"],episode["world_cell"]))
    for name in ("P125:1","P141:4"):
        case=next(r for r in ledger["objects"] if r["id"]==name)
        chosen_val.update(r["index"] for r in _even(case["observations"],12))
    selection=dict(train=chosen_train,check=chosen_check,val=sorted(chosen_val))
    report=dict(checkpoint_sha256=file_sha256(BEST_C),resources=runtime_snapshot(),
        plan=plan["case_tests"]["normal"]["feature_test"],selection=selection,
        source_manifests={k:v["sha256"] for k,v in manifests.items()},matched_training_profiles=matched,
        point_rule="Up to 2048 normal and 512 anomaly points uniformly per scan, deterministic RNG seed 431 + manifest index; include all named normal patches additionally. Same points for both feature representations and original C scores.",
        validation_rule="Eight highest-loss episodes per focused normal parent, max-loss frame per episode; twelve evenly spaced observations of each of P125:1/P141:4. Evaluation only, never readout fitting.")
    report["plan"]["readout"]="Training-only standardization; balanced logistic loss plus 0.5e-3 squared weights; zero initialization; L-BFGS with maxiter=500, gtol=1e-7, ftol=1e-12; no tuning"
    report["plan"]["matched_context_check"]="Two 448-dimensional inputs: six unchanged context tokens plus either the pre-interaction point state or the fused point state. Both readouts have 449 parameters. The existing 64-dimensional readouts are query-only/fused-only references; they cannot isolate loss of backbone context information."
    if report["checkpoint_sha256"]!=plan["checkpoint_sha256"]:raise ValueError("C changed")
    write_json(output/"features.json",report)
    device=torch.device("cuda")
    model,saved_model=load_model(BEST_C,device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    captured=defaultdict(list)
    def hook(name):
        def receive(module,inputs,result):captured[name].append(result.detach())
        return receive
    def head_input(module,inputs):captured["post"].append(inputs[0].detach())
    def context_input(module,inputs):captured["context"].append(inputs[0].detach())
    handles=[model.conditional.detail.register_forward_hook(hook("query")),
        model.conditional.sensor.register_forward_hook(hook("sensor")),
        model.point_detail.register_forward_hook(hook("detail")),model.point_position.register_forward_hook(hook("position")),
        model.head.register_forward_pre_hook(head_input),
        model.conditional.layers[0]["key"].register_forward_pre_hook(context_input)]
    outputs={name:[] for name in ("pre","post","context","labels","scores","split","source_index","slots","focus_normal")}
    normal_mass=defaultdict(float)
    rank_values,_,_,_,_,normal_weights=rank_data()
    start=time.perf_counter()
    for split_number,(split,indices) in enumerate(selection.items()):
        loader=DataLoader(PreparedScans(manifests[split]),batch_size=None,sampler=indices,num_workers=workers,
            prefetch_factor=1,pin_memory=True,generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            for sample in loader:
                index=int(sample["index"])
                target=sample["targets"].numpy()
                rng=np.random.default_rng(431+index)
                point_indices=[rng.choice(np.flatnonzero(target==label),min(limit,int((target==label).sum())),replace=False)
                    for label,limit in ((0,2048),(1,512))]
                patch_mask=np.zeros(len(target),bool)
                xyz=sample["xyzi"].numpy()[:,:3]
                if split=="train":
                    grid=np.floor(xyz/.75).astype(int)
                    for cell in train_patches[index]:patch_mask|=(target==0)&np.all(grid==cell,axis=1)
                if split=="val" and index in focused_patches:
                    record=manifests[split]["records"][index]
                    pose=poses_for(Path(record["scan"]).parents[1])[record["frame"]]
                    world=xyz.astype(float)@pose[:3,:3].T+pose[:3,3]
                    grid=np.floor(world/PATCH).astype(int)
                    # Patch identities are restricted to the original parent surface.
                    offset=next(r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"] if r["index"]==index)
                    _,_,cached_scores,meta=validation_arrays(offset,record)
                    normal=np.flatnonzero(target==0)
                    saved=np.load(output/f"surface_{record['sequence']}.npz")
                    component=saved["component"][np.searchsorted(saved["keys"],cell_keys(world[normal],meta["semantic"][normal]))]
                    for parent,cell in focused_patches[index]:
                        selected=normal[(component==int(parent.split(":")[1]))&np.all(grid[normal]==cell,axis=1)]
                        normal_mass[parent]+=float(normal_weights[np.searchsorted(rank_values,cached_scores[selected])].sum())
                        patch_mask[selected]=True
                selected=np.unique(np.r_[*point_indices,np.flatnonzero(patch_mask)])
                captured.clear()
                batch=to_device(sample,device)
                with autocast(device):prediction=model(batch)
                # Match the original mixed-precision addition order of the point state.
                pre_voxel=captured["query"][0]+captured["sensor"][0]
                pre=(pre_voxel[batch["inverse"]]+torch.cat(captured["detail"]))+torch.cat(captured["position"])
                post=torch.cat(captured["post"])
                outputs["pre"].append(pre[selected].float().cpu().numpy())
                outputs["post"].append(post[selected].float().cpu().numpy())
                outputs["context"].append(captured["context"][0][batch["inverse"][selected]].flatten(1).float().cpu().numpy())
                outputs["labels"].append(target[selected])
                outputs["scores"].append(prediction[selected].cpu().numpy())
                outputs["split"].append(np.full(len(selected),split_number,np.int8))
                outputs["source_index"].append(np.full(len(selected),index,np.int32))
                outputs["slots"].append(sample["slots"].numpy()[selected])
                outputs["focus_normal"].append(patch_mask[selected] if split=="val" else np.zeros(len(selected),bool))
        print(f"frozen features {split}: {len(indices)} complete scans, {time.perf_counter()-start:.1f}s",flush=True)
    for handle in handles:handle.remove()
    report["C_parameters_unchanged"]=all(torch.equal(v.detach().cpu(),saved_model["model"][k]) for k,v in model.state_dict().items())
    if not report["C_parameters_unchanged"]:raise ValueError("feature extraction changed C")
    del model,saved_model,captured,batch,pre,post,prediction
    torch.cuda.empty_cache()
    data={k:np.concatenate(v) for k,v in outputs.items()}
    data["checkpoint_sha256"]=np.array(report["checkpoint_sha256"])
    train_mask=data["split"]==0
    y=data["labels"][train_mask].astype(float)
    weights=np.where(y==1,.5/(y==1).sum(),.5/(y==0).sum())
    report["readouts"]={}
    predictions={"C":data["scores"]}
    with threadpool_limits(limits=6):
        for name in ("pre","post","input","context_post"):
            source="pre" if name=="input" else "post" if name=="context_post" else name
            x=(np.concatenate((data[source],data["context"]),axis=1) if name in ("input","context_post") else data[source]).astype(float)
            center,scale=x[train_mask].mean(0),np.maximum(x[train_mask].std(0),1e-4)
            x=(x-center)/scale
            fitted=x[train_mask]
            def objective(theta):
                scores=fitted@theta[:-1]+theta[-1]
                residual=weights*(expit(scores)-y)
                loss=float((weights*(np.logaddexp(0.,scores)-y*scores)).sum()+.5e-3*(theta[:-1]**2).sum())
                gradient=np.r_[fitted.T@residual+1e-3*theta[:-1],residual.sum()]
                return loss,gradient
            result=minimize(objective,np.zeros(x.shape[1]+1),jac=True,method="L-BFGS-B",options=dict(maxiter=500,gtol=1e-7,ftol=1e-12))
            if not result.success:raise RuntimeError(f"linear readout failed: {result.message}")
            predictions[name]=x@result.x[:-1]+result.x[-1]
            report["readouts"][name]=dict(parameters=result.x.tolist(),center=center.tolist(),scale=scale.tolist(),
                dimension=x.shape[1],loss=float(result.fun),iterations=int(result.nit),gradient_max=float(np.abs(result.jac).max()))
    report["metrics"]={}
    for split_number,split in enumerate(selection):
        selected=data["split"]==split_number
        report["metrics"][split]=dict(normal=int((data["labels"][selected]==0).sum()),anomaly=int((data["labels"][selected]==1).sum()),models={})
        for name,scores in predictions.items():
            curve=score_curve(scores[selected],data["labels"][selected])
            metrics=curve_summary(curve)
            if split=="val":
                threshold=curve["operating"][1]["threshold"]
                metrics["focused_normal_points"]=int(data["focus_normal"].sum())
                metrics["focused_normal_FP_at_subset_75_recall"]=int((scores[data["focus_normal"]]>=threshold).sum())
            report["metrics"][split]["models"][name]=metrics
    report.update(seconds=time.perf_counter()-start,focused_normal_AP_mass=dict(normal_mass),
        interpretation="Same sampled points, frozen C and subset AP only. pre/post are 64-dimensional query-only/fused-only references. input/context_post share identical six-scale context tokens and 449 readout parameters; only the point state differs. Raw context bypasses compression in both wide readouts, so this does not alone prove information loss by the deployed fusion or an architecture's causal effect. C is unchanged.")
    np.savez_compressed(output/"features.npz",**data,**{f"{name}_scores":score for name,score in predictions.items() if name!="C"})
    write_json(output/"features.json",report)
    print({k:{name:row["AP"] for name,row in value["models"].items()} for k,value in report["metrics"].items()},flush=True)


def review(output):
    """Render actual local returns and summarize tested explanations separately."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties,findfont
    from matplotlib.ft2font import FT2Font
    for font in ("times.ttf","simsun.ttc"):fontManager.addfont(f"/mnt/c/Windows/Fonts/{font}")
    chinese=FontProperties(fname="/mnt/c/Windows/Fonts/simsun.ttc")
    plt.rcParams.update({"font.family":"Times New Roman","pdf.fonttype":42,"font.size":10})
    plan=json.loads((output/"focus.json").read_text())
    train=json.loads((output/"train_profiles.json").read_text())["records"]
    val=json.loads((output/"val_profiles.json").read_text())["records"]
    links=json.loads((output/"material_links.json").read_text())["records"]
    observation=json.loads((output/"observations.json").read_text()) if (output/"observations.json").exists() else None
    probe=json.loads((output/"features.json").read_text()) if (output/"features.json").exists() else None
    data=Scans(load_manifest("results/data/native/train.json","train"))
    old=Scans(load_manifest("assets/train.json","train"))
    manifest=load_manifest("assets/val.json","val")
    offsets={r["index"]:r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]}

    def cloud(row,kind):
        if kind=="val":
            record=manifest["records"][row["index"]]
            xyzi,target,_,meta=validation_arrays(offsets[row["index"]],record)
            if row["kind"]=="anomaly":
                chosen=np.flatnonzero((target==1)&(meta["instance"]==row["selector"]["instance"]))
            else:
                pose=poses_for(Path(record["scan"]).parents[1])[record["frame"]]
                normal=np.flatnonzero(target==0)
                world=xyzi[normal,:3].astype(float)@pose[:3,:3].T+pose[:3,3]
                saved=np.load(output/f"surface_{record['sequence']}.npz")
                component=saved["component"][np.searchsorted(saved["keys"],cell_keys(world,meta["semantic"][normal]))]
                chosen=normal[(component==row["selector"]["component"])&
                    np.all(np.floor(world/PATCH).astype(int)==row["selector"]["world_cell"],axis=1)]
            title=f"{row['case']} / {row['frame']:06d}"
        else:
            sample=(old if kind=="old" else data)[row["old_index"] if kind=="old" else row["index"]]
            xyzi,target=sample["xyzi"],sample["targets"]
            if kind=="old" or row["kind"]=="anomaly":
                chosen=np.flatnonzero(target==1)
            else:
                chosen=np.flatnonzero((target==0)&np.all(np.floor(xyzi[:,:3]/.75).astype(int)==row["selector"]["cell"],axis=1))
            title=f"{'old' if kind=='old' else 'train'} #{row['old_index'] if kind=='old' else row['index']} / {row['frame']:06d}"
        if len(chosen)!=row["points"]:raise ValueError("review cloud does not match material identity")
        xyz=xyzi[:,:3]-np.median(xyzi[chosen,:3],axis=0)
        return xyz,chosen,title+f"\nn={len(chosen)}, score median={row['score_quantiles'][1]:.3f}"

    def draw(rows,stem,title):
        fig=plt.figure(figsize=(4*len(rows[0]),3.5*len(rows)))
        for i,panels in enumerate(rows):
            radius=max(1.,np.ceil(max(np.abs(x[indices]).max() for x,indices,_ in panels)*2+.5)/2)
            for j,(xyz,selected,label) in enumerate(panels):
                ax=fig.add_subplot(len(rows),len(panels),i*len(panels)+j+1,projection="3d")
                near=np.flatnonzero(np.linalg.norm(xyz,axis=1)<radius*1.5)
                near=near[np.linspace(0,len(near)-1,min(2000,len(near))).round().astype(int)]
                ax.scatter(*xyz[near].T,c=".75",s=.6,alpha=.35,rasterized=True)
                ax.scatter(*xyz[selected].T,c="#1768ac",s=6,rasterized=True)
                ax.set(xlim=(-radius,radius),ylim=(-radius,radius),zlim=(-radius,radius),xlabel="x (m)",ylabel="y (m)",zlabel="z (m)")
                ax.set_title(label,fontsize=9)
                ax.view_init(elev=25,azim=-55)
        fig.suptitle(title,fontproperties=chinese)
        fig.tight_layout(rect=(0,0,1,.97))
        fig.canvas.draw()
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_text():continue
            expected="SimSun" if any('\u4e00'<=c<='\u9fff' for c in text.get_text()) else "Times New Roman"
            if FT2Font(findfont(text.get_fontproperties(),fallback_to_default=False)).family_name!=expected:
                raise ValueError("review figure font fallback")
        fig.savefig(output/f"{stem}.pdf")
        fig.savefig(output/f"{stem}.png",dpi=160)
        plt.close(fig)

    panels=[]
    normal_reviews=[]
    for parent in plan["normal"]:
        for episode in parent["episodes"][:2]:
            qi=max((i for i,r in enumerate(val) if r["selector"].get("fragment")==episode["id"]),key=lambda i:val[i]["represented_AP_loss"])
            match=min((n for n in links[qi]["neighbors"] if train[n["profile"]]["domain"]=="stu"),key=lambda n:n["distance"])
            candidate=train[match["profile"]]
            panels.append([cloud(val[qi],"val"),cloud(candidate,"train")])
            normal_reviews.append(dict(fragment=episode["id"],query_index=val[qi]["index"],query_AP=val[qi]["represented_AP_loss"],
                train_index=candidate["index"],train_profile=match["profile"],distance=match["distance"],blocks=match["blocks"],
                query_BCE=val[qi]["mean_BCE"],train_BCE=candidate["mean_BCE"]))
    draw(panels,"fragments","主要正常局部及其同传感器训练近邻")
    panels=[]
    if observation and observation.get("matches"):
        for match in sorted(observation["matches"],key=lambda r:-r["AP_loss"])[:2]:
            query=next(r for r in val if r["case"]=="P125:1" and r["index"]==match["index"])
            panels.append([cloud(query,"val")]+[cloud(observation["records"][match["matches"][name]["record"]],"old") for name in ("selected","unselected")])
    relevant=[r for r in plan["candidate_audit"] if r["case"]=="P141:4" and r["neighbors"]["full"]["index"]==5080]
    for row in sorted(relevant,key=lambda r:-r["AP_loss"])[:2]:
        query=next(r for r in val if r["case"]==row["case"] and r["index"]==row["index"])
        panels.append([cloud(query,"val")]+[cloud(train[row["neighbors"][name]["profile"]],"train") for name in ("full","geometry")])
    draw(panels,"relevance","真实失分观测与不同条件下检索的训练候选")
    summary=dict(checkpoint_sha256=plan["checkpoint_sha256"],normal_local_reviews=normal_reviews,
        normal=[dict(parent=r["parent"],AP_loss=r["AP_loss"],episodes=len(r["episodes"]),selected_episodes=sum(e["selected"] for e in r["episodes"]),
            profiled_observations=len(r["profile_queries"]),profiled_AP=r["selected_AP"],unprofiled_AP=r["AP_loss"]-r["selected_AP"]) for r in plan["normal"]],
        case_findings={},cause_confirmed_AP=0.,causal_unresolved_AP=plan["unexplained_AP"],
        interpretation="Hypothesis checks narrow next actions; no controlled intervention yet identifies the cause of a positive amount of complete-validation AP loss.")
    for case in ("P125:1","P141:4"):
        audit=[r for r in plan["candidate_audit"] if r["case"]==case]
        others=[r for r in audit if r["neighbors"]["full"]["index"]!=5080]
        summary["case_findings"][case]=dict(AP_loss=sum(r["AP_loss"] for r in audit),
            candidate_5080_AP={name:sum(r["AP_loss"] for r in audit if r["neighbors"][name]["index"]==5080) for name in ("full","geometry","without_sampling")},
            other_full_neighbors_BCE=sum(r["AP_loss"]*r["neighbors"]["full"]["BCE"] for r in others)/sum(r["AP_loss"] for r in others))
    if observation and observation.get("matches"):
        total=sum(r["AP_loss"] for r in observation["matches"])
        summary["case_findings"]["P125:1"]["same_world_observations"]=dict(scans=len(observation["records"]),new_forward_scans=observation["new_forward_scans"],
            closer_unselected_AP=sum(r["AP_loss"] for r in observation["matches"] if r["matches"]["unselected"]["distance"]<r["matches"]["selected"]["distance"]),
            weighted_candidate_BCE={name:sum(r["AP_loss"]*observation["records"][r["matches"][name]["record"]]["mean_BCE"] for r in observation["matches"])/total for name in ("selected","unselected")},
            finding="New within-world views improve nearest descriptor distances but are mostly already recognized by C; this does not support prioritizing more repetitions of these geometries. Material equivalence to real 125 remains unverified.")
    summary["case_findings"]["P141:4"]["finding"]="5080's relevance is not confirmed by the geometry-only check. This does not prove it unrelated, but is insufficient to call 141 undertrained; conditional fitting remains unstarted."
    if probe and "metrics" in probe:
        summary["feature_readout"]=dict(metrics=probe["metrics"],normal_AP_mass=probe["focused_normal_AP_mass"],
            finding="The query-only reference omits backbone context and cannot diagnose its loss in fusion. Compare input/context_post for equal context and capacity, and treat any gain from bypassing context compression as a clue requiring a trained architecture control.",
            limitation=probe["interpretation"])
        summary["feature_readout"].update(_readout_cases(output,plan,probe,val,links,train))
    path=output/"precision.json"
    numerical=json.loads(path.read_text()) if path.exists() else None
    if numerical and numerical.get("complete_validation") and "metrics" in numerical:
        if numerical["checkpoint_sha256"]!=plan["checkpoint_sha256"]:raise ValueError("numerical C differs")
        summary["numerical_intervention"]=dict(report=str(path),metrics=numerical["metrics"],
            finding="A controlled change of rotary-position arithmetic isolated a numerical information defect and measured its complete-ranking effect. The remaining material and learning causes are not thereby identified.")
        summary["historical_AP_partition_scope"]="cause_confirmed_AP is an exclusive historical partition, not a claim that no numerical mechanism has been identified; use numerical_intervention for its measured effect."
        summary["interpretation"]="The material/readout fields are historical observations before the rotary fix. Numerical and material interventions below retain their separate initializations, score populations and causal limits."
    path = output / "transfer.json"
    if path.exists():
        transfer_report = json.loads(path.read_text())
        if "findings" in transfer_report:
            summary["material_intervention"] = dict(report=str(path), findings=transfer_report["findings"],
                comparison=transfer_report["comparison"], scope=transfer_report["scope"])
            if "local_comparison" in transfer_report:
                summary["material_intervention"]["local_comparison"] = transfer_report["local_comparison"]
            summary["material_intervention"]["evidence_scope"] = transfer_report["evidence_scope"]
    write_json(output/"review.json",summary)


def _readout_cases(output,plan,probe,queries,links,training):
    """Trace the existing readout predictions back to each inspected normal patch."""
    from .diagnose import score_curve,curve_summary
    manifest=load_manifest("assets/val.json","val")
    offsets={r["index"]:r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]}
    with np.load(output/"features.npz") as saved:
        if str(saved["checkpoint_sha256"])!=plan["checkpoint_sha256"]:raise ValueError("feature checkpoint differs")
        selected=saved["split"]==2
        indices,slots,labels=[saved[k][selected] for k in ("source_index","slots","labels")]
        scores={name:saved["scores" if name=="C" else f"{name}_scores"][selected]
            for name in probe["metrics"]["val"]["models"]}
        focus_mask=saved["focus_normal"][selected]
    # The wide readouts see precisely the same context coordinates and scaling.
    a,b=(probe["readouts"][name] for name in ("input","context_post"))
    if len(a["parameters"])!=449 or len(b["parameters"])!=449:raise ValueError("unequal readout capacity")
    for name in ("center","scale"):
        if not np.array_equal(a[name][64:],b[name][64:]):raise ValueError("context scaling differs")
    cached=np.empty(len(indices),np.float32)
    frames={}
    for index in np.unique(indices):
        row=manifest["records"][index]
        xyzi,target,prediction,meta=validation_arrays(offsets[index],row)
        positions=np.flatnonzero(indices==index)
        take=np.searchsorted(meta["slot"],slots[positions])
        if not np.array_equal(meta["slot"][take],slots[positions]) or not np.array_equal(target[take],labels[positions]):
            raise ValueError("readout points differ from fixed validation identities")
        cached[positions]=prediction[take]
        frames[index]=(xyzi,target,meta)
    thresholds={name:row["operating"][1]["threshold"] for name,row in probe["metrics"]["val"]["models"].items()}
    rows,used=[],np.zeros(len(indices),bool)
    for parent in plan["normal"]:
        for episode in parent["episodes"][:8]:
            qi=max((i for i,r in enumerate(queries) if r["selector"].get("fragment")==episode["id"]),
                key=lambda i:queries[i]["represented_AP_loss"])
            query=queries[qi]
            index=query["index"]
            row=manifest["records"][index]
            xyzi,target,meta=frames[index]
            pose=poses_for(Path(row["scan"]).parents[1])[row["frame"]]
            normal=np.flatnonzero(target==0)
            world=xyzi[normal,:3].astype(float)@pose[:3,:3].T+pose[:3,3]
            with np.load(output/f"surface_{row['sequence']}.npz") as surface:
                component=surface["component"][np.searchsorted(surface["keys"],cell_keys(world,meta["semantic"][normal]))]
            chosen=normal[(component==query["selector"]["component"])&
                np.all(np.floor(world/PATCH).astype(int)==query["selector"]["world_cell"],axis=1)]
            positions=np.flatnonzero(indices==index)
            take=positions[np.searchsorted(slots[positions],meta["slot"][chosen])]
            if not np.array_equal(slots[take],meta["slot"][chosen]) or len(take)!=query["points"] or used[take].any():
                raise ValueError("normal feature patches are incomplete or overlapping")
            used[take]=True
            nearest=min((n for n in links[qi]["neighbors"] if training[n["profile"]]["domain"]=="stu"),key=lambda n:n["distance"])
            material=training[nearest["profile"]]
            fp={name:s[take]>=thresholds[name] for name,s in scores.items()}
            rows.append(dict(parent=parent["parent"],fragment=episode["id"],index=index,frame=row["frame"],
                points=len(take),AP_mass=query["represented_AP_loss"],
                train_index=material["index"],train_profile=nearest["profile"],train_BCE=material["mean_BCE"],
                train_visits=material["visits"],distance_blocks=nearest["blocks"],
                FP={name:int(v.sum()) for name,v in fp.items()},
                input_only_FP=int((fp["input"]&~fp["context_post"]).sum()),
                fused_only_FP=int((~fp["input"]&fp["context_post"]).sum())))
    if not np.array_equal(used,focus_mask):raise ValueError("normal case summary omits focused points")
    for parent,mass in probe["focused_normal_AP_mass"].items():
        if abs(sum(r["AP_mass"] for r in rows if r["parent"]==parent)-mass)>1e-10:raise ValueError("normal AP scope differs")
    reference=curve_summary(score_curve(cached,labels))
    difference=np.abs(cached-scores["C"])
    return dict(normal_cases=rows,thresholds=thresholds,
        normal_case_scope="All sixteen selected peak patches, with each model's threshold at the same target recall on the sampled diagnostic subset. AP_mass is their fixed C global loss allocation, not a gain from a readout or an estimate for the whole episode.",
        numerical_check=dict(points=len(cached),changed_scores=int((difference>0).sum()),max_absolute_score_change=float(difference.max()),
            mean_absolute_score_change=float(difference.mean()),cached_C_subset_AP=reference["AP"],
            fresh_C_subset_AP=probe["metrics"]["val"]["models"]["C"]["AP"],
            interpretation="Compare the same C and raw point identities. Agreement on this subset does not establish universal deterministic inference. The complete AP ledger continues to use the original official predictions."))


def precision(output,workers,limit=None):
    """Isolate rotary-position arithmetic and compare every exposed feature element."""
    import torch
    from torch.utils.data import DataLoader
    from .evaluate import PreparedScans,autocast,load_model
    from .model import GRID_SIZE,to_device
    from .diagnose import score_curve,curve_summary
    from .attribute import loss_weights
    disk_check(3_000_000_000)
    ledger=json.loads((output/"ledger.json").read_text())
    manifest=load_manifest("assets/val.json","val")
    offsets={r["index"]:r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]}
    indices=sorted(offsets)
    probes={}
    for case in ledger["objects"]:
        failure=max(case["observations"],key=lambda r:r["AP_loss"])
        success=min(case["observations"],key=lambda r:r["AP_loss"]/r["count"])
        for role,row in (("failure",failure),("reference",success)):
            probes.setdefault(row["index"],[]).append((case["id"],role))
    for case in ledger["surfaces"][:2]:
        worst=max(case["observations"],key=lambda r:r[3])
        probes.setdefault(worst[0],[]).append((case["id"],"failure"))
    if limit is not None:
        priority=list(probes)
        priority=priority[:4]+priority[-2:]+priority[4:-2]
        indices=list(dict.fromkeys(priority))[:limit]
    cases=ledger["objects"]+ledger["surfaces"]
    case_number={r["id"]:i for i,r in enumerate(cases)}
    report=dict(checkpoint=str(BEST_C),checkpoint_sha256=file_sha256(BEST_C),resources=runtime_snapshot(),
        model_source=file_sha256("vendor/litept/pointrope.py"),manifest=manifest["sha256"],indices=indices,
        complete_validation=limit is None,
        intervention=dict(fixed="C weights, full raw scans, labels, point order, voxel membership, neighborhoods, all other mixed-precision operations",
            changed="Only PointROPE arithmetic executes outside autocast; clear both arms' rotary lookup caches before each scan",
            controls="Original arithmetic, repeated original pass on the first scan, corrected rotary arithmetic; no optimizer or training updates",
            supports="Identical pre-rotary features followed by changed rotary coordinates and downstream features isolates a computational information error. Only measured full-ranking improvement supports it as a remediable factor for a case.",
            excludes="Any observed paired difference cannot be caused by adding training material or training updates; these causes may still contribute to remaining errors.",
            limits="An inference repair does not identify every historical learning cause or establish benefits after retraining. Opposite AP marginal views must not be added."),
        feature_scope="Every scalar of point detail, embedding, five encoders, decoder, every attention QKV and rotated Q/K, initial query, six context tokens, two fused states, point state, and hidden scoring state. Shared voxel vectors are compared once and mapped to every original point. Other temporary arithmetic tensors are not claimed as stored or inspected.")
    if report["checkpoint_sha256"]!=ledger["checkpoint_sha256"]:raise ValueError("C differs from ledger")
    write_json(output/"precision.json",report)
    count=sum(offsets[i]["stop"]-offsets[i]["start"] for i in indices)
    before=np.lib.format.open_memmap(output/"precision_base.npy",mode="w+",dtype=np.float32,shape=(count,))
    after=np.lib.format.open_memmap(output/"precision_fixed.npy",mode="w+",dtype=np.float32,shape=(count,))
    first_file=np.lib.format.open_memmap(output/"precision_stages.npy",mode="w+",dtype=np.uint8,shape=(count,))
    point_cases=np.empty(count,np.int32)
    labels=np.empty(count,np.int8)
    device=torch.device("cuda")
    model,saved_model=load_model(BEST_C,device)
    torch.cuda.reset_peak_memory_stats()
    corrected=False
    captures,levels,attention,inverse_maps={},{},{},[]
    contexts={}
    def emit(name,value,level):
        captures.setdefault(name,[]).append(value.detach().clone())
        levels[name]=level
    def point_hook(name):
        def receive(module,inputs,result):emit(name,result,"point")
        return receive
    def sparse_hook(name,level):
        def receive(module,inputs,result):emit(name,result.feat,level)
        return receive
    def conditional_input(module,inputs):
        inverse_maps.extend(inputs[2])
    def tensor_input(name):
        def receive(module,inputs):emit(name,inputs[0].flatten(1),0)
        return receive
    handles=[model.detail.register_forward_hook(point_hook("point_detail")),
        model.backbone.embedding.register_forward_hook(sparse_hook("embedding",0)),
        model.backbone.dec.register_forward_hook(sparse_hook("decoder",0)),
        model.conditional.register_forward_pre_hook(conditional_input),
        model.conditional.layers[0]["query"].register_forward_pre_hook(tensor_input("initial_query")),
        model.conditional.layers[0]["key"].register_forward_pre_hook(tensor_input("context_tokens")),
        model.head.register_forward_pre_hook(lambda module,inputs:emit("point_state",inputs[0],"point")),
        model.head[2].register_forward_hook(point_hook("score_hidden"))]
    for i,encoder in enumerate(model.backbone.enc):handles.append(encoder.register_forward_hook(sparse_hook(f"encoder_{i}",i)))
    for i,layer in enumerate(model.conditional.layers):
        handles.append(layer["final_norm"].register_forward_hook(
            lambda module,inputs,result,n=f"fusion_{i}":emit(n,result,0)))
    for name,module in model.named_modules():
        if type(module).__name__!="PointROPEAttention":continue
        def attention_input(module,inputs,n=name):
            point=inputs[0]
            _,unpad,_=point.get_padding_and_inverse(module.patch_size)
            attention[n]=dict(level=round(np.log2(point.grid_size/GRID_SIZE)),
                inverse=unpad[point.serialized_inverse[module.order_index]],calls=0)
        def qkv_output(module,inputs,result,n=name):emit(n+".qkv",result,attention[n]["level"])
        def rotary_input(module,inputs,n=name):
            # Explicit arms also reproduce the original arithmetic after a call-site repair.
            context=torch.autocast("cuda",dtype=torch.bfloat16,enabled=not corrected)
            context.__enter__()
            contexts[n]=context
        def rotary_output(module,inputs,result,n=name):
            if n in contexts:contexts.pop(n).__exit__(None,None,None)
            if result is None:return
            info=attention[n]
            # Undo attention serialization/padding before tracing original points.
            value=result[0].transpose(0,1).reshape(result.shape[2],-1)[info["inverse"]]
            emit(n+(".rotary_q" if info["calls"]==0 else ".rotary_k"),value,info["level"])
            info["calls"]+=1
        handles.extend((module.register_forward_pre_hook(attention_input),module.qkv.register_forward_hook(qkv_output),
            module.rope.register_forward_pre_hook(rotary_input),
            module.rope.register_forward_hook(rotary_output,always_call=True)))
    rotary=[m for m in model.modules() if type(m).__name__=="PointROPE"]
    stage_names=None
    totals={}
    feature_cases=None
    examples=[]
    rows=[]
    pose_cache,surface_cache={},{}
    cached_scores=np.load(OUTPUT/"lr_val.npy",mmap_mode="r")
    cached_meta=np.load(OUTPUT/"val_points.npy",mmap_mode="r")
    baseline_difference=dict(changed=0,max_absolute=0.,sum_absolute=0.)
    loader=DataLoader(PreparedScans(manifest),batch_size=None,sampler=indices,num_workers=workers,
        prefetch_factor=1,pin_memory=True,generator=torch.Generator().manual_seed(0))
    start=time.perf_counter()
    cursor=0
    with torch.no_grad():
        for number,sample in enumerate(loader,1):
            index=int(sample["index"])
            row=manifest["records"][index]
            target=sample["targets"].numpy()
            valid=target>=0
            offset=offsets[index]
            meta=cached_meta[offset["start"]:offset["stop"]]
            if not np.array_equal(sample["slots"].numpy()[valid],meta["slot"]) or not np.array_equal(target[valid],meta["target"]):
                raise ValueError("precision probe changed official point identities")
            batch=to_device(sample,device)
            states=[]
            maps=[]
            predictions=[]
            for arm in (("original","repeat","corrected") if number==1 else ("original","corrected")):
                corrected=arm=="corrected"
                captures.clear();levels.clear();attention.clear();inverse_maps.clear()
                for module in rotary:module.cache.clear()
                with autocast(device):prediction=model(batch)
                current={k:torch.cat(v) for k,v in captures.items()}
                if not torch.isfinite(prediction).all():raise ValueError("nonfinite paired prediction")
                if arm=="repeat":
                    difference=np.abs(prediction.cpu().numpy()-predictions[0])
                    report["repeat_control"]=dict(index=index,changed=int((difference!=0).sum()),
                        max_absolute=float(difference.max()),feature_changed_values={
                            k:int((v!=states[0][k]).sum()) for k,v in current.items()})
                    if any(not torch.equal(a,b) for a,b in zip(maps[0],inverse_maps)):
                        raise ValueError("repeat changed voxel membership")
                    del current
                    continue
                predictions.append(prediction.cpu().numpy())
                states.append(current)
                maps.append([v.clone() for v in inverse_maps])
            if any(not torch.equal(a,b) for a,b in zip(*maps)):raise ValueError("numerical intervention changed voxel membership")
            if stage_names is None:
                stage_names=list(states[0])
                if len(stage_names)>=255:raise ValueError("too many traced stages")
                feature_cases=np.zeros((len(cases),len(stage_names)+1),np.int64)
                totals={k:dict(values=0,changed=0,sum_absolute=0.,max_absolute=0.) for k in stage_names}
            if list(states[1])!=stage_names:raise ValueError("feature identities differ between arms")
            first=torch.zeros(len(target),dtype=torch.uint8,device=device)
            raw_points=np.flatnonzero(valid)
            ids=np.full(len(meta),-1,np.int32)
            positive=meta["target"]==1
            for instance in np.unique(meta["instance"][positive]):
                ids[positive&(meta["instance"]==instance)]=case_number[f"P{row['sequence']}:{instance}"]
            sequence=row["sequence"]
            if sequence not in pose_cache:
                pose_cache[sequence]=poses_for(Path(row["scan"]).parents[1])
                with np.load(output/f"surface_{sequence}.npz") as surface:surface_cache[sequence]={k:surface[k] for k in ("keys","component")}
            normal=np.flatnonzero(~positive)
            pose=pose_cache[sequence][row["frame"]]
            world=sample["xyzi"].numpy()[raw_points[normal],:3].astype(float)@pose[:3,:3].T+pose[:3,3]
            surface=surface_cache[sequence]
            components=surface["component"][np.searchsorted(surface["keys"],cell_keys(world,meta["semantic"][normal]))]
            translation=np.array([case_number[f"N{sequence}:{j}"] for j in range(int(surface["component"].max())+1)])
            ids[normal]=translation[components]
            if (ids<0).any():raise ValueError("unassigned official point")
            chosen=[]
            for case,role in probes.get(index,[]):
                if case not in ("P125:1","P141:4",*PARENTS):continue
                positions=np.flatnonzero(ids==case_number[case])
                if not len(positions):continue
                score=predictions[0][raw_points[positions]]
                chosen_index=np.argmin(score) if (case.startswith("P")== (role=="failure")) else np.argmax(score)
                p=int(raw_points[positions[chosen_index]])
                chosen.append((p,dict(case=case,role=role,index=index,frame=row["frame"],slot=int(sample["slots"][p]),
                    original_score=float(predictions[0][p]),corrected_score=float(predictions[1][p]),features={})))
            for si,name in enumerate(stage_names,1):
                a,b=states[0][name],states[1][name]
                if a.shape!=b.shape:raise ValueError("feature shape changed")
                delta=(a.float()-b.float()).abs()
                per_row=(delta!=0).any(1)
                mapping=None if levels[name]=="point" else maps[0][levels[name]][batch["inverse"]]
                changed=per_row if mapping is None else per_row[mapping]
                first[(first==0)&changed]=si
                stat=totals[name]
                stat["values"]+=delta.numel()
                stat["changed"]+=int((delta!=0).sum())
                stat["sum_absolute"]+=float(delta.sum(dtype=torch.float64))
                stat["max_absolute"]=max(stat["max_absolute"],float(delta.max()))
                for p,example in chosen:
                    at=p if mapping is None else int(mapping[p])
                    example["features"][name]=dict(original=a[at].float().cpu().tolist(),corrected=b[at].float().cpu().tolist())
            stop=cursor+len(meta)
            before[cursor:stop]=predictions[0][valid]
            after[cursor:stop]=predictions[1][valid]
            first_file[cursor:stop]=first.cpu().numpy()[valid]
            labels[cursor:stop]=meta["target"]
            point_cases[cursor:stop]=ids
            np.add.at(feature_cases,(ids,first_file[cursor:stop]),1)
            difference=np.abs(before[cursor:stop]-cached_scores[offset["start"]:offset["stop"]])
            baseline_difference["changed"]+=int((difference>0).sum())
            baseline_difference["sum_absolute"]+=float(difference.sum(dtype=np.float64))
            baseline_difference["max_absolute"]=max(baseline_difference["max_absolute"],float(difference.max()))
            examples.extend(e for _,e in chosen)
            rows.append(dict(index=index,sequence=sequence,frame=row["frame"],start=cursor,stop=stop))
            cursor=stop
            del states,current,maps,prediction,delta,a,b,batch,captures,first
            captures={}
            if number%50==0 or number==1:print(f"precision pair {number}/{len(indices)}; {time.perf_counter()-start:.1f}s",flush=True)
    for handle in handles:handle.remove()
    if cursor!=count:raise ValueError("incomplete paired point stream")
    report["parameters_unchanged"]=all(torch.equal(v.detach().cpu(),saved_model["model"][k]) for k,v in model.state_dict().items())
    if not report["parameters_unchanged"]:raise ValueError("C changed during diagnostic")
    before.flush();after.flush();first_file.flush()
    report.update(forward_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        point_count=count,offsets=rows,stages=stage_names,feature_comparison=totals,baseline_difference=baseline_difference)
    del model,saved_model
    torch.cuda.empty_cache()
    curves={name:score_curve(scores,labels) for name,scores in (("original",before),("corrected",after))}
    report["metrics"]={k:curve_summary(v) for k,v in curves.items()}
    counts=np.bincount(point_cases,minlength=len(cases))
    losses,errors={},{}
    for name,scores in (("original",before),("corrected",after)):
        curve=curves[name]
        _,ploss,nloss=loss_weights(curve["positive"],curve["negative"])
        losses[name]=np.zeros(len(cases),np.float64)
        errors[name]=np.zeros((len(cases),len(curve["operating"])),np.int64)
        # One streamed point pass replaces a full-data search for every case.
        for begin in range(0,count,1_000_000):
            end=min(begin+1_000_000,count)
            ids=point_cases[begin:end]
            score=np.asarray(scores[begin:end])
            positive=labels[begin:end]==1
            rank=np.searchsorted(curve["values"],score)
            weights=np.where(positive,ploss[rank],nloss[rank])
            losses[name]+=np.bincount(ids,weights=weights,minlength=len(cases))
            for j,op in enumerate(curve["operating"]):
                wrong=np.where(positive,score<op["threshold"],score>=op["threshold"])
                errors[name][:,j]+=np.bincount(ids[wrong],minlength=len(cases))
        expected=100-curve["AP"]
        for selected in (slice(0,len(ledger["objects"])),slice(len(ledger["objects"]),None)):
            if not np.isclose(losses[name][selected].sum(),expected,atol=1e-8,rtol=0):
                raise ValueError("case AP attribution does not sum to the measured gap")
    case_rows=[]
    for ci,case in enumerate(cases):
        if not counts[ci]:continue
        entry=dict(case=case["id"],points=int(counts[ci]),original_full_AP_loss=case["AP_loss"],
            first_feature_change_counts=feature_cases[ci].tolist(),arms={})
        for name in curves:
            curve=curves[name]
            operating={str(op["target"]):int(errors[name][ci,j]) for j,op in enumerate(curve["operating"])}
            entry["arms"][name]=dict(AP_loss=float(losses[name][ci]),errors=operating)
        entry["loss_reduction"]=entry["arms"]["original"]["AP_loss"]-entry["arms"]["corrected"]["AP_loss"]
        case_rows.append(entry)
    report["cases"]=case_rows
    report["seconds"]=time.perf_counter()-start
    write_json(output/"precision.json",report,indent=None)
    write_json(output/"feature_values.json",dict(checkpoint_sha256=report["checkpoint_sha256"],examples=examples,
        scope="Complete vectors for selected failure/reference points; every exposed scalar for all input points was compared in the streamed pass. Values diagnose a paired numerical intervention, not a learned-feature semantic label."),indent=None)
    print(report["metrics"],flush=True)


def _geometry_init():
    global _geometry_sources
    import torch
    torch.set_num_threads(1)
    _geometry_sources={name:Scans(load_manifest(path,kind),cache_size=2) for name,path,kind in
        (("train","results/data/native/train.json","train"),("val","assets/val.json","val"))}


def _geometry_scan(task):
    """Use exact multiplication by 20 as an independent 5 cm cell reference."""
    from .model import voxelize
    split,index=task
    sample=_geometry_sources[split][index]
    xyz=sample["xyzi"]
    target=sample["targets"]
    batch=voxelize(xyz)
    inv,order,ptr,grid,mean,offset=[batch[k].numpy() for k in
        ("inverse","order","pointer","grid","voxel_xyzi","offset")]
    raw=np.floor(xyz[:,:3].astype(np.float64)*20).astype(np.int64)
    shift=(raw.min(0)//16)*16
    mismatch=np.any(grid[inv]+shift!=raw,axis=1)
    counts=np.bincount(inv,minlength=len(grid))
    reference=np.stack([np.bincount(inv,weights=xyz[:,j],minlength=len(grid))/counts for j in range(4)],1).astype(np.float32)
    expected_offset=(xyz[:,:3].astype(np.float64)*20-raw-.5).astype(np.float32)
    normal=np.bincount(inv[target==0],minlength=len(grid))
    positive=np.bincount(inv[target==1],minlength=len(grid))
    mixed=(normal>0)&(positive>0)
    take=mixed[inv]&(target>=0)
    conflict=np.zeros(len(xyz),bool)
    if take.any():
        _,at=np.unique(xyz[take],axis=0,return_inverse=True)
        p=np.bincount(at[target[take]==1],minlength=at.max()+1)
        n=np.bincount(at[target[take]==0],minlength=at.max()+1)
        conflict[take]=((p>0)&(n>0))[at]
    row=_geometry_sources[split].records[index]
    result=dict(split=split,index=index,group=row.get("group","STU_validation"),points=len(xyz),voxels=len(grid),
        grid_mismatch=int(mismatch.sum()),invalid_point_order=int(not np.array_equal(np.sort(order),np.arange(len(xyz)))),
        segment_mismatch=int(not np.array_equal(inv[order],np.repeat(np.arange(len(grid)),np.diff(ptr)))),
        mean_max_error=float(np.abs(mean-reference).max()),offset_max_error=float(np.abs(offset-expected_offset).max()),
        offset_max_absolute=float(np.abs(offset).max()),max_point_centroid_distance=float(np.linalg.norm(xyz[:,:3]-mean[inv,:3],axis=1).max()),
        hierarchy_mismatch=sum(int(np.any((grid[inv]//factor)!=(raw//factor-shift//factor),axis=1).sum()) for factor in (2,4,8,16)),
        naive_fp32_grid_differences=int(np.any(np.floor(xyz[:,:3]/np.float32(.05)).astype(np.int64)!=raw,axis=1).sum()),
        mixed_voxels=int(mixed.sum()),mixed_normal_points=int(normal[mixed].sum()),mixed_anomaly_points=int(positive[mixed].sum()),
        identical_input_normal=int((conflict&(target==0)).sum()),identical_input_anomaly=int((conflict&(target==1)).sum()))
    if result["grid_mismatch"] or result["invalid_point_order"] or result["segment_mismatch"] or result["hierarchy_mismatch"]:
        raise ValueError(f"incorrect point membership: {result}")
    if result["mean_max_error"]>1e-6 or result["offset_max_error"]>2e-7:
        raise ValueError(f"voxel arithmetic exceeds its FP32 representation tolerance: {result}")
    return result


def model_check(output,workers):
    """Check cell membership on both complete pools, then trace actual model indices."""
    import torch
    from .evaluate import autocast,load_model
    from .model import to_device
    from vendor.litept.model import PointROPEAttention,GridPooling,GridUnpooling
    manifest={"train":load_manifest("results/data/native/train.json","train"),"val":load_manifest("assets/val.json","val")}
    selected={"train":list(range(len(manifest["train"]["records"]))),
        "val":[i for i,r in enumerate(manifest["val"]["records"]) if r["eligible"]]}
    report=dict(checkpoint_sha256=file_sha256(BEST_C),sources={k:v["sha256"] for k,v in manifest.items()},
        resources=runtime_snapshot(),scope="All current training scans and eligible official validation scans for CPU geometry; explicitly selected full scans for GPU layer semantics. No input, label, grid size, weight or training change.",
        tolerances=dict(voxel_mean_absolute=1e-6,normalized_offset_absolute=2e-7,pooling_coordinate_absolute=1e-5,pooled_detail_absolute=2e-5))
    disk_check(100_000_000)
    start=time.perf_counter()
    tasks=[(name,i) for name,indices in selected.items() for i in indices]
    with ProcessPoolExecutor(max_workers=workers,initializer=_geometry_init) as executor:
        rows=[]
        for i,row in enumerate(executor.map(_geometry_scan,tasks,chunksize=8),1):
            rows.append(row)
            if i%500==0:print(f"geometry {i}/{len(tasks)}; {time.perf_counter()-start:.1f}s",flush=True)
    report["geometry_seconds"]=time.perf_counter()-start
    report["geometry_frames"]=rows
    report["geometry_groups"]={}
    maximum=("mean_max_error","offset_max_error","offset_max_absolute","max_point_centroid_distance")
    sums=("points","voxels","grid_mismatch","invalid_point_order","segment_mismatch","hierarchy_mismatch",
        "naive_fp32_grid_differences","mixed_voxels","mixed_normal_points","mixed_anomaly_points","identical_input_normal","identical_input_anomaly")
    for group in sorted({r["group"] for r in rows}):
        group_rows=[r for r in rows if r["group"]==group]
        report["geometry_groups"][group]=dict(scans=len(group_rows),**{k:sum(r[k] for r in group_rows) for k in sums},
            **{k:max(r[k] for r in group_rows) for k in maximum})
    write_json(output/"model_check.json",report,indent=None)
    chosen={"val":set(_even(selected["val"],16)),"train":set()}
    examples=json.loads((output/"feature_values.json").read_text())["examples"]
    chosen["val"].update(e["index"] for e in examples)
    for group in sorted({r["group"] for r in manifest["train"]["records"]}):
        chosen["train"].update(_even([i for i,r in enumerate(manifest["train"]["records"]) if r["group"]==group],4))
    # Target the largest label-sharing case before seeing its internal model values.
    for split in chosen:
        chosen[split].add(max((r for r in rows if r["split"]==split),key=lambda r:r["mixed_anomaly_points"])["index"])
    device=torch.device("cuda")
    model,saved=load_model(BEST_C,device)
    sources={k:Scans(v,cache_size=2) for k,v in manifest.items()}
    torch.cuda.reset_peak_memory_stats()
    frame,projection,geometry,ancestry,details,handles={},{},{},None,[],[]
    def check_close(name,a,b,tolerance=0.):
        error=float((a.double()-b.double()).abs().max()) if a.numel() else 0.
        frame[name]=max(frame.get(name,0.),error)
        if error>tolerance:raise ValueError(f"model semantic mismatch {name}: {error}")
    def pooling_input(module,inputs,n):
        parent=inputs[0]
        geometry[n]=(parent.grid_coord.clone(),parent.coord.clone())
    def projected(module,inputs,result,n):projection[n]=result.detach()
    def pooled_before_norm(module,inputs,n):
        point=inputs[0]
        parent_grid,parent_coord=geometry[n]
        inv=point.pooling_inverse
        check_close("pool_membership",point.grid_coord[inv],parent_grid//2)
        counts=torch.bincount(inv,minlength=len(point.coord))
        expected=torch.zeros_like(point.coord,dtype=torch.float64).index_add_(0,inv,parent_coord.double())/counts[:,None]
        check_close("pool_coordinate",point.coord,expected,1e-5)
        value=projection.pop(n)
        expected=value.new_full((len(point.coord),value.shape[1]),-torch.inf)
        expected.scatter_reduce_(0,inv[:,None].expand_as(value),value,reduce="amax",include_self=True)
        check_close("pool_max",point.feat,expected)
    def unpool_input(module,inputs,n):geometry[n]=inputs[0].pooling_parent.grid_coord.clone()
    def unpool_output(module,inputs,result,n):check_close("unpool_parent_order",result.grid_coord,geometry[n])
    for name,module in model.named_modules():
        if isinstance(module,GridPooling):
            handles.extend((module.register_forward_pre_hook(lambda m,a,n=name:pooling_input(m,a,n)),
                module.proj.register_forward_hook(lambda m,a,r,n=name:projected(m,a,r,n)),
                module.norm.register_forward_pre_hook(lambda m,a,n=name:pooled_before_norm(m,a,n))))
        elif isinstance(module,GridUnpooling):
            handles.extend((module.register_forward_pre_hook(lambda m,a,n=name:unpool_input(m,a,n)),
                module.register_forward_hook(lambda m,a,r,n=name:unpool_output(m,a,r,n))))
        elif isinstance(module,PointROPEAttention):
            def verify_attention(module,inputs):
                point=inputs[0]
                pad,unpad,cu=point.get_padding_and_inverse(module.patch_size)
                order=point.serialized_order[module.order_index][pad]
                inverse=unpad[point.serialized_inverse[module.order_index]]
                check_close("attention_restore",order[inverse],torch.arange(len(point.feat),device=device))
                if int(cu[0])!=0 or int(cu[-1])!=len(order) or int(torch.diff(cu).max())>module.patch_size:
                    raise ValueError("invalid attention segment boundaries")
                frame["attention_calls"]=frame.get("attention_calls",0)+1
            handles.append(module.register_forward_pre_hook(verify_attention))
    handles.append(model.detail.register_forward_hook(lambda m,a,r:details.append(r.detach())))
    def encoder_output(module,inputs,result,level):
        nonlocal ancestry
        if level:ancestry=result.pooling_inverse[ancestry]
        geometry[f"level{level}"]=(result.grid_coord.clone(),result.coord.clone(),ancestry.clone())
    for level,encoder in enumerate(model.backbone.enc):
        handles.append(encoder.register_forward_hook(lambda m,a,r,level=level:encoder_output(m,a,r,level)))
    def conditional_input(module,inputs):
        pooled,xyz,indices,coords,features=inputs
        root_grid=geometry["level0"][0]
        for level,index in enumerate(indices):
            at=0 if level==5 else level
            grid,coordinate,expected=geometry[f"level{at}"]
            check_close("conditional_index",index,expected)
            check_close("conditional_cell",grid[index],root_grid//(2**at))
            check_close("conditional_coordinate",coords[level],coordinate)
        if not torch.equal(coords[5][indices[5]],xyz):raise ValueError("decoder token is not at its original root voxel")
        value=torch.cat(details).double()
        inv=current_batch["inverse"]
        counts=torch.bincount(inv,minlength=len(xyz))
        mean=value.new_zeros((len(xyz),64)).index_add_(0,inv,value)/counts[:,None]
        maximum=value.new_full((len(xyz),64),-torch.inf)
        maximum.scatter_reduce_(0,inv[:,None].expand_as(value),value,reduce="amax",include_self=True)
        check_close("detail_mean",pooled[:,:64],mean,2e-5)
        check_close("detail_max",pooled[:,64:],maximum)
    handles.append(model.conditional.register_forward_pre_hook(conditional_input))
    report["gpu_frames"]=[]
    with torch.no_grad():
        for split in ("train","val"):
            for index in sorted(chosen[split]):
                sample=sources[split][index]
                current_batch=to_device(prepare_scan(sample),device)
                frame=dict(split=split,index=index,points=len(sample["xyzi"]))
                details.clear();geometry.clear();projection.clear()
                ancestry=torch.arange(len(current_batch["grid"]),device=device)
                with autocast(device):score=model(current_batch)
                if len(score)!=len(sample["slots"]) or not torch.isfinite(score).all():raise ValueError("point output identity differs")
                frame["normal_points"]=int((sample["targets"]==0).sum())
                frame["anomaly_points"]=int((sample["targets"]==1).sum())
                report["gpu_frames"].append(frame)
    for handle in handles:handle.remove()
    report["parameters_unchanged"]=all(torch.equal(v.detach().cpu(),saved["model"][k]) for k,v in model.state_dict().items())
    if not report["parameters_unchanged"]:raise ValueError("model checking changed C")
    report["seconds"]=time.perf_counter()-start
    report["peak_cuda_bytes"]=torch.cuda.max_memory_allocated()
    report["interpretation"]="Exact membership and index checks diagnose implementation. Mixed-label cells quantify a modeling limitation, not a proven AP cause: point details and point labels remain separate. Coarse coordinates are equal-child-voxel means by design, not point-count-weighted centroids."
    write_json(output/"model_check.json",report,indent=None)
    print(dict(groups=report["geometry_groups"],gpu_scans=len(report["gpu_frames"]),seconds=report["seconds"]),flush=True)
