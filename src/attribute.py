"""Exhaustive AP accounting with recurrent objects and spatial surface groups.

Loss accounting is exact at the saved score ties. Spatial grouping and material
retrieval are diagnostic descriptions; neither establishes a causal mechanism.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
from scipy.sparse.csgraph import connected_components

from .data import Scans, file_sha256, load_manifest, rigid, write_json
from .diagnose import BEST_C, OUTPUT, write_csv
from .train import disk_check, runtime_snapshot


CELL = .5
RECALL_BANDS = (0., .25, .5, .75, .9, .95, .99, 1.)


def poses_for(directory):
    """Use the same camera-to-LiDAR conjugation as the STU training reader."""
    line = next(s for s in (directory / "calib.txt").read_text().splitlines() if s.startswith("Tr:"))
    transform = np.eye(4)
    transform[:3] = np.asarray(line.split(":", 1)[1].split(), float).reshape(3, 4)
    rigid(transform)
    values = np.loadtxt(directory / "poses.txt").reshape(-1, 3, 4)
    poses = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
    poses[:, :3] = values
    poses = np.linalg.inv(transform) @ poses @ transform
    for pose in poses:
        rigid(pose)
    return poses


def cell_keys(xyz, semantic):
    grid = np.floor(xyz / CELL).astype(np.int64) + 32768
    if grid.min() < 0 or grid.max() >= 65536:
        raise ValueError("world coordinates exceed loss-ledger cell encoding")
    x, y, z = grid.astype(np.uint64).T
    return (semantic.astype(np.uint64) << 48) | (x << 32) | (y << 16) | z


def surface_groups(keys, orientation=None):
    """Join nearby cells only within one raw class and geometric surface type."""
    rows, columns = [], []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                delta = (x << 32) + (y << 16) + z
                if delta <= 0:
                    continue
                other = keys + np.uint64(delta)
                at = np.searchsorted(keys, other)
                valid = at < len(keys)
                found = np.flatnonzero(valid)
                found = found[keys[at[found]] == other[found]]
                # Prevent packed-axis overflow from joining different semantics.
                found = found[(keys[found] >> 48) == (keys[at[found]] >> 48)]
                if orientation is not None:
                    found = found[orientation[found] == orientation[at[found]]]
                rows.append(found)
                columns.append(at[found])
    a, b = np.concatenate(rows), np.concatenate(columns)
    graph = coo_matrix((np.ones(len(a), np.uint8), (a, b)), shape=(len(keys), len(keys))).tocsr()
    return connected_components(graph, directed=False)[1].astype(np.int32)


def surface_orientation(centers):
    """Separate horizontal, vertical, oblique and nonplanar surface candidates."""
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    distances, neighbors = tree.query(centers, k=min(16, len(centers)), workers=1)
    local = centers[neighbors] - centers[neighbors].mean(1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", local, local) / local.shape[1]
    values, vectors = np.linalg.eigh(covariance)
    vertical = np.abs(vectors[:, 2, 0])
    planar = (values[:, 1] - values[:, 0]) / np.maximum(values[:, 2], 1e-12) >= .3
    supported = (distances[:, -1] <= 1.5) & (values[:, 1] > 1e-5)
    return np.where(~(planar & supported), 3, np.where(vertical >= np.cos(np.pi/6), 0,
                    np.where(vertical <= .5, 1, 2))).astype(np.int8)


def rank_data():
    rows = list(csv.DictReader((OUTPUT / "lr_pr.csv").open(encoding="utf-8-sig")))
    values = np.array([float(r["threshold"]) for r in rows])[::-1]
    tp, fp = (np.array([int(r[k]) for r in rows]) for k in ("tp", "fp"))
    positive, negative = np.diff(np.r_[0, tp])[::-1], np.diff(np.r_[0, fp])[::-1]
    denominator, positive_loss, normal_loss = loss_weights(positive, negative)
    return values, positive, negative, denominator, positive_loss, normal_loss


def loss_weights(positive, negative):
    """Weights in ascending score order, with complete ties including the point."""
    tp = np.cumsum(positive[::-1])[::-1]
    fp = np.cumsum(negative[::-1])[::-1]
    denominator = tp + fp
    return denominator, 100 / positive.sum() * fp / denominator, np.cumsum(positive / denominator) * 100 / positive.sum()


def recall_loss_weights(positive, negative):
    """Split score ties at recall-band boundaries without reordering points."""
    _, weights, _ = loss_weights(positive, negative)
    end = np.cumsum(positive[::-1])[::-1] / positive.sum()
    begin = end - positive / positive.sum()
    return np.stack([weights*np.divide(np.maximum(0.,np.minimum(end,hi)-np.maximum(begin,lo)),
        end-begin,out=np.zeros_like(end),where=end>begin) for lo,hi in zip(RECALL_BANDS[:-1],RECALL_BANDS[1:])])


def _sequence(task):
    sequence, records, offsets, destination = task
    start_time = time.perf_counter()
    values, positive, negative, denominator, ploss, nloss = rank_data()
    global_neg_above = np.cumsum(negative[::-1])[::-1]
    score_file = np.load(OUTPUT / "lr_val.npy", mmap_mode="r")
    meta_file = np.load(OUTPUT / "val_points.npy", mmap_mode="r")
    directory = Path(records[0]["scan"]).parents[1]
    poses = poses_for(directory)

    def read(record, offset):
        meta = meta_file[offset["start"]:offset["stop"]]
        score = score_file[offset["start"]:offset["stop"]]
        xyzi = np.fromfile(record["scan"], dtype="<f4").reshape(-1, 4)[meta["slot"]]
        pose = poses[record["frame"]]
        world = xyzi[:, :3].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
        normal = meta["target"] == 0
        return meta, score, xyzi, world, normal

    key_parts, centers_parts, count_parts = [], [], []
    for record, offset in zip(records, offsets):
        meta, _, _, world, normal = read(record, offset)
        unique, inverse, frequency = np.unique(cell_keys(world[normal], meta["semantic"][normal]), return_inverse=True, return_counts=True)
        key_parts.append(unique)
        centers_parts.append(np.stack([np.bincount(inverse, weights=world[normal, j]) for j in range(3)], axis=1))
        count_parts.append(frequency)
    keys, inverse = np.unique(np.concatenate(key_parts), return_inverse=True)
    mass = np.bincount(inverse, weights=np.concatenate(count_parts))
    totals = np.concatenate(centers_parts)
    centers = np.stack([np.bincount(inverse, weights=totals[:, j]) / mass for j in range(3)], axis=1)
    del key_parts, centers_parts, count_parts, totals, inverse
    orientation = surface_orientation(centers)
    component = surface_groups(keys, orientation)
    count, width = int(component.max()) + 1, len(values)
    np.savez_compressed(Path(destination) / f"surface_{sequence}.npz", keys=keys, component=component, orientation=orientation,
                        cell=CELL, pose_sha256=file_sha256(directory / "poses.txt"),
                        calibration_sha256=file_sha256(directory / "calib.txt"))
    # Sparse histograms retain every normal point without writing duplicate clouds.
    histograms, observations, anomaly = [], defaultdict(list), defaultdict(list)
    for record, offset in zip(records, offsets):
        meta, score, xyzi, world, normal = read(record, offset)
        rank = np.searchsorted(values, score)
        if not np.array_equal(values[rank], score):
            raise ValueError("saved point scores do not match exact PR ties")
        key = cell_keys(world[normal], meta["semantic"][normal])
        group = component[np.searchsorted(keys, key)]
        code, frequency = np.unique(group.astype(np.int64) * width + rank[normal], return_counts=True)
        histograms.append(coo_matrix((frequency, (code // width, code % width)), shape=(count, width)).tocsr())
        sizes = np.bincount(group, minlength=count)
        losses = np.bincount(group, weights=nloss[rank[normal]], minlength=count)
        frame_normal_hist = np.bincount(rank[normal], minlength=width)
        frame_neg_above = np.cumsum(frame_normal_hist[::-1])[::-1]
        for c in np.flatnonzero(sizes):
            observations[int(c)].append([offset["index"], record["frame"], int(sizes[c]), float(losses[c])])
        for instance in np.unique(meta["instance"][~normal]):
            chosen = np.flatnonzero((~normal) & (meta["instance"] == instance))
            ranks, counts = np.unique(rank[chosen], return_counts=True)
            within = frame_neg_above[rank[chosen]] - .5*frame_normal_hist[rank[chosen]]
            across = global_neg_above[rank[chosen]] - .5*negative[rank[chosen]] - within
            anomaly[int(instance)].append(dict(index=offset["index"], frame=record["frame"],
                count=len(chosen), AP_loss=float(ploss[rank[chosen]].sum()),
                ranks=ranks.tolist(), counts=counts.tolist(),
                center_world=np.median(world[chosen], axis=0).tolist(),
                within_scan_rank_error=float((within/normal.sum()).mean()),
                cross_scan_rank_error=float((across/(negative.sum()-normal.sum())).mean()),
                within_scan_AP_loss=float((frame_neg_above[rank[chosen]]/denominator[rank[chosen]]).sum()*100/positive.sum()),
                score_quantiles=np.quantile(score[chosen], [.1, .5, .9]).tolist(),
                range_quantiles=np.quantile(meta["range"][chosen], [.1, .5, .9]).tolist()))
        if len(histograms) == 16:
            histograms = [sum(histograms)]
    histogram = sum(histograms)
    save_npz(Path(destination) / f"ranks_{sequence}.npz", histogram)
    surface = []
    for c in range(count):
        positions = keys[component == c]
        grid = np.stack([(positions >> shift) & 65535 for shift in (32, 16, 0)], axis=1).astype(float) - 32768
        obs = observations[c]
        surface.append(dict(id=f"N{sequence}:{c}", sequence=sequence, component=c,
            semantic=int(positions[0] >> np.uint64(48)), cells=len(positions),
            orientation=int(orientation[np.flatnonzero(component==c)[0]]),
            bounds_world=[(grid.min(0) * CELL).tolist(), ((grid.max(0) + 1) * CELL).tolist()],
            observations=obs, points=sum(r[2] for r in obs), AP_loss=sum(r[3] for r in obs)))
    objects = []
    for instance, obs in sorted(anomaly.items()):
        episodes = []
        for row in obs:
            if not episodes or row["frame"] != episodes[-1][-1] + 1:
                episodes.append([])
            episodes[-1].append(row["frame"])
        objects.append(dict(id=f"P{sequence}:{instance}", sequence=sequence, instance=instance,
            observations=obs, episodes=episodes, points=sum(r["count"] for r in obs),
            AP_loss=sum(r["AP_loss"] for r in obs),
            within_scan_rank_error=sum(r["within_scan_rank_error"]*r["count"] for r in obs)/sum(r["count"] for r in obs),
            cross_scan_rank_error=sum(r["cross_scan_rank_error"]*r["count"] for r in obs)/sum(r["count"] for r in obs),
            within_scan_AP_loss=sum(r["within_scan_AP_loss"] for r in obs),
            center_world_span=np.ptp([r["center_world"] for r in obs], axis=0).tolist()))
    # Check the physical meaning of pose-based linking, not just matrix shape.
    from scipy.spatial import cKDTree
    adjacent = [i for i in range(len(records)-1) if records[i+1]["frame"] == records[i]["frame"]+1]
    registration = []
    for i in np.unique(np.linspace(0,len(adjacent)-1,min(4,len(adjacent))).round().astype(int)):
        j = adjacent[i]
        a = read(records[j], offsets[j])
        b = read(records[j+1], offsets[j+1])
        selected_a = np.flatnonzero(a[4] & (a[0]["range"]>=5) & (a[0]["range"]<30))
        selected_b = b[4] & (b[0]["range"]>=5) & (b[0]["range"]<30)
        chosen = selected_a[np.linspace(0,len(selected_a)-1,min(1024,len(selected_a))).round().astype(int)]
        aligned = cKDTree(b[3][selected_b]).query(a[3][chosen])[0]
        raw = cKDTree(b[2][selected_b,:3]).query(a[2][chosen,:3])[0]
        registration.append(dict(frames=[records[j]["frame"],records[j+1]["frame"]],
            aligned_distance=np.quantile(aligned,[.5,.9]).tolist(),uncompensated_distance=np.quantile(raw,[.5,.9]).tolist()))
    print(f"AP ledger sequence {sequence}: {len(objects)} annotation cases, {count} surface candidates, "
          f"{time.perf_counter()-start_time:.1f}s", flush=True)
    return dict(sequence=sequence, surfaces=surface, objects=objects, registration=registration,
                pose_sha256=file_sha256(directory / "poses.txt"),
                calibration_sha256=file_sha256(directory / "calib.txt"))


def account(output, workers):
    """Partition 100-AP exactly; retain both sides without double counting them."""
    output.mkdir(parents=True, exist_ok=True)
    disk_check(1_000_000_000)
    write_json(output / "ledger_resources.json", runtime_snapshot())
    manifest = load_manifest("assets/val.json", "val")
    source = json.loads((OUTPUT / "lr.json").read_text())
    prediction_checkpoint = Path(source["checkpoint"])
    if source["checkpoint_sha256"] != file_sha256(prediction_checkpoint):
        raise ValueError("C source predictions have changed")
    import torch
    retained = torch.load(BEST_C,map_location="cpu",weights_only=False)["model"]
    predicted = torch.load(prediction_checkpoint,map_location="cpu",weights_only=False)["model"]
    if retained.keys()!=predicted.keys() or any(not torch.equal(retained[k],predicted[k]) for k in retained):
        raise ValueError("retained C parameters differ from the prediction model")
    del retained,predicted
    metadata = json.loads((OUTPUT / "val_offsets.json").read_text())
    if metadata["manifest"]!=manifest["sha256"] or source["splits"]["val"]["manifest_sha256"]!=manifest["sha256"]:
        raise ValueError("C prediction and point identities use different validation manifests")
    offsets = metadata["rows"]
    tasks = []
    for sequence in sorted({r["sequence"] for r in offsets}):
        selected = [r for r in offsets if r["sequence"] == sequence]
        tasks.append((sequence, [manifest["records"][r["index"]] for r in selected], selected, str(output)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        sequences = list(executor.map(_sequence, tasks))
    objects = sorted([r for s in sequences for r in s["objects"]], key=lambda r:-r["AP_loss"])
    surfaces = [r for s in sequences for r in s["surfaces"]]
    values, positive, negative, denominator, ploss, nloss = rank_data()
    object_histogram = np.zeros((len(objects), len(values)), np.int64)
    for i, obj in enumerate(objects):
        for obs in obj["observations"]:
            np.add.at(object_histogram[i], obs["ranks"], obs["counts"])
    if not np.array_equal(object_histogram.sum(0), positive):
        raise ValueError("anomaly cases omit or duplicate evaluated points")
    band_loss = object_histogram @ recall_loss_weights(positive,negative).T
    for row, bands in zip(objects,band_loss):
        row["AP_loss_by_recall"] = bands.tolist()
        if abs(bands.sum()-row["AP_loss"])>1e-7:
            raise ValueError("recall bands omit an object's loss")
    per_normal = np.cumsum(object_histogram / denominator[None, :], axis=1) * 100 / positive.sum()
    matrix, normal_histogram = [], np.zeros(len(values), np.int64)
    for s in sequences:
        histogram = load_npz(output / f"ranks_{s['sequence']}.npz")
        matrix.append(np.asarray(histogram @ per_normal.T))
        normal_histogram += np.asarray(histogram.sum(0)).ravel()
    if not np.array_equal(normal_histogram, negative):
        raise ValueError("normal surfaces omit or duplicate evaluated points")
    matrix = np.concatenate(matrix)
    expected = 100 - source["splits"]["val"]["metrics"]["AP"]
    if not np.allclose(matrix.sum(0), [r["AP_loss"] for r in objects], rtol=0, atol=1e-7):
        raise ValueError("object and normal-surface AP accounting disagree")
    if abs(matrix.sum() - expected) > 1e-7:
        raise ValueError("the complete ledger does not sum to 100-AP")
    for i, row in enumerate(objects):
        order = np.argsort(-matrix[:, i], kind="stable")
        row["normal_drivers"] = [[surfaces[j]["id"], float(matrix[j, i])] for j in order if matrix[j, i] > 0]
    for j, row in enumerate(surfaces):
        row["anomaly_drivers"] = [[objects[i]["id"], float(matrix[j, i])] for i in np.argsort(-matrix[j]) if matrix[j, i] > 0]
    edges = [dict(anomaly=objects[i]["id"], normal=surfaces[j]["id"], AP_loss=float(matrix[j, i]))
             for j, i in zip(*np.nonzero(matrix))]
    write_csv(output / "relations.csv", sorted(edges, key=lambda r:-r["AP_loss"]))
    result = dict(checkpoint=str(BEST_C), checkpoint_sha256=file_sha256(BEST_C),
        prediction_checkpoint=str(prediction_checkpoint), prediction_checkpoint_sha256=source["checkpoint_sha256"],
        retained_parameters_equal_prediction=True,
        validation_manifest=manifest["sha256"], metrics=source["splits"]["val"]["metrics"],
        evaluated_points=int(positive.sum()+negative.sum()), anomaly_points=int(positive.sum()),
        normal_points=int(negative.sum()), AP_loss=expected, assigned_AP_loss=float(matrix.sum()),
        unassigned_AP_loss=0., accounting_residual=float(expected-matrix.sum()), zero_loss_normal_points=int(negative[nloss==0].sum()),
        zero_loss_anomaly_points=int(positive[ploss==0].sum()),
        recall_bands=RECALL_BANDS,AP_loss_by_recall=band_loss.sum(0).tolist(),
        definitions=dict(point_loss="100/P * FP(s)/(TP(s)+FP(s)); every positive and complete score tie retained",
            relation="100/P * sum over positives in object of N_surface(score>=s)/(TP(s)+FP(s))",
            aggregation="Repeated observations are merged into cases, never removed from official point counts",
            rank_error="For each anomalous point: fraction of normal points outranking it, with half credit for ties. Same-scan and other-scan denominators are separate; average over actual anomaly points. Conditional normal distributions differ, so this is not a calibration-cause test.",
            surfaces="Per-sequence 26-connected 0.5 m world cells within one raw semantic and geometric orientation class; candidate surfaces, not certified instances. Orientation: 0 horizontal, 1 vertical, 2 oblique, 3 nonplanar or insufficient support. Raw semantics are not certified physical categories.",
            views="Object and normal-surface tables are two marginal views of the SAME gap; never add their totals",
            causality="Exact accounting is descriptive. A positive loss is not evidence for a particular cause or attainable gain"),
        poses=[{k:s[k] for k in ("sequence","pose_sha256","calibration_sha256","registration")} for s in sequences],
        objects=objects, surfaces=sorted(surfaces,key=lambda r:-r["AP_loss"]))
    write_json(output / "ledger.json", result, indent=None)
    print(f"Complete AP ledger: {len(objects)} object cases, {len(surfaces)} surface candidates, "
          f"gap={matrix.sum():.10f}, residual={expected-matrix.sum():.3g}", flush=True)


def signature(xyzi, targets, chosen):
    """Measured shape, sampling, response and context; no learned score in retrieval."""
    from scipy.spatial import cKDTree
    cloud = np.asarray(xyzi[chosen], np.float64)
    xyz = cloud[:, :3]
    center = np.median(xyz, axis=0)
    centered = xyz - xyz.mean(0)
    eigen, axes = np.linalg.eigh(centered.T @ centered / max(len(xyz), 1))
    eigen = np.maximum(eigen, 0.)
    projected = centered @ axes
    extent = np.quantile(projected, .95, axis=0) - np.quantile(projected, .05, axis=0)
    sampled = xyz[np.linspace(0, len(xyz)-1, min(128, len(xyz))).round().astype(int)]
    distances = np.linalg.norm(sampled[:, None] - sampled[None, :], axis=2)
    pairs = distances[np.triu_indices(len(sampled), 1)]
    pair_quantiles = np.quantile(pairs, [.1, .25, .5, .75, .9]) if len(pairs) else np.zeros(5)
    if len(xyz) >= 4:
        gap, neighbors = cKDTree(xyz).query(sampled, k=min(12, len(xyz)))
        near = xyz[neighbors] - xyz[neighbors].mean(1, keepdims=True)
        local_eigen, local_axes = np.linalg.eigh(np.einsum("nki,nkj->nij", near, near))
        normals = local_axes[:, :, 0]
        # Normal dispersion and local residual distinguish curved/thick observations.
        normal_moment = np.linalg.eigvalsh(normals.T @ normals / len(normals))
        residual = np.quantile(local_eigen[:, 0] / np.maximum(local_eigen.sum(1), 1e-12), [.1, .5, .9])
        spacing = np.quantile(gap[:, 1], [.1, .5, .9])
    else:
        normal_moment, residual, spacing = np.zeros(3), np.zeros(3), np.zeros(3)
    surrounding = np.linalg.norm(xyzi[:, :3] - center, axis=1) <= 2.
    surrounding[chosen] = False
    background = np.asarray(xyzi[surrounding & (targets == 0), :3], np.float64)
    if len(background) >= 3:
        delta = background - background.mean(0)
        bg_eigen, bg_axes = np.linalg.eigh(delta.T @ delta / len(background))
        bg_spread = np.sqrt(np.maximum(bg_eigen, 0.))
        proximity = np.quantile(cKDTree(background).query(sampled)[0], [.1, .5, .9])
        relative_height = center[2] - np.quantile(background[:, 2], .1)
        angle = abs(float(axes[:, 0] @ bg_axes[:, 0]))
    else:
        bg_spread, proximity, relative_height, angle = np.zeros(3), np.zeros(3), 0., 0.
    geometry = np.r_[np.log1p(extent), np.log1p(pair_quantiles), eigen / max(eigen.sum(), 1e-12), normal_moment, residual]
    sampling = np.r_[np.log1p(len(cloud)), np.log1p(np.quantile(np.linalg.norm(xyz, axis=1), [.1, .5, .9])),
                     np.log1p(spacing)]
    response = np.quantile(cloud[:, 3], [.1, .5, .9])
    context = np.r_[np.log1p(len(background)), np.log1p(bg_spread), np.log1p(proximity), relative_height, angle]
    return np.r_[geometry, sampling, response, context].astype(np.float32)


def pair_errors(positive, negative):
    """Conditional opposite-class outranking rates; ties contribute one half."""
    normal_above = np.cumsum(negative[::-1])[::-1] - .5*negative
    anomaly_below = np.cumsum(positive) - .5*positive
    return normal_above/negative.sum(), anomaly_below/positive.sum()


def _profile_init(output, mode, queries, ranking):
    global _profile_output, _profile_mode, _profile_queries, _profile_data, _profile_frames, _profile_rank, _profile_reference
    _profile_output, _profile_mode, _profile_queries = Path(output), mode, queries
    training = json.loads((_profile_output / "train.json").read_text())
    _profile_frames = training["frames"]
    values, positive, negative = ranking
    anomaly_error, normal_error = pair_errors(positive, negative)
    tp, fp = (np.cumsum(x[::-1])[::-1] for x in (positive,negative))
    _profile_reference = values, anomaly_error, normal_error, tp/(tp+fp)
    if mode == "train":
        _profile_data = Scans(load_manifest("results/data/native/train.json", "train"))
        _profile_rank = np.load(_profile_output / "train_scores.npy", mmap_mode="r")
    else:
        _profile_data = load_manifest("assets/val.json", "val")
        _profile_rank = np.load(OUTPUT / "lr_val.npy", mmap_mode="r")


def _profiles(index):
    from .diagnose import validation_arrays
    query = _profile_queries[index]
    if _profile_mode == "train":
        row = _profile_data.records[index]
        sample = _profile_data[index]
        xyzi, targets, slots = [sample[k] for k in ("xyzi", "targets", "slots")]
        frame = _profile_frames[index]
        scores = np.full(len(targets), np.nan, np.float32)
        scores[targets >= 0] = _profile_rank[frame["start"]:frame["stop"]]
        items = []
        for obs in query:
            chosen = np.searchsorted(slots, obs["slots"])
            if not np.array_equal(slots[chosen], obs["slots"]):
                raise ValueError("training material slot identity changed")
            selector = dict(instance=obs["instance"]) if obs["kind"] == "anomaly" else dict(cell=np.floor(xyzi[chosen[0], :3]/.75).astype(int).tolist())
            items.append((chosen, dict(kind=obs["kind"], selector=selector)))
        # Add score-independent normal references throughout the sensor range.
        normal = np.flatnonzero(targets == 0)
        band = np.digitize(np.linalg.norm(xyzi[normal, :3], axis=1), [5, 10, 20, 30])
        grid = np.floor(xyzi[:, :3] / .75).astype(np.int32)
        seen = {tuple(item[1]["selector"]["cell"]) for item in items if item[1]["kind"] == "normal"}
        for b in np.unique(band):
            cells, frequency = np.unique(grid[normal[band == b]], axis=0, return_counts=True)
            cell = cells[int(np.argmax(frequency))]
            if tuple(cell) in seen:
                continue
            seen.add(tuple(cell))
            chosen = np.flatnonzero((targets == 0) & np.all(grid == cell, axis=1))
            items.append((chosen, dict(kind="normal", selector=dict(cell=cell.tolist()))))
        common = dict(index=index, frame=row["frame"], domain="nuscenes" if row.get("source") == "nuscenes" else "stu",
                      group=row["group"], unit=row.get("scene", row.get("world", "206")), visits=frame["visits_before_C"])
    else:
        offset, anomaly_queries, surface_queries = query
        row = _profile_data["records"][index]
        xyzi, targets, scores, meta = validation_arrays(offset, row)
        slots = meta["slot"]
        items = []
        for obs in anomaly_queries:
            chosen = np.flatnonzero((targets == 1) & (meta["instance"] == obs["instance"]))
            items.append((chosen, dict(case=obs["case"], kind="anomaly", selector=dict(instance=obs["instance"]),
                                       represented_AP_loss=obs["AP_loss"])))
        if surface_queries:
            saved = np.load(_profile_output / f"surface_{row['sequence']}.npz", allow_pickle=False)
            pose = poses_for(Path(row["scan"]).parents[1])[row["frame"]]
            normal = np.flatnonzero(targets == 0)
            world = xyzi[normal, :3].astype(float) @ pose[:3, :3].T + pose[:3, 3]
            keys = cell_keys(world, meta["semantic"][normal])
            component = saved["component"][np.searchsorted(saved["keys"], keys)]
            values, _, _, _, _, penalty = rank_data()
            weights = penalty[np.searchsorted(values, scores[normal])]
            grid = np.floor(xyzi[:, :3] / .75).astype(np.int32)
            for obs in surface_queries:
                candidates = np.flatnonzero(component == obs["component"])
                if "world_cell" in obs:
                    picked = candidates[np.all(np.floor(world[candidates]/.75).astype(int)==obs["world_cell"],axis=1)]
                    chosen = normal[picked]
                    mass = float(weights[picked].sum())
                    if len(chosen)!=obs["expected_points"] or abs(mass-obs["expected_AP"])>1e-8:
                        raise ValueError("focused local points differ from their AP fragment")
                    selector = dict(component=obs["component"],world_cell=obs["world_cell"],fragment=obs["fragment"])
                else:
                    cells, inverse = np.unique(grid[normal[candidates]], axis=0, return_inverse=True)
                    losses = np.bincount(inverse, weights=weights[candidates])
                    cell = int(np.argmax(losses))
                    chosen, mass = normal[candidates[inverse == cell]], float(losses[cell])
                    selector = dict(component=obs["component"],cell=cells[cell].tolist())
                items.append((chosen, dict(case=obs["case"], kind="normal", selector=selector,
                    represented_AP_loss=mass, semantic=obs["semantic"])))
        common = dict(index=index, frame=row["frame"], sequence=row["sequence"], domain="stu", group="validation")
    output = []
    sorted_class = [np.sort(scores[targets==label]) for label in (0,1)]
    values, anomaly_error, normal_error, precision = _profile_reference
    for chosen, item in items:
        s = scores[chosen]
        label = item["kind"] == "anomaly"
        rank = np.searchsorted(values,s)
        if not np.array_equal(values[rank],s):
            raise ValueError("material scores are absent from their reference pool")
        other = sorted_class[1-int(label)]
        below = .5*(np.searchsorted(other,s,side="left")+np.searchsorted(other,s,side="right"))
        within = float(((len(other)-below) if label else below).mean()/len(other)) if len(other) else None
        details = dict(**common, **item, points=len(chosen), score_quantiles=np.quantile(s, [.1, .5, .9]).tolist(),
                       mean_BCE=float(np.logaddexp(0., -s if label else s).mean()),
                       reference_pair_error=float((anomaly_error if label else normal_error)[rank].mean()),
                       within_scan_pair_error=within,
                       global_precision_at_score=float(precision[rank].mean()) if label else None,
                       fraction_wrong_at_logit_zero=float(((s < 0) if label else (s >= 0)).mean()))
        output.append((details, signature(xyzi, targets, chosen)))
    return output


def materials(output, workers):
    """Inspect every anomaly observation and a representative of every loss-bearing surface."""
    ledger = json.loads((output / "ledger.json").read_text())
    training = json.loads((output / "train.json").read_text())
    if training["checkpoint_sha256"] != ledger["checkpoint_sha256"]:
        raise ValueError("materials and ledger use different C models")
    if training["train_manifest"]!=load_manifest("results/data/native/train.json","train")["sha256"]:
        raise ValueError("training material identities have changed")
    train_queries = defaultdict(list)
    for obs in training["observations"]:
        train_queries[obs["index"]].append(obs)
    offsets = {r["index"]:r for r in json.loads((OUTPUT / "val_offsets.json").read_text())["rows"]}
    val_queries = {i:(r, [], []) for i, r in offsets.items()}
    for obj in ledger["objects"]:
        for obs in obj["observations"]:
            val_queries[obs["index"]][1].append(dict(case=obj["id"], instance=obj["instance"], AP_loss=obs["AP_loss"]))
    focused = json.loads((output/"focus.json").read_text()).get("normal",[]) if (output/"focus.json").exists() else []
    for case in focused:
        for obs in case["profile_queries"]:
            val_queries[obs["index"]][2].append(obs)
    for surface in ledger["surfaces"]:
        if surface["AP_loss"] <= 0 or any(r["parent"]==surface["id"] for r in focused):
            continue
        # The uninspected part of a surface remains explicitly unresolved.
        obs = max(surface["observations"], key=lambda r:r[3])
        val_queries[obs[0]][2].append(dict(case=surface["id"], component=surface["component"], semantic=surface["semantic"]))
    from .diagnose import score_curve
    curve = score_curve(np.load(output/"train_scores.npy",mmap_mode="r"),np.load(output/"train_labels.npy",mmap_mode="r"))
    if abs(curve["AP"]-training["pooled_training"]["AP"])>1e-9:
        raise ValueError("training score population changed")
    ranks = dict(train=tuple(curve[k] for k in ("values","positive","negative")),val=rank_data()[:3])
    for mode, queries in (("train", train_queries), ("val", val_queries)):
        rows, vectors = [], []
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, initializer=_profile_init,
                initargs=(str(output), mode, queries, ranks[mode])) as executor:
            for n, batch in enumerate(executor.map(_profiles, sorted(queries), chunksize=8), 1):
                for row, vector in batch:
                    rows.append(row)
                    vectors.append(vector)
                if n % 400 == 0:
                    print(f"material profiles {mode}: {n}/{len(queries)}, {time.perf_counter()-start:.1f}s", flush=True)
        np.save(output / f"{mode}_profiles.npy", np.stack(vectors))
        write_json(output / f"{mode}_profiles.json", dict(checkpoint_sha256=ledger["checkpoint_sha256"],
            source_manifest=training["train_manifest"] if mode == "train" else ledger["validation_manifest"],
            seconds=time.perf_counter()-start, records=rows,
            feature_blocks=dict(geometry=[0,17], sampling=[17,24], response=[24,27], context=[27,36]),
            ranking_reference=dict(normal_points=int(ranks[mode][2].sum()), anomaly_points=int(ranks[mode][1].sum()),
                AP=curve["AP"] if mode=="train" else ledger["metrics"]["AP"],
                definition="Opposite-class outranking fraction, half credit for ties. Training profiles use all frozen training points; validation profiles use all eligible validation points. Within-scan rates use only that frame; null means no opposite class. Different reference populations are not a causal calibration comparison."),
            scope="Measured observation profiles, not proof of physical material/shape equivalence. Native intensities are not cross-sensor calibrated.",
            normal_selection="Training: prior high-score cells plus most populated 0.75 m cell per range band. Validation: focused parents use all observations of selected world-cell episodes in focus.json; other surfaces use one max-loss patch. Remaining AP is explicitly unresolved."), indent=None)
        print(f"material profiles {mode}: {len(rows)} observations, {time.perf_counter()-start:.1f}s", flush=True)


def evidence(output, workers):
    """Attach inspected material candidates and explicit causal unknowns to all cases."""
    from scipy.spatial import cKDTree
    ledger = json.loads((output / "ledger.json").read_text())
    focus = json.loads((output/"focus.json").read_text()) if (output/"focus.json").exists() else None
    train = json.loads((output / "train_profiles.json").read_text())
    val = json.loads((output / "val_profiles.json").read_text())
    if train["checkpoint_sha256"] != val["checkpoint_sha256"] or train["checkpoint_sha256"] != ledger["checkpoint_sha256"]:
        raise ValueError("material profiles and AP ledger must use C")
    if val["source_manifest"]!=ledger["validation_manifest"] or train["source_manifest"]!=load_manifest("results/data/native/train.json","train")["sha256"]:
        raise ValueError("material profiles use different source populations")
    x = np.load(output / "train_profiles.npy").astype(float)
    q = np.load(output / "val_profiles.npy").astype(float)
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError("nonfinite measured material profiles")
    neighbors = [[] for _ in q]
    calibrations = []
    for kind in ("anomaly", "normal"):
        query_indices = np.array([i for i,r in enumerate(val["records"]) if r["kind"] == kind])
        for domain in ("stu", "nuscenes"):
            indices = np.array([i for i,r in enumerate(train["records"]) if r["kind"] == kind and r["domain"] == domain])
            center = np.median(x[indices], axis=0)
            scale = np.maximum(np.quantile(x[indices], .75, axis=0) - np.quantile(x[indices], .25, axis=0), .05)
            # Equal total scale for the four measured blocks; scores never enter distance.
            for lo, hi in train["feature_blocks"].values():
                scale[lo:hi] *= np.sqrt(hi-lo)
            transformed = (x[indices] - center) / scale
            tree = cKDTree(transformed)
            distance, at = tree.query((q[query_indices]-center)/scale, k=min(32,len(indices)), workers=workers)
            need = 1 if kind == "normal" and domain == "stu" else 3
            for pos, qi in enumerate(query_indices):
                used = set()
                candidates, distances = at[pos], distance[pos]
                k = len(candidates)
                while True:
                    chosen = []
                    used.clear()
                    for candidate, total in zip(candidates, distances):
                        ti = int(indices[candidate])
                        row = train["records"][ti]
                        unit = "206" if kind == "normal" and domain == "stu" else row["unit"]
                        if unit in used:
                            continue
                        used.add(unit)
                        delta = (q[qi]-x[ti])/scale
                        chosen.append(dict(profile=ti, distance=float(total),
                            blocks={name:float(np.linalg.norm(delta[lo:hi])) for name,(lo,hi) in train["feature_blocks"].items()}))
                        if len(chosen) == need:
                            break
                    if len(chosen) >= need or k >= len(indices):
                        break
                    k = min(k*4,len(indices))
                    distances,candidates = tree.query((q[qi]-center)/scale,k=k)
                neighbors[qi].extend(chosen)
            calibrations.append(dict(kind=kind,domain=domain,center=center.tolist(),scale=scale.tolist()))
    by_case = defaultdict(list)
    for i, row in enumerate(val["records"]):
        by_case[row["case"]].append(i)
    links = [dict(query=i,case=r["case"],neighbors=neighbors[i]) for i,r in enumerate(val["records"])]
    write_json(output / "material_links.json", dict(records=links,normalizations=calibrations,
        training_manifest=train["source_manifest"], validation_manifest=val["source_manifest"],
        scope="All anomaly observations; focused normal parents use multiple world-cell episodes, other normal surfaces one representative. Exact measured nearest candidates, not certified material equivalence."), indent=None)
    reviewed = {}
    if focus:
        from .probe import review
        review(output)
        reviewed = json.loads((output/"review.json").read_text())
    report, rows = [], []
    total = ledger["AP_loss"]
    for kind, cases in (("anomaly",ledger["objects"]),("normal",ledger["surfaces"])):
        cumulative = 0.
        for case in cases:
            case_id, loss = case["id"], case["AP_loss"]
            cumulative += loss
            query_ids = by_case[case_id]
            represented = sum(val["records"][i]["represented_AP_loss"] for i in query_ids)
            material_mass = defaultdict(float)
            closest_bce, val_bce, train_rank, val_rank, denominator = 0.,0.,0.,0.,0.
            for i in query_ids:
                query = val["records"][i]
                weight = query["represented_AP_loss"]
                same_domain = [n for n in neighbors[i] if train["records"][n["profile"]]["domain"] == "stu"]
                closest = min(same_domain,key=lambda n:n["distance"])
                closest_bce += weight*train["records"][closest["profile"]]["mean_BCE"]
                val_bce += weight*query["mean_BCE"]
                train_rank += weight*train["records"][closest["profile"]]["reference_pair_error"]
                val_rank += weight*query["reference_pair_error"]
                denominator += weight
                for n in neighbors[i]:
                    material_mass[n["profile"]] += weight
            top = sorted(material_mass,key=lambda i:-material_mass[i])[:6]
            references = [dict(profile=i,**train["records"][i],represented_query_AP=material_mass[i]) for i in top]
            comparable_bce = closest_bce / denominator if denominator else None
            actual_bce = val_bce / denominator if denominator else None
            if loss <= 0:
                clue, test = "当前排序未产生 AP 失分", "保留正常参照，不由该项安排修改"
                cause = "当前无失分"
            elif kind == "anomaly" and case["instance"] == 0:
                clue = "实例身份未分配，单点不足以核验几何与素材等价"
                test = "先核对原始槽位、邻帧和标注身份；评价点保持不变"
                cause = "证据不足"
            elif comparable_bce is not None and comparable_bce >= np.log(2):
                clue = "同传感器近邻训练候选也有较大分类损失；是否为可信相关素材、是否训练不足尚未区分"
                test = "确认近邻素材关系后，固定这些训练样本做可拟合性短探针并跟踪独立检查；不直接扩大总体步数"
                cause = "证据不足"
            else:
                clue = "已检索到分类损失较低的训练近邻；几何、响应和背景等价尚未证实，泛化差异不能认定为学错"
                test = "先复查三维形态、表面响应和背景关系；素材可信后比较浅层与融合表示的同容量读出，区分信息不足与读出损失"
                cause = "证据不足"
            if loss>0 and kind=="normal":
                clue += "；结构身份为几何候选，代表局部之外的错误尚未完成素材核对"
            if kind=="anomaly" and loss>0:
                clue += ("；同帧错序率不低于跨帧，不支持用跨扫描偏移统一解释" if case["within_scan_rank_error"]>=case["cross_scan_rank_error"]
                         else "；跨帧错序率较高，但尚未排除正常结构组成差异")
            modification = "待原因证据，不改变数据、模型或训练设置" if loss>0 else "无需修改"
            if focus and loss>0:
                if case_id=="P125:1":
                    views = reviewed.get("case_findings",{}).get(case_id,{}).get("same_world_observations")
                    if views:
                        clue += f"；前三个近邻世界的{views['scans']}条既有观测已检查，未选最近素材按查询失分加权BCE={views['weighted_candidate_BCE']['unselected']:.6g}；描述更近不等于几何与响应已覆盖"
                    test = "固定125第133、127帧及原始扫描，比较当前记录5340与同世界未选观测的几何、响应和相邻结构；只改变候选观测，只有实际相关性改善才支持重选代表帧，总描述距离下降不作为充分证据"
                    modification = "不优先增加已检查三个世界的重复训练；下一项核验独立形态、响应及背景关系，确认具体覆盖缺口后再选择或补充对应观测"
                elif case_id=="P141:4":
                    audit_rows=[r for r in focus["candidate_audit"] if r["case"]==case_id]
                    full=sum(r["AP_loss"] for r in audit_rows if r["neighbors"]["full"]["index"]==5080)
                    geometry=sum(r["AP_loss"] for r in audit_rows if r["neighbors"]["geometry"]["index"]==5080)
                    clue += f"；5080在全描述最近邻中对应{full:.5f}个AP百分点，仅几何最近邻中对应{geometry:.5f}；尚未确认其相关性，不能归为没学够"
                    test = "先复核141仅几何近邻与真实失败观测，5080暂不作为拟合依据；相关素材确认后执行focus.json中最多40步的四帧拟合检查，并同时检查独立观测"
                    modification = "暂不依据5080提高取样频次或启动拟合；确认相关素材后，仅当拟合同时改善独立相关观测才考虑增加其使用，训练样本单独改善仅说明局部拟合成功"
                elif kind=="normal" and any(r["parent"]==case_id for r in focus.get("normal",[])):
                    clue += f"；已分解连续局部片段并计算{len(query_ids)}个实际观测描述，代表{represented:.5f}个AP百分点；固定特征对照见features.json，不能直接作融合因果结论"
                    patches = [r for r in reviewed.get("feature_readout",{}).get("normal_cases",[]) if r["parent"]==case_id]
                    if patches:
                        before,after=(sum(r["FP"][name] for r in patches) for name in ("input","context_post"))
                        clue += f"；相同上下文的449参数读出在本项{sum(r['points'] for r in patches)}个检查点上误报为{before}/{after}，阈值分别按诊断子集75%召回确定，不能当作完整AP变化"
                    if case_id=="N141:75":
                        test = "固定314–380连续片段及原始正常标签，核验377帧高分局部与训练记录5059的形态、表面和邻接结构；只改变用于比较的训练正常候选，先区分相似描述与可信相关素材，再考虑增加相关训练观测使用"
                        modification = "优先核对5059自身误报和实际相关性；相关性成立再做正常观测拟合及独立观测检查，现有读出结果不支持直接替换融合模块"
                    else:
                        test = "固定87–168连续片段，核验135帧两个高分局部与记录5757、6006中已低分正常局部的实际形态和背景；只改变用于比较的训练正常候选，缺少可信对应才支持补正常覆盖，不能仅凭读出分数差认定融合削弱"
                        modification = "先核对真实高分局部与已识别训练正常素材之间的形态及背景差异；保留C和现有融合，确认缺失观测后再补相应正常素材"
                elif kind=="anomaly":
                    worst=max(case["observations"],key=lambda r:r["AP_loss"])
                    test = f"本轮未启动本项原因实验；后续从{case_id}的第{worst['frame']}帧及正常侧{case['normal_drivers'][0][0]}核验素材，再指定唯一改动"
                else:
                    worst=max(case["observations"],key=lambda r:r[3])
                    test = f"本轮未启动本项原因实验；后续先细分{case_id}第{worst[1]}帧的实际高分局部，确认原始编码{case['semantic']}所对应的物理结构"
            item = dict(case=case_id, kind=kind, AP_loss=loss, cumulative_loss_percent=100*cumulative/total,
                points=case["points"], observations=len(case["observations"]),
                AP_loss_by_recall=case.get("AP_loss_by_recall"),
                profiled_observations=len(query_ids), material_profiled_AP=represented,
                material_unprofiled_AP=max(0.,loss-represented), cause_confirmed_AP=0., causal_unresolved_AP=loss,
                training_candidates=[dict(profile=r["profile"],represented_query_AP=r["represented_query_AP"]) for r in references],
                query_profiles=query_ids, weighted_training_BCE=comparable_bce, weighted_validation_BCE=actual_bce,
                weighted_training_rank_error=train_rank/denominator if denominator else None,
                weighted_validation_rank_error=val_rank/denominator if denominator else None,
                within_scan_rank_error=case.get("within_scan_rank_error"),cross_scan_rank_error=case.get("cross_scan_rank_error"),
                cause=cause, evidence=clue, next_discriminating_test=test,
                proposed_modification=modification,
                counterevidence="观测描述近邻不证明覆盖；低训练损失不证明学对或学错；一两次访问不证明训练不足",
                identity_basis="序列和原始异常实例号；连续观测分段保存在 ledger.json" if kind=="anomaly" else "同序列、原始标签、几何方向及世界坐标连通表面；没有正常实例真值",
                drivers=(case["normal_drivers"] if kind=="anomaly" else case["anomaly_drivers"])[:5])
            report.append(item)
            refs = "；".join(f"{r['domain']} 记录{r['index']} / 素材{r['profile']} / BCE={r['mean_BCE']:.5g} / 错序率={r['reference_pair_error']:.5g} / 访问{r['visits']}次" for r in references)
            rows.append(dict(案例=case_id,视角="异常对象" if kind=="anomaly" else "正常结构候选",
                固定排序失分百分点=loss,该视角累计失分比例=100*cumulative/total,有效点次=case["points"],观测次数=len(case["observations"]),
                各召回区间失分=case.get("AP_loss_by_recall"),同帧错序概率=case.get("within_scan_rank_error"),跨帧错序概率=case.get("cross_scan_rank_error"),
                对应训练素材=refs,训练素材学习表现=f"近邻训练 BCE={comparable_bce}; 验证观测 BCE={actual_bce}; 均仅为分类诊断",
                训练素材池内错序概率=item["weighted_training_rank_error"],验证观测池内错序概率=item["weighted_validation_rank_error"],
                原因类别=cause,原因证据=clue,修改方案=item["proposed_modification"],区分原因的小实验=test,
                尚未核对素材的失分百分点=item["material_unprofiled_AP"],尚未确认原因的失分百分点=loss,
                主要对侧案例="；".join(f"{key}:{value:.6g}" for key,value in item["drivers"][:5])))
    write_csv(output / "attribution.csv", [r for r in rows if r["视角"]=="异常对象"])
    write_csv(output / "normal_cases.csv", [r for r in rows if r["视角"]=="正常结构候选"])
    summary = dict(AP_loss=total,score_accounted_AP=ledger["assigned_AP_loss"], score_unassigned_AP=ledger["unassigned_AP_loss"],
        anomaly_cases=len(ledger["objects"]),normal_surface_candidates=len(ledger["surfaces"]),
        normal_positive_loss_cases=sum(r["AP_loss"]>0 for r in ledger["surfaces"]),
        material_profiled_AP={kind:sum(r["material_profiled_AP"] for r in report if r["kind"]==kind) for kind in ("anomaly","normal")},
        material_unprofiled_AP={kind:sum(r["material_unprofiled_AP"] for r in report if r["kind"]==kind) for kind in ("anomaly","normal")},
        confirmed_causes={"样本覆盖不足":0.,"样本有了但没学够":0.,"样本有了但学错了":0.,"证据不足":total},
        normal_case_identity="Automatically linked surface candidates, not certified physical objects",
        interpretation="Complete rank accounting is not completed causal attribution. The two marginal views overlap and must not be added.",
        results=report)
    write_json(output / "attribution.json", summary, indent=None)
    plot_evidence(output,ledger,train,val,links)
    print({k:v for k,v in summary.items() if k!="results"},flush=True)


def plot_evidence(output, ledger, training, validation, links):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager, FontProperties, findfont
    from matplotlib.ft2font import FT2Font
    from .diagnose import validation_arrays
    for font in ("times.ttf","simsun.ttc"):
        fontManager.addfont(f"/mnt/c/Windows/Fonts/{font}")
    chinese = FontProperties(fname="/mnt/c/Windows/Fonts/simsun.ttc")
    plt.rcParams.update({"font.family":"Times New Roman","pdf.fonttype":42,"font.size":11})

    def save(fig, stem):
        fig.canvas.draw()
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_text():
                continue
            expected="SimSun" if any('\u4e00'<=c<='\u9fff' for c in text.get_text()) else "Times New Roman"
            actual=FT2Font(findfont(text.get_fontproperties(),fallback_to_default=False)).family_name
            if actual!=expected:
                raise ValueError(f"unexpected report font {actual}, expected {expected}")
        fig.savefig(output/f"{stem}.pdf")
        fig.savefig(output/f"{stem}.png",dpi=170)
        plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(12,4.2))
    bottom=np.zeros(7)
    top=ledger["objects"][:6]
    for obj in top:
        mass=np.array(obj["AP_loss_by_recall"])
        axes[0].bar(range(7),mass,bottom=bottom,label=obj["id"])
        bottom+=mass
    axes[0].bar(range(7),np.array(ledger["AP_loss_by_recall"])-bottom,bottom=bottom,color=".75",label="其余案例")
    axes[0].set_xticks(range(7),[f"{lo*100:g}–{hi*100:g}" for lo,hi in zip(RECALL_BANDS[:-1],RECALL_BANDS[1:])])
    axes[0].set_xlabel("全局召回率区间（百分比）",fontproperties=chinese)
    axes[0].set_ylabel("失分百分点",fontproperties=chinese)
    legend=axes[0].legend(fontsize=8,ncol=2)
    for text in legend.get_texts():
        if text.get_text()=="其余案例":text.set_fontproperties(chinese)
    for name,key in (("异常标注案例","objects"),("正常表面候选","surfaces")):
        values=np.cumsum([r["AP_loss"] for r in ledger[key]])/ledger["AP_loss"]*100
        axes[1].plot(np.arange(1,len(values)+1),values,label=name)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("按失分排序的案例数",fontproperties=chinese)
    axes[1].set_ylabel("累计失分份额（百分比）",fontproperties=chinese)
    axes[1].legend(prop=chinese)
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    save(fig,"loss")

    val_manifest=load_manifest("assets/val.json","val")
    offsets={r["index"]:r for r in json.loads((OUTPUT/"val_offsets.json").read_text())["rows"]}
    train_manifest=load_manifest("results/data/native/train.json","train")
    scans=Scans(train_manifest)
    train_scores=np.load(output/"train_scores.npy",mmap_mode="r")
    train_frames=json.loads((output/"train.json").read_text())["frames"]
    fig=plt.figure(figsize=(12,13))
    for row_number,obj in enumerate(ledger["objects"][:4]):
        qi=max((i for i,r in enumerate(validation["records"]) if r["case"]==obj["id"]),
               key=lambda i:validation["records"][i]["represented_AP_loss"])
        query=validation["records"][qi]
        selected=[query]+[training["records"][min((r for r in links[qi]["neighbors"] if training["records"][r["profile"]]["domain"]==domain),key=lambda r:r["distance"])["profile"]]
                         for domain in ("stu","nuscenes")]
        panels=[]
        for col,item in enumerate(selected):
            if col==0:
                xyzi,target,score,meta=validation_arrays(offsets[item["index"]],val_manifest["records"][item["index"]])
                chosen=np.flatnonzero((target==1)&(meta["instance"]==item["selector"]["instance"]))
                title=f"{obj['id']} / {item['frame']:06d}"
            else:
                sample=scans[item["index"]]
                xyzi,target,slots=[sample[k] for k in ("xyzi","targets","slots")]
                record=train_manifest["records"][item["index"]]
                ids=np.zeros(sample["slot_count"],np.int32)
                with np.load(record["delta"],allow_pickle=False) as delta:
                    if record.get("source")=="nuscenes":ids[delta["slots"]]=delta["object_ids"]
                    else:ids[delta["source_slot"]]=delta["packed_labels"]>>16
                chosen=np.flatnonzero((target==1)&(ids[slots]==item["selector"]["instance"]))
                frame=train_frames[item["index"]]
                score=np.full(len(target),np.nan,np.float32)
                score[target>=0]=train_scores[frame["start"]:frame["stop"]]
                title=f"{'STU' if col==1 else 'nuScenes'} #{item['index']}"
            assert len(chosen)==item["points"]
            xyz=xyzi[:,:3]-np.median(xyzi[chosen,:3],axis=0)
            panels.append((xyz,chosen,score,title))
        radius=max(1.,np.ceil(max(np.abs(xyz[chosen]).max() for xyz,chosen,_,_ in panels)*2+.5)/2)
        for col,(xyz,chosen,score,title) in enumerate(panels):
            ax=fig.add_subplot(4,3,row_number*3+col+1,projection="3d")
            near=np.flatnonzero(np.linalg.norm(xyz,axis=1)<radius*1.5)
            near=near[np.linspace(0,len(near)-1,min(2000,len(near))).round().astype(int)]
            ax.scatter(*xyz[near].T,c=".75",s=.5,alpha=.4,rasterized=True)
            colors=ax.scatter(*xyz[chosen].T,c=score[chosen],vmin=-8,vmax=36,cmap="viridis",s=5,rasterized=True)
            ax.set(xlim=(-radius,radius),ylim=(-radius,radius),zlim=(-radius,radius),xlabel="x (m)",ylabel="y (m)",zlabel="z (m)")
            ax.set_title(f"{title}\nn={len(chosen)}, median={np.median(score[chosen]):.3f}",fontsize=10)
            ax.view_init(elev=25,azim=-55)
    fig.suptitle("主要失分观测与两域训练近邻的实际回波",fontproperties=chinese)
    fig.tight_layout(rect=(0,0,.92,.97))
    bar=fig.colorbar(colors,cax=fig.add_axes([.94,.25,.015,.5]))
    bar.set_label("异常分数",fontproperties=chinese)
    save(fig,"materials")
