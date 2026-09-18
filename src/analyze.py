"""Describe every raw STU/206 scan for the V4 data design, without synthesizing data.

The source/pose and instance-distance conventions follow AJAE-v3/observations.py.
Counts use original return records; no model, crop, registration or point removal
is used to improve the observations. Histogram bins are display bins only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

# Keep process-level parallelism from multiplying BLAS/OpenMP thread pools.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


LABELS = {
    0: "未标注", 1: "离群标签", 2: "异常", 10: "汽车", 11: "自行车",
    13: "公共汽车", 15: "摩托车", 16: "轨道车辆", 18: "卡车",
    20: "其他车辆", 30: "行人", 31: "骑自行车者", 32: "骑摩托车者",
    40: "道路", 44: "停车区域", 48: "人行道", 49: "其他地面",
    50: "建筑", 51: "围栏", 52: "其他结构", 60: "车道标线",
    70: "植被", 71: "树干", 72: "地形", 80: "杆状物", 81: "交通标志",
    99: "其他物体", 252: "运动汽车", 253: "运动骑自行车者",
    254: "运动行人", 255: "运动骑摩托车者", 256: "运动轨道车辆",
    257: "运动公共汽车", 258: "运动卡车", 259: "运动其他车辆",
}
QUANTILES = (0, .05, .25, .5, .75, .95, .99, 1)
Q_NAMES = ("min", "p05", "p25", "median", "p75", "p95", "p99", "max")
RANGE_EDGES = np.r_[2.5, np.arange(3, 51)]
COUNT_EDGES = np.r_[.5, 1.5, 2.5, 3.5, 4.5, 9.5, 19.5, 49.5, 99.5,
                    199.5, 499.5, 999.5, 1999.5, np.inf]
REFERENCES = ((10, 78, 164, 240), (11, 1, 293, 316), (15, 6, 45, 85))


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return None
    result = dict(zip(Q_NAMES, map(float, np.quantile(values, QUANTILES))))
    # Preserve the source float32 median convention for measured ranges.
    result["median"] = float(np.median(values))
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_inputs(root, rays_path):
    directory = root / "train" / "206"
    scans = sorted(p.stem for p in (directory / "velodyne").glob("*.bin"))
    labels = sorted(p.stem for p in (directory / "labels").glob("*.label"))
    expected = [f"{i:06d}" for i in range(449)]
    if scans != expected or labels != expected:
        raise ValueError("206 must have corresponding scans/labels for all frames 0..448")
    calibration = {}
    for line in (directory / "calib.txt").read_text().splitlines():
        key, text = line.split(":", 1)
        matrix = np.eye(4)
        matrix[:3] = np.fromstring(text, sep=" ").reshape(3, 4)
        calibration[key] = matrix
    camera = np.loadtxt(directory / "poses.txt").reshape(-1, 3, 4)
    if len(camera) != 449 or not np.isfinite(camera).all():
        raise ValueError("invalid or missing source poses")
    poses = np.broadcast_to(np.eye(4), (449, 4, 4)).copy()
    poses[:, :3] = camera
    transform = calibration["Tr"]
    # Preserve the published camera-to-LiDAR composition and source frame order.
    poses = np.stack([np.linalg.inv(transform) @ pose @ transform for pose in poses])
    with np.load(rays_path, allow_pickle=False) as saved:
        gamma, origin_x, origin_z = saved["even_params"]
        local, shift = saved["even_local"], saved["integer_shift"]
    angle = math.pi + gamma - 2 * math.pi * (np.arange(1024)[None] - shift[:, None]) / 1024
    cosine, sine = np.cos(angle), np.sin(angle)
    directions = np.stack((cosine * local[:, None, 0] - sine * local[:, None, 1],
                           sine * local[:, None, 0] + cosine * local[:, None, 1],
                           np.broadcast_to(local[:, None, 2], angle.shape)), axis=-1)
    origins = np.stack((origin_x * cosine, origin_x * sine,
                        np.full_like(cosine, origin_z)), axis=-1)
    canonical = np.arange(128)[:, None] * 1024 + (np.arange(1024)[None] - shift[:, None]) % 1024
    return directory, poses, directions.reshape(-1, 3), origins.reshape(-1, 3), np.argsort(canonical.ravel()), local


def init_worker(directory, poses, directions, origins, canonical_order):
    global _directory, _poses, _directions, _origins, _canonical_order
    _directory, _poses = directory, poses
    _directions, _origins, _canonical_order = directions, origins, canonical_order


def frame_analysis(frame_id):
    scan_path = _directory / "velodyne" / f"{frame_id:06d}.bin"
    label_path = _directory / "labels" / f"{frame_id:06d}.label"
    if scan_path.stat().st_size % 16 or label_path.stat().st_size % 4:
        raise ValueError(f"malformed binary lengths at frame {frame_id}")
    xyzi = np.fromfile(scan_path, dtype="<f4").reshape(-1, 4)
    packed = np.fromfile(label_path, dtype="<u4")
    if len(xyzi) != 131072 or len(packed) != len(xyzi) or not np.isfinite(xyzi).all():
        raise ValueError(f"invalid point/label layout or nonfinite value at frame {frame_id}")
    raw, instance = packed & 65535, packed >> 16
    unknown = set(map(int, np.unique(raw))) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown raw semantics at frame {frame_id}: {unknown}")
    actual = np.any(xyzi[:, :3] != 0, axis=1)
    radius = np.linalg.norm(xyzi[:, :3], axis=1)
    inside = actual & (radius >= 2.5) & (radius <= 50)
    normal = inside & (raw != 0) & (raw != 2)
    anomaly = inside & (raw == 2)
    row = dict(frame=frame_id, slots=len(xyzi), returns_all=int(actual.sum()),
               empty_slots=int((~actual).sum()), empty_nonzero_label=int(np.count_nonzero(packed[~actual])),
               empty_nonzero_intensity=int(np.count_nonzero(xyzi[~actual, 3])),
               returns_2p5_50m=int(inside.sum()), valid_normal=int(normal.sum()),
               valid_anomaly=int(anomaly.sum()), ignored_in_range=int((inside & (raw == 0)).sum()),
               returns_below_2p5m=int((actual & (radius < 2.5)).sum()),
               returns_above_50m=int((actual & (radius > 50)).sum()),
               native_anomaly_all=int((actual & (raw == 2)).sum()),
               normal_without_instance=int((normal & (instance == 0)).sum()),
               normal_with_instance=int((normal & (instance > 0)).sum()))
    points = xyzi[actual]
    ordered = points[np.lexsort(tuple(points[:, i] for i in (3, 2, 1, 0)))]
    row["coincident_xyz_extra_records"] = int(np.all(np.diff(ordered[:, :3], axis=0) == 0, axis=1).sum())
    row["identical_xyzi_extra_records"] = int(np.all(np.diff(ordered, axis=0) == 0, axis=1).sum())
    for name, values in (("range_all", radius[actual]), ("range_normal", radius[normal]),
                         ("intensity_all", xyzi[actual, 3]), ("intensity_normal", xyzi[normal, 3])):
        row.update({f"{name}_{k}": v for k, v in quantiles(values).items()})
    row["negative_intensity"] = int((xyzi[actual, 3] < 0).sum())
    row["zero_intensity"] = int((xyzi[actual, 3] == 0).sum())
    row["empty_intensity_min"] = float(xyzi[~actual, 3].min())
    row["empty_intensity_max"] = float(xyzi[~actual, 3].max())
    row["intensity_quantization_residual_max"] = float(np.max(np.abs(
        xyzi[actual, 3].astype(float) * 3500 - np.rint(xyzi[actual, 3].astype(float) * 3500))))
    ray_vector = xyzi[actual, :3].astype(float) - _origins[actual]
    along = np.einsum("ij,ij->i", ray_vector, _directions[actual]) / np.square(_directions[actual]).sum(1)
    residual = np.linalg.norm(ray_vector - along[:, None] * _directions[actual], axis=1)
    row["rays_behind_origin"] = int((along < 0).sum())
    row.update({f"ray_residual_m_{k}": v for k, v in quantiles(residual).items()})
    # Neighbour distances describe pairs of measured rays, not surface continuity.
    xyz_grid = xyzi[_canonical_order, :3].reshape(128, 1024, 3)
    mask_grid = normal[_canonical_order].reshape(128, 1024)
    for axis, name in ((1, "horizontal"), (0, "vertical")):
        if axis == 1:
            valid_pairs = mask_grid & np.roll(mask_grid, -1, axis)
            delta = xyz_grid - np.roll(xyz_grid, -1, axis)
        else:
            valid_pairs = mask_grid[:-1] & mask_grid[1:]
            delta = np.diff(xyz_grid, axis=0)
        distances = np.linalg.norm(delta[valid_pairs], axis=1)
        row[f"{name}_pairs"] = len(distances)
        row.update({f"{name}_spacing_m_{k}": v for k, v in quantiles(distances).items()})
    classes = []
    for semantic in np.unique(raw):
        all_mask, valid_mask = actual & (raw == semantic), inside & (raw == semantic)
        q = quantiles(radius[valid_mask])
        classes.append(dict(frame=frame_id, semantic=int(semantic), category=LABELS[int(semantic)],
                            returns_all=int(all_mask.sum()), returns_2p5_50m=int(valid_mask.sum()),
                            returns_with_instance=int((valid_mask & (instance > 0)).sum()),
                            instance_ids=sorted(map(int, np.unique(instance[valid_mask & (instance > 0)]))),
                            range_median=None if q is None else q["median"]))
    objects = {}
    for identity in np.unique(packed[actual & (instance > 0) & (raw > 2)]):
        semantic, number = int(identity) & 65535, int(identity) >> 16
        mask = actual & (packed == identity)
        valid_mask = mask & inside
        cloud = xyzi[mask, :3].astype(float)
        world = cloud @ _poses[frame_id, :3, :3].T + _poses[frame_id, :3, 3]
        key = f"206:{semantic}:{number}"
        observation = dict(structure_id=key, semantic=semantic, instance_id=number, frame=frame_id,
                           returns_all=int(mask.sum()), returns_2p5_50m=int(valid_mask.sum()))
        for name, values in (("range_all_m", radius[mask]), ("range_valid_m", radius[valid_mask]),
                             ("intensity_valid", xyzi[valid_mask, 3])):
            q = quantiles(values)
            observation.update({f"{name}_{k}": None if q is None else q[k] for k in ("min", "median", "max")})
        center = cloud.mean(0)
        observation["azimuth_deg"] = float(np.rad2deg(np.arctan2(center[1], center[0])))
        for axis, value in zip("xyz", world.mean(0)):
            observation[f"observed_centroid_world_{axis}_m"] = float(value)
        objects[key] = (observation, world)
    numeric_ids = defaultdict(set)
    for key in objects:
        _, semantic, number = map(int, key.split(":"))
        numeric_ids[number].add(semantic)
    row["reused_numeric_instance_ids"] = ";".join(str(k) for k,v in sorted(numeric_ids.items()) if len(v)>1)
    return row, classes, objects, radius[normal], xyzi[normal, 3], residual


def cloud_distances(first, second):
    """Equal standing for both observed surfaces; no registration or motion fit."""
    forward = cKDTree(second).query(first, workers=1)[0]
    backward = cKDTree(first).query(second, workers=1)[0]
    return np.maximum(np.quantile(forward, [.5, .95]), np.quantile(backward, [.5, .95]))


def track_analysis(item):
    key, entries = item
    observations, clouds = zip(*entries)
    ids = [row["frame"] for row in observations]
    for i in range(1, len(entries)):
        median, tail = cloud_distances(clouds[i - 1], clouds[i])
        observations[i].update(previous_observed_frame=ids[i - 1], gap_frames=ids[i] - ids[i - 1],
                               nn_world_median_m=float(median), nn_world_p95_m=float(tail))
    world = np.concatenate(clouds)
    valid = [row for row in observations if row["returns_2p5_50m"] > 0]
    n = np.array([r["returns_2p5_50m"] for r in valid])
    distance = np.array([r["range_valid_m_median"] for r in valid])
    rho = float(spearmanr(n, distance).statistic) if len(n) > 2 and np.ptp(n) and np.ptp(distance) else None
    result = dict(structure_id=key, semantic=observations[0]["semantic"],
                  category=LABELS[observations[0]["semantic"]], instance_id=observations[0]["instance_id"],
                  moving_label=observations[0]["semantic"] >= 252,
                  observed_frames_all=len(entries), observed_frames_in_range=len(valid),
                  first_frame_all=ids[0], last_frame_all=ids[-1],
                  first_frame_in_range=valid[0]["frame"] if valid else None,
                  last_frame_in_range=valid[-1]["frame"] if valid else None,
                  unobserved_frames_inside_span=ids[-1] - ids[0] + 1 - len(ids),
                  maximum_gap_frames=int(max(np.diff(ids), default=0)),
                  returns_all=sum(r["returns_all"] for r in observations),
                  returns_2p5_50m=int(n.sum()), frames_1_to_4=int(((n >= 1) & (n <= 4)).sum()),
                  spearman_count_distance=rho)
    for name, values in (("count", n), ("distance", distance),
                         ("adjacent_nn_median", [r["nn_world_median_m"] for r in observations[1:]]),
                         ("adjacent_nn_p95", [r["nn_world_p95_m"] for r in observations[1:]])):
        q = quantiles(values)
        result.update({f"{name}_{k}": None if q is None else q[k] for k in ("min", "median", "p95", "max")})
    for axis, lower, upper in zip("xyz", world.min(0), world.max(0)):
        result[f"world_{axis}_min_m"], result[f"world_{axis}_max_m"] = float(lower), float(upper)
    thirds = []
    blocks = [np.concatenate([clouds[i] for i in group]) for group in np.array_split(np.arange(len(clouds)), 3) if len(group)]
    for first, second in zip(blocks, blocks[1:]):
        thirds.append(cloud_distances(first, second).tolist())
    result["temporal_thirds_nn_median_max_m"] = max((r[0] for r in thirds), default=None)
    result["temporal_thirds_nn_p95_max_m"] = max((r[1] for r in thirds), default=None)
    segments = []
    for group in np.split(np.arange(len(valid)), np.flatnonzero(np.diff([r["frame"] for r in valid]) != 1) + 1):
        if not len(group):
            continue
        rows = [valid[i] for i in group]
        segments.append(dict(structure_id=key, semantic=result["semantic"],
                             start_frame=rows[0]["frame"], end_frame=rows[-1]["frame"], frames=len(rows),
                             count_min=min(r["returns_2p5_50m"] for r in rows),
                             count_max=max(r["returns_2p5_50m"] for r in rows),
                             frames_1_to_4=sum(r["returns_2p5_50m"] <= 4 for r in rows),
                             distance_min_m=min(r["range_valid_m_median"] for r in rows),
                             distance_max_m=max(r["range_valid_m_median"] for r in rows)))
    result["contiguous_visible_segments"] = len(segments)
    result["longest_contiguous_visible_frames"] = max((s["frames"] for s in segments), default=0)
    return result, list(observations), segments


def analyze(args):
    import psutil
    started = time.monotonic()
    inputs = read_inputs(args.data_root, args.rays)
    directory, poses, directions, origins, order, local = inputs
    init_worker(*inputs[:5])
    frames, class_frames, tracks = [], [], defaultdict(list)
    ranges, intensities, residuals = [], [], []
    peak_rss = 0
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=inputs[:5]) as pool:
        for result in pool.map(frame_analysis, range(449), chunksize=2):
            row, classes, objects, distance, intensity, residual = result
            frames.append(row)
            class_frames.extend(classes)
            for key, observation in objects.items():
                tracks[key].append(observation)
            ranges.append(distance); intensities.append(intensity); residuals.append(residual)
            if len(frames) % 64 == 0 or len(frames) == 449:
                processes = [psutil.Process()] + psutil.Process().children()
                peak_rss = max(peak_rss, sum(p.memory_info().rss for p in processes if p.is_running()))
                print(json.dumps(dict(frames=len(frames), elapsed_s=round(time.monotonic() - started, 2),
                                      rss_bytes=peak_rss)), flush=True)
        # Frame arrays are bounded by 206; the much smaller instance clouds are reused.
        track_results = list(pool.map(track_analysis, sorted(tracks.items()), chunksize=1))
    structures, observations, segments = [], [], []
    for structure, rows, intervals in track_results:
        structures.append(structure); segments.extend(intervals)
        observed = {row["frame"]: row for row in rows}
        for frame in range(449):
            observations.append(observed.get(frame, dict(structure_id=structure["structure_id"],
                                semantic=structure["semantic"], instance_id=structure["instance_id"],
                                frame=frame, returns_all=0, returns_2p5_50m=0)))
    classes = []
    for semantic, name in LABELS.items():
        selected = [r for r in class_frames if r["semantic"] == semantic]
        members = [s for s in structures if s["semantic"] == semantic]
        classes.append(dict(semantic=semantic, category=name, observed_frames=sum(r["returns_all"] > 0 for r in selected),
                            returns_all=sum(r["returns_all"] for r in selected),
                            returns_2p5_50m=sum(r["returns_2p5_50m"] for r in selected),
                            returns_with_instance=sum(r["returns_with_instance"] for r in selected),
                            labeled_identities=len(members),
                            repeated_identities=sum(s["observed_frames_all"] > 1 for s in members),
                            instance_frames_in_range=sum(s["observed_frames_in_range"] for s in members),
                            frames_1_to_4=sum(s["frames_1_to_4"] for s in members)))
    rotation = poses[:, :3, :3]
    trajectory = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    valid_rows = [r for r in observations if r["returns_2p5_50m"] > 0]
    n = np.array([r["returns_2p5_50m"] for r in valid_rows])
    d = np.array([r["range_valid_m_median"] for r in valid_rows])
    histogram, _, _ = np.histogram2d(d, n, bins=(RANGE_EDGES, COUNT_EDGES))
    balanced = np.zeros_like(histogram)
    for structure in structures:
        rows = [r for r in valid_rows if r["structure_id"] == structure["structure_id"]]
        if rows:
            counts, _, _ = np.histogram2d([r["range_valid_m_median"] for r in rows],
                                         [r["returns_2p5_50m"] for r in rows], bins=(RANGE_EDGES, COUNT_EDGES))
            balanced += counts / len(rows)
    balanced /= sum(s["observed_frames_in_range"] > 0 for s in structures)
    totals = {key: sum(f[key] for f in frames) for key in (
        "slots", "returns_all", "empty_slots", "empty_nonzero_label", "empty_nonzero_intensity",
        "returns_2p5_50m", "valid_normal", "valid_anomaly", "ignored_in_range", "returns_below_2p5m",
        "returns_above_50m", "native_anomaly_all", "normal_without_instance", "normal_with_instance",
        "coincident_xyz_extra_records", "identical_xyzi_extra_records", "negative_intensity", "zero_intensity",
        "rays_behind_origin")}
    summary = dict(source=str(directory), frames=449,
                   scope="Raw 206 only; descriptive V4 analysis; no synthetic or model result.",
                   totals=totals,
                   trajectory=dict(length_m=float(steps.sum()), start_end_displacement_m=float(np.linalg.norm(trajectory[-1]-trajectory[0])),
                                   step_m=quantiles(steps), minimum_xyz_m=trajectory.min(0).tolist(), maximum_xyz_m=trajectory.max(0).tolist(),
                                   rotation_orthogonality_error_max=float(np.max(np.abs(rotation.transpose(0,2,1) @ rotation - np.eye(3)))),
                                   rotation_determinant_min=float(np.linalg.det(rotation).min()),
                                   rotation_determinant_max=float(np.linalg.det(rotation).max()),
                                   timestamps_present=False),
                   ray_model=dict(source=str(args.rays.resolve()), beams=128, columns=1024,
                                  horizontal_step_deg=360/1024,
                                  elevations_deg=quantiles(np.rad2deg(np.arcsin(local[:, 2]))),
                                  vertical_gap_deg=quantiles(np.diff(np.sort(np.rad2deg(np.arcsin(local[:, 2])))))),
                   structures=dict(identities=len(structures), repeated_identities=sum(s["observed_frames_all"]>1 for s in structures),
                                   moving_label_identities=sum(s["moving_label"] for s in structures),
                                   observed_instance_frames_all=sum(s["observed_frames_all"] for s in structures),
                                   observed_instance_frames_in_range=len(valid_rows),
                                   frames_1_to_4=int((n<=4).sum()), identities_with_1_to_4=sum(s["frames_1_to_4"]>0 for s in structures),
                                   count=quantiles(n), range_median_m=quantiles(d),
                                   contiguous_segments=len(segments),
                                   identities_with_count_and_distance_variation=sum(s["spearman_count_distance"] is not None for s in structures)),
                   joint_distribution=dict(distance_edges_m=RANGE_EDGES.tolist(), count_edges=[None if np.isinf(v) else v for v in COUNT_EDGES],
                                           observation_counts=histogram.astype(int).tolist(), structure_equal_weight=balanced.tolist()),
                   definitions=dict(valid="actual return; float32 Euclidean norm in [2.5,50]; raw !=0; anomaly raw==2",
                                    identity="206:raw semantic:nonzero instance ID; labels are not a verified persistent object identity",
                                    range="full min/median/max retained; joint plots use in-range return-distance median",
                                    quantiles="median uses numpy.median, preserving float32 range precision; other quantiles use linear interpolation",
                                    zeros="no observed return of that label identity; distances remain missing; no interpolation",
                                    segments="maximal consecutive frame runs with >=1 in-range return of one labeled identity",
                                    geometry="observed surfaces in inv(Tr) @ pose_camera @ Tr world; no registration; not full object shape/height",
                                    binning="display-only distance/count bins; no V4 coverage or acceptance thresholds are defined",
                                    frame_counts="normal reference observations retain 1-4 returns; anomaly >=5 frame eligibility is not imposed on pure normal data",
                                    normal_training="user confirmed zero-anomaly scans may supply only normal supervision; sampler/loss unspecified"))
    summary["structures"]["structure_equal_weight_fraction_1_to_4"] = float(np.mean([
        s["frames_1_to_4"] / s["observed_frames_in_range"] for s in structures if s["observed_frames_in_range"]]))
    summary["structures"]["reused_numeric_ids"] = sorted({int(i) for row in frames
        for i in row["reused_numeric_instance_ids"].split(";") if i})
    summary["reference_examples"] = []
    for semantic, number, first, last in REFERENCES:
        key = f"206:{semantic}:{number}"
        entries = [(row, cloud) for row,cloud in tracks[key] if first <= row["frame"] <= last]
        rows, clouds = zip(*entries)
        support = []
        for i, cloud in enumerate(clouds):
            other = np.concatenate([c for j,c in enumerate(clouds) if j!=i])
            support.append(float(np.quantile(cKDTree(other).query(cloud,workers=1)[0],.95)))
        summary["reference_examples"].append(dict(
            structure_id=key, start_frame=first, end_frame=last, observed_frames=len(rows),
            count=quantiles([r["returns_2p5_50m"] for r in rows]),
            distance_m=quantiles([r["range_valid_m_median"] for r in rows]),
            frames_1_to_4=sum(1<=r["returns_2p5_50m"]<=4 for r in rows),
            other_frames_nn_p95_m=quantiles(support),
            use="previous exploratory example, remeasured here; not a V4 selection or matching threshold"))
    for name, arrays in (("normal_range_m", ranges), ("normal_intensity", intensities), ("ray_residual_m", residuals)):
        values = np.concatenate(arrays)
        summary[name] = quantiles(values)
        if name == "normal_range_m":
            summary["normal_range_histogram"] = np.histogram(values, bins=RANGE_EDGES)[0].tolist()
        del values
        arrays.clear()
    # Independent accounting checks protect denominators and zero-observation semantics.
    assert totals["slots"] == totals["empty_slots"] + totals["returns_all"]
    assert totals["returns_all"] == totals["returns_below_2p5m"] + totals["returns_2p5_50m"] + totals["returns_above_50m"]
    assert totals["returns_2p5_50m"] == totals["valid_normal"] + totals["valid_anomaly"] + totals["ignored_in_range"]
    assert sum(r["returns_all"] for r in classes) == totals["returns_all"]
    assert sum(r["returns_2p5_50m"] for r in valid_rows) == sum(s["returns_2p5_50m"] for s in structures)
    assert sum(s["frames"] for s in segments) == len(valid_rows)
    assert int(histogram.sum()) == len(valid_rows) and np.isclose(balanced.sum(), 1)
    for i, row in enumerate(frames):
        row.update({f"sensor_world_{axis}_m": float(trajectory[i,j]) for j,axis in enumerate("xyz")})
        row["travel_from_previous_m"] = None if i==0 else float(steps[i-1])
    summary["execution"] = dict(workers=args.workers, numerical_threads=1, elapsed_seconds=time.monotonic()-started,
                                sampled_peak_process_rss_bytes=peak_rss,
                                command=f"python src/analyze.py --data-root {args.data_root} --rays {args.rays} --output {args.output} --workers {args.workers}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("frames",frames),("labels",classes),("observations",observations),("structures",structures),("segments",segments)):
        write_csv(output / f"{name}.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("totals", "trajectory", "structures", "normal_range_m", "normal_intensity", "ray_residual_m", "execution")},ensure_ascii=False,allow_nan=False),flush=True)


def report(output):
    """Render the reviewed measurements with explicit Chinese/Latin font runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.ft2font import FT2Font
    from matplotlib.textpath import TextToPath

    def load(name):
        with (output / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    frames, labels, observations, structures = [load(n) for n in ("frames", "labels", "observations", "structures")]
    summary = json.loads((output / "summary.json").read_text())
    total, identities = summary["totals"], summary["structures"]
    paths = ("/mnt/c/Windows/Fonts/times.ttf", "/mnt/c/Windows/Fonts/simsun.ttc")
    faces = [FT2Font(path) for path in paths]
    if [face.family_name for face in faces] != ["Times New Roman", "SimSun"]:
        raise ValueError("required original fonts are unavailable")
    for path in paths:
        fontManager.addfont(path)
    plt.rcParams.update({"font.family":"Times New Roman", "pdf.fonttype":42,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "font.size":9, "axes.titlesize":11, "axes.labelsize":9,
                         "savefig.dpi":220})
    metrics = TextToPath()
    widths = {}
    figure = None
    page_number = 0
    pdf = PdfPages(output / "report.pdf", metadata={"Title":"206 序列分析", "Author":"AJAE V4"})

    def font_index(c):
        return int("\u2e80" <= c <= "\u9fff" or "\uff00" <= c <= "\uffef")

    def advance(c, size):
        key = (c, size)
        if key not in widths:
            index = font_index(c)
            if not faces[index].get_char_index(ord(c)):
                raise ValueError(f"required font lacks character {c!r}")
            widths[key] = metrics.get_text_width_height_descent(c, FontProperties(fname=paths[index], size=size), False)[0]
        return widths[key]

    def rich(text, x, y, size=10.5, color="#20252b"):
        # Coordinates and advances are physical PDF points, independent of DPI.
        start = 0
        while start < len(text):
            index = font_index(text[start]); end = start + 1
            while end < len(text) and font_index(text[end]) == index:
                end += 1
            span = text[start:end]
            figure.text(x / 595.44, y / 841.68, span,
                        fontproperties=FontProperties(fname=paths[index], size=size), color=color)
            x += sum(advance(c, size) for c in span)
            start = end

    def text(value, y, size=10.5, x=45, width=505, leading=17):
        line = ""; used = 0
        for character in str(value) + "\n":
            if character == "\n":
                rich(line,x,y,size); y -= leading; line=""; used=0
                continue
            length = advance(character,size)
            if used + length > width:
                rich(line,x,y,size); y -= leading; line=""; used=0
            line += character; used += length
        return y - 7

    def page(title):
        nonlocal figure,page_number
        if figure is not None:
            pdf.savefig(figure); plt.close(figure)
        page_number += 1
        figure = plt.figure(figsize=(8.27,11.69))
        rich("AJAE V4 · 206 序列分析",45,808,10,"#56606b")
        rich(title,45,775,18)
        rich(str(page_number),540,25,9,"#56606b")
        figure.add_artist(plt.Line2D([45/595.44,550/595.44],[794/841.68]*2,
                                    transform=figure.transFigure,color="#b0b7be",lw=.7))
        return 745

    def table(headers, rows, y, columns, size=9.5, height=21):
        for values in [headers] + rows:
            x=45
            for value,width in zip(values,columns):
                rich(str(value),x,y,size)
                x += width
            y -= height
        return y - 12

    def numbers(key):
        return np.array([float(r[key]) for r in frames])

    def pct(value, denominator):
        return f"{100*value/denominator:.2f}%"

    y=page("数据基础与主要结论")
    y=text("本次读取 STU/train/206 的全部 449 帧原始扫描、标签、相机位姿和标定。统计服务于总方案的广覆盖数据构造、正常观测参照及单帧分割监督；没有生成异常、训练模型或计算异常检测成绩。",y)
    y=table(["项目","全序列结果"],[
        ["原始文件记录",f"{total['slots']:,}"],
        ["实际回波",f"{total['returns_all']:,}"],
        ["坐标全零的空记录",f"{total['empty_slots']:,}"],
        ["2.5–50 米内实际回波",f"{total['returns_2p5_50m']:,}"],
        ["范围内有效正常点",f"{total['valid_normal']:,}"],
        ["范围内忽略标签点",f"{total['ignored_in_range']:,}"],
        ["全扫描原生异常回波",f"{total['native_anomaly_all']:,}"],
        ["每帧实际回波最少／中位／最多", " / ".join(f"{int(v):,}" for v in np.quantile(numbers('returns_all'),[0,.5,1]))],
    ],y,[285,220])
    y=text("449 帧扫描、449 份标签和 449 个位姿逐帧对应，帧号为 0–448。扫描均为 131,072 条四维记录，坐标和强度均为有限数。未发现帧内相同坐标或相同四维记录的额外副本；统计保留原始文件记录，未按坐标删点。",y)
    y=text(f"全部 {total['empty_slots']:,} 个空记录都具有非零强度，且原始标签为 0。判定实际回波必须检查坐标是否全零，不能仅检查强度。范围外的 {total['returns_below_2p5m']+total['returns_above_50m']:,} 个实际回波仍属于模型完整扫描输入。",y)
    y=text("206 的原生异常点数为零。按照本轮已确认的规则，这些纯正常帧可提供正常监督；异常分割训练还需要合成异常。本文所称有效正常点，指坐标非全零、三维欧氏距离位于含边界的 2.5–50 米、原始标签非 0 且非 2 的点。",y)
    text("这些结果确认原始数据的读取与监督支持，不能证明异常可学习、合成数据充分或模型性能提升。",y)

    y=page("语义组成与实例身份覆盖")
    y=text("下表列出实际出现的原始语义类别。比例以范围内有效正常点为分母；原始标签 0 不参与正常监督。实例身份必须同时包含序列、原始语义类别和非零编号。",y)
    rows=[]
    for r in labels:
        if int(r["returns_all"]):
            rows.append([r["semantic"],r["category"],f"{int(r['returns_2p5_50m']):,}",
                         "—" if int(r["semantic"]) in (0,2) else pct(int(r["returns_2p5_50m"]),total["valid_normal"]),r["labeled_identities"]])
    y=table(["原始标签","类别","范围内回波","正常占比","实例身份"],rows,y,[60,132,135,90,70],height=19,size=9)
    y=text(f"有效正常点中，仅 {total['normal_with_instance']:,} 点具有非零实例编号，占 {pct(total['normal_with_instance'],total['valid_normal'])}。道路、建筑、植被、树干、杆状物、交通标志等在本序列中没有非零实例编号。它们继续提供正常背景监督，不能把整类点当成一个物体统计。",y)
    extra = {int(r['semantic']):int(r['returns_2p5_50m']) for r in labels}
    text(f"原始标签 1、52、99 分别有 {extra[1]:,}、{extra[52]:,}、{extra[99]:,} 个范围内回波。它们在官方二元口径中属于正常点，不能沿用旧语义训练映射将其丢弃。标签 1 缺乏明确正常结构语义，因此不据此构造实例参照。",y)

    y=page("轨迹、位姿与射线一致性")
    trajectory=summary["trajectory"]
    y=text(f"传感器沿原始位姿累计移动 {trajectory['length_m']:.3f} 米，首尾位移 {trajectory['start_end_displacement_m']:.3f} 米。每相邻帧的移动中位数为 {trajectory['step_m']['median']:.3f} 米，最大为 {trajectory['step_m']['max']:.3f} 米。文件没有时间戳，因此不将这些量换算为速度。",y)
    ax=figure.add_axes([.12,.47,.78,.32])
    xy=np.column_stack((numbers("sensor_world_x_m"),numbers("sensor_world_y_m")))
    ax.plot(*xy.T,color="#bcc3cc",lw=1)
    points=ax.scatter(*xy.T,c=np.arange(449),s=9,cmap="viridis")
    ax.scatter(*xy[0],marker="o",s=35,c="#1b6b48",label="Start")
    ax.scatter(*xy[-1],marker="s",s=35,c="#943f36",label="End")
    ax.set(xlabel="World x (m)",ylabel="World y (m)");ax.set_aspect("equal",adjustable="datalim")
    figure.colorbar(points,ax=ax,label="Frame",pad=.03);ax.legend(loc="best",frameon=False)
    y=345
    ray=summary["ray_model"]; residual=summary["ray_residual_m"]
    y=text(f"世界坐标采用 inv(Tr) × pose_camera × Tr，旋转矩阵的最大正交误差约为 {trajectory['rotation_orthogonality_error_max']:.2e}。该世界坐标继承初始雷达参考轴，不代表地理方位；没有重新配准或改变原始轨迹。",y)
    y=text(f"已有射线标定包含 128 束、每圈 1,024 列，名义水平间隔 {ray['horizontal_step_deg']:.7f} 度。束仰角从 {ray['elevations_deg']['min']:.3f} 至 {ray['elevations_deg']['max']:.3f} 度，相邻束仰角间隔中位数为 {ray['vertical_gap_deg']['median']:.3f} 度。",y)
    y=text(f"本次对全部实际回波计算到对应标定射线的垂直距离，中位数 {residual['median']*1000:.3f} 毫米，95 分位 {residual['p95']*1000:.3f} 毫米，99 分位 {residual['p99']*1000:.3f} 毫米，最大 {residual['max']*1000:.3f} 毫米；没有回波位于射线原点后方。",y)
    text("这支持已有标定与 206 文件记录的几何一致性。该标定原本由 206 拟合，因此这不是独立传感器精度验证，也不自动确定 V4 允许的合成误差。",y)

    page("回波距离、强度与采样间距")
    ax=figure.add_axes([.13,.61,.77,.24])
    ax.plot(np.arange(449),numbers("returns_all"),label="All actual returns",lw=1)
    ax.plot(np.arange(449),numbers("valid_normal"),label="Valid normal returns",lw=1)
    ax.set(xlabel="Frame",ylabel="Return count");ax.legend(frameon=False)
    ax=figure.add_axes([.13,.32,.77,.21])
    edges=np.array(summary["joint_distribution"]["distance_edges_m"])
    ax.bar(edges[:-1],np.array(summary["normal_range_histogram"])/total["valid_normal"]*100,
           width=np.diff(edges),align="edge",color="#356b88",edgecolor="white",linewidth=.3)
    ax.set(xlabel="Range (m)",ylabel="Normal returns (%)",xlim=(2.5,50))
    y=218
    distance=summary["normal_range_m"]; intensity=summary["normal_intensity"]
    y=text(f"范围内正常点的距离中位数为 {distance['median']:.3f} 米，75 分位为 {distance['p75']:.3f} 米，95 分位为 {distance['p95']:.3f} 米。全扫描中最远实际回波为 {numbers('range_all_max').max():.3f} 米。点数多集中在近处，不能用累计点数代替不同位置的覆盖。",y)
    y=text(f"有效正常点原始强度范围为 {intensity['min']:.6f}–{intensity['max']:.6f}，中位数 {intensity['median']:.6f}。强度可以超过 1；分析未截断或重新归一化。逐帧的相邻水平射线正常回波间距中位数再取中位数为 {np.median(numbers('horizontal_spacing_m_median')):.4f} 米，垂直方向为 {np.median(numbers('vertical_spacing_m_median')):.4f} 米。",y)
    text("间距统计要求两条相邻射线均有范围内正常回波，但不要求属于同一表面；距离跳变可能跨越物体边界或遮挡边界，不能直接解释为表面粗糙度。",y)

    page("正常观测的回波数量与距离")
    joint=summary["joint_distribution"]
    distributions=[np.array(joint["observation_counts"])/identities["observed_instance_frames_in_range"]*100,
                   np.array(joint["structure_equal_weight"])*100]
    positive=np.concatenate([a[a>0] for a in distributions]);norm=LogNorm(positive.min(),positive.max())
    count_labels=["1","2","3","4","5–9","10–19","20–49","50–99","100–199","200–499","500–999","1000–1999","2000+"]
    for i,(values,title) in enumerate(zip(distributions,("Each observed instance-frame has equal weight","Each labeled identity has equal total weight"))):
        ax=figure.add_axes([.13,.57-i*.31,.70,.235])
        mesh=ax.pcolormesh(edges,np.arange(14),np.ma.masked_equal(values.T,0),norm=norm,cmap="viridis",rasterized=True)
        ax.set(yticks=np.arange(13)+.5,yticklabels=count_labels,xlabel="Median range of in-range returns (m)",ylabel="Return count",title=title)
        figure.colorbar(mesh,cax=figure.add_axes([.855,.57-i*.31,.022,.235]),label="Mass (%)")
    y=170
    y=text(f"有范围内回波的实例观测共 {identities['observed_instance_frames_in_range']:,} 次，其中 {identities['frames_1_to_4']:,} 次只有 1–4 个回波，占 {pct(identities['frames_1_to_4'],identities['observed_instance_frames_in_range'])}，涉及 {identities['identities_with_1_to_4']} 个标注身份。每个身份等权后，这一比例为 {100*identities['structure_equal_weight_fraction_1_to_4']:.2f}%。",y)
    text("两图分别归一化为 100%。第一图容易受到长期可见物体的影响；第二图用于观察身份覆盖差异，不规定训练抽样权重。图中分箱只用于显示，不是 V4 的正式覆盖组合、远距定义或匹配容差。",y)

    page("全部标注身份的可见区间")
    keys=[s["structure_id"] for s in structures]
    matrix=np.zeros((len(keys),449));indices={key:i for i,key in enumerate(keys)}
    for row in observations:
        matrix[indices[row["structure_id"]],int(row["frame"])]=int(row["returns_2p5_50m"])
    ax=figure.add_axes([.19,.19,.69,.65])
    mesh=ax.imshow(np.ma.masked_equal(np.log10(1+matrix),0),aspect="auto",interpolation="nearest",cmap="viridis",extent=(-.5,448.5,len(keys)-.5,-.5),rasterized=True)
    ax.set(yticks=np.arange(len(keys)),yticklabels=keys,xlabel="Frame",ylabel="Labeled identity")
    ax.tick_params(axis="y",labelsize=6.2,length=0)
    figure.colorbar(mesh,cax=figure.add_axes([.90,.19,.02,.65]),label="log10(1 + in-range returns)")
    y=130
    y=text(f"{identities['identities']} 个标注身份中，{identities['repeated_identities']} 个跨帧出现，{identities['moving_label_identities']} 个带运动语义标签。全部实际观测有 {identities['observed_instance_frames_all']:,} 次，其中 {identities['observed_instance_frames_in_range']:,} 次具有范围内回波；按连续有范围内回波分成 {identities['contiguous_segments']} 段。多个片段仍属于原标注身份，不作为新增独立结构。",y)
    text("空白只表示该帧没有该标注身份的范围内回波，不能区分距离过远、遮挡、漏标或物体离开。时间片段应据这些真实观测安排，不能用插值补出回波。",y)

    y=page("跨帧关联的证据与限制")
    y=text(f"同一实例数字在不同语义类别间重复使用。本次发现同时复用的数字包括 {'、'.join(map(str,identities['reused_numeric_ids']))}，因此仅使用实例数字会把不同对象混在一起。完整标识保留原始语义类别，运动与非运动标签也不自动合并。",y)
    y=text("对每个身份，将所有观测按原始位姿变换至世界坐标。相继两次观测分别计算双向最近点距离，再取两个方向中位数及 95 分位的较大值。另将按帧序排列的观测等分为三组，比较相邻两组汇集表面的距离。它们描述观测表面一致性，不是物体速度或身份真值。",y)
    jumps=sorted([r for r in observations if r.get("nn_world_median_m")],key=lambda r:float(r["nn_world_median_m"]),reverse=True)[:6]
    y=table(["标注身份","前帧→后帧","间隔帧数","双向中位距离"],
            [[r["structure_id"],f"{r['previous_observed_frame']}→{r['frame']}",r["gap_frames"],f"{float(r['nn_world_median_m']):.3f} 米"] for r in jumps],y,[140,130,90,140],size=9)
    y=text("较大的跨帧距离并不自动表明标注错误。运动、长时间无回波、物体不同部位被观测以及位姿误差，都可能造成变化。反过来，重复编号和较小距离也不能单独认证固定物体。本次保留全部观测与间隔，不设自动通过或剔除阈值。",y)
    y=text("没有实例编号的杆状物、树干和交通标志等，仍提供正常点及同帧上下文。若后续需要把其中某片区域作为跨帧参照，必须在落实时明确具体区域及关联依据；当前不能把未定义的区域曲线补写成已测得结果。",y)
    text("可见点的世界包围范围只覆盖已观测表面，不能当作完整物体尺寸、真实高度、接地状态或可放置空间。206 的本轮分析没有把任何候选位置认定为已经满足异常接地与无穿插要求。",y)

    page("此前三个参照片段的全段复核")
    for i,example in enumerate(summary["reference_examples"]):
        rows=[r for r in observations if r["structure_id"]==example["structure_id"] and example["start_frame"]<=int(r["frame"])<=example["end_frame"]]
        x=[int(r["frame"]) for r in rows]
        for j,(field,title) in enumerate((("returns_2p5_50m","Return count"),("range_valid_m_median","Median range (m)"))):
            ax=figure.add_axes([.12+j*.43,.64-i*.205,.34,.13])
            ax.plot(x,[float(r[field]) for r in rows],color="#356b88",lw=1)
            ax.set(xlabel="Frame",ylabel=title,title=example["structure_id"] if j==0 else "")
    y=155
    for example,category in zip(summary['reference_examples'],('汽车','自行车','摩托车')):
        y=text(f"{category} {example['structure_id']}，第 {example['start_frame']}–{example['end_frame']} 帧：{example['count']['min']:.0f}–{example['count']['max']:.0f} 个回波，距离中位数范围 {example['distance_m']['min']:.2f}–{example['distance_m']['max']:.2f} 米。",y,size=10)
    text(f"三个片段共 {sum(r['observed_frames'] for r in summary['reference_examples'])} 次观测、{sum(r['frames_1_to_4'] for r in summary['reference_examples'])} 次 1–4 点观测。摩托车例显示相近距离可以对应不同回波数量；原因尚不能归结为单一遮挡因素。三例不作为 V4 的唯一参照，也不构成总体代表性或已完成异常匹配的证据。",y,size=10)

    y=page("对两阶段数据构造的具体支持")
    for title,body in (
        ("基础数据", "206 提供完整正常背景、可用的位姿轨迹及多种语义结构。点分布明显偏向近处，而不同正常物体的可见区间和回波数量差异较大。首轮合成需要在整个数据池检查几何、位置与实际观测覆盖，不能用多次重复同一背景的累计点数替代多样性。"),
        ("针对性数据", "本次为全部标注身份保留逐帧数量、距离、观测间隔和世界几何证据，并给出连续可见片段。后续可据具体可信参照选择异常几何、尺寸与世界位置，再由原始射线自然形成观测；本次没有确定匹配容差，也没有逐帧增删点来拟合曲线。"),
        ("少回波监督", f"{identities['identities_with_1_to_4']} 个标注身份出现过范围内 1–4 点观测，说明少回波正常观测实际存在。它们属于正常参照统计，不受异常整帧至少 5 点的门槛排除。合成之后须先计算所有物体的联合遮挡，再应用整帧异常计数规则。"),
        ("尚不能从原始序列给出的结论", "没有植入异常，便没有异常引起的背景变化真值，也不能判断弱背景变化异常是否覆盖充分。缺少完整物体几何和具体放置，就不能给出低矮异常覆盖数或合法放置数量。没有模型预测，便不能计算 AP、AUROC、FPR95，也不能确认对 COVAL 的性能增益。"),
        ("独立性", "206 始终只是一条原始背景序列。其不同帧、同一物体的多个片段以及未来的多个合成版本都有关联。本次没有读取 STU 真实异常评价集来构造训练分布，所得统计不构成独立泛化证据。"),
    ):
        y=text(title,y,size=12)
        y=text(body,y)
    text("下一项能改变数据构造判断的动作，是在实际开展异常放置时，依据明确的正常参照和覆盖目标，核查自然生成的数量—距离变化、接地、穿插与联合遮挡。相关未定数值在落实时再逐项确认。",y)

    y=page("复算入口、统计单位与交付文件")
    y=text("原始输入：/home/jasongao/Data/STU/train/206。射线参考：/home/jasongao/Study/AJAE/assets/rays.npz。脚本独立读取数据，没有导入旧训练代码、旧合成池或旧实验成绩。",y)
    y=table(["文件","统计单位与用途"],[
        ["frames.csv","449 行；完整扫描、标签、距离、强度、射线与位姿统计"],
        ["labels.csv","原始语义类别；回波和实例身份覆盖"],
        ["observations.csv",f"{len(observations):,} 行；{len(structures)} 个标注身份 × 449 帧，含零观测"],
        ["structures.csv",f"{len(structures)} 行；每个标注身份的观测与几何汇总"],
        ["segments.csv",f"{identities['contiguous_segments']} 行；连续具有范围内回波的自然片段"],
        ["summary.json","总体统计、分箱数据、定义与执行信息"],
    ],y,[140,365],height=24,size=9)
    y=text("距离：原始 float32 坐标的三维欧氏距离，含 2.5 与 50 米边界。观测表同时保存全部回波距离与范围内距离的最小值、中位数和最大值；联合图用范围内距离中位数。中位数使用 np.median，其中 float32 距离保持原精度；其他分位数使用线性插值。观测之间存在相关性，本文不把分位数或相关系数解释为独立样本推断。",y)
    y=text("数量：按实际文件回波计数。零回波的距离、强度和几何字段为空，不能按数值 0 使用。连续片段由相邻帧是否至少有一个范围内回波直接确定，没有附加最短长度。标注身份、物理物体数、片段数和实例观测次数分别报告。",y)
    y=text("完整扫描的坐标、标签与文件长度已经逐帧核查；总体点数、距离分区、标签分区、实例观测与片段计数相互核对。图表使用同一批保存的统计表，不另行抽样计算。中文逐字符使用宋体，英文、数字与图轴使用 Times New Roman。",y)
    y=text("运行环境使用已有 AJAE Python 环境，在 AJAE-v4 根目录运行：\nPYTHONDONTWRITEBYTECODE=1 /home/jasongao/Study/AJAE/.venv/bin/python src/analyze.py --workers 6\n仅重建报告：在上述命令后加 --report-only。",y,size=9.5)
    text("标签和评价依据：STU 官方 compute_point_level_ood.py 与 Mask4Former3D/conf/semantic-kitti.yaml。报告中的数字均来自本次 206 实际读取；没有模拟数字或预测成绩。",y)
    pdf.savefig(figure);plt.close(figure);pdf.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--rays",type=Path,default=Path("/home/jasongao/Study/AJAE/assets/rays.npz"))
    parser.add_argument("--output",type=Path,default=Path("results/206"))
    parser.add_argument("--workers",type=int,required=True)
    parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if not args.report_only:
        analyze(args)
    report(args.output)


if __name__ == "__main__":
    main()
