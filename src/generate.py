"""Freeze fixed physical worlds as complete sequences of independent scans."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy.stats import spearmanr

from .data import FrozenDataset, FrozenFrame, _atomic_json, host_disk, source_identity
from .profile import GROUND, ground_relation, visible_shape
from .protocol import load_protocol
from .scene import STUSequence
from .coverage import conditions
from .supervision import ScanGeometry, surface_targets, surface_probe
from .render import (
    MaterialSpec,
    ObservedObstacleIndex,
    QualifiedSupportPool,
    ShapeSpec,
    WorldSpec,
    calibrated_ray_grid,
    calibrate_sensor,
    load_sensor_calibration,
    observed_normal_collision,
    place_object,
    render_frames,
    save_sensor_calibration,
    shape_geometry,
    qualify_grounding,
    _frame_trace_context,
    _object_hits,
)


# Fixed coarse observation bins; they never encode a particular validation object.
INF = float("inf")
BINS = {
    "count": (0, 1, 5, 20, 100, 500, INF),
    "range": (0, 2.5, 10, 20, 35, 50, 80, 120, INF),
    **{
        k: (0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, INF)
        for k in ("length", "width", "height")
    },
    **{
        k: (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.000001)
        for k in ("linearity", "planarity", "scattering")
    },
    "same_instance_neighbor_distance": (0, 0.05, 0.1, 0.25, 0.5, 1, 2, INF),
    "background_neighbors": (0, 1, 3, 10, 30, 100, 300, INF),
    "intensity_contrast": (-INF, -0.5, -0.2, -0.05, 0, 0.05, 0.2, 0.5, INF),
    "nearest_road_fraction": (0, 0.5, 0.9, 1.000001),
    "ground_height_median": (-INF, 0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, INF),
    **{
        k: (0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.000001)
        for k in ("visible_fraction", "eligible_fraction")
    },
    "longest_visible_run": (0, 10, 25, 50, 100, 200, 400, 800, INF),
    "range_span": (0, 10, 20, 35, 50, 80, 120, INF),
    "range_step": (0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, INF),
    "count_step": (0, 0.05, 0.1, 0.25, 0.5, 1, 2, INF),
    "height_step": (0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, INF),
    "range_count_correlation": (-1.000001, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 1.000001),
}


def finite_median(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def histogram(values, edges):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    counts = np.r_[np.histogram(values[finite], edges)[0], np.count_nonzero(~finite)]
    if counts.sum() != len(values):
        raise ValueError("observation lies outside its declared histogram support")
    return counts / max(1, len(values))


def trajectory(rows):
    count = np.array([r["count"] for r in rows])
    ranges = np.array([r.get("range") for r in rows], float)
    heights = np.array([r.get("ground_height_median") for r in rows], float)
    visible = count > 0
    runs = np.diff(np.flatnonzero(np.diff(np.r_[False, visible, False])))
    adjacent = visible[1:] & visible[:-1]
    observed = ranges[np.isfinite(ranges)]
    correlation = None
    if visible.sum() >= 5 and np.ptp(count[visible]) and np.ptp(ranges[visible]):
        correlation = float(spearmanr(ranges[visible], count[visible]).statistic)
    return dict(
        visible_fraction=float(visible.mean()),
        eligible_fraction=float(np.mean([r["in_range"] >= 5 for r in rows])),
        longest_visible_run=int(max(runs[::2], default=0)),
        range_span=float(np.ptp(np.quantile(observed, [0.05, 0.95])))
        if len(observed)
        else None,
        range_step=finite_median(np.abs(np.diff(ranges))[adjacent]),
        count_step=finite_median(np.abs(np.diff(np.log1p(count)))[adjacent]),
        height_step=finite_median(np.abs(np.diff(heights))[adjacent]),
        range_count_correlation=correlation,
        count_p95=float(np.quantile(count[visible], 0.95)) if visible.any() else 0.0,
        visible_frames=int(visible.sum()),
        eligible_frames=int(np.count_nonzero([r["in_range"] >= 5 for r in rows])),
    )


def distributions(rows):
    summary = trajectory(rows)
    hist = {}
    for key, edges in BINS.items():
        values = [summary[key]] if key in summary else [r.get(key) for r in rows]
        hist[key] = histogram(values, edges).tolist()
    # Count and range are coupled by visibility; retain their joint distribution.
    counts = np.array([r["count"] for r in rows])
    ranges = np.array([r.get("range") for r in rows], float)
    ci = np.searchsorted(BINS["count"], counts, side="right") - 1
    ri = np.searchsorted(BINS["range"], ranges, side="right") - 1
    ri[~np.isfinite(ranges)] = len(BINS["range"]) - 1
    joint = np.bincount(
        ci * len(BINS["range"]) + ri,
        minlength=(len(BINS["count"]) - 1) * len(BINS["range"]),
    )
    hist["count_range"] = (joint / len(rows)).tolist()
    return summary, hist


def observation(rendered):
    source = rendered.source
    ids = np.flatnonzero(rendered.inserted_mask)
    row = dict(frame=source.frame_id, count=len(ids), in_range=0)
    if not len(ids):
        return row
    raw = source.xyzi[ids]
    ranges = np.linalg.norm(raw[:, :3], axis=1)
    xyz = raw[:, :3].astype(np.float64)
    row.update(
        in_range=int(np.count_nonzero((ranges >= 2.5) & (ranges <= 50))),
        range=float(np.median(ranges)),
    )
    row.update({key: value for key, value in visible_shape(xyz).items() if key in BINS})
    row["same_instance_neighbor_distance"] = (
        float(np.median(cKDTree(xyz).query(xyz, k=2)[0][:, 1]))
        if len(ids) >= 2
        else None
    )
    slots = source.real_slots
    semantic = source.labels.semantic[slots]
    normal = slots[(semantic != 0) & (semantic != 2)]
    if len(normal):
        tree = cKDTree(source.xyzi[normal, :3])
        nearest = tree.query(xyz)[1]
        row["nearest_road_fraction"] = float(
            np.mean(source.labels.semantic[normal[nearest]] == 40)
        )
        neighborhoods = tree.query_ball_point(xyz, 0.5)
        row["background_neighbors"] = float(np.median([len(n) for n in neighborhoods]))
        contrast = [
            float(raw[i, 3]) - source.xyzi[normal[n], 3].astype(float).mean()
            for i, n in enumerate(neighborhoods)
            if len(n) >= 3
        ]
        row["intensity_contrast"] = finite_median(contrast)
    ground = source.xyzi[slots[np.isin(semantic, GROUND)], :3].astype(np.float64)
    relation = ground_relation(
        xyz, ground, cKDTree(ground[:, :2]) if len(ground) else None
    )
    row["ground_height_median"] = relation["ground_height_median"]
    row["ground_status"] = relation["ground_status"]
    return row


def sample_support(sequence, rng, config, footprint_radius, background, rejections, references=None):
    poses = np.stack([sequence.lidar_pose(f) for f in sequence.frame_ids])
    accepted = 0
    for attempt in range(config["maximum_support_attempts"]):
        reference = None
        if references is not None:
            available = [r for r in references if r["slots"]]
            if not available:
                rejections["no_native_normal_reference"] += 1
                return
            record = available[int(rng.integers(len(available)))]
            frame = sequence[record["frame"]]
            if source_identity(frame) != record["source_identity"]:
                raise ValueError("native reference source changed")
            ids = np.asarray(record["slots"], np.int32)
            point = frame.xyzi[int(rng.choice(ids)), :3]
            cluster = np.linalg.norm(frame.xyzi[ids, :3] - point, axis=1) <= config["reference_cluster_radius_m"]
            if cluster.sum() < config["minimum_reference_positions"]:
                rejections["normal_reference_cluster_too_small"] += 1
                continue
            reference = dict(frame=frame.frame_id, source_identity=record["source_identity"],
                             slots=ids[cluster].tolist(), offsets=np.asarray(record["offsets"])[cluster].tolist())
        else:
            frame = sequence[int(rng.integers(len(sequence)))]
        ground_slots = frame.real_slots[np.isin(frame.labels.semantic[frame.real_slots], config["semantics"])]
        xyz = frame.xyzi[ground_slots, :3].astype(np.float64)
        distance = np.linalg.norm(xyz, axis=1)
        choices = np.flatnonzero((distance >= config["proposal_range_m"][0]) & (distance <= config["proposal_range_m"][1]))
        if reference is not None:
            separation = np.linalg.norm(xyz[choices, :2] - point[:2], axis=1)
            lo, hi = config["reference_clearance_m"]
            choices = choices[(separation >= footprint_radius + lo) & (separation <= footprint_radius + hi)]
        if not len(choices):
            rejections["no_ground_in_proposal_range"] += 1
            continue
        selected = int(rng.choice(choices))
        center = xyz[selected, :2]
        points = xyz[np.linalg.norm(xyz[:, :2] - center, axis=1) <= config["plane_radius_m"]]
        if len(points) < config["minimum_ground_points"]:
            rejections["insufficient_ground"] += 1
            continue
        design = np.column_stack((points[:, :2] - center, np.ones(len(points))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
        residual = points[:, 2] - design @ coefficients
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual) <= max(0.05, 3 * mad)
        if rank != 3 or keep.sum() < config["minimum_ground_points"]:
            rejections["reference_rank_or_inlier_support"] += 1
            continue
        coefficients, _, rank, _ = np.linalg.lstsq(design[keep], points[keep, 2], rcond=None)
        rmse = float(np.sqrt(np.mean((points[keep, 2] - design[keep] @ coefficients) ** 2)))
        try:
            hull = ConvexHull(design[keep, :2])
        except QhullError:
            rejections["support_hull"] += 1
            continue
        if (rank != 3 or rmse > config["maximum_plane_rmse_m"]
                or np.linalg.norm(coefficients[:2]) > np.tan(np.deg2rad(config["maximum_slope_degrees"]))
                or np.linalg.eigvalsh(np.cov(design[keep, :2], rowvar=False))[0] < config["minimum_xy_eigenvalue"]
                or np.any(hull.equations[:, -1] > -footprint_radius)):
            rejections["plane_quality_or_full_footprint"] += 1
            continue
        rotation, translation = frame.lidar_pose[:3, :3], frame.lidar_pose[:3, 3]
        anchor = np.r_[center, coefficients[2]] @ rotation.T + translation
        trajectory_range = np.linalg.norm(poses[:, :3, 3] - anchor, axis=1)
        if not config["minimum_trajectory_clearance_m"] <= trajectory_range.min() <= config["maximum_closest_trajectory_distance_m"]:
            rejections["trajectory_clearance"] += 1
            continue
        real = frame.real_slots
        nearby = real[np.linalg.norm(frame.xyzi[real, :3] - np.r_[center, coefficients[2]], axis=1) <= config["background_radius_m"]]
        semantic = frame.labels.semantic[nearby]
        known = frame.labels.semantic_target[nearby] != 255
        ground_ids = set(semantic[known & np.isin(semantic, config["semantics"])].tolist())
        structures = int(np.sum(known & ~np.isin(semantic, GROUND)))
        category = "structured" if len(ground_ids) >= 2 or structures >= 3 else "ground"
        if background != "any" and category != background:
            rejections["background_stratum"] += 1
            continue
        normal = rotation @ np.r_[-coefficients[:2], 1.0]
        normal /= np.linalg.norm(normal)
        slot = int(ground_slots[selected])
        pool = QualifiedSupportPool(np.array([0]), np.array([frame.labels.semantic[slot]]),
                                    np.array([frame.frame_id]), np.array([slot]), np.array([distance[selected]]),
                                    np.array([0], np.uint64), anchor[None], normal[None],
                                    np.array([-normal @ anchor]), sequence.spec.sequence_id)
        yield pool, dict(plane_rmse_m=rmse, plane_support=int(keep.sum()), proposal_attempt=attempt,
                         closest_trajectory_distance_m=float(trajectory_range.min()), background=category,
                         background_semantic_counts={str(k): int(v) for k, v in Counter(semantic[known].tolist()).items()},
                         normal_reference=reference)
        accepted += 1
        if accepted == config["support_candidates"]:
            break


def nearby_obstacles(source, item):
    center = (
        np.asarray(item.translation_world_m) - source.lidar_pose[:3, 3]
    ) @ source.lidar_pose[:3, :3]
    slots = source.real_slots
    xyz = source.xyzi[slots, :3].astype(np.float64)
    semantic = source.labels.semantic[slots]
    # A bounding sphere is conservative; the existing exact signed-distance test decides collision.
    chosen = (
        (semantic != 0)
        & ~np.isin(semantic, GROUND)
        & (np.linalg.norm(xyz - center, axis=1) <= item.bounding_radius_m + 0.05)
    )
    slots, xyz = slots[chosen], xyz[chosen]
    if not len(slots):
        return None
    world = xyz @ source.lidar_pose[:3, :3].T + source.lidar_pose[:3, 3]
    identities = (np.uint64(source.frame_id) << np.uint64(32)) | slots.astype(np.uint64)
    return ObservedObstacleIndex(world, identities)


def make_shape(rng, profile, config):
    dimensions = np.exp([rng.uniform(*np.log(profile[key])) for key in ("length_m", "width_m", "height_m")])
    family = profile["shape"]
    if family == "single":
        scales, offsets = np.array([[.5, .5, .5]]), np.zeros((1, 3))
    elif family == "step":
        # The second part is taller: the first primitive cannot define object height.
        scales = np.array([[.45, .30, .30], [.27, .30, .50]])
        offsets = np.array([[-.15, 0, -.20], [.25, 0, 0]])
    elif family == "elbow":
        scales = np.array([[.45, .20, .5], [.20, .35, .5], [.30, .20, .5]])
        offsets = np.array([[0, 0, 0], [.35, .20, 0], [-.10, .35, 0]])
    elif family == "bridge":
        scales = np.array([[.22, .35, .40], [.22, .35, .40], [.50, .35, .28]])
        offsets = np.array([[-.30, 0, -.10], [.30, 0, -.10], [0, 0, .22]])
    else:
        raise ValueError("unknown declared shape family")
    lower, upper = (offsets - scales).min(axis=0), (offsets + scales).max(axis=0)
    factors = dimensions / (upper - lower)
    offsets = (offsets - (upper + lower) / 2) * factors
    scales *= factors
    exponent_range = config["exponents" if family == "single" else "composed_exponents"]
    shape = ShapeSpec(tuple(map(tuple, scales)), tuple(map(tuple, offsets)),
                      tuple(map(tuple, rng.uniform(*exponent_range, (len(scales), 2)))),
                      (0.,) * len(scales), ("union",) * len(scales))
    geometry = shape_geometry(shape)
    if not np.allclose([geometry[k] for k in ("length_m", "width_m", "height_m")], dimensions, atol=3e-6, rtol=0):
        raise ValueError("complete shape bounds differ from requested dimensions")
    return shape, geometry


def ray_observation(source, world, grid, sensor, geometry):
    """Use the renderer's exact surface competition; signal draws never enter placement scores."""
    item = world.objects[0]
    slots, directions_sensor, directions, origins_sensor, origins, native = _frame_trace_context(source, grid)
    center, rotation = np.asarray(item.translation_world_m), np.asarray(item.rotation_world_from_local)
    origin, direction = (origins - center) @ rotation, directions @ rotation
    lower, upper = np.asarray(geometry["lower_local_m"]), np.asarray(geometry["upper_local_m"])
    parallel = np.abs(direction) < 1e-15
    safe = np.where(parallel, 1., direction)
    a, b = (lower - origin) / safe, (upper - origin) / safe
    a[parallel], b[parallel] = -np.inf, np.inf
    enter, leave = np.minimum(a, b).max(axis=1), np.maximum(a, b).min(axis=1)
    box = ((leave >= np.maximum(enter, 0)) & (enter < native - world.tie_tolerance_m)
           & ~np.any(parallel & ((origin < lower) | (origin > upper)), axis=1))
    competition = _object_hits(origins, directions, world, grid, sensor, source.frame_id, canonical_ray_slots=slots)
    surface = np.isfinite(competition.distance_m) & (competition.distance_m < native - world.tie_tolerance_m)
    returned = surface & competition.returned
    if np.any(surface & ~box):
        raise ValueError("true foreground surface lies outside the complete object bounds")
    unique = lambda mask: len(np.unique(grid.canonical_ray_by_slot[slots[mask]]))
    xyz = (origins_sensor[returned] + competition.distance_m[returned, None] * directions_sensor[returned]).astype(np.float32)
    ranges = np.linalg.norm(xyz, axis=1)
    return dict(frame=source.frame_id,
                range_m=float(np.linalg.norm(center - source.lidar_pose[:3, 3])),
                available_box_rays=unique(box), foreground_surface_rays=unique(surface),
                final_anomaly_rays=unique(returned), final_anomaly_slots=int(returned.sum()),
                in_range_anomaly_slots=int(np.sum((ranges >= 2.5) & (ranges <= 50)))), surface


def far_ray_opportunities(sequence, world, grid, sensor, geometry, check_ranges):
    center = np.asarray(world.objects[0].translation_world_m)
    distances = np.array([np.linalg.norm(center - sequence.lidar_pose(f)[:3, 3]) for f in sequence.frame_ids])
    far = np.flatnonzero((distances >= 35) & (distances <= 50))
    if not len(far):
        return []
    frames = sorted({int(far[np.argmin(np.abs(distances[far] - value))]) for value in check_ranges})
    return [ray_observation(sequence[frame], world, grid, sensor, geometry)[0] for frame in frames]


def make_world(sequence, seed, config, profile, grid, sensor, references=None):
    rng = np.random.default_rng(seed)
    shape, geometry = make_shape(rng, profile, config["shape"])
    grounding = qualify_grounding(shape)
    if not grounding.passed:
        raise ValueError("shape_grounding_unreliable")
    material = MaterialSpec(float(rng.uniform(*config["shape"]["material_quantile"])),
                            float(rng.uniform(*config["shape"]["material_roughness"])))
    rejections, candidates = Counter(), []
    for pool, support in sample_support(sequence, rng, config["placement"], geometry["footprint_radius_m"],
                                       profile["background"], rejections, references):
        frame = sequence[int(pool.frames[0])]
        slots = frame.real_slots
        selected = slots[(frame.labels.semantic[slots] != 0) & ~np.isin(frame.labels.semantic[slots], GROUND)]
        obstacles = ObservedObstacleIndex(frame.xyzi[selected, :3].astype(float) @ frame.lidar_pose[:3, :3].T + frame.lidar_pose[:3, 3],
                                         (np.uint64(frame.frame_id) << np.uint64(32)) | selected.astype(np.uint64))
        yaw = float(rng.uniform(-np.pi, np.pi))
        if profile["view"] == "low_far":
            poses = np.array([sequence.lidar_pose(f)[:3, 3] for f in sequence.frame_ids])
            direction = pool.anchors_world_m[0] - poses[np.argmin(np.abs(np.linalg.norm(poses - pool.anchors_world_m[0], axis=1) - 42))]
            yaw = float(np.arctan2(direction[1], direction[0]) + np.pi / 2)
        for yaw_offset in config["proposals"]["yaw_offsets_rad"]:
            angle = yaw + yaw_offset
            try:
                item, placement = place_object(shape, material, pool, obstacles, object_id=1, label="anomaly-proxy",
                                               proposal_namespace=f"single-frame-content/{seed}", proposal_stream=0,
                                               yaw_rad=angle, material_seed=seed, yaw_seed=seed, shape_seed=seed,
                                               proposal_rows=[0], maximum_candidates=1, grounding_eligibility=grounding)
            except ValueError:
                rejections["local_collision_or_grounding"] += 1
                continue
            world = WorldSpec(seed, sequence.spec.sequence_id, (item,))
            if profile["view"] == "low_far":
                observations = far_ray_opportunities(sequence, world, grid, sensor, geometry, config["proposals"]["far_check_ranges_m"])
            else:
                observed, foreground = ray_observation(frame, world, grid, sensor, geometry)
                if np.any(foreground[support["normal_reference"]["slots"]]):
                    rejections["native_reference_would_be_occluded"] += 1
                    continue
                observations = [observed]
            hits = [r["foreground_surface_rays"] for r in observations]
            # Geometry selects supports; final counts only verify the same fixed draw after full rendering.
            score = (sum(n >= 5 for n in hits), sum(min(n, 12) for n in hits)) if hits else (-1, 0)
            candidates.append((score, item, dict(**support, placement=clean_json(placement.to_dict()),
                                                support_plane=dict(anchor_world_m=pool.anchors_world_m[0].tolist(),
                                                                   normal_world=pool.normals_world[0].tolist(), offset=float(pool.offsets[0])),
                                                yaw_rad=angle, ray_observations=observations)))
    if not candidates:
        raise ValueError("no_physical_support:" + json.dumps(dict(rejections), sort_keys=True))
    best = max(range(len(candidates)), key=lambda i: (candidates[i][0], -i))
    _, item, chosen = candidates[best]
    return WorldSpec(seed, sequence.spec.sequence_id, (item,)), dict(
        **chosen, geometry=geometry, shape_family=profile["shape"], candidate_category=profile["name"],
        proposal_intent=profile["view"], support_rejections=dict(rejections),
        support_proposals=[dict(frame=c[2]["placement"]["support_frame"], slot=c[2]["placement"]["support_slot"],
                                yaw_rad=c[2]["yaw_rad"], ray_observations=c[2]["ray_observations"], chosen=i == best)
                           for i, c in enumerate(candidates)],
        connectivity=shape.continuous_connectivity_certificate().state)


def scan_normal_references(data_root, frame, config):
    from numba import set_num_threads
    set_num_threads(1)
    source = STUSequence.open(data_root, protocol=load_protocol(), partition="train", sequence_id=201,
                              label_mode="required")[frame]
    empty = np.zeros(source.slot_count, bool)
    identity = source_identity(source)
    sample = FrozenFrame(source, identity, empty, empty)
    slots = source.real_slots
    geometry = ScanGeometry(source.xyzi[slots], slots, config["supervision"]["common"]["sampling_scale"])
    parameters = config["supervision"]["C3"]["parameters"]
    target = surface_targets(source, sample, [geometry], [sample.anomaly_target[slots]], [], parameters)[parameters["minimum_visible_support_points"]][0]
    ranges = np.linalg.norm(source.xyzi[slots, :3], axis=1)
    lo, hi = config["placement"]["proposal_range_m"]
    selected = ((sample.anomaly_target[slots] == 0) & target["surface_valid"]
                & (target["surface_offset_z"] >= -.2) & (target["surface_offset_z"] <= -.05)
                & (ranges >= lo) & (ranges <= hi))
    rows = np.sort(geometry.first[selected[geometry.first]])
    return dict(frame=frame, source_identity=identity, slots=slots[rows].tolist(),
                offsets=target["surface_offset_z"][rows].tolist())


def check_normal_reference(sample, original, reference, parameters, minimum):
    """A native witness must survive unchanged and remain locally visible with the anomaly."""
    slots = np.asarray(reference["slots"], np.int32)
    if source_identity(original) != reference["source_identity"]:
        raise ValueError("native reference source changed after placement")
    unchanged = (np.array_equal(sample.source.xyzi[slots], original.xyzi[slots])
                 and np.array_equal(sample.source.labels.packed[slots], original.labels.packed[slots])
                 and np.all(sample.anomaly_target[slots] == 0))
    anomalies = sample.source.xyzi[sample.inserted_mask, :3]
    ranges = np.linalg.norm(anomalies, axis=1)
    eligible = int(np.sum((ranges >= 2.5) & (ranges <= 50)))
    probes = []
    if unchanged and len(anomalies):
        near = cKDTree(anomalies.astype(float)).query(original.xyzi[slots, :3])[0] <= 2
        for i in np.flatnonzero(near):
            point = original.xyzi[slots[i], :3].astype(float)
            probe = surface_probe(point, original, sample, sample.source.real_slots, parameters)
            if probe["valid"] and -.2 <= probe["offset_z_m"] <= -.05:
                if abs(probe["offset_z_m"] - reference["offsets"][i]) > 1e-6:
                    raise ValueError("the same original normal surface target changed")
                probes.append(dict(source_slot=int(slots[i]), semantic=int(original.labels.semantic[slots[i]]), **probe))
    return dict(frame=original.frame_id, protected_positions=len(slots), unchanged=bool(unchanged),
                eligible_anomaly_returns=eligible, adjacent_valid_positions=len(probes), probes=probes,
                achieved=bool(unchanged and eligible >= 5 and len(probes) >= minimum))


def content_summary(rows, placement):
    world = placement["geometry"]
    result = {key: dict(frames=0, anomaly_returns=0) for key in conditions(world, dict(count=0, in_range=0))}
    for row in rows:
        flags = conditions(world, row)
        for key, active in flags.items():
            if active:
                result[key]["frames"] += 1
                result[key]["anomaly_returns"] += row["count"]
    return result


def generate_candidate(data_root, output, split, index, config, identity):
    started = time.perf_counter()
    split_config = config["dataset"]["splits"][split]
    sequence_id = split_config["source_sequence"]
    seed = int(
        np.random.SeedSequence([config["seed"], config["proposals"]["seed_namespace"], sequence_id, index]).generate_state(1)[
            0
        ]
    )
    directory = Path(output) / split / f"supplement_{index:03d}"
    sequence = STUSequence.open(
        data_root,
        protocol=load_protocol(),
        partition="train",
        sequence_id=sequence_id,
        label_mode="required",
    )
    if len(sequence) != split_config["frames_per_world"]:
        raise ValueError("normal source length changed")
    report = dict(
        index=index,
        seed=seed,
        source_sequence=sequence_id,
        configuration_identity=identity,
    )
    try:
        grid, sensor = load_sensor_calibration(Path(output) / "calibration.pt")
        profile = config["proposals"]["categories"][split][index // config["proposals"]["repeats_per_source"]]
        report["candidate_category"] = profile["name"]
        references = (json.loads((Path(output) / "normal_reference.json").read_text())["scans"]
                      if profile["view"] == "normal_reference" else None)
        world, placement = make_world(sequence, seed, config, profile, grid, sensor, references)
    except ValueError as error:
        report.update(status="rejected", reason=str(error), frames=[],
                      seconds=time.perf_counter() - started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        _atomic_json(directory / "manifest.json", report)
        return report
    definition = dict(world=world.to_dict(), generation=placement)
    world_path = directory / "world.json"
    if world_path.exists() and json.loads(world_path.read_text()) != definition:
        raise ValueError(
            "interrupted world definition differs from the current fixed world"
        )
    _atomic_json(world_path, definition)
    rows, collision = [], None
    local_bounds = world.objects[0].shape.tight_continuous_outer_bounds(
        z_slabs=256, safety_margin_m=1e-6
    )
    written_bytes = sum(p.stat().st_size for p in (directory / "frames").glob("*.npz"))

    def record_sample(sample):
        row = observation(sample)
        row.update(
            occluded=int(sample.occluded_original_mask.sum()),
            slots=sample.source.slot_count,
            real_returns=sample.source.real_count,
        )
        rows.append(row)

    def sources():
        nonlocal collision
        for source in sequence:
            obstacles = nearby_obstacles(source, world.objects[0])
            if obstacles is not None:
                hit, minimum, _ = observed_normal_collision(
                    world.objects[0],
                    obstacles,
                    penetration_m=config["placement"]["deep_penetration_m"],
                    local_bounds=local_bounds,
                )
                if hit:
                    if collision is None:
                        collision = dict(frame=source.frame_id, minimum_sdf_m=minimum)
            path = directory / "frames" / f"{source.frame_id:06d}.npz"
            if path.exists():
                record_sample(FrozenFrame.load(path, source, world.identity))
            else:
                yield source

    for rendered in render_frames(sources(), world, grid, sensor):
        original = sequence[rendered.frame_id]
        sample = FrozenFrame(
            rendered.source,
            world.identity,
            rendered.inserted_mask,
            rendered.occluded_original_mask,
        )
        path = directory / "frames" / f"{rendered.frame_id:06d}.npz"
        sample.save(path, original)
        written_bytes += path.stat().st_size
        if written_bytes > 256 * 1024**2:
            raise OSError("candidate reached the bounded 256 MiB working allocation")
        restored = FrozenFrame.load(path, original, world.identity)
        for actual, expected in (
            (restored.source.xyzi, rendered.xyzi),
            (restored.source.labels.packed, rendered.packed_labels),
            (
                restored.source.labels.semantic_target,
                rendered.source.labels.semantic_target,
            ),
            (restored.source.real_slots, rendered.visible_slots),
            (restored.inserted_mask, rendered.inserted_mask),
            (restored.occluded_original_mask, rendered.occluded_original_mask),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(
                    "saved single-frame reconstruction differs from physical render"
                )
        record_sample(sample)
    reason = "observed_normal_deep_penetration" if collision else None
    stats, hist = distributions(rows) if rows else ({}, {})
    if len(rows) != len(sequence):
        raise ValueError("world stopped before its final source frame")
    if collision is None:
        if (
            stats["eligible_frames"]
            < config["qualification"]["minimum_eligible_frames"]
        ):
            reason = "no_eligible_anomaly_observation"
    content = content_summary(rows, placement)
    for probe in placement["ray_observations"]:
        row = rows[probe["frame"]]
        if row["count"] != probe["final_anomaly_slots"] or row["in_range"] != probe["in_range_anomaly_slots"]:
            raise ValueError("full rendering changed the fixed proposal signal draw")
    purpose = dict(name=profile["view"], achieved=content["low_far_eligible"]["frames"] > 0)
    if profile["view"] == "normal_reference":
        reference = placement["normal_reference"]
        original = sequence[reference["frame"]]
        sample = FrozenFrame.load(directory / "frames" / f"{original.frame_id:06d}.npz", original, world.identity)
        purpose.update(check_normal_reference(sample, original, reference, config["supervision"]["C3"]["parameters"],
                                              config["placement"]["minimum_reference_positions"]))
    purpose["achieved"] &= placement["geometry"]["height_m"] <= .2
    physical_reason = reason
    if reason is None and not purpose["achieved"]:
        reason = "declared_content_purpose_not_observed"
    report.update(
        status="qualified" if reason is None else "rejected",
        reason=reason,
        collision=collision,
        world_identity=world.identity,
        frames=rows,
        trajectory=stats,
        histograms=hist,
        content=content,
        purpose=purpose,
        physical_reason=physical_reason,
        content_check_frames=[purpose["frame"]] if "frame" in purpose else [],
        shape_family=placement["shape_family"],
        background=placement["background"],
        geometry=placement["geometry"],
        seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        reconstruction="all_rendered_frames_equal_xyzi_packed_labels_slots_and_masks",
    )
    _atomic_json(directory / "manifest.json", report)
    return report


def select_worlds(base, reports, split):
    """Retain the complete base in its original order; append successful supplements only."""
    accepted = [r for r in sorted(reports, key=lambda r: r["index"])
                if r["status"] == "qualified" and r["purpose"]["achieved"]]
    worlds = [dict(entry) for entry in base]
    worlds.extend(dict(path=f"{split}/supplement_{r['index']:03d}", world_identity=r["world_identity"],
                       candidate_index=r["index"], origin="supplement") for r in accepted)
    return worlds, dict(base_worlds=len(base), accepted_supplements=[r["index"] for r in accepted],
                        rejected_supplements={str(r["index"]): r["reason"] for r in reports if r not in accepted},
                        rule="all_base_worlds_in_original_order_then_physical_and_purpose_valid_supplements")


def prepare_calibration(data_root, output, config):
    calibration = config["calibration"]
    sequence = STUSequence.open(
        data_root,
        protocol=load_protocol(),
        partition="train",
        sequence_id=206,
        label_mode="required",
    )
    identities = [source_identity(source) for source in sequence]
    signature = dict(
        configuration=calibration,
        source_frames=identities,
        rays_sha256=hashlib.sha256(Path(calibration["rays"]).read_bytes()).hexdigest(),
    )
    identity = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()
    path = output / "calibration.pt"
    reuse = Path(config["calibration_source"]) if config.get("calibration_source") else None
    if not path.exists() and reuse is not None:
        _, sensor = load_sensor_calibration(reuse)
        if json.loads(dict(sensor.provenance).get("input_identity", '""')) != identity:
            raise ValueError("reused calibration does not match the complete normal source inputs")
        shutil.copyfile(reuse, path)
    if path.exists():
        _, sensor = load_sensor_calibration(path)
        if json.loads(dict(sensor.provenance).get("input_identity", '""')) != identity:
            raise ValueError(
                "sensor calibration does not belong to these current normal inputs"
            )
    else:
        grid = calibrated_ray_grid(calibration["rays"])
        start = time.perf_counter()
        sensor = calibrate_sensor(
            sequence,
            grid,
            source_sequence_id=206,
            provenance={
                "input_identity": identity,
                "source_frame_identities": identities,
            },
            **{
                k: calibration[k]
                for k in (
                    "seed",
                    "quantile_count",
                    "maximum_samples_per_cell",
                    "minimum_cell_count",
                )
            },
        )
        save_sensor_calibration(path, grid, sensor)
        print(
            json.dumps(
                dict(
                    event="calibrated",
                    frames=len(sequence),
                    seconds=time.perf_counter() - start,
                )
            ),
            flush=True,
        )
    return dict(
        input_identity=identity,
        frame_count=len(sequence),
        intensity_min=sensor.intensity_min,
        intensity_max=sensor.intensity_max,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--config", type=Path, default=Path("protocol/v1.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        parser.error("workers must fit the CPU affinity")
    config = json.loads(args.config.read_text())
    output = args.output or Path(config["dataset"]["directory"])
    base_dir = Path(config["dataset"]["base_directory"])
    base_root = json.loads((base_dir / "manifest.json").read_text())
    base_entries = {}
    for split, item in config["dataset"]["splits"].items():
        base_dataset = FrozenDataset(base_dir, args.data_root, split, allow_candidates=True)
        entries = base_root["splits"][split]["worlds"]
        if len(entries) != item["base_worlds"] or len(base_dataset) != item["base_worlds"] * item["frames_per_world"]:
            raise ValueError("the authorized base-world membership changed")
        base_entries[split] = [dict(entry, path=os.path.relpath((base_dir / entry["path"]).resolve(), output.resolve()), origin="base")
                               for entry in entries]
    science = {k: config[k] for k in ("seed", "dataset", "calibration", "placement", "shape", "qualification", "proposals")}
    science["normal_reference_surface"] = config["supervision"]["C3"]["parameters"]
    science["sampling_scale"] = config["supervision"]["common"]["sampling_scale"]
    if config["qualification"]["reference_use"] != "normal_sources_and_predeclared_content_only":
        raise ValueError("this generator does not use real-label distribution fitting")
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("generate.py", "render.py", "data.py", "profile.py", "coverage.py", "supervision.py")}
    identity = hashlib.sha256(json.dumps(dict(science=science, implementation=implementation, base=base_entries), sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["status"] == "frozen":
            raise FileExistsError("the first-round dataset is already frozen")
        if existing["configuration_identity"] != identity:
            raise ValueError("supplement directory belongs to different generation inputs")
    schedule = []
    for split, item in config["dataset"]["splits"].items():
        profiles = config["proposals"]["categories"][split]
        count = len(profiles) * config["proposals"]["repeats_per_source"]
        if item["candidates"] != count:
            raise ValueError("supplement budget must equal declared categories times repeats")
        for index in range(count):
            seed = int(np.random.SeedSequence([config["seed"], config["proposals"]["seed_namespace"], item["source_sequence"], index]).generate_state(1)[0])
            schedule.append(dict(split=split, index=index, seed=seed,
                                 category=profiles[index // config["proposals"]["repeats_per_source"]]["name"]))
    disk = host_disk()
    active_peak = args.workers * (256 * 1024**2 + 2 * 393216 * 32)
    allocation = len(schedule) * 256 * 1024**2
    peak = allocation + active_peak + 256 * 1024**2
    if disk["SizeRemaining"] - peak < disk["reserve_bytes"]:
        raise OSError("supplement peak allocation would enter the physical E: reserve")
    manifest = dict(format="stu-frozen-dataset", status="building", configuration_identity=identity,
                    configuration=science, implementation=implementation, splits={}, generation={},
                    base_dataset=dict(directory=str(base_dir), configuration_identity=base_root["configuration_identity"]),
                    information_use="normal_sources_only;_retain_all_32_authorized_base_worlds;_no_val19_numeric_fitting",
                    planned_candidates=schedule,
                    execution=dict(workers=args.workers, host_before=disk, peak_budget_bytes=peak))
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, manifest)
    started = time.perf_counter()
    manifest["calibration"] = prepare_calibration(args.data_root, output, config)
    reference_path = output / "normal_reference.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        if reference["configuration_identity"] != identity:
            raise ValueError("normal reference cache belongs to different inputs")
        sequence = STUSequence.open(args.data_root, protocol=load_protocol(), partition="train", sequence_id=201, label_mode="required")
        if any(source_identity(sequence[r["frame"]]) != r["source_identity"] for r in reference["scans"]):
            raise ValueError("normal reference scan changed")
    else:
        frames = config["proposals"]["normal_reference_frames"]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(frames)), mp_context=mp.get_context("spawn")) as pool:
            scans = list(pool.map(scan_normal_references, [str(args.data_root)] * len(frames), frames, [config] * len(frames)))
        _atomic_json(reference_path, dict(configuration_identity=identity, scans=scans))
        print(json.dumps(dict(event="native_references", frames={r["frame"]: len(r["slots"]) for r in scans})), flush=True)
    reports, jobs = {split: [] for split in config["dataset"]["splits"]}, []
    for planned in schedule:
        split, index = planned["split"], planned["index"]
        path = output / split / f"supplement_{index:03d}" / "manifest.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report["configuration_identity"] != identity:
                raise ValueError("cached supplement belongs to another generator")
            reports[split].append(report)
        else:
            jobs.append((split, index))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(generate_candidate, str(args.data_root), str(output), split, index, config, identity): (split, index)
                   for split, index in jobs}
        for future in as_completed(futures):
            split, index = futures[future]
            report = future.result()
            reports[split].append(report)
            print(json.dumps(dict(event="supplement", split=split, index=index, status=report["status"],
                                  reason=report["reason"], purpose={k: v for k, v in report.get("purpose", {}).items() if k != "probes"},
                                  content=report.get("content"), seconds=report["seconds"])), flush=True)
            remaining = host_disk()
            used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            if used > allocation or remaining["SizeRemaining"] < remaining["reserve_bytes"] + active_peak:
                for pending in futures:
                    pending.cancel()
                raise OSError("generation allocation reached; completed worlds are retained")
    for split, item in config["dataset"]["splits"].items():
        values = sorted(reports[split], key=lambda r: r["index"])
        worlds, selection = select_worlds(base_entries[split], values, split)
        manifest["generation"][split] = values
        manifest["splits"][split] = dict(source_sequence=item["source_sequence"], samples=len(worlds) * item["frames_per_world"],
                                         selection=selection, worlds=worlds)
    # A completed failed purpose limits the experiment; it never triggers another candidate batch.
    manifest["status"] = "frozen"
    manifest["execution"].update(seconds=time.perf_counter() - started, host_after=host_disk(),
                                 bytes_on_disk=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()))
    _atomic_json(manifest_path, manifest)
    for split in manifest["splits"]:
        FrozenDataset(output, args.data_root, split)
    print(json.dumps(dict(event="frozen", directory=str(output),
                          samples={s: x["samples"] for s, x in manifest["splits"].items()},
                          seconds=manifest["execution"]["seconds"])), flush=True)


if __name__ == "__main__":
    main()
