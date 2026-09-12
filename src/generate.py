"""Freeze fixed physical worlds as complete sequences of independent scans."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
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
    render_frame,
    save_sensor_calibration,
    shape_geometry,
    shape_relations,
    primary_structure,
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
        # Prefer the scheduled trajectory quarter, then use the remaining bounded draws globally.
        interval = config.get("frame_interval", (0, len(sequence))) if attempt < config["maximum_support_attempts"] // 2 else (0, len(sequence))
        if references is not None:
            first, stop = interval
            available = [r for r in references if r["slots"] and first <= r["frame"] < stop]
            if not available:
                rejections["no_native_normal_reference"] += 1
                continue
            record = available[int(rng.integers(len(available)))]
            witness = sequence[record["frame"]]
            if source_identity(witness) != record["source_identity"]:
                raise ValueError("native reference source changed")
            ids = np.asarray(record["slots"], np.int32)
            point = witness.xyzi[record["anchor_slot"], :3]
            cluster = np.linalg.norm(witness.xyzi[ids, :3] - point, axis=1) <= config["reference_cluster_radius_m"]
            if cluster.sum() < config["minimum_reference_positions"]:
                rejections["normal_reference_cluster_too_small"] += 1
                continue
            reference = dict(frame=witness.frame_id, source_identity=record["source_identity"],
                             slots=ids[cluster].tolist(), kind=record["kind"])
            # Physical placement may use a denser view of the same source location.
            point_world = point.astype(float) @ witness.lidar_pose[:3, :3].T + witness.lidar_pose[:3, 3]
            distances = np.linalg.norm(poses[:, :3, 3] - point_world, axis=1)
            lo, hi = config["normal_reference_support_view_range_m"]
            support_frames = np.flatnonzero((distances >= lo) & (distances <= hi))
            if not len(support_frames):
                rejections["no_nearby_support_view"] += 1
                continue
            frame = sequence[int(rng.choice(support_frames))]
            point = (point_world - frame.lidar_pose[:3, 3]) @ frame.lidar_pose[:3, :3]
        else:
            frames = config.get("frame_candidates")
            frame = sequence[int(rng.choice(frames)) if frames else int(rng.integers(*interval))]
        ground_slots = frame.real_slots[np.isin(frame.labels.semantic[frame.real_slots], config["semantics"])]
        xyz = frame.xyzi[ground_slots, :3].astype(np.float64)
        distance = np.linalg.norm(xyz, axis=1)
        choices = np.flatnonzero((distance >= config["proposal_range_m"][0]) & (distance <= config["proposal_range_m"][1]))
        if config.get("target_region") and reference is None:
            world_xy = (xyz[choices] @ frame.lidar_pose[:3, :3].T + frame.lidar_pose[:3, 3])[:, :2]
            region = np.floor(world_xy / config["region_grid_m"]).astype(int)
            choices = choices[np.all(region == config["target_region"], axis=1)]
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
    elif family == "cross":
        scales = np.array([[.50, .10, .50], [.10, .50, .50]])
        offsets = np.zeros((2, 3))
    else:
        raise ValueError("unknown declared shape family")
    # Part proportions, junctions and relative yaw vary before physical checks.
    scales *= rng.uniform(*config["part_scale_multiplier"], scales.shape)
    if len(scales) > 1:
        offsets += rng.uniform(-config["part_offset_jitter"], config["part_offset_jitter"], offsets.shape)
    if profile.get("relation") == "multiple_contact":
        # Both feet must reach the same local support plane after part variation.
        bottom = np.min(offsets[:2, 2] - scales[:2, 2])
        offsets[:2, 2] = bottom + scales[:2, 2]
    yaws = rng.uniform(*config["part_yaw_rad"], len(scales)) if len(scales) > 1 else np.zeros(1)
    lower, upper = (offsets - scales).min(axis=0), (offsets + scales).max(axis=0)
    factors = dimensions / (upper - lower)
    offsets = (offsets - (upper + lower) / 2) * factors
    scales *= factors
    exponent_range = config["exponents" if family == "single" else "composed_exponents"]
    shape = ShapeSpec(tuple(map(tuple, scales)), tuple(map(tuple, offsets)),
                      tuple(map(tuple, rng.uniform(*exponent_range, (len(scales), 2)))),
                      tuple(yaws), ("union",) * len(scales))
    geometry = shape_geometry(shape)
    if not np.isclose(geometry["height_m"], dimensions[2], atol=3e-6, rtol=0):
        raise ValueError("local height changed under within-part horizontal rotations")
    return shape, geometry


def ray_observation(source, world, grid, sensor, geometry, reference_slots=None):
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
    record = dict(frame=source.frame_id,
                range_m=float(np.linalg.norm(center - source.lidar_pose[:3, 3])),
                available_box_rays=unique(box), foreground_surface_rays=unique(surface),
                potential_new_rays=unique(surface & ~np.isfinite(native)),
                potential_changed_native_rays=unique(surface & np.isfinite(native)),
                final_anomaly_rays=unique(returned), final_anomaly_slots=int(returned.sum()),
                in_range_anomaly_slots=int(np.sum((ranges >= 2.5) & (ranges <= 50))))
    if reference_slots is not None:
        # Select physical opportunities independently of stochastic signal returns.
        points = origins_sensor[surface] + competition.distance_m[surface, None] * directions_sensor[surface]
        distance = np.linalg.norm(points, axis=1)
        inside = (distance >= 2.5) & (distance <= 50)
        protected = np.asarray(reference_slots, np.int32)
        protected = protected[~surface[protected]]
        near = (cKDTree(points[inside]).query(source.xyzi[protected, :3])[0] <= 2
                if inside.any() else np.zeros(len(protected), bool))
        record.update(native_joint_surface_rays=len(np.unique(grid.canonical_ray_by_slot[slots[surface][inside]])),
                      native_joint_positions=len(np.unique(source.xyzi[protected[near], :3], axis=0)),
                      protected_slots=protected.tolist())
    return record, surface


def surface_opportunities(sequence, world, grid, sensor, geometry, config):
    center = np.asarray(world.objects[0].translation_world_m)
    distances = np.array([np.linalg.norm(center - sequence.lidar_pose(f)[:3, 3]) for f in sequence.frame_ids])
    records = []
    for band, lo, hi in (("near", 2.5, 10), ("middle", 10, 35), ("far", 35, 50.00001)):
        frames = np.flatnonzero((distances >= lo) & (distances < hi))
        if len(frames):
            chosen = frames[np.unique(np.linspace(0, len(frames)-1,
                            min(len(frames), config["opportunity_frames_per_band"]), dtype=int))]
            records.extend(dict(ray_observation(sequence[int(frame)], world, grid, sensor, geometry)[0],
                                distance_band=band) for frame in chosen)
    return records


def opportunity_passes(probes, profile, config):
    """Rank legal proposals by exact intersections, independently of signal draws."""
    kind = profile["opportunity"]
    threshold = config["surface_ray_requirements"][kind]
    if kind in ("far_dense", "low_far"):
        selected = [p for p in probes if p["distance_band"] == "far" and p["foreground_surface_rays"] >= threshold]
    elif kind == "near_sparse":
        selected = [p for p in probes if p["distance_band"] == "near" and
                    threshold[0] <= p["foreground_surface_rays"] <= threshold[1]]
    elif kind == "weak_background":
        selected = [p for p in probes if p["foreground_surface_rays"] >= threshold
                    and p["potential_changed_native_rays"] <= 4]
    else:
        selected = [p for p in probes if p["foreground_surface_rays"] >= threshold]
    return len(selected) >= config["minimum_opportunity_frames"]


def make_world(sequence, seed, config, profile, grid, sensor, references=None):
    """Choose physical supports before inspecting any final signal counts."""
    rejections = Counter()
    for attempt in range(config["proposals"]["shape_attempts"]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 10, attempt]))
        try:
            shape, geometry = make_shape(rng, profile, config["shape"])
            grounding = qualify_grounding(shape)
            if not grounding.passed:
                raise ValueError("shape_grounding_unreliable")
            if primary_structure(shape_relations(shape, config["research_coverage"]["morphology"])) != profile["structure"]:
                raise ValueError("actual_geometry_relation_not_supported")
            break
        except ValueError as error:
            rejections[str(error)] += 1
    else:
        raise ValueError("bounded_shape_search_failed:" + json.dumps(dict(rejections)))
    material = MaterialSpec(float(rng.uniform(*config["shape"]["material_quantile"])),
                            float(rng.uniform(*config["shape"]["material_roughness"])))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20]))
    placement_config = dict(config["placement"], frame_interval=profile["frame_interval"],
                            target_region=profile["target_region"], frame_candidates=profile["frame_candidates"],
                            region_grid_m=config["research_coverage"]["regions"]["grid_m"])
    if references is not None:
        placement_config.update(proposal_range_m=[5, 50], minimum_reference_positions=config["proposals"]["normal_context"]["minimum_positions"],
                                reference_cluster_radius_m=2, reference_clearance_m=[.05, 1.5])
    bounds = (np.asarray(geometry["lower_local_m"]), np.asarray(geometry["upper_local_m"]))
    candidates = []
    for pool, support in sample_support(sequence, rng, placement_config, geometry["footprint_radius_m"],
                                       profile["background"], rejections, references):
        frame = sequence[int(pool.frames[0])]
        slots = frame.real_slots
        selected = slots[(frame.labels.semantic[slots] != 0) & ~np.isin(frame.labels.semantic[slots], GROUND)]
        obstacles = ObservedObstacleIndex(frame.xyzi[selected, :3].astype(float) @ frame.lidar_pose[:3, :3].T + frame.lidar_pose[:3, 3],
                                         (np.uint64(frame.frame_id) << np.uint64(32)) | selected.astype(np.uint64))
        yaw = float(rng.uniform(-np.pi, np.pi))
        if references is not None:
            witness = sequence[support["normal_reference"]["frame"]]
            sight = pool.anchors_world_m[0] - witness.lidar_pose[:3, 3]
            yaw = float(np.arctan2(sight[1], sight[0]) + np.pi / 2 +
                        rng.uniform(*config["proposals"]["native_yaw_offset_rad"]))
        worlds, placements = [], []
        try:
            for offset in profile["yaw_offsets_rad"]:
                item, placement = place_object(shape, material, pool, obstacles, object_id=1, label="anomaly-proxy",
                    proposal_namespace=f"physical-expansion/{seed}", proposal_stream=0,
                    yaw_rad=yaw + offset, material_seed=seed, yaw_seed=seed, shape_seed=seed,
                    proposal_rows=[0], maximum_candidates=1, grounding_eligibility=grounding)
                world = WorldSpec(seed, sequence.spec.sequence_id, (item,))
                if references is not None:
                    witness = sequence[support["normal_reference"]["frame"]]
                    joint, _ = ray_observation(witness, world, grid, sensor, geometry,
                                              support["normal_reference"]["slots"])
                    if (joint["native_joint_positions"] < config["proposals"]["normal_context"]["minimum_positions"]
                            or joint["native_joint_surface_rays"] < config["proposals"]["native_surface_minimum"]):
                        raise ValueError("no_joint_native_surface_opportunity")
                    support = dict(support, normal_reference=dict(support["normal_reference"],
                                   slots=joint["protected_slots"]), native_surface_opportunity=joint)
                worlds.append(world)
                placements.append(dict(**support, placement=clean_json(placement.to_dict()),
                    support_plane=dict(anchor_world_m=pool.anchors_world_m[0].tolist(),
                                       normal_world=pool.normals_world[0].tolist(), offset=float(pool.offsets[0])),
                    yaw_rad=yaw + offset, geometry=geometry, shape_family=profile["shape"],
                    candidate_category=profile["name"], proposal_intent=profile["opportunity"],
                    combination=profile["combination"],
                    shape_attempt=attempt, nominal_dimensions_m={k: profile[k] for k in ("length_m", "width_m", "height_m")},
                    connectivity=shape.continuous_connectivity_certificate().state))
        except ValueError as error:
            rejections[str(error)] += 1
            continue
        probes = surface_opportunities(sequence, worlds[0], grid, sensor, geometry, config["proposals"])
        achieved = opportunity_passes(probes, profile, config["proposals"])
        candidates.append((achieved, worlds, placements, probes))
        if achieved:
            break
    # A content-poor but legal candidate is retained for measured pool selection.
    for achieved, worlds, placements, probes in sorted(candidates, key=lambda x: not x[0]):
        collision = None
        for original in sequence:
            for world in worlds:
                nearby = nearby_obstacles(original, world.objects[0])
                if nearby is not None:
                    hit, minimum, _ = observed_normal_collision(world.objects[0], nearby,
                        penetration_m=config["placement"]["deep_penetration_m"], local_bounds=bounds)
                    if hit:
                        collision = dict(frame=original.frame_id, minimum_sdf_m=minimum)
                        break
            if collision:
                break
        if collision:
            rejections["complete_trajectory_collision"] += 1
            continue
        for world, placement in zip(worlds, placements):
            placement.update(ray_observations=probes, support_rejections=dict(rejections),
                             proposal_opportunities_met=achieved,
                             physical_check="no_deep_penetration_in_any_original_source_frame")
        return worlds, placements
    raise ValueError("bounded_physical_support_search_failed:" + json.dumps(dict(rejections)))


def scan_normal_references(data_root, sequence_id, frame, config, reference):
    from numba import set_num_threads
    from .profile import observed_geometry
    from .coverage import native_context_types
    set_num_threads(1)
    source = STUSequence.open(data_root, protocol=load_protocol(), partition="train", sequence_id=sequence_id,
                              label_mode="required")[frame]
    slots = source.real_slots
    values = observed_geometry(source.xyzi[slots], slots)
    inside = (values["range"] >= 5) & (values["range"] <= 50) & (source.labels.semantic_target[slots] != 255)
    kinds = native_context_types(values, reference, config["research_coverage"]["normal_contrasts"]["sparse_neighbor_threshold"])
    ground = source.xyzi[slots[np.isin(source.labels.semantic[slots], config["placement"]["semantics"])] , :3]
    if not len(ground):
        return []
    context = config["proposals"]["normal_context"]
    distance, nearest = cKDTree(ground[:, :2]).query(source.xyzi[slots, :2])
    height = source.xyzi[slots, 2] - ground[nearest, 2]
    usable = inside & (distance <= context["maximum_ground_xy_distance_m"]) & (np.abs(height) <= context["maximum_height_above_ground_m"])
    records, identity = [], source_identity(source)
    for kind, mask in kinds.items():
        selected = slots[mask & usable]
        selected = selected[np.unique(source.xyzi[selected, :3], axis=0, return_index=True)[1]]
        if len(selected) < context["minimum_positions"]:
            continue
        xyz = source.xyzi[selected, :3]
        world = xyz.astype(float) @ source.lidar_pose[:3, :3].T + source.lidar_pose[:3, 3]
        # Spread native anchors spatially; store a small cluster, not repeated full scans.
        _, candidates = np.unique(np.floor(world / .5).astype(int), axis=0, return_index=True)
        if len(candidates) > context["anchors_per_kind_per_frame"]:
            candidates = candidates[np.linspace(0, len(candidates)-1, context["anchors_per_kind_per_frame"], dtype=int)]
        distance, neighbors = cKDTree(xyz).query(xyz[candidates], k=np.arange(1, context["neighbors_per_anchor"]+1),
                                                distance_upper_bound=context["radius_m"])
        for anchor, ns, ds in zip(candidates, neighbors, distance):
            cluster = ns[np.isfinite(ds)]
            if len(cluster) < context["minimum_positions"]:
                continue
            region = np.floor(world[anchor, :2] / config["research_coverage"]["regions"]["grid_m"]).astype(int)
            records.append(dict(frame=frame, source_identity=identity, slots=selected[cluster].tolist(),
                anchor_slot=int(selected[anchor]), kind=kind, region=region.tolist(), anchor_world_m=world[anchor].tolist()))
    return records


def check_normal_reference(sample, original, reference):
    """A native witness must survive unchanged and remain locally visible with the anomaly."""
    slots = np.asarray(reference["slots"], np.int32)
    if source_identity(original) != reference["source_identity"]:
        raise ValueError("native reference source changed after placement")
    unchanged = (np.array_equal(sample.source.xyzi[slots], original.xyzi[slots])
                 and np.array_equal(sample.source.labels.packed[slots], original.labels.packed[slots])
                 and np.all(sample.anomaly_target[slots] == 0))
    anomalies = sample.source.xyzi[sample.inserted_mask, :3]
    near = cKDTree(anomalies.astype(float)).query(original.xyzi[slots, :3])[0] <= 2 if len(anomalies) else np.zeros(len(slots), bool)
    return dict(frame=original.frame_id, protected_positions=len(slots), unchanged=bool(unchanged),
                adjacent_positions=int(near.sum()), source_slots=slots.tolist(), kind=reference["kind"],
                interpretation="native_positions_only; final_joint_geometry_coverage_is_measured_separately")


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


def expansion_schedule(config, round_number=1, references=None, source_poses=None):
    """Enumerate the thirty constrained cells before candidate rendering."""
    proposals = config["proposals"]
    if round_number != 1:
        raise ValueError("the declared candidate sweep has one batch")
    record = json.loads((Path(config["research_coverage"]["output"]) / "inventory.json").read_text())
    groups = []
    for split, item in config["dataset"]["splits"].items():
        base = [w for w in record["worlds"] if w["split"] == split]
        regions = sorted({w["regions"][0] for w in base})
        available = references[split] if references else None
        if available is not None and source_poses is not None:
            distance = cKDTree(source_poses[split]).query([r["anchor_world_m"] for r in available])[0]
            available = [r for r, d in zip(available, distance)
                         if d <= config["placement"]["normal_reference_support_view_range_m"][1]]
        cell = 0
        for structure, template in proposals["structures"].items():
            for height, height_range in proposals["heights"].items():
                for background in proposals["backgrounds"]:
                    combination = f"{structure}/{height}/{background}"
                    count = proposals.get("cell_candidates", {}).get(split, {}).get(
                        combination, proposals["candidates_per_cell"][split])
                    if not count:
                        cell += 1
                        continue
                    anchors = ({"/".join(map(str, r["region"])): r["anchor_world_m"]
                                for r in available if r["kind"] == background} if references else
                               {w["regions"][0]: w["anchor_world_m"] for w in base})
                    available_regions = sorted(anchors) or regions
                    for repeat in range(count):
                        index = proposals.get("index_start", 2000) + cell*256 + repeat
                        region_index = int(np.floor(repeat * len(available_regions) / count))
                        region = available_regions[(region_index + cell) % len(available_regions)]
                        anchor = anchors.get(region, base[0]["anchor_world_m"])
                        profile = dict(template, name=combination, combination=combination, structure=structure,
                            native_context=background, background="any", height_m=height_range,
                            target_region=list(map(int, region.split("/"))), anchor_world_m=anchor,
                            frame_interval=[0, item["frames_per_world"]], frame_candidates=[], yaw_offsets_rad=[0.],
                            opportunity="visible")
                        if structure == "solid" and height == "low":
                            profile.update(length_m=[1.3, 1.8], width_m=[.55, .8], height_m=[.18, .195])
                        if structure == "solid" and height == "raised":
                            profile.update(length_m=[1.5, 2.6], width_m=[.8, 1.6], height_m=[.8, 1.6])
                        if structure == "elongated":
                            profile.update(length_m=[2.2, 3.2], width_m=[.16, .3])
                            if height == "raised":
                                profile["height_m"] = [.4, .65]
                        if structure == "sheet" and height == "raised":
                            profile.update(length_m=[.7, 1.5], width_m=[.06, .12], height_m=[.5, 1.2])
                        if height == "low" and structure in ("branched", "contacts"):
                            profile.update(length_m=[1.5, 1.8], width_m=[.8, 1.], height_m=[.18, .195])
                        seed = int(np.random.SeedSequence([config["seed"], proposals["seed_namespace"],
                                                          item["source_sequence"], index]).generate_state(1)[0])
                        groups.append(dict(split=split, group=index, round=1, seed=seed, members=[index], profile=profile))
                    cell += 1
    return groups


def generate_group(data_root, output, planned, config, identity):
    """Freeze one geometry parent and its within-source orientation variants."""
    started = time.perf_counter()
    split, seed = planned["split"], planned["seed"]
    source_id = config["dataset"]["splits"][split]["source_sequence"]
    directories = [Path(output) / split / f"world_{i:03d}" for i in planned["members"]]
    if all((d / "manifest.json").exists() for d in directories):
        reports = [json.loads((d / "manifest.json").read_text()) for d in directories]
        if any(r["configuration_identity"] != identity for r in reports):
            raise ValueError("cached expansion group belongs to different inputs")
        return reports
    global _generation_inputs
    cache_key = (str(data_root), str(output), identity, source_id)
    if "_generation_inputs" not in globals() or _generation_inputs[0] != cache_key:
        sequence = STUSequence.open(data_root, protocol=load_protocol(), partition="train",
                                   sequence_id=source_id, label_mode="required")
        grid, sensor = load_sensor_calibration(Path(output) / "calibration.pt")
        references = json.loads((Path(output) / "normal_reference.json").read_text())["scans"][split]
        poses = np.array([sequence.lidar_pose(f)[:3, 3] for f in sequence.frame_ids])
        distances = cKDTree(poses).query([r["anchor_world_m"] for r in references])[0]
        references = [r for r, d in zip(references, distances)
                      if d <= config["placement"]["normal_reference_support_view_range_m"][1]]
        _generation_inputs = (cache_key, sequence, grid, sensor, references)
    _, sequence, grid, sensor, references = _generation_inputs
    profile = planned["profile"]
    references = [r for r in references if r["kind"] == profile["native_context"] and r["region"] == profile["target_region"]]
    poses = np.array([sequence.lidar_pose(f)[:3, 3] for f in sequence.frame_ids])
    distance = np.linalg.norm(poses - np.asarray(profile["anchor_world_m"]), axis=1)
    profile = dict(profile, frame_candidates=np.flatnonzero((distance >= 5) & (distance <= 18)).tolist())
    common = dict(seed=seed, source_sequence=source_id, configuration_identity=identity,
                  family_id=f"{split}/{planned['group']}", paired=len(directories) == 2,
                  combination=profile["combination"],
                  candidate_category=profile["name"], shape_family=profile["shape"])
    try:
        worlds, placements = make_world(sequence, seed, config, profile, grid, sensor, references)
    except ValueError as error:
        reports = [dict(common, index=i, status="rejected", reason=str(error), frames=[],
                        seconds=time.perf_counter() - started) for i in planned["members"]]
        for directory, report in zip(directories, reports):
            _atomic_json(directory / "manifest.json", report)
        return reports
    for directory, world, placement in zip(directories, worlds, placements):
        path = directory / "world.json"
        definition = dict(world=world.to_dict(), generation=placement)
        if path.exists() and json.loads(path.read_text()) != definition:
            raise ValueError("interrupted world definition differs from its fixed physical group")
        _atomic_json(path, definition)
    rows = [[] for _ in worlds]
    written = [sum(p.stat().st_size for p in (d / "frames").glob("*.npz")) for d in directories]
    render_started = time.perf_counter()
    for original in sequence:
        # The source scan is read once for the orientation pair; signal streams remain fixed.
        for member, (world, directory) in enumerate(zip(worlds, directories)):
            path = directory / "frames" / f"{original.frame_id:06d}.npz"
            if path.exists():
                sample = FrozenFrame.load(path, original, world.identity)
            else:
                rendered = render_frame(original, world, grid, sensor)
                sample = FrozenFrame(rendered.source, world.identity, rendered.inserted_mask, rendered.occluded_original_mask)
                sample.save(path, original)
                written[member] += path.stat().st_size
                if written[member] > config["proposals"]["world_bytes_limit"]:
                    raise OSError("world reached its declared allocation")
                restored = FrozenFrame.load(path, original, world.identity)
                for actual, expected in ((restored.source.xyzi, sample.source.xyzi),
                                         (restored.source.labels.packed, sample.source.labels.packed),
                                         (restored.source.labels.semantic_target, sample.source.labels.semantic_target),
                                         (restored.source.real_slots, sample.source.real_slots),
                                         (restored.inserted_mask, sample.inserted_mask),
                                         (restored.occluded_original_mask, sample.occluded_original_mask)):
                    if not np.array_equal(actual, expected):
                        raise ValueError("saved full-scan reconstruction differs from physical rendering")
            row = observation(sample)
            row.update(occluded=int(sample.occluded_original_mask.sum()), slots=sample.source.slot_count,
                       real_returns=sample.source.real_count)
            rows[member].append(row)
    reports = []
    for member, (directory, world, placement, observed) in enumerate(zip(directories, worlds, placements, rows)):
        if len(observed) != len(sequence):
            raise ValueError("a physical world must contain its complete source trajectory")
        for probe in placement["ray_observations"]:
            row = observed[probe["frame"]]
            if row["count"] != probe["final_anomaly_slots"] or row["in_range"] != probe["in_range_anomaly_slots"]:
                raise ValueError("full rendering changed a fixed proposal signal draw")
        context = None
        if references is not None:
            reference = placement["normal_reference"]
            original = sequence[reference["frame"]]
            sample = FrozenFrame.load(directory / "frames" / f"{original.frame_id:06d}.npz", original, world.identity)
            context = check_normal_reference(sample, original, reference)
            if not context["unchanged"]:
                raise ValueError("protected native context changed during rendering")
        stats, hist = distributions(observed)
        report = dict(common, index=planned["members"][member], variant=member, status="qualified", reason=None,
                      world_identity=world.identity, frames=observed, trajectory=stats, histograms=hist,
                      content=content_summary(observed, placement), normal_reference=context,
                      geometry=placement["geometry"], background=placement["background"],
                      content_check_frames=[context["frame"]] if context is not None else [],
                      group_seconds=time.perf_counter() - started,
                      seconds=(time.perf_counter() - started) / len(worlds),
                      render_seconds_per_world=(time.perf_counter() - render_started) / len(worlds),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      bytes_on_disk=written[member], reconstruction="all_frames_equal_xyzi_labels_slots_and_masks",
                      acceptance="physical_legality; geometric_opportunities_are_preferences; final_content_measured_separately")
        _atomic_json(directory / "manifest.json", report)
        reports.append(report)
    return reports


def select_worlds(base, reports, split):
    """Collect legal candidates; final balanced membership is selected from all worlds."""
    accepted = [r for r in sorted(reports, key=lambda r: r["index"]) if r["status"] == "qualified"]
    worlds = [dict(entry) for entry in base]
    worlds.extend(dict(path=f"{split}/world_{r['index']:03d}", world_identity=r["world_identity"],
                       candidate_index=r["index"], origin="constrained", cohort="candidate",
                       family_id=r["family_id"], paired=r["paired"], variant=r["variant"]) for r in accepted)
    return worlds, dict(original_worlds=len(base), accepted=[r["index"] for r in accepted],
                        rejected={str(r["index"]): r["reason"] for r in reports if r["status"] != "qualified"},
                        rule="all_existing_and_physically_legal_new_candidates_pending_balanced_selection")


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


def dataset_storage(directory):
    """Count referenced files once, including base provenance but excluding unused worlds."""
    files, directories, visited = set(), set(), set()
    def add_tree(path):
        directories.add(path)
        for item in path.rglob("*"):
            (files if item.is_file() else directories).add(item.resolve())
    root = json.loads((directory / "manifest.json").read_text())
    for part in root["splits"].values():
        for entry in part["worlds"]:
            add_tree((directory / entry["path"]).resolve())
    current = directory.resolve()
    while current not in visited:
        visited.add(current)
        directories.add(current)
        files.update(p.resolve() for p in current.iterdir() if p.is_file())
        record = json.loads((current / "manifest.json").read_text())
        base = record.get("base_dataset", {}).get("directory")
        if base is None:
            break
        current = Path(base).resolve()
    files.add(Path(root["configuration"]["calibration"]["rays"]).resolve())
    unique = {}
    for path in files:
        stat = path.stat()
        unique[(stat.st_dev, stat.st_ino)] = stat
    return dict(files=len(unique), content_bytes=sum(s.st_size for s in unique.values()),
                allocated_file_bytes=sum(s.st_blocks * 512 for s in unique.values()),
                allocated_directory_bytes=sum(p.stat().st_blocks * 512 for p in directories),
                scope="referenced_worlds_and_root_provenance_and_rays_no_raw_scans_or_unused_worlds")



def complete_candidates(output, manifest, data_root, require_complete):
    """Collect existing immutable worlds without rendering or changing their identities."""
    config = manifest["configuration"]
    base = Path(manifest["base_dataset"]["directory"])
    original = json.loads((base / "manifest.json").read_text())
    reports = {split: [] for split in config["dataset"]["splits"]}
    for plans in manifest["planned_rounds"].values():
        for planned in plans:
            for index in planned["members"]:
                path = output / planned["split"] / f"world_{index:03d}" / "manifest.json"
                if not path.exists():
                    if require_complete:
                        raise ValueError(f"candidate generation is incomplete: {path}")
                    continue
                report = json.loads(path.read_text())
                if report["configuration_identity"] != manifest["configuration_identity"]:
                    raise ValueError("saved parent belongs to another generation definition")
                reports[planned["split"]].append(report)
    for split, item in config["dataset"]["splits"].items():
        entries = [dict(e, path=os.path.relpath((base/e["path"]).resolve(), output.resolve()))
                   for e in original["splits"][split]["worlds"]]
        worlds, selection = select_worlds(entries, reports[split], split)
        manifest["generation"][split] = [{k: v for k, v in r.items() if k not in ("frames", "histograms")}
                                         for r in reports[split]]
        manifest["splits"][split] = dict(source_sequence=item["source_sequence"],
            samples=len(worlds)*item["frames_per_world"], selection=selection, worlds=worlds)
    if require_complete:
        manifest["completed_rounds"] = sorted(map(int, manifest["planned_rounds"]))
        manifest["status"] = "candidates_complete"
    _atomic_json(output / "manifest.json", manifest)
    manifest["execution"]["storage"] = dataset_storage(output)
    _atomic_json(output / "manifest.json", manifest)
    if require_complete:
        for split in reports:
            FrozenDataset(output, data_root, split, allow_candidates=True)
    print(json.dumps(dict(event=manifest["status"], directory=str(output),
        worlds={s: len(x["worlds"]) for s,x in manifest["splits"].items()},
        samples={s: x["samples"] for s,x in manifest["splits"].items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--config", type=Path, default=Path("protocol/data.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--pilot", action="store_true", help="run the first scheduled parent per source within the declared budget")
    parser.add_argument("--collect-only", action="store_true", help="collect all completed candidate reports without generating any world")
    args = parser.parse_args()
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        parser.error("workers must fit the CPU affinity")
    config = json.loads(args.config.read_text())
    output = args.output or Path(config["proposals"]["output"])
    if (output / "manifest.json").exists() and not args.collect_only:
        saved = json.loads((output / "manifest.json").read_text())
        if saved.get("status") == "frozen":
            print(json.dumps(dict(event="dataset_already_frozen", directory=str(output),
                worlds={s: len(p["worlds"]) for s, p in saved["splits"].items()})), flush=True)
            return
    if args.collect_only:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["status"] not in ("building", "candidates_complete"):
            raise ValueError("collection is only for an unfinished candidate manifest")
        manifest["execution"]["collection"] = dict(host=host_disk(),
            generation_restarted=False, rule="all_declared_candidate_reports_must_exist")
        complete_candidates(output, manifest, args.data_root, True)
        return
    base_dir = Path(config["dataset"]["base_directory"])
    base_root = json.loads((base_dir / "manifest.json").read_text())
    base_entries, source_poses = {}, {}
    for split in config["dataset"]["splits"]:
        dataset = FrozenDataset(base_dir, args.data_root, split, allow_candidates=True)
        source_poses[split] = np.asarray([dataset.sequence.lidar_pose(f)[:3, 3] for f in dataset.sequence.frame_ids])
        entries = base_root["splits"][split]["worlds"]
        base_entries[split] = [dict(entry, path=os.path.relpath((base_dir / entry["path"]).resolve(), output.resolve()))
                               for entry in entries]
    science = {k: config[k] for k in ("seed", "calibration", "placement", "shape", "qualification", "proposals", "research_coverage")}
    science["dataset"] = dict(directory=str(output), base_directory=str(base_dir), splits={
        s: {k: item[k] for k in ("source_sequence", "frames_per_world")} for s, item in config["dataset"]["splits"].items()})
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("generate.py", "render.py", "data.py", "profile.py")}
    from .coverage import native_context_types
    implementation["native_context_types"] = hashlib.sha256(inspect.getsource(native_context_types).encode()).hexdigest()
    identity = hashlib.sha256(json.dumps(dict(science=science, implementation=implementation, base=base_entries), sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    disk = host_disk()
    counts = {s: sum(config["proposals"].get("cell_candidates", {}).get(s, {}).get(
        f"{shape}/{height}/{background}", config["proposals"]["candidates_per_cell"][s])
        for shape in config["proposals"]["structures"] for height in config["proposals"]["heights"]
        for background in config["proposals"]["backgrounds"]) for s in base_entries}
    allocation = sum(n * (config["proposals"]["world_bytes_limit"] +
        4096*(config["dataset"]["splits"][s]["frames_per_world"]+3)) for s,n in counts.items())
    active_peak = args.workers * (config["proposals"]["world_bytes_limit"] + 2 * 393216 * 32)
    peak = allocation + active_peak + 256 * 1024**2
    if disk["SizeRemaining"] - peak < disk["reserve_bytes"]:
        raise OSError("targeted generation would enter the physical E: reserve")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["configuration_identity"] != identity:
            raise ValueError("targeted dataset belongs to different generation inputs")
    else:
        manifest = dict(format="stu-frozen-dataset", status="building", configuration_identity=identity,
            configuration=science, implementation=implementation,
            splits={s: dict(source_sequence=config["dataset"]["splits"][s]["source_sequence"],
                            samples=base_root["splits"][s]["samples"], worlds=w) for s, w in base_entries.items()},
            generation={}, base_dataset=dict(directory=str(base_dir), configuration_identity=base_root["configuration_identity"]),
            information_use="normal_206_201_only; old_and_new_worlds_share_240_120_final_quotas; no_val19_fitting",
            planned_rounds={}, completed_rounds=[], execution=dict(host_before=disk, base_storage=dataset_storage(base_dir), runs=[]))
    key = str(args.round)
    if args.round in manifest["completed_rounds"]:
        print(json.dumps(dict(event="round_already_complete", round=args.round)), flush=True)
        return
    manifest["status"] = "building"
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, manifest)
    started = time.perf_counter()
    manifest["calibration"] = prepare_calibration(args.data_root, output, config)
    reference_path = output / "normal_reference.json"
    previous_reference = base_dir / "normal_reference.json"
    if not reference_path.exists() and previous_reference.exists():
        old = base_root["configuration"]
        unchanged = (old["calibration"] == config["calibration"]
            and old["research_coverage"]["normal_contrasts"] == config["research_coverage"]["normal_contrasts"]
            and old["proposals"]["normal_context"] == config["proposals"]["normal_context"]
            and all(base_root["implementation"][k] == implementation[k] for k in ("profile.py", "native_context_types")))
        if unchanged:
            reference = json.loads(previous_reference.read_text())
            reference.update(configuration_identity=identity, reused_from=str(previous_reference),
                reuse_basis="same_original_geometry_normal_reference_calibration_and_anchor_rules; every_used_source_identity_is_checked")
            if config["proposals"].get("measured_contexts_only"):
                # Reuse observed normal clusters, then draw new geometry parents and fixed signals.
                for split, entries in base_entries.items():
                    witnesses = defaultdict(list)
                    for entry in entries:
                        saved = json.loads((output / entry["path"] / "manifest.json").read_text())
                        native = saved.get("normal_reference")
                        if (saved.get("status") == "qualified" and native and native.get("unchanged")
                                and native.get("adjacent_positions", 0) >= 5
                                and saved["frames"][native["frame"]]["in_range"] >= 5):
                            witnesses[native["kind"], native["frame"]].append(set(native["source_slots"]))
                    reference["scans"][split] = [r for r in reference["scans"][split]
                        if any(len(set(r["slots"]) & slots) >= 5 for slots in witnesses[r["kind"], r["frame"]])]
                reference["selection"] = "original_native_clusters_with_a_previously_measured_joint_witness; new_geometry_not_signal_rerolls"
            _atomic_json(reference_path, reference)
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        if reference["configuration_identity"] != identity:
            raise ValueError("normal context cache belongs to different inputs")
    else:
        from .coverage import geometry_reference
        fitted = geometry_reference()
        reference_quantiles = {k: fitted[k] for k in ("edges", "quantiles")}
        del fitted
        jobs = [(s, item["source_sequence"], f) for s, item in config["dataset"]["splits"].items()
                for f in range(item["frames_per_world"])]
        scans = {s: [] for s in base_entries}
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(scan_normal_references, str(args.data_root), seq, f, config, reference_quantiles): s for s, seq, f in jobs}
            for future in as_completed(futures):
                scans[futures[future]].extend(future.result())
        for records in scans.values():
            records.sort(key=lambda r: r["frame"])
        reference = dict(configuration_identity=identity, scans=scans, definition="full_original_scan_native_geometry_clusters; unchanged_206_normal_reference; no_auxiliary_training_target")
        _atomic_json(reference_path, reference)
    print(json.dumps(dict(event="native_context_anchors", anchors={s: dict(Counter(r['kind'] for r in v))
                        for s, v in reference["scans"].items()})), flush=True)
    if key not in manifest["planned_rounds"]:
        manifest["planned_rounds"][key] = expansion_schedule(config, args.round, reference["scans"], source_poses)
        _atomic_json(manifest_path, manifest)
    schedule = manifest["planned_rounds"][key]
    pilots = {s: next((p["group"] for p in schedule if p["split"] == s), None) for s in base_entries}
    jobs = []
    for planned in schedule:
        path = output / planned["split"] / f"world_{planned['members'][0]:03d}" / "manifest.json"
        if not path.exists() and (not args.pilot or planned["group"] == pilots[planned["split"]]):
            jobs.append(planned)
    last_host_check = time.monotonic()
    with ProcessPoolExecutor(max_workers=min(args.workers, max(1, len(jobs))), mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(generate_group, str(args.data_root), str(output), planned, config, identity): planned for planned in jobs}
        for future in as_completed(futures):
            planned, values = futures[future], future.result()
            print(json.dumps(dict(event="target_parent", split=planned["split"], target=planned["profile"]["name"],
                group=planned["group"], results=[{k: v.get(k) for k in ("index", "status", "reason", "seconds", "peak_rss_bytes", "bytes_on_disk")} for v in values])), flush=True)
            if time.monotonic() - last_host_check >= 60:
                remaining = host_disk()
                last_host_check = time.monotonic()
                if remaining["SizeRemaining"] < remaining["reserve_bytes"] + active_peak:
                    for pending in futures:
                        pending.cancel()
                    raise OSError("generation stopped at the physical host reserve; completed worlds retained")
    manifest["execution"]["runs"].append(dict(round=args.round, pilot=args.pilot, workers=args.workers, groups_run=len(jobs),
        seconds=time.perf_counter()-started, host_after=host_disk(), peak_budget_bytes=peak))
    complete_candidates(output, manifest, args.data_root, not args.pilot)


if __name__ == "__main__":
    main()
