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

from .data import FrozenFrame, _atomic_json, host_disk, source_identity
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
    render_frames,
    save_sensor_calibration,
    shape_geometry,
    qualify_grounding,
    _frame_trace_context,
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


def sample_support(sequence, rng, config, footprint_radius, background, rejections):
    poses = np.stack([sequence.lidar_pose(f) for f in sequence.frame_ids])
    accepted = 0
    for attempt in range(config["maximum_support_attempts"]):
        frame = sequence[int(rng.integers(len(sequence)))]
        ground_slots = frame.real_slots[np.isin(frame.labels.semantic[frame.real_slots], config["semantics"])]
        xyz = frame.xyzi[ground_slots, :3].astype(np.float64)
        distance = np.linalg.norm(xyz, axis=1)
        choices = np.flatnonzero((distance >= config["proposal_range_m"][0]) & (distance <= config["proposal_range_m"][1]))
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
        if category != background:
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
                         background_semantic_counts={str(k): int(v) for k, v in Counter(semantic[known].tolist()).items()})
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


def far_ray_opportunities(sequence, item, grid, geometry):
    center = np.asarray(item.translation_world_m)
    distances = np.array([np.linalg.norm(center - sequence.lidar_pose(f)[:3, 3]) for f in sequence.frame_ids])
    far = np.flatnonzero((distances >= 35) & (distances <= 50))
    if not len(far):
        return []
    frames = sorted({int(far[np.argmin(np.abs(distances[far] - value))]) for value in (35., 42.5, 50.)})
    rotation = np.asarray(item.rotation_world_from_local)
    lower, upper = np.asarray(geometry["lower_local_m"]), np.asarray(geometry["upper_local_m"])
    observations = []
    for frame in frames:
        source = sequence[frame]
        slots, _, directions, _, origins, native = _frame_trace_context(source, grid)
        origin, direction = (origins - center) @ rotation, directions @ rotation
        # Box intersections are proposal opportunities, never rendered return counts.
        parallel = np.abs(direction) < 1e-15
        safe_direction = np.where(parallel, 1., direction)
        a, b = (lower - origin) / safe_direction, (upper - origin) / safe_direction
        a[parallel], b[parallel] = -np.inf, np.inf
        enter, leave = np.minimum(a, b).max(axis=1), np.maximum(a, b).min(axis=1)
        available = (leave >= np.maximum(enter, 0)) & (enter < native - 1e-6)
        available &= ~np.any(parallel & ((origin < lower) | (origin > upper)), axis=1)
        count = len(np.unique(grid.canonical_ray_by_slot[slots[available]]))
        observations.append(dict(frame=frame, range_m=float(distances[frame]), available_box_rays=count))
    return observations


def make_world(sequence, seed, config, profile, grid):
    rng = np.random.default_rng(seed)
    shape, geometry = make_shape(rng, profile, config["shape"])
    grounding = qualify_grounding(shape)
    if not grounding.passed:
        raise ValueError("shape_grounding_unreliable")
    material = MaterialSpec(float(rng.uniform(*config["shape"]["material_quantile"])),
                            float(rng.uniform(*config["shape"]["material_roughness"])))
    rejections, candidates = Counter(), []
    for pool, support in sample_support(sequence, rng, config["placement"], geometry["footprint_radius_m"], profile["background"], rejections):
        frame = sequence[int(pool.frames[0])]
        slots = frame.real_slots
        selected = slots[(frame.labels.semantic[slots] != 0) & ~np.isin(frame.labels.semantic[slots], GROUND)]
        obstacles = ObservedObstacleIndex(frame.xyzi[selected, :3].astype(float) @ frame.lidar_pose[:3, :3].T + frame.lidar_pose[:3, 3],
                                         (np.uint64(frame.frame_id) << np.uint64(32)) | selected.astype(np.uint64))
        yaw = float(rng.uniform(-np.pi, np.pi))
        if profile["view"] != "ordinary":
            poses = np.array([sequence.lidar_pose(f)[:3, 3] for f in sequence.frame_ids])
            direction = pool.anchors_world_m[0] - poses[np.argmin(np.abs(np.linalg.norm(poses - pool.anchors_world_m[0], axis=1) - 42.5))]
            yaw = float(np.arctan2(direction[1], direction[0]) + np.pi / 2 + rng.uniform(-.35, .35))
        try:
            item, placement = place_object(shape, material, pool, obstacles, object_id=1, label="anomaly-proxy",
                                           proposal_namespace=f"single-frame-content/{seed}", proposal_stream=0,
                                           yaw_rad=yaw, material_seed=seed, yaw_seed=seed, shape_seed=seed,
                                           proposal_rows=[0], maximum_candidates=1, grounding_eligibility=grounding)
        except ValueError:
            rejections["local_collision_or_grounding"] += 1
            continue
        opportunities = far_ray_opportunities(sequence, item, grid, geometry) if profile["view"] != "ordinary" else []
        rays = [r["available_box_rays"] for r in opportunities]
        target = profile["proposal_ray_target"]
        score = (sum(n >= 5 for n in rays), -sum(abs(n - target) for n in rays)) if opportunities else (-1, 0)
        candidates.append((score, item, dict(**support, placement=clean_json(placement.to_dict()),
                                            support_plane=dict(anchor_world_m=pool.anchors_world_m[0].tolist(),
                                                               normal_world=pool.normals_world[0].tolist(), offset=float(pool.offsets[0])),
                                            yaw_rad=yaw, far_ray_opportunities=opportunities)))
    if not candidates:
        raise ValueError("no_physical_support:" + json.dumps(dict(rejections), sort_keys=True))
    best = max(range(len(candidates)), key=lambda i: (candidates[i][0], -i))
    _, item, chosen = candidates[best]
    return WorldSpec(seed, sequence.spec.sequence_id, (item,)), dict(
        **chosen, geometry=geometry, shape_family=profile["shape"], candidate_category=profile["name"],
        proposal_intent=profile["view"], support_rejections=dict(rejections),
        support_proposals=[dict(frame=c[2]["placement"]["support_frame"], slot=c[2]["placement"]["support_slot"],
                                far_ray_opportunities=c[2]["far_ray_opportunities"], chosen=i == best)
                           for i, c in enumerate(candidates)],
        connectivity=shape.continuous_connectivity_certificate().state)


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
        np.random.SeedSequence([config["seed"], sequence_id, index]).generate_state(1)[
            0
        ]
    )
    directory = Path(output) / split / f"candidate_{index:03d}"
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
        profile = config["proposals"]["categories"][index // config["proposals"]["repeats_per_source"]]
        report["candidate_category"] = profile["name"]
        world, placement = make_world(sequence, seed, config, profile, grid)
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
    report.update(
        status="qualified" if reason is None else "rejected",
        reason=reason,
        collision=collision,
        world_identity=world.identity,
        frames=rows,
        trajectory=stats,
        histograms=hist,
        content=content_summary(rows, placement),
        shape_family=placement["shape_family"],
        background=placement["background"],
        geometry=placement["geometry"],
        seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        reconstruction="all_rendered_frames_equal_xyzi_packed_labels_slots_and_masks",
    )
    _atomic_json(directory / "manifest.json", report)
    return report


def select_worlds(reports, number=None):
    """Round-robin content strata; frames and auxiliary validity never determine weight."""
    available = sorted((r for r in reports if r["status"] == "qualified"), key=lambda r: r["index"])
    number = len(available) if number is None else number
    if number > len(available):
        raise ValueError("selection requests more qualified worlds than exist")
    conditions_order = ("far_few_eligible", "far_denser", "low_far_eligible", "few_eligible", "low_eligible", "all")
    chosen, cells, shapes, backgrounds = [], Counter(), Counter(), Counter()
    while len(chosen) < number:
        for condition in conditions_order:
            members = [r for r in available if r["content"][condition]["frames"] > 0]
            if not members:
                continue
            best = min(members, key=lambda r: (cells[(r["shape_family"], r["background"])],
                                               shapes[r["shape_family"]], backgrounds[r["background"]], r["index"]))
            chosen.append(best)
            available.remove(best)
            cells[(best["shape_family"], best["background"])] += 1
            shapes[best["shape_family"]] += 1
            backgrounds[best["background"]] += 1
            if len(chosen) == number:
                break
    return chosen, dict(order=[r["index"] for r in chosen],
                        conditions={c: sum(r["content"][c]["frames"] > 0 for r in chosen) for c in conditions_order},
                        shape_worlds=dict(shapes), background_worlds=dict(backgrounds),
                        rule="content_round_robin_then_least_represented_shape_background_then_index")


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
    parser.add_argument("--select", type=int, nargs=2, metavar=("TRAIN_WORLDS", "VALIDATION_WORLDS"),
                        help="select and freeze explicit quotas after candidate inspection")
    args = parser.parse_args()
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)) or (args.select and min(args.select) < 1):
        parser.error("workers must fit the CPU affinity; explicit quotas must be positive")
    config = json.loads(args.config.read_text())
    output = args.output or Path(config["dataset"]["directory"])
    science = {k: config[k] for k in ("seed", "dataset", "calibration", "placement", "shape", "qualification", "proposals")}
    if config["qualification"]["reference_use"] != "normal_sources_and_predeclared_content_only":
        raise ValueError("this generator requires the current content specification without real-label fitting")
    implementation = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__), Path(__file__).with_name("render.py"), Path(__file__).with_name("data.py"),
                       Path(__file__).with_name("profile.py"), Path(__file__).with_name("coverage.py"))}
    identity = hashlib.sha256(json.dumps(dict(science=science, implementation=implementation), sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["status"] == "frozen":
            raise FileExistsError("the dataset is already frozen")
        if existing["configuration_identity"] != identity:
            raise ValueError("candidate directory belongs to different generation inputs")
    count = len(config["proposals"]["categories"]) * config["proposals"]["repeats_per_source"]
    schedule = []
    for split, item in config["dataset"]["splits"].items():
        if item["candidates"] != count:
            raise ValueError("candidate budget must cover exactly the declared categories and repeats")
        for index in range(count):
            seed = int(np.random.SeedSequence([config["seed"], item["source_sequence"], index]).generate_state(1)[0])
            schedule.append(dict(split=split, index=index, seed=seed,
                                 category=config["proposals"]["categories"][index // config["proposals"]["repeats_per_source"]]["name"]))
    disk = host_disk()
    # Each candidate is streamed and bounded; include active atomic/raw frame copies.
    active_peak = args.workers * (256 * 1024**2 + 2 * 393216 * 32)
    allocation = len(schedule) * 256 * 1024**2
    peak = allocation + active_peak + 256 * 1024**2
    if disk["SizeRemaining"] - peak < disk["reserve_bytes"]:
        raise OSError("candidate peak allocation would enter the physical E: reserve")
    manifest = dict(format="stu-frozen-dataset", status="building", configuration_identity=identity,
                    configuration=science, implementation=implementation, splits={}, generation={},
                    information_use="normal_sources_only_no_val19_numeric_reference_or_world_reuse",
                    planned_candidates=schedule,
                    execution=dict(workers=args.workers, host_before=disk, allocation_bytes=allocation,
                                   active_candidate_peak_bytes=active_peak, peak_budget_bytes=peak))
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, manifest)
    manifest["calibration"] = prepare_calibration(args.data_root, output, config)
    reports, jobs = {split: [] for split in config["dataset"]["splits"]}, []
    for planned in schedule:
        split, index = planned["split"], planned["index"]
        path = output / split / f"candidate_{index:03d}" / "manifest.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report["configuration_identity"] != identity:
                raise ValueError("cached candidate belongs to another generator")
            reports[split].append(report)
        else:
            jobs.append((split, index))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(generate_candidate, str(args.data_root), str(output), split, index, config, identity): (split, index)
                   for split, index in jobs}
        for future in as_completed(futures):
            split, index = futures[future]
            report = future.result()
            reports[split].append(report)
            print(json.dumps(dict(event="candidate", split=split, index=index, category=report.get("candidate_category"),
                                  status=report["status"], reason=report["reason"], content=report.get("content"),
                                  seconds=report.get("seconds"), peak_rss_bytes=report.get("peak_rss_bytes"))), flush=True)
            if sum(map(len, reports.values())) % 4 == 0:
                remaining = host_disk()
                used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                if used > allocation or remaining["SizeRemaining"] < remaining["reserve_bytes"] + active_peak:
                    for pending in futures:
                        pending.cancel()
                    raise OSError("generation allocation reached; complete candidates are retained")
    for split, item in config["dataset"]["splits"].items():
        values = sorted(reports[split], key=lambda r: r["index"])
        # Complete, physically valid candidates remain inspectable even without eligible observations.
        complete = [r for r in values if len(r["frames"]) == item["frames_per_world"] and not r.get("collision")]
        _, order = select_worlds(values)
        manifest["generation"][split] = values
        manifest["splits"][split] = dict(source_sequence=item["source_sequence"],
                                         samples=len(complete) * item["frames_per_world"],
                                         selection_preview=order,
                                         worlds=[dict(path=f"{split}/candidate_{r['index']:03d}", world_identity=r["world_identity"],
                                                      candidate_index=r["index"]) for r in complete])
    manifest["status"] = "candidates_complete"
    if args.select:
        for (split, item), number in zip(config["dataset"]["splits"].items(), args.select):
            chosen, selection = select_worlds(reports[split], number)
            manifest["splits"][split].update(samples=number * item["frames_per_world"], selection=selection,
                                             worlds=[dict(path=f"{split}/candidate_{r['index']:03d}", world_identity=r["world_identity"],
                                                          candidate_index=r["index"]) for r in chosen])
        manifest["status"] = "frozen"
    manifest["execution"].update(seconds=time.perf_counter() - started, host_after=host_disk(),
                                 bytes_on_disk=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()))
    _atomic_json(manifest_path, manifest)
    print(json.dumps(dict(event=manifest["status"], directory=str(output),
                          samples={s: x["samples"] for s, x in manifest["splits"].items()},
                          seconds=manifest["execution"]["seconds"])), flush=True)


if __name__ == "__main__":
    main()
