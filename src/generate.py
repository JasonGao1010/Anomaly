"""Freeze fixed physical worlds as complete sequences of independent scans."""

from __future__ import annotations

import argparse
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


def reference(profile, margins):
    worlds, unknown = [], 0
    for directory in sorted(Path(profile).glob("[0-9]*")):
        frame_count = sum(1 for _ in (directory / "frames.jsonl").open())
        items = [json.loads(line) for line in (directory / "instances.jsonl").open()]
        shapes = {(r["frame"], r["instance"]): r for r in items}
        with np.load(directory / "anomaly.npz") as points:
            unknown += int(np.count_nonzero(points["instance"] == 0))
            for iid in np.unique(points["instance"]):
                if iid == 0:
                    continue
                rows = [dict(frame=f, count=0, in_range=0) for f in range(frame_count)]
                chosen = points["instance"] == iid
                for f in np.unique(points["frame"][chosen]):
                    mask = chosen & (points["frame"] == f)
                    row = rows[int(f)]
                    row.update(
                        count=int(mask.sum()),
                        in_range=int(points["inside"][mask].sum()),
                        range=finite_median(points["distance"][mask]),
                    )
                    for key in (
                        "same_instance_neighbor_distance",
                        "background_neighbors",
                        "intensity_contrast",
                    ):
                        row[key] = finite_median(points[key][mask])
                    row["nearest_road_fraction"] = float(
                        np.mean(points["nearest_normal_semantic"][mask] == 40)
                    )
                    item = shapes[(int(f), int(iid))]
                    for key in (
                        "length",
                        "width",
                        "height",
                        "linearity",
                        "planarity",
                        "scattering",
                        "ground_height_median",
                    ):
                        row[key] = item[key]
                stats, hist = distributions(rows)
                worlds.append(
                    dict(
                        sequence=int(directory.name),
                        instance=int(iid),
                        trajectory=stats,
                        histograms=hist,
                    )
                )
    if len({w["sequence"] for w in worlds}) != 19:
        raise ValueError("matching requires the complete current 19-sequence profile")
    mean = {
        key: np.mean([w["histograms"][key] for w in worlds], axis=0).tolist()
        for key in worlds[0]["histograms"]
    }
    envelope = {}
    for key, margin in margins.items():
        values = [
            w["trajectory"][key] for w in worlds if w["trajectory"][key] is not None
        ]
        envelope[key] = [max(0.0, min(values) - margin), max(values) + margin]
    return dict(
        worlds=worlds,
        histograms=mean,
        trajectory_envelope=envelope,
        unknown_instance_points=unknown,
        weighting="equal_object_trajectories_then_equal_feature_groups",
    )


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


def sample_support(sequence, rng, config, footprint_radius):
    poses = np.stack([sequence.lidar_pose(f) for f in sequence.frame_ids])
    for _ in range(config["maximum_support_attempts"]):
        frame = sequence[int(rng.integers(len(sequence)))]
        ground_slots = frame.real_slots[
            np.isin(frame.labels.semantic[frame.real_slots], config["semantics"])
        ]
        xyz = frame.xyzi[ground_slots, :3].astype(np.float64)
        distance = np.linalg.norm(xyz, axis=1)
        choices = np.flatnonzero(
            (distance >= config["proposal_range_m"][0])
            & (distance <= config["proposal_range_m"][1])
        )
        if not len(choices):
            continue
        selected = int(rng.choice(choices))
        center = xyz[selected, :2]
        nearby = np.linalg.norm(xyz[:, :2] - center, axis=1) <= config["plane_radius_m"]
        points = xyz[nearby]
        if len(points) < config["minimum_ground_points"]:
            continue
        design = np.column_stack((points[:, :2] - center, np.ones(len(points))))
        coefficients = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
        residual = points[:, 2] - design @ coefficients
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual) <= max(0.05, 3 * mad)
        if keep.sum() < config["minimum_ground_points"]:
            continue
        coefficients = np.linalg.lstsq(design[keep], points[keep, 2], rcond=None)[0]
        rmse = float(
            np.sqrt(np.mean((points[keep, 2] - design[keep] @ coefficients) ** 2))
        )
        try:
            hull = ConvexHull(design[keep, :2])
        except QhullError:
            continue
        if (
            rmse > config["maximum_plane_rmse_m"]
            or np.linalg.norm(coefficients[:2])
            > np.tan(np.deg2rad(config["maximum_slope_degrees"]))
            or np.linalg.eigvalsh(np.cov(design[keep, :2], rowvar=False))[0]
            < config["minimum_xy_eigenvalue"]
            or np.any(hull.equations[:, -1] > -footprint_radius)
        ):
            continue
        rotation, translation = frame.lidar_pose[:3, :3], frame.lidar_pose[:3, 3]
        anchor = np.r_[center, coefficients[2]] @ rotation.T + translation
        trajectory_range = np.linalg.norm(poses[:, :2, 3] - anchor[:2], axis=1)
        if (
            not config["minimum_trajectory_clearance_m"]
            <= trajectory_range.min()
            <= config["maximum_closest_trajectory_distance_m"]
        ):
            continue
        normal = rotation @ np.r_[-coefficients[:2], 1.0]
        normal /= np.linalg.norm(normal)
        slot = int(ground_slots[selected])
        pool = QualifiedSupportPool(
            np.array([0]),
            np.array([frame.labels.semantic[slot]]),
            np.array([frame.frame_id]),
            np.array([slot]),
            np.array([distance[selected]]),
            np.array([0], np.uint64),
            anchor[None],
            normal[None],
            np.array([-normal @ anchor]),
            sequence.spec.sequence_id,
        )
        return pool, dict(
            plane_rmse_m=rmse,
            plane_support=int(keep.sum()),
            closest_trajectory_distance_m=float(trajectory_range.min()),
        )
    raise ValueError("no_single_scan_support_with_full_trajectory_clearance")


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


def make_world(sequence, seed, config):
    rng = np.random.default_rng(seed)
    shape_config = config["shape"]
    length = float(np.exp(rng.uniform(*np.log(shape_config["length_m"]))))
    width = max(0.05, length * float(rng.uniform(*shape_config["width_ratio"])))
    height = float(np.exp(rng.uniform(*np.log(shape_config["height_m"]))))
    shape = ShapeSpec(
        ((length / 2, width / 2, height / 2),),
        ((0.0, 0.0, 0.0),),
        (tuple(rng.uniform(*shape_config["exponents"], 2)),),
        (0.0,),
        ("union",),
    )
    material = MaterialSpec(
        float(rng.uniform(*shape_config["material_quantile"])),
        float(rng.uniform(*shape_config["material_roughness"])),
    )
    pool, support = sample_support(
        sequence, rng, config["placement"], shape.bound_radius_m
    )
    # Placement reuses the renderer's support/contact calculation. Full-sequence collisions follow below.
    frame = sequence[int(pool.frames[0])]
    real = frame.real_slots
    normal = real[
        (frame.labels.semantic[real] != 0)
        & ~np.isin(frame.labels.semantic[real], GROUND)
    ]
    points = (
        frame.xyzi[normal, :3].astype(float) @ frame.lidar_pose[:3, :3].T
        + frame.lidar_pose[:3, 3]
    )
    obstacles = ObservedObstacleIndex(
        points, (np.uint64(frame.frame_id) << np.uint64(32)) | normal.astype(np.uint64)
    )
    item, placement = place_object(
        shape,
        material,
        pool,
        obstacles,
        object_id=1,
        label="anomaly-proxy",
        proposal_namespace=f"single-frame-v1/{seed}",
        proposal_stream=0,
        yaw_rad=float(rng.uniform(*shape_config["yaw_rad"])),
        material_seed=seed,
        yaw_seed=seed,
        shape_seed=seed,
        proposal_rows=[0],
        maximum_candidates=1,
    )
    return WorldSpec(seed, sequence.spec.sequence_id, (item,)), dict(
        **support,
        placement=clean_json(placement.to_dict()),
        support_plane=dict(
            anchor_world_m=pool.anchors_world_m[0].tolist(),
            normal_world=pool.normals_world[0].tolist(),
            offset=float(pool.offsets[0]),
        ),
    )


def generate_candidate(data_root, output, split, index, config, target, identity):
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
        world, placement = make_world(sequence, seed, config)
    except ValueError as error:
        report.update(status="rejected", reason=str(error), frames=[])
        _atomic_json(directory / "manifest.json", report)
        return report
    grid, sensor = load_sensor_calibration(Path(output) / "calibration.pt")
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
                    collision = dict(frame=source.frame_id, minimum_sdf_m=minimum)
                    return
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
    stats, hist = distributions(rows) if rows and collision is None else ({}, {})
    if collision is None:
        if len(rows) != len(sequence):
            raise ValueError("world stopped before its final source frame")
        if (
            stats["eligible_frames"]
            < config["qualification"]["minimum_eligible_frames"]
        ):
            reason = "no_eligible_anomaly_observation"
        for key, (low, high) in target["trajectory_envelope"].items():
            if stats[key] is None or not low <= stats[key] <= high:
                reason = f"trajectory_outside_reference_envelope:{key}"
                break
    report.update(
        status="qualified" if reason is None else "rejected",
        reason=reason,
        collision=collision,
        world_identity=world.identity,
        frames=rows,
        trajectory=stats,
        histograms=hist,
        seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        reconstruction="all_rendered_frames_equal_xyzi_packed_labels_slots_and_masks",
    )
    if reason is not None:
        frame_directory = directory / "frames"
        if frame_directory.exists():
            shutil.rmtree(frame_directory)
    _atomic_json(directory / "manifest.json", report)
    return report


def discrepancy(histograms, target):
    return {
        key: float(0.5 * np.abs(np.asarray(histograms[key]) - target[key]).sum())
        for key in target
    }


def select_worlds(reports, number, target):
    reports = sorted(
        (r for r in reports if r["status"] == "qualified"), key=lambda r: r["index"]
    )
    if len(reports) < number:
        raise RuntimeError(
            f"only {len(reports)} complete worlds qualified; {number} required"
        )
    keys = tuple(target)
    matrix = np.array(
        [np.concatenate([r["histograms"][k] for k in keys]) for r in reports]
    )
    reference_vector = np.concatenate([target[k] for k in keys])

    def loss(total, count):
        return float(np.abs(total / count - reference_vector).sum() / (2 * len(keys)))

    selected, total = [], np.zeros(matrix.shape[1])
    while len(selected) < number:
        best = min(
            (i for i in range(len(reports)) if i not in selected),
            key=lambda i: (
                loss(total + matrix[i], len(selected) + 1),
                reports[i]["index"],
            ),
        )
        selected.append(best)
        total += matrix[best]
    # A deterministic finite descent improves the aggregate without changing any world.
    while True:
        improvement = None
        current = loss(total, number)
        for position, old in enumerate(selected):
            for new in range(len(reports)):
                if new in selected:
                    continue
                value = loss(total - matrix[old] + matrix[new], number)
                if value < current - 1e-12:
                    current, improvement = value, (position, old, new)
        if improvement is None:
            break
        position, old, new = improvement
        selected[position] = new
        total += matrix[new] - matrix[old]
    chosen = sorted((reports[i] for i in selected), key=lambda r: r["index"])
    mean = {
        key: np.mean([r["histograms"][key] for r in chosen], axis=0).tolist()
        for key in keys
    }
    before = {
        key: np.mean([r["histograms"][key] for r in reports], axis=0).tolist()
        for key in keys
    }
    return chosen, dict(
        selected_histograms=mean,
        qualified_candidate_histograms=before,
        selected_total_variation=discrepancy(mean, target),
        qualified_candidate_total_variation=discrepancy(before, target),
        mean_total_variation=loss(total, number),
    )


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
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/jasongao/Data/STU")
    )
    parser.add_argument("--config", type=Path, default=Path("protocol/v1.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--pilot",
        type=int,
        help="process this many candidates per source, retaining valid frozen frames for the full run",
    )
    args = parser.parse_args()
    if args.workers < 1 or (args.pilot is not None and args.pilot < 1):
        parser.error("workers and pilot candidates must be positive")
    config = json.loads(args.config.read_text())
    output = args.output or Path(config["dataset"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    science = {
        key: config[key]
        for key in (
            "seed",
            "dataset",
            "calibration",
            "placement",
            "shape",
            "qualification",
        )
    }
    implementation = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path(__file__),
            Path(__file__).with_name("render.py"),
            Path(__file__).with_name("data.py"),
            Path(__file__).with_name("profile.py"),
        )
    }
    identity = hashlib.sha256(
        json.dumps(
            dict(science=science, implementation=implementation), sort_keys=True
        ).encode()
    ).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["status"] == "frozen":
            raise FileExistsError(
                "the dataset is already frozen; use its single-frame reader"
            )
        if existing["configuration_identity"] != identity:
            raise ValueError(
                "unfinished candidate data belongs to different generation inputs"
            )
    disk = host_disk()
    # Hard upper bound: every slot could change, including temporary atomic files.
    sizes = {}
    for split, item in config["dataset"]["splits"].items():
        sequence = STUSequence.open(
            args.data_root,
            protocol=load_protocol(),
            partition="train",
            sequence_id=item["source_sequence"],
            label_mode="required",
        )
        slots = sum(p.stat().st_size // 16 for p in sequence._scan_paths.values())
        sizes[split] = slots * 32 * item["candidates"]
    # Actual deltas are sparse; enforce a bounded working allocation during generation.
    active_peak = args.workers * (256 * 1024**2 + 2 * 393216 * 32)
    allocation = min(
        20_000_000_000, disk["SizeRemaining"] - disk["reserve_bytes"] - active_peak
    )
    if allocation < 1_000_000_000:
        raise OSError("insufficient host space for the bounded generation allocation")
    manifest = dict(
        format="stu-frozen-dataset",
        status="building",
        configuration_identity=identity,
        configuration=science,
        implementation=implementation,
        splits={},
        execution=dict(
            workers=args.workers,
            host_before=disk,
            allocation_bytes=allocation,
            active_candidate_peak_bytes=active_peak,
            dense_uncompressed_upper_bytes=sizes,
        ),
        generation={},
    )
    _atomic_json(manifest_path, manifest)
    manifest["calibration"] = prepare_calibration(args.data_root, output, config)
    target = reference(
        config["qualification"]["reference"],
        config["qualification"]["envelope_margins"],
    )
    _atomic_json(output / "reference.json", clean_json(target))
    started = time.perf_counter()
    reports = {split: [] for split in config["dataset"]["splits"]}
    jobs = []
    for split, item in config["dataset"]["splits"].items():
        limit = (
            min(args.pilot, item["candidates"]) if args.pilot else item["candidates"]
        )
        for index in range(limit):
            path = output / split / f"candidate_{index:03d}" / "manifest.json"
            if path.exists():
                report = json.loads(path.read_text())
                if report["configuration_identity"] != identity:
                    raise ValueError("cached candidate belongs to another generator")
                reports[split].append(report)
            else:
                jobs.append((split, index))
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                generate_candidate,
                str(args.data_root),
                str(output),
                split,
                index,
                config,
                target,
                identity,
            ): (split, index)
            for split, index in jobs
        }
        for future in as_completed(futures):
            split, index = futures[future]
            report = future.result()
            reports[split].append(report)
            print(
                json.dumps(
                    dict(
                        event="candidate",
                        split=split,
                        index=index,
                        status=report["status"],
                        reason=report["reason"],
                        trajectory=report.get("trajectory"),
                        seconds=report.get("seconds"),
                        peak_rss_bytes=report.get("peak_rss_bytes"),
                    )
                ),
                flush=True,
            )
            if sum(map(len, reports.values())) % 8 == 0:
                remaining = host_disk()
                used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                if (
                    used > allocation
                    or remaining["SizeRemaining"]
                    < remaining["reserve_bytes"] + 2_000_000_000
                ):
                    for pending in futures:
                        pending.cancel()
                    raise OSError(
                        "bounded synthetic working allocation reached; completed candidates are retained"
                    )
    manifest["execution"]["seconds"] = time.perf_counter() - started
    for split, values in reports.items():
        manifest["generation"][split] = [
            dict(
                index=r["index"],
                seed=r["seed"],
                status=r["status"],
                reason=r["reason"],
                trajectory=r.get("trajectory"),
                collision=r.get("collision"),
                seconds=r.get("seconds"),
                peak_rss_bytes=r.get("peak_rss_bytes"),
            )
            for r in sorted(values, key=lambda r: r["index"])
        ]
    if args.pilot:
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                dict(event="pilot_complete", status="building", directory=str(output))
            ),
            flush=True,
        )
        return
    selected = {}
    for split, item in config["dataset"]["splits"].items():
        chosen, comparison = select_worlds(
            reports[split], item["worlds"], target["histograms"]
        )
        selected[split] = chosen
        manifest["splits"][split] = dict(
            source_sequence=item["source_sequence"],
            samples=item["worlds"] * item["frames_per_world"],
            worlds=[],
            comparison=comparison,
        )
    for split, chosen in selected.items():
        for world_index, report in enumerate(chosen):
            source = output / split / f"candidate_{report['index']:03d}"
            destination = output / split / f"world_{world_index:03d}"
            source.rename(destination)
            manifest["splits"][split]["worlds"].append(
                dict(
                    path=str(destination.relative_to(output)),
                    world_identity=report["world_identity"],
                    candidate_index=report["index"],
                )
            )
        for directory in (output / split).glob("candidate_*"):
            shutil.rmtree(directory)
    manifest["status"] = "frozen"
    manifest["execution"]["host_after"] = host_disk()
    manifest["execution"]["bytes_on_disk"] = sum(
        p.stat().st_size for p in output.rglob("*") if p.is_file()
    )
    _atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            dict(
                event="frozen",
                samples={s: x["samples"] for s, x in manifest["splits"].items()},
                bytes_on_disk=manifest["execution"]["bytes_on_disk"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
