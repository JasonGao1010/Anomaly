"""Source-only nuScenes observations and visible-surface road insertions.

Only measured surface triangles and existing receiver returns are used. This is
a local visibility approximation, not a complete object or firing simulation.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import ijson
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.spatial.transform import Rotation

from .data import SOURCE_VERSION, identity, nuscenes_mapping, write_json


RECIPE = dict(
    supervision="Only inserted debris is positive; original debris is ignored",
    support="Observed connected semantic patches; open adjacent-beam triangles",
    visibility="Nearest triangle intersections on existing measured directions",
    normal_control="The same insertion applied to measured normal road users",
    pairing="Unmodified original returns are the normal auxiliary reference",
    source_view="Rigid road-tangent rotation preserves the primary viewing side and road-relative height",
    source_range_ratio=[.75, 1.25], scale=1., positive_fraction=.75,
    min_donor_points=8, max_azimuth_steps=2.5, max_adjacent_beams=1,
    depth_jump_m=.2, depth_jump_per_angular_distance=2.,
    support_residual_m=.12, collision_margin_m=.15, placement_attempts=16,
    donor_identity="Semantic-label connected patch without instance annotations; spatial merging is not instance ground truth",
    evaluation="anomaly >= 5 marks official-style eligibility; training retains all records",
)
_DONORS = None
_MAPPING = None
_OUTPUT = None


def _rows(path):
    with Path(path).open("rb") as stream:
        yield from ijson.items(stream, "item", use_float=True)


def _pose(row):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(row["rotation"])[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = row["translation"]
    return matrix


def sources(root):
    """Read every labeled official keyframe, independently of prior manifests."""
    from nuscenes.utils.splits import create_splits_scenes
    root = Path(root).resolve(strict=True)
    meta = root / "v1.0-trainval"
    official = create_splits_scenes()
    membership = {name: split for split in ("train", "val") for name in official[split]}
    scenes = {r["token"]: r for r in _rows(meta / "scene.json")}
    if set(membership) != {r["name"] for r in scenes.values()}:
        raise ValueError("the complete official nuScenes trainval scenes are required")
    samples = {r["token"]: r for r in _rows(meta / "sample.json")}
    sample_index = {token: i for i, token in enumerate(sorted(samples))}
    labels = {r["sample_data_token"]: r["filename"] for r in _rows(meta / "lidarseg.json")}
    selected = []
    for row in _rows(meta / "sample_data.json"):
        if row["token"] in labels:
            if not row["is_key_frame"] or not row["filename"].startswith("samples/LIDAR_TOP/"):
                raise ValueError("lidarseg must refer to original LIDAR_TOP keyframes")
            selected.append(row)
    if len(selected) != len(labels) or len(selected) != len(samples):
        raise ValueError("every original keyframe must have exactly one lidarseg observation")
    needed = {r["ego_pose_token"] for r in selected}
    poses = {r["token"]: _pose(r) for r in _rows(meta / "ego_pose.json") if r["token"] in needed}
    calibrations = {r["token"]: _pose(r) for r in _rows(meta / "calibrated_sensor.json")}
    records = {"train": [], "val": []}
    for row in selected:
        sample = samples[row["sample_token"]]
        scene = scenes[sample["scene_token"]]
        split = membership[scene["name"]]
        records[split].append(dict(
            source="nuscenes", scene=scene["name"], sample_token=sample["token"], token=row["token"],
            log_token=scene["log_token"], timestamp=int(row["timestamp"]),
            frame=sample_index[sample["token"]], scan=str(root / row["filename"]),
            label=str(root / labels[row["token"]]), group="normal_nuscenes", subset=split,
            pose=(poses[row["ego_pose_token"]] @ calibrations[row["calibrated_sensor_token"]]).tolist()))
    for split in records:
        records[split].sort(key=lambda r: (r["scene"], r["timestamp"], r["token"]))
    logs = {split: {r["log_token"] for r in rows} for split, rows in records.items()}
    if logs["train"] & logs["val"]:
        raise ValueError("official partitions share acquisition logs")
    return records, nuscenes_mapping(root)


def _read(record):
    raw = np.fromfile(record["scan"], dtype="<f4")
    labels = np.fromfile(record["label"], dtype=np.uint8)
    if raw.size != 5 * len(labels) or not len(labels):
        raise ValueError("raw nuScenes point and lidarseg counts differ")
    raw = raw.reshape(-1, 5)
    if not np.isfinite(raw).all() or np.any((raw[:, 4] < 0) | (raw[:, 4] > 31)):
        raise ValueError("invalid original nuScenes observation")
    return raw, labels


def _counts(raw, labels, mapping):
    distance = np.linalg.norm(raw[:, :3], axis=1)
    actual = np.any(raw[:, :3] != 0, axis=1)
    normal = (mapping[labels] == 1) & actual & (distance >= 2.5) & (distance <= 50.)
    return dict(points=int(actual.sum()), slots=len(raw), normal=int(normal.sum()), anomaly=0, eligible=False)


def _components(xyz):
    if len(xyz) < RECIPE["min_donor_points"]:
        return []
    radius = np.clip(.035 * np.median(np.linalg.norm(xyz, axis=1)), .25, .8)
    pairs = cKDTree(xyz).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz)))
    count, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == label) for label in range(count)
            if np.count_nonzero(labels == label) >= RECIPE["min_donor_points"]]


def _triangles(xyz, rings, beam_rank, azimuth_step):
    distance = np.linalg.norm(xyz, axis=1)
    direction = xyz / distance[:, None]
    center = np.arctan2(xyz[:, 1].mean(), xyz[:, 0].mean())
    azimuth = (np.arctan2(xyz[:, 1], xyz[:, 0]) - center + np.pi) % (2 * np.pi) - np.pi
    elevation = np.arcsin(np.clip(direction[:, 2], -1, 1))
    angular = np.column_stack((azimuth, elevation))
    if np.ptp(elevation) < 1e-5 or np.ptp(azimuth) < 1e-5:
        return np.empty((0, 3), dtype=np.int32)
    try:
        triangles = Delaunay(angular).simplices
    except QhullError:
        return np.empty((0, 3), dtype=np.int32)
    rank = beam_rank[rings]
    valid = (np.ptp(rank[triangles], axis=1) <= RECIPE["max_adjacent_beams"])
    valid &= np.ptp(azimuth[triangles], axis=1) <= RECIPE["max_azimuth_steps"] * azimuth_step
    # Never span range discontinuities or fill unobserved angular holes.
    for first, second in ((0, 1), (1, 2), (2, 0)):
        a, b = triangles[:, first], triangles[:, second]
        chord = np.linalg.norm(direction[a] - direction[b], axis=1) * np.minimum(distance[a], distance[b])
        valid &= abs(distance[a] - distance[b]) <= RECIPE["depth_jump_m"] + RECIPE["depth_jump_per_angular_distance"] * chord
    area = np.linalg.norm(np.cross(xyz[triangles[:, 1]] - xyz[triangles[:, 0]],
                                   xyz[triangles[:, 2]] - xyz[triangles[:, 0]]), axis=1)
    return triangles[valid & (area > 1e-5)].astype(np.int32)


def _ground(points, center, radius):
    selected = points[np.linalg.norm(points[:, :2] - center, axis=1) <= radius]
    if len(selected) < 8:
        return None
    design = np.column_stack((selected[:, :2] - center, np.ones(len(selected))))
    plane, _, rank, _ = np.linalg.lstsq(design, selected[:, 2], rcond=None)
    if rank < 3 or np.quantile(abs(design @ plane - selected[:, 2]), .9) > RECIPE["support_residual_m"]:
        return None
    if np.linalg.norm(plane[:2]) > .3:
        return None
    return plane


def _basis(view, plane):
    """An orthonormal road frame preserves metric shape on sloping surfaces."""
    normal = np.r_[-plane[:2], 1.]
    normal /= np.linalg.norm(normal)
    forward = np.r_[view, plane[:2] @ view]
    forward /= np.linalg.norm(forward)
    return np.column_stack((forward, np.cross(normal, forward), normal))


def _scene_donors(task):
    records, mapping = task
    lookup = np.asarray([r["target"] for r in mapping], np.uint8)
    names = {r["name"]: r["raw"] for r in mapping}
    ground_ids = [r["raw"] for r in mapping if r["name"] in ("flat.driveable_surface", "flat.sidewalk", "flat.terrain")]
    control_ids = [r["raw"] for r in mapping if r["target"] and
                   (r["name"].startswith("human.") or r["name"] in ("vehicle.car", "vehicle.bicycle", "vehicle.motorcycle"))]
    rows, debris, controls = [], [], {}
    for record in records:
        raw, labels = _read(record)
        if np.any(labels >= len(lookup)):
            raise ValueError("unknown original lidarseg category")
        rows.append(dict(record, **_counts(raw, labels, lookup)))
        transform = np.asarray(record["pose"])
        world = raw[:, :3].astype(float) @ transform[:3, :3].T + transform[:3, 3]
        ground = world[np.isin(labels, ground_ids)]
        if len(ground) < 8:
            continue
        distance = np.linalg.norm(raw[:, :3], axis=1)
        real = distance > 2.5
        rings = raw[:, 4].astype(int)
        elevation = np.arcsin(np.clip(raw[:, 2] / np.maximum(distance, 1e-8), -1, 1))
        beam_pitch = [np.median(elevation[real & (rings == i)]) if np.any(real & (rings == i)) else -10. + i for i in range(32)]
        rank = np.argsort(np.argsort(beam_pitch))
        step = 2 * np.pi / max(1, len(raw) / 32)
        for category in [names["movable_object.debris"], *control_ids]:
            slots = np.flatnonzero((labels == category) & real & (distance <= 50.))
            parts = _components(raw[slots, :3])
            is_debris = category == names["movable_object.debris"]
            if not is_debris and parts:
                parts = [max(parts, key=len)]
            for component in parts:
                selected = slots[component]
                triangles = _triangles(raw[selected, :3].astype(float), rings[selected], rank, step)
                if not len(triangles):
                    continue
                xyz = world[selected]
                center = np.median(xyz[:, :2], axis=0)
                radius = max(.75, float(np.linalg.norm(xyz[:, :2] - center, axis=1).max()) + .5)
                if radius > 5.:
                    continue
                plane = _ground(ground, center, radius + 1.)
                if plane is None:
                    continue
                origin = np.r_[center, plane[2]]
                vector = center - transform[:2, 3]
                source_range = float(np.linalg.norm(vector))
                if source_range < 2.5:
                    continue
                forward = vector / source_range
                basis = _basis(forward, plane)
                local = (xyz - origin) @ basis
                if local[:, 2].min() < -.2 or local[:, 2].max() < .15:
                    continue
                donor = dict(id=f"{record['token']}:{int(selected.min())}", scene=record["scene"],
                    token=record["token"], sample_token=record["sample_token"], log_token=record["log_token"],
                    scan=record["scan"], category=mapping[category]["name"],
                    slots=selected.astype(np.int32), xyz=local.astype(np.float32),
                    intensity=(raw[selected, 3] / 255.).astype(np.float32), triangles=triangles,
                    center=origin, basis=basis.tolist(), support_plane=plane.tolist(),
                    range=source_range, kind="anomaly" if is_debris else "control")
                if is_debris:
                    previous = next((i for i, old in enumerate(debris)
                                     if np.linalg.norm(old["center"] - origin) < .75), None)
                    if previous is None:
                        debris.append(donor)
                    elif len(triangles) > len(debris[previous]["triangles"]):
                        debris[previous] = donor
                elif category not in controls or len(triangles) > len(controls[category]["triangles"]):
                    controls[category] = donor
    return rows, debris + list(controls.values())


def _intersections(directions, vertices, triangles, intensities):
    """Two-sided intersections with measured open facets, with no back-face fill."""
    nearest = np.full(len(directions), np.inf)
    intensity = np.zeros(len(directions))
    for start in range(0, len(triangles), 128):
        face = vertices[triangles[start:start + 128]]
        a, edge1, edge2 = face[:, 0], face[:, 1] - face[:, 0], face[:, 2] - face[:, 0]
        p = np.cross(directions[:, None], edge2[None])
        determinant = np.einsum("rti,ti->rt", p, edge1)
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=abs(determinant) > 1e-10)
        u = -np.einsum("ti,rti->rt", a, p) * inverse
        q = np.cross(-a, edge1)
        v = np.einsum("ri,ti->rt", directions, q) * inverse
        distance = np.einsum("ti,ti->t", edge2, q)[None] * inverse
        valid = (abs(determinant) > 1e-10) & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 0)
        distance = np.where(valid, distance, np.inf)
        choice = distance.argmin(axis=1)
        row = np.arange(len(directions))
        closer = distance[row, choice] < nearest
        chosen = triangles[start:start + 128][choice]
        value = ((1 - u[row, choice] - v[row, choice]) * intensities[chosen[:, 0]] +
                 u[row, choice] * intensities[chosen[:, 1]] + v[row, choice] * intensities[chosen[:, 2]])
        nearest[closer] = distance[row, choice][closer]
        intensity[closer] = value[closer]
    return nearest, intensity


def transplant(raw, labels, record, donor, road_id, rng):
    """Return a sparse foreground replacement, or None for unsupported placement."""
    transform = np.asarray(record["pose"])
    world = raw[:, :3].astype(float) @ transform[:3, :3].T + transform[:3, 3]
    distance = np.linalg.norm(raw[:, :3], axis=1)
    road_mask = labels == road_id
    road = world[road_mask]
    if len(road) < 8:
        return None
    horizontal = np.linalg.norm(road[:, :2] - transform[:2, 3], axis=1)
    near, far = RECIPE["source_range_ratio"]
    candidates = np.flatnonzero((horizontal >= max(2.5, near * donor["range"])) &
                                (horizontal <= min(50., far * donor["range"])))
    if not len(candidates):
        return None
    local = donor["xyz"].astype(float)
    lower, upper = local.min(axis=0), local.max(axis=0)
    radius = max(.75, float(np.linalg.norm(local[:, :2], axis=1).max()) + .5)
    origin_sensor = transform[:3, 3]
    directions = np.divide(raw[:, :3], distance[:, None], out=np.zeros((len(raw), 3), float), where=distance[:, None] > 0)
    for candidate in rng.permutation(candidates)[:RECIPE["placement_attempts"]]:
        center = road[candidate, :2]
        plane = _ground(road, center, radius + .5)
        if plane is None:
            continue
        vector = center - origin_sensor[:2]
        vector /= np.linalg.norm(vector)
        basis = _basis(vector, plane)
        origin = np.r_[center, plane[2]]
        # Normal and anomalous donors share the same footprint collision rule.
        nearby = np.linalg.norm(world[:, :2] - center, axis=1) <= radius
        relative = (world[nearby] - origin) @ basis
        margin = RECIPE["collision_margin_m"]
        occupied = np.all((relative[:, :2] >= lower[:2] - margin) & (relative[:, :2] <= upper[:2] + margin), axis=1)
        occupied &= (relative[:, 2] > -RECIPE["support_residual_m"]) & (relative[:, 2] <= upper[2] + margin)
        occupied &= ~road_mask[nearby]
        if occupied.any():
            continue
        placed_world = local @ basis.T + origin
        vertices = (placed_world - origin_sensor) @ transform[:3, :3]
        centroid = vertices.mean(axis=0)
        extent = np.linalg.norm(vertices - centroid, axis=1).max()
        center_range = np.linalg.norm(centroid)
        covered = np.ones(len(raw), dtype=bool)
        if extent < center_range:
            cone = np.sqrt(max(0., 1 - (extent / center_range) ** 2))
            covered = directions @ (centroid / center_range) >= cone
        slots = np.flatnonzero((distance > 0) & covered)
        if not len(slots):
            continue
        hit, intensity = _intersections(directions[slots], vertices, donor["triangles"], donor["intensity"])
        # Visibility precedes supervision: an out-of-range foreground still
        # occludes its background and remains part of the observed point cloud.
        visible = (hit > 0) & (hit < distance[slots] - 1e-4)
        slots, hit, intensity = slots[visible], hit[visible], intensity[visible]
        if not len(slots):
            continue
        xyzi = np.column_stack((directions[slots] * hit[:, None], intensity)).astype(np.float32)
        return slots.astype(np.int32), xyzi, dict(position_world=origin.tolist(),
            yaw_rad=float(np.arctan2(vector[1], vector[0])), range=float(np.linalg.norm(origin - origin_sensor)),
            source_range=float(donor["range"]), scale=1., basis_world=basis.tolist(),
            support_plane=plane.tolist(), surface="open measured triangles")
    return None


def _initialize(donors, mapping, output):
    global _DONORS, _MAPPING, _OUTPUT
    _DONORS, _MAPPING, _OUTPUT = donors, mapping, Path(output)


def _generate(record):
    seed = int.from_bytes(hashlib.sha256((SOURCE_VERSION + record["token"]).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    kind = "anomaly" if seed % 4 else "control"
    donors = [d for d in _DONORS[kind] if d["scene"] != record["scene"]]
    if not donors:
        return record, None
    donor = donors[int(rng.integers(len(donors)))]
    raw, labels = _read(record)
    road_id = next(row["raw"] for row in _MAPPING if row["name"] == "flat.driveable_surface")
    result = transplant(raw, labels, record, donor, road_id, rng)
    if result is None:
        return record, None
    slots, xyzi, placement = result
    delta_path = _OUTPUT / record["subset"] / f"{record['token']}.npz"
    value = 2 if kind == "anomaly" else 1
    np.savez_compressed(delta_path, slots=slots, xyzi=xyzi,
                        labels=np.full(len(slots), value, dtype=np.uint32), token=np.asarray(record["token"]))
    lookup = np.asarray([r["target"] for r in _MAPPING], np.uint8)
    old_range = np.linalg.norm(raw[slots, :3], axis=1)
    removed_normal = int(((lookup[labels[slots]] == 1) & (old_range >= 2.5) & (old_range <= 50.)).sum())
    stored_range = np.linalg.norm(xyzi[:, :3], axis=1)
    supervised = int(((stored_range >= 2.5) & (stored_range <= 50.)).sum())
    normal = record["normal"] - removed_normal + (supervised if value == 1 else 0)
    anomaly = supervised if value == 2 else 0
    inserted = dict(record, group=f"{kind}_nuscenes", delta=str(delta_path),
        normal=normal, anomaly=anomaly, eligible=anomaly >= 5, placement=placement,
        donor=donor["id"], source_scene=donor["scene"], range=placement["range"])
    return record, inserted


def build(root, output, workers):
    """Build both official source splits; never read STU or an older generated set."""
    output = Path(output).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; existing generated observations are not reused")
    records, mapping = sources(root)
    manifests = {}
    for split in ("train", "val"):
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        scenes = {}
        for record in records[split]:
            scenes.setdefault(record["scene"], []).append(record)
        original, donors = [], []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, (rows, patches) in enumerate(executor.map(_scene_donors, [(rows, mapping) for rows in scenes.values()]), 1):
                original.extend(rows)
                donors.extend(patches)
                if index % 25 == 0 or index == len(scenes):
                    print(f"nuScenes {split} original scenes {index}/{len(scenes)}; observed donor patches {len(donors)}", flush=True)
        pools = {kind: [donor for donor in donors if donor["kind"] == kind] for kind in ("anomaly", "control")}
        if any(not pool for pool in pools.values()):
            raise ValueError(f"{split} has no supported measured anomaly or control surfaces")
        original.sort(key=lambda r: (r["scene"], r["timestamp"], r["token"]))
        rows, attempted = [], {"anomaly": 0, "control": 0}
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize, initargs=(pools, mapping, str(output))) as executor:
            for index, (record, inserted) in enumerate(executor.map(_generate, original, chunksize=16), 1):
                rows.append(record)
                if inserted is not None:
                    rows.append(inserted)
                    attempted[inserted["group"].split("_")[0]] += 1
                if index % 1000 == 0 or index == len(original):
                    print(f"nuScenes {split} insertions {index}/{len(original)}: {attempted}", flush=True)
        catalog = {donor["id"]: dict(
            {key: donor[key] for key in ("scene", "token", "sample_token", "log_token", "scan", "category",
                                         "kind", "basis", "support_plane", "range")},
            center=donor["center"].tolist(), slots=donor["slots"].tolist()) for donor in donors}
        manifest = dict(version=SOURCE_VERSION, kind=split, mapping=mapping, records=rows, donors=catalog,
            directory=str(directory), root=str(Path(root).resolve()),
            split=dict(name=split, scenes=sorted(scenes), logs=sorted({r["log_token"] for r in original})),
            recipe=dict(RECIPE, donor_patches={kind: len(pool) for kind, pool in pools.items()},
                        original_frames=len(original), successful_insertions=attempted))
        manifest["sha256"] = identity(manifest)
        write_json(output / f"{split}.json", manifest, indent=None)
        manifests[split] = manifest
    return manifests
