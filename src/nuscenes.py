"""Source-only nuScenes observations and visible-surface road insertions.

Only measured surface triangles and existing receiver returns are used. This is
a local visibility approximation, not a complete object or firing simulation.
"""

from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
import hashlib
import json
import itertools
from pathlib import Path
import shutil
import time

import ijson
import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.transform import Rotation

from .data import SOURCE_VERSION, identity, nuscenes_mapping, write_json


RECIPE = dict(
    revision="instance-coverage-1",
    supervision="Only inserted, separable debris/pushable road obstacles are positive; original void remains ignored",
    support="Unique instance-box and semantic-label intersection; open adjacent-beam triangles",
    visibility="Nearest triangle intersections on existing measured directions",
    normal_control="The same insertion applied to measured normal road users",
    pairing="Unmodified original returns are the normal auxiliary reference",
    source_view="Rigid road-tangent rotation preserves the primary viewing side and road-relative height",
    scale=1., minimum_target_range="source observation range; no closer-view surface completion",
    distance_edges=[2.5, 10., 20., 30., 40., 50.], max_views=6, requests_per_view_bin=2,
    train_requests=dict(reference=14065, coverage=14065, control=14065),
    val_requests=dict(reference=2000, coverage=2000, control=2000),
    min_donor_points=8, max_azimuth_steps=2.5, max_adjacent_beams=1,
    depth_jump_m=.2, depth_jump_per_angular_distance=2.,
    support_residual_m=.12, collision_margin_m=.15, placement_attempts=16,
    donor_identity="Official instance token within scene; cross-scene identity is not guaranteed",
    evaluation="anomaly >= 5 eligibility; main metrics use reference role; training keeps 1-4 positive points",
    reference_distance="Sample a bin proportional to the receiver's measured road-return counts; not natural anomaly prevalence",
    control_match="Same range bin, visible-count ratio in [0.5,2], angular-span ratio in [0.5,2] where measurable, overlapping intensity q10-q90",
    normal_library="Deterministic instance subset: at most 256 per mapped normal category in train, 64 in val",
    allocation="Interleave reference and coverage requests; balance annotated instance, then view-bin use; geometry groups unavailable",
    allocation_ties="First 64 SHA256 bits of donor_id XOR request seed; matched controls use donor hash",
    shape_holdout="Unavailable without validated cross-instance geometry labels; keep all 600 requested slots unfilled",
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
    if {r["sample_token"] for r in selected} != set(samples):
        raise ValueError("labeled observations must cover each sample exactly once")
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
    counts = Counter(r["scene"] for rows in records.values() for r in rows)
    if len(counts) != len(scenes) or any(counts[s["name"]] != s["nbr_samples"] for s in scenes.values()):
        raise ValueError("labeled keyframe counts differ from original scene metadata")
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


def _background_scene(task):
    records, mapping = task
    lookup = np.asarray([r["target"] for r in mapping], np.uint8)
    counts, categories = Counter(), np.zeros(32, dtype=np.int64)
    ranges = np.zeros(5, dtype=np.int64)
    rows = []
    for record in records:
        raw, labels = _read(record)
        if np.any(labels >= len(lookup)):
            raise ValueError("unknown original lidarseg label")
        row = dict(record, role="original", **_counts(raw, labels, lookup))
        rows.append(row)
        actual = np.any(raw[:, :3] != 0, axis=1)
        distance = np.linalg.norm(raw[:, :3], axis=1)
        valid = actual & (distance >= 2.5) & (distance <= 50.)
        normal = valid & (lookup[labels] == 1)
        counts.update({key: row[key] for key in ("points", "slots", "normal")})
        counts.update(ignored_in_range=int((valid & ~normal).sum()),
                      outside_range=int((actual & ~valid).sum()), empty_slots=int((~actual).sum()))
        categories += np.bincount(labels, minlength=32)
        ranges += np.histogram(distance[normal], [2.5, 10., 20., 30., 40., 50.])[0]
    return rows, counts, categories, ranges


def _backgrounds(root, output, workers, records, mapping):
    """Preserve complete raw scans; only trusted in-range points are supervised."""
    logs = {r["token"]: r["location"] for r in _rows(Path(root) / "v1.0-trainval/log.json")}
    manifests = {}
    for split, original in records.items():
        scenes = defaultdict(list)
        for record in original:
            scenes[record["scene"]].append(record)
        rows, counts = [], Counter()
        categories, ranges = np.zeros(32, dtype=np.int64), np.zeros(5, dtype=np.int64)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_background_scene, [(r, mapping) for r in scenes.values()]):
                observed, totals, raw_counts, distance_counts = result
                rows.extend(observed)
                counts.update(totals)
                categories += raw_counts
                ranges += distance_counts
        # An ignored point remains observed context, never an implicit normal label.
        assert counts["points"] == counts["normal"] + counts["ignored_in_range"] + counts["outside_range"]
        summary = dict(frames=len(rows), scenes=len(scenes), logs=len({r["log_token"] for r in rows}),
                       **counts, anomaly=0, raw_class_points=categories.tolist(),
                       normal_distance_edges_m=[2.5, 10., 20., 30., 40., 50.],
                       normal_distance_points=ranges.tolist(),
                       location_frames=dict(Counter(logs[r["log_token"]] for r in rows)))
        manifest = dict(version=SOURCE_VERSION, kind=split, root=str(Path(root).resolve()),
            directory=str(output), mapping=mapping, records=rows,
            split=dict(name=split, scenes=sorted(scenes), logs=sorted({r["log_token"] for r in rows})),
            recipe=dict(stage="background", partition="Official nuScenes train/val scenes; disjoint acquisition logs",
                input="Complete original measured returns; no point removal by semantic label or supervision range",
                supervision="20 known normal classes within 2.5-50 m; all other points ignored; no positive labels",
                validation="Normal-field and false-positive analysis only; anomaly metrics require later positive examples"),
            summary=summary)
        manifest["sha256"] = identity(manifest)
        manifests[split] = manifest
        print("nuScenes background " + json.dumps(dict(subset=split, **summary)), flush=True)
    for split, manifest in manifests.items():
        write_json(output / f"{split}.json", manifest, indent=None)
    return manifests


def _seed(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "little")


def _inside(world, annotation):
    rotation = _pose(annotation)[:3, :3]
    local = (world - annotation["translation"]) @ rotation
    # nuScenes stores width, length, height; box x is length and box y is width.
    half = np.asarray(annotation["size"])[[1, 0, 2]] * .5
    return np.all(abs(local) <= half + 1e-5, axis=1)


def _instance_slots(world, labels, annotation, category, other_boxes=()):
    selected = (labels == category) & _inside(world, annotation)
    for other in other_boxes:
        if other["instance_token"] != annotation["instance_token"]:
            selected &= ~_inside(world, other)
    return np.flatnonzero(selected)


def _annotations(root, records, mapping):
    meta = Path(root) / "v1.0-trainval"
    categories = {r["token"]: r["name"] for r in _rows(meta / "category.json")}
    names = {r["name"]: r["raw"] for r in mapping}
    normal = {r["name"] for r in mapping if r["target"] == 1 and
              (r["name"].startswith("human.") or r["name"] in (
                  "vehicle.car", "vehicle.bicycle", "vehicle.motorcycle",
                  "movable_object.trafficcone", "movable_object.barrier"))}
    anomaly = {"movable_object.debris", "movable_object.pushable_pullable"}
    instances = {r["token"]: dict(r, category=categories[r["category_token"]])
                 for r in _rows(meta / "instance.json") if categories[r["category_token"]] in normal | anomaly}
    sample = {r["sample_token"]: r for rows in records.values() for r in rows}
    # First establish partition identity before selecting a bounded normal library.
    for row in _rows(meta / "sample_annotation.json"):
        instance = instances.get(row["instance_token"])
        if instance is not None and row["token"] == instance["first_annotation_token"]:
            source = sample[row["sample_token"]]
            instance.update(subset=source["subset"], scene=source["scene"], log_token=source["log_token"])
    selected = {token for token, row in instances.items() if row["category"] in anomaly}
    for split, limit in (("train", 256), ("val", 64)):
        for category in sorted(normal):
            candidates = [token for token, row in instances.items() if row["subset"] == split and row["category"] == category]
            selected.update(sorted(candidates, key=_seed)[:limit])
    all_instances = instances
    instances = {token: dict(row, kind="anomaly" if row["category"] in anomaly else "control")
                 for token, row in instances.items() if token in selected}
    annotations = defaultdict(list)
    for row in _rows(meta / "sample_annotation.json"):
        instance = all_instances.get(row["instance_token"])
        if instance is not None:
            annotations[row["sample_token"]].append(dict(
                {key: row[key] for key in ("token", "instance_token", "translation", "rotation", "size", "num_lidar_pts")},
                category=names[instance["category"]], kind="anomaly" if instance["category"] in anomaly else "control",
                extract=row["instance_token"] in selected))
    return annotations, instances


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


def _ground(points, center, radius, diagnostics=None):
    diagnostics = {} if diagnostics is None else diagnostics
    selected = points[np.linalg.norm(points[:, :2] - center, axis=1) <= radius]
    diagnostics["support_points"] = len(selected)
    if len(selected) < 8:
        diagnostics["reason"] = "support_too_sparse"
        return None
    design = np.column_stack((selected[:, :2] - center, np.ones(len(selected))))
    plane, _, rank, _ = np.linalg.lstsq(design, selected[:, 2], rcond=None)
    if rank < 3:
        diagnostics["reason"] = "support_rank_deficient"
        return None
    if np.quantile(abs(design @ plane - selected[:, 2]), .9) > RECIPE["support_residual_m"]:
        diagnostics["reason"] = "support_nonplanar"
        return None
    if np.linalg.norm(plane[:2]) > .3:
        diagnostics["reason"] = "support_slope"
        return None
    diagnostics["reason"] = "supported"
    return plane


def _basis(view, plane):
    """An orthonormal road frame preserves metric shape on sloping surfaces."""
    normal = np.r_[-plane[:2], 1.]
    normal /= np.linalg.norm(normal)
    forward = np.r_[view, plane[:2] @ view]
    forward /= np.linalg.norm(forward)
    return np.column_stack((forward, np.cross(normal, forward), normal))


def _scene_donors(task):
    records, mapping, annotations = task
    lookup = np.asarray([r["target"] for r in mapping], np.uint8)
    names = {r["name"]: r["raw"] for r in mapping}
    ground_ids = [r["raw"] for r in mapping if r["name"] in ("flat.driveable_surface", "flat.sidewalk", "flat.terrain")]
    human_ids = [r["raw"] for r in mapping if r["name"].startswith("human.")]
    rows, observations, rejected = [], defaultdict(list), Counter()
    for record in records:
        raw, labels = _read(record)
        if np.any(labels >= len(lookup)):
            raise ValueError("unknown original lidarseg category")
        rows.append(dict(record, **_counts(raw, labels, lookup)))
        rows[-1]["road_histogram"] = np.histogram(np.linalg.norm(raw[labels == names["flat.driveable_surface"], :3], axis=1),
                                                RECIPE["distance_edges"])[0].tolist()
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
        boxes = annotations.get(record["sample_token"], [])
        for annotation in boxes:
            if not annotation["extract"]:
                continue
            category = annotation["category"]
            instance = annotation["instance_token"]
            if annotation["num_lidar_pts"] < RECIPE["min_donor_points"]:
                rejected["box_too_sparse"] += 1
                continue
            others = [row for row in boxes if row["category"] == category]
            selected = _instance_slots(world, labels, annotation, category, others)
            selected = selected[real[selected] & (distance[selected] <= 50.)]
            if len(selected) < RECIPE["min_donor_points"]:
                rejected["semantic_surface_too_sparse"] += 1
                continue
            # Exclude an occupied cart rather than relabeling its user's returns.
            if annotation["kind"] == "anomaly" and np.any(_inside(world[np.isin(labels, human_ids)], annotation)):
                rejected["occupied_object"] += 1
                continue
            if len(selected):
                triangles = _triangles(raw[selected, :3].astype(float), rings[selected], rank, step)
                if not len(triangles):
                    rejected["no_observed_triangles"] += 1
                    continue
                xyz = world[selected]
                center = np.asarray(annotation["translation"])[:2]
                radius = max(.75, float(np.linalg.norm(xyz[:, :2] - center, axis=1).max()) + .5)
                if radius > 5.:
                    rejected["oversize_surface"] += 1
                    continue
                plane = _ground(ground, center, radius + 1.)
                if plane is None:
                    rejected["uncertain_support"] += 1
                    continue
                origin = np.r_[center, plane[2]]
                vector = center - transform[:2, 3]
                source_range = float(np.linalg.norm(np.asarray(annotation["translation"]) - transform[:3, 3]))
                if source_range < 2.5:
                    continue
                forward = vector / np.linalg.norm(vector)
                basis = _basis(forward, plane)
                local = (xyz - origin) @ basis
                if local[:, 2].min() < -RECIPE["support_residual_m"]:
                    rejected["below_support"] += 1
                    continue
                box_rotation = _pose(annotation)[:3, :3]
                viewpoint = (transform[:3, 3] - annotation["translation"]) @ box_rotation
                donor = dict(id=f"{instance}:{record['token']}", instance=instance, scene=record["scene"],
                    token=record["token"], sample_token=record["sample_token"], log_token=record["log_token"],
                    scan=record["scan"], category=mapping[category]["name"], subset=record["subset"],
                    timestamp=record["timestamp"], annotation=annotation["token"],
                    slots=selected.astype(np.int32), xyz=local.astype(np.float32),
                    source_view=viewpoint / np.linalg.norm(viewpoint),
                    sensor_local=((transform[:3, 3] - origin) @ basis).astype(np.float32),
                    object_center_local=((np.asarray(annotation["translation"]) - origin) @ basis).astype(np.float32),
                    size=np.asarray(annotation["size"])[[1, 0, 2]],
                    intensity=(raw[selected, 3] / 255.).astype(np.float32), triangles=triangles,
                    center=origin, basis=basis.tolist(), support_plane=plane.tolist(),
                    range=source_range, kind=annotation["kind"])
                observations[instance].append(donor)
    donors = []
    for views in observations.values():
        # Maximin viewing-direction selection starts at the best measured surface.
        remaining = sorted(views, key=lambda d: (-len(d["triangles"]), d["id"]))
        chosen = [remaining.pop(0)]
        while remaining and len(chosen) < RECIPE["max_views"]:
            angle = np.array([min(1 - np.clip(d["source_view"] @ old["source_view"], -1, 1)
                                 for old in chosen) for d in remaining])
            index = int(np.argmax(angle))
            if angle[index] < 1 - np.cos(np.deg2rad(3.)):
                break
            chosen.append(remaining.pop(index))
        donors.extend(chosen)
    return rows, donors, dict(rejected)


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


def _render_surface(raw, transform, donor, origin, basis):
    """Resample one fixed world surface on the receiver's measured directions.

    World placement never follows the receiver. Empty hits distinguish unknown
    surface sides from real foreground occlusion; supervision range is irrelevant.
    """
    empty = np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)
    local = np.asarray(donor["xyz"], dtype=float)
    transform, origin, basis = np.asarray(transform), np.asarray(origin), np.asarray(basis)
    origin_sensor = transform[:3, 3]
    vertices = (local @ basis.T + origin - origin_sensor) @ transform[:3, :3]
    distance = np.linalg.norm(raw[:, :3], axis=1)
    directions = np.divide(raw[:, :3], distance[:, None], out=np.zeros((len(raw), 3), float),
                           where=distance[:, None] > 0)
    centroid = vertices.mean(axis=0)
    extent = np.linalg.norm(vertices - centroid, axis=1).max()
    center_range = np.linalg.norm(centroid)
    covered = np.ones(len(raw), dtype=bool)
    if extent < center_range:
        cone = np.sqrt(max(0., 1 - (extent / center_range) ** 2))
        covered = directions @ (centroid / center_range) >= cone
    slots = np.flatnonzero((distance > 0) & covered)
    if not len(slots):
        return *empty, "no_receiver_ray_in_cone"
    triangles = donor["triangles"]
    if "sensor_local" in donor:
        face = local[triangles]
        normals = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
        midpoint = face.mean(axis=1)
        current_sensor = (origin_sensor - origin) @ basis
        original_side = np.einsum("ij,ij->i", normals, donor["sensor_local"] - midpoint)
        target_side = np.einsum("ij,ij->i", normals, current_sensor - midpoint)
        triangles = triangles[(original_side * target_side) > 0]
    if not len(triangles):
        return *empty, "unobserved_surface_side"
    hit, intensity = _intersections(directions[slots], vertices, triangles, donor["intensity"])
    visible = (hit > 0) & (hit < distance[slots] - 1e-4)
    if not visible.any():
        reason = "surface_occluded" if np.isfinite(hit).any() else "no_surface_intersection"
        return *empty, reason
    slots, hit, intensity = slots[visible], hit[visible], intensity[visible]
    xyzi = np.column_stack((directions[slots] * hit[:, None], intensity)).astype(np.float32)
    return slots.astype(np.int32), xyzi, "visible"


def _visible_attributes(xyzi):
    distance = np.linalg.norm(xyzi[:, :3], axis=1)
    points = xyzi[(distance >= 2.5) & (distance <= 50.)]
    if not len(points):
        return dict(angular_span=[0., 0.], intensity_interval=[0., 0.])
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    center = np.arctan2(np.sin(azimuth).mean(), np.cos(azimuth).mean())
    azimuth = (azimuth - center + np.pi) % (2 * np.pi) - np.pi
    elevation = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1))
    return dict(angular_span=[float(np.ptp(azimuth)), float(np.ptp(elevation))],
                intensity_interval=np.quantile(points[:, 3], [.1, .9]).tolist())


def transplant(raw, labels, record, donor, road_id, rng, distance_bin=None, diagnostics=None, match=None):
    """Return a sparse foreground replacement, or None for unsupported placement."""
    transform = np.asarray(record["pose"])
    world = raw[:, :3].astype(float) @ transform[:3, :3].T + transform[:3, 3]
    road_mask = labels == road_id
    road = world[road_mask]
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(reason="no_road", attempts=0, attempt_failures={})
    failures = diagnostics["attempt_failures"]
    def reject(reason):
        failures[reason] = failures.get(reason, 0) + 1
        diagnostics["reason"] = reason
    if len(road) < 8:
        return None
    center_distance = np.linalg.norm(road - transform[:3, 3], axis=1)
    lower_range, upper_range = (2.5, 50.) if distance_bin is None else RECIPE["distance_edges"][distance_bin:distance_bin + 2]
    center_local = np.asarray(donor.get("object_center_local", [0., 0., 0.]))
    margin_center = np.linalg.norm(center_local)
    candidates = np.flatnonzero((center_distance >= max(lower_range, donor["range"]) - margin_center) &
                                (center_distance <= upper_range + margin_center))
    if not len(candidates):
        diagnostics["reason"] = "no_supported_road_range"
        return None
    local = donor["xyz"].astype(float)
    lower, upper = local.min(axis=0), local.max(axis=0)
    radius = max(.75, float(np.linalg.norm(local[:, :2], axis=1).max()) + .5)
    origin_sensor = transform[:3, 3]
    diagnostics["reason"] = "placement_exhausted"
    for candidate in rng.permutation(candidates)[:RECIPE["placement_attempts"]]:
        diagnostics["attempts"] += 1
        center = road[candidate, :2]
        support = {}
        plane = _ground(road, center, radius + .5, support)
        if plane is None:
            reject(support["reason"])
            continue
        # The fitted road must support the chosen anchor, not just nearby points.
        if abs(plane[2] - road[candidate, 2]) > RECIPE["support_residual_m"]:
            reject("support_anchor_disagreement")
            continue
        vector = center - origin_sensor[:2]
        vector /= np.linalg.norm(vector)
        basis = _basis(vector, plane)
        origin = np.r_[center, plane[2]]
        object_center = origin + center_local @ basis.T
        placed_range = float(np.linalg.norm(object_center - origin_sensor))
        if (placed_range < max(lower_range, donor["range"]) or placed_range > upper_range
                or (upper_range != 50. and placed_range == upper_range)):
            reject("object_center_outside_range")
            continue
        # Normal and anomalous donors share the same footprint collision rule.
        nearby = np.linalg.norm(world[:, :2] - center, axis=1) <= radius
        relative = (world[nearby] - origin) @ basis
        margin = RECIPE["collision_margin_m"]
        occupied = np.all((relative[:, :2] >= lower[:2] - margin) & (relative[:, :2] <= upper[:2] + margin), axis=1)
        occupied &= (relative[:, 2] > -RECIPE["support_residual_m"]) & (relative[:, 2] <= upper[2] + margin)
        occupied &= ~road_mask[nearby]
        if occupied.any():
            reject("footprint_collision")
            continue
        slots, xyzi, reason = _render_surface(raw, transform, donor, origin, basis)
        if reason != "visible":
            reject(reason)
            continue
        diagnostics["reason"] = "visible"
        # The request keeps its donor; bounded matching never swaps to an easier object.
        supervised = (np.linalg.norm(xyzi[:, :3], axis=1) >= 2.5) & (np.linalg.norm(xyzi[:, :3], axis=1) <= 50.)
        if match and supervised.any():
            actual = int(supervised.sum())
            if not (.5 <= actual / max(match["visible_points"], 1) <= 2.):
                diagnostics["reason"] = "control_point_support_mismatch"
                continue
            attributes = _visible_attributes(xyzi)
            span, target_span = np.asarray(attributes["angular_span"]), np.asarray(match["angular_span"])
            measurable = (span > 1e-5) & (target_span > 1e-5)
            if np.any((span[measurable] / target_span[measurable] < .5) |
                      (span[measurable] / target_span[measurable] > 2.)):
                diagnostics["reason"] = "control_angular_support_mismatch"
                continue
            low, high = attributes["intensity_interval"]
            target_low, target_high = match["intensity_interval"]
            if low > target_high + 1 / 255 or target_low > high + 1 / 255:
                diagnostics["reason"] = "control_intensity_support_mismatch"
                continue
        return slots.astype(np.int32), xyzi, dict(position_world=origin.tolist(),
            object_center_world=object_center.tolist(),
            yaw_rad=float(np.arctan2(vector[1], vector[0])), range=placed_range,
            source_range=float(donor["range"]), scale=1., basis_world=basis.tolist(),
            support_plane=plane.tolist(), surface="open measured triangles")
    return None


def _initialize(donors, mapping, output):
    global _DONORS, _MAPPING, _OUTPUT
    _DONORS, _MAPPING, _OUTPUT = donors, mapping, Path(output)


def _generate(request):
    record, role = request["record"], request["role"]
    outcome = {key: value for key, value in request.items() if key not in ("record", "match")}
    outcome.update(token=record["token"], subset=record["subset"])
    if request.get("donor") is None:
        reason = "geometry_holdout_unavailable" if request["holdout"] is True else "no_donor_budget"
        outcome.update(reason=reason, visible_points=0)
        return None, outcome
    donor = _DONORS[request["donor"]]
    rng = np.random.default_rng(_seed((RECIPE["revision"], record["token"], role)))
    kind = donor["kind"]
    raw, labels = _read(record)
    road_id = next(row["raw"] for row in _MAPPING if row["name"] == "flat.driveable_surface")
    result = transplant(raw, labels, record, donor, road_id, rng, request["bin"], outcome, request.get("match"))
    if result is None:
        outcome["visible_points"] = 0
        return None, outcome
    slots, xyzi, placement = result
    value = 2 if kind == "anomaly" else 1
    lookup = np.asarray([r["target"] for r in _MAPPING], np.uint8)
    old_range = np.linalg.norm(raw[slots, :3], axis=1)
    removed_normal = int(((lookup[labels[slots]] == 1) & (old_range >= 2.5) & (old_range <= 50.)).sum())
    stored_range = np.linalg.norm(xyzi[:, :3], axis=1)
    supervised = int(((stored_range >= 2.5) & (stored_range <= 50.)).sum())
    outcome.update(visible_points=supervised, changed_slots=len(slots), placement_range=placement["range"])
    if not supervised:
        outcome["reason"] = "only_outside_supervision"
        return None, outcome
    normal = record["normal"] - removed_normal + (supervised if value == 1 else 0)
    anomaly = supervised if value == 2 else 0
    valid_ranges = stored_range[(stored_range >= 2.5) & (stored_range <= 50.)]
    filename = f"{record['token']}_{role}.npz"
    delta_path = _OUTPUT / ".building" / record["subset"] / filename
    np.savez_compressed(delta_path, slots=slots, xyzi=xyzi,
                        labels=np.full(len(slots), value, dtype=np.uint32), token=np.asarray(record["token"]))
    outcome.update(reason="accepted", point_range_median=float(np.median(valid_ranges)),
                   point_histogram=np.histogram(valid_ranges, RECIPE["distance_edges"])[0].tolist(),
                   **_visible_attributes(xyzi))
    inserted = dict(record, group=f"{kind}_nuscenes", delta=str(_OUTPUT / record["subset"] / filename),
        normal=normal, anomaly=anomaly, eligible=anomaly >= 5, placement=placement,
        donor=donor["id"], source_scene=donor["scene"], range=placement["range"], role=role,
        instance=donor["instance"], geometry_group=donor["geometry_group"],
        geometry_reliable=donor["geometry_reliable"], shape_holdout=donor["shape_holdout"],
        point_range_median=outcome["point_range_median"], point_histogram=outcome["point_histogram"],
        requested_bin=request["bin"], visible_points=supervised,
        angular_span=outcome["angular_span"], intensity_interval=outcome["intensity_interval"],
        matched_anomaly_token=request["match"]["token"] if request.get("match") else None)
    return inserted, outcome


def _requests(records, split):
    scenes = defaultdict(list)
    for record in records:
        scenes[record["scene"]].append(record)
    ordered = [row for batch in itertools.zip_longest(
        *(sorted(scenes[name], key=lambda r: _seed(r["token"])) for name in sorted(scenes, key=_seed)))
        for row in batch if row is not None]
    budget = RECIPE[f"{split}_requests"]
    required = budget["reference"] + budget["coverage"] + (budget["control"] if split == "val" else 0)
    if len(ordered) < required:
        raise ValueError("the agreed request budget requires all official keyframes")
    requests = [("reference", row, None, None) for row in ordered[:budget["reference"]]]
    coverage = ordered[budget["reference"]:budget["reference"] + budget["coverage"]]
    requests.extend(("coverage", row, i % 5, (i // 5 < 120) if split == "val" else False)
                    for i, row in enumerate(coverage))
    controls = (ordered[required - budget["control"]:required] if split == "val" else
                sorted(ordered, key=lambda r: _seed("control" + r["token"]))[:budget["control"]])
    requests.extend(("control", row, None, None) for row in controls)
    return requests


def _assign(requests, donors, usage, matches=None):
    """Allocate identities before placement; failures consume their original quota."""
    identifiers = [d["id"] for d in donors]
    scenes = {value: i for i, value in enumerate(dict.fromkeys(d["scene"] for d in donors))}
    groups = {value: i for i, value in enumerate(dict.fromkeys(d["geometry_group"] for d in donors))}
    instances = {value: i for i, value in enumerate(dict.fromkeys(d["instance"] for d in donors))}
    group_index = np.array([groups[d["geometry_group"]] for d in donors], dtype=np.intp)
    instance_index = np.array([instances[d["instance"]] for d in donors], dtype=np.intp)
    group_usage = np.zeros(len(groups), dtype=np.int64)
    instance_usage = np.zeros(len(instances), dtype=np.int64)
    counts = np.array([[usage[identifier, b] for b in range(5)] for identifier in identifiers],
                      dtype=np.int64).reshape(-1, 5)
    kind = np.array([d["kind"] == "control" for d in donors], dtype=bool)
    scene = np.array([scenes[d["scene"]] for d in donors], dtype=np.intp)
    ranges = np.array([d["range"] for d in donors], dtype=float)
    held = np.array([d["shape_holdout"] for d in donors], dtype=bool)
    reliable = np.array([d["geometry_reliable"] for d in donors], dtype=bool)
    support = np.array([len(d["slots"]) for d in donors], dtype=float)
    tie = np.array([_seed(identifier) for identifier in identifiers], dtype=np.uint64)
    assigned = []
    edges = RECIPE["distance_edges"]
    for role, record, distance_bin, holdout in requests:
        request_seed = _seed((record["token"], role))
        rng = np.random.default_rng(request_seed)
        match = None if matches is None else matches.get(record["token"])
        if match is not None:
            distance_bin = min(4, max(0, int(np.searchsorted(edges, match["point_range_median"], side="right") - 1)))
        if distance_bin is None:
            frequencies = np.asarray(record["road_histogram"], dtype=float)
            distance_bin = int(rng.choice(5, p=frequencies / frequencies.sum())) if frequencies.sum() else int(rng.integers(5))
        eligible = ((kind == (role == "control")) & (scene != scenes.get(record["scene"], -1))
                    & (ranges < edges[distance_bin + 1])
                    & (counts[:, distance_bin] < RECIPE["requests_per_view_bin"]))
        if holdout is not None:
            eligible &= held == holdout
        if holdout is True:
            eligible &= reliable
        candidates = np.flatnonzero(eligible)
        selected = None
        if len(candidates):
            if match is not None:
                target_range = max(match["point_range_median"], 2.5)
                estimated = np.maximum(support[candidates] * (ranges[candidates] / target_range) ** 2, 1)
                mismatch = abs(np.log(estimated / max(match["visible_points"], 1)))
                candidates = candidates[mismatch == mismatch.min()]
            # Lexicographic minima need no full sort; donor order breaks exact hash ties.
            for value, index in ((group_usage, group_index), (instance_usage, instance_index)):
                current = value[index[candidates]]
                candidates = candidates[current == current.min()]
            if match is None:
                current = counts[candidates, distance_bin]
                candidates = candidates[current == current.min()]
                # Stable pre-generation tie rule avoids hashing every token/view pair.
                priority = tie[candidates] ^ np.uint64(request_seed)
            else:
                priority = tie[candidates]
            selected = int(candidates[np.argmin(priority)])
            counts[selected, distance_bin] += 1
            usage[identifiers[selected], distance_bin] += 1
            group_usage[group_index[selected]] += 1
            instance_usage[instance_index[selected]] += 1
        assigned.append(dict(record=record, role=role, bin=distance_bin, holdout=holdout,
                             donor=None if selected is None else identifiers[selected], match=match))
    return assigned


def _catalog(donors):
    keep = ("instance", "scene", "token", "sample_token", "log_token", "scan", "category", "subset",
            "kind", "basis", "support_plane", "range", "annotation", "timestamp", "geometry_group",
            "geometry_reliable", "shape_holdout")
    return {d["id"]: dict({key: d[key] for key in keep}, center=d["center"].tolist(),
                          slots=d["slots"].tolist(), triangles=len(d["triangles"]),
                          size=d["size"].tolist(), source_view=d["source_view"].tolist(),
                          object_center_local=d["object_center_local"].tolist()) for d in donors}


def _selected_surfaces(root, records, mapping, catalog):
    """Use exactly the reviewed observations, not an unreviewed frame union."""
    selected = {(x["sample_token"], x["instance"]): x for x in catalog["selected"]}
    source = {r["token"]: r for rows in records.values() for r in rows}
    wanted = {x["sample_token"] for x in selected.values()}
    names = {r["name"]: r["raw"] for r in mapping}
    meta = Path(root) / "v1.0-trainval"
    categories = {r["token"]: r["name"] for r in _rows(meta / "category.json")}
    instances = {r["token"]: categories[r["category_token"]] for r in _rows(meta / "instance.json")}
    annotations, collision_boxes = defaultdict(list), defaultdict(list)
    for row in _rows(meta / "sample_annotation.json"):
        box = {k: row[k] for k in ("translation", "rotation", "size")}
        collision_boxes[row["sample_token"]].append(box)
        if row["sample_token"] in wanted:
            x = selected.get((row["sample_token"], row["instance_token"]))
            annotations[row["sample_token"]].append(dict(row,
                category=names[instances[row["instance_token"]]], kind="anomaly",
                extract=x is not None, family=x["appearance"] if x else "unused"))
    scenes = defaultdict(dict)
    for x in selected.values():
        r = source[x["token"]]
        if any(r[k] != x[k] for k in ("sample_token", "scene", "subset", "log_token", "scan", "label", "pose")):
            raise ValueError("selected object provenance differs from the original background")
        if names[x["category"]] != x["raw_label"]:
            raise ValueError("selected object category differs from original lidarseg")
        scenes[r["scene"]][r["token"]] = r
    donors, rejected = [], Counter()
    for scene in scenes.values():
        _, extracted, failures = _scene_donors((list(scene.values()), mapping, annotations))
        for d in extracted:
            x = selected[d["sample_token"], d["instance"]]
            if not np.array_equal(d["slots"], x["point_slots"]):
                raise ValueError("surface extraction changed the reviewed point selection")
            d.update(review_id=x["review_id"], appearance=x["appearance"],
                     box_rotation_local=np.asarray(d["basis"]).T @ _pose(x["box"])[:3, :3])
        donors.extend(extracted)
        rejected.update(failures)
    return donors, collision_boxes, dict(rejected)


def _box_entry(directions, sensor, rotation, size):
    """Distance to a rigid box from a ray origin, all in one coordinate frame."""
    rays = directions @ rotation
    start = sensor @ rotation
    half = np.asarray(size) * .5
    parallel = abs(rays) < 1e-12
    lo = np.divide(-half - start, rays, out=np.zeros_like(rays), where=~parallel)
    hi = np.divide(half - start, rays, out=np.zeros_like(rays), where=~parallel)
    inside = abs(start) <= half
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(lo, hi)).max(axis=1)
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(lo, hi)).min(axis=1)
    return np.where(far >= np.maximum(near, 0.), np.maximum(near, 0.), np.inf)


def _sequence_collision(frames, donor, origin, basis, boxes, road_id):
    """Reject discrete scene conflicts before rendering any sequence fragment."""
    center = origin + donor["object_center_local"] @ basis.T
    rotation = basis @ donor["box_rotation_local"]
    half = donor["size"] * .5
    margin = RECIPE["collision_margin_m"]
    world_half = abs(rotation) @ half
    low, high = center - world_half - margin, center + world_half + margin
    poses = np.array([np.asarray(r["pose"])[:3, 3] for r, _, _, _ in frames])
    relative = (poses - center) @ rotation
    # A conservative near-field clearance, not a calibrated ego vehicle model.
    if np.any(np.linalg.norm(np.maximum(abs(relative) - half, 0.), axis=1) < 2.5):
        return "ego_clearance"
    for record, raw, labels, world in frames:
        road = world[(labels == road_id) &
                     (np.linalg.norm(world[:, :2] - origin[:2], axis=1) <= np.linalg.norm(half[:2]) + .5)]
        if len(road) and np.quantile(abs((road - origin) @ basis[:, 2]), .9) > RECIPE["support_residual_m"]:
            return "sequence_road_disagreement"
        near = np.all((world >= low) & (world <= high), axis=1)
        local = (world[near] - center) @ rotation
        occupied = np.all(abs(local) <= half + margin, axis=1) & (labels[near] != road_id)
        if occupied.any():
            return "observed_object_collision"
        for box in boxes.get(record["sample_token"], ()):
            extent = abs(_pose(box)[:3, :3]) @ (np.asarray(box["size"])[[1, 0, 2]] * .5)
            other = np.asarray(box["translation"])
            # Enclosing world boxes are conservative under rotation and slope.
            if np.all(other + extent >= low) and np.all(other - extent <= high):
                return "annotated_object_collision"
    return None


def _sequence(task):
    records, donor, mapping, boxes, output = task
    lookup = np.array([r["target"] for r in mapping], np.uint32)
    road_id = next(r["raw"] for r in mapping if r["name"] == "flat.driveable_surface")
    frames, original = [], []
    for record in records:
        raw, labels = _read(record)
        pose = np.asarray(record["pose"])
        world = raw[:, :3].astype(float) @ pose[:3, :3].T + pose[:3, 3]
        frames.append((record, raw, labels, world))
        original.append(dict(record, role="original", **_counts(raw, labels, lookup)))
    scene = records[0]["scene"]
    result = dict(scene=scene, subset=records[0]["subset"], donor=donor["id"],
                  instance=donor["instance"], attempts=0, failures={}, frames=[])
    placement = None
    # The source is assigned before placement; failed scenes never swap donors.
    anchors = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    for index in anchors:
        record, raw, labels, _ = frames[index]
        for attempt in range(4):
            result["attempts"] += 1
            diagnostics = {}
            candidate = transplant(raw, labels, record, donor, road_id,
                np.random.default_rng(_seed(("sequence", scene, donor["id"], index, attempt))),
                diagnostics=diagnostics)
            if candidate is None:
                reason = diagnostics["reason"]
            else:
                proposed = candidate[2]
                origin, basis = np.array(proposed["position_world"]), np.array(proposed["basis_world"])
                reason = _sequence_collision(frames, donor, origin, basis, boxes, road_id)
                if reason is None:
                    placement = proposed
                    result["anchor_token"] = record["token"]
                    break
            result["failures"][reason] = result["failures"].get(reason, 0) + 1
        if placement is not None:
            break
    if placement is None:
        result["status"] = "no_sequence_placement"
        result["frames"] = [dict(token=r["token"], timestamp=r["timestamp"], supported=False,
                                 status="no_sequence_placement") for r in records]
        return original, [], result
    result.update(status="placed", placement=placement)
    origin, basis = np.array(placement["position_world"]), np.array(placement["basis_world"])
    center = origin + donor["object_center_local"] @ basis.T
    rotation = basis @ donor["box_rotation_local"]
    facets = donor["xyz"][donor["triangles"]].astype(float)
    normals = np.cross(facets[:, 1] - facets[:, 0], facets[:, 2] - facets[:, 0])
    midpoint = facets.mean(axis=1)
    source_side = np.einsum("ij,ij->i", normals, donor["sensor_local"] - midpoint)
    source_direction = donor["sensor_local"] - donor["object_center_local"]
    generated, segment, previous_supported = [], -1, False
    for (record, raw, labels, _), background in zip(frames, original):
        pose = np.asarray(record["pose"])
        sensor = (pose[:3, 3] - origin) @ basis
        current_side = np.einsum("ij,ij->i", normals, sensor - midpoint)
        distance_to_center = float(np.linalg.norm(center - pose[:3, 3]))
        status = dict(token=record["token"], timestamp=record["timestamp"], range=distance_to_center)
        # These frames remain in the timeline, not as false zero-anomaly samples.
        same_halfspace = source_direction @ (sensor - donor["object_center_local"]) > 0
        reason = ("closer_than_source" if distance_to_center + 1e-6 < donor["range"] else
                  "unobserved_surface_side" if not same_halfspace or
                  not np.any(source_side * current_side > 0) else None)
        if reason:
            status.update(status=reason, supported=False)
            result["frames"].append(status)
            previous_supported = False
            continue
        if not previous_supported:
            segment += 1
        previous_supported = True
        slots, xyzi, reason = _render_surface(raw, pose, donor, origin, basis)
        distance = np.linalg.norm(raw[:, :3], axis=1)
        direction = np.divide(raw[:, :3], distance[:, None], out=np.zeros((len(raw), 3), float),
                              where=distance[:, None] > 0)
        entry = _box_entry(direction @ pose[:3, :3].T, pose[:3, 3] - center, rotation, donor["size"])
        uncertain = (entry < distance - 1e-4) & (distance > 0)
        uncertain[slots] = False
        ignored = np.flatnonzero(uncertain)
        changed = np.sort(np.concatenate((slots, ignored))).astype(np.int32)
        replacement = raw[changed, :4].copy()
        replacement[:, 3] /= 255.
        replacement_labels = np.zeros(len(changed), np.uint32)
        where = np.searchsorted(changed, slots)
        replacement[where], replacement_labels[where] = xyzi, 2
        truth = lookup[labels].copy()
        truth[changed] = replacement_labels
        observed = raw[:, :4].copy()
        observed[changed, :3] = replacement[:, :3]
        ranges = np.linalg.norm(observed[:, :3], axis=1)
        supervised = (ranges >= 2.5) & (ranges <= 50.) & np.any(observed[:, :3] != 0, axis=1)
        positive = ranges[(truth == 2) & supervised]
        histogram = np.histogram(positive, RECIPE["distance_edges"])[0].tolist()
        filename = f"{record['token']}.npz"
        if len(changed):
            np.savez_compressed(Path(output) / record["subset"] / filename,
                slots=changed, xyzi=replacement, labels=replacement_labels, token=np.asarray(record["token"]))
        row = dict(background, group="anomaly_nuscenes", role="sequence", donor=donor["id"],
            instance=donor["instance"], source_scene=donor["scene"], segment=segment,
            normal=int(((truth == 1) & supervised).sum()), anomaly=len(positive), eligible=len(positive) >= 5,
            point_histogram=histogram, visible_points=len(slots), uncertain_points=len(ignored),
            range=distance_to_center, placement=placement)
        if len(changed):
            row["delta"] = str(Path(output) / record["subset"] / filename)
        generated.append(row)
        status.update(status=reason, supported=True, segment=segment, visible_points=len(slots),
                      anomaly=len(positive), uncertain_points=len(ignored),
                      uncertain_supervised_points=int((uncertain & supervised).sum()),
                      point_histogram=histogram)
        result["frames"].append(status)
    result["segments"] = segment + 1
    if not generated:
        result["status"] = "no_supported_interval"
    return original, generated, result


def _sequences(root, output, workers, records, mapping, objects):
    """One fixed placement per scene; supported time intervals keep their identity."""
    catalog = json.loads(Path(objects).read_text())
    donors, boxes, rejected = _selected_surfaces(root, records, mapping, catalog)
    report = dict(stage="fixed_sequence", objects=str(Path(objects).resolve()),
        selected_objects=len(catalog["selected"]), surfaces=len(donors), surface_failures=rejected,
        scope="Partial measured surfaces on existing return directions; unknown occlusion ignored",
        geometry="One reviewed observation per object; no multiframe deformation or backside completion",
        collision="All labeled keyframes: observed points and conservative annotation bounds; ego clearance 2.5 m",
        view_support="Source viewing halfspace and at least one front-facing measured facet; target center no closer than source",
        unknown_occlusion="Original context kept but label ignored behind an unmeasured possible box surface",
        intensity="Measured vertex interpolation, not a calibrated range/material response", scenes=[])
    surfaces = {d["id"]: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in d.items()} for d in donors}
    report["surfaces"] = surfaces
    tasks, usage = [], Counter()
    for split in ("train", "val"):
        (output / split).mkdir(parents=True, exist_ok=True)
        scenes = defaultdict(list)
        for row in records[split]:
            scenes[row["scene"]].append(row)
        available = [d for d in donors if d["subset"] == split]
        if not available:
            raise ValueError(f"no reviewed surface supports {split}")
        for scene, rows in scenes.items():
            donor = min((d for d in available if d["scene"] != scene),
                        key=lambda d: (usage[d["instance"]], _seed((scene, d["id"]))))
            usage[donor["instance"]] += 1
            tasks.append((rows, donor, mapping,
                          {r["sample_token"]: boxes[r["sample_token"]] for r in rows}, str(output)))
    output_rows = {s: [] for s in records}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, (original, generated, scene) in enumerate(executor.map(_sequence, tasks, chunksize=1)):
            output_rows[scene["subset"]].extend(original + generated)
            report["scenes"].append(scene)
            if (i + 1) % 25 == 0:
                print(f"nuScenes fixed sequences {i + 1}/{len(tasks)}", flush=True)
    summaries, manifests = {}, {}
    for split, rows in output_rows.items():
        scenes = [r for r in report["scenes"] if r["subset"] == split]
        generated = [r for r in rows if r["role"] == "sequence"]
        frames = [f for s in scenes for f in s["frames"]]
        summaries[split] = dict(scenes=len(scenes), placed_scenes=sum(s["status"] == "placed" for s in scenes),
            originals=sum(r["role"] == "original" for r in rows), generated=len(generated),
            eligible=sum(r["eligible"] for r in generated), zero_positive=sum(r["anomaly"] == 0 for r in generated),
            one_to_four_points=sum(0 < r["anomaly"] < 5 for r in generated),
            positive_points=sum(r["anomaly"] for r in generated),
            changed_returns=sum(r["visible_points"] for r in generated),
            uncertain_points=sum(r["uncertain_points"] for r in generated),
            frame_states=dict(Counter(f["status"] for f in frames)),
            instances=len({r["instance"] for r in generated}),
            positive_distance_points=np.sum([r["point_histogram"] for r in generated], axis=0).tolist() if generated else [0]*5)
        manifest = dict(version=SOURCE_VERSION, kind=split, mapping=mapping, records=rows,
            directory=str(output / split), root=str(Path(root).resolve()),
            split=dict(name=split, scenes=sorted({r["scene"] for r in rows}),
                       logs=sorted({r["log_token"] for r in rows})),
            recipe=dict(stage="fixed_sequence", report=str(output / "sequences.json"), summary=summaries[split]))
        manifest["sha256"] = identity(manifest)
        write_json(output / f"{split}.json", manifest, indent=None)
        manifests[split] = manifest
    report.update(summary=summaries, seconds=time.monotonic() - started)
    write_json(output / "sequences.json", report, indent=None)
    print(json.dumps(summaries), flush=True)
    return manifests


def build(root, output, workers, *, background_only=False, objects=None):
    """Census actual surfaces, then materialize bounded source-only requests."""
    output = Path(output).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    if (background_only or objects is not None) and output.exists() and any(output.iterdir()):
        raise ValueError("background or sequence construction requires an empty output directory")
    if background_only and objects is not None:
        raise ValueError("background-only and object placement are mutually exclusive")
    staging = output / ".building"
    if staging.exists():
        raise ValueError("an unfinished build exists; inspect it before replacement")
    if output.exists() and any(output.iterdir()) and not all((output / f"{s}.json").exists() for s in ("train", "val")):
        raise ValueError("output is not an existing complete nuScenes build")
    started = time.monotonic()
    records, mapping = sources(root)
    if background_only:
        return _backgrounds(root, output, workers, records, mapping)
    if objects is not None:
        return _sequences(root, output, workers, records, mapping, objects)
    annotations, instances = _annotations(root, records, mapping)
    print(f"nuScenes selected candidate instances {dict(Counter((d['subset'] + ':' + d['kind']) for d in instances.values()))}", flush=True)
    original, donors, failures = {}, [], {}
    for split in ("train", "val"):
        scenes = defaultdict(list)
        for record in records[split]:
            scenes[record["scene"]].append(record)
        original[split], failures[split] = [], Counter()
        tasks = [(rows, mapping, {r["sample_token"]: annotations.get(r["sample_token"], []) for r in rows})
                 for rows in scenes.values()]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, (rows, views, rejected) in enumerate(executor.map(_scene_donors, tasks), 1):
                original[split].extend(rows)
                donors.extend(views)
                failures[split].update(rejected)
                if index % 25 == 0 or index == len(scenes):
                    print(f"nuScenes census {split} scenes {index}/{len(scenes)}; measured views {len(donors)}", flush=True)
    del annotations, records, tasks
    geometry = group_geometry(donors)
    census = dict(recipe=RECIPE, geometry=geometry, extraction_failures=failures,
                  candidates={s: dict(Counter(d["category"] for d in instances.values() if d["subset"] == s)) for s in original},
                  surfaces={s: {kind: dict(instances=len({d['instance'] for d in donors if d['subset'] == s and d['kind'] == kind}),
                                           views=sum(d['subset'] == s and d['kind'] == kind for d in donors))
                                for kind in ("anomaly", "control")} for s in original})
    print("nuScenes usable measured surfaces " + json.dumps(census["surfaces"]), flush=True)
    print("nuScenes geometry groups " + json.dumps({k: geometry[k] for k in (
        "anomaly_instances", "reliable_instances", "groups", "largest_group", "development_reliable_groups",
        "heldout_fraction", "removed_training_observations")}), flush=True)
    views_by_instance = defaultdict(list)
    for donor in donors:
        views_by_instance[donor["instance"]].append(donor)
    census["instances"] = {token: dict(subset=row["subset"], category=row["category"], kind=row["kind"],
        views=len(views_by_instance[token]),
        range_eligible_views=[sum(d["range"] < upper for d in views_by_instance[token])
                              for upper in RECIPE["distance_edges"][1:]],
        geometry_group=views_by_instance[token][0]["geometry_group"] if views_by_instance[token] else None,
        geometry_reliable=bool(views_by_instance[token] and views_by_instance[token][0]["geometry_reliable"]),
        shape_holdout=bool(views_by_instance[token] and views_by_instance[token][0]["shape_holdout"]))
        for token, row in instances.items()}
    census["range_eligible_definition"] = "Source range below target-bin upper bound is necessary, not sufficient for visible placement; actual outcomes are counted separately."
    write_json(staging / "census.json", census)
    manifests = {}
    for split in ("train", "val"):
        (staging / split).mkdir(parents=True, exist_ok=True)
        available = [d for d in donors if d["subset"] == split and not (split == "train" and d["shape_holdout"])]
        original[split].sort(key=lambda r: (r["scene"], r["timestamp"], r["token"]))
        requests = _requests(original[split], split)
        usage, rows, outcomes = Counter(), list(original[split]), []
        for record in rows:
            record["role"] = "original"
        by_role = [[r for r in requests if r[0] == role] for role in ("reference", "coverage")]
        anomaly_requests = [r for pair in itertools.zip_longest(*by_role) for r in pair if r is not None]
        assigned = _assign(anomaly_requests, available, usage)
        source = {d["id"]: d for d in available}
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize, initargs=(source, mapping, str(output))) as executor:
            for index, (inserted, outcome) in enumerate(executor.map(_generate, assigned, chunksize=8), 1):
                outcomes.append(outcome)
                if inserted is not None:
                    rows.append(inserted)
                if index % 1000 == 0 or index == len(assigned):
                    print(f"nuScenes {split} anomaly requests {index}/{len(assigned)}; accepted {len(rows)-len(original[split])}", flush=True)
            controls = [r for r in requests if r[0] == "control"]
            accepted = {r["token"]: r for r in rows if r["role"] in ("reference", "coverage")}
            if split == "val":
                reference = [r for r in accepted.values() if r["role"] == "reference"]
                coverage = [r for r in accepted.values() if r["role"] == "coverage"]
                matches = {request[1]["token"]: pool[i % len(pool)] for half, pool in enumerate((reference, coverage)) if pool
                           for i, request in enumerate(controls[half * 1000:(half + 1) * 1000])}
            else:
                matches = accepted
            assigned = _assign(controls, available, usage, matches)
            for index, (inserted, outcome) in enumerate(executor.map(_generate, assigned, chunksize=8), 1):
                outcomes.append(outcome)
                if inserted is not None:
                    rows.append(inserted)
                if index % 1000 == 0 or index == len(assigned):
                    print(f"nuScenes {split} control requests {index}/{len(assigned)}", flush=True)
        catalog = _catalog([d for d in donors if d["subset"] == split])
        summary = dict(requests=dict(Counter(r["role"] for r in outcomes)),
                       outcomes=dict(Counter(r["reason"] for r in outcomes)),
                       records=dict(Counter(r["role"] for r in rows)),
                       eligible_reference=sum(r["eligible"] and r["role"] == "reference" for r in rows),
                       eligible_coverage=sum(r["eligible"] and r["role"] == "coverage" for r in rows),
                       positive_points=sum(r["anomaly"] for r in rows),
                       per_distance=[dict(bin=b, requested=sum(r["bin"] == b and r["role"] != "control" for r in outcomes),
                                          coverage_requests=sum(r["bin"] == b and r["role"] == "coverage" for r in outcomes),
                                          accepted=sum(r["requested_bin"] == b for r in rows if r["anomaly"]),
                                          instances=len({r["instance"] for r in rows if r["anomaly"] and r["requested_bin"] == b}),
                                          eligible=sum(r["eligible"] and r["requested_bin"] == b for r in rows if r["anomaly"]),
                                          few_point=sum(0 < r["anomaly"] < 5 and r["requested_bin"] == b for r in rows if r["anomaly"]),
                                          points=sum(r["point_histogram"][b] for r in rows if r["anomaly"])) for b in range(5)])
        census[split] = summary
        if split == "val" and not summary["eligible_reference"]:
            raise ValueError(f"{split} has no evaluable reference anomalies; generated data remain unpublished")
        manifest = dict(version=SOURCE_VERSION, kind=split, mapping=mapping, records=rows, donors=catalog,
                        directory=str(output / split), root=str(Path(root).resolve()),
                        split=dict(name=split, scenes=sorted({r["scene"] for r in original[split]}),
                                   logs=sorted({r["log_token"] for r in original[split]})),
                        recipe=dict(RECIPE, original_frames=len(original[split]), summary=summary),
                        evaluation_role="reference" if split == "val" else None)
        manifest["sha256"] = identity(manifest)
        write_json(staging / f"{split}.json", manifest, indent=None)
        write_json(staging / f"{split}_requests.json", outcomes, indent=None)
        manifests[split] = manifest
    census["seconds"] = time.monotonic() - started
    write_json(staging / "census.json", census)
    # Publish only a complete replacement. Old raw scans are never copied or deleted.
    for split in ("train", "val"):
        directory = output / split
        directory.mkdir(exist_ok=True)
        keep = {Path(r["delta"]).name for r in manifests[split]["records"] if r.get("delta")}
        for path in (staging / split).iterdir():
            path.replace(directory / path.name)
        for path in directory.glob("*.npz"):
            if path.name not in keep:
                path.unlink()
    for path in staging.glob("*.json"):
        path.replace(output / path.name)
    shutil.rmtree(staging)
    print("nuScenes completed " + json.dumps({s: census[s] for s in ("train", "val")}), flush=True)
    return manifests



def group_geometry(donors):
    """Instance identity is observed; cross-instance shape identity is not.

    Repeat-view errors alone do not calibrate cross-instance false matches.
    Do not turn this unavailable measurement into exclusions or shape claims.
    """
    instances = {}
    for donor in donors:
        signature = (donor["subset"], donor["kind"])
        previous = instances.setdefault(donor["instance"], signature)
        if previous != signature:
            raise ValueError("one annotated instance crosses source partitions or roles")
        donor.update(geometry_group=None, geometry_reliable=False, shape_holdout=False)
    return dict(method="Official annotated-instance allocation; geometric identity unavailable",
                unavailable_reason="Partial visible-surface repeatability does not validate cross-instance shape similarity",
                anomaly_instances=sum(kind == "anomaly" for _, kind in instances.values()),
                reliable_instances=0, groups=None, largest_group=None, development_reliable_groups=0,
                heldout_groups=[], heldout_fraction=0., removed_training_instances=[],
                removed_training_observations=0, planned_holdout_requests=600,
                scope="Scene/log and annotated-instance split; no unseen-geometry or cross-scene physical-identity guarantee")
