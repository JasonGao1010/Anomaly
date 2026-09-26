"""Prepare original nuScenes normal scans and exact reviewed semantic labels."""

from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
import json
from pathlib import Path

import ijson
import numpy as np
from scipy.spatial.transform import Rotation

from .data import SOURCE_VERSION, identity, nuscenes_mapping, nuscenes_truth, write_json


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


def _counts(raw, truth):
    distance = np.linalg.norm(raw[:, :3], axis=1)
    actual = np.any(raw[:, :3] != 0, axis=1)
    normal = (truth == 1) & actual & (distance >= 2.5) & (distance <= 50.)
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
        truth = nuscenes_truth(record, labels, mapping)
        row = dict(record, role="original", **_counts(raw, truth))
        rows.append(row)
        actual = np.any(raw[:, :3] != 0, axis=1)
        distance = np.linalg.norm(raw[:, :3], axis=1)
        valid = actual & (distance >= 2.5) & (distance <= 50.)
        normal = valid & (truth == 1)
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
                       supplemental_normal_frames=sum("normal_slots" in r for r in rows),
                       supplemental_normal_points=sum(len(r.get("normal_slots", [])) for r in rows),
                       normal_distance_edges_m=[2.5, 10., 20., 30., 40., 50.],
                       normal_distance_points=ranges.tolist(),
                       location_frames=dict(Counter(logs[r["log_token"]] for r in rows)))
        manifest = dict(version=SOURCE_VERSION, kind=split, root=str(Path(root).resolve()),
            directory=str(output), mapping=mapping, records=rows,
            split=dict(name=split, scenes=sorted(scenes), logs=sorted({r["log_token"] for r in rows})),
            recipe=dict(stage="background", partition="Official nuScenes train/val scenes; disjoint acquisition logs",
                input="Complete original measured returns; no point removal by semantic label or supervision range",
                supervision=f"{sum(r['target'] == 1 for r in mapping)} admitted normal classes within 2.5-50 m; unresolved categories ignored; no positive labels",
                supplemental_normals="Exact reviewed native point indices only; no category-wide or temporal label propagation",
                validation="Normal semantic development only; unknown detection uses separate STU scans"),
            summary=summary)
        manifest["sha256"] = identity(manifest)
        manifests[split] = manifest
        print("nuScenes background " + json.dumps(dict(subset=split, **summary)), flush=True)
    for split, manifest in manifests.items():
        write_json(output / f"{split}.json", manifest, indent=None)
    return manifests


def _normal_annotations(records, mapping, path):
    """Attach exact reviewed slots; no category-wide or cross-frame propagation."""
    annotations = json.loads(Path(path).read_text())["records"]
    source = {r["token"]: r for rows in records.values() for r in rows}
    seen = set()
    for annotation in annotations:
        token = annotation["token"]
        if token in seen or token not in source:
            raise ValueError("duplicate or unknown supplemental annotation token")
        seen.add(token)
        record = source[token]
        if any(record[k] != annotation[k] for k in ("scene", "subset")) or annotation["semantic"] != "building":
            raise ValueError("supplemental annotation identity or admitted semantics differ")
        checked = dict(record, normal_slots=annotation["point_slots"])
        _, labels = _read(record)
        nuscenes_truth(checked, labels, mapping)
        for rows in records.values():
            for row in rows:
                if row["token"] == token:
                    row["normal_slots"] = annotation["point_slots"]
                    row["normal_annotation"] = str(Path(path).resolve())


def build(root, output, workers, *, normal_annotations=None):
    """Index original normal scans without modifying their measured returns."""
    output = Path(output).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("background construction requires an empty output directory")
    records, mapping = sources(root)
    if normal_annotations is not None:
        _normal_annotations(records, mapping, normal_annotations)
    return _backgrounds(root, output, workers, records, mapping)
