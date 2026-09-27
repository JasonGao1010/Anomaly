"""Read complete normal training scans and unchanged STU evaluation returns."""

from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import math
import os
import time

import numpy as np


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


SOURCE_VERSION = "AJAE-V4-NS"
MANIFEST_VERSION = "AJAE-V4-F240-R1"
DATA_ROOT = Path("/home/jasongao/Data/STU")
NUSCENES_ROOT = Path("/home/jasongao/Data/Nuscenes")
# Retain known road-scene classes; unresolved source categories receive no target.
NUSCENES_NORMAL = frozenset((
    "human.pedestrian.adult", "human.pedestrian.child",
    "human.pedestrian.construction_worker", "human.pedestrian.police_officer",
    "vehicle.bicycle", "vehicle.emergency.ambulance", "vehicle.emergency.police",
    "vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.car", "vehicle.construction",
    "vehicle.motorcycle", "vehicle.trailer", "vehicle.truck", "flat.driveable_surface",
    "flat.sidewalk", "flat.terrain", "static.vegetation",
))


def readonly(values):
    result = np.ascontiguousarray(values)
    result.setflags(write=False)
    return result


def rigid(matrix):
    """Check a source rigid transform at the original reader's file precision."""
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("pose must be a finite 4 x 4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-9, rtol=0):
        raise ValueError("invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=1e-3):
        raise ValueError("pose rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1, abs_tol=1e-3, rel_tol=1e-3):
        raise ValueError("pose rotation determinant is not +1")


@dataclass(frozen=True, slots=True)
class Frame:
    """Original file slots, optional truth and a sensor-to-world pose.

    Only xyzi and the return selection belong to model input. Source identity,
    pose and labels remain separate metadata for construction and supervision.
    """

    frame_id: int
    xyzi: np.ndarray
    pose: np.ndarray
    labels: np.ndarray | None
    sequence_id: int = 206
    partition: str = "train"
    actual: np.ndarray = field(init=False)
    range_m: np.ndarray = field(init=False)
    semantic: np.ndarray | None = field(init=False)
    instance: np.ndarray | None = field(init=False)

    def __post_init__(self):
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise ValueError("sequence_id must be a nonnegative integer")
        xyzi = np.asarray(self.xyzi)
        if xyzi.dtype != np.float32 or xyzi.ndim != 2 or xyzi.shape[1] != 4:
            raise TypeError("xyzi must be float32[N,4]")
        if not np.isfinite(xyzi).all():
            raise ValueError("xyzi contains nonfinite values")
        pose = np.asarray(self.pose)
        if pose.dtype != np.float64:
            raise TypeError("pose must be float64[4,4]")
        rigid(pose)
        packed = None if self.labels is None else np.asarray(self.labels)
        if packed is not None and (packed.dtype != np.uint32 or packed.shape != (len(xyzi),)):
            raise TypeError("packed labels must be uint32[N] aligned with xyzi")
        semantic = None if packed is None else (packed & 65535).astype(np.uint16)
        instance = None if packed is None else (packed >> 16).astype(np.uint16)
        # XYZ, not intensity or label, identifies an actual return. Never deduplicate.
        for name, values in (
            ("xyzi", xyzi.copy()), ("pose", pose.copy()),
            ("labels", None if packed is None else packed.copy()),
            ("actual", np.any(xyzi[:, :3] != 0, axis=1)),
            ("range_m", np.linalg.norm(xyzi[:, :3], axis=1)),
            ("semantic", semantic), ("instance", instance),
        ):
            object.__setattr__(self, name, None if values is None else readonly(values))

    @property
    def return_slots(self):
        return np.flatnonzero(self.actual)


def point_targets(frame):
    """Return -1/0/1 for ignored/normal/anomalous points before frame selection."""
    if frame.labels is None:
        raise ValueError("supervision requires ground-truth labels")
    labels = unified_labels(frame.labels)
    valid = frame.actual & (frame.range_m >= 2.5) & (frame.range_m <= 50) & (labels != 0)
    targets = np.full(len(frame.xyzi), -1, dtype=np.int8)
    targets[valid] = (labels[valid] == 2).astype(np.int8)
    return targets


def unified_labels(packed):
    """STU stores raw semantics in the low 16 bits; raw 1 is a valid inlier."""
    raw = np.asarray(packed) & 65535
    unknown = np.setdiff1d(np.unique(raw), list(LABELS))
    if len(unknown):
        raise ValueError(f"unrecognized raw STU semantic labels: {unknown.tolist()}")
    return np.where(raw == 0, 0, np.where(raw == 2, 2, 1)).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class Supervision:
    targets: np.ndarray
    normal_count: int
    anomaly_count: int
    allow_normal: bool = False

    @property
    def eligible(self):
        return self.anomaly_count >= 5 or (
            self.allow_normal and self.anomaly_count == 0 and self.normal_count > 0)


def supervision(frame, *, allow_normal=False):
    """Training may retain trusted normal scans; official evaluation never does."""
    targets = point_targets(frame)
    normal, anomaly = int((targets == 0).sum()), int((targets == 1).sum())
    if anomaly < 5 and not (allow_normal and anomaly == 0 and normal > 0):
        targets.fill(-1)
    # No per-object threshold: 1-4 point objects stay anomalous in eligible frames.
    return Supervision(readonly(targets), normal, anomaly, allow_normal)


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value, *, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_scan(scan, label=None, *, partition="val", expected=None, io_timing=None):
    """Decode unchanged file slots; optionally measure file reads without decoding."""
    scan = Path(scan)
    start = time.perf_counter()
    raw_scan = scan.read_bytes()
    read_seconds = time.perf_counter() - start
    size = len(raw_scan)
    if not size or size % 16:
        raise ValueError(f"invalid XYZI byte length: {scan}")
    xyzi = np.frombuffer(raw_scan, dtype="<f4").reshape(-1, 4)
    packed = None
    if label is not None:
        label = Path(label)
        start = time.perf_counter()
        raw_label = label.read_bytes()
        read_seconds += time.perf_counter() - start
        if len(raw_label) != len(xyzi) * 4:
            raise ValueError(f"point/label count mismatch: {scan}")
        packed = np.frombuffer(raw_label, dtype="<u4")
        unified_labels(packed)
    if expected is not None:
        actual = (hashlib.sha256(raw_scan).hexdigest(),
                  None if label is None else hashlib.sha256(raw_label).hexdigest())
        if actual != tuple(expected):
            raise ValueError(f"STU observation changed after manifest creation: {scan}")
    result = Frame(int(scan.stem), xyzi, np.eye(4), packed,
                   sequence_id=int(scan.parent.parent.name), partition=partition)
    if io_timing is not None:
        io_timing["seconds"] = read_seconds
    return result


def _census_real(task):
    scan, label, partition = task
    frame = read_scan(scan, label, partition=partition)
    selected = supervision(frame) if label is not None else None
    return dict(sequence=frame.sequence_id, frame=frame.frame_id, scan=str(scan),
                label=None if label is None else str(label), scan_sha256=hashlib.sha256(frame.xyzi.tobytes()).hexdigest(),
                label_sha256=None if label is None else hashlib.sha256(frame.labels.tobytes()).hexdigest(),
                points=int(frame.actual.sum()), slots=len(frame.xyzi),
                normal=None if selected is None else selected.normal_count,
                anomaly=None if selected is None else selected.anomaly_count,
                eligible=None if selected is None else selected.eligible)


def make_real_manifest(directory, *, partition="val", workers=4):
    if partition not in ("val", "test") or workers < 1:
        raise ValueError("STU manifests require val/test and positive workers")
    directory = Path(directory).resolve(strict=True)
    sequences = sorted(p for p in directory.iterdir() if p.is_dir() and p.name.isdigit())
    if not sequences:
        raise ValueError(f"no official STU sequences in {directory}")
    tasks = []
    for sequence in sequences:
        scans = sorted((sequence / "velodyne").glob("*.bin"))
        labels = sorted((sequence / "labels").glob("*.label"))
        if not scans or (labels and [p.stem for p in scans] != [p.stem for p in labels]) or (partition == "val" and not labels):
            raise ValueError(f"missing or unmatched STU scans/labels: {sequence}")
        if not labels:
            labels = [None] * len(scans)
        tasks.extend((s, t, partition) for s, t in zip(scans, labels))
    if len({t is None for _, t, _ in tasks}) != 1:
        raise ValueError("a STU evaluation population cannot mix labeled and hidden scans")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_census_real, tasks, chunksize=8))
    result = dict(version=MANIFEST_VERSION, kind=partition, directory=str(directory),
                  sequences=[p.name for p in sequences], records=records)
    result["sha256"] = identity(result)
    return result


def load_manifest(path, kind):
    value = json.loads(Path(path).read_text())
    expected = value.pop("sha256")
    if identity(value) != expected or value["version"] not in (MANIFEST_VERSION, SOURCE_VERSION) or value["kind"] != kind:
        raise ValueError(f"invalid {kind} manifest identity: {path}")
    value["sha256"] = expected
    return value


def nuscenes_mapping(root=NUSCENES_ROOT):
    categories = json.loads((Path(root) / "lidarseg/category.json").read_text())
    if sorted(row["index"] for row in categories) != list(range(32)):
        raise ValueError("the original 32-class lidarseg taxonomy is required")
    if not NUSCENES_NORMAL <= {row["name"] for row in categories}:
        raise ValueError("known normal classes are absent from the lidarseg taxonomy")
    reasons = {
        "noise": "Invalid or unidentifiable returns; no supervision",
        "animal": "Animal behavior and STU normal/ignore boundary are not specified",
        "human.pedestrian.personal_mobility": "Person and mobility device are not separately labeled; ignore mixed category",
        "human.pedestrian.stroller": "Person/device boundary is not established; ignore mixed category",
        "human.pedestrian.wheelchair": "Person/device boundary is not established; ignore mixed category",
        "movable_object.barrier": "Temporary barriers are not identical to SemanticKITTI fences; target boundary unresolved",
        "movable_object.trafficcone": "No public point-level STU normal/ignore correspondence; do not infer anomaly from missing class name",
        "movable_object.debris": "Native debris has no trusted normal correspondence and remains ignored",
        "movable_object.pushable_pullable": "Native devices have no trusted normal correspondence and remain ignored",
        "static_object.bicycle_rack": "Rack is not a bicycle; no reliable normal/ignore correspondence",
        "flat.other": "Mixed water, rail tracks, islands and stairs cannot be separated with this label",
        "static.manmade": "Mixed normal infrastructure and STU-ignored parking meters, utility boxes and lamps; ignore entire unresolved category",
        "static.other": "Unresolved static objects; no normal or anomaly target",
        "vehicle.ego": "Ego platform is outside road-object supervision",
    }
    return [dict(raw=row["index"], name=row["name"],
                 target=int(row["name"] in NUSCENES_NORMAL),
                 decision="normal" if row["name"] in NUSCENES_NORMAL else "ignore",
                 reason=reasons.get(row["name"],
                     "Expected road user" if row["name"].startswith(("human.", "vehicle.")) else
                     "Road, sidewalk, terrain or vegetation in the shared normal semantics"))
            for row in sorted(categories, key=lambda row: row["index"])]


def nuscenes_truth(record, labels, mapping):
    """Apply exact reviewed normal labels; unresolved source categories stay ignored."""
    truth = np.asarray([row["target"] for row in mapping], np.uint32)[labels]
    if "normal_slots" in record:
        slots = np.asarray(record["normal_slots"])
        raw_id = next(row["raw"] for row in mapping if row["name"] == "static.manmade")
        if (slots.ndim != 1 or slots.dtype.kind not in "iu" or not len(slots)
                or np.any(slots < 0) or np.any(slots >= len(labels))
                or np.any(np.diff(slots) <= 0) or np.any(labels[slots] != raw_id)):
            raise ValueError("supplemental normal points do not match the reviewed native category")
        truth[slots] = 1
    return truth


def read_nuscenes(record, mapping):
    """Keep one native sweep; only convert the sensor's fixed intensity units."""
    if record.get("delta"):
        raise ValueError("normal source reading requires unchanged native returns")
    raw = np.fromfile(record["scan"], dtype="<f4")
    labels = np.fromfile(record["label"], dtype=np.uint8)
    if raw.size % 5 or raw.size // 5 != labels.size or not labels.size:
        raise ValueError("nuScenes point/label records do not correspond")
    raw = raw.reshape(-1, 5)
    if not np.isfinite(raw).all() or np.any((raw[:, 4] < 0) | (raw[:, 4] > 31)):
        raise ValueError("invalid native nuScenes returns")
    if np.any(labels >= len(mapping)):
        raise ValueError("unknown nuScenes lidarseg label")
    xyzi = raw[:, :4].copy()
    # This is the official LitePT nuScenes input convention, not per-frame scaling.
    xyzi[:, 3] /= 255.
    truth = nuscenes_truth(record, labels, mapping)
    return Frame(record["frame"], xyzi, np.asarray(record.get("pose", np.eye(4)), dtype=float),
                 truth, sequence_id=0, partition="train")


NORMAL_CLASSES = ("car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
                  "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
                  "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign")


STU_NORMAL_SEMANTICS = {
    10: 0, 252: 0, 11: 1, 15: 2, 18: 3, 258: 3,
    13: 4, 16: 4, 20: 4, 256: 4, 257: 4, 259: 4,
    30: 5, 254: 5, 31: 6, 253: 6, 32: 7, 255: 7,
    40: 8, 60: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13,
    70: 14, 71: 15, 72: 16, 80: 17, 81: 18,
}


# Coarse source labels supervise allowed sets, never invented fine annotations.
# Reviewed exact point slots and class definitions are in assets/normal.json.
NUSCENES_NORMAL_SETS = {
    2: (5, 6, 7), 3: (5, 6, 7), 4: (5, 6, 7), 6: (5, 6, 7), 14: (1, 6), 15: (4,), 16: (4,),
    17: (0,), 18: (4,), 19: (0, 3, 4), 20: (0, 1, 2, 3, 4, 6, 7), 21: (2, 4, 7),
    22: (4,), 23: (3,), 24: (8, 9, 10, 11), 26: (10,), 27: (10, 14, 16), 30: (14, 15),
}


NORMAL_ANNOTATIONS = Path(__file__).resolve().parents[1] / "assets" / "normal.json"


@lru_cache(maxsize=2)
def normal_cycle_annotations(root, metadata_sha256):
    """Use official per-instance rider attributes; missing attributes stay coarse."""
    import ijson
    meta = Path(root) / "v1.0-trainval"
    categories = {r["token"]: r["name"] for r in json.loads((meta / "category.json").read_text())}
    raw_class = {"vehicle.bicycle": 14, "vehicle.motorcycle": 21}
    instances = {r["token"]: raw_class[categories[r["category_token"]]]
                 for r in json.loads((meta / "instance.json").read_text())
                 if categories[r["category_token"]] in raw_class}
    attributes = {r["token"]: r["name"] for r in json.loads((meta / "attribute.json").read_text())}
    grouped = {}
    with (meta / "sample_annotation.json").open("rb") as stream:
        for row in ijson.items(stream, "item", use_float=True):
            category = instances.get(row["instance_token"])
            if category is None:
                continue
            states = {attributes[token] for token in row["attribute_tokens"]}
            if {"cycle.with_rider", "cycle.without_rider"} <= states:
                raise ValueError("a cycle has contradictory official rider attributes")
            box = {key: row[key] for key in ("token", "instance_token", "translation", "rotation", "size")}
            box.update(raw_class=category, with_rider="cycle.with_rider" in states)
            grouped.setdefault(row["sample_token"], []).append(box)
    return grouped


def attach_normal_annotations(records, root, *, directory=None):
    """Attach observed point labels and official cycle attributes to their own scans."""
    names = ("category", "instance", "attribute", "sample_annotation")
    provenance = {name: file_sha256(Path(root) / "v1.0-trainval" / (name + ".json")) for name in names}
    cycles = normal_cycle_annotations(str(Path(root).resolve()), tuple(provenance.values()))
    reviewed = json.loads(NORMAL_ANNOTATIONS.read_text())["records"]
    reviewed_sha256 = file_sha256(NORMAL_ANNOTATIONS)
    source = {row["token"]: row for row in records}
    if len(source) != len(records):
        raise ValueError("normal source records contain duplicate lidar observations")
    for row in records:
        if row["sample_token"] in cycles:
            row["normal_cycles"] = cycles[row["sample_token"]]
        if row.get("normal_slots"):
            # Reviewed annotations travel with their manifest, not an old workspace.
            path = (Path(directory) / Path(row["normal_annotation"]).name
                    if directory is not None else Path(row["normal_annotation"]))
            row["normal_annotation"] = str(path.resolve())
            annotations = json.loads(path.read_text())["records"]
            match = [a for a in annotations if a["token"] == row["token"]]
            if (len(match) != 1 or match[0]["semantic"] != "building"
                    or any(match[0][key] != row[key] for key in ("scene", "subset"))
                    or match[0]["point_slots"] != row["normal_slots"]):
                raise ValueError("reviewed building slots do not match the original annotation")
            row["normal_fine"] = [dict(semantic="building", raw_class=28,
                point_slots=row["normal_slots"], provenance=str(path.resolve()),
                provenance_sha256=file_sha256(path))]
    for annotation in reviewed:
        row = source.get(annotation["token"])
        if row is None:
            continue
        if any(annotation[key] != row[key] for key in ("scene", "subset", "sample_token")):
            raise ValueError("reviewed normal point annotation belongs to another scan or partition")
        row.setdefault("normal_fine", []).append(annotation)
    for row in records:
        categories = {NORMAL_CLASSES.index(a["semantic"]) for a in row.get("normal_fine", ())}
        candidates = {6 for box in row.get("normal_cycles", ())
                      if box["raw_class"] == 14 and box["with_rider"]}
        if candidates or categories:
            # Replay eligibility requires actual in-range, uniquely assigned returns.
            allowed = read_normal_record(row)["allowed"]
            singleton = allowed.sum(1) == 1
            categories = {category for category in categories | candidates
                          if bool(np.any(singleton & allowed[:, category]))}
        row["refinement_classes"] = sorted(categories)
        row["normal_refinement"] = dict(version="normal-class-refinement-1",
            identity=identity(dict(cycles=row.get("normal_cycles", []), fine=row.get("normal_fine", []))),
            source_metadata_sha256=provenance, reviewed_sha256=reviewed_sha256,
            cycles="Official nuScenes sample_annotation, instance and cycle.with_rider attributes; unique cuboid and source-label intersection")
    return records


def refine_normal_labels(record, xyzi, label, allowed, slots):
    """Refine only uniquely attributed points; retain all unresolved label sets."""
    from scipy.spatial.transform import Rotation
    boxes = record.get("normal_cycles", ())
    if boxes:
        pose = np.asarray(record["pose"], dtype=np.float64)
        for category, fine in ((14, (6,)), (21, (4, 7))):
            selected = np.flatnonzero(label == category)
            candidates = [box for box in boxes if box["raw_class"] == category]
            if not len(selected) or not candidates:
                continue
            world = xyzi[selected, :3].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
            inside = []
            for box in candidates:
                rotation = Rotation.from_quat(np.asarray(box["rotation"])[[1, 2, 3, 0]]).as_matrix()
                local = (world - box["translation"]) @ rotation
                # Official cuboids use width/length/height; never enlarge a box.
                inside.append(np.all(abs(local) < np.asarray(box["size"])[[1, 0, 2]] * .5, axis=1))
            inside = np.stack(inside)
            unique = inside.sum(0) == 1
            for box, matched in zip(candidates, inside):
                # A non-rider attribute does not exclude a person standing nearby.
                if box["with_rider"]:
                    points = selected[unique & matched]
                    allowed[points] = False
                    allowed[np.ix_(points, fine)] = True
    claimed = set()
    for annotation in record.get("normal_fine", ()):
        points = np.asarray(annotation["point_slots"])
        if (points.ndim != 1 or points.dtype.kind not in "iu" or not len(points)
                or len(points) != len(np.unique(points))
                or np.any(points < 0) or claimed.intersection(points.tolist())):
            raise ValueError("reviewed normal slots are invalid or overlap")
        claimed.update(points.tolist())
        position = np.searchsorted(slots, points)
        if np.any(position >= len(slots)) or not np.array_equal(slots[position], points):
            raise ValueError("reviewed normal annotation points are absent from this scan")
        if not np.all(label[position] == annotation["raw_class"]):
            raise ValueError("reviewed normal point annotation contradicts its original source labels")
        category = NORMAL_CLASSES.index(annotation["semantic"])
        allowed[position] = False
        allowed[position, category] = True


def normal_records(source, *, development=False, directory=None, root=None):
    """Only original source scans or the explicitly allowed normal STU sequences."""
    if source == "nuscenes":
        directory = Path(directory or "results/data/background")
        kind = "val" if development else "train"
        manifest = load_manifest(directory / (kind + ".json"), kind)
        other_kind = "train" if development else "val"
        other = load_manifest(directory / (other_kind + ".json"), other_kind)
        # Partition identity is checked before labels are attached or scans trained.
        for split, value in ((kind, manifest), (other_kind, other)):
            rows = value["records"]
            if (value["version"] != SOURCE_VERSION
                    or any(row.get("source") != "nuscenes" or row.get("subset") != split
                           or row.get("delta") or row.get("anomaly", 0) for row in rows)):
                raise ValueError("normal source records do not belong to their original partition")
            for key in ("token", "sample_token", "scan", "label"):
                if len({row[key] for row in rows}) != len(rows):
                    raise ValueError("duplicate normal source identity: " + key)
        for key in ("scene", "log_token", "token", "sample_token", "scan", "label"):
            if {row[key] for row in manifest["records"]} & {row[key] for row in other["records"]}:
                raise ValueError("normal source training/development overlap: " + key)
        records = manifest["records"]
        for row in records:
            for key in ("scan", "label"):
                digest = file_sha256(row[key])
                expected = row.get(key + "_sha256")
                if expected is not None and expected != digest:
                    raise ValueError("normal annotation source file changed: " + key)
                # The run identity must include the bytes, including unrefined source scans.
                row[key + "_sha256"] = digest
        return attach_normal_annotations(records, manifest["root"], directory=directory)
    if source not in ("206", "201") or development != (source == "201"):
        raise ValueError("normal protocol permits 206 training and 201 development only")
    directory = Path(root or DATA_ROOT) / "train" / source
    calibration = {}
    for line in (directory / "calib.txt").read_text().splitlines():
        if line.strip():
            key, value = line.split(":", 1)
            if key in calibration:
                raise ValueError("duplicate normal sequence calibration: " + key)
            matrix = np.eye(4)
            matrix[:3] = np.fromstring(value, sep=" ").reshape(3, 4)
            calibration[key] = matrix
    camera = np.loadtxt(directory / "poses.txt").reshape(-1, 3, 4)
    poses = np.broadcast_to(np.eye(4), (len(camera), 4, 4)).copy()
    poses[:, :3] = camera
    transform = calibration["Tr"]
    rigid(transform)
    poses = np.linalg.inv(transform) @ poses @ transform
    scans = sorted((directory / "velodyne").glob("*.bin"))
    labels = sorted((directory / "labels").glob("*.label"))
    expected = [f"{index:06d}" for index in range(449 if source == "206" else 682)]
    if ([p.stem for p in scans] != expected or [p.stem for p in labels] != expected
            or len(scans) != len(poses)):
        raise ValueError("normal sequence must preserve all original scan, label and pose identities")
    for pose in poses:
        rigid(pose)
    return [dict(source="normal_stu", scene=source, frame=int(scan.stem), scan=str(scan),
                 subset="val" if development else "train",
                 label=str(directory / "labels" / (scan.stem + ".label")),
                 scan_sha256=file_sha256(scan),
                 label_sha256=file_sha256(directory / "labels" / (scan.stem + ".label")),
                 pose=poses[i].tolist())
            for i, scan in enumerate(scans)]


def read_normal_record(record):
    if record["source"] not in ("nuscenes", "normal_stu"):
        raise ValueError("normal training requires original nuScenes or STU records")
    source = record["source"] == "nuscenes"
    buffers = {}
    for key in ("scan", "label"):
        content = Path(record[key]).read_bytes()
        expected = [item[key + "_sha256"] for item in (record, *record.get("normal_fine", ()))
                    if key + "_sha256" in item]
        if expected:
            actual_sha256 = hashlib.sha256(content).hexdigest()
            if any(value != actual_sha256 for value in expected):
                raise ValueError("normal annotation source file changed: " + key)
        buffers[key] = content
    # Decode the exact verified bytes; do not reopen a potentially replaced file.
    raw = np.frombuffer(buffers["scan"], dtype="<f4").reshape(-1, 5 if source else 4)
    label = np.frombuffer(buffers["label"], dtype=np.uint8 if source else "<u4")
    if not len(raw) or len(raw) != len(label) or not np.isfinite(raw).all():
        raise ValueError("normal scan and point labels do not correspond")
    if source and (np.any(label >= 32) or np.any((raw[:, 4] < 0) | (raw[:, 4] > 31))):
        raise ValueError("unknown original nuScenes category or beam identity")
    if not source:
        unified_labels(label)
    pose = np.asarray(record["pose"], dtype=np.float64)
    rigid(pose)
    label = label.astype(np.int64) & 65535
    if not source and np.any(label == 2):
        raise ValueError("anomaly label entered normal-only training/development")
    actual = np.any(raw[:, :3] != 0, axis=1)
    xyzi = raw[actual, :4].copy()
    if source:
        xyzi[:, 3] /= 255.
    label = label[actual]
    allowed = np.zeros((len(label), 19), dtype=bool)
    mapping = NUSCENES_NORMAL_SETS if source else {k: (v,) for k, v in STU_NORMAL_SEMANTICS.items()}
    for key, values in mapping.items():
        allowed[np.ix_(label == key, values)] = True
    if source:
        refine_normal_labels(record, xyzi, label, allowed, np.flatnonzero(actual))
    distance = np.linalg.norm(xyzi[:, :3], axis=1)
    allowed[(distance < 2.5) | (distance > 50)] = False
    return dict(xyzi=xyzi, allowed=allowed, slots=np.flatnonzero(actual), slot_count=len(raw),
                pose=pose)


class NormalScans:
    """Real normal labels, full input context and bounded supervised queries."""

    def __init__(self, records, *, augment=False, queries=4096, seed=206):
        if queries <= 0:
            raise ValueError("normal supervision requires a positive query budget")
        self.records, self.augment, self.queries, self.seed = records, augment, queries, seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        from .model import voxelize
        from .normal import hypothesis_observation
        epoch, index = index if isinstance(index, tuple) else (0, index)
        raw = read_normal_record(self.records[index])
        if self.augment:
            angle = np.random.default_rng(np.random.SeedSequence([self.seed, index, epoch, 613])).uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            raw["xyzi"][:, :3] = raw["xyzi"][:, :3] @ rotation.T
        result = voxelize(raw["xyzi"])
        result.update(allowed=torch.from_numpy(raw["allowed"]), slots=torch.from_numpy(raw["slots"]),
                      slot_count=raw["slot_count"], index=index,
                      observation=hypothesis_observation(raw["xyzi"]))
        valid = np.flatnonzero(raw["allowed"].any(1))
        # Fixed class-balanced query inclusion; the backbone still sees every return.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, epoch, 7291]))
        chosen = []
        for category in range(19):
            candidates = np.flatnonzero(raw["allowed"][:, category])
            if len(candidates):
                chosen.extend(rng.choice(candidates, min(len(candidates), self.queries // 19), replace=False))
        chosen = np.unique(chosen)
        remaining = np.setdiff1d(valid, chosen, assume_unique=True)
        chosen = np.r_[chosen, rng.choice(remaining, min(len(remaining), max(0, self.queries - len(chosen))), replace=False)]
        result["queries"] = torch.from_numpy(np.sort(chosen).astype(np.int64))
        return result


class Scans:
    """Read real STU evaluation scans without changing their original point slots."""

    def __init__(self, manifest):
        if manifest["version"] != MANIFEST_VERSION or manifest["kind"] not in ("val", "test"):
            raise ValueError("STU evaluation requires an unchanged real-scan manifest")
        self.manifest = manifest
        self.records = manifest["records"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        frame = read_scan(record["scan"], record["label"], partition=self.manifest["kind"],
                          expected=(record["scan_sha256"], record["label_sha256"]))
        if frame.labels is None:
            if self.manifest["kind"] != "test":
                raise ValueError("only hidden test scans may omit labels")
            if (int(frame.actual.sum()), len(frame.xyzi)) != (record["points"], record["slots"]):
                raise ValueError("hidden test point identities differ from manifest")
            return dict(xyzi=frame.xyzi[frame.actual].copy(), slots=frame.return_slots,
                        targets=None, instance=None, slot_count=len(frame.xyzi), index=index)
        targets = point_targets(frame)
        observed = (int(frame.actual.sum()), int((targets == 0).sum()),
                    int((targets == 1).sum()), len(frame.xyzi))
        expected = tuple(record[k] for k in ("points", "normal", "anomaly", "slots"))
        if observed != expected:
            raise ValueError(f"decoded counts differ from manifest: {observed} != {expected}")
        return dict(xyzi=frame.xyzi[frame.actual].copy(), slots=frame.return_slots,
                    targets=targets[frame.actual].copy(), instance=frame.instance[frame.actual].copy(),
                    slot_count=len(frame.xyzi), index=index)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare normal nuScenes sources or real STU validation scans.")
    parser.add_argument("operation", choices=("nuscenes", "val", "test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nuscenes-root", type=Path, default=NUSCENES_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--workers", type=int, default=min(4, len(os.sched_getaffinity(0))))
    parser.add_argument("--normal-annotations", type=Path, help="Exact reviewed native source labels")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.operation == "nuscenes":
        if args.output is None:
            parser.error("nuscenes requires an explicit new --output directory")
        from .nuscenes import build
        build(args.nuscenes_root, args.output, args.workers, normal_annotations=args.normal_annotations)
        return
    if args.normal_annotations is not None:
        parser.error("--normal-annotations requires nuscenes")
    output = args.output or Path(f"assets/{args.operation}.json")
    if output.exists():
        parser.error("evaluation manifest already exists")
    manifest = make_real_manifest(args.data_root / args.operation, partition=args.operation, workers=args.workers)
    write_json(output, manifest)
    print(json.dumps(dict(scans=len(manifest["records"]),
        eligible=sum(bool(r["eligible"]) for r in manifest["records"]),
        labeled=all(r["label"] is not None for r in manifest["records"]), sha256=manifest["sha256"])))


if __name__ == "__main__":
    main()
