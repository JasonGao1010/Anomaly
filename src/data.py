"""Read complete STU and nuScenes observations with separate training/evaluation rules.

File and pose conventions are taken from AJAE-v3/src/scene.py. Ray calibration
uses AJAE/assets/rays.npz and the formula already measured in the 206 analysis.
"""

from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
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
RAYS_PATH = Path(__file__).resolve().parents[1] / "assets" / "rays.npz"
VERSION = "AJAE-V4-F240-R2"
PILOT_VERSION = "AJAE-V4-P1"
CONTINUATION_VERSION = "AJAE-V4-P2"
NATIVE_VERSION = "AJAE-V4-N1"
NDP_VERSION = "AJAE-V4-NDP"
SOURCE_VERSION = "AJAE-V4-NS"
# R2 changes the training budget; retain the exact R1 observations and manifests.
MANIFEST_VERSION = "AJAE-V4-F240-R1"
DATA_ROOT = Path("/home/jasongao/Data/STU")
POOL_ROOT = Path("/home/jasongao/Study/AJAE/results/synthetic")
NUSCENES_ROOT = Path("/home/jasongao/Data/Nuscenes")
# Retain known road-scene classes; ambiguous entities and noise receive no target.
NUSCENES_NORMAL = frozenset((
    "human.pedestrian.adult", "human.pedestrian.child",
    "human.pedestrian.construction_worker", "human.pedestrian.police_officer",
    "vehicle.bicycle", "vehicle.emergency.ambulance", "vehicle.emergency.police",
    "vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.car", "vehicle.construction",
    "vehicle.motorcycle", "vehicle.trailer", "vehicle.truck", "flat.driveable_surface",
    "flat.sidewalk", "flat.terrain", "static.vegetation",
))

# Empirical redundancy tolerances, fixed before model evaluation. They organize
# observations; none of them changes point labels, network inputs or evaluation.
DIVERSITY = dict(range_ratio=1.25, view_degrees=15., count_ratio=1.6,
                 spread_ratio=1.6, shape_change=.15, intensity_change=.15,
                 context_change=.35, context_count_ratio=2., context_radius_m=3.,
                 normal_translation_m=3., normal_rotation_degrees=10.,
                 normal_cell_count_scale=32., normal_cell_count_ratio=2.)


def observation_descriptor(xyzi, object_pose, sensor_pose, context):
    """Describe a measured object view; context is counts in fixed semantic groups."""
    xyz = xyzi[:, :3].astype(float)
    eig = np.maximum(np.linalg.eigvalsh(np.cov(xyz.T)), 0)[::-1] if len(xyz) >= 3 else np.zeros(3)
    direction = (np.asarray(sensor_pose)[:3, 3] - np.asarray(object_pose)[:3, 3]) @ np.asarray(object_pose)[:3, :3]
    direction /= np.linalg.norm(direction)
    context = np.asarray(context, float)
    return np.r_[np.log(np.median(np.linalg.norm(xyz, axis=1))), direction, np.log(len(xyz)),
                 np.log(.05 + 2 * np.sqrt(eig)),
                 (eig[0]-eig[1])/max(eig[0], 1e-12), (eig[1]-eig[2])/max(eig[0], 1e-12),
                 np.log1p(np.quantile(xyzi[:, 3], [.1, .5, .9])), np.log1p(context.sum()),
                 context/max(context.sum(), 1.)].tolist()


def observation_distance(features, reference):
    """A view is redundant only when all recorded conditions/outcomes are close."""
    x, y, p = np.asarray(features), np.asarray(reference), DIVERSITY
    delta = abs(x-y)
    angle = np.arccos(np.clip(x[:, 1:4] @ y[1:4], -1, 1))
    return np.maximum.reduce((delta[:, 0]/np.log(p["range_ratio"]),
        angle/np.deg2rad(p["view_degrees"]), delta[:, 4]/np.log(p["count_ratio"]),
        delta[:, 5:8].max(1)/np.log(p["spread_ratio"]), delta[:, 8:10].max(1)/p["shape_change"],
        delta[:, 10:13].max(1)/p["intensity_change"], delta[:, 13]/np.log(p["context_count_ratio"]),
        delta[:, 14:].sum(1)/p["context_change"]))


def select_observations(features, required=(), *, normal=False):
    """Farthest-first covering with a distance tolerance, never a frame quota."""
    x = np.asarray(features, float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("observation descriptors must be a nonempty finite matrix")
    chosen = list(dict.fromkeys(map(int, required)))
    distance = lambda i: np.max(abs(x-x[i]), axis=1) if normal else observation_distance(x, x[i])
    if not chosen:
        chosen = [int(np.argmin(np.square(x-np.median(x, axis=0)).mean(1)))]
    nearest = np.full(len(x), np.inf)
    for index in chosen:
        nearest = np.minimum(nearest, distance(index))
    nearest[chosen] = 0
    while nearest.max() > 1. + 1e-9:
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, distance(index))
        nearest[chosen] = 0
    return sorted(chosen), nearest


def context_groups(semantic, mapping=None):
    """Road/other ground, vegetation, vehicles, fence/barrier, built, people, other."""
    if mapping is None:
        groups = ((40, 44), (48, 49, 72), (70, 71), (10, 11, 13, 15, 16, 18, 20, 252, 256, 257, 258, 259),
                  (51,), (50, 52, 80, 81), (30, 31, 32, 253, 254, 255))
        return np.select([np.isin(semantic, g) for g in groups], np.arange(len(groups)), default=7)
    table = []
    for row in mapping:
        name = row["name"]
        table.append(0 if name == "flat.driveable_surface" else 1 if name.startswith("flat.") else
            2 if name == "static.vegetation" else 3 if name.startswith("vehicle.") else
            4 if name == "movable_object.barrier" else 5 if name == "static.manmade" else
            6 if name.startswith("human.") else 7)
    return np.asarray(table)[semantic]


def normal_descriptor(frame, raw_labels):
    """Whole-scan pose and trusted-normal class/range distribution; no cropping."""
    normal = point_targets(frame) == 0
    band = np.searchsorted([10., 20., 35.], frame.range_m[normal], side="right")
    counts = np.bincount(np.asarray(raw_labels)[normal]*4+band, minlength=128)
    p = DIVERSITY
    return np.r_[frame.pose[:3, 3]/p["normal_translation_m"],
        frame.pose[:3, :3].ravel()/(2*np.sin(np.deg2rad(p["normal_rotation_degrees"])/2)),
        np.log1p(counts/p["normal_cell_count_scale"])/np.log(p["normal_cell_count_ratio"])].tolist()


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


class STUSequence:
    """Stream the 449 original training scans that define the V4 background."""

    def __init__(self, data_root):
        self.directory = Path(data_root).expanduser().resolve(strict=True) / "train" / "206"
        expected = [f"{index:06d}" for index in range(449)]
        for name, suffix in (("velodyne", ".bin"), ("labels", ".label")):
            if sorted(path.stem for path in (self.directory / name).glob(f"*{suffix}")) != expected:
                raise ValueError(f"{name} must cover every 206 frame 0..448 exactly")
        calibration = {}
        for line in (self.directory / "calib.txt").read_text().splitlines():
            if not line.strip():
                continue
            key, text = line.split(":", 1)
            if key in calibration:
                raise ValueError(f"duplicate calibration key: {key}")
            matrix = np.eye(4)
            matrix[:3] = np.asarray([float(v) for v in text.split()]).reshape(3, 4)
            if not np.isfinite(matrix).all():
                raise ValueError("calibration contains nonfinite values")
            calibration[key] = matrix
        transform = calibration["Tr"]
        rigid(transform)
        camera = np.loadtxt(self.directory / "poses.txt")
        if camera.shape != (449, 12):
            raise ValueError("206 poses must contain one 3 x 4 matrix per scan")
        poses = np.broadcast_to(np.eye(4), (449, 4, 4)).copy()
        poses[:, :3] = camera.reshape(449, 3, 4)
        # KITTI camera poses are conjugated by Tr; they are not LiDAR poses directly.
        inverse = np.linalg.inv(transform)
        self.poses = readonly(np.stack([inverse @ pose @ transform for pose in poses]))
        for pose in self.poses:
            rigid(pose)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, frame_id):
        if type(frame_id) is not int or not 0 <= frame_id < len(self):
            raise IndexError(frame_id)
        scan = self.directory / "velodyne" / f"{frame_id:06d}.bin"
        label = self.directory / "labels" / f"{frame_id:06d}.label"
        if scan.stat().st_size != 131072 * 16 or label.stat().st_size != 131072 * 4:
            raise ValueError(f"invalid 206 point/label byte lengths at frame {frame_id}")
        frame = Frame(frame_id, np.fromfile(scan, dtype="<f4").reshape(-1, 4),
                      self.poses[frame_id], np.fromfile(label, dtype="<u4"))
        unknown = set(map(int, np.unique(frame.semantic))) - set(LABELS)
        if unknown:
            raise ValueError(f"unknown 206 raw semantics at frame {frame_id}: {unknown}")
        return frame


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


def legacy_source_identity(frame):
    """Reproduce the source binding already stored in AJAE/V3 delta files.

    The old 19-class map is storage metadata only, never a V4 learning target.
    """
    groups = ((10, 252), (11,), (15,), (18, 258), (13, 16, 20, 256, 257, 259),
              (30, 254), (31, 253), (32, 255), (40, 60), (44,), (48,), (49,),
              (50,), (51,), (70,), (71,), (72,), (80,), (81,))
    mapping = np.full(65536, 255, np.uint8)
    for target, raw in enumerate(groups):
        mapping[list(raw)] = target
    digest = hashlib.sha256(f"{frame.partition}/{frame.sequence_id}/{frame.frame_id}".encode())
    for values in (frame.xyzi, frame.labels, frame.pose, mapping[frame.semantic]):
        digest.update(values.tobytes())
    return digest.hexdigest()


def read_delta(path):
    """Read an existing one-object delta without constructing or filtering a scan."""
    fields = {"format", "source_identity", "world_identity", "source_slot", "xyzi",
              "packed_labels", "inserted_slot", "occluded_slot"}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != fields or saved["format"].item() != "stu-frozen-frame":
            raise ValueError("unrecognized saved scan delta")
        delta = {name: saved[name] for name in fields - {"format"}}
    for name in ("source_identity", "world_identity"):
        delta[name] = str(delta[name].item())
    for name in ("source_slot", "inserted_slot", "occluded_slot"):
        slots = delta[name]
        if slots.dtype != np.int32 or slots.ndim != 1 or np.any(slots < 0) or np.any(np.diff(slots) <= 0):
            raise ValueError(f"invalid sorted slot set: {name}")
    slots, inserted, occluded = (delta[name] for name in ("source_slot", "inserted_slot", "occluded_slot"))
    if not np.array_equal(slots, np.union1d(inserted, occluded)):
        raise ValueError("changed slots do not equal inserted/occluded slots")
    xyzi, packed = delta["xyzi"], delta["packed_labels"]
    if xyzi.dtype != np.float32 or xyzi.shape != (len(slots), 4) or not np.isfinite(xyzi).all():
        raise ValueError("invalid saved XYZI")
    if packed.dtype != np.uint32 or packed.shape != (len(slots),):
        raise ValueError("invalid saved labels")
    selected = np.isin(slots, inserted, assume_unique=True)
    if np.any(~np.any(xyzi[selected, :3] != 0, axis=1)) or np.any(packed[selected] != (np.uint32(60001) << np.uint32(16) | np.uint32(2))):
        raise ValueError("saved anomaly returns/labels disagree")
    if np.any(xyzi[~selected] != 0) or np.any(packed[~selected] != 0):
        raise ValueError("opaque occlusion without return must clear XYZI and labels")
    return delta


def validate_delta(delta, original, world_identity, *, source_identity=None):
    """Bind delta rows to their original scan and fixed world before using them.

    Analysis may reuse the source binding computed once for this exact Frame.
    """
    identity = legacy_source_identity(original) if source_identity is None else source_identity
    if delta["source_identity"] != identity or delta["world_identity"] != world_identity:
        raise ValueError("saved delta belongs to a different source or world")
    slots, inserted, occluded = (delta[name] for name in ("source_slot", "inserted_slot", "occluded_slot"))
    if len(slots) and slots[-1] >= len(original.xyzi):
        raise ValueError("saved delta slot exceeds the original scan")
    if np.any(~original.actual[occluded]):
        raise ValueError("an originally empty slot cannot be occluded")
    if not np.array_equal(inserted[original.actual[inserted]], np.intersect1d(inserted, occluded, assume_unique=True)):
        raise ValueError("inserted returns over original returns must record occlusion")


def restore_delta(path, original, world_identity):
    """Restore full XYZI and raw labels; the caller applies the common supervision rule."""
    delta = read_delta(path)
    validate_delta(delta, original, world_identity)
    xyzi, labels = original.xyzi.copy(), original.labels.copy()
    xyzi[delta["source_slot"]] = delta["xyzi"]
    labels[delta["source_slot"]] = delta["packed_labels"]
    return Frame(original.frame_id, xyzi, original.pose, labels,
                 sequence_id=original.sequence_id, partition=original.partition)


@dataclass(frozen=True, slots=True)
class Rays:
    """Calibrated origins and directions in original file-slot order."""

    directions: np.ndarray
    origins: np.ndarray
    canonical_ids: np.ndarray
    local: np.ndarray
    beam_ids: np.ndarray | None = None
    returned: np.ndarray | None = None

    def __post_init__(self):
        count = len(self.directions)
        if self.directions.shape != (count, 3) or self.origins.shape != (count, 3):
            raise ValueError("ray origins and directions must be aligned [N,3]")
        if not np.isfinite(self.directions).all() or not np.isfinite(self.origins).all():
            raise ValueError("ray geometry must be finite")
        if not np.allclose(np.linalg.norm(self.directions, axis=1), 1, atol=1e-7, rtol=1e-7):
            raise ValueError("ray directions must be unit vectors")
        if self.canonical_ids.shape != (count,) or not np.issubdtype(self.canonical_ids.dtype, np.integer) or not np.array_equal(np.sort(self.canonical_ids), np.arange(count)):
            raise ValueError("canonical ray IDs must form a complete permutation")
        if self.local.ndim != 2 or self.local.shape[1] != 3 or not len(self.local) or count % len(self.local) or not np.isfinite(self.local).all():
            raise ValueError("local beam vectors must be finite [beam,3] dividing the slot count")
        for name in ("directions", "origins", "canonical_ids", "local"):
            object.__setattr__(self, name, readonly(getattr(self, name).copy()))
        if self.beam_ids is not None:
            beam = np.asarray(self.beam_ids)
            if beam.shape != (count,) or not np.issubdtype(beam.dtype, np.integer) or np.any((beam < 0) | (beam >= len(self.local))):
                raise ValueError("explicit beam IDs must identify every native firing slot")
            object.__setattr__(self, "beam_ids", readonly(beam.copy()))
        if self.returned is not None:
            returned = np.asarray(self.returned)
            if returned.shape != (count,) or returned.dtype != np.bool_:
                raise ValueError("native return flags must align with the firing slots")
            object.__setattr__(self, "returned", readonly(returned.copy()))


def read_rays(path=RAYS_PATH):
    """Read the measured 206 ray parameters; historical passed flags are not evidence."""
    with np.load(path, allow_pickle=False) as saved:
        parameters = np.asarray(saved["even_params"], dtype=np.float64)
        local = np.asarray(saved["even_local"], dtype=np.float64)
        shifts = saved["integer_shift"]
    if parameters.shape != (3,) or local.shape != (128, 3) or shifts.shape != (128,):
        raise ValueError("invalid 206 ray calibration array shapes")
    if not np.issubdtype(shifts.dtype, np.integer) or not np.isfinite(parameters).all():
        raise ValueError("invalid 206 ray calibration parameters")
    gamma, origin_x, origin_z = parameters
    # Keep the measured encoder gauge and integer row shifts, including empty slots.
    angle = math.pi + gamma - 2 * math.pi * (np.arange(1024)[None] - shifts[:, None]) / 1024
    cosine, sine = np.cos(angle), np.sin(angle)
    directions = np.stack((cosine * local[:, None, 0] - sine * local[:, None, 1],
                           sine * local[:, None, 0] + cosine * local[:, None, 1],
                           np.broadcast_to(local[:, None, 2], angle.shape)), axis=-1)
    origins = np.stack((origin_x * cosine, origin_x * sine, np.full_like(cosine, origin_z)), axis=-1)
    canonical = np.arange(128)[:, None] * 1024 + (np.arange(1024)[None] - shifts[:, None]) % 1024
    return Rays(directions.reshape(-1, 3), origins.reshape(-1, 3), canonical.ravel(), local)


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
        if expected is not None and (hashlib.sha256(raw_scan).hexdigest(),
                                     hashlib.sha256(raw_label).hexdigest()) != expected:
            raise ValueError(f"STU observation changed after manifest creation: {scan}")
    result = Frame(int(scan.stem), xyzi, np.eye(4), packed,
                   sequence_id=int(scan.parent.parent.name), partition=partition)
    if io_timing is not None:
        io_timing["seconds"] = read_seconds
    return result


def _census_source(task):
    data_root, pool_root, frame_id, worlds = task
    sequence = STUSequence(data_root)
    original = sequence[frame_id]
    source = legacy_source_identity(original)
    target = point_targets(original)
    normal, anomaly = int((target == 0).sum()), int((target == 1).sum())
    source_record = dict(frame=frame_id, source_identity=source,
                         scan=str(sequence.directory / "velodyne" / f"{frame_id:06d}.bin"),
                         label=str(sequence.directory / "labels" / f"{frame_id:06d}.label"))
    source_record["scan_sha256"] = file_sha256(source_record["scan"])
    source_record["label_sha256"] = file_sha256(source_record["label"])
    rows, skipped = [], 0
    for world in worlds:
        paths = [Path(pool_root) / p / "frames" / f"{frame_id:06d}.npz" for p in world["paths"]
                 if frame_id in world["frames_by_path"][p]]
        if not paths:
            continue
        hashes = [file_sha256(p) for p in paths]
        if len(set(hashes)) != 1:
            raise ValueError(f"conflicting duplicate world/frame: {world['id']}/{frame_id}")
        delta = read_delta(paths[0])
        validate_delta(delta, original, world["id"], source_identity=source)
        changed = delta["source_slot"]
        replacement = delta["xyzi"]
        replacement_labels = unified_labels(delta["packed_labels"])
        distance = np.linalg.norm(replacement[:, :3], axis=1)
        actual = np.any(replacement[:, :3] != 0, axis=1)
        valid = actual & (distance >= 2.5) & (distance <= 50) & (replacement_labels != 0)
        # Exact sparse replacement of full-scan counts; no object-level filtering.
        n_normal = normal - int((target[changed] == 0).sum()) + int((valid & (replacement_labels == 1)).sum())
        n_anomaly = anomaly - int((target[changed] == 1).sum()) + int((valid & (replacement_labels == 2)).sum())
        n_real = int(original.actual.sum() - original.actual[changed].sum() + actual.sum())
        if n_anomaly < 5:
            skipped += 1
            continue
        row = dict(world=world["id"], frame=frame_id, delta=str(paths[0].resolve()),
                   delta_sha256=hashes[0], source_identity=source, points=n_real,
                   normal=n_normal, anomaly=n_anomaly, slots=len(original.xyzi))
        row["content_sha256"] = identity(row)
        rows.append(row)
    return source_record, rows, skipped


def _census_real(task):
    scan, label, partition = task
    frame = read_scan(scan, label, partition=partition)
    selected = supervision(frame)
    return dict(sequence=frame.sequence_id, frame=frame.frame_id, scan=str(scan),
                label=str(label), scan_sha256=hashlib.sha256(frame.xyzi.tobytes()).hexdigest(),
                label_sha256=hashlib.sha256(frame.labels.tobytes()).hexdigest(),
                points=int(frame.actual.sum()), slots=len(frame.xyzi),
                normal=selected.normal_count, anomaly=selected.anomaly_count,
                eligible=selected.eligible)


def make_manifest(data_root=DATA_ROOT, pool_root=POOL_ROOT, workers=4):
    data_root, pool_root = Path(data_root).resolve(), Path(pool_root).resolve()
    pool_path = pool_root / "manifest.json"
    pool = json.loads(pool_path.read_text())
    split = pool["splits"]["train"]
    if split["source_sequence"] != 206:
        raise ValueError("F240 requires the saved 206 training pool")
    worlds = {}
    for entry in split["worlds"]:
        path, key = entry["path"], entry["world_identity"]
        folder = pool_root / path
        saved = json.loads((folder / "manifest.json").read_text())
        if saved["world_identity"] != key or saved["source_sequence"] != 206:
            raise ValueError(f"world source mismatch: {folder}")
        files = sorted((folder / "frames").glob("*.npz"))
        frames = sorted({int(p.stem) for p in files})
        if not frames or frames != sorted({int(r["frame"]) for r in saved["frames"]}):
            raise ValueError(f"saved frame index differs from actual files: {folder}")
        if len(frames) != len(files) or frames[0] < 0 or frames[-1] >= 449:
            raise ValueError(f"invalid or ambiguous frame names: {folder}")
        world = worlds.setdefault(key, dict(id=key, paths=[], frames_by_path={}, sources={}))
        if path not in world["paths"]:
            world["paths"].append(path)
            world["frames_by_path"][path] = frames
            world["sources"][path] = dict(manifest_sha256=file_sha256(folder / "manifest.json"),
                                          world_sha256=file_sha256(folder / "world.json"),
                                          generation_version=saved["configuration_identity"])
    if len(worlds) != 240:
        raise ValueError(f"expected exactly 240 distinct saved worlds, got {len(worlds)}")
    ordered = sorted(worlds.values(), key=lambda w: w["id"])
    frame_ids = sorted({f for w in ordered for fs in w["frames_by_path"].values() for f in fs})
    tasks = [(str(data_root), str(pool_root), f, ordered) for f in frame_ids]
    records, sources, skipped = [], [], 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, (source, rows, omitted) in enumerate(executor.map(_census_source, tasks), 1):
            sources.append(source)
            records.extend(rows)
            skipped += omitted
            if i % 25 == 0 or i == len(tasks):
                print(f"206 census {i}/{len(tasks)}: eligible={len(records)} skipped={skipped}", flush=True)
    records.sort(key=lambda r: (r["world"], r["frame"]))
    result = dict(version=MANIFEST_VERSION, kind="train", data_root=str(data_root), pool_root=str(pool_root),
                  pool_version=pool["configuration_identity"], pool_sha256=file_sha256(pool_path),
                  worlds=ordered, sources=sources, records=records, skipped=skipped,
                  calibration_sha256=file_sha256(data_root / "train/206/calib.txt"),
                  poses_sha256=file_sha256(data_root / "train/206/poses.txt"))
    result["sha256"] = identity(result)
    return result


def make_real_manifest(directory, *, partition="val", workers=4):
    directory = Path(directory).resolve(strict=True)
    sequences = sorted(p for p in directory.glob("1[0-9][0-9]") if p.is_dir())
    if not sequences:
        raise ValueError(f"no official STU sequences in {directory}")
    tasks = []
    for sequence in sequences:
        scans = sorted((sequence / "velodyne").glob("*.bin"))
        labels = sorted((sequence / "labels").glob("*.label"))
        if not scans or [p.stem for p in scans] != [p.stem for p in labels]:
            raise ValueError(f"missing or unmatched STU scans/labels: {sequence}")
        tasks.extend((s, t, partition) for s, t in zip(scans, labels))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_census_real, tasks, chunksize=8))
    result = dict(version=MANIFEST_VERSION, kind=partition, directory=str(directory),
                  sequences=[p.name for p in sequences], records=records)
    result["sha256"] = identity(result)
    return result


def load_manifest(path, kind):
    value = json.loads(Path(path).read_text())
    expected = value.pop("sha256")
    if identity(value) != expected or value["version"] not in (MANIFEST_VERSION, PILOT_VERSION, NATIVE_VERSION, NDP_VERSION, SOURCE_VERSION) or value["kind"] != kind:
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
        "animal": "Animal behavior and STU normal/ignore boundary are not specified; no stationary obstacle donor",
        "human.pedestrian.personal_mobility": "Person and mobility device are not separately labeled; ignore mixed category",
        "human.pedestrian.stroller": "Person/device boundary is not established; ignore, never an obstacle donor",
        "human.pedestrian.wheelchair": "Person/device boundary is not established; ignore, never an obstacle donor",
        "movable_object.barrier": "Temporary barriers are not identical to SemanticKITTI fences; target boundary unresolved",
        "movable_object.trafficcone": "No public point-level STU normal/ignore correspondence; do not infer anomaly from missing class name",
        "movable_object.debris": "Native debris remains ignored; only admitted, explicitly inserted objects are positive",
        "movable_object.pushable_pullable": "Native devices remain ignored; only admitted, unoccupied inserted obstacles are positive",
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


def nuscenes_poses(records, root=NUSCENES_ROOT):
    """Stream only selected sweeps and preceding poses, retaining world coordinates."""
    import ijson
    from scipy.spatial.transform import Rotation
    root = Path(root) / "v1.0-trainval"
    selected = {r["token"] for r in records}
    with (root / "sample_data.json").open("rb") as stream:
        scans = {r["token"]: r for r in ijson.items(stream, "item") if r["token"] in selected}
    if set(scans) != selected:
        raise ValueError("selected nuScenes sweep metadata is incomplete")
    adjacent = {r["prev"] or r["next"] for r in scans.values()} - set(scans)
    with (root / "sample_data.json").open("rb") as stream:
        scans.update({r["token"]: r for r in ijson.items(stream, "item") if r["token"] in adjacent})
    pose_tokens = {r["ego_pose_token"] for r in scans.values()}
    with (root / "ego_pose.json").open("rb") as stream:
        poses = {r["token"]: r for r in ijson.items(stream, "item") if r["token"] in pose_tokens}
    calibrations = {r["token"]: r for r in json.loads((root / "calibrated_sensor.json").read_text())}

    def matrix(row):
        result = np.eye(4)
        q = np.asarray(row["rotation"], float)
        result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        result[:3, 3] = np.asarray(row["translation"], float)
        return result

    output = []
    for record in records:
        row = scans[record["token"]]
        neighbor = scans[row["prev"] or row["next"]]
        transform = matrix(calibrations[row["calibrated_sensor_token"]])
        pose = matrix(poses[row["ego_pose_token"]]) @ transform
        other = matrix(poses[neighbor["ego_pose_token"]]) @ transform
        dt = (int(row["timestamp"]) - int(neighbor["timestamp"])) / 1e6
        if dt == 0 or (row["prev"] and dt < 0) or (not row["prev"] and dt > 0):
            raise ValueError("adjacent native sweep timestamps are not ordered")
        output.append(dict(record, pose=pose.tolist(), adjacent_pose=other.tolist(),
                           adjacent_seconds=dt, calibrated_sensor=row["calibrated_sensor_token"]))
    return output


def native_keyframes(reference, root=NUSCENES_ROOT):
    """All labeled keyframes in the already authorized training logs/scenes."""
    import ijson
    root = Path(root)
    split = reference["split"]
    allowed = set(split["train"])
    blocked = {split["logs"][s] for s in split["check"]}
    scenes = {r["token"]: r for r in json.loads((root/"v1.0-trainval/scene.json").read_text()) if r["name"] in allowed}
    if any(r["log_token"] in blocked for r in scenes.values()):
        raise ValueError("a held-out acquisition log entered expansion")
    samples = {r["token"]: scenes[r["scene_token"]] for r in json.loads((root/"v1.0-trainval/sample.json").read_text())
               if r["scene_token"] in scenes}
    labels = {r["sample_data_token"]: r["filename"] for r in json.loads((root/"v1.0-trainval/lidarseg.json").read_text())}
    old = {r["token"]: r for r in reference["records"] if r.get("source") == "nuscenes"}
    records = []
    with (root/"v1.0-trainval/sample_data.json").open("rb") as stream:
        for row in ijson.items(stream, "item"):
            if not row["is_key_frame"] or row["sample_token"] not in samples or not row["filename"].startswith("samples/LIDAR_TOP/"):
                continue
            scene = samples[row["sample_token"]]
            record = dict(source="nuscenes", scene=scene["name"], token=row["token"], frame=len(records),
                scan=str(root/row["filename"]), label=str(root/labels[row["token"]]),
                subset="train", group="normal_nuscenes", timestamp=int(row["timestamp"]),
                sample_token=row["sample_token"], log_token=scene["log_token"])
            if record["token"] in old and record["frame"] != old[record["token"]]["frame"]:
                raise ValueError("frame identity changed in the stateless observation random stream")
            records.append(record)
    if len(records) != sum(r["nbr_samples"] for r in scenes.values()) or not set(old) <= {r["token"] for r in records}:
        raise ValueError("incomplete labeled trajectory keyframes")
    return nuscenes_poses(records, root)


def nuscenes_rays(record):
    """Recover native 32-beam slots in the motion-compensated reference frame.

    Column times interpolate adjacent poses; zero-range records refine translation.
    Observed directions are measured, whereas missing-return directions are fitted.
    This is an empirical acquisition model, not recovered raw firing timestamps.
    """
    import warnings
    from scipy.spatial.transform import Rotation
    raw = np.fromfile(record["scan"], dtype="<f4").reshape(-1, 5)
    if len(raw) % 32 or not np.array_equal(raw[:, 4], np.tile(np.arange(32), len(raw) // 32)):
        raise ValueError("nuScenes firing order is not the native 32-beam column layout")
    count = len(raw) // 32
    xyz = raw[:, :3].astype(float).reshape(count, 32, 3)
    relative = np.linalg.inv(np.asarray(record["pose"])) @ np.asarray(record["adjacent_pose"])
    # A skipped preceding sweep must not double the current scan's rotation period.
    duration = min(abs(record["adjacent_seconds"]), count * 46.08e-6)
    fraction = (1 - np.arange(count) / count) * duration / record["adjacent_seconds"]
    rotations = Rotation.from_rotvec(fraction[:, None] * Rotation.from_matrix(relative[:3, :3]).as_rotvec()).as_matrix()
    origin = fraction[:, None] * relative[:3, 3]
    # Distinct beams cannot produce exactly coincident near-origin surface hits.
    # Repeated XYZ values therefore recover motion-shifted zero-depth slots even
    # when adjacent-pose interpolation is inaccurate during rapid ego motion.
    multiplicity = (xyz[:, :, None] == xyz[:, None, :]).all(-1).sum(-1)
    most = multiplicity.argmax(1)
    measured = xyz[np.arange(count), most]
    valid_origin = (multiplicity.max(1) >= 2) & (np.linalg.norm(measured - origin, axis=1) < 1.)
    correction = measured[valid_origin] - origin[valid_origin]
    if valid_origin.any():
        for axis in range(3):
            origin[:, axis] += np.interp(np.arange(count), np.flatnonzero(valid_origin), correction[:, axis])
    local = np.einsum("ncj,njk->nck", xyz - origin[:, None], rotations)
    distance = np.linalg.norm(local, axis=2)
    returned = distance > .05
    azimuth = np.arctan2(local[:, :, 1], local[:, :, 0])
    elevation = np.arcsin(np.clip(local[:, :, 2] / np.maximum(distance, 1e-12), -1, 1))
    usable = returned & (distance >= 2.5) & (distance <= 80)
    expected = -2 * np.pi * np.arange(count) / count
    wrap = lambda value: (value + np.pi) % (2 * np.pi) - np.pi
    phase = np.median(wrap(azimuth - expected[:, None])[usable])
    column = expected + phase
    row_offset = np.zeros(32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(2):
            residual = np.where(usable, wrap(azimuth - column[:, None]), np.nan)
            row_offset = np.nan_to_num(np.nanmedian(residual, axis=0))
            residual = np.where(usable, wrap(azimuth - column[:, None] - row_offset), np.nan)
            adjustment = np.nanmedian(residual, axis=1)
            valid = np.isfinite(adjustment)
            if not valid.any():
                raise ValueError("no measured directions support the native firing model")
            column += np.interp(np.arange(count), np.flatnonzero(valid), adjustment[valid])
        angles = np.nanmedian(np.where(usable, elevation, np.nan), axis=0)
    nominal = np.deg2rad(-30.6666666667 + np.arange(32) * 4 / 3)
    angles = np.where(np.isfinite(angles), angles, nominal)
    az = column[:, None] + row_offset
    fitted = np.stack((np.cos(angles)[None] * np.cos(az), np.cos(angles)[None] * np.sin(az),
                       np.broadcast_to(np.sin(angles), az.shape)), -1)
    measured_direction = local / np.maximum(distance[..., None], 1e-12)
    angular_error = np.rad2deg(np.arccos(np.clip((fitted * measured_direction).sum(-1)[usable], -1, 1)))
    direction = np.where(returned[..., None], measured_direction, fitted)
    direction = np.einsum("ncj,nkj->nck", direction, rotations)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    local_beams = np.column_stack((np.cos(angles), np.zeros(32), np.sin(angles)))
    rays = Rays(direction.reshape(-1, 3), np.repeat(origin, 32, axis=0), np.arange(len(raw)),
                local_beams, np.tile(np.arange(32), count), returned.ravel())
    diagnostics = dict(columns=count, beams=32, measured_returns=int(returned.sum()),
                       inferred_empty_rays=int((~returned).sum()), duration_seconds=float(duration),
                       angular_residual_deg=dict(zip(("median", "p95", "p99", "max"),
                           map(float, np.quantile(angular_error, [.5, .95, .99, 1])))),
                       origin_correction_max_m=float(np.linalg.norm(correction, axis=1).max()) if len(correction) else None)
    return rays, diagnostics


def nuscenes_truth(record, labels, mapping):
    """Apply reviewed point labels before any synthetic foreground replacement."""
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
    if "delta" in record:
        with np.load(record["delta"], allow_pickle=False) as delta:
            if str(delta["token"]) != record["token"]:
                raise ValueError("nuScenes delta belongs to a different original sweep")
            xyzi[delta["slots"]], truth[delta["slots"]] = delta["xyzi"], delta["labels"]
    return Frame(record["frame"], xyzi, np.asarray(record.get("pose", np.eye(4)), dtype=float),
                 truth, sequence_id=0, partition="train")


def _census_normal(task):
    record, mapping = task
    try:
        frame = read_nuscenes(record, mapping)
        target = point_targets(frame)
        record = dict(record, points=int(frame.actual.sum()), slots=len(frame.xyzi),
                      normal=int((target == 0).sum()), anomaly=int((target == 1).sum()))
        if record["anomaly"] or not record["normal"]:
            raise ValueError("normal nuScenes scan has inconsistent supervision")
        return record, None
    except (OSError, ValueError) as error:
        return None, dict(record, error=f"{type(error).__name__}: {error}")


def make_normal_manifest(root, output, *, scenes=16, check_scenes=4, seed=2064, expanded=False, workers=4):
    """Select official train scenes before reading labels; hold out whole logs."""
    import ast
    import urllib.request
    import ijson
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/utils/splits.py"
    with urllib.request.urlopen(url, timeout=30) as response:
        split_source = response.read()
    lists = {}
    for node in ast.parse(split_source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for name in node.targets:
                if isinstance(name, ast.Name):
                    lists[name.id] = ast.literal_eval(node.value)
    official = set(lists["train_detect"] + lists["train_track"])
    scene_rows = json.loads((root / "v1.0-trainval/scene.json").read_text())
    groups = {}
    for row in scene_rows:
        if row["name"] in official:
            groups.setdefault(row["log_token"], []).append(row)
    rng = np.random.default_rng(seed)
    logs = sorted(groups)
    selected = []
    for idx in rng.permutation(len(logs))[:scenes + check_scenes]:
        choices = sorted(groups[logs[idx]], key=lambda row: row["name"])
        selected.append(choices[int(rng.integers(len(choices)))])
    if len(selected) != scenes + check_scenes:
        raise ValueError("insufficient independent training logs")
    split = dict(seed=seed, official_source=url,
                 source_sha256=hashlib.sha256(split_source).hexdigest(),
                 train=[row["name"] for row in selected[:scenes]],
                 check=[row["name"] for row in selected[scenes:]],
                 logs={row["name"]: row["log_token"] for row in selected},
                 geometry_train=[f"proxy-{i:03d}" for i in range(12)],
                 geometry_check=[f"proxy-{i:03d}" for i in range(12, 15)],
                 excluded_stu=[201], stu_train=[206])
    reference = load_manifest(output / "normal.json", "normal") if expanded else None
    if expanded:
        original = reference["split"]
        blocked = {original["logs"][name] for name in original["check"]}
        by_name = {row["name"]: row for row in scene_rows}
        if any(by_name[name]["log_token"] != original["logs"][name] for name in original["check"]):
            raise ValueError("internal check scene/log identity changed")
        excluded = sorted(row["name"] for row in scene_rows
                          if row["name"] in official and row["log_token"] in blocked)
        selected = sorted((row for row in scene_rows if row["name"] in official
                           and row["log_token"] not in blocked), key=lambda row: row["name"])
        split = dict(original, official_source=url, source_sha256=hashlib.sha256(split_source).hexdigest(),
                     train=[row["name"] for row in selected],
                     logs={**{name: original["logs"][name] for name in original["check"]},
                           **{row["name"]: row["log_token"] for row in selected}},
                     excluded_logs=sorted(blocked), excluded_scenes=excluded)
    else:
        write_json(output / "split.json", split)
    mapping = nuscenes_mapping(root)
    if expanded and mapping != reference["mapping"]:
        raise ValueError("background expansion must preserve the verified label mapping")
    if not expanded:
        write_json(output / "labels.json", dict(
            unified={"0": "忽略", "1": "正常", "2": "合成异常代理"},
            nuscenes=mapping,
            stu=[dict(raw=k, name=v, target=0 if k == 0 else 2 if k == 2 else 1)
                 for k, v in LABELS.items()],
            intensity={"STU": "原始值", "nuScenes": "原始强度除以固定常数255"},
            train_frames="可信正常且有有效监督，或2.5–50米内异常总数不少于5；1–4跳过",
            evaluation="固定官方实现；异常总数少于5均跳过，包括零异常帧",
            category_source=str(root / "lidarseg/category.json")))
    tokens = {row["token"]: row["name"] for row in selected}
    samples = {row["token"]: tokens[row["scene_token"]]
               for row in json.loads((root / "v1.0-trainval/sample.json").read_text())
               if row["scene_token"] in tokens}
    labels = {row["sample_data_token"]: row["filename"]
              for row in json.loads((root / "v1.0-trainval/lidarseg.json").read_text())}
    records = []
    with (root / "v1.0-trainval/sample_data.json").open("rb") as stream:
        for row in ijson.items(stream, "item"):
            if (not row["is_key_frame"] or row["sample_token"] not in samples
                    or not row["filename"].startswith("samples/LIDAR_TOP/")):
                continue
            scene = samples[row["sample_token"]]
            record = dict(source="nuscenes", scene=scene, token=row["token"],
                          frame=len(records), scan=str(root / row["filename"]),
                          label=str(root / labels[row["token"]]),
                          subset="train" if scene in split["train"] else "check",
                          group="normal_nuscenes")
            if expanded:
                record.update(timestamp=int(row["timestamp"]), sample_token=row["sample_token"],
                              log_token=split["logs"][scene])
            records.append(record)
    counts = {row["name"]: sum(r["scene"] == row["name"] for r in records) for row in selected}
    if any(counts[row["name"]] != row["nbr_samples"] for row in selected):
        raise ValueError("selected scenes have missing labeled keyframes")
    if expanded:
        chosen = []
        for scene in split["train"]:
            rows = sorted((r for r in records if r["scene"] == scene), key=lambda r: r["timestamp"])
            # Select temporal coverage before examining labels or model scores.
            indices = np.rint(np.linspace(0, len(rows) - 1, min(8, len(rows)))).astype(int)
            chosen.extend(rows[i] for i in indices)
        records = chosen
    candidate_count = len(records)
    valid, skipped = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for number, (record, error) in enumerate(executor.map(
                _census_normal, ((row, mapping) for row in records), chunksize=16), 1):
            if error:
                if not expanded:
                    raise ValueError(error)
                skipped.append(error)
            else:
                valid.append(record)
            if number % 500 == 0 or number == candidate_count:
                print(f"normal census {number}/{candidate_count}: valid={len(valid)} skipped={len(skipped)}", flush=True)
    records = valid
    counts = {row["name"]: sum(r["scene"] == row["name"] for r in records) for row in selected}
    result = dict(version=PILOT_VERSION, kind="normal", mapping=mapping,
                  split=split, records=records, scene_counts=counts,
                  source=json.loads((root / "lidarseg/source.json").read_text()))
    if expanded:
        result.update(reference_manifest=reference["sha256"], selection=dict(
            official_train_scenes=len(official), excluded_scenes=len(excluded),
            candidate_scenes=len(selected), available_scenes=sum(n > 0 for n in counts.values()),
            available_logs=len({r["log_token"] for r in records}), frames_per_scene=8,
            method="rounded linspace over chronological native keyframes, including endpoints",
            candidate_frames=candidate_count, valid_frames=len(records), skipped=skipped,
            normal_points=sum(r["normal"] for r in records)))
    result["sha256"] = identity(result)
    write_json(output / ("background.json" if expanded else "normal.json"), result)
    print(json.dumps(dict(normal_scans=len(records), scenes=sum(n > 0 for n in counts.values()),
                          selection=result.get("selection"))), flush=True)
    return result


def replace_background(train, normal_path, reference_path):
    """Replace only native nuScenes records; retain every other record verbatim."""
    normal = load_manifest(normal_path, "normal")
    blocked = {train["split"]["logs"][name] for name in train["split"]["check"]}
    rows = normal["records"]
    if (normal["mapping"] != train["mapping"] or normal["reference_manifest"] != train["normal_manifest"]
            or set(normal["split"]["excluded_logs"]) != blocked
            or normal["split"]["check"] != train["split"]["check"]
            or len({r["token"] for r in rows}) != len(rows)
            or any(r["log_token"] in blocked or r["subset"] != "train" or r["source"] != "nuscenes"
                   or r["group"] != "normal_nuscenes" or r["anomaly"] or r["normal"] <= 0 for r in rows)):
        raise ValueError("expanded normal sources violate label or held-out-log boundaries")
    result = dict(train, records=[r for r in train["records"] if r["group"] != "normal_nuscenes"] + rows,
                  split=normal["split"], normal_manifest=normal["sha256"], normal_path=str(Path(normal_path).resolve()),
                  reference_train_manifest=train["sha256"], reference_train_path=str(Path(reference_path).resolve()))
    result.pop("sha256")
    result["sha256"] = identity(result)
    return result


def make_pilot_manifest(base_path, output):
    """Mix existing exposure with new training sources; checks never enter training."""
    output = Path(output)
    base = load_manifest(base_path, "train")
    normal = load_manifest(output / "normal.json", "normal")
    targeted = json.loads((output / "targeted.json").read_text())
    split = json.loads((output / "split.json").read_text())
    if normal["split"] != split or split["stu_train"] != [206] or split["excluded_stu"] != [201]:
        raise ValueError("inconsistent training source split")
    if set(split["train"]) & set(split["check"]) or set(split["geometry_train"]) & set(split["geometry_check"]):
        raise ValueError("internal check sources overlap training")
    records = [dict(row, group="base", subset="train") for row in base["records"]]
    for entry in targeted["worlds"]:
        if entry["accepted"] and entry["subset"] == "train":
            world = json.loads(Path(entry["path"]).read_text())
            if world["geometry"] not in split["geometry_train"]:
                raise ValueError("unexpected targeted training geometry")
            records.extend(world["frames"])
    records.extend(row for row in normal["records"] if row["subset"] == "train")
    sequence = STUSequence(base["data_root"])
    for source in base["sources"]:
        frame = sequence[source["frame"]]
        selected = supervision(frame, allow_normal=True)
        if (not selected.eligible or selected.anomaly_count or
                legacy_source_identity(frame) != source["source_identity"]):
            raise ValueError("raw 206 scan is not the trusted normal source")
        records.append(dict(source="normal_stu", group="normal_stu", subset="train",
                            frame=frame.frame_id, points=int(frame.actual.sum()), slots=len(frame.xyzi),
                            normal=selected.normal_count, anomaly=0))
    if any(row["subset"] != "train" or 1 <= row["anomaly"] <= 4 for row in records):
        raise ValueError("an ineligible scan entered pilot training")
    result = dict(base, version=PILOT_VERSION, records=records, mapping=normal["mapping"],
                  split=split, base_manifest=base["sha256"], normal_manifest=normal["sha256"],
                  targeted_source=str((output / "targeted.json").resolve()))
    result.pop("sha256")
    result["sha256"] = identity(result)
    write_json(output / "train.json", result)
    counts = {group: sum(row["group"] == group for row in records)
              for group in ("base", "targeted", "normal_nuscenes", "normal_stu")}
    print(json.dumps(dict(training_scans=counts, sha256=result["sha256"])), flush=True)
    return result


def _stu_diversity_frame(task):
    """Read a raw frame once for all existing world observations at that time."""
    from scipy.spatial import cKDTree
    frame_id, records = task
    source = _diversity_sequence[frame_id]
    if legacy_source_identity(source) != records[0]["source_identity"]:
        raise ValueError("STU training background changed")
    original_ids = np.flatnonzero(source.actual)
    tree = cKDTree(source.xyzi[original_ids, :3])
    groups = context_groups(source.semantic)
    result = []
    for record in records:
        delta = read_delta(record["delta"])
        if str(delta["world_identity"]) != record["world"] or str(delta["source_identity"]) != record["source_identity"]:
            raise ValueError("STU observation has the wrong source identity")
        xyz = delta["xyzi"]
        radius = np.linalg.norm(xyz[:, :3], axis=1)
        mask = ((delta["packed_labels"] & 65535) == 2) & (radius >= 2.5) & (radius <= 50)
        if int(mask.sum()) != record["anomaly"]:
            raise ValueError("STU anomaly supervision changed during expansion")
        obj = _diversity_worlds[record["world"]]
        center = (obj["pose"][:3, 3]-source.pose[:3, 3]) @ source.pose[:3, :3]
        context = original_ids[tree.query_ball_point(center, DIVERSITY["context_radius_m"])]
        context = context[~np.isin(context, delta["occluded_slot"])]
        features = observation_descriptor(xyz[mask], obj["pose"], source.pose,
                                         np.bincount(groups[context], minlength=8))
        result.append((record["world"], frame_id, features))
    return result


def stu_observations(base, workers):
    """Measure every eligible existing STU view with one raw read per time step."""
    import multiprocessing as mp
    global _diversity_sequence, _diversity_worlds
    _diversity_sequence = STUSequence(base["data_root"])
    _diversity_worlds, by_frame = {}, {}
    for entry in base["worlds"]:
        directory = Path(base["pool_root"])/entry["paths"][0]
        obj = json.loads((directory/"world.json").read_text())["world"]["objects"][0]
        metadata = json.loads((directory/"manifest.json").read_text())
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = obj["rotation_world_from_local"], obj["translation_world_m"]
        _diversity_worlds[entry["id"]] = dict(pose=pose, geometry=identity(obj["shape"]),
            source_family=metadata.get("family_id", "unrecorded:"+identity(obj["shape"])))
    for record in base["records"]:
        by_frame.setdefault(record["frame"], []).append(record)
    features = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        for rows in pool.map(_stu_diversity_frame, sorted(by_frame.items())):
            features.update({(world, frame): vector for world, frame, vector in rows})
    return _diversity_worlds, features


def expand_stu(reference, output, workers):
    """Add nonredundant existing views while retaining every baseline STU record."""
    base = load_manifest(Path(__file__).resolve().parents[1]/"assets/train.json", "train")
    if base["sha256"] != reference["base_manifest"]:
        raise ValueError("expanded STU observations must come from the original eligible pool")
    destination = Path(output)/"stu.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["reference"] != reference["sha256"] or saved["tolerances"] != DIVERSITY:
            raise ValueError("existing STU selection belongs to another specification")
        return saved
    worlds, features = stu_observations(base, workers)
    by_world = {}
    for record in base["records"]:
        by_world.setdefault(record["world"], []).append(record)
    mandatory = {(r["world"], r["frame"]) for r in reference["records"] if r["group"] == "anomaly_stu"}
    chosen, summaries = [], []
    for world, records in sorted(by_world.items()):
        records = sorted(records, key=lambda r: r["frame"])
        vectors = [features[world, r["frame"]] for r in records]
        required = [i for i, r in enumerate(records) if (world, r["frame"]) in mandatory]
        keep, nearest = select_observations(vectors, required)
        chosen.extend(dict(records[i], group="anomaly_stu", subset="train", observation=vectors[i],
                           geometry=[worlds[world]["geometry"]],
                           source_family=worlds[world]["source_family"]) for i in keep)
        summaries.append(dict(world=world, candidates=len(records), base=len(required), retained=len(keep),
            frames=[records[i]["frame"] for i in keep], omitted_max_distance=float(nearest.max())))
    result = dict(reference=reference["sha256"], source=base["sha256"], tolerances=DIVERSITY,
                  candidates=len(base["records"]), records=chosen, worlds=summaries,
                  selection="all baseline views retained; farthest-first until every omitted view is within tolerance of a retained view")
    write_json(destination, result)
    return result


def make_native_manifest(base_path, output):
    """One finite two-domain pool: native 4+4 scenes and three STU views per world."""
    output = Path(output)
    base = load_manifest(base_path, "train")
    native = load_manifest(output / "native/manifest.json", "native")
    original = load_manifest(output / "train.json", "train")
    blocked = {native["split"]["logs"][name] for name in native["split"]["check"]}
    if any(r["log_token"] in blocked or r["subset"] != "train" for r in native["records"]):
        raise ValueError("internal-check acquisition log entered native training")
    if native["mapping"] != original["mapping"] or base["sha256"] != original["base_manifest"]:
        raise ValueError("the established normal labels or STU source pool changed")
    grouped = {}
    for row in base["records"]:
        grouped.setdefault(row["world"], []).append(row)
    representatives, selection = [], []
    for world in base["worlds"]:
        rows = sorted(grouped[world["id"]], key=lambda r: r["frame"])
        saved = json.loads((Path(base["pool_root"]) / world["paths"][0] / "manifest.json").read_text())
        observed = {r["frame"]: r for r in saved["frames"]}
        features = np.asarray([[np.log1p(r["anomaly"]), observed[r["frame"]]["range"] / 10,
            np.log1p(observed[r["frame"]]["occluded"]), observed[r["frame"]]["linearity"],
            observed[r["frame"]]["planarity"]] for r in rows], dtype=float)
        # Older summaries omit shape statistics for sparse objects. Recompute
        # those descriptors from actual anomaly returns instead of inventing zeros.
        for index in np.flatnonzero(~np.isfinite(features).all(1)):
            delta = read_delta(rows[index]["delta"])
            xyz = delta["xyzi"][:, :3].astype(float)
            distance = np.linalg.norm(xyz, axis=1)
            points = xyz[((delta["packed_labels"] & 65535) == 2) & (distance >= 2.5) & (distance <= 50)]
            eig = np.maximum(np.linalg.eigvalsh(np.cov(points.T)), 0)
            features[index, 3:] = ((eig[2]-eig[1])/max(eig[2], 1e-12),
                                   (eig[1]-eig[0])/max(eig[2], 1e-12))
        if not np.isfinite(features).all():
            raise ValueError("STU observation descriptors must be measured and finite")
        scale = np.maximum(np.quantile(features, .75, axis=0) - np.quantile(features, .25, axis=0), .1)
        normalized = (features - np.median(features, axis=0)) / scale
        chosen = [int(np.argmin(np.square(normalized).sum(1)))]
        while len(chosen) < min(3, len(rows)):
            distance = np.square(normalized[:, None] - normalized[chosen]).sum(-1).min(1)
            distance[chosen] = -1
            chosen.append(int(np.argmax(distance)))
        representatives.extend(dict(rows[i], group="anomaly_stu", subset="train") for i in chosen)
        selection.append(dict(world=world["id"], frames=[rows[i]["frame"] for i in chosen],
                              descriptors=features[chosen].tolist()))
    stu_normal = [r for r in original["records"] if r.get("source") == "normal_stu"]
    if len(stu_normal) != 449 or len(representatives) != 720:
        raise ValueError("the STU supplement requires 449 original and 3 x 240 representative scans")
    records = native["records"] + representatives + stu_normal
    if any(1 <= r["anomaly"] <= 4 or r["normal"] + r["anomaly"] == 0 for r in records):
        raise ValueError("an ineligible scan entered the two-domain pool")
    result = dict(base, version=NATIVE_VERSION, records=records, mapping=native["mapping"],
        split=native["split"], native_manifest=native["sha256"], base_manifest=base["sha256"],
        selection=selection, representative_rule="median observation then farthest observations in point count, range, occlusion and local shape")
    result.pop("sha256")
    result["sha256"] = identity(result)
    write_json(output / "native/train.json", result)
    print(json.dumps(dict(samples=len(records), native=len(native["records"]), stu=len(representatives)+449)), flush=True)
    return result


def make_ndp_manifest(data_root, output):
    """Generate the public fixed Perlin sequence; store only modified raw slots."""
    from vendor.ndp.augmentation import perlin_raise, REVISION, SEED
    output = Path(output).resolve()
    if (output / "train.json").exists():
        raise FileExistsError("the NDP training manifest already exists")
    output.mkdir(parents=True, exist_ok=True)
    sequence = STUSequence(data_root)
    # Upstream uses TWO independent streams, both advanced in sorted frame order.
    parameters, noise = np.random.RandomState(SEED), np.random.default_rng(SEED)
    records, sources = [], []
    started = time.perf_counter()
    for index in range(len(sequence)):
        original = sequence[index]
        clean = point_targets(original)
        if (clean == 1).any() or not (clean == 0).any():
            raise ValueError("NDP requires unmodified normal 206 scans")
        radius, strength = 1.25 + parameters.uniform(-.5, .25), .5 + parameters.uniform(-.25, .5)
        xyzi, semantic = perlin_raise(original.xyzi.copy(), original.semantic.copy(),
            patch_radius=radius, strength=strength, rng=noise, debug=False)
        labels = (original.labels & 0xFFFF0000) | semantic.astype(np.uint32)
        changed = np.flatnonzero((xyzi != original.xyzi).any(1) | (labels != original.labels))
        frame = Frame(index, xyzi, original.pose, labels)
        target = point_targets(frame)
        source = dict(frame=index, source_identity=legacy_source_identity(original),
            scan=str(sequence.directory / "velodyne" / f"{index:06d}.bin"),
            label=str(sequence.directory / "labels" / f"{index:06d}.label"),
            normal=int((clean == 0).sum()))
        source.update(scan_sha256=file_sha256(source["scan"]), label_sha256=file_sha256(source["label"]))
        sources.append(source)
        delta = output / f"{index:06d}.npz"
        np.savez_compressed(delta, frame=index, source_identity=source["source_identity"],
                            slots=changed, xyzi=xyzi[changed], labels=labels[changed])
        records.append(dict(source="perlin", group="perlin_stu", frame=index, subset="train",
            delta=str(delta), delta_sha256=file_sha256(delta), points=int(frame.actual.sum()),
            slots=len(xyzi), normal=int((target == 0).sum()), anomaly=int((target == 1).sum()),
            changed=len(changed), radius_m=radius, strength_m=strength))
        if (index + 1) % 50 == 0 or index + 1 == len(sequence):
            print(f"Perlin 206: {index + 1}/{len(sequence)}", flush=True)
    result = dict(version=NDP_VERSION, kind="train", data_root=str(Path(data_root).resolve()),
        sources=sources, records=records,
        calibration_sha256=file_sha256(sequence.directory / "calib.txt"),
        poses_sha256=file_sha256(sequence.directory / "poses.txt"),
        recipe=dict(repository="https://github.com/343gltysprk/ndp", revision=REVISION,
            script="ood_augmentation.py", seed=SEED, sequence="206", frames=len(sequence),
            radius_m=[.75, 1.5], strength_m=[.25, 1.], target_ratio=.3, grid_res=192,
            base_res=[3, 3], octaves=3, persistence=.55, lacunarity=2.,
            raise_threshold_m=.01, dbscan_eps_m=.1, dbscan_min_samples=1,
            train_frame_filter="all 449 frames; no evaluation five-anomaly-point filter",
            normal_reference="same unmodified source scan; auxiliary likelihood only",
            augmentation=False,
            targets="unchanged model labels: raw 2 anomaly, raw 0 ignored, other labels normal, range 2.5-50 m",
            generation_seconds=time.perf_counter() - started))
    result["sha256"] = identity(result)
    write_json(output / "train.json", result)
    print(json.dumps(dict(scans=len(records), anomaly_points=sum(r["anomaly"] for r in records),
        sparse_anomaly_frames=sum(0 < r["anomaly"] < 5 for r in records),
        unchanged_frames=sum(r["changed"] == 0 for r in records), seconds=result["recipe"]["generation_seconds"])), flush=True)
    return result


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
# Coarse source labels supervise sets, never invented fine target annotations.
NUSCENES_NORMAL_SETS = {
    2: (5,), 3: (5,), 4: (5,), 6: (5,), 14: (1, 6), 15: (4,), 16: (4,),
    17: (0,), 18: (4,), 19: (0, 3, 4), 20: (0, 2, 4, 7), 21: (2, 7),
    22: (4,), 23: (3,), 24: (8, 9), 26: (10,), 27: (16,), 30: (14, 15),
}


def normal_records(source, *, development=False):
    """Only original source scans or the explicitly allowed normal STU sequences."""
    if source == "nuscenes":
        path = Path("results/data/background") / ("val.json" if development else "train.json")
        records = json.loads(path.read_text())["records"]
        if any(row.get("delta") or row.get("anomaly", 0) for row in records):
            raise ValueError("normal-only training cannot consume inserted foregrounds")
        return records
    if source not in ("206", "201") or development != (source == "201"):
        raise ValueError("normal protocol permits 206 training and 201 development only")
    directory = DATA_ROOT / "train" / source
    calibration = {}
    for line in (directory / "calib.txt").read_text().splitlines():
        if line.strip():
            key, value = line.split(":", 1)
            matrix = np.eye(4)
            matrix[:3] = np.fromstring(value, sep=" ").reshape(3, 4)
            calibration[key] = matrix
    camera = np.loadtxt(directory / "poses.txt").reshape(-1, 3, 4)
    poses = np.broadcast_to(np.eye(4), (len(camera), 4, 4)).copy()
    poses[:, :3] = camera
    transform = calibration["Tr"]
    poses = np.linalg.inv(transform) @ poses @ transform
    scans = sorted((directory / "velodyne").glob("*.bin"))
    if len(scans) != len(poses):
        raise ValueError("normal sequence poses and scans differ")
    return [dict(source="normal_stu", scene=source, frame=int(scan.stem), scan=str(scan),
                 label=str(directory / "labels" / (scan.stem + ".label")), pose=poses[i].tolist())
            for i, scan in enumerate(scans)]


def read_normal_record(record):
    source = record["source"] == "nuscenes"
    raw = np.fromfile(record["scan"], dtype="<f4").reshape(-1, 5 if source else 4)
    label = np.fromfile(record["label"], dtype=np.uint8 if source else "<u4")
    if len(raw) != len(label) or not np.isfinite(raw).all():
        raise ValueError("normal scan and point labels do not correspond")
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
    distance = np.linalg.norm(xyzi[:, :3], axis=1)
    allowed[(distance < 2.5) | (distance > 50)] = False
    return dict(xyzi=xyzi, allowed=allowed, slots=np.flatnonzero(actual), slot_count=len(raw),
                pose=np.asarray(record["pose"], dtype=np.float64))


class NormalScans:
    """Real normal labels, full input context and bounded supervised queries."""

    def __init__(self, records, *, augment=False, queries=4096):
        if queries <= 0:
            raise ValueError("normal supervision requires a positive query budget")
        self.records, self.augment, self.queries = records, augment, queries

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        from .model import voxelize
        from .normal import hypothesis_observation
        epoch, index = index if isinstance(index, tuple) else (0, index)
        raw = read_normal_record(self.records[index])
        if self.augment:
            angle = np.random.default_rng(np.random.SeedSequence([index, epoch, 613])).uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            raw["xyzi"][:, :3] = raw["xyzi"][:, :3] @ rotation.T
        result = voxelize(raw["xyzi"])
        result.update(allowed=torch.from_numpy(raw["allowed"]), slots=torch.from_numpy(raw["slots"]),
                      slot_count=raw["slot_count"], index=index,
                      observation=hypothesis_observation(raw["xyzi"]))
        valid = np.flatnonzero(raw["allowed"].any(1))
        # Fixed class-balanced query inclusion; the backbone still sees every return.
        rng = np.random.default_rng(np.random.SeedSequence([index, epoch, 7291]))
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
    """Fixed manifest reader. Metadata and truth never enter model features."""

    def __init__(self, manifest, *, cache_size=8):
        self.manifest = manifest
        self.records = manifest["records"]
        self.cache_size = cache_size
        self.cache = OrderedDict()
        if manifest["version"] == SOURCE_VERSION and any(r.get("source") != "nuscenes" for r in self.records):
            raise ValueError("source-only records must identify raw nuScenes scans")
        self.sequence = (STUSequence(manifest["data_root"])
                         if "sources" in manifest and manifest["version"] != SOURCE_VERSION else None)
        if self.sequence is not None:
            for name, expected in (("calib.txt", manifest["calibration_sha256"]),
                                   ("poses.txt", manifest["poses_sha256"])):
                if file_sha256(self.sequence.directory / name) != expected:
                    raise ValueError(f"206 coordinate metadata changed: {name}")
            self.sources = {r["frame"]: r for r in manifest["sources"]}

    def __len__(self):
        return len(self.records)

    def _source(self, frame_id):
        record = self.sources[frame_id]
        stamp = tuple((Path(record[k]).stat().st_size, Path(record[k]).stat().st_mtime_ns)
                      for k in ("scan", "label"))
        if frame_id in self.cache:
            old_stamp, frame = self.cache.pop(frame_id)
            if old_stamp != stamp:
                raise ValueError("source scan changed after being read")
        else:
            frame = self.sequence[frame_id]
            if legacy_source_identity(frame) != record["source_identity"]:
                raise ValueError("source scan differs from the fixed manifest")
        self.cache[frame_id] = stamp, frame
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return frame

    def __getitem__(self, index):
        record = self.records[index]
        if record.get("source") == "nuscenes":
            frame = read_nuscenes(record, self.manifest["mapping"])
        elif record.get("source") == "normal_stu":
            frame = self._source(record["frame"])
        elif record.get("source") in ("targeted", "rendered_stu", "perlin"):
            original = self._source(record["frame"])
            if record["source"] == "perlin" and file_sha256(record["delta"]) != record["delta_sha256"]:
                raise ValueError("Perlin modification changed after manifest creation")
            with np.load(record["delta"], allow_pickle=False) as delta:
                if (int(delta["frame"]) != original.frame_id or
                        str(delta["source_identity"]) != self.sources[original.frame_id]["source_identity"]):
                    raise ValueError("targeted observation belongs to a different original scan")
                if record["source"] == "rendered_stu" and str(delta["world"]) != record["world"]:
                    raise ValueError("rendered STU observation belongs to another fixed world")
                xyzi, labels = original.xyzi.copy(), original.labels.copy()
                xyzi[delta["slots"]], labels[delta["slots"]] = delta["xyzi"], delta["labels"]
            frame = Frame(original.frame_id, xyzi, original.pose, labels)
        elif self.sequence is not None:
            original = self._source(record["frame"])
            raw = Path(record["delta"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["delta_sha256"]:
                raise ValueError(f"saved delta changed: {record['delta']}")
            delta = read_delta(io.BytesIO(raw))
            validate_delta(delta, original, record["world"], source_identity=record["source_identity"])
            xyzi, labels = original.xyzi.copy(), original.labels.copy()
            xyzi[delta["source_slot"]] = delta["xyzi"]
            labels[delta["source_slot"]] = delta["packed_labels"]
            frame = Frame(original.frame_id, xyzi, original.pose, labels)
        else:
            frame = read_scan(record["scan"], record["label"], partition=self.manifest["kind"],
                              expected=(record["scan_sha256"], record["label_sha256"]))
        target = point_targets(frame)
        # Source training retains sparse anomalies and unmodified normal scans;
        # the official five-point rule selects evaluation frames only.
        selected = (Supervision(target, int((target == 0).sum()), int((target == 1).sum()), True)
                    if self.manifest["version"] in (NDP_VERSION, SOURCE_VERSION) else
                    supervision(frame, allow_normal=self.manifest["version"] in (PILOT_VERSION, NATIVE_VERSION)))
        if (self.manifest["kind"] == "train" and self.manifest["version"] not in (NDP_VERSION, SOURCE_VERSION)
                and not selected.eligible):
            raise ValueError("training scan is ineligible under the frame rule")
        if (self.manifest["version"] == SOURCE_VERSION and self.manifest["kind"] != "train"
                and record["eligible"] != (selected.anomaly_count >= 5)):
            raise ValueError("source development eligibility must follow the official five-point rule")
        observed = (int(frame.actual.sum()), selected.normal_count, selected.anomaly_count, len(frame.xyzi))
        expected = tuple(record[k] for k in ("points", "normal", "anomaly", "slots"))
        if observed != expected:
            raise ValueError(f"decoded counts differ from manifest: {observed} != {expected}")
        return dict(xyzi=frame.xyzi[frame.actual].copy(), slots=frame.return_slots,
                    targets=target[frame.actual].copy(),
                    slot_count=len(frame.xyzi), index=index)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare nuScenes source data or historical experiment manifests.")
    parser.add_argument("operation", choices=("normal", "expand", "mix", "native", "legacy", "ndp", "nuscenes"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nuscenes-root", type=Path, default=NUSCENES_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--pool-root", type=Path, default=POOL_ROOT)
    parser.add_argument("--train", type=Path, default=Path("assets/train.json"))
    parser.add_argument("--val", type=Path, default=Path("assets/val.json"))
    parser.add_argument("--workers", type=int, default=min(4, len(os.sched_getaffinity(0))))
    parser.add_argument("--background-only", action="store_true", help="nuScenes: split original backgrounds without extracting or inserting objects")
    parser.add_argument("--objects", type=Path, help="nuScenes: reviewed object catalog for one fixed placement per sequence")
    parser.add_argument("--normal-annotations", type=Path, help="nuScenes: reviewed native point labels; unresolved points remain ignored")
    args = parser.parse_args()
    if args.background_only and args.operation != "nuscenes":
        parser.error("--background-only is only supported by the nuscenes operation")
    if args.objects is not None and (args.operation != "nuscenes" or args.background_only):
        parser.error("--objects requires nuscenes without --background-only")
    if args.normal_annotations is not None and (args.operation != "nuscenes" or not (args.background_only or args.objects)):
        parser.error("--normal-annotations requires nuScenes backgrounds or reviewed object sequences")
    if args.operation == "nuscenes":
        if args.output is None:
            parser.error("nuscenes requires an explicit new --output directory")
        from .nuscenes import build
        build(args.nuscenes_root, args.output, args.workers,
              background_only=args.background_only, objects=args.objects, normal_annotations=args.normal_annotations)
        return
    if args.output is None:
        args.output = Path("results/data")
    if args.operation == "ndp":
        make_ndp_manifest(args.data_root, args.output)
        return
    if args.operation in ("normal", "expand"):
        if (args.output / ("background.json" if args.operation == "expand" else "normal.json")).exists():
            parser.error("normal sources already exist")
        make_normal_manifest(args.nuscenes_root, args.output, expanded=args.operation == "expand", workers=args.workers)
        return
    if args.operation == "mix":
        make_pilot_manifest(args.train, args.output)
        return
    if args.operation == "native":
        make_native_manifest(args.train, args.output)
        return
    if args.train.exists() or args.val.exists():
        parser.error("manifest already exists; fixed manifests must not be silently replaced")
    train = make_manifest(args.data_root, args.pool_root, args.workers)
    val = make_real_manifest(args.data_root / "val", workers=args.workers)
    write_json(args.train, train)
    write_json(args.val, val)
    print(json.dumps(dict(train=len(train["records"]), worlds=len(train["worlds"]),
                          train_sha256=train["sha256"], val=len(val["records"]),
                          val_eligible=sum(r["eligible"] for r in val["records"]),
                          val_sha256=val["sha256"]), indent=2))


if __name__ == "__main__":
    main()
