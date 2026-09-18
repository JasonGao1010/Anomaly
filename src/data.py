"""Read complete STU/206 observations and apply the V4 supervision rules.

File and pose conventions are taken from AJAE-v3/src/scene.py. Ray calibration
uses AJAE/assets/rays.npz and the formula already measured in the 206 analysis.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import math

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
    valid = frame.actual & (frame.range_m >= 2.5) & (frame.range_m <= 50) & (frame.semantic != 0)
    targets = np.full(len(frame.xyzi), -1, dtype=np.int8)
    targets[valid] = (frame.semantic[valid] == 2).astype(np.int8)
    return targets


@dataclass(frozen=True, slots=True)
class Supervision:
    targets: np.ndarray
    normal_count: int
    anomaly_count: int

    @property
    def eligible(self):
        return self.anomaly_count >= 5


def supervision(frame):
    """Apply the updated plan: every frame with fewer than five anomalies is skipped."""
    targets = point_targets(frame)
    normal, anomaly = int((targets == 0).sum()), int((targets == 1).sum())
    if anomaly < 5:
        targets.fill(-1)
    # No per-object threshold: 1-4 point objects stay anomalous in eligible frames.
    return Supervision(readonly(targets), normal, anomaly)


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
