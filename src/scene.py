#!/usr/bin/env python3
"""Read one STU scan with its original file-slot identity and labels."""

from __future__ import annotations

import os
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCAN_CHANNELS = 4
SCAN_DTYPE = np.dtype("<f4")
LABEL_DTYPE = np.dtype("<u4")
RIGID_ATOL = 1.0e-3
IDENTITY_ATOL = 1.0e-9
SOURCE_FRAME_CACHE_SIZE = 1
# These values reproduce the identity stored in each existing delta; they are
# storage metadata, not a supervision target for the next training design.
STORED_CLASS_MAP = {
    0: 255, 1: 255, 2: 255, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4,
    18: 3, 20: 4, 30: 5, 31: 6, 32: 7, 40: 8, 44: 9, 48: 10,
    49: 11, 50: 12, 51: 13, 52: 255, 60: 8, 70: 14, 71: 15,
    72: 16, 80: 17, 81: 18, 99: 255, 252: 0, 253: 6, 254: 5,
    255: 7, 256: 4, 257: 4, 258: 3, 259: 4,
}
SOURCE_COUNTS = {201: 682, 206: 449}
# Released train/201 copies are file-layout aliases, not a coordinate deduplication rule.
DUPLICATE_201_RAY_LAYOUT = {
    0: (0, ((0, 131072, 0), (131072, 131072, 0), (262144, 131072, 0))),
    1: (0, ((0, 131072, 0), (131072, 131072, 0), (262144, 131072, 0))),
    2: (29184, ((0, 29184, 0), (29184, 131072, 0), (160256, 131072, 0))),
    3: (0, ((0, 131072, 0), (131072, 131072, 0))),
}


class SceneDataError(ValueError):
    """Report malformed STU data or an invalid scene relation."""


def _plain_int(name: str, value: int, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{name} must be an integer >= {minimum}")
    return value


def _freeze(array: np.ndarray) -> np.ndarray:
    frozen = np.ascontiguousarray(array)
    frozen.setflags(write=False)
    return frozen


def _finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        count = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise SceneDataError(f"{name} contains {count} non-finite value(s)")


def _rigid(name: str, matrix: np.ndarray) -> None:
    if matrix.shape != (4, 4):
        raise SceneDataError(f"{name} must have shape (4, 4)")
    _finite(name, matrix)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=IDENTITY_ATOL):
        raise SceneDataError(f"{name} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        atol=RIGID_ATOL,
        rtol=RIGID_ATOL,
    ):
        raise SceneDataError(f"{name} rotation is not orthonormal")
    if not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=RIGID_ATOL, rel_tol=RIGID_ATOL
    ):
        raise SceneDataError(f"{name} rotation determinant is not +1")


@dataclass(frozen=True, slots=True)
class PointLabels:
    """Packed, raw, and optional STU-normal labels aligned with one point array."""

    packed: np.ndarray
    semantic: np.ndarray
    instance: np.ndarray
    semantic_target: np.ndarray | None = None

    def __post_init__(self) -> None:
        count = self.packed.size
        if self.packed.dtype != np.uint32 or self.packed.shape != (count,):
            raise TypeError("packed labels must be uint32[N]")
        if self.semantic.dtype != np.uint16 or self.semantic.shape != (count,):
            raise TypeError("semantic labels must be uint16[N]")
        if self.instance.dtype != np.uint16 or self.instance.shape != (count,):
            raise TypeError("instance labels must be uint16[N]")
        if not np.array_equal(
            self.semantic, (self.packed & np.uint32(0xFFFF)).astype(np.uint16)
        ):
            raise SceneDataError("semantic labels do not match packed labels")
        if not np.array_equal(
            self.instance, (self.packed >> np.uint32(16)).astype(np.uint16)
        ):
            raise SceneDataError("instance labels do not match packed labels")
        if self.semantic_target is not None:
            if self.semantic_target.dtype != np.uint8 or self.semantic_target.shape != (
                count,
            ):
                raise TypeError("semantic_target must be uint8[N]")
            valid = (self.semantic_target <= np.uint8(18)) | (
                self.semantic_target == np.uint8(255)
            )
            if not np.all(valid):
                raise SceneDataError(
                    "semantic targets must be class 0 through 18 or ignore 255"
                )
            self.semantic_target.setflags(write=False)
        self.packed.setflags(write=False)
        self.semantic.setflags(write=False)
        self.instance.setflags(write=False)


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """One complete STU scan with original file slots and read-only arrays."""

    partition: str
    sequence_id: int
    frame_id: int
    xyzi: np.ndarray
    lidar_pose: np.ndarray
    zero_slot_mask: np.ndarray = field(init=False)
    real_slots: np.ndarray = field(init=False)
    duplicate_ray_slots: np.ndarray | None = field(init=False)
    observation_slots: np.ndarray = field(init=False)
    record_inverse: np.ndarray = field(init=False)
    labels: PointLabels | None

    def __post_init__(self) -> None:
        if self.partition not in {"train", "val", "test", "fixture"}:
            raise SceneDataError("SourceFrame.partition is invalid")
        _plain_int("sequence_id", self.sequence_id)
        _plain_int("frame_id", self.frame_id)
        count = self.xyzi.shape[0]
        if self.xyzi.dtype != np.float32 or self.xyzi.shape != (count, 4):
            raise TypeError("xyzi must be float32[N,4]")
        if self.lidar_pose.dtype != np.float64:
            raise TypeError("lidar_pose must be float64[4,4]")
        _rigid("lidar_pose", self.lidar_pose)
        _finite("xyzi", self.xyzi)
        # Empty slots remain in storage; real_slots selects actual returns.
        zero = np.all(self.xyzi[:, :3] == np.float32(0.0), axis=1)
        object.__setattr__(self, "zero_slot_mask", zero)
        object.__setattr__(self, "real_slots", np.flatnonzero(~zero).astype(np.int32))
        if self.labels is not None and self.labels.packed.size != count:
            raise SceneDataError("scan and label slot counts differ")
        mapping = None
        if (self.partition == "train" and self.sequence_id == 201
                and self.frame_id in DUPLICATE_201_RAY_LAYOUT and count > 131072):
            start, runs = DUPLICATE_201_RAY_LAYOUT[self.frame_id]
            if sum(length for _, length, _ in runs) != count:
                raise SceneDataError("201 duplicate-block slot count differs from its known layout")
            mapping = np.concatenate([np.arange(ray, ray + length, dtype=np.int32)
                                      for _, length, ray in runs])
        object.__setattr__(self, "duplicate_ray_slots", mapping)
        self.validate_duplicate_values(self.xyzi, "XYZI")
        if self.labels is not None:
            self.validate_duplicate_values(self.labels.packed, "packed labels")
            if self.labels.semantic_target is not None:
                self.validate_duplicate_values(self.labels.semantic_target, "semantic targets")
        if mapping is None:
            slots, inverse = self.real_slots, np.arange(self.real_count, dtype=np.int64)
        else:
            # Frame 2 starts with a partial copy; always select its following complete block.
            rays = np.flatnonzero(~zero[start:start + 131072]).astype(np.int32)
            slots = start + rays
            inverse = np.searchsorted(rays, mapping[self.real_slots])
            mapping.setflags(write=False)
        object.__setattr__(self, "observation_slots", _freeze(slots))
        object.__setattr__(self, "record_inverse", _freeze(inverse))
        for array in (
            self.xyzi,
            self.lidar_pose,
            self.zero_slot_mask,
            self.real_slots,
        ):
            array.setflags(write=False)

    @property
    def slot_count(self) -> int:
        return int(self.xyzi.shape[0])

    @property
    def real_count(self) -> int:
        return int(self.real_slots.size)

    def validate_duplicate_values(self, values: np.ndarray, name: str) -> None:
        """Check actual raw or rendered contents before sharing any observation."""
        if self.duplicate_ray_slots is not None:
            start = DUPLICATE_201_RAY_LAYOUT[self.frame_id][0]
            if not np.array_equal(values, values[start:start + 131072][self.duplicate_ray_slots]):
                raise SceneDataError(f"201 duplicate-block {name} differ from the complete block")


def make_source_frame(
    frame_id: int,
    xyzi: np.ndarray,
    lidar_pose: np.ndarray,
    labels: PointLabels | None = None,
    *,
    partition: str,
    sequence_id: int,
) -> SourceFrame:
    """Build one original or deterministically rendered STU frame."""

    frame = _plain_int("frame_id", frame_id)
    array = np.asarray(xyzi)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 4:
        raise TypeError("xyzi must be float32[N,4]")
    pose = np.asarray(lidar_pose)
    if pose.dtype != np.float64:
        raise TypeError("lidar_pose must be float64[4,4]")
    _rigid("lidar_pose", pose)
    owned = _freeze(array.copy())
    return SourceFrame(
        partition=partition,
        sequence_id=_plain_int("sequence_id", sequence_id),
        frame_id=frame,
        xyzi=owned,
        lidar_pose=_freeze(pose.copy()),
        labels=labels,
    )


def _matrix(values: Sequence[float], name: str) -> np.ndarray:
    if len(values) not in {12, 16}:
        raise SceneDataError(f"{name} must contain 12 or 16 numbers")
    matrix = np.eye(4, dtype=np.float64)
    if len(values) == 12:
        matrix[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    else:
        matrix[:] = np.asarray(values, dtype=np.float64).reshape(4, 4)
    _rigid(name, matrix)
    return matrix


def read_calibration(path: Path) -> Mapping[str, np.ndarray]:
    """Read all KITTI-style calibration matrices."""

    resolved = path.expanduser().resolve(strict=True)
    result: dict[str, np.ndarray] = {}
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                raise SceneDataError(f"invalid calibration line {line_number}")
            key, text = line.split(":", 1)
            key = key.strip()
            if not key or key in result:
                raise SceneDataError(f"invalid calibration key on line {line_number}")
            try:
                values = [float(item) for item in text.split()]
            except ValueError as error:
                raise SceneDataError(
                    f"non-numeric calibration value on line {line_number}"
                ) from error
            result[key] = _freeze(_matrix(values, f"calibration {key}"))
    if "Tr" not in result:
        raise SceneDataError("calibration must contain Tr")
    return result


def read_poses(path: Path) -> np.ndarray:
    """Read KITTI-style camera poses before LiDAR calibration."""

    resolved = path.expanduser().resolve(strict=True)
    poses: list[np.ndarray] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                values = [float(item) for item in line.split()]
            except ValueError as error:
                raise SceneDataError(
                    f"non-numeric pose value on line {line_number}"
                ) from error
            poses.append(_matrix(values, f"pose {line_number - 1}"))
    if not poses:
        raise SceneDataError("pose file is empty")
    return _freeze(np.stack(poses, axis=0))


def _indexed_files(directory: Path, suffix: str) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    indexed: dict[int, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        if not path.stem.isdigit():
            raise SceneDataError(f"file name must be numeric: {path.name}")
        frame = int(path.stem)
        if frame in indexed:
            raise SceneDataError(f"duplicate numeric frame id {frame}")
        indexed[frame] = path
    if not indexed:
        raise SceneDataError(f"no {suffix} files in {directory}")
    if sorted(indexed) != list(range(len(indexed))):
        raise SceneDataError(f"{directory} frame ids must be contiguous from zero")
    return indexed


class STUSequence:
    """Read either source sequence used by the existing synthetic samples."""

    def __init__(self, data_root: Path | str, sequence_id: int) -> None:
        self.sequence_id = _plain_int("sequence_id", sequence_id)
        if sequence_id not in SOURCE_COUNTS:
            raise SceneDataError("synthetic sources must be train/201 or train/206")
        self.sequence_dir = (
            Path(data_root).expanduser().resolve(strict=True) / "train" / str(sequence_id)
        )
        if not self.sequence_dir.is_dir():
            raise NotADirectoryError(self.sequence_dir)

        self._scan_paths = _indexed_files(self.sequence_dir / "velodyne", ".bin")
        self.frame_count = len(self._scan_paths)
        self.frame_ids = tuple(range(self.frame_count))
        if self.frame_count != SOURCE_COUNTS[sequence_id]:
            raise SceneDataError("source scan count differs from the saved sample set")

        calibration = read_calibration(self.sequence_dir / "calib.txt")
        camera_poses = read_poses(self.sequence_dir / "poses.txt")
        if camera_poses.shape[0] != self.frame_count:
            raise SceneDataError("pose count does not match scan count")
        # Preserve the calibration order used to construct the saved identities.
        lidar_from_camera = np.linalg.inv(calibration["Tr"])
        lidar_poses = np.stack(
            [lidar_from_camera @ pose @ calibration["Tr"] for pose in camera_poses]
        )
        for frame, pose in enumerate(lidar_poses):
            _rigid(f"LiDAR pose {frame}", pose)
        self._lidar_poses = _freeze(lidar_poses.astype(np.float64, copy=False))

        self._label_paths = _indexed_files(self.sequence_dir / "labels", ".label")
        if sorted(self._label_paths) != list(self.frame_ids):
            raise SceneDataError("labels must cover every scan")
        self._semantic_target_lut = np.full(1 << 16, -1, dtype=np.int16)
        for raw, target in STORED_CLASS_MAP.items():
            self._semantic_target_lut[raw] = target
        self._semantic_target_lut.setflags(write=False)
        self._frames: OrderedDict[int, SourceFrame] = OrderedDict()
        self._cache_frames = SOURCE_FRAME_CACHE_SIZE


    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, frame_id: int) -> SourceFrame:
        return self.source_frame(frame_id)


    def source_frame(self, frame_id: int) -> SourceFrame:
        frame = _plain_int("frame_id", frame_id)
        if frame >= self.frame_count:
            raise IndexError(frame)
        cached = self._frames.pop(frame, None)
        if cached is not None:
            self._frames[frame] = cached
            return cached
        path = self._scan_paths[frame]
        record_bytes = SCAN_CHANNELS * SCAN_DTYPE.itemsize
        if path.stat().st_size <= 0 or path.stat().st_size % record_bytes:
            raise SceneDataError(f"invalid scan byte length: {path}")
        with path.open("rb") as stream:
            raw = np.fromfile(stream, dtype=SCAN_DTYPE)
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if raw.size % SCAN_CHANNELS:
            raise SceneDataError(f"scan cannot be reshaped to N x 4: {path}")
        xyzi = raw.reshape(-1, SCAN_CHANNELS).astype(np.float32, copy=False)
        _finite(f"scan {frame}", xyzi)
        result = make_source_frame(
            frame,
            xyzi,
            self._lidar_poses[frame],
            self._read_labels(frame, xyzi.shape[0]),
            partition="train",
            sequence_id=self.sequence_id,
        )
        self._frames[frame] = result
        while len(self._frames) > self._cache_frames:
            self._frames.popitem(last=False)
        return result

    def _read_labels(self, frame: int, slot_count: int) -> PointLabels:
        path = self._label_paths[frame]
        if path.stat().st_size <= 0 or path.stat().st_size % LABEL_DTYPE.itemsize:
            raise SceneDataError(f"invalid label byte length: {path}")
        with path.open("rb") as stream:
            packed = np.fromfile(stream, dtype=LABEL_DTYPE).astype(
                np.uint32, copy=False
            )
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if packed.size != slot_count:
            raise SceneDataError(
                f"frame {frame} has {slot_count} scan slots but {packed.size} labels"
            )
        semantic = (packed & np.uint32(0xFFFF)).astype(np.uint16, copy=False)
        instance = (packed >> np.uint32(16)).astype(np.uint16, copy=False)
        mapped = self._semantic_target_lut[semantic]
        if np.any(mapped < 0):
            unknown = sorted(map(int, np.unique(semantic[mapped < 0])))
            raise SceneDataError(f"normal frame {frame} has unmapped labels {unknown}")
        return PointLabels(
            packed=_freeze(packed),
            semantic=_freeze(semantic),
            instance=_freeze(instance),
            semantic_target=_freeze(mapped.astype(np.uint8)),
        )
