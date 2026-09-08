#!/usr/bin/env python3
"""Read one STU scan with its original file-slot identity and labels."""

from __future__ import annotations

import argparse
import json
import os
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from .protocol import STUProtocol, SequenceSpec, load_protocol


SCAN_CHANNELS = 4
SCAN_DTYPE = np.dtype("<f4")
LABEL_DTYPE = np.dtype("<u4")
RIGID_ATOL = 1.0e-3
IDENTITY_ATOL = 1.0e-9
SOURCE_FRAME_CACHE_SIZE = 1
ANOMALY_IGNORE = np.int8(-1)
ANOMALY_NORMAL = np.int8(0)
ANOMALY_POSITIVE = np.int8(1)


class SceneDataError(ValueError):
    """Report malformed STU data or an invalid scene relation."""


class LabelMode(str, Enum):
    """Choose whether a caller is allowed to read labels."""

    REQUIRED = "required"
    FORBIDDEN = "forbidden"


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


def official_stu_coordinates(xyzi: np.ndarray, lidar_pose: np.ndarray) -> np.ndarray:
    """Reproduce STU's released pre-voxel coordinate formula exactly."""

    array = np.asarray(xyzi)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 4:
        raise TypeError("xyzi must be float32[N,4]")
    if array.shape[0] == 0:
        raise SceneDataError("a scan must contain at least one file slot")
    _finite("xyzi", array)
    pose = np.asarray(lidar_pose)
    if pose.dtype != np.float64:
        raise TypeError("lidar_pose must be float64[4,4]")
    _rigid("lidar_pose", pose)

    # STU transposes its stored standard pose before this row-vector operation.
    # Keeping T_W<-S standard here gives the equivalent R p + t transform.
    coordinates = array[:, :3] @ pose[:3, :3].T + pose[:3, 3]
    return _freeze(coordinates.astype(np.float64, copy=False))


def official_stu_features(xyzi: np.ndarray, lidar_pose: np.ndarray) -> np.ndarray:
    """Compute STU's intensity and scan-centred distance input channels."""

    array = np.asarray(xyzi)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 4:
        raise TypeError("xyzi must be float32[N,4]")
    coordinates = official_stu_coordinates(array, lidar_pose)
    return _features_from_coordinates(array, coordinates)


def _features_from_coordinates(array, coordinates):
    """Reuse the exact world coordinates when constructing an immutable scan."""
    center = coordinates.mean(axis=0)
    distance = np.linalg.norm(coordinates - center, axis=1)[:, None]
    return _freeze(np.hstack((array[:, 3:4], distance)).astype(np.float32, copy=False))


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

    @property
    def group_key(self) -> np.ndarray:
        return self.packed

    @property
    def anomaly(self) -> np.ndarray:
        result = self.semantic == np.uint16(2)
        result.setflags(write=False)
        return result

    @property
    def binary_valid(self) -> np.ndarray:
        result = self.semantic != np.uint16(0)
        result.setflags(write=False)
        return result

    @property
    def anomaly_target(self) -> np.ndarray:
        """Return the three-state target: ignore=-1, normal=0, anomaly=1."""

        result = np.full(self.semantic.shape, ANOMALY_NORMAL, dtype=np.int8)
        result[self.semantic == np.uint16(0)] = ANOMALY_IGNORE
        result[self.semantic == np.uint16(2)] = ANOMALY_POSITIVE
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """One complete STU file-slot scan and its frozen official model inputs."""

    partition: str
    sequence_id: int
    frame_id: int
    xyzi: np.ndarray
    lidar_pose: np.ndarray
    coordinates: np.ndarray = field(init=False)
    features: np.ndarray = field(init=False)
    zero_slot_mask: np.ndarray = field(init=False)
    real_slots: np.ndarray = field(init=False)
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
        # Derive arrays once from the validated source; callers cannot supply conflicting copies.
        object.__setattr__(
            self, "coordinates", official_stu_coordinates(self.xyzi, self.lidar_pose)
        )
        object.__setattr__(
            self, "features", _features_from_coordinates(self.xyzi, self.coordinates)
        )
        zero = np.all(self.xyzi[:, :3] == np.float32(0.0), axis=1)
        object.__setattr__(self, "zero_slot_mask", zero)
        object.__setattr__(self, "real_slots", np.flatnonzero(~zero).astype(np.int32))
        _finite("official STU coordinates", self.coordinates)
        _finite("official STU features", self.features)
        if self.labels is not None and self.labels.packed.size != count:
            raise SceneDataError("scan and label slot counts differ")
        for array in (
            self.xyzi,
            self.lidar_pose,
            self.coordinates,
            self.features,
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

    def restore_real(self, values: np.ndarray) -> np.ndarray:
        """Restore visible-return values to this frame's complete file-slot order."""

        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] != self.real_count:
            raise ValueError(
                f"values must have leading size {self.real_count}, got {array.shape}"
            )
        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError("values must use a numeric or boolean dtype")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("values must be finite")
        output = np.zeros((self.slot_count, *array.shape[1:]), dtype=array.dtype)
        output[self.real_slots] = array
        return _freeze(output)


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


def locate_sequence(
    data_root: Path | str,
    partition: str,
    sequence_id: int,
    *,
    protocol: STUProtocol,
) -> Path:
    """Resolve one protocol sequence without searching alternative layouts."""

    if partition not in {"train", "val", "test"}:
        raise ValueError("partition must be train, val, or test")
    identifier = _plain_int("sequence_id", sequence_id)
    protocol.sequence(partition, identifier)
    path = (
        Path(data_root).expanduser().resolve(strict=True) / partition / str(identifier)
    )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path.resolve()


class STUSequence:
    """Read one protocol-assigned STU sequence with a bounded source-frame cache."""

    def __init__(
        self,
        sequence_dir: Path | str,
        *,
        protocol: STUProtocol,
        spec: SequenceSpec,
        label_mode: LabelMode | str,
    ) -> None:
        if not isinstance(protocol, STUProtocol):
            raise TypeError("protocol must be STUProtocol")
        if not isinstance(spec, SequenceSpec):
            raise TypeError("spec must be SequenceSpec")
        if protocol.sequence(spec.partition, spec.sequence_id) != spec:
            raise SceneDataError("sequence spec is not part of this protocol")
        self.protocol = protocol
        self.sequence_dir = Path(sequence_dir).expanduser().resolve(strict=True)
        if not self.sequence_dir.is_dir():
            raise NotADirectoryError(self.sequence_dir)
        if (
            self.sequence_dir.name != str(spec.sequence_id)
            or self.sequence_dir.parent.name != spec.partition
        ):
            raise SceneDataError("sequence directory does not match protocol identity")
        self.label_mode = LabelMode(label_mode)
        if self.label_mode is LabelMode.REQUIRED and not spec.labels_available:
            raise SceneDataError("labels are unavailable for this protocol role")

        self._scan_paths = _indexed_files(self.sequence_dir / "velodyne", ".bin")
        self.frame_count = len(self._scan_paths)
        self.frame_ids = tuple(range(self.frame_count))
        # Public sequence lengths come from the released scan inventory.
        self.spec = spec.with_observed_frame_count(self.frame_count)

        calibration = read_calibration(self.sequence_dir / "calib.txt")
        camera_poses = read_poses(self.sequence_dir / "poses.txt")
        if camera_poses.shape[0] != self.frame_count:
            raise SceneDataError("pose count does not match scan count")
        lidar_from_camera = np.linalg.inv(calibration["Tr"])
        lidar_poses = np.stack(
            [lidar_from_camera @ pose @ calibration["Tr"] for pose in camera_poses]
        )
        for frame, pose in enumerate(lidar_poses):
            _rigid(f"LiDAR pose {frame}", pose)
        self._lidar_poses = _freeze(lidar_poses.astype(np.float64, copy=False))

        self._label_paths: dict[int, Path] | None = None
        if self.label_mode is LabelMode.REQUIRED:
            paths = _indexed_files(self.sequence_dir / "labels", ".label")
            if sorted(paths) != list(self.frame_ids):
                raise SceneDataError("labels must cover every scan")
            self._label_paths = paths
        self._semantic_target_lut = np.full(1 << 16, -1, dtype=np.int16)
        for raw, target in protocol.semantic_class_map.items():
            self._semantic_target_lut[raw] = target
        self._semantic_target_lut.setflags(write=False)
        self._frames: OrderedDict[int, SourceFrame] = OrderedDict()
        self._cache_frames = SOURCE_FRAME_CACHE_SIZE

    @classmethod
    def open(
        cls,
        data_root: Path | str,
        *,
        protocol: STUProtocol,
        partition: str,
        sequence_id: int,
        label_mode: LabelMode | str,
    ) -> "STUSequence":
        spec = protocol.sequence(partition, sequence_id)
        return cls(
            locate_sequence(
                data_root,
                partition,
                sequence_id,
                protocol=protocol,
            ),
            protocol=protocol,
            spec=spec,
            label_mode=label_mode,
        )

    @property
    def labels_available(self) -> bool:
        return self._label_paths is not None

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, frame_id: int) -> SourceFrame:
        return self.source_frame(frame_id)

    def __iter__(self) -> Iterator[SourceFrame]:
        for frame_id in self.frame_ids:
            yield self.source_frame(frame_id)

    def lidar_pose(self, frame_id: int) -> np.ndarray:
        frame = _plain_int("frame_id", frame_id)
        if frame >= self.frame_count:
            raise IndexError(frame)
        return self._lidar_poses[frame]

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
            partition=self.spec.partition,
            sequence_id=self.spec.sequence_id,
        )
        self._frames[frame] = result
        while len(self._frames) > self._cache_frames:
            self._frames.popitem(last=False)
        return result

    def _read_labels(self, frame: int, slot_count: int) -> PointLabels | None:
        if self._label_paths is None:
            return None
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
        semantic_target: np.ndarray | None = None
        if self.spec.partition == "train":
            mapped = self._semantic_target_lut[semantic]
            if np.any(mapped < 0):
                unknown = sorted(map(int, np.unique(semantic[mapped < 0])))
                raise SceneDataError(
                    f"normal frame {frame} has unmapped labels {unknown}"
                )
            semantic_target = mapped.astype(np.uint8)
        return PointLabels(
            packed=_freeze(packed),
            semantic=_freeze(semantic),
            instance=_freeze(instance),
            semantic_target=None
            if semantic_target is None
            else _freeze(semantic_target),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val"), required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--labels", choices=tuple(mode.value for mode in LabelMode), default="forbidden"
    )
    args = parser.parse_args()
    protocol = (
        load_protocol() if args.protocol is None else load_protocol(args.protocol)
    )
    sequence = STUSequence.open(
        args.data_root,
        protocol=protocol,
        partition=args.partition,
        sequence_id=args.sequence,
        label_mode=args.labels,
    )
    frame = sequence.source_frame(args.frame)
    print(
        json.dumps(
            {
                "partition": frame.partition,
                "sequence": frame.sequence_id,
                "frame": frame.frame_id,
                "sequence_frames": len(sequence),
                "file_slots": frame.slot_count,
                "real_returns": frame.real_count,
                "zero_coordinate_slots": frame.slot_count - frame.real_count,
                "labels_read": frame.labels is not None,
                "input": "current scan xyzi in sensor coordinates",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
