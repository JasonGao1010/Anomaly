#!/usr/bin/env python3
"""Read STU scans and build AJAE's causal per-frame input windows.

Each scan keeps its original returns and reproduces the released STU
pre-voxel coordinates and two input features. Ego poses also describe where
the same returns lie in the current sensor coordinate system. The model can
therefore encode every frame exactly as STU did before feature-level alignment.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

try:
    from .protocol import (
        AJAEProtocol,
        FrameSpan,
        SequenceSpec,
        WINDOW_FRAMES,
        load_protocol,
    )
except ImportError:  # Direct script execution.
    from protocol import (
        AJAEProtocol,
        FrameSpan,
        SequenceSpec,
        WINDOW_FRAMES,
        load_protocol,
    )


SCAN_CHANNELS = 4
SCAN_DTYPE = np.dtype("<f4")
LABEL_DTYPE = np.dtype("<u4")
RIGID_ATOL = 1.0e-3
IDENTITY_ATOL = 1.0e-9


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
    """Reproduce the released STU coordinates before sparse quantization."""

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

    coordinates = array[:, :3] @ pose[:3, :3] + pose[3, :3]
    return _freeze(coordinates.astype(np.float64, copy=False))


def official_stu_features(xyzi: np.ndarray, lidar_pose: np.ndarray) -> np.ndarray:
    """Compute STU's intensity and scan-centred distance input channels."""

    array = np.asarray(xyzi)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 4:
        raise TypeError("xyzi must be float32[N,4]")
    coordinates = official_stu_coordinates(array, lidar_pose)
    center = coordinates.mean(axis=0)
    distance = np.linalg.norm(coordinates - center, axis=1)[:, None]
    return _freeze(np.hstack((array[:, 3:4], distance)).astype(np.float32, copy=False))


@dataclass(frozen=True, slots=True)
class PointLabels:
    """Labels aligned with one point array and isolated from model features."""

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
        """Return the complete raw semantic-instance identity."""

        return self.packed

    @property
    def anomaly(self) -> np.ndarray:
        result = self.semantic == np.uint16(2)
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """One complete STU scan with raw returns and official model inputs."""

    frame_id: int
    xyzi: np.ndarray
    lidar_pose: np.ndarray
    coordinates: np.ndarray
    features: np.ndarray
    zero_slot_mask: np.ndarray
    real_slots: np.ndarray
    labels: PointLabels | None

    def __post_init__(self) -> None:
        _plain_int("frame_id", self.frame_id)
        count = self.xyzi.shape[0]
        if self.xyzi.dtype != np.float32 or self.xyzi.shape != (count, 4):
            raise TypeError("xyzi must be float32[N,4]")
        if self.lidar_pose.dtype != np.float64:
            raise TypeError("lidar_pose must be float64[4,4]")
        _rigid("lidar_pose", self.lidar_pose)
        if self.coordinates.dtype != np.float64 or self.coordinates.shape != (count, 3):
            raise TypeError("coordinates must be float64[N,3]")
        if self.features.dtype != np.float32 or self.features.shape != (count, 2):
            raise TypeError("features must be float32[N,2]")
        if self.zero_slot_mask.dtype != np.bool_ or self.zero_slot_mask.shape != (
            count,
        ):
            raise TypeError("zero_slot_mask must be bool[N]")
        if self.real_slots.dtype != np.int32 or self.real_slots.ndim != 1:
            raise TypeError("real_slots must be int32[M]")
        _finite("xyzi", self.xyzi)
        _finite("official STU coordinates", self.coordinates)
        _finite("official STU features", self.features)
        if not np.array_equal(
            self.coordinates, official_stu_coordinates(self.xyzi, self.lidar_pose)
        ):
            raise SceneDataError("coordinates differ from STU's official definition")
        if not np.array_equal(self.features[:, 0], self.xyzi[:, 3]):
            raise SceneDataError("the first STU feature must be raw intensity")
        if not np.array_equal(
            self.features, official_stu_features(self.xyzi, self.lidar_pose)
        ):
            raise SceneDataError("features differ from STU's official definition")
        expected_zero = np.all(self.xyzi[:, :3] == np.float32(0.0), axis=1)
        if not np.array_equal(self.zero_slot_mask, expected_zero):
            raise SceneDataError("zero-slot mask does not match raw coordinates")
        expected_real = np.flatnonzero(~expected_zero).astype(np.int32)
        if not np.array_equal(self.real_slots, expected_real):
            raise SceneDataError("real slots do not complement zero-coordinate slots")
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
        """Restore values for real returns to this frame's original file slots."""

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
) -> SourceFrame:
    """Build one real or counterfactual frame with the sole STU input formula."""

    frame = _plain_int("frame_id", frame_id)
    array = np.asarray(xyzi)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 4:
        raise TypeError("xyzi must be float32[N,4]")
    pose = np.asarray(lidar_pose)
    if pose.dtype != np.float64:
        raise TypeError("lidar_pose must be float64[4,4]")
    _rigid("lidar_pose", pose)
    owned = _freeze(array.copy())
    coordinates = official_stu_coordinates(owned, pose)
    zero_mask = _freeze(np.all(owned[:, :3] == np.float32(0.0), axis=1))
    real_slots = _freeze(np.flatnonzero(~zero_mask).astype(np.int32, copy=False))
    return SourceFrame(
        frame_id=frame,
        xyzi=owned,
        lidar_pose=_freeze(pose.copy()),
        coordinates=coordinates,
        features=official_stu_features(owned, pose),
        zero_slot_mask=zero_mask,
        real_slots=real_slots,
        labels=labels,
    )


@dataclass(frozen=True, slots=True)
class WindowFrame:
    """One source frame plus its age and rigid relation to the current frame."""

    source: SourceFrame
    age: int
    source_to_current: np.ndarray

    def __post_init__(self) -> None:
        _plain_int("frame age", self.age)
        if self.age >= WINDOW_FRAMES:
            raise SceneDataError("frame age exceeds the causal window")
        if self.source_to_current.dtype != np.float64:
            raise TypeError("source_to_current must be float64[4,4]")
        _rigid("source_to_current", self.source_to_current)
        self.source_to_current.setflags(write=False)


@dataclass(frozen=True, slots=True)
class WindowMembers:
    """Real-return identities and geometry used after per-frame encoding."""

    coordinates_current: np.ndarray
    source_frame: np.ndarray
    source_slot: np.ndarray
    frame_age: np.ndarray
    frame_offsets: np.ndarray

    def __post_init__(self) -> None:
        count = self.coordinates_current.shape[0]
        if (
            self.coordinates_current.dtype != np.float32
            or self.coordinates_current.shape != (count, 3)
        ):
            raise TypeError("coordinates_current must be float32[M,3]")
        for name, array, dtype in (
            ("source_frame", self.source_frame, np.int32),
            ("source_slot", self.source_slot, np.int32),
            ("frame_age", self.frame_age, np.uint8),
        ):
            if array.dtype != dtype or array.shape != (count,):
                raise TypeError(f"{name} must be {np.dtype(dtype).name}[M]")
        if self.frame_offsets.dtype != np.int64 or self.frame_offsets.ndim != 1:
            raise TypeError("frame_offsets must be int64[F+1]")
        if self.frame_offsets.size < 2 or int(self.frame_offsets[0]) != 0:
            raise SceneDataError("frame offsets must begin at zero")
        if np.any(np.diff(self.frame_offsets) < 0):
            raise SceneDataError("frame offsets must be nondecreasing")
        if int(self.frame_offsets[-1]) != count:
            raise SceneDataError("last frame offset must equal member count")
        _finite("current-frame member coordinates", self.coordinates_current)
        for array in (
            self.coordinates_current,
            self.source_frame,
            self.source_slot,
            self.frame_age,
            self.frame_offsets,
        ):
            array.setflags(write=False)

    @property
    def count(self) -> int:
        return int(self.coordinates_current.shape[0])

    @property
    def current_slice(self) -> slice:
        return slice(int(self.frame_offsets[-2]), int(self.frame_offsets[-1]))

    def frame_slice(self, local_frame: int) -> slice:
        index = _plain_int("local_frame", local_frame)
        if index + 1 >= self.frame_offsets.size:
            raise IndexError(index)
        return slice(int(self.frame_offsets[index]), int(self.frame_offsets[index + 1]))


@dataclass(frozen=True, slots=True)
class SceneWindow:
    """The causal scene input for one current-frame AJAE prediction."""

    spec: SequenceSpec
    current_frame: int
    frames: tuple[WindowFrame, ...]
    members: WindowMembers
    labels: PointLabels | None

    def __post_init__(self) -> None:
        _plain_int("current_frame", self.current_frame)
        if not 1 <= len(self.frames) <= WINDOW_FRAMES:
            raise SceneDataError("a causal window must contain one through five frames")
        frame_ids = tuple(item.source.frame_id for item in self.frames)
        if frame_ids[-1] != self.current_frame:
            raise SceneDataError("the last window frame must be current")
        if frame_ids != tuple(range(frame_ids[0], self.current_frame + 1)):
            raise SceneDataError("window frames must be consecutive")
        expected_ages = tuple(self.current_frame - frame for frame in frame_ids)
        if tuple(item.age for item in self.frames) != expected_ages:
            raise SceneDataError("frame ages do not match source frames")
        if not np.allclose(
            self.frames[-1].source_to_current,
            np.eye(4, dtype=np.float64),
            atol=IDENTITY_ATOL,
            rtol=0.0,
        ):
            raise SceneDataError("the current-frame transform must be identity")
        if self.members.frame_offsets.size != len(self.frames) + 1:
            raise SceneDataError("member offsets do not match window frames")
        if self.labels is not None and self.labels.packed.size != self.members.count:
            raise SceneDataError("window labels do not match real members")

    @property
    def current(self) -> SourceFrame:
        return self.frames[-1].source

    def restore_current(self, values: np.ndarray) -> np.ndarray:
        """Restore current real-return predictions to official file order."""

        return self.current.restore_real(values)


def assemble_window(
    spec: SequenceSpec,
    current_frame: int,
    sources: Sequence[SourceFrame],
) -> SceneWindow:
    """Assemble real or rendered source scans with one shared point-order rule."""

    if not isinstance(spec, SequenceSpec):
        raise TypeError("spec must be SequenceSpec")
    current = _plain_int("current_frame", current_frame)
    frames_source = tuple(sources)
    if not frames_source or len(frames_source) > WINDOW_FRAMES:
        raise SceneDataError("sources must contain one through five scans")
    frame_ids = tuple(source.frame_id for source in frames_source)
    if frame_ids[-1] != current:
        raise SceneDataError("the last source scan must be current")
    if frame_ids != tuple(range(frame_ids[0], current + 1)):
        raise SceneDataError("source scans must be consecutive")
    labels_present = tuple(source.labels is not None for source in frames_source)
    if len(set(labels_present)) != 1:
        raise SceneDataError("all source scans must have the same label availability")

    current_pose = frames_source[-1].lidar_pose
    frames: list[WindowFrame] = []
    coordinates: list[np.ndarray] = []
    source_frames: list[np.ndarray] = []
    source_slots: list[np.ndarray] = []
    frame_ages: list[np.ndarray] = []
    offsets = [0]
    packed: list[np.ndarray] = []
    semantic: list[np.ndarray] = []
    instance: list[np.ndarray] = []
    semantic_targets: list[np.ndarray] = []

    for source in frames_source:
        transform = np.linalg.solve(current_pose, source.lidar_pose)
        _rigid(f"relative pose {current}<-{source.frame_id}", transform)
        age = current - source.frame_id
        frames.append(
            WindowFrame(
                source=source,
                age=age,
                source_to_current=_freeze(transform.astype(np.float64, copy=False)),
            )
        )
        slots = source.real_slots
        xyz = source.xyzi[slots, :3].astype(np.float64, copy=False)
        aligned = xyz @ transform[:3, :3].T + transform[:3, 3]
        coordinates.append(aligned.astype(np.float32))
        source_frames.append(np.full(slots.size, source.frame_id, dtype=np.int32))
        source_slots.append(slots.copy())
        frame_ages.append(np.full(slots.size, age, dtype=np.uint8))
        offsets.append(offsets[-1] + int(slots.size))
        if source.labels is not None:
            packed.append(source.labels.packed[slots])
            semantic.append(source.labels.semantic[slots])
            instance.append(source.labels.instance[slots])
            if source.labels.semantic_target is not None:
                semantic_targets.append(source.labels.semantic_target[slots])

    members = WindowMembers(
        coordinates_current=_freeze(np.concatenate(coordinates, axis=0)),
        source_frame=_freeze(np.concatenate(source_frames)),
        source_slot=_freeze(np.concatenate(source_slots)),
        frame_age=_freeze(np.concatenate(frame_ages)),
        frame_offsets=_freeze(np.asarray(offsets, dtype=np.int64)),
    )
    labels: PointLabels | None = None
    if labels_present[0]:
        labels = PointLabels(
            packed=_freeze(np.concatenate(packed)),
            semantic=_freeze(np.concatenate(semantic)),
            instance=_freeze(np.concatenate(instance)),
            semantic_target=(
                _freeze(np.concatenate(semantic_targets)) if semantic_targets else None
            ),
        )
    return SceneWindow(
        spec=spec,
        current_frame=current,
        frames=tuple(frames),
        members=members,
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


def locate_sequence(data_root: Path | str, partition: str, sequence_id: int) -> Path:
    """Resolve one protocol sequence without searching alternative layouts."""

    if partition not in {"train", "val", "test"}:
        raise ValueError("partition must be train, val, or test")
    identifier = _plain_int("sequence_id", sequence_id)
    path = (
        Path(data_root).expanduser().resolve(strict=True) / partition / str(identifier)
    )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path.resolve()


class STUSequence:
    """Read one complete protocol-assigned STU sequence."""

    def __init__(
        self,
        sequence_dir: Path | str,
        *,
        protocol: AJAEProtocol,
        spec: SequenceSpec,
        label_mode: LabelMode | str,
    ) -> None:
        if not isinstance(protocol, AJAEProtocol):
            raise TypeError("protocol must be AJAEProtocol")
        if not isinstance(spec, SequenceSpec):
            raise TypeError("spec must be SequenceSpec")
        if protocol.sequence(spec.partition, spec.sequence_id) != spec:
            raise SceneDataError("sequence spec is not part of this protocol")
        self.protocol = protocol
        self.spec = spec
        self.sequence_dir = Path(sequence_dir).expanduser().resolve(strict=True)
        if not self.sequence_dir.is_dir():
            raise NotADirectoryError(self.sequence_dir)
        if self.sequence_dir.name != str(spec.sequence_id):
            raise SceneDataError("sequence directory does not match sequence identity")
        if self.sequence_dir.parent.name != spec.partition:
            raise SceneDataError("sequence directory does not match partition")

        self.label_mode = LabelMode(label_mode)
        if self.label_mode is LabelMode.REQUIRED and not spec.labels_available:
            raise SceneDataError("labels are unavailable for this protocol role")

        self._scan_paths = _indexed_files(self.sequence_dir / "velodyne", ".bin")
        self.frame_count = len(self._scan_paths)
        if spec.span is not None and spec.span != FrameSpan(0, self.frame_count):
            raise SceneDataError(
                f"{spec.partition}/{spec.sequence_id} has {self.frame_count} scans, "
                f"expected {spec.frames}"
            )
        self.span = FrameSpan(0, self.frame_count)

        calibration = read_calibration(self.sequence_dir / "calib.txt")
        camera_poses = read_poses(self.sequence_dir / "poses.txt")
        if camera_poses.shape[0] != self.frame_count:
            raise SceneDataError("pose count does not match scan count")
        lidar_from_camera = np.linalg.inv(calibration["Tr"])
        lidar_poses = np.stack(
            [lidar_from_camera @ pose @ calibration["Tr"] for pose in camera_poses],
            axis=0,
        )
        for frame, pose in enumerate(lidar_poses):
            _rigid(f"LiDAR pose {frame}", pose)
        self._lidar_poses = _freeze(lidar_poses.astype(np.float64, copy=False))

        self._label_paths: dict[int, Path] | None = None
        if self.label_mode is LabelMode.REQUIRED:
            paths = _indexed_files(self.sequence_dir / "labels", ".label")
            if sorted(paths) != list(range(self.frame_count)):
                raise SceneDataError("labels must cover every scan")
            self._label_paths = paths

        self._semantic_target_lut = np.full(1 << 16, -1, dtype=np.int16)
        for raw, target in protocol.normal_training_class_map.items():
            self._semantic_target_lut[raw] = target
        self._semantic_target_lut.setflags(write=False)

        self._frames: OrderedDict[int, SourceFrame] = OrderedDict()

    @classmethod
    def open(
        cls,
        data_root: Path | str,
        *,
        protocol: AJAEProtocol,
        partition: str,
        sequence_id: int,
        label_mode: LabelMode | str,
    ) -> STUSequence:
        spec = protocol.sequence(partition, sequence_id)
        return cls(
            locate_sequence(data_root, partition, sequence_id),
            protocol=protocol,
            spec=spec,
            label_mode=label_mode,
        )

    @property
    def labels_available(self) -> bool:
        return self._label_paths is not None

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, frame_id: int) -> SceneWindow:
        frame = _plain_int("frame_id", frame_id)
        if frame >= self.frame_count:
            raise IndexError(frame)
        return self.window(frame)

    def __iter__(self) -> Iterator[SceneWindow]:
        for frame in range(self.frame_count):
            yield self.window(frame)

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
        raw = np.fromfile(path, dtype=SCAN_DTYPE)
        if raw.size % SCAN_CHANNELS:
            raise SceneDataError(f"scan cannot be reshaped to N x 4: {path}")
        xyzi = raw.reshape(-1, SCAN_CHANNELS).astype(np.float32, copy=False)
        _finite(f"scan {frame}", xyzi)

        labels = self._read_labels(frame, xyzi.shape[0])
        result = make_source_frame(frame, xyzi, self._lidar_poses[frame], labels)
        self._frames[frame] = result
        while len(self._frames) > WINDOW_FRAMES:
            self._frames.popitem(last=False)
        return result

    def _read_labels(self, frame: int, slot_count: int) -> PointLabels | None:
        if self._label_paths is None:
            return None
        path = self._label_paths[frame]
        if path.stat().st_size <= 0 or path.stat().st_size % LABEL_DTYPE.itemsize:
            raise SceneDataError(f"invalid label byte length: {path}")
        packed = np.fromfile(path, dtype=LABEL_DTYPE).astype(np.uint32, copy=False)
        if packed.size != slot_count:
            raise SceneDataError(
                f"frame {frame} has {slot_count} scan slots but {packed.size} labels"
            )
        semantic = (packed & np.uint32(0xFFFF)).astype(np.uint16, copy=False)
        instance = (packed >> np.uint32(16)).astype(np.uint16, copy=False)
        semantic_target: np.ndarray | None = None
        if self.spec.role in {"normal_training", "normal_validation"}:
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
            semantic_target=(
                None if semantic_target is None else _freeze(semantic_target)
            ),
        )

    def window(self, current_frame: int) -> SceneWindow:
        frame = _plain_int("current_frame", current_frame)
        if frame >= self.frame_count:
            raise IndexError(frame)
        frame_ids = self.protocol.window_frame_ids(
            self.spec.partition, self.spec.sequence_id, frame
        )
        return self._assemble(frame_ids, frame)

    def _assemble(self, frame_ids: tuple[int, ...], current_frame: int) -> SceneWindow:
        return assemble_window(
            self.spec,
            current_frame,
            tuple(self.source_frame(frame_id) for frame_id in frame_ids),
        )

    def audit(self, *, deep: bool = False) -> dict[str, object]:
        """Describe source data without running a model or metric."""

        result: dict[str, object] = {
            "partition": self.spec.partition,
            "sequence": self.spec.sequence_id,
            "role": self.spec.role,
            "frames": self.frame_count,
            "labels_read": self.labels_available,
            "model_input": {
                "coordinates": "released_STU_pre_voxel_coordinates",
                "features": ["intensity", "official_STU_distance"],
            },
        }
        if not deep:
            return result
        slots = 0
        real = 0
        for frame in range(self.frame_count):
            source = self.source_frame(frame)
            slots += source.slot_count
            real += source.real_count
        result.update(
            {
                "file_slots": slots,
                "real_returns": real,
                "zero_coordinate_slots": slots - real,
            }
        )
        return result


def summarize_window(window: SceneWindow) -> dict[str, object]:
    return {
        "partition": window.spec.partition,
        "sequence": window.spec.sequence_id,
        "current_frame": window.current_frame,
        "frame_ids": [item.source.frame_id for item in window.frames],
        "frame_ages": [item.age for item in window.frames],
        "input_slots_by_frame": [item.source.slot_count for item in window.frames],
        "real_members": window.members.count,
        "current_real_returns": window.current.real_count,
        "feature_channels": 2,
        "labels_read": window.labels is not None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect AJAE schema-28 STU windows.")
    parser.add_argument("--split", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val", "test"), required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument(
        "--labels",
        choices=tuple(mode.value for mode in LabelMode),
        required=True,
    )
    parser.add_argument("--frame", type=int, action="append")
    parser.add_argument("--check-all", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_protocol() if args.split is None else load_protocol(args.split)
    sequence = STUSequence.open(
        args.data_root,
        protocol=protocol,
        partition=args.partition,
        sequence_id=args.sequence,
        label_mode=args.labels,
    )
    frames = args.frame or list(
        dict.fromkeys((0, min(WINDOW_FRAMES - 1, sequence.frame_count - 1)))
    )
    output = {
        "sequence": sequence.audit(deep=args.check_all),
        "windows": [summarize_window(sequence.window(frame)) for frame in frames],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
