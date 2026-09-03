#!/usr/bin/env python3
"""Read STU sequences and assemble symmetric five-scan windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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
        ExperimentCondition,
        SequenceSpec,
        WINDOW_MEMBER_OFFSETS,
        load_protocol,
    )
except ImportError:  # Direct script execution.
    from protocol import (
        AJAEProtocol,
        ExperimentCondition,
        SequenceSpec,
        WINDOW_MEMBER_OFFSETS,
        load_protocol,
    )


SCAN_CHANNELS = 4
SCAN_DTYPE = np.dtype("<f4")
LABEL_DTYPE = np.dtype("<u4")
RIGID_ATOL = 1.0e-3
IDENTITY_ATOL = 1.0e-9
RAY_MAPPING_DOMAIN = 128 * 1024
SOURCE_FRAME_CACHE_SIZE = 16
LOGGER = logging.getLogger(__name__)


class SceneDataError(ValueError):
    """Report malformed STU data or an invalid scene relation."""


class LabelMode(str, Enum):
    """Choose whether a caller is allowed to read labels."""

    REQUIRED = "required"
    FORBIDDEN = "forbidden"


_SEALED_ACCESS_KEY = object()


class _SealedSequenceAccess:
    """Carry a validated method-freeze decision into the lowest data loader."""

    __slots__ = ("condition", "partition", "protocol")

    def __init__(
        self,
        protocol: AJAEProtocol,
        partition: str,
        condition: str,
        *,
        key: object,
    ) -> None:
        if key is not _SEALED_ACCESS_KEY:
            raise SceneDataError("sealed sequence access must come from the evaluator")
        if partition not in {"val", "test"}:
            raise SceneDataError("only validation and test sequences are sealed")
        self.protocol = protocol
        self.partition = partition
        self.condition = condition


def _grant_sealed_sequence_access(
    protocol: AJAEProtocol,
    *,
    partition: str,
    condition: str,
) -> _SealedSequenceAccess:
    """Create a loader capability only after the evaluator validates method freeze."""

    return _SealedSequenceAccess(
        protocol,
        partition,
        condition,
        key=_SEALED_ACCESS_KEY,
    )


def _require_sealed_sequence_access(
    protocol: AJAEProtocol,
    partition: str,
    access: _SealedSequenceAccess | None,
    *,
    sequence_id: int,
) -> None:
    if partition in {"val", "test"} and (
        not isinstance(access, _SealedSequenceAccess)
        or access.protocol is not protocol
        or access.partition != partition
    ):
        message = f"{partition} sequences are sealed until the evaluator validates method freeze"
        LOGGER.warning(
            "Refused sealed sequence access: partition=%s sequence=%s",
            partition,
            sequence_id,
        )
        raise SceneDataError(message)


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


def canonical_ray_mapping_digest(mapping: np.ndarray) -> str:
    """Bind a complete slot-to-ray permutation to immutable calibration bytes."""

    values = np.asarray(mapping)
    if values.dtype != np.int32 or values.shape != (RAY_MAPPING_DOMAIN,):
        raise TypeError("canonical ray mapping must be int32[131072]")
    if np.unique(values).size != values.size or np.any(
        (values < 0) | (values >= RAY_MAPPING_DOMAIN)
    ):
        raise SceneDataError("canonical ray mapping must be a complete permutation")
    return hashlib.sha256(
        b"AJAE-schema30-OS1-128-slot-to-ray\0" + values.tobytes(order="C")
    ).hexdigest()


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
    center = coordinates.mean(axis=0)
    distance = np.linalg.norm(coordinates - center, axis=1)[:, None]
    return _freeze(np.hstack((array[:, 3:4], distance)).astype(np.float32, copy=False))


@dataclass(frozen=True, slots=True)
class RayId:
    """One canonical OS1-128 beam and azimuth-column identity."""

    beam_id: int
    azimuth_column: int

    def __post_init__(self) -> None:
        if not 0 <= _plain_int("RayId.beam_id", self.beam_id) < 128:
            raise SceneDataError("RayId.beam_id must lie in [0,127]")
        if not 0 <= _plain_int("RayId.azimuth_column", self.azimuth_column) < 1024:
            raise SceneDataError("RayId.azimuth_column must lie in [0,1023]")


@dataclass(frozen=True, slots=True)
class PointId:
    """Stable identity of one visible return in the canonical ray grid."""

    frame_id: int
    ray: RayId

    def __post_init__(self) -> None:
        _plain_int("PointId.frame_id", self.frame_id)
        if not isinstance(self.ray, RayId):
            raise TypeError("PointId.ray must be RayId")


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


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """One complete STU file-slot scan and its frozen official model inputs."""

    partition: str
    sequence_id: int
    frame_id: int
    xyzi: np.ndarray
    lidar_pose: np.ndarray
    coordinates: np.ndarray
    features: np.ndarray
    zero_slot_mask: np.ndarray
    real_slots: np.ndarray
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
    zero_mask = _freeze(np.all(owned[:, :3] == np.float32(0.0), axis=1))
    real_slots = _freeze(np.flatnonzero(~zero_mask).astype(np.int32, copy=False))
    return SourceFrame(
        partition=partition,
        sequence_id=_plain_int("sequence_id", sequence_id),
        frame_id=frame,
        xyzi=owned,
        lidar_pose=_freeze(pose.copy()),
        coordinates=official_stu_coordinates(owned, pose),
        features=official_stu_features(owned, pose),
        zero_slot_mask=zero_mask,
        real_slots=real_slots,
        labels=labels,
    )


@dataclass(frozen=True, slots=True)
class WindowReferencePose:
    """A five-pose-symmetric window frame expressed in world coordinates.

    ``rotation`` is :math:`R_{W<-G}` and ``translation`` is the window origin
    in world coordinates. Input sensor poses are always :math:`T_{W<-S}`.
    """

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation)
        translation = np.asarray(self.translation)
        if rotation.dtype != np.float64 or rotation.shape != (3, 3):
            raise TypeError("WindowReferencePose.rotation must be float64[3,3]")
        if translation.dtype != np.float64 or translation.shape != (3,):
            raise TypeError("WindowReferencePose.translation must be float64[3]")
        _finite("window reference rotation", rotation)
        _finite("window reference translation", translation)
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3, dtype=np.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        ):
            raise SceneDataError(
                "window reference rotation is not numerically orthogonal"
            )
        if not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-10, rel_tol=1.0e-10
        ):
            raise SceneDataError("window reference rotation determinant is not +1")
        object.__setattr__(self, "rotation", _freeze(rotation.copy()))
        object.__setattr__(self, "translation", _freeze(translation.copy()))
        identity = np.eye(4, dtype=np.float64)
        world_from_window = self.world_from_window
        window_from_world = self.window_from_world
        if not (
            np.allclose(
                world_from_window @ window_from_world,
                identity,
                atol=1.0e-10,
                rtol=1.0e-10,
            )
            and np.allclose(
                window_from_world @ world_from_window,
                identity,
                atol=1.0e-10,
                rtol=1.0e-10,
            )
        ):
            raise SceneDataError("window reference transforms are not mutual inverses")

    @classmethod
    def from_sensor_poses(
        cls, sensor_to_world: Sequence[np.ndarray]
    ) -> "WindowReferencePose":
        """Compute the translation mean and SO(3) chordal mean of five T_W<-S poses."""

        poses = tuple(np.asarray(pose) for pose in sensor_to_world)
        if len(poses) != len(WINDOW_MEMBER_OFFSETS):
            raise SceneDataError(
                "a window reference requires exactly five sensor poses"
            )
        for index, pose in enumerate(poses):
            if pose.dtype != np.float64:
                raise TypeError(f"sensor_to_world[{index}] must be float64[4,4]")
            _rigid(f"sensor_to_world[{index}]", pose)

        # Canonical accumulation makes floating-point construction independent
        # of the caller's scan order without assigning a privileged scan.
        ordered = tuple(sorted(poses, key=lambda pose: pose.tobytes(order="C")))
        translation = np.sum(
            np.stack([pose[:3, 3] for pose in ordered]), axis=0, dtype=np.float64
        ) / float(len(ordered))
        rotation_sum = np.sum(
            np.stack([pose[:3, :3] for pose in ordered]), axis=0, dtype=np.float64
        )
        left, _, right_t = np.linalg.svd(rotation_sum)
        correction = np.eye(3, dtype=np.float64)
        correction[2, 2] = -1.0 if np.linalg.det(left @ right_t) < 0.0 else 1.0
        rotation = left @ correction @ right_t
        return cls(
            rotation.astype(np.float64, copy=False),
            translation.astype(np.float64, copy=False),
        )

    @property
    def world_from_window(self) -> np.ndarray:
        """Return T_W<-G, mapping symmetric-window coordinates to world coordinates."""

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.translation
        return _freeze(transform)

    @property
    def window_from_world(self) -> np.ndarray:
        """Return T_G<-W, the exact rigid inverse of ``world_from_window``."""

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation.T
        transform[:3, 3] = -(self.rotation.T @ self.translation)
        return _freeze(transform)


@dataclass(frozen=True, slots=True)
class WindowFrame:
    """One source scan and its transform into the symmetric window frame."""

    source: SourceFrame
    scan_group: int
    source_to_window: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceFrame):
            raise TypeError("WindowFrame.source must be SourceFrame")
        group = _plain_int("WindowFrame.scan_group", self.scan_group)
        if group >= len(WINDOW_MEMBER_OFFSETS):
            raise SceneDataError("WindowFrame.scan_group must lie in [0,4]")
        transform = np.asarray(self.source_to_window)
        if transform.dtype != np.float64:
            raise TypeError("source_to_window must be float64[4,4]")
        _rigid("source_to_window", transform)
        object.__setattr__(self, "source_to_window", _freeze(transform.copy()))


@dataclass(frozen=True, slots=True)
class WindowPoints:
    """All five scans' visible returns in symmetric window coordinates."""

    coordinates: np.ndarray
    scan_group: np.ndarray
    source_frame: np.ndarray
    source_slot: np.ndarray
    source_ray: np.ndarray
    ray_mapping_audited: bool
    ray_mapping_digest: str | None

    def __post_init__(self) -> None:
        count = self.coordinates.shape[0]
        if self.coordinates.dtype != np.float32 or self.coordinates.shape != (count, 3):
            raise TypeError("coordinates must be float32[M,3]")
        for name, array, dtype in (
            ("scan_group", self.scan_group, np.int8),
            ("source_frame", self.source_frame, np.int32),
            ("source_slot", self.source_slot, np.int32),
            ("source_ray", self.source_ray, np.int32),
        ):
            if array.dtype != dtype or array.shape != (count,):
                raise TypeError(f"{name} must be {np.dtype(dtype).name}[M]")
        if type(self.ray_mapping_audited) is not bool:
            raise TypeError("ray_mapping_audited must be boolean")
        if self.ray_mapping_audited:
            if (
                not isinstance(self.ray_mapping_digest, str)
                or len(self.ray_mapping_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.ray_mapping_digest
                )
            ):
                raise SceneDataError(
                    "an audited ray mapping requires its calibration digest"
                )
        elif self.ray_mapping_digest is not None:
            raise SceneDataError(
                "an unaudited ray mapping cannot carry an audit digest"
            )
        _finite("symmetric-window coordinates", self.coordinates)
        if np.any(
            (self.scan_group < 0) | (self.scan_group >= len(WINDOW_MEMBER_OFFSETS))
        ):
            raise SceneDataError("scan groups must lie in [0,4]")
        if np.any(self.source_frame < 0) or np.any(self.source_slot < 0):
            raise SceneDataError(
                "source frame and slot identities must be non-negative"
            )
        if np.any((self.source_ray < 0) | (self.source_ray >= RAY_MAPPING_DOMAIN)):
            raise SceneDataError("canonical ray IDs must lie in [0,131071]")
        if count > 1:
            same_group = self.scan_group[1:] == self.scan_group[:-1]
            if np.any(same_group & (self.source_frame[1:] != self.source_frame[:-1])):
                raise SceneDataError("one scan group cannot mix source frames")
            if np.any(same_group & (self.source_slot[1:] <= self.source_slot[:-1])):
                raise SceneDataError(
                    "source slots must increase within each scan group"
                )
            starts = np.concatenate(
                (np.asarray([True]), self.scan_group[1:] != self.scan_group[:-1])
            )
            started_groups = self.scan_group[starts]
            if np.unique(started_groups).size != started_groups.size:
                raise SceneDataError(
                    "a scan group must occupy one contiguous point block"
                )
        for array in (
            self.coordinates,
            self.scan_group,
            self.source_frame,
            self.source_slot,
            self.source_ray,
        ):
            array.setflags(write=False)

    @property
    def count(self) -> int:
        return int(self.coordinates.shape[0])

    def point_id(self, index: int) -> PointId:
        point = _plain_int("point index", index)
        if point >= self.count:
            raise IndexError(point)
        ray = int(self.source_ray[point])
        return PointId(
            int(self.source_frame[point]),
            RayId(ray // 1024, ray % 1024),
        )


@dataclass(frozen=True, slots=True)
class SceneWindow:
    """One complete five-scan observation in a symmetric reference frame."""

    spec: SequenceSpec
    condition: ExperimentCondition
    window_start: int
    frame_ids: tuple[int, ...]
    reference_pose: WindowReferencePose
    frames: tuple[WindowFrame, ...]
    points: WindowPoints
    labels: PointLabels | None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, SequenceSpec):
            raise TypeError("spec must be SequenceSpec")
        if not isinstance(self.condition, ExperimentCondition):
            raise TypeError("condition must be ExperimentCondition")
        start = _plain_int("window_start", self.window_start)
        declared_ids = tuple(self.frame_ids)
        if any(type(frame_id) is not int for frame_id in declared_ids):
            raise TypeError("frame_ids must contain plain integers")
        expected_ids = tuple(start + offset for offset in WINDOW_MEMBER_OFFSETS)
        if declared_ids != expected_ids:
            raise SceneDataError(
                "frame_ids must be the five consecutive IDs from window_start"
            )
        if tuple(self.spec.window_frame_ids(start)) != declared_ids:
            raise SceneDataError(
                "window identity is not legal for the sequence specification"
            )
        if not isinstance(self.reference_pose, WindowReferencePose):
            raise TypeError("reference_pose must be WindowReferencePose")
        if len(self.frames) != len(WINDOW_MEMBER_OFFSETS):
            raise SceneDataError("a SceneWindow must contain exactly five source scans")

        source_ids = tuple(item.source.frame_id for item in self.frames)
        if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(
            declared_ids
        ):
            raise SceneDataError(
                "window source scans must match the declared frame IDs once each"
            )
        canonical_group = {
            frame_id: index for index, frame_id in enumerate(declared_ids)
        }
        expected_pose = WindowReferencePose.from_sensor_poses(
            tuple(item.source.lidar_pose for item in self.frames)
        )
        if not np.allclose(
            self.reference_pose.world_from_window,
            expected_pose.world_from_window,
            atol=IDENTITY_ATOL,
            rtol=IDENTITY_ATOL,
        ):
            raise SceneDataError(
                "window reference pose does not match all five sensor poses"
            )
        window_from_world = self.reference_pose.window_from_world
        for item in self.frames:
            group = canonical_group[item.source.frame_id]
            if item.scan_group != group:
                raise SceneDataError(
                    "scan group does not match the declared source frame"
                )
            expected_transform = window_from_world @ item.source.lidar_pose
            if not np.allclose(
                item.source_to_window,
                expected_transform,
                atol=IDENTITY_ATOL,
                rtol=IDENTITY_ATOL,
            ):
                raise SceneDataError("source-to-window transform is inconsistent")
            mask = self.points.scan_group == group
            if not np.all(self.points.source_frame[mask] == item.source.frame_id):
                raise SceneDataError(
                    "point frame identities do not match their scan group"
                )
            if not np.array_equal(
                self.points.source_slot[mask], item.source.real_slots
            ):
                raise SceneDataError("point slots do not match visible source returns")
        if self.points.count != sum(item.source.real_count for item in self.frames):
            raise SceneDataError(
                "window points do not contain every visible return exactly once"
            )
        if self.labels is not None and self.labels.packed.size != self.points.count:
            raise SceneDataError("window labels do not match visible returns")

    def frame_for_id(self, frame_id: int) -> WindowFrame:
        """Return one member by stable source-frame identity, independent of row order."""

        identifier = _plain_int("frame_id", frame_id)
        for frame in self.frames:
            if frame.source.frame_id == identifier:
                return frame
        raise KeyError(identifier)

    def restore_source_frame(self, frame_id: int, values: np.ndarray) -> np.ndarray:
        """Restore a full-window value array to one source scan's file-slot order."""

        frame = self.frame_for_id(frame_id)
        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] != self.points.count:
            raise ValueError(
                f"values must have leading size {self.points.count}, got {array.shape}"
            )
        mask = self.points.source_frame == frame.source.frame_id
        return frame.source.restore_real(array[mask])


def assemble_window(
    spec: SequenceSpec,
    window_start: int,
    frame_ids: Sequence[int],
    sources: Sequence[SourceFrame],
    *,
    condition: ExperimentCondition | str = ExperimentCondition.B3,
    canonical_ray_by_slot: np.ndarray | Mapping[int, np.ndarray] | None = None,
    ray_mapping_audited: bool = False,
    ray_mapping_digest: str | None = None,
) -> SceneWindow:
    """Assemble all five scans while preserving order-independent point identity."""

    if not isinstance(spec, SequenceSpec):
        raise TypeError("spec must be SequenceSpec")
    selected = ExperimentCondition(condition)
    start = _plain_int("window_start", window_start)
    declared_ids = tuple(frame_ids)
    if any(type(frame_id) is not int for frame_id in declared_ids):
        raise TypeError("frame_ids must contain plain integers")
    expected_ids = tuple(start + offset for offset in WINDOW_MEMBER_OFFSETS)
    if (
        declared_ids != expected_ids
        or tuple(spec.window_frame_ids(start)) != declared_ids
    ):
        raise SceneDataError("frame_ids do not define a legal five-scan window")

    source_frames = tuple(sources)
    if len(source_frames) != len(WINDOW_MEMBER_OFFSETS):
        raise SceneDataError("a window requires exactly five source scans")
    source_ids = tuple(source.frame_id for source in source_frames)
    if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(declared_ids):
        raise SceneDataError("sources must contain each declared frame exactly once")
    if any(
        source.partition != spec.partition or source.sequence_id != spec.sequence_id
        for source in source_frames
    ):
        raise SceneDataError(
            "source identity does not match the sequence specification"
        )
    labels_present = tuple(source.labels is not None for source in source_frames)
    if len(set(labels_present)) != 1:
        raise SceneDataError("all scans must have the same label availability")
    targets_present = tuple(
        source.labels is not None and source.labels.semantic_target is not None
        for source in source_frames
    )
    if labels_present[0] and len(set(targets_present)) != 1:
        raise SceneDataError("all labels must have the same STU-target availability")
    if ray_mapping_audited and canonical_ray_by_slot is None:
        raise SceneDataError(
            "an audited window requires an explicit calibrated mapping"
        )
    if not ray_mapping_audited and ray_mapping_digest is not None:
        raise SceneDataError("an unaudited window cannot carry a calibration digest")

    reference_pose = WindowReferencePose.from_sensor_poses(
        tuple(source.lidar_pose for source in source_frames)
    )
    window_from_world = reference_pose.window_from_world
    canonical_group = {frame_id: index for index, frame_id in enumerate(declared_ids)}
    frames: list[WindowFrame] = []
    coordinates: list[np.ndarray] = []
    scan_groups: list[np.ndarray] = []
    source_ids_by_point: list[np.ndarray] = []
    source_slots: list[np.ndarray] = []
    source_rays: list[np.ndarray] = []
    packed: list[np.ndarray] = []
    semantic: list[np.ndarray] = []
    instance: list[np.ndarray] = []
    semantic_targets: list[np.ndarray] = []

    for source in source_frames:
        group = canonical_group[source.frame_id]
        # Poses are T_W<-S; composition gives T_G<-S for symmetric coordinates.
        transform = window_from_world @ source.lidar_pose
        _rigid(f"source-to-window pose {source.frame_id}", transform)
        transform = _freeze(transform.astype(np.float64, copy=False))
        frames.append(WindowFrame(source, group, transform))
        slots = source.real_slots
        source_xyz = source.xyzi[slots, :3].astype(np.float64, copy=False)
        aligned = source_xyz @ transform[:3, :3].T + transform[:3, 3]
        coordinates.append(aligned.astype(np.float32))
        scan_groups.append(np.full(slots.size, group, dtype=np.int8))
        source_ids_by_point.append(np.full(slots.size, source.frame_id, dtype=np.int32))
        source_slots.append(slots.copy())
        if canonical_ray_by_slot is None:
            mapping = np.arange(source.slot_count, dtype=np.int32)
        elif isinstance(canonical_ray_by_slot, Mapping):
            if source.frame_id not in canonical_ray_by_slot:
                raise SceneDataError(
                    f"canonical ray mapping lacks frame {source.frame_id}"
                )
            mapping = np.asarray(canonical_ray_by_slot[source.frame_id])
        else:
            mapping = np.asarray(canonical_ray_by_slot)
        if mapping.dtype != np.int32 or mapping.shape != (source.slot_count,):
            raise TypeError("canonical_ray_by_slot must provide int32[slot]")
        if np.unique(mapping).size != mapping.size or np.any(
            (mapping < 0) | (mapping >= RAY_MAPPING_DOMAIN)
        ):
            raise SceneDataError(
                "canonical ray mapping must be an in-range one-to-one map"
            )
        if (
            ray_mapping_audited
            and canonical_ray_mapping_digest(mapping) != ray_mapping_digest
        ):
            raise SceneDataError(
                "canonical ray mapping does not match its calibration digest"
            )
        source_rays.append(mapping[slots].copy())
        if source.labels is not None:
            packed.append(source.labels.packed[slots])
            semantic.append(source.labels.semantic[slots])
            instance.append(source.labels.instance[slots])
            if source.labels.semantic_target is not None:
                semantic_targets.append(source.labels.semantic_target[slots])

    points = WindowPoints(
        coordinates=_freeze(np.concatenate(coordinates)),
        scan_group=_freeze(np.concatenate(scan_groups)),
        source_frame=_freeze(np.concatenate(source_ids_by_point)),
        source_slot=_freeze(np.concatenate(source_slots)),
        source_ray=_freeze(np.concatenate(source_rays)),
        ray_mapping_audited=ray_mapping_audited,
        ray_mapping_digest=ray_mapping_digest,
    )
    labels: PointLabels | None = None
    if labels_present[0]:
        labels = PointLabels(
            packed=_freeze(np.concatenate(packed)),
            semantic=_freeze(np.concatenate(semantic)),
            instance=_freeze(np.concatenate(instance)),
            semantic_target=(
                _freeze(np.concatenate(semantic_targets))
                if targets_present[0]
                else None
            ),
        )
    return SceneWindow(
        spec=spec,
        condition=selected,
        window_start=start,
        frame_ids=declared_ids,
        reference_pose=reference_pose,
        frames=tuple(frames),
        points=points,
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
    protocol: AJAEProtocol,
    sealed_access: _SealedSequenceAccess | None = None,
) -> Path:
    """Resolve one protocol sequence without searching alternative layouts."""

    if partition not in {"train", "val", "test"}:
        raise ValueError("partition must be train, val, or test")
    identifier = _plain_int("sequence_id", sequence_id)
    _require_sealed_sequence_access(
        protocol,
        partition,
        sealed_access,
        sequence_id=identifier,
    )
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
        protocol: AJAEProtocol,
        spec: SequenceSpec,
        label_mode: LabelMode | str,
        sealed_access: _SealedSequenceAccess | None = None,
    ) -> None:
        if not isinstance(protocol, AJAEProtocol):
            raise TypeError("protocol must be AJAEProtocol")
        if not isinstance(spec, SequenceSpec):
            raise TypeError("spec must be SequenceSpec")
        if protocol.sequence(spec.partition, spec.sequence_id) != spec:
            raise SceneDataError("sequence spec is not part of this protocol")
        _require_sealed_sequence_access(
            protocol,
            spec.partition,
            sealed_access,
            sequence_id=spec.sequence_id,
        )
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
        # Hidden sequence lengths become observable only after opening their files.
        self.spec = spec.with_observed_frame_count(self.frame_count)
        self.span = self.spec.span
        self.window_starts = self.spec.legal_window_starts()

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
        for raw, target in protocol.normal_training_class_map.items():
            self._semantic_target_lut[raw] = target
        self._semantic_target_lut.setflags(write=False)
        self._frames: OrderedDict[int, SourceFrame] = OrderedDict()
        self._cache_frames = SOURCE_FRAME_CACHE_SIZE

    @classmethod
    def open(
        cls,
        data_root: Path | str,
        *,
        protocol: AJAEProtocol,
        partition: str,
        sequence_id: int,
        label_mode: LabelMode | str,
        sealed_access: _SealedSequenceAccess | None = None,
    ) -> "STUSequence":
        spec = protocol.sequence(partition, sequence_id)
        return cls(
            locate_sequence(
                data_root,
                partition,
                sequence_id,
                protocol=protocol,
                sealed_access=sealed_access,
            ),
            protocol=protocol,
            spec=spec,
            label_mode=label_mode,
            sealed_access=sealed_access,
        )

    @property
    def labels_available(self) -> bool:
        return self._label_paths is not None

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, window_start: int) -> SceneWindow:
        return self.window(window_start)

    def __iter__(self) -> Iterator[SceneWindow]:
        for window_start in self.window_starts:
            yield self.window(window_start)

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
        packed = np.fromfile(path, dtype=LABEL_DTYPE).astype(np.uint32, copy=False)
        if packed.size != slot_count:
            raise SceneDataError(
                f"frame {frame} has {slot_count} scan slots but {packed.size} labels"
            )
        semantic = (packed & np.uint32(0xFFFF)).astype(np.uint16, copy=False)
        instance = (packed >> np.uint32(16)).astype(np.uint16, copy=False)
        semantic_target: np.ndarray | None = None
        if self.spec.supports_counterfactuals:
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

    def window(
        self,
        window_start: int,
        *,
        condition: ExperimentCondition | str = ExperimentCondition.B3,
        canonical_ray_by_slot: np.ndarray | Mapping[int, np.ndarray] | None = None,
        ray_mapping_audited: bool = False,
        ray_mapping_digest: str | None = None,
    ) -> SceneWindow:
        start = _plain_int("window_start", window_start)
        selected = ExperimentCondition(condition)
        if start not in frozenset(self.window_starts):
            raise SceneDataError(
                f"frame {start} is not a legal five-scan window for "
                f"{self.spec.partition}/{self.spec.sequence_id}"
            )
        frame_ids = self.spec.window_frame_ids(start)
        if frame_ids[-1] >= self.frame_count:
            raise SceneDataError("protocol window refers to a missing source frame")
        return assemble_window(
            self.spec,
            start,
            frame_ids,
            tuple(self.source_frame(frame_id) for frame_id in frame_ids),
            condition=selected,
            canonical_ray_by_slot=canonical_ray_by_slot,
            ray_mapping_audited=ray_mapping_audited,
            ray_mapping_digest=ray_mapping_digest,
        )

    def audit(self, *, deep: bool = False) -> dict[str, object]:
        """Describe source data without running a model or metric."""

        result: dict[str, object] = {
            "partition": self.spec.partition,
            "sequence": self.spec.sequence_id,
            "role": self.spec.role,
            "source_frames": self.frame_count,
            "legal_windows": len(self.window_starts),
            "first_window_start": self.window_starts[0],
            "last_window_start": self.window_starts[-1],
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
        for frame_id in self.frame_ids:
            source = self.source_frame(frame_id)
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
        "condition": window.condition.value,
        "window_start": window.window_start,
        "frame_ids": list(window.frame_ids),
        "source_order": [item.source.frame_id for item in window.frames],
        "scan_groups": [item.scan_group for item in window.frames],
        "input_slots_by_frame": [item.source.slot_count for item in window.frames],
        "visible_returns": window.points.count,
        "visible_returns_by_frame": [item.source.real_count for item in window.frames],
        "feature_channels": 2,
        "labels_read": window.labels is not None,
        "ray_mapping_audited": window.points.ray_mapping_audited,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect AJAE schema-32 STU windows.")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val", "test"), required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument(
        "--labels", choices=tuple(mode.value for mode in LabelMode), required=True
    )
    parser.add_argument("--window-start", type=int, action="append")
    parser.add_argument(
        "--condition",
        choices=tuple(item.value for item in ExperimentCondition),
        default=ExperimentCondition.B3.value,
    )
    parser.add_argument("--check-all", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    condition = ExperimentCondition(args.condition)
    starts = sequence.window_starts
    selected_starts = args.window_start or [starts[0], starts[-1]]
    output = {
        "sequence": sequence.audit(deep=args.check_all),
        "windows": [
            summarize_window(sequence.window(start, condition=condition))
            for start in selected_starts
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
