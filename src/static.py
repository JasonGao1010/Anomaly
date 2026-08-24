#!/usr/bin/env python3
"""STU-initialized current-anchored causal window model used by AJAE.

The shared frozen STU stem encodes each source scan. Trainable p16, p8, and p4
history residuals read aligned causal evidence and modify only current-frame
point features and logits. Oracle labels select historical evidence during the
mechanism experiment but are never exposed to the temporal module or anomaly
head.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .protocol import AJAEProtocol, WINDOW_FRAMES
    from .scene import SceneWindow
except ImportError:  # Direct script execution.
    from protocol import AJAEProtocol, WINDOW_FRAMES
    from scene import SceneWindow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STU_REPOSITORY = (
    PROJECT_ROOT.parent / "DynaCAN-deps" / "stu_dataset" / "Mask4Former3D"
)
NUM_NORMAL_CLASSES = 19
NUM_QUERIES = 100
MASK_DIM = 128
VOXEL_SIZE_METRES = 0.05
MINIMUM_INSTANCE_CENTROID_POINTS = 8
MAXIMUM_OBSERVED_SPEED_MPS = 30.0
INSTANCE_DISPLACEMENT_TOLERANCE_METRES = 1.0
MAXIMUM_TEMPORAL_LOGIT_CORRECTION = 4.0
STU_MODEL_STATE_FORMAT = "ajae-stu-normal-model-state-v2"
STU_MODEL_STATE_CONVERSION_RULE = "extract_exact_model_prefix_strip_once_v1"
# Frozen by the controlled model.* extraction from the official STU release.
STU_CHECKPOINT_BYTES = 476_261_075
STU_CHECKPOINT_SHA256 = (
    "743b10d39c4076d98533bf1e84d389ad2703016904d31146e48919618b07b67a"
)
STU_MODEL_STATE_BYTES = 158_806_731
STU_MODEL_STATE_SHA256 = (
    "bd62c2ace0fd13911e2ba81f4969ca6633e73ec5270ffc0b1bd61840b05f924d"
)
STU_MODEL_STATE_TENSOR_SHA256 = (
    "0be4805592a3d064b21655c6c6eeeb7227322c9670873345be52747b0a24d1fb"
)


class StaticModelError(ValueError):
    """Report an invalid model input, checkpoint, target, or output."""


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        count = int((~torch.isfinite(value)).sum().item())
        raise StaticModelError(f"{name} contains {count} non-finite value(s)")


def _as_tensor(array: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.array(array, copy=True)).to(device=device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(state: Mapping[str, object]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in canonical key order."""

    if any(not isinstance(name, str) for name in state):
        raise StaticModelError("STU model state must use string tensor names")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, Tensor):
            raise StaticModelError("STU model state must map strings to tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def stu_model_state_path(checkpoint: Path) -> Path:
    """Resolve the restricted-loader tensor file actually consumed by AJAE."""

    source = checkpoint.expanduser().resolve(strict=True)
    return source.with_name(f"{source.stem}.model_state.pt").resolve(strict=True)


def stu_source_manifest(repository: Path | str) -> dict[str, object]:
    """Hash every official STU Python and YAML source consumed by AJAE."""

    root = Path(repository).expanduser().resolve(strict=True)
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml"}
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    payload = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "root": str(root),
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "files": records,
    }


def _validated_stu_weights(
    checkpoint: Path,
) -> tuple[Mapping[str, Tensor], dict[str, object]]:
    """Validate both official bytes and the safely converted tensor payload."""

    source = checkpoint.expanduser().resolve(strict=True)
    converted = stu_model_state_path(source)
    source_sha256 = _sha256(source)
    converted_sha256 = _sha256(converted)
    if source.stat().st_size != STU_CHECKPOINT_BYTES or source_sha256 != (
        STU_CHECKPOINT_SHA256
    ):
        raise StaticModelError("STU checkpoint is not the frozen official release")
    if converted.stat().st_size != STU_MODEL_STATE_BYTES or converted_sha256 != (
        STU_MODEL_STATE_SHA256
    ):
        raise StaticModelError("converted STU checkpoint is not the frozen extraction")
    try:
        payload = torch.load(converted, map_location="cpu", weights_only=True)
    except Exception as error:
        raise StaticModelError(
            "converted STU checkpoint cannot be safely loaded"
        ) from error
    required = {
        "format",
        "conversion_rule",
        "source_checkpoint_bytes",
        "source_checkpoint_sha256",
        "tensor_sha256",
        "state_dict",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise StaticModelError("converted STU checkpoint has an invalid format")
    if payload["format"] != STU_MODEL_STATE_FORMAT:
        raise StaticModelError("converted STU checkpoint version is unsupported")
    if payload["conversion_rule"] != STU_MODEL_STATE_CONVERSION_RULE:
        raise StaticModelError(
            "converted STU checkpoint uses an unsupported extraction"
        )
    if payload["source_checkpoint_bytes"] != STU_CHECKPOINT_BYTES:
        raise StaticModelError("converted state refers to a different checkpoint size")
    if payload["source_checkpoint_sha256"] != STU_CHECKPOINT_SHA256:
        raise StaticModelError("converted state does not match the STU checkpoint")
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise StaticModelError("converted STU state_dict is empty")
    tensor_sha256 = model_state_sha256(state)
    if payload["tensor_sha256"] != STU_MODEL_STATE_TENSOR_SHA256:
        raise StaticModelError("converted STU checkpoint declares the wrong tensors")
    if tensor_sha256 != STU_MODEL_STATE_TENSOR_SHA256:
        raise StaticModelError("converted STU tensor content has the wrong identity")
    if _sha256(converted) != converted_sha256:
        raise StaticModelError("converted STU checkpoint changed while it was loaded")
    return state, {
        "checkpoint_bytes": source.stat().st_size,
        "checkpoint_sha256": source_sha256,
        "model_state_bytes": converted.stat().st_size,
        "model_state_sha256": converted_sha256,
        "model_state_tensor_sha256": tensor_sha256,
    }


def stu_weight_identity(checkpoint: Path) -> dict[str, object]:
    """Return the validated identities of both STU weight artifacts."""

    _, identity = _validated_stu_weights(checkpoint)
    return identity


@dataclass(frozen=True, slots=True)
class FrameInput:
    """One label-free source scan prepared by :mod:`scene`."""

    frame_id: int
    age: int
    coordinates: np.ndarray
    features: np.ndarray
    real_slots: np.ndarray
    lidar_pose: np.ndarray
    source_to_current: np.ndarray

    def __post_init__(self) -> None:
        count = self.coordinates.shape[0]
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise TypeError("frame_id must be a non-negative integer")
        if type(self.age) is not int or not 0 <= self.age < WINDOW_FRAMES:
            raise TypeError("age must lie between zero and four")
        if self.coordinates.dtype != np.float64 or self.coordinates.shape != (
            count,
            3,
        ):
            raise TypeError("coordinates must be float64[N,3]")
        if self.features.dtype != np.float32 or self.features.shape != (count, 2):
            raise TypeError("features must be float32[N,2]")
        if self.real_slots.dtype != np.int32 or self.real_slots.ndim != 1:
            raise TypeError("real_slots must be int32[M]")
        for name, matrix in (
            ("lidar_pose", self.lidar_pose),
            ("source_to_current", self.source_to_current),
        ):
            if matrix.dtype != np.float64 or matrix.shape != (4, 4):
                raise TypeError(f"{name} must be float64[4,4]")
        if (
            not np.isfinite(self.coordinates).all()
            or not np.isfinite(self.features).all()
        ):
            raise StaticModelError("frame input contains non-finite values")
        if np.any(self.real_slots < 0) or np.any(self.real_slots >= count):
            raise StaticModelError("real slot lies outside its source scan")


@dataclass(frozen=True, slots=True)
class StaticInput:
    """The complete label-free causal window consumed by AJAE."""

    partition: str
    sequence_id: int
    current_frame: int
    frames: tuple[FrameInput, ...]
    member_coordinates: np.ndarray
    member_ages: np.ndarray
    frame_offsets: np.ndarray
    current_slot_count: int
    current_real_slots: np.ndarray

    def __post_init__(self) -> None:
        if not 1 <= len(self.frames) <= WINDOW_FRAMES:
            raise StaticModelError("a static input needs one through five frames")
        if self.frames[-1].frame_id != self.current_frame or self.frames[-1].age != 0:
            raise StaticModelError("the last input frame must be current")
        count = self.member_coordinates.shape[0]
        if (
            self.member_coordinates.dtype != np.float32
            or self.member_coordinates.shape != (count, 3)
        ):
            raise TypeError("member_coordinates must be float32[M,3]")
        if self.member_ages.dtype != np.uint8 or self.member_ages.shape != (count,):
            raise TypeError("member_ages must be uint8[M]")
        if self.frame_offsets.dtype != np.int64 or self.frame_offsets.shape != (
            len(self.frames) + 1,
        ):
            raise TypeError("frame_offsets must be int64[F+1]")
        if int(self.frame_offsets[0]) != 0 or int(self.frame_offsets[-1]) != count:
            raise StaticModelError("frame offsets do not span all members")
        if np.any(np.diff(self.frame_offsets) < 0):
            raise StaticModelError("frame offsets must be nondecreasing")
        expected = np.concatenate(
            [
                np.full(frame.real_slots.size, frame.age, dtype=np.uint8)
                for frame in self.frames
            ]
        )
        if not np.array_equal(self.member_ages, expected):
            raise StaticModelError("member ages do not match source frames")
        if self.current_real_slots.dtype != np.int32:
            raise TypeError("current_real_slots must be int32")
        if self.current_real_slots.size != self.current_count:
            raise StaticModelError(
                "current slot identity does not match current members"
            )

    @classmethod
    def from_window(cls, window: SceneWindow) -> StaticInput:
        """Strip labels from a scene window without changing point order."""

        if not isinstance(window, SceneWindow):
            raise TypeError("window must be SceneWindow")
        frames = tuple(
            FrameInput(
                frame_id=item.source.frame_id,
                age=item.age,
                coordinates=item.source.coordinates,
                features=item.source.features,
                real_slots=item.source.real_slots,
                lidar_pose=item.source.lidar_pose,
                source_to_current=item.source_to_current,
            )
            for item in window.frames
        )
        return cls(
            partition=window.spec.partition,
            sequence_id=window.spec.sequence_id,
            current_frame=window.current_frame,
            frames=frames,
            member_coordinates=window.members.coordinates_current,
            member_ages=window.members.frame_age,
            frame_offsets=window.members.frame_offsets,
            current_slot_count=window.current.slot_count,
            current_real_slots=window.current.real_slots,
        )

    @property
    def member_count(self) -> int:
        return int(self.member_coordinates.shape[0])

    @property
    def current_slice(self) -> slice:
        return slice(int(self.frame_offsets[-2]), int(self.frame_offsets[-1]))

    @property
    def current_count(self) -> int:
        return int(self.frame_offsets[-1] - self.frame_offsets[-2])

    def restore_current(self, values: Tensor) -> Tensor:
        if values.ndim < 1 or values.shape[0] != self.current_count:
            raise StaticModelError("current prediction has the wrong point count")
        output = values.new_zeros((self.current_slot_count, *values.shape[1:]))
        slots = torch.from_numpy(self.current_real_slots.astype(np.int64)).to(
            values.device
        )
        output[slots] = values
        return output

    def history_subset(self, length: int) -> "StaticInput":
        """Keep the current scan and the nearest ``length`` causal scans."""

        if type(length) is not int or not 0 <= length < WINDOW_FRAMES:
            raise ValueError("history length must be 0, 1, 2, 3, or 4")
        selected_frames = tuple(
            frame for frame in self.frames if frame.age == 0 or frame.age <= length
        )
        start_frame = len(self.frames) - len(selected_frames)
        start_member = int(self.frame_offsets[start_frame])
        offsets = np.ascontiguousarray(
            self.frame_offsets[start_frame:] - start_member, dtype=np.int64
        )
        return StaticInput(
            partition=self.partition,
            sequence_id=self.sequence_id,
            current_frame=self.current_frame,
            frames=selected_frames,
            member_coordinates=np.ascontiguousarray(
                self.member_coordinates[start_member:], dtype=np.float32
            ),
            member_ages=np.ascontiguousarray(
                self.member_ages[start_member:], dtype=np.uint8
            ),
            frame_offsets=offsets,
            current_slot_count=self.current_slot_count,
            current_real_slots=np.ascontiguousarray(
                self.current_real_slots, dtype=np.int32
            ),
        )


@dataclass(frozen=True, slots=True)
class HistorySamplingOffsets:
    """Object-aware Oracle offsets used only by sparse interpolation.

    ``query_offsets[age, i]`` is added to current point ``i`` before querying a
    source frame of that age. ``object_membership`` and
    ``object_membership_by_age`` are Oracle-only supervision used to construct
    selected historical evidence. They are never exposed to a temporal module
    or anomaly head.
    """

    current_coordinates: np.ndarray
    query_offsets: np.ndarray
    object_membership: np.ndarray
    object_membership_by_age: tuple[np.ndarray, ...] | None = None

    def __post_init__(self) -> None:
        count = self.current_coordinates.shape[0]
        if (
            self.current_coordinates.dtype != np.float32
            or self.current_coordinates.shape != (count, 3)
        ):
            raise TypeError("current_coordinates must be float32[N,3]")
        if self.query_offsets.dtype != np.float32 or self.query_offsets.shape != (
            WINDOW_FRAMES,
            count,
            3,
        ):
            raise TypeError("query_offsets must be float32[5,N,3]")
        if self.object_membership.dtype != np.bool_ or self.object_membership.shape != (
            count,
        ):
            raise TypeError("object_membership must be bool[N]")
        membership_by_age = self.object_membership_by_age
        if membership_by_age is None:
            membership_by_age = (
                np.ascontiguousarray(self.object_membership, dtype=np.bool_),
                *(np.empty(0, dtype=np.bool_) for _ in range(WINDOW_FRAMES - 1)),
            )
            object.__setattr__(self, "object_membership_by_age", membership_by_age)
        if len(membership_by_age) != WINDOW_FRAMES:
            raise TypeError("object_membership_by_age must contain ages 0 through 4")
        for age, membership in enumerate(membership_by_age):
            if membership.dtype != np.bool_ or membership.ndim != 1:
                raise TypeError(
                    f"object_membership_by_age[{age}] must be a one-dimensional bool array"
                )
        if not np.array_equal(membership_by_age[0], self.object_membership):
            raise StaticModelError("age-zero object membership differs from current")
        if not (
            np.isfinite(self.current_coordinates).all()
            and np.isfinite(self.query_offsets).all()
        ):
            raise StaticModelError("history sampling offsets contain non-finite values")
        if np.count_nonzero(self.query_offsets[0]):
            raise StaticModelError("current-frame query offsets must be exactly zero")
        self.current_coordinates.setflags(write=False)
        self.query_offsets.setflags(write=False)
        self.object_membership.setflags(write=False)
        for membership in membership_by_age:
            membership.setflags(write=False)

    @classmethod
    def fixed(
        cls,
        current_coordinates: np.ndarray,
        object_membership: np.ndarray | None = None,
        object_membership_by_age: tuple[np.ndarray, ...] | None = None,
    ) -> "HistorySamplingOffsets":
        coordinates = np.ascontiguousarray(current_coordinates, dtype=np.float32)
        membership = (
            np.zeros(coordinates.shape[0], dtype=np.bool_)
            if object_membership is None
            else np.ascontiguousarray(object_membership, dtype=np.bool_)
        )
        return cls(
            current_coordinates=coordinates,
            query_offsets=np.zeros(
                (WINDOW_FRAMES, coordinates.shape[0], 3), dtype=np.float32
            ),
            object_membership=membership,
            object_membership_by_age=object_membership_by_age,
        )

    def fixed_like(self) -> "HistorySamplingOffsets":
        """Keep Oracle object support while removing its motion displacement."""

        return HistorySamplingOffsets.fixed(
            self.current_coordinates,
            self.object_membership,
            self.object_membership_by_age,
        )

    def sham(self) -> "HistorySamplingOffsets":
        """Reverse every nonzero query offset while preserving its magnitude."""

        offsets = np.ascontiguousarray(-self.query_offsets, dtype=np.float32)
        offsets[0] = 0.0
        return HistorySamplingOffsets(
            current_coordinates=np.ascontiguousarray(
                self.current_coordinates, dtype=np.float32
            ),
            query_offsets=offsets,
            object_membership=np.ascontiguousarray(
                self.object_membership, dtype=np.bool_
            ),
            object_membership_by_age=tuple(
                np.ascontiguousarray(value, dtype=np.bool_)
                for value in self.object_membership_by_age or ()
            ),
        )


@dataclass(slots=True)
class _WindowStemFrame:
    """Frozen frame-local STU features retained by the window prototype."""

    frame_id: int
    age: int
    p1: Any | None
    p2: Any | None
    p4: Any
    p8: Any
    p16: Any | None
    p8_decoded: Any | None
    inverse_map: Tensor
    real_slots: Tensor
    raw_coordinates: Tensor
    sparse_input: Any
    rotation: Tensor
    source_to_current: Tensor


@dataclass(frozen=True, slots=True)
class HistoryCandidate:
    """One history feature plus supervision metadata excluded from Q/K/V."""

    feature: Tensor
    valid: Tensor
    age: int
    same_object: Tensor | None = None
    target_weight: Tensor | None = None


@dataclass(frozen=True, slots=True)
class HistoryMatchMass:
    """Aggregation and age-local direct-supervision masses for one sparse scale."""

    same_object: Tensor
    null: Tensor
    has_same_object: Tensor
    target_weight: Tensor
    direct_same_object: Tensor
    direct_null: Tensor
    direct_has_same_object: Tensor
    direct_real_valid: Tensor
    ages: tuple[int, ...]

    def __post_init__(self) -> None:
        count = self.same_object.shape[0]
        age_count = len(self.ages)
        if (
            self.same_object.shape != (count,)
            or self.null.shape != (count,)
            or self.has_same_object.shape != (count,)
            or self.target_weight.shape != (count,)
            or self.direct_same_object.shape != (count, age_count)
            or self.direct_null.shape != (count, age_count)
            or self.direct_has_same_object.shape != (count, age_count)
            or self.direct_real_valid.shape != (count, age_count)
        ):
            raise StaticModelError("history match masses do not align as [N]")
        if (
            self.has_same_object.dtype != torch.bool
            or self.direct_has_same_object.dtype != torch.bool
            or self.direct_real_valid.dtype != torch.bool
        ):
            raise TypeError("history match truth must be boolean")
        if self.ages != tuple(sorted(set(self.ages))) or any(
            not 1 <= age < WINDOW_FRAMES for age in self.ages
        ):
            raise StaticModelError("history match ages are invalid")
        if bool((self.target_weight < 0.0).any()) or bool(
            (self.target_weight > 1.0).any()
        ):
            raise StaticModelError("history match target weights must be in [0,1]")
        for name, value in (
            ("same_object_attention_mass", self.same_object),
            ("null_attention_mass", self.null),
            ("direct_same_object_attention_mass", self.direct_same_object),
            ("direct_null_attention_mass", self.direct_null),
            ("history_match_target_weight", self.target_weight),
        ):
            _finite(name, value)
        for name, value in (
            ("same-object", self.same_object),
            ("null", self.null),
            ("direct same-object", self.direct_same_object),
            ("direct null", self.direct_null),
        ):
            if bool((value < 0.0).any()) or bool((value > 1.0 + 1.0e-5).any()):
                raise StaticModelError(f"{name} attention mass must be in [0,1]")


class _CausalWindowResidual(nn.Module):
    """Read aligned history with one learned-score zero-value null per age."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.age = nn.Embedding(WINDOW_FRAMES, channels)
        nn.init.zeros_(self.age.weight)
        # Values have no affine term, so an empty history tensor cannot become
        # a learned history-presence token. Age conditions matching scores only.
        self.normalization = nn.LayerNorm(channels, elementwise_affine=False)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        current: Tensor,
        history: Sequence[HistoryCandidate | tuple[Tensor, Tensor, int]],
        *,
        return_match_mass: bool = False,
    ) -> (
        tuple[Tensor, Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor, HistoryMatchMass]
    ):
        candidates: list[HistoryCandidate] = []
        for item in history:
            if isinstance(item, HistoryCandidate):
                candidate = item
            else:
                if len(item) != 3:
                    raise StaticModelError("history candidate tuple must have three fields")
                candidate = HistoryCandidate(*item)
            count = current.shape[0]
            if (
                candidate.feature.shape != current.shape
                or candidate.valid.shape != (count,)
                or candidate.valid.dtype != torch.bool
                or type(candidate.age) is not int
            ):
                raise StaticModelError("history candidate does not align with current")
            if (candidate.same_object is None) != (candidate.target_weight is None):
                raise StaticModelError("history match metadata must be complete or absent")
            if candidate.same_object is not None:
                assert candidate.target_weight is not None
                if (
                    candidate.same_object.shape != (count,)
                    or candidate.same_object.dtype != torch.bool
                    or candidate.target_weight.shape != (count,)
                    or not candidate.target_weight.is_floating_point()
                ):
                    raise StaticModelError("history match metadata does not align")
                if bool((candidate.same_object & ~candidate.valid).any()):
                    raise StaticModelError("an invalid candidate cannot be a truth match")
            candidates.append(candidate)
        if not candidates:
            valid = torch.zeros(
                current.shape[0], dtype=torch.bool, device=current.device
            )
            output = (current, torch.zeros_like(current), valid)
            if not return_match_mass:
                return output
            return (
                *output,
                HistoryMatchMass(
                    same_object=current.new_zeros(current.shape[0]),
                    null=current.new_ones(current.shape[0]),
                    has_same_object=valid,
                    target_weight=current.new_zeros(current.shape[0]),
                    direct_same_object=current.new_zeros((current.shape[0], 0)),
                    direct_null=current.new_zeros((current.shape[0], 0)),
                    direct_has_same_object=torch.zeros(
                        (current.shape[0], 0),
                        dtype=torch.bool,
                        device=current.device,
                    ),
                    direct_real_valid=torch.zeros(
                        (current.shape[0], 0),
                        dtype=torch.bool,
                        device=current.device,
                    ),
                    ages=(),
                ),
            )
        ages = tuple(sorted({candidate.age for candidate in candidates}))
        if any(not 1 <= age < WINDOW_FRAMES for age in ages):
            raise StaticModelError("history candidates contain an invalid age")
        metadata_present = [candidate.same_object is not None for candidate in candidates]
        if any(metadata_present) and not all(metadata_present):
            raise StaticModelError("history candidates mix supervised and unsupervised slots")
        target_weight = current.new_zeros(current.shape[0])
        if all(metadata_present):
            first = candidates[0].target_weight
            assert first is not None
            target_weight = first.to(device=current.device, dtype=current.dtype)
            if any(
                candidate.target_weight is None
                or not torch.equal(
                    candidate.target_weight.to(
                        device=current.device, dtype=current.dtype
                    ),
                    target_weight,
                )
                for candidate in candidates[1:]
            ):
                raise StaticModelError("history candidates disagree on target weights")
        features = [candidate.feature for candidate in candidates]
        masks = [candidate.valid for candidate in candidates]
        candidate_ages = [candidate.age for candidate in candidates]
        null = torch.zeros_like(current)
        # Null candidates share the scoring projections but contribute zero value.
        features.extend(null for _ in ages)
        masks.extend(
            torch.ones(current.shape[0], dtype=torch.bool, device=current.device)
            for _ in ages
        )
        candidate_ages.extend(ages)
        normalized = torch.stack(
            [self.normalization(feature) for feature in features], dim=1
        )
        keys = torch.stack(
            [
                value + self.age.weight[age].to(dtype=value.dtype)[None, :]
                for value, age in zip(
                    normalized.unbind(dim=1), candidate_ages, strict=True
                )
            ],
            dim=1,
        )
        valid = torch.stack(masks, dim=1)
        real_valid = torch.stack([candidate.valid for candidate in candidates], dim=1)
        query = self.query(self.normalization(current))
        score = (query[:, None, :] * self.key(keys)).sum(dim=-1)
        score = score / math.sqrt(current.shape[1])
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        any_valid = real_valid.any(dim=1, keepdim=True)
        weight = torch.softmax(score, dim=1)
        context = (weight[..., None] * self.value(normalized)).sum(dim=1)
        residual = self.output(context)
        # A null match is a strict identity path, including after output biases
        # have learned nonzero values on valid history.
        residual = residual * any_valid.to(dtype=residual.dtype)
        output = current + residual, context, any_valid.squeeze(1)
        if not return_match_mass:
            return output
        if all(metadata_present):
            same_object = torch.stack(
                [
                    candidate.same_object.to(device=current.device)
                    for candidate in candidates
                    if candidate.same_object is not None
                ],
                dim=1,
            )
        else:
            same_object = torch.zeros_like(real_valid)
        real_count = len(candidates)
        direct_same_object: list[Tensor] = []
        direct_null: list[Tensor] = []
        direct_has_same_object: list[Tensor] = []
        direct_real_valid: list[Tensor] = []
        # Direct correspondence is an age-local decision. It reuses the exact
        # aggregation scores but does not let visibility at one age suppress
        # the explicit null target at a different age.
        for null_offset, age in enumerate(ages):
            age_real_indices = [
                index
                for index, candidate_age in enumerate(candidate_ages[:real_count])
                if candidate_age == age
            ]
            if not age_real_indices:
                raise StaticModelError("a null candidate lacks real age peers")
            columns = age_real_indices + [real_count + null_offset]
            age_weight = torch.softmax(score[:, columns], dim=1)
            age_truth = same_object[:, age_real_indices]
            direct_same_object.append(
                (age_weight[:, :-1] * age_truth.to(dtype=age_weight.dtype)).sum(
                    dim=1
                )
            )
            direct_null.append(age_weight[:, -1])
            direct_has_same_object.append(age_truth.any(dim=1))
            direct_real_valid.append(real_valid[:, age_real_indices].any(dim=1))
        return (
            *output,
            HistoryMatchMass(
                same_object=(
                    weight[:, :real_count]
                    * same_object.to(dtype=weight.dtype)
                ).sum(dim=1),
                null=weight[:, real_count:].sum(dim=1),
                has_same_object=same_object.any(dim=1),
                target_weight=target_weight,
                direct_same_object=torch.stack(direct_same_object, dim=1),
                direct_null=torch.stack(direct_null, dim=1),
                direct_has_same_object=torch.stack(
                    direct_has_same_object, dim=1
                ),
                direct_real_valid=torch.stack(direct_real_valid, dim=1),
                ages=ages,
            ),
        )


def _official_modules(repository: Path) -> tuple[Any, Any, Any]:
    resolved = repository.expanduser().resolve(strict=True)
    if not (resolved / "models" / "mask4former.py").is_file():
        raise FileNotFoundError("STU repository lacks models/mask4former.py")
    loaded = sys.modules.get("models")
    if loaded is not None:
        location = Path(loaded.__file__).resolve()
        if resolved not in location.parents:
            raise StaticModelError("another top-level models package is already loaded")
    elif str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    try:
        models = importlib.import_module("models")
        me = importlib.import_module("MinkowskiEngine")
        omega = importlib.import_module("omegaconf")
    except ImportError as error:
        raise RuntimeError(
            "AJAE requires the official STU environment with MinkowskiEngine, "
            "Hydra, OmegaConf, and PyTorch3D"
        ) from error
    return models, me, omega


def _build_official_model(repository: Path) -> nn.Module:
    models, _, omega = _official_modules(repository)
    backbone = omega.OmegaConf.create(
        {
            "_target_": "models.Res16UNet34C",
            "config": {
                "dialations": [1, 1, 1, 1],
                "conv1_kernel_size": 5,
                "bn_momentum": 0.02,
            },
            "in_channels": 2,
            "out_channels": NUM_NORMAL_CLASSES,
        }
    )
    return models.Mask4Former3D(
        backbone=backbone,
        num_queries=NUM_QUERIES,
        num_heads=8,
        num_decoders=3,
        num_levels=4,
        sample_sizes=[4000, 8000, 16000, 32000],
        mask_dim=MASK_DIM,
        dim_feedforward=1024,
        num_labels=NUM_NORMAL_CLASSES,
    )


def load_stu_weights(model: nn.Module, checkpoint: Path) -> None:
    """Load the safe tensor extraction and bind it to the official checkpoint."""

    state, _ = _validated_stu_weights(checkpoint)
    model.load_state_dict(state, strict=True)


class CurrentPointPrediction:
    """Frozen spatial features and the trainable single-frame anomaly baseline."""

    def __init__(self, *, features: Tensor, logits: Tensor) -> None:
        count = features.shape[0]
        if features.shape != (count, MASK_DIM) or logits.shape != (count,):
            raise StaticModelError("current point prediction has invalid shapes")
        _finite("current_point_features", features)
        _finite("current_point_logits", logits)
        self.features = features
        self.logits = logits


class HistoryPointPrediction:
    """One history-conditioned correction of a shared current prediction."""

    def __init__(
        self,
        *,
        logits: Tensor,
        correction: Tensor,
        point_history_support: Tensor,
        history_coverage: Tensor,
        scale_residuals: tuple[Tensor, Tensor, Tensor] | None = None,
        match_mass_by_scale: tuple[
            HistoryMatchMass, HistoryMatchMass, HistoryMatchMass
        ]
        | None = None,
    ) -> None:
        count = logits.shape[0]
        expected = {
            "correction": (count,),
            "point_history_support": (count,),
            "history_coverage": (3,),
        }
        for name, shape in expected.items():
            value = locals()[name]
            if value.shape != shape:
                raise StaticModelError(f"{name} has shape {tuple(value.shape)}")
            _finite(name, value)
        if point_history_support.dtype != torch.bool:
            raise TypeError("point history support must be boolean")
        _finite("history_point_logits", logits)
        self.logits = logits
        self.correction = correction
        self.point_history_support = point_history_support
        self.history_coverage = history_coverage
        if scale_residuals is not None:
            if len(scale_residuals) != 3:
                raise StaticModelError("scale residual audit requires p16, p8, and p4")
            for value in scale_residuals:
                if value.ndim != 2:
                    raise StaticModelError("scale residual audit tensor must be [N,C]")
                _finite("scale_history_residual", value)
        self.scale_residuals = scale_residuals
        if match_mass_by_scale is not None and len(match_mass_by_scale) != 3:
            raise StaticModelError("match supervision requires p16, p8, and p4 masses")
        if match_mass_by_scale is not None and scale_residuals is not None:
            for residual, mass in zip(
                scale_residuals, match_mass_by_scale, strict=True
            ):
                if mass.same_object.shape[0] != residual.shape[0]:
                    raise StaticModelError(
                        "match supervision does not align with its sparse scale"
                    )
        self.match_mass_by_scale = match_mass_by_scale


class WindowDetectorPrototype(nn.Module):
    """Shared 3D STU stem with current-anchored factorized window updates."""

    def __init__(
        self,
        *,
        checkpoint: Path | str = PROJECT_ROOT / "weights" / "59p6pq_ens1.ckpt",
        official_repository: Path | str = DEFAULT_STU_REPOSITORY,
    ) -> None:
        super().__init__()
        self.official_repository = (
            Path(official_repository).expanduser().resolve(strict=True)
        )
        self.stu = _build_official_model(self.official_repository)
        load_stu_weights(self.stu, Path(checkpoint))
        for parameter in self.stu.parameters():
            parameter.requires_grad_(False)

        backbone = self.stu.backbone
        channels = tuple(backbone.PLANES)
        if channels != (32, 64, 128, 256, 256, 128, 96, 96):
            raise StaticModelError(f"unexpected STU backbone widths: {channels}")
        self.temporal_p16 = _CausalWindowResidual(256)
        self.temporal_p8 = _CausalWindowResidual(256)
        # Step-8 occupancy loses many 1--5 point anomalies, so only the
        # current p4 skip receives this small supplemental history residual.
        self.temporal_p4 = _CausalWindowResidual(64)
        self.point_anomaly_head = nn.Sequential(
            nn.Linear(MASK_DIM, MASK_DIM),
            nn.ReLU(),
            nn.Linear(MASK_DIM, 1),
        )
        # A bias-free readout keeps zero history on the exact identity path while
        # a nonzero derivative so zero-initialized sparse residuals can learn.
        self.temporal_point_delta = nn.Linear(MASK_DIM, 1, bias=False)

    @classmethod
    def from_protocol(
        cls,
        protocol: AJAEProtocol,
        *,
        project_root: Path | str = PROJECT_ROOT,
        official_repository: Path | str = DEFAULT_STU_REPOSITORY,
    ) -> WindowDetectorPrototype:
        if not isinstance(protocol, AJAEProtocol):
            raise TypeError("protocol must be AJAEProtocol")
        return cls(
            checkpoint=protocol.checkpoint_path(project_root),
            official_repository=official_repository,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def train(self, mode: bool = True) -> WindowDetectorPrototype:
        super().train(mode)
        # Frozen running statistics are required for exact zero-time fallback.
        self.stu.eval()
        return self

    def new_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("stu."):
                yield parameter

    def current_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the single-frame point baseline only."""

        yield from self.point_anomaly_head.parameters()

    def temporal_parameters(self) -> Iterator[nn.Parameter]:
        """History-only parameters used by the Oracle upper-bound experiment."""

        for module in (
            self.temporal_p16,
            self.temporal_p8,
            self.temporal_p4,
            self.temporal_point_delta,
        ):
            yield from module.parameters()

    @staticmethod
    def _require_map(value: Any | None, name: str) -> Any:
        if value is None:
            raise StaticModelError(f"current STU stem lacks {name}")
        return value

    def _voxelize_stem(self, frame: FrameInput) -> _WindowStemFrame:
        """Run the shared official spatial stem through block3."""

        _, me, _ = _official_modules(self.official_repository)
        coordinates, features, unique, inverse = me.utils.sparse_quantize(
            coordinates=frame.coordinates,
            features=frame.features,
            return_index=True,
            return_inverse=True,
            quantization_size=VOXEL_SIZE_METRES,
        )
        if not isinstance(features, Tensor):
            features = torch.from_numpy(features)
        coordinates, features = me.utils.sparse_collate(
            [coordinates], [features.float()]
        )
        sparse = me.SparseTensor(
            coordinates=coordinates,
            features=features,
            device=self.device,
        )
        backbone = self.stu.backbone
        with torch.no_grad():
            out = backbone.relu(backbone.bn0(backbone.conv0p1s1(sparse)))
            p1 = out
            out = backbone.relu(backbone.bn1(backbone.conv1p1s2(out)))
            p2 = backbone.block1(out)
            out = backbone.relu(backbone.bn2(backbone.conv2p2s2(p2)))
            p4 = backbone.block2(out)
            out = backbone.relu(backbone.bn3(backbone.conv3p4s2(p4)))
            p8 = backbone.block3(out)

        unique_np = (
            unique.detach().cpu().numpy() if isinstance(unique, Tensor) else unique
        )
        raw = np.column_stack(
            (
                frame.coordinates[np.asarray(unique_np, dtype=np.int64)],
                np.zeros(len(unique_np), dtype=np.float64),
            )
        )
        return _WindowStemFrame(
            frame_id=frame.frame_id,
            age=frame.age,
            p1=p1 if frame.age == 0 else None,
            p2=p2 if frame.age == 0 else None,
            p4=p4,
            p8=p8,
            p16=None,
            p8_decoded=None,
            inverse_map=torch.as_tensor(inverse, dtype=torch.long, device=self.device),
            real_slots=torch.from_numpy(frame.real_slots.astype(np.int64)).to(
                self.device
            ),
            raw_coordinates=torch.from_numpy(raw).float().to(self.device),
            sparse_input=sparse,
            rotation=_as_tensor(frame.lidar_pose[:3, :3], self.device).float(),
            source_to_current=_as_tensor(frame.source_to_current, self.device).float(),
        )

    def _spatial_mid(self, frame: _WindowStemFrame) -> None:
        """Compute frozen per-frame spatial block4 and block5 features."""

        _, me, _ = _official_modules(self.official_repository)
        backbone = self.stu.backbone
        with torch.no_grad():
            out = backbone.relu(backbone.bn4(backbone.conv4p8s2(frame.p8)))
            frame.p16 = backbone.block4(out)
            out = backbone.relu(backbone.bntr4(backbone.convtr4p16s2(frame.p16)))
            frame.p8_decoded = backbone.block5(me.cat(out, frame.p8))

    def _source_query_coordinates(
        self,
        current_map: Any,
        current: _WindowStemFrame,
        source: _WindowStemFrame,
        query_offset: Tensor | None = None,
    ) -> Tensor:
        """Map current STU voxel centers into a historical STU coordinate frame."""

        official_current = current_map.C[:, 1:].to(dtype=current_map.F.dtype)
        official_current = official_current * VOXEL_SIZE_METRES
        current_sensor = official_current @ current.rotation.T
        if query_offset is not None:
            if query_offset.shape != current_sensor.shape:
                raise StaticModelError("history query offsets do not align")
            current_sensor = current_sensor + query_offset.to(
                device=current_sensor.device, dtype=current_sensor.dtype
            )
        transform = source.source_to_current
        source_sensor = (current_sensor - transform[:3, 3]) @ transform[:3, :3]
        source_official = source_sensor @ source.rotation
        batch = source_official.new_zeros((source_official.shape[0], 1))
        return torch.cat((batch, source_official / VOXEL_SIZE_METRES), dim=1)

    @staticmethod
    def _object_query_hypothesis(
        query_coordinates: Tensor,
        sampling: HistorySamplingOffsets,
        age: int,
    ) -> tuple[Tensor, Tensor]:
        """Expose one object-motion candidate at every sparse query.

        Generated-object membership recovers the known trajectory displacement
        but never controls candidate presence.  Static and object hypotheses
        therefore have identical interfaces on normal and generated points.
        """

        count = query_coordinates.shape[0]
        offsets = np.zeros((count, 3), dtype=np.float32)
        object_indices = np.flatnonzero(sampling.object_membership)
        if count == 0 or object_indices.size == 0:
            return (
                torch.from_numpy(offsets).to(query_coordinates.device),
                torch.zeros(count, dtype=torch.bool, device=query_coordinates.device),
            )
        object_offsets = sampling.query_offsets[age, object_indices]
        if not np.allclose(object_offsets, object_offsets[0], rtol=0.0, atol=1.0e-6):
            raise StaticModelError("one Oracle object received inconsistent offsets")
        offsets[:] = object_offsets[0]
        return (
            torch.from_numpy(offsets).to(query_coordinates.device),
            torch.ones(count, dtype=torch.bool, device=query_coordinates.device),
        )

    @staticmethod
    def _scale_object_fraction(
        sparse_map: Any,
        frame: _WindowStemFrame,
        object_membership: np.ndarray,
    ) -> tuple[Tensor, Tensor]:
        """Compute the exact generated-object fraction for each sparse voxel."""

        count = int(frame.real_slots.numel())
        if object_membership.dtype != np.bool_ or object_membership.shape != (count,):
            raise StaticModelError(
                "scale object membership does not align with returns"
            )
        input_rows = frame.inverse_map[frame.real_slots]
        point_coordinates = (
            frame.sparse_input.C[input_rows, 1:]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        query_coordinates = (
            sparse_map.C[:, 1:].detach().cpu().numpy().astype(np.int64, copy=False)
        )
        stride = np.asarray(tuple(int(value) for value in sparse_map.tensor_stride))
        if stride.shape != (3,) or np.any(stride < 1):
            raise StaticModelError("sparse map has an invalid three-dimensional stride")
        point_keys = np.floor_divide(point_coordinates, stride[None, :])
        query_keys = np.floor_divide(query_coordinates, stride[None, :])
        unique, inverse = np.unique(point_keys, axis=0, return_inverse=True)
        totals = np.bincount(inverse, minlength=unique.shape[0])
        positives = np.bincount(
            inverse,
            weights=object_membership.astype(np.float64),
            minlength=unique.shape[0],
        )
        lookup = {tuple(key): index for index, key in enumerate(unique.tolist())}
        query_index = np.fromiter(
            (lookup.get(tuple(key), -1) for key in query_keys.tolist()),
            dtype=np.int64,
            count=query_keys.shape[0],
        )
        defined_np = query_index >= 0
        fraction_np = np.zeros(query_keys.shape[0], dtype=np.float32)
        fraction_np[defined_np] = (
            positives[query_index[defined_np]] / totals[query_index[defined_np]]
        ).astype(np.float32)
        fraction = torch.from_numpy(fraction_np).to(
            device=sparse_map.F.device, dtype=sparse_map.F.dtype
        )
        defined = torch.from_numpy(defined_np).to(device=sparse_map.F.device)
        return fraction, defined

    @staticmethod
    def _interpolated_object_evidence(
        source_present: Tensor,
        input_index: Tensor,
        output_index: Tensor,
        output_count: int,
    ) -> Tensor:
        """Mark queries whose actual interpolation edges touch object evidence."""

        evidence_count = torch.zeros(
            output_count, dtype=torch.int32, device=source_present.device
        )
        if input_index.numel():
            evidence_count.index_add_(
                0,
                output_index.long(),
                source_present[input_index.long()].to(dtype=torch.int32),
            )
        return evidence_count > 0

    @staticmethod
    def _oracle_mix(
        static_feature: Tensor,
        static_valid: Tensor,
        object_feature: Tensor,
        object_evidence_valid: Tensor,
        current_fraction: Tensor,
        current_defined: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the frozen Oracle-Mix formula without renormalizing absence."""

        if (
            static_feature.shape != object_feature.shape
            or static_valid.shape != (static_feature.shape[0],)
            or object_evidence_valid.shape != static_valid.shape
            or current_fraction.shape != static_valid.shape
            or current_defined.shape != static_valid.shape
        ):
            raise StaticModelError("Oracle-Mix inputs do not align")
        static_mass = (1.0 - current_fraction).clamp(0.0, 1.0)
        object_mass = current_fraction.clamp(0.0, 1.0)
        static_used = current_defined & (static_mass > 0.0) & static_valid
        object_used = current_defined & (object_mass > 0.0) & object_evidence_valid
        mixed = (
            static_mass[:, None]
            * static_used[:, None].to(dtype=static_feature.dtype)
            * static_feature
            + object_mass[:, None]
            * object_used[:, None].to(dtype=object_feature.dtype)
            * object_feature
        )
        return mixed, static_used | object_used

    @staticmethod
    def _history_content_control(
        feature: Tensor,
        valid: Tensor,
        control: str,
        *,
        age: int,
        hypothesis: int,
    ) -> Tensor:
        """Apply a validation-only content intervention without changing support."""

        if control == "actual":
            return feature
        if control == "null":
            return torch.zeros_like(feature)
        if control != "shuffle":
            raise ValueError("history content control must be actual, null, or shuffle")
        output = feature.clone()
        indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if indices.numel() < 2:
            output[indices] = 0.0
            return output
        shift = 1 + (age + hypothesis) % (indices.numel() - 1)
        output[indices] = feature[torch.roll(indices, shifts=int(shift))]
        return output

    def _aligned_history(
        self,
        current_map: Any,
        current: _WindowStemFrame,
        history: Sequence[_WindowStemFrame],
        attribute: str,
        sampling: HistorySamplingOffsets,
        *,
        content_control: str = "actual",
        hypothesis_order: str = "static_object",
        correspondence: str = "oracle_proposal",
    ) -> list[HistoryCandidate]:
        if hypothesis_order not in {"static_object", "object_static"}:
            raise ValueError("history hypothesis order is invalid")
        if correspondence not in {"oracle_select", "oracle_proposal"}:
            raise ValueError("history correspondence must be Oracle Select or Proposal")
        _, me, _ = _official_modules(self.official_repository)
        interpolation = me.MinkowskiInterpolation(return_kernel_map=True)
        aligned: list[HistoryCandidate] = []
        current_fraction, current_defined = self._scale_object_fraction(
            current_map, current, sampling.object_membership
        )
        target_weight = torch.where(
            current_defined, current_fraction, torch.zeros_like(current_fraction)
        )
        memberships = sampling.object_membership_by_age
        if memberships is None:
            raise StaticModelError("history correspondence lacks object identity")

        for source in history:
            source_map = self._require_map(getattr(source, attribute), attribute)
            query = self._source_query_coordinates(current_map, current, source)
            query = query.to(dtype=source_map.F.dtype)
            feature, (static_input, static_output) = interpolation(source_map, query)
            valid = torch.zeros(query.shape[0], dtype=torch.bool, device=self.device)
            valid[static_output.long()] = True

            object_feature = feature
            object_valid = valid
            object_input = torch.empty(0, dtype=torch.long, device=self.device)
            object_output = torch.empty(0, dtype=torch.long, device=self.device)
            official = current_map.C[:, 1:].to(dtype=current_map.F.dtype)
            current_sensor = official * VOXEL_SIZE_METRES @ current.rotation.T
            if bool(sampling.object_membership.any()):
                object_offset, object_support = self._object_query_hypothesis(
                    current_sensor, sampling, source.age
                )
                object_query = self._source_query_coordinates(
                    current_map, current, source, object_offset
                ).to(dtype=source_map.F.dtype)
                object_feature, (object_input, object_output) = interpolation(
                    source_map, object_query
                )
                object_valid = torch.zeros(
                    object_query.shape[0], dtype=torch.bool, device=self.device
                )
                object_valid[object_output.long()] = True
                object_valid &= object_support

            source_fraction, source_defined = self._scale_object_fraction(
                source_map, source, memberships[source.age]
            )
            source_present = source_defined & (source_fraction > 0.0)
            object_evidence = self._interpolated_object_evidence(
                source_present,
                object_input,
                object_output,
                object_feature.shape[0],
            )

            if correspondence == "oracle_select":
                mixed, mixed_valid = self._oracle_mix(
                    feature,
                    valid,
                    object_feature,
                    object_valid & object_evidence,
                    current_fraction,
                    current_defined,
                )
                controlled = self._history_content_control(
                    mixed,
                    mixed_valid,
                    content_control,
                    age=source.age,
                    hypothesis=0,
                )
                if content_control == "null":
                    mixed_valid = torch.zeros_like(mixed_valid)
                candidates = (
                    HistoryCandidate(controlled, mixed_valid, source.age),
                )
            else:
                static_evidence = self._interpolated_object_evidence(
                    source_present,
                    static_input,
                    static_output,
                    feature.shape[0],
                )
                static_valid = valid
                proposal_object_valid = object_valid
                static_feature = self._history_content_control(
                    feature,
                    static_valid,
                    content_control,
                    age=source.age,
                    hypothesis=0,
                )
                proposal_object_feature = self._history_content_control(
                    object_feature,
                    proposal_object_valid,
                    content_control,
                    age=source.age,
                    hypothesis=1,
                )
                if content_control == "null":
                    static_valid = torch.zeros_like(static_valid)
                    proposal_object_valid = torch.zeros_like(proposal_object_valid)
                static_same_object = static_valid & static_evidence
                object_same_object = proposal_object_valid & object_evidence
                candidates = (
                    HistoryCandidate(
                        static_feature,
                        static_valid,
                        source.age,
                        static_same_object,
                        target_weight,
                    ),
                    HistoryCandidate(
                        proposal_object_feature,
                        proposal_object_valid,
                        source.age,
                        object_same_object,
                        target_weight,
                    ),
                )
            if hypothesis_order == "object_static":
                candidates = tuple(reversed(candidates))
            aligned.extend(candidates)
        return aligned

    def _sparse_like(self, sparse_map: Any, features: Tensor) -> Any:
        _, me, _ = _official_modules(self.official_repository)
        return me.SparseTensor(
            features=features,
            coordinate_map_key=sparse_map.coordinate_map_key,
            coordinate_manager=sparse_map.coordinate_manager,
        )

    def _decode_p8(self, p16: Any, p8_skip: Any) -> Any:
        _, me, _ = _official_modules(self.official_repository)
        backbone = self.stu.backbone
        out = backbone.relu(backbone.bntr4(backbone.convtr4p16s2(p16)))
        return backbone.block5(me.cat(out, p8_skip))

    def _decode_current_high(
        self,
        p8: Any,
        p4_skip: Any,
        p2_skip: Any,
        p1_skip: Any,
    ) -> tuple[Any, Any, Any]:
        """Run block6--block8 only for the current frame."""

        _, me, _ = _official_modules(self.official_repository)
        backbone = self.stu.backbone
        out = backbone.relu(backbone.bntr5(backbone.convtr5p8s2(p8)))
        p4 = backbone.block6(me.cat(out, p4_skip))
        out = backbone.relu(backbone.bntr6(backbone.convtr6p4s2(p4)))
        p2 = backbone.block7(me.cat(out, p2_skip))
        out = backbone.relu(backbone.bntr7(backbone.convtr7p2s2(p2)))
        p1 = backbone.block8(me.cat(out, p1_skip))
        return p4, p2, p1

    @staticmethod
    def _validate_sampling(
        inputs: StaticInput, sampling: HistorySamplingOffsets
    ) -> None:
        """Require Oracle sampling identities to match this exact five-frame window."""

        if not isinstance(inputs, StaticInput):
            raise TypeError("sampling validation requires StaticInput")
        if not isinstance(sampling, HistorySamplingOffsets):
            raise TypeError("sampling validation requires HistorySamplingOffsets")
        current_count = inputs.current_count
        expected_coordinates = inputs.member_coordinates[inputs.current_slice]
        if (
            sampling.current_coordinates.dtype != np.float32
            or sampling.current_coordinates.shape != (current_count, 3)
            or not np.isfinite(sampling.current_coordinates).all()
            or not np.array_equal(sampling.current_coordinates, expected_coordinates)
        ):
            raise StaticModelError(
                "history sampling coordinates do not match current real points"
            )
        if (
            sampling.query_offsets.dtype != np.float32
            or sampling.query_offsets.shape != (WINDOW_FRAMES, current_count, 3)
            or not np.isfinite(sampling.query_offsets).all()
        ):
            raise StaticModelError(
                "history sampling offsets do not align with all five frame ages"
            )
        if np.count_nonzero(sampling.query_offsets[0]):
            raise StaticModelError("current-frame sampling offsets must be zero")
        if (
            sampling.object_membership.dtype != np.bool_
            or sampling.object_membership.shape != (current_count,)
        ):
            raise StaticModelError(
                "current object identity does not align with current real points"
            )

        memberships = sampling.object_membership_by_age
        if memberships is None or len(memberships) != WINDOW_FRAMES:
            raise StaticModelError(
                "history sampling requires one object identity vector per frame age"
            )
        frame_count_by_age: dict[int, int] = {}
        for frame in inputs.frames:
            if frame.age in frame_count_by_age:
                raise StaticModelError("static input contains a duplicate frame age")
            frame_count_by_age[frame.age] = int(frame.real_slots.size)
        for age, membership in enumerate(memberships):
            expected_count = frame_count_by_age.get(age, 0)
            if membership.dtype != np.bool_ or membership.shape != (expected_count,):
                raise StaticModelError(
                    f"age-{age} object identity does not align with its real returns"
                )
        if not np.array_equal(memberships[0], sampling.object_membership):
            raise StaticModelError("age-zero object identity differs from current")

    def _current_point_path(self, current: _WindowStemFrame) -> CurrentPointPrediction:
        p8 = self._require_map(current.p8_decoded, "p8_decoded")
        p1_skip = self._require_map(current.p1, "p1")
        p2_skip = self._require_map(current.p2, "p2")
        _, _, p1 = self._decode_current_high(p8, current.p4, p2_skip, p1_skip)
        point_map = self.stu.point_features_head(p1)
        features = point_map.F[current.inverse_map[current.real_slots]]
        logits = self.point_anomaly_head(features).squeeze(1)
        return CurrentPointPrediction(features=features, logits=logits)

    def forward_current(self, inputs: StaticInput) -> CurrentPointPrediction:
        """Evaluate only the current spatial path, irrespective of supplied history."""

        if not isinstance(inputs, StaticInput):
            raise TypeError("forward_current accepts only StaticInput")
        current = self._voxelize_stem(inputs.frames[-1])
        self._spatial_mid(current)
        return self._current_point_path(current)

    def _history_point_branch(
        self,
        current: _WindowStemFrame,
        history: Sequence[_WindowStemFrame],
        sampling: HistorySamplingOffsets,
        current_prediction: CurrentPointPrediction,
        *,
        use_p4: bool,
        content_control: str = "actual",
        hypothesis_order: str = "static_object",
        correspondence: str = "oracle_proposal",
        aligned_by_scale: Mapping[
            str, Sequence[HistoryCandidate | tuple[Tensor, Tensor, int]]
        ]
        | None = None,
        point_to_p4: Tensor | None = None,
    ) -> HistoryPointPrediction:
        p16_base = self._require_map(current.p16, "p16")
        p8_base = self._require_map(current.p8_decoded, "p8_decoded")
        if aligned_by_scale is None:
            aligned_p16 = self._aligned_history(
                p16_base,
                current,
                history,
                "p16",
                sampling,
                content_control=content_control,
                hypothesis_order=hypothesis_order,
                correspondence=correspondence,
            )
        else:
            aligned_p16 = list(aligned_by_scale["p16"])
        p16_feature, _, valid_p16, match_p16 = self.temporal_p16(
            p16_base.F, aligned_p16, return_match_mass=True
        )
        p16 = self._sparse_like(p16_base, p16_feature)

        p8_current = self._decode_p8(p16, current.p8)
        if aligned_by_scale is None:
            aligned_p8 = self._aligned_history(
                p8_base,
                current,
                history,
                "p8_decoded",
                sampling,
                content_control=content_control,
                hypothesis_order=hypothesis_order,
                correspondence=correspondence,
            )
        else:
            aligned_p8 = list(aligned_by_scale["p8"])
        p8_feature, _, valid_p8, match_p8 = self.temporal_p8(
            p8_current.F, aligned_p8, return_match_mass=True
        )
        p8 = self._sparse_like(p8_current, p8_feature)

        if use_p4:
            if aligned_by_scale is None:
                aligned_p4 = self._aligned_history(
                    current.p4,
                    current,
                    history,
                    "p4",
                    sampling,
                    content_control=content_control,
                    hypothesis_order=hypothesis_order,
                    correspondence=correspondence,
                )
            else:
                aligned_p4 = list(aligned_by_scale["p4"])
            p4_feature, _, valid_p4, match_p4 = self.temporal_p4(
                current.p4.F, aligned_p4, return_match_mass=True
            )
            p4_skip = self._sparse_like(current.p4, p4_feature)
        else:
            p4_feature = current.p4.F
            valid_p4 = torch.zeros(
                current.p4.F.shape[0], dtype=torch.bool, device=self.device
            )
            p4_skip = current.p4
            match_p4 = HistoryMatchMass(
                same_object=current.p4.F.new_zeros(current.p4.F.shape[0]),
                null=current.p4.F.new_ones(current.p4.F.shape[0]),
                has_same_object=valid_p4,
                target_weight=current.p4.F.new_zeros(current.p4.F.shape[0]),
                direct_same_object=current.p4.F.new_zeros(
                    (current.p4.F.shape[0], 0)
                ),
                direct_null=current.p4.F.new_zeros((current.p4.F.shape[0], 0)),
                direct_has_same_object=torch.zeros(
                    (current.p4.F.shape[0], 0),
                    dtype=torch.bool,
                    device=self.device,
                ),
                direct_real_valid=torch.zeros(
                    (current.p4.F.shape[0], 0),
                    dtype=torch.bool,
                    device=self.device,
                ),
                ages=(),
            )

        p1_skip = self._require_map(current.p1, "p1")
        p2_skip = self._require_map(current.p2, "p2")
        _, _, p1 = self._decode_current_high(p8, p4_skip, p2_skip, p1_skip)
        point_map = self.stu.point_features_head(p1)
        features = point_map.F[current.inverse_map[current.real_slots]]
        history_difference = features - current_prediction.features
        delta = MAXIMUM_TEMPORAL_LOGIT_CORRECTION * torch.tanh(
            self.temporal_point_delta(history_difference).squeeze(1)
        )
        if point_to_p4 is None:
            official_p4 = current.p4.C[:, 1:].to(dtype=current.p4.F.dtype)
            p4_sensor = official_p4 * VOXEL_SIZE_METRES @ current.rotation.T
            nearest_p4 = cKDTree(
                p4_sensor.detach().float().cpu().numpy().astype(np.float64)
            ).query(sampling.current_coordinates.astype(np.float64), k=1)[1]
            point_to_p4 = torch.as_tensor(
                np.asarray(nearest_p4, dtype=np.int64),
                dtype=torch.long,
                device=self.device,
            )
        elif point_to_p4.shape != (sampling.current_coordinates.shape[0],):
            raise StaticModelError("cached p4 point map does not align")
        point_support = valid_p4[point_to_p4.to(device=self.device, dtype=torch.long)]
        correction = point_support.to(dtype=delta.dtype) * delta
        coverage = torch.stack(
            (
                valid_p16.float().mean(),
                valid_p8.float().mean(),
                valid_p4.float().mean(),
            )
        )
        return HistoryPointPrediction(
            logits=current_prediction.logits + correction,
            correction=correction,
            point_history_support=point_support,
            history_coverage=coverage,
            scale_residuals=(
                p16_feature - p16_base.F,
                p8_feature - p8_current.F,
                p4_feature - current.p4.F,
            ),
            match_mass_by_scale=(match_p16, match_p8, match_p4),
        )

    def _prepare_point_history(
        self, inputs: StaticInput
    ) -> tuple[_WindowStemFrame, tuple[_WindowStemFrame, ...], CurrentPointPrediction]:
        current = self._voxelize_stem(inputs.frames[-1])
        self._spatial_mid(current)
        current_prediction = self._current_point_path(current)
        history = tuple(self._voxelize_stem(frame) for frame in inputs.frames[:-1])
        for frame in history:
            self._spatial_mid(frame)
        return current, history, current_prediction

    @staticmethod
    def _sparse_cache_payload(sparse_map: Any) -> dict[str, object]:
        return {
            "coordinates": sparse_map.C.detach().to(device="cpu", dtype=torch.int32),
            "features": sparse_map.F.detach().to(device="cpu", dtype=torch.float32),
            "tensor_stride": [int(value) for value in sparse_map.tensor_stride],
        }

    def prepare_frozen_history_cache(
        self,
        inputs: StaticInput,
        sampling: HistorySamplingOffsets,
        *,
        include_validation_controls: bool,
    ) -> dict[str, object]:
        """Compile frozen STU maps and aligned candidates into plain FP32 tensors."""

        self._validate_sampling(inputs, sampling)
        current, history, current_prediction = self._prepare_point_history(inputs)
        routes: dict[str, tuple[HistorySamplingOffsets, str]] = {
            "oracle_select": (sampling, "oracle_select"),
            "oracle_proposal": (sampling, "oracle_proposal"),
        }
        if include_validation_controls:
            routes.update(
                {
                    "fixed_select": (sampling.fixed_like(), "oracle_select"),
                    "sham_select": (sampling.sham(), "oracle_select"),
                }
            )
        scale_specs = (
            ("p16", self._require_map(current.p16, "p16"), "p16"),
            (
                "p8",
                self._require_map(current.p8_decoded, "p8_decoded"),
                "p8_decoded",
            ),
            ("p4", current.p4, "p4"),
        )
        candidates: dict[str, object] = {}
        for route_name, (route_sampling, correspondence) in routes.items():
            route_scales: dict[str, object] = {}
            for scale, current_map, attribute in scale_specs:
                aligned = self._aligned_history(
                    current_map,
                    current,
                    history,
                    attribute,
                    route_sampling,
                    correspondence=correspondence,
                )
                records: list[dict[str, object]] = []
                for candidate in aligned:
                    record: dict[str, object] = {
                        "feature": candidate.feature.detach().to(
                            device="cpu", dtype=torch.float32
                        ),
                        "valid": candidate.valid.detach().to(
                            device="cpu", dtype=torch.bool
                        ),
                        "age": candidate.age,
                    }
                    if candidate.same_object is not None:
                        assert candidate.target_weight is not None
                        record["same_object"] = candidate.same_object.detach().to(
                            device="cpu", dtype=torch.bool
                        )
                        record["target_weight"] = candidate.target_weight.detach().to(
                            device="cpu", dtype=torch.float32
                        )
                    records.append(record)
                route_scales[scale] = records
            candidates[route_name] = route_scales
        official_p4 = current.p4.C[:, 1:].to(dtype=current.p4.F.dtype)
        p4_sensor = official_p4 * VOXEL_SIZE_METRES @ current.rotation.T
        point_to_p4 = cKDTree(
            p4_sensor.detach().float().cpu().numpy().astype(np.float64)
        ).query(sampling.current_coordinates.astype(np.float64), k=1)[1]
        return {
            "format": "ajae-frozen-history-cache-v3",
            "current": {
                "frame_id": current.frame_id,
                "maps": {
                    "p1": self._sparse_cache_payload(
                        self._require_map(current.p1, "p1")
                    ),
                    "p2": self._sparse_cache_payload(
                        self._require_map(current.p2, "p2")
                    ),
                    "p4": self._sparse_cache_payload(current.p4),
                    "p8": self._sparse_cache_payload(current.p8),
                    "p16": self._sparse_cache_payload(
                        self._require_map(current.p16, "p16")
                    ),
                    "p8_decoded": self._sparse_cache_payload(
                        self._require_map(current.p8_decoded, "p8_decoded")
                    ),
                },
                "inverse_map": current.inverse_map.detach().cpu(),
                "real_slots": current.real_slots.detach().cpu(),
                "rotation": current.rotation.detach().float().cpu(),
                "source_to_current": current.source_to_current.detach().float().cpu(),
                "point_features": current_prediction.features.detach().float().cpu(),
                "point_to_p4": torch.as_tensor(
                    np.asarray(point_to_p4, dtype=np.int64), dtype=torch.long
                ),
                "current_coordinates": torch.from_numpy(
                    np.array(sampling.current_coordinates, copy=True)
                ),
            },
            "candidates": candidates,
        }

    def _hydrate_sparse_map(
        self,
        payload: Mapping[str, object],
        coordinate_manager: Any,
    ) -> Any:
        _, me, _ = _official_modules(self.official_repository)
        coordinates = payload.get("coordinates")
        features = payload.get("features")
        stride = payload.get("tensor_stride")
        if (
            not isinstance(coordinates, Tensor)
            or coordinates.dtype != torch.int32
            or coordinates.ndim != 2
            or coordinates.shape[1] != 4
            or not isinstance(features, Tensor)
            or features.dtype != torch.float32
            or features.shape[0] != coordinates.shape[0]
            or not isinstance(stride, list)
            or len(stride) != 3
        ):
            raise StaticModelError("cached sparse map is malformed")
        return me.SparseTensor(
            features=features.to(self.device),
            coordinates=coordinates.to(self.device),
            tensor_stride=[int(value) for value in stride],
            coordinate_manager=coordinate_manager,
            device=self.device,
        )

    def _hydrate_frozen_history_cache(
        self, payload: Mapping[str, object]
    ) -> tuple[_WindowStemFrame, CurrentPointPrediction, Tensor]:
        if payload.get("format") != "ajae-frozen-history-cache-v3":
            raise StaticModelError("frozen history cache format changed")
        current_payload = payload.get("current")
        if not isinstance(current_payload, Mapping):
            raise StaticModelError("frozen history cache lacks current maps")
        maps = current_payload.get("maps")
        if not isinstance(maps, Mapping):
            raise StaticModelError("frozen history cache map table is malformed")
        _, me, _ = _official_modules(self.official_repository)
        manager = me.CoordinateManager(
            D=3, coordinate_map_type=me.CoordinateMapType.CUDA
        )
        hydrated = {
            name: self._hydrate_sparse_map(value, manager)
            for name, value in maps.items()
            if isinstance(value, Mapping)
        }
        if set(hydrated) != {"p1", "p2", "p4", "p8", "p16", "p8_decoded"}:
            raise StaticModelError("frozen history cache does not contain all maps")

        def tensor(name: str, dtype: torch.dtype) -> Tensor:
            value = current_payload.get(name)
            if not isinstance(value, Tensor) or value.dtype != dtype:
                raise StaticModelError(f"frozen history cache lacks {name}")
            return value.to(self.device)

        inverse_map = tensor("inverse_map", torch.long)
        real_slots = tensor("real_slots", torch.long)
        features = tensor("point_features", torch.float32)
        current = _WindowStemFrame(
            frame_id=int(current_payload["frame_id"]),
            age=0,
            p1=hydrated["p1"],
            p2=hydrated["p2"],
            p4=hydrated["p4"],
            p8=hydrated["p8"],
            p16=hydrated["p16"],
            p8_decoded=hydrated["p8_decoded"],
            inverse_map=inverse_map,
            real_slots=real_slots,
            raw_coordinates=torch.empty(0, 4, device=self.device),
            sparse_input=hydrated["p1"],
            rotation=tensor("rotation", torch.float32),
            source_to_current=tensor("source_to_current", torch.float32),
        )
        prediction = CurrentPointPrediction(
            features=features,
            logits=self.point_anomaly_head(features).squeeze(1),
        )
        return current, prediction, tensor("point_to_p4", torch.long)

    def _cached_candidates(
        self,
        payload: Mapping[str, object],
        route: str,
        length: int,
        control: str,
        hypothesis_order: str,
        correspondence: str,
    ) -> dict[str, list[HistoryCandidate]]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, Mapping):
            raise StaticModelError("frozen cache lacks candidate routes")
        route_payload = candidates.get(route)
        if not isinstance(route_payload, Mapping):
            raise StaticModelError(f"frozen cache lacks route {route}")
        output: dict[str, list[HistoryCandidate]] = {}
        for scale in ("p16", "p8", "p4"):
            records = route_payload.get(scale)
            if not isinstance(records, list):
                raise StaticModelError(f"frozen cache lacks {route}/{scale}")
            by_age: dict[int, list[HistoryCandidate]] = {}
            for record in records:
                if not isinstance(record, Mapping):
                    raise StaticModelError("cached history candidate is malformed")
                feature = record.get("feature")
                valid = record.get("valid")
                age = record.get("age")
                same_object = record.get("same_object")
                target_weight = record.get("target_weight")
                if (
                    not isinstance(feature, Tensor)
                    or feature.dtype != torch.float32
                    or not isinstance(valid, Tensor)
                    or valid.dtype != torch.bool
                    or type(age) is not int
                ):
                    raise StaticModelError("cached history candidate tensors changed")
                metadata_present = same_object is not None or target_weight is not None
                if metadata_present and (
                    not isinstance(same_object, Tensor)
                    or same_object.dtype != torch.bool
                    or same_object.shape != valid.shape
                    or not isinstance(target_weight, Tensor)
                    or target_weight.dtype != torch.float32
                    or target_weight.shape != valid.shape
                ):
                    raise StaticModelError("cached match metadata changed")
                if (correspondence == "oracle_proposal") != metadata_present:
                    raise StaticModelError(
                        "cached match metadata does not match its correspondence route"
                    )
                if age <= length:
                    by_age.setdefault(age, []).append(
                        HistoryCandidate(
                            feature.to(self.device),
                            valid.to(self.device),
                            age,
                            None
                            if same_object is None
                            else same_object.to(self.device),
                            None
                            if target_weight is None
                            else target_weight.to(self.device),
                        )
                    )
            selected: list[HistoryCandidate] = []
            for age in sorted(by_age):
                pair = by_age[age]
                expected = 1 if correspondence == "oracle_select" else 2
                if len(pair) != expected:
                    raise StaticModelError(
                        "cached evidence slots do not match the correspondence route"
                    )
                if correspondence == "oracle_select" and control == "shuffle":
                    candidate = pair[0]
                    shuffled = self._history_content_control(
                        candidate.feature,
                        candidate.valid,
                        "shuffle",
                        age=age,
                        hypothesis=0,
                    )
                    pair = [HistoryCandidate(shuffled, candidate.valid, age)]
                elif control == "null":
                    pair = [
                        HistoryCandidate(
                            torch.zeros_like(candidate.feature),
                            torch.zeros_like(candidate.valid),
                            candidate.age,
                            None
                            if candidate.same_object is None
                            else torch.zeros_like(candidate.same_object),
                            candidate.target_weight,
                        )
                        for candidate in pair
                    ]
                elif control != "actual":
                    raise ValueError("cached content control is invalid")
                if hypothesis_order == "object_static":
                    pair = list(reversed(pair))
                selected.extend(pair)
            output[scale] = selected
        return output

    def forward_cached_history_controls(
        self,
        payload: Mapping[str, object],
        *,
        sampling: HistorySamplingOffsets,
        requests: Mapping[str, tuple[str, int, str, str, str]],
    ) -> tuple[CurrentPointPrediction, dict[str, HistoryPointPrediction]]:
        """Run trainable temporal paths from a graph-free frozen window cache."""

        current, current_prediction, point_to_p4 = self._hydrate_frozen_history_cache(
            payload
        )
        cached_coordinates = payload["current"]["current_coordinates"]
        if not isinstance(cached_coordinates, Tensor) or not np.array_equal(
            cached_coordinates.cpu().numpy(), sampling.current_coordinates
        ):
            raise StaticModelError("cached current coordinates changed")
        outputs: dict[str, HistoryPointPrediction] = {}
        for name, (route, length, control, order, correspondence) in requests.items():
            if length not in {1, 2, 4}:
                raise ValueError("cached history controls require K in {1,2,4}")
            aligned = self._cached_candidates(
                payload, route, length, control, order, correspondence
            )
            outputs[name] = self._history_point_branch(
                current,
                (),
                sampling,
                current_prediction,
                use_p4=True,
                content_control=control,
                hypothesis_order=order,
                correspondence=correspondence,
                aligned_by_scale=aligned,
                point_to_p4=point_to_p4,
            )
        return current_prediction, outputs

    def forward_history(
        self,
        inputs: StaticInput,
        *,
        sampling: HistorySamplingOffsets,
        content_control: str = "actual",
        hypothesis_order: str = "static_object",
        correspondence: str = "oracle_proposal",
    ) -> tuple[CurrentPointPrediction, HistoryPointPrediction]:
        """Evaluate one physical history against its frozen current baseline."""

        if not isinstance(inputs, StaticInput):
            raise TypeError("forward_history accepts only StaticInput")
        self._validate_sampling(inputs, sampling)
        current, history, current_prediction = self._prepare_point_history(inputs)
        prediction = self._history_point_branch(
            current,
            history,
            sampling,
            current_prediction,
            use_p4=True,
            content_control=content_control,
            hypothesis_order=hypothesis_order,
            correspondence=correspondence,
        )
        return current_prediction, prediction

    def forward_history_controls(
        self,
        inputs: StaticInput,
        requests: Mapping[str, tuple[HistorySamplingOffsets, int, str, str, str]],
    ) -> tuple[CurrentPointPrediction, dict[str, HistoryPointPrediction]]:
        """Evaluate validation interventions from one shared window encoding."""

        if not isinstance(inputs, StaticInput):
            raise TypeError("forward_history_controls accepts only StaticInput")
        if not requests:
            raise ValueError("at least one history intervention is required")
        current, full_history, current_prediction = self._prepare_point_history(inputs)
        outputs: dict[str, HistoryPointPrediction] = {}
        for name, request in requests.items():
            if len(request) != 5:
                raise ValueError("history control request must contain five fields")
            sampling, length, control, hypothesis_order, correspondence = request
            self._validate_sampling(inputs, sampling)
            if type(length) is not int or length not in {1, 2, 4}:
                raise ValueError("history controls require K in {1,2,4}")
            history = tuple(frame for frame in full_history if frame.age <= length)
            outputs[name] = self._history_point_branch(
                current,
                history,
                sampling,
                current_prediction,
                use_p4=True,
                content_control=control,
                hypothesis_order=hypothesis_order,
                correspondence=correspondence,
            )
        return current_prediction, outputs


def _observed_instance_displacements(
    coordinates: Tensor,
    ages: Tensor,
    group_key: Tensor,
    semantic: Tensor,
) -> tuple[Tensor, Tensor]:
    """Estimate centroid drift only for diagnostic normal-instance strata."""

    instance = torch.bitwise_right_shift(group_key, 16)
    valid_point = (semantic != 255) & (group_key != 0) & (instance > 0)
    keys = torch.unique(group_key[valid_point], sorted=True)
    displacement_by_point = coordinates.new_zeros(coordinates.shape)
    valid = torch.zeros(
        coordinates.shape[0], dtype=torch.bool, device=coordinates.device
    )
    for key in keys:
        group = valid_point & (group_key == key)
        current = group & (ages == 0)
        if int(current.sum()) < MINIMUM_INSTANCE_CENTROID_POINTS:
            continue
        current_center = coordinates[current].mean(dim=0)
        for age in torch.unique(ages[group]):
            if int(age) == 0:
                # Current members are fixed to zero offset by the architecture;
                # including their zero loss would only dilute historical motion.
                continue
            selected = group & (ages == age)
            if int(selected.sum()) < MINIMUM_INSTANCE_CENTROID_POINTS:
                continue
            source_center = coordinates[selected].mean(dim=0)
            displacement = current_center - source_center
            maximum_displacement = (
                INSTANCE_DISPLACEMENT_TOLERANCE_METRES
                + MAXIMUM_OBSERVED_SPEED_MPS * 0.1 * float(age)
            )
            if float(displacement.norm()) > maximum_displacement:
                continue
            displacement_by_point[selected] = displacement
            valid[selected] = True
    return displacement_by_point, valid


@dataclass(frozen=True, slots=True)
class ProposalMatchNullLoss:
    """Direct same-object or null supervision on three sparse attention scales."""

    total: Tensor
    match: Tensor
    null: Tensor
    match_by_scale: tuple[Tensor, Tensor, Tensor]
    null_by_scale: tuple[Tensor, Tensor, Tensor]
    match_queries_by_scale: tuple[int, int, int]
    null_queries_by_scale: tuple[int, int, int]
    structural_null_queries_by_scale: tuple[int, int, int]


def proposal_match_null_loss(
    prediction: HistoryPointPrediction,
) -> ProposalMatchNullLoss:
    """Supervise same-object or learnable null decisions independently by age."""

    masses = prediction.match_mass_by_scale
    if masses is None:
        raise StaticModelError("history prediction lacks match supervision masses")

    def weighted_negative_log(
        probability: Tensor, weight: Tensor, selected: Tensor
    ) -> Tensor:
        if not bool(selected.any()):
            return probability.sum() * 0.0
        selected_probability = probability[selected].float().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        selected_weight = weight[selected].float()
        denominator = selected_weight.sum()
        if not bool(denominator > 0.0):
            raise StaticModelError("selected match targets have zero total weight")
        return -(selected_weight * selected_probability.log()).sum() / denominator

    match_by_scale: list[Tensor] = []
    null_by_scale: list[Tensor] = []
    match_counts: list[int] = []
    null_counts: list[int] = []
    structural_null_counts: list[int] = []
    eligible_count = 0
    for mass in masses:
        if not mass.ages:
            match_by_scale.append(mass.same_object.sum() * 0.0)
            null_by_scale.append(mass.null.sum() * 0.0)
            match_counts.append(0)
            null_counts.append(0)
            structural_null_counts.append(0)
            continue
        weight = mass.target_weight[:, None].expand_as(mass.direct_same_object)
        eligible = weight > 0.0
        match_target = eligible & mass.direct_has_same_object
        absent_target = eligible & ~mass.direct_has_same_object
        # When every real candidate is invalid, null already has probability
        # one and supplies no learnable candidate-rejection decision.
        null_target = absent_target & mass.direct_real_valid
        structural_null = absent_target & ~mass.direct_real_valid
        eligible_count += int(eligible.sum().item())
        match_counts.append(int(match_target.sum().item()))
        null_counts.append(int(null_target.sum().item()))
        structural_null_counts.append(int(structural_null.sum().item()))
        match_by_scale.append(
            weighted_negative_log(
                mass.direct_same_object, weight, match_target
            )
        )
        null_by_scale.append(
            weighted_negative_log(mass.direct_null, weight, null_target)
        )
    if eligible_count == 0:
        raise StaticModelError("Proposal match loss has no generated-object queries")
    match = torch.stack(match_by_scale).mean()
    null = torch.stack(null_by_scale).mean()
    total = match + null
    for name, value in (
        ("proposal_match_null_total", total),
        ("proposal_match", match),
        ("proposal_null", null),
    ):
        _finite(name, value)
    return ProposalMatchNullLoss(
        total=total,
        match=match,
        null=null,
        match_by_scale=tuple(match_by_scale),
        null_by_scale=tuple(null_by_scale),
        match_queries_by_scale=tuple(match_counts),
        null_queries_by_scale=tuple(null_counts),
        structural_null_queries_by_scale=tuple(structural_null_counts),
    )


@dataclass(frozen=True, slots=True)
class OracleTemporalLoss:
    """Classification, normal safety, and class-balanced residual magnitude."""

    total: Tensor
    window_bce: Tensor
    normal_safety: Tensor
    magnitude: Tensor
    anomaly_magnitude: Tensor
    normal_magnitude: Tensor


def oracle_temporal_loss(
    window_logits: Tensor,
    current_logits: Tensor,
    current_anomaly: Tensor,
    current_valid: Tensor,
    *,
    safety_weight: float = 1.0,
    magnitude_weight: float = 0.1,
    smooth_l1_beta: float = 1.0,
) -> OracleTemporalLoss:
    """Train finite history corrections without a logit-ordering target."""

    if window_logits.ndim != 1 or current_logits.shape != window_logits.shape:
        raise StaticModelError("current and window logits must align as [N]")
    if (
        current_anomaly.shape != window_logits.shape
        or current_anomaly.dtype != torch.bool
        or current_valid.shape != window_logits.shape
        or current_valid.dtype != torch.bool
    ):
        raise StaticModelError("Oracle temporal targets must be aligned bool[N]")
    safe_weight = float(safety_weight)
    mag_weight = float(magnitude_weight)
    beta = float(smooth_l1_beta)
    if not all(math.isfinite(value) for value in (safe_weight, mag_weight, beta)):
        raise StaticModelError("Oracle temporal loss weights must be finite")
    if safe_weight < 0.0 or mag_weight < 0.0 or beta <= 0.0:
        raise StaticModelError("Oracle temporal loss weights are outside their range")
    anomaly = current_valid & current_anomaly
    normal = current_valid & ~current_anomaly
    if not bool(anomaly.any()) or not bool(normal.any()):
        raise StaticModelError("Oracle temporal loss requires both point classes")

    current = current_logits.detach().to(
        device=window_logits.device, dtype=window_logits.dtype
    )
    delta = window_logits.float() - current.float()
    positive_bce = F.binary_cross_entropy_with_logits(
        window_logits[anomaly].float(),
        torch.ones_like(window_logits[anomaly], dtype=torch.float32),
    )
    negative_bce = F.binary_cross_entropy_with_logits(
        window_logits[normal].float(),
        torch.zeros_like(window_logits[normal], dtype=torch.float32),
    )
    window_bce = 0.5 * (positive_bce + negative_bce)
    normal_safety = F.relu(delta[normal]).mean()
    anomaly_magnitude = F.smooth_l1_loss(
        delta[anomaly],
        torch.zeros_like(delta[anomaly]),
        reduction="mean",
        beta=beta,
    )
    normal_magnitude = F.smooth_l1_loss(
        delta[normal],
        torch.zeros_like(delta[normal]),
        reduction="mean",
        beta=beta,
    )
    magnitude = 0.5 * (anomaly_magnitude + normal_magnitude)
    total = window_bce + safe_weight * normal_safety + mag_weight * magnitude
    for name, value in (
        ("oracle_temporal_total", total),
        ("oracle_temporal_window_bce", window_bce),
        ("oracle_temporal_normal_safety", normal_safety),
        ("oracle_temporal_magnitude", magnitude),
    ):
        _finite(name, value)
    return OracleTemporalLoss(
        total=total,
        window_bce=window_bce,
        normal_safety=normal_safety,
        magnitude=magnitude,
        anomaly_magnitude=anomaly_magnitude,
        normal_magnitude=normal_magnitude,
    )
