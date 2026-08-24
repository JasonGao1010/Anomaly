#!/usr/bin/env python3
"""Run AJAE's preregistered three-arm selector deconfounding experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

CPU_THREADS = 12
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch import nn  # noqa: E402

try:
    from .protocol import AJAEProtocol, WINDOW_FRAMES, load_protocol
    from .scene import (
        LabelMode,
        PointLabels,
        SceneWindow,
        SourceFrame,
        STUSequence,
        assemble_window,
        make_source_frame,
    )
    from .static import (
        DEFAULT_STU_REPOSITORY,
        HistorySamplingOffsets,
        INSTANCE_DISPLACEMENT_TOLERANCE_METRES,
        MINIMUM_INSTANCE_CENTROID_POINTS,
        StaticInput,
        WindowDetectorPrototype,
        _observed_instance_displacements,
        oracle_temporal_loss,
        proposal_match_null_loss,
        stu_source_manifest,
        stu_weight_identity,
    )
except ImportError:
    from protocol import AJAEProtocol, WINDOW_FRAMES, load_protocol
    from scene import (
        LabelMode,
        PointLabels,
        SceneWindow,
        SourceFrame,
        STUSequence,
        assemble_window,
        make_source_frame,
    )
    from static import (
        DEFAULT_STU_REPOSITORY,
        HistorySamplingOffsets,
        INSTANCE_DISPLACEMENT_TOLERANCE_METRES,
        MINIMUM_INSTANCE_CENTROID_POINTS,
        StaticInput,
        WindowDetectorPrototype,
        _observed_instance_displacements,
        oracle_temporal_loss,
        proposal_match_null_loss,
        stu_source_manifest,
        stu_weight_identity,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FORMAT = "ajae-intensity-calibration-v2"
MECHANISM_MANIFEST_FORMAT = "ajae-oracle-mechanism-source-v3"
ORACLE_TEMPORAL_FORMAT = "ajae-selector-deconfounding-v1"
ORACLE_TEMPORAL_STATE_FORMAT = "ajae-selector-deconfounding-states-v1"
ORACLE_TEMPORAL_PROGRESS_FORMAT = "ajae-selector-deconfounding-progress-v1"
ORACLE_TEMPORAL_PROTOCOL_FORMAT = "ajae-oracle-temporal-protocol-v6"
STAGE0_AUDIT_FORMAT = "ajae-selector-loss-gradient-audit-v2"
DEFAULT_ORACLE_TEMPORAL_PROTOCOL = PROJECT_ROOT / "oracle.json"
ORACLE_HISTORY_LENGTHS = (1, 2, 4)
SELECTOR_ARMS = (
    "clean_select",
    "proposal_direct",
    "proposal_classification",
)
SELECTOR_CONDITIONS = {
    "clean_select": "Clean-Select-4",
    "proposal_direct": "Proposal-Direct-4",
    "proposal_classification": "Proposal-Classification-4",
}
DEFAULT_GPU_MEMORY_FRACTION = 0.55
MAXIMUM_GPU_MEMORY_FRACTION = 0.70
RANGE_EDGES = np.asarray((0.0, 10.0, 20.0, 30.0, 40.0, 50.0, np.inf))
INCIDENCE_EDGES = np.asarray((0.0, math.pi / 6, math.pi / 3, math.pi / 2 + 1e-6))
LASER_BEAMS = 128
EPSILON = 1.0e-6
SOURCE_IDENTITY_FILES = {
    "train.py": PROJECT_ROOT / "src" / "train.py",
    "static.py": PROJECT_ROOT / "src" / "static.py",
    "scene.py": PROJECT_ROOT / "src" / "scene.py",
    "protocol.py": PROJECT_ROOT / "src" / "protocol.py",
}


def _project_relative_path(path: Path | str) -> str:
    """Serialize a filesystem path relative to the repository root."""

    resolved = Path(path).expanduser().resolve()
    return Path(os.path.relpath(resolved, PROJECT_ROOT)).as_posix()


class TrainingError(RuntimeError):
    """Report a scientifically invalid or failed AJAE training operation."""


class NoEligibleGroundError(TrainingError):
    """The current scan has no labeled ground return suitable for insertion."""


def _finite_number(name: str, value: float, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else "finite "
        raise ValueError(f"{name} must be a {qualifier}number")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        _json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_tree_identity(sequence_root: Path) -> dict[str, object]:
    """Bind one raw STU sequence without persisting a derived manifest."""

    root = sequence_root.expanduser().resolve(strict=True)
    records = []
    for path in sorted(
        (value for value in root.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )

    def component(name: str) -> dict[str, object]:
        if name == "metadata":
            selected = [record for record in records if "/" not in record["path"]]
        else:
            selected = [
                record
                for record in records
                if str(record["path"]).startswith(f"{name}/")
            ]
        return {
            "file_count": len(selected),
            "bytes": sum(int(record["bytes"]) for record in selected),
            "manifest_sha256": _sha256_json(selected),
        }

    return {
        "identity_algorithm": (
            "sha256(canonical JSON list of relative path, byte count, and "
            "per-file SHA-256 in lexical path order)"
        ),
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "manifest_sha256": _sha256_json(records),
        "components": {
            name: component(name) for name in ("velodyne", "labels", "metadata")
        },
    }


def _tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash tensor names, types, shapes, and exact bytes in a frozen state."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def training_code_identity(
    protocol: AJAEProtocol,
    official_repository: Path | str = DEFAULT_STU_REPOSITORY,
) -> dict[str, object]:
    """Hash the protocol, key source files, and pretrained weights."""

    sources = {name: _sha256_file(path) for name, path in SOURCE_IDENTITY_FILES.items()}
    weights = stu_weight_identity(protocol.checkpoint_path(PROJECT_ROOT))
    stu_sources = stu_source_manifest(official_repository)
    return {
        "protocol_sha256": _sha256_json(protocol.plain_document()),
        "source_sha256": sources,
        "stu_checkpoint_sha256": weights["checkpoint_sha256"],
        "stu_model_state_sha256": weights["model_state_sha256"],
        "stu_model_state_tensor_sha256": weights["model_state_tensor_sha256"],
        "stu_source_identity": {
            "file_count": stu_sources["file_count"],
            "total_bytes": stu_sources["total_bytes"],
            "manifest_sha256": stu_sources["manifest_sha256"],
        },
    }


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """All numerical choices that define one AJAE optimization run."""

    new_lr: float
    weight_decay: float
    seed: int
    minimum_current_anomaly_points: int = 1
    gpu_memory_fraction: float = DEFAULT_GPU_MEMORY_FRACTION

    def __post_init__(self) -> None:
        _finite_number("new_lr", self.new_lr, positive=True)
        if _finite_number("weight_decay", self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            type(self.minimum_current_anomaly_points) is not int
            or self.minimum_current_anomaly_points < 1
        ):
            raise ValueError("minimum_current_anomaly_points must be positive")
        fraction = _finite_number("gpu_memory_fraction", self.gpu_memory_fraction)
        if not 0.0 < fraction <= MAXIMUM_GPU_MEMORY_FRACTION:
            raise ValueError(
                "gpu_memory_fraction must be greater than zero and at most 0.70"
            )


@dataclass(frozen=True, slots=True)
class TrajectoryPlan:
    """One physical synthetic trajectory fixed before parameter updates."""

    seed: int
    angular_scale_rad: float
    radial_speed_mps: float
    anchor_mode: str
    current_anomaly_points: int
    trajectory_profile: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("plan seed must be non-negative")
        controls = TrajectoryControls(
            self.angular_scale_rad, self.radial_speed_mps, self.anchor_mode
        )
        if controls not in TRAJECTORY_CONTROLS:
            raise TrainingError("a stored plan uses unknown trajectory controls")
        if self.trajectory_profile != _trajectory_profile_name(controls):
            raise TrainingError("a stored plan has inconsistent trajectory controls")
        if (
            type(self.current_anomaly_points) is not int
            or self.current_anomaly_points < 1
        ):
            raise TrainingError("a stored plan has no current-frame anomaly return")


@dataclass(slots=True)
class RenderedWindow:
    counterfactual: SceneWindow
    synthetic_members: np.ndarray
    proposal_parameters: dict[str, object] | None = None


def oracle_history_sampling(rendered: RenderedWindow) -> HistorySamplingOffsets:
    """Build exact synthetic-trajectory queries without exposing supervision.

    Normal returns use the same per-point interface with zero offset.  The STU
    source provides persistent instance IDs but no ground-truth object poses;
    centroid drift is therefore not promoted to an Oracle motion label.
    """

    if not isinstance(rendered, RenderedWindow):
        raise TypeError("rendered must be RenderedWindow")
    parameters = rendered.proposal_parameters
    if not isinstance(parameters, Mapping):
        raise TrainingError("Oracle sampling requires trajectory metadata")
    velocity_value = parameters.get("velocity_world_mps")
    if not isinstance(velocity_value, list):
        raise TrainingError("Oracle sampling lacks the generated world velocity")
    velocity_world = np.asarray(velocity_value, dtype=np.float64)
    if velocity_world.shape != (3,) or not np.isfinite(velocity_world).all():
        raise TrainingError("Oracle world velocity is malformed")

    window = rendered.counterfactual
    current = window.members.current_slice
    coordinates = np.ascontiguousarray(
        window.members.coordinates_current[current], dtype=np.float32
    )
    synthetic = rendered.synthetic_members[current]
    if synthetic.dtype != np.bool_ or synthetic.shape != (coordinates.shape[0],):
        raise TrainingError("Oracle synthetic identity does not align with current")
    velocity_current = velocity_world @ window.current.lidar_pose[:3, :3]
    offsets = np.zeros((WINDOW_FRAMES, coordinates.shape[0], 3), dtype=np.float32)
    for age in range(1, WINDOW_FRAMES):
        # Historical position = current position - elapsed-time displacement.
        offsets[age, synthetic] = np.asarray(
            -0.1 * age * velocity_current, dtype=np.float32
        )
    membership_by_age: list[np.ndarray | None] = [None] * WINDOW_FRAMES
    for frame_index, item in enumerate(window.frames):
        member_slice = window.members.frame_slice(frame_index)
        membership_by_age[item.age] = np.ascontiguousarray(
            rendered.synthetic_members[member_slice], dtype=np.bool_
        )
    if any(value is None for value in membership_by_age):
        raise TrainingError("Oracle sampling requires all five frame ages")
    return HistorySamplingOffsets(
        current_coordinates=coordinates,
        query_offsets=np.ascontiguousarray(offsets),
        object_membership=np.ascontiguousarray(synthetic, dtype=np.bool_),
        object_membership_by_age=tuple(
            value for value in membership_by_age if value is not None
        ),
    )


def _binary_anomaly_targets(
    raw_semantic: np.ndarray, synthetic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the public binary anomaly rule to rendered current returns."""

    if (
        raw_semantic.ndim != 1
        or raw_semantic.dtype != np.uint16
        or synthetic.shape != raw_semantic.shape
        or synthetic.dtype != np.bool_
    ):
        raise TrainingError("binary anomaly targets require uint16[N] and bool[N]")
    # AJAE's binary task follows the public anomaly labels, not the 19-class
    # STU semantic-head ignore map: raw 0 is ignored, raw 2 is anomalous, and
    # every other nonzero semantic is a normal negative.
    anomaly = synthetic | (raw_semantic == np.uint16(2))
    valid = (raw_semantic != np.uint16(0)) | synthetic
    return np.ascontiguousarray(anomaly), np.ascontiguousarray(valid)


def _oracle_current_targets(
    rendered: RenderedWindow, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return point supervision without placing labels in model inputs."""

    current = rendered.counterfactual.members.current_slice
    synthetic = rendered.synthetic_members[current]
    source = rendered.counterfactual.current
    assert source.labels is not None
    raw_semantic = source.labels.semantic[source.real_slots]
    anomaly, valid = _binary_anomaly_targets(raw_semantic, synthetic)
    return (
        torch.from_numpy(anomaly).to(device=device, dtype=torch.bool),
        torch.from_numpy(valid).to(device=device, dtype=torch.bool),
    )


def _render_oracle_trajectory(
    window: SceneWindow,
    plan: TrajectoryPlan,
    generator: "CounterfactualGenerator",
) -> RenderedWindow:
    """Render the plan-selected physical trajectory exactly once."""

    rendered = generator.render(
        window,
        seed=plan.seed,
        trajectory_controls=TrajectoryControls(
            plan.angular_scale_rad,
            plan.radial_speed_mps,
            plan.anchor_mode,
        ),
        # Static plans fix the complete planar velocity to zero.
        lateral_speed_mps=(0.0 if plan.radial_speed_mps == 0.0 else None),
    )
    current = rendered.counterfactual.members.current_slice
    if (
        int(np.count_nonzero(rendered.synthetic_members[current]))
        != plan.current_anomaly_points
    ):
        raise TrainingError("single Oracle trajectory did not reproduce its plan")
    return rendered


@dataclass(frozen=True, slots=True)
class TrajectoryControls:
    """Physical controls for one preregistered synthetic trajectory."""

    angular_scale_rad: float
    radial_speed_mps: float
    anchor_mode: str

    def __post_init__(self) -> None:
        angular_scale = _finite_number(
            "angular_scale_rad", self.angular_scale_rad, positive=True
        )
        if angular_scale > 0.05:
            raise ValueError("angular_scale_rad must be at most 0.05 radians")
        radial_speed = _finite_number("radial_speed_mps", self.radial_speed_mps)
        if not 0.0 <= radial_speed <= 20.0:
            raise ValueError("radial_speed_mps must be in [0, 20]")
        if self.anchor_mode not in {"uniform_ground", "near_foreground"}:
            raise ValueError("anchor_mode must be uniform_ground or near_foreground")


def _trajectory_profile_name(controls: TrajectoryControls) -> str:
    size = "small" if controls.angular_scale_rad == 0.006 else "broad"
    motion = "static" if controls.radial_speed_mps == 0.0 else "moving"
    anchor = "uniform" if controls.anchor_mode == "uniform_ground" else "edge"
    return f"{size}-{motion}-{anchor}"


TRAJECTORY_CONTROLS = tuple(
    TrajectoryControls(angular_scale, radial_speed, anchor_mode)
    for angular_scale in (0.006, 0.018)
    for radial_speed in (0.0, 15.0)
    for anchor_mode in ("uniform_ground", "near_foreground")
)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_value(value), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nested_state_equal(left: object, right: object) -> bool:
    """Compare RNG or optimizer-free audit state without coercing tensors."""

    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(
            left, right
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(
                _nested_state_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _repeat_noise_relative_comparison(
    first: Tensor,
    repeated: Tensor,
    swapped: Tensor,
    swapped_repeated: Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    repeat_noise_multiplier: float,
) -> dict[str, float | bool]:
    """Compare a permutation against two same-order numerical baselines."""

    tensors = (first, repeated, swapped, swapped_repeated)
    if any(value.shape != first.shape for value in tensors[1:]):
        return {
            "reference_scale": math.inf,
            "base_tolerance": math.inf,
            "repeat_noise": math.inf,
            "swap_difference": math.inf,
            "allowed_swap_difference": math.inf,
            "repeat_stable": False,
            "passed": False,
        }

    def maximum_difference(left: Tensor, right: Tensor) -> float:
        return float(
            (left.detach().float() - right.detach().float()).abs().max().cpu()
        )

    reference_scale = max(
        float(value.detach().float().abs().max().cpu()) for value in tensors
    )
    base = absolute_tolerance + relative_tolerance * reference_scale
    repeat_noise = max(
        maximum_difference(first, repeated),
        maximum_difference(swapped, swapped_repeated),
    )
    swap_difference = max(
        maximum_difference(first, swapped),
        maximum_difference(repeated, swapped_repeated),
    )
    allowed = base + repeat_noise_multiplier * repeat_noise
    repeat_stable = repeat_noise <= base
    return {
        "reference_scale": reference_scale,
        "base_tolerance": base,
        "repeat_noise": repeat_noise,
        "swap_difference": swap_difference,
        "allowed_swap_difference": allowed,
        "repeat_stable": repeat_stable,
        "passed": repeat_stable and swap_difference <= allowed,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _member_identity(window: SceneWindow) -> np.ndarray:
    return np.column_stack(
        (window.members.source_frame, window.members.source_slot)
    ).astype(np.int64, copy=False)


def _slot_directions(frame: SourceFrame) -> np.ndarray:
    """Recover every physical ray direction from the organized 128-beam scan."""

    if frame.slot_count % LASER_BEAMS:
        raise TrainingError(
            f"frame {frame.frame_id} has {frame.slot_count} slots, not a 128-beam grid"
        )
    columns = frame.slot_count // LASER_BEAMS
    xyz = frame.xyzi[:, :3].reshape(columns, LASER_BEAMS, 3).astype(np.float64)
    valid = ~frame.zero_slot_mask.reshape(columns, LASER_BEAMS)
    ranges = np.linalg.norm(xyz, axis=2)
    known = xyz / np.maximum(ranges[..., None], EPSILON)

    azimuth = np.arctan2(xyz[..., 1], xyz[..., 0])
    column_vector = np.where(valid, np.exp(1j * azimuth), 0.0).sum(axis=1)
    known_columns = np.flatnonzero(np.abs(column_vector) > EPSILON)
    if known_columns.size < 2:
        raise TrainingError("a scan does not contain enough rays to recover azimuth")
    known_azimuth = np.unwrap(np.angle(column_vector[known_columns]))
    column_azimuth = np.interp(
        np.arange(columns, dtype=np.float64), known_columns, known_azimuth
    )

    elevation = np.arctan2(xyz[..., 2], np.linalg.norm(xyz[..., :2], axis=2))
    beam_elevation = np.full(LASER_BEAMS, np.nan, dtype=np.float64)
    for beam in range(LASER_BEAMS):
        if bool(valid[:, beam].any()):
            beam_elevation[beam] = np.median(elevation[valid[:, beam], beam])
    known_beams = np.flatnonzero(np.isfinite(beam_elevation))
    if known_beams.size < 2:
        raise TrainingError("a scan does not contain enough rays to recover elevation")
    beam_elevation = np.interp(
        np.arange(LASER_BEAMS, dtype=np.float64),
        known_beams,
        beam_elevation[known_beams],
    )

    cosine = np.cos(beam_elevation)[None, :]
    reconstructed = np.stack(
        (
            cosine * np.cos(column_azimuth)[:, None],
            cosine * np.sin(column_azimuth)[:, None],
            np.broadcast_to(np.sin(beam_elevation)[None, :], valid.shape),
        ),
        axis=2,
    )
    reconstructed[valid] = known[valid]
    return np.ascontiguousarray(reconstructed.reshape(-1, 3), dtype=np.float64)


def _stratum_key(value: Sequence[int]) -> str:
    return ":".join(str(int(item)) for item in value)


@dataclass(slots=True)
class GeneratorCalibration:
    """Normal-206 intensity distributions used by the ray renderer."""

    sensor: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        if not self.sensor:
            raise TrainingError("intensity calibration is empty")
        for key, values in self.sensor.items():
            if not key or values.ndim != 1 or values.shape[0] < 1:
                raise TrainingError("intensity distribution has an invalid shape")
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise TrainingError("intensity distribution is non-finite")

    @classmethod
    def from_payload(cls, payload: object) -> "GeneratorCalibration":
        if not isinstance(payload, Mapping):
            raise TrainingError("intensity calibration has an invalid format")
        if payload["format"] != CALIBRATION_FORMAT:
            raise TrainingError("intensity calibration version is unsupported")
        sensor_value = payload["sensor"]
        if not isinstance(sensor_value, Mapping):
            raise TrainingError("intensity calibration is invalid")
        sensor = {}
        for key, value in sensor_value.items():
            if not isinstance(key, str) or not isinstance(value, Tensor):
                raise TrainingError("intensity distribution is invalid")
            sensor[key] = value.cpu().numpy().astype(np.float32, copy=True)
        return cls(sensor=sensor)


class CounterfactualGenerator:
    """Render one stable moving procedural object on the original LiDAR rays."""

    def __init__(
        self,
        sensor_reference: Mapping[str, np.ndarray],
        *,
        minimum_current_points: int,
    ) -> None:
        if not sensor_reference:
            raise TrainingError("normal-206 sensor reference is empty")
        self.minimum_current_points = minimum_current_points
        self._sensor_reference = dict(sensor_reference)
        self._reference_cache = dict(sensor_reference)

    def _reference(self, key: str) -> np.ndarray:
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached
        range_bin, beam, _ = key.split(":")
        candidates = [
            value
            for name, value in self._sensor_reference.items()
            if name.startswith(f"{range_bin}:{beam}:")
        ]
        if candidates:
            reference = np.concatenate(candidates, axis=0)
        else:
            candidates = [
                value
                for name, value in self._sensor_reference.items()
                if name.startswith(f"{range_bin}:")
            ]
            if not candidates:
                raise TrainingError(
                    f"normal 206 has no sensor reference for range bin {range_bin}"
                )
            reference = np.concatenate(candidates, axis=0)
        self._reference_cache[key] = reference
        return reference

    @staticmethod
    def _rotation(yaw: float) -> np.ndarray:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    @staticmethod
    def _intersection(
        origin: np.ndarray,
        directions: np.ndarray,
        center: np.ndarray,
        rotation: np.ndarray,
        radii: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_origin = (origin - center) @ rotation
        local_direction = directions @ rotation
        inverse_square = 1.0 / np.square(radii)
        a = np.sum(np.square(local_direction) * inverse_square, axis=1)
        b = 2.0 * np.sum(local_origin * local_direction * inverse_square, axis=1)
        c = float(np.sum(np.square(local_origin) * inverse_square) - 1.0)
        discriminant = np.square(b) - 4.0 * a * c
        valid = discriminant >= 0.0
        distance = np.full(directions.shape[0], np.inf, dtype=np.float64)
        root = np.sqrt(np.maximum(discriminant, 0.0))
        near = (-b - root) / np.maximum(2.0 * a, EPSILON)
        far = (-b + root) / np.maximum(2.0 * a, EPSILON)
        selected = np.where(near > 0.05, near, far)
        valid &= selected > 0.05
        distance[valid] = selected[valid]
        return distance, valid

    def _sample_intensity(
        self,
        ranges: np.ndarray,
        beams: np.ndarray,
        incidence: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        range_bin = np.searchsorted(RANGE_EDGES, ranges, side="right") - 1
        incidence_bin = np.searchsorted(INCIDENCE_EDGES, incidence, side="right") - 1
        incidence_bin = np.clip(incidence_bin, 0, len(INCIDENCE_EDGES) - 2)
        result = np.empty(ranges.shape[0], dtype=np.float32)
        for index, values in enumerate(
            zip(range_bin, beams, incidence_bin, strict=True)
        ):
            reference = self._reference(_stratum_key(values))
            result[index] = reference[rng.integers(reference.shape[0])]
        return result

    @staticmethod
    def _trajectory_anchor(
        current: SourceFrame,
        current_slots: np.ndarray,
        current_range: np.ndarray,
        ground: np.ndarray,
        rng: np.random.Generator,
        mode: str,
    ) -> tuple[int, bool, float]:
        """Choose ground contact by range and optional native foreground geometry."""

        candidates = np.flatnonzero(
            ground & (current_range >= 8.0) & (current_range <= 50.0)
        )
        if candidates.size == 0:
            raise NoEligibleGroundError(
                "current scan has no eligible normal ground return"
            )

        assert current.labels is not None
        semantic = current.labels.semantic
        original_range = np.linalg.norm(current.xyzi[:, :3].astype(np.float64), axis=1)
        original_range[current.zero_slot_mask] = np.inf
        foreground = (
            (~current.zero_slot_mask)
            & (semantic != np.uint16(0))
            & (~np.isin(semantic, (40, 44, 48, 60)))
        )
        columns = current.slot_count // LASER_BEAMS
        foreground_range = np.where(foreground, original_range, np.inf).reshape(
            columns, LASER_BEAMS
        )
        nearest_foreground = foreground_range.min(axis=1)
        local_foreground = np.full(columns, np.inf, dtype=np.float64)
        # Adjacent azimuth columns represent real neighbouring rays. No wrap is
        # used, so the two scan boundaries cannot become artificial neighbours.
        for offset in range(-3, 4):
            source_start = max(0, -offset)
            source_stop = min(columns, columns - offset)
            target_start = source_start + offset
            target_stop = source_stop + offset
            local_foreground[target_start:target_stop] = np.minimum(
                local_foreground[target_start:target_stop],
                nearest_foreground[source_start:source_stop],
            )
        candidate_columns = current_slots[candidates] // LASER_BEAMS
        near_foreground = (
            local_foreground[candidate_columns] + 0.75 < current_range[candidates]
        )
        if mode == "near_foreground":
            candidates = candidates[near_foreground]
            if candidates.size == 0:
                raise NoEligibleGroundError(
                    "current scan has no ground contact near a foreground depth edge"
                )

        target_range = float(rng.uniform(8.0, 50.0))
        nearest_count = min(12, candidates.size)
        nearest = np.argpartition(
            np.abs(current_range[candidates] - target_range), nearest_count - 1
        )[:nearest_count]
        selected = int(candidates[int(rng.choice(nearest))])
        selected_column = int(current_slots[selected] // LASER_BEAMS)
        selected_near_foreground = bool(
            local_foreground[selected_column] + 0.75 < current_range[selected]
        )
        return selected, selected_near_foreground, target_range

    def render(
        self,
        window: SceneWindow,
        *,
        seed: int,
        trajectory_controls: TrajectoryControls,
        lateral_speed_mps: float | None = None,
    ) -> RenderedWindow:
        if window.labels is None or window.labels.semantic_target is None:
            raise TrainingError("counterfactual rendering requires normal labels")
        rng = np.random.default_rng(seed)
        current = window.current
        assert current.labels is not None
        current_slots = current.real_slots
        current_xyz = current.xyzi[current_slots, :3]
        current_range = np.linalg.norm(current_xyz, axis=1)
        ground = np.isin(current.labels.semantic[current_slots], (40, 44, 48, 60))
        anchor_index, near_foreground, target_range = self._trajectory_anchor(
            current,
            current_slots,
            current_range,
            ground,
            rng,
            trajectory_controls.anchor_mode,
        )
        anchor = current_xyz[anchor_index].astype(np.float64)
        anchor_range = float(current_range[anchor_index])
        projected_half_size = anchor_range * trajectory_controls.angular_scale_rad
        # Angular scale controls the number of native rays that can hit the object.
        radii = np.asarray(
            (
                np.clip(projected_half_size * rng.uniform(0.8, 1.5), 0.12, 1.35),
                np.clip(projected_half_size * rng.uniform(0.6, 1.0), 0.10, 0.90),
                np.clip(projected_half_size * rng.uniform(1.0, 1.7), 0.18, 1.40),
            ),
            dtype=np.float64,
        )
        proposal_parameters: dict[str, object] = {
            "style": "single_trajectory",
            "anchor_mode": trajectory_controls.anchor_mode,
            "anchor_near_foreground": near_foreground,
            "target_anchor_range_m": target_range,
            "actual_anchor_range_m": anchor_range,
            "angular_scale_rad": trajectory_controls.angular_scale_rad,
            "radial_speed_mps": trajectory_controls.radial_speed_mps,
        }
        # Two homothetic ellipsoids whose normalized centre distance lies
        # strictly between |1-s| and 1+s have intersecting surfaces. This gives
        # one connected object rather than an occasionally hidden inner shell.
        second_scale = float(rng.uniform(0.45, 0.8))
        second_radii = radii * second_scale
        offset_angle = float(rng.uniform(-math.pi, math.pi))
        normalized_distance = 0.75 * (1.0 + second_scale)
        second_offset = (
            radii
            * normalized_distance
            * np.asarray(
                (math.cos(offset_angle), math.sin(offset_angle), 0.0),
                dtype=np.float64,
            )
        )
        yaw = float(rng.uniform(-math.pi, math.pi))
        object_rotation = self._rotation(yaw)
        pose = current.lidar_pose
        radial_world = anchor @ pose[:3, :3].T
        radial_world[2] = 0.0
        radial_world /= max(float(np.linalg.norm(radial_world)), EPSILON)
        lateral_world = np.asarray(
            (-radial_world[1], radial_world[0], 0.0), dtype=np.float64
        )
        lateral_speed = (
            float(rng.uniform(-1.5, 1.5))
            if lateral_speed_mps is None
            else _finite_number("lateral_speed_mps", lateral_speed_mps)
        )
        if abs(lateral_speed) > 1.5:
            raise ValueError("lateral_speed_mps must lie in [-1.5, 1.5]")
        # Positive radial motion places the object closer in causal history.
        velocity = (
            trajectory_controls.radial_speed_mps * radial_world
            + lateral_speed * lateral_world
        )
        proposal_parameters.update(
            {
                "lateral_speed_mps": lateral_speed,
                "radii_m": radii.tolist(),
                "velocity_world_mps": velocity.tolist(),
            }
        )
        center_sensor = anchor + np.asarray((0.0, 0.0, radii[2]), dtype=np.float64)
        center_world = center_sensor @ pose[:3, :3].T + pose[:3, 3]
        synthetic_instance = np.uint16(60000 + seed % 5000)

        rendered: list[SourceFrame] = []
        synthetic_slots: list[np.ndarray] = []
        for item in window.frames:
            source = item.source
            assert source.labels is not None
            source_pose = source.lidar_pose
            time_offset = -0.1 * item.age
            moving_center = center_world + velocity * time_offset
            origin_world = source_pose[:3, 3]
            slots = np.arange(source.slot_count, dtype=np.int32)
            sensor_points = source.xyzi[:, :3].astype(np.float64)
            original_range = np.linalg.norm(sensor_points, axis=1)
            original_range[source.zero_slot_mask] = np.inf
            sensor_direction = _slot_directions(source)
            world_direction = sensor_direction @ source_pose[:3, :3].T

            first_distance, first_valid = self._intersection(
                origin_world,
                world_direction,
                moving_center,
                object_rotation,
                radii,
            )
            second_center = moving_center + second_offset @ object_rotation.T
            second_distance, second_valid = self._intersection(
                origin_world,
                world_direction,
                second_center,
                object_rotation,
                second_radii,
            )
            distance = np.minimum(first_distance, second_distance)
            valid = (first_valid | second_valid) & (distance + 0.05 < original_range)
            hit_slots = slots[valid]
            hit_mask = np.zeros(source.slot_count, dtype=np.bool_)
            hit_mask[hit_slots] = True
            synthetic_slots.append(hit_mask)

            xyzi = source.xyzi.copy()
            if hit_slots.size:
                hit_world = (
                    origin_world + distance[valid, None] * world_direction[valid]
                )
                hit_sensor = (hit_world - origin_world) @ source_pose[:3, :3]
                first_surface = first_distance[valid] <= second_distance[valid]
                chosen_center = np.where(
                    first_surface[:, None],
                    moving_center[None, :],
                    second_center[None, :],
                )
                chosen_radii = np.where(
                    first_surface[:, None], radii[None, :], second_radii[None, :]
                )
                local = (hit_world - chosen_center) @ object_rotation
                local_normal = local / np.square(chosen_radii)
                world_normal = local_normal @ object_rotation.T
                world_normal /= np.maximum(
                    np.linalg.norm(world_normal, axis=1, keepdims=True), EPSILON
                )
                incidence = np.arccos(
                    np.clip(
                        np.abs(np.sum(world_normal * -world_direction[valid], axis=1)),
                        0.0,
                        1.0,
                    )
                )
                xyzi[hit_slots, :3] = hit_sensor.astype(np.float32)
                intensity_rng = np.random.default_rng(
                    np.random.SeedSequence((seed, source.frame_id, 2))
                )
                xyzi[hit_slots, 3] = self._sample_intensity(
                    distance[valid],
                    slots[valid] % LASER_BEAMS,
                    incidence,
                    intensity_rng,
                )

            semantic = source.labels.semantic.copy()
            instance = source.labels.instance.copy()
            semantic_target = source.labels.semantic_target.copy()
            semantic[hit_slots] = np.uint16(2)
            instance[hit_slots] = synthetic_instance
            semantic_target[hit_slots] = np.uint8(255)
            packed = semantic.astype(np.uint32) | (
                instance.astype(np.uint32) << np.uint32(16)
            )
            labels = PointLabels(
                packed=packed,
                semantic=semantic,
                instance=instance,
                semantic_target=semantic_target,
            )
            rendered.append(
                make_source_frame(source.frame_id, xyzi, source.lidar_pose, labels)
            )

        current_count = int(np.count_nonzero(synthetic_slots[-1]))
        if current_count < self.minimum_current_points:
            raise TrainingError(
                f"counterfactual current object has only {current_count} returns"
            )
        counterfactual = assemble_window(
            window.spec, window.current_frame, tuple(rendered)
        )
        member_masks = np.concatenate(
            [
                mask[source.real_slots]
                for mask, source in zip(synthetic_slots, rendered, strict=True)
            ]
        )
        return RenderedWindow(
            counterfactual=counterfactual,
            synthetic_members=member_masks,
            proposal_parameters=proposal_parameters,
        )


def _current_counterfactual(rendered: RenderedWindow) -> SceneWindow:
    """Use the same single-frame STU feature space as the current baseline."""

    return assemble_window(
        rendered.counterfactual.spec,
        rendered.counterfactual.current_frame,
        (rendered.counterfactual.frames[-1].source,),
    )


def _plan_key(role: str, epoch: int, frame: int) -> str:
    if role not in {"train", "validation"}:
        raise ValueError("plan role must be train or validation")
    return f"{role}:{epoch}:{frame}"


def _balanced_point_bce(logits: Tensor, anomaly: Tensor, valid: Tensor) -> Tensor:
    """Give synthetic anomaly and valid normal points equal aggregate weight."""

    positive = valid & anomaly
    negative = valid & ~anomaly
    if not bool(positive.any()) or not bool(negative.any()):
        raise TrainingError("balanced point loss requires both target classes")
    positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[positive].float(), torch.ones_like(logits[positive]).float()
    )
    negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[negative].float(), torch.zeros_like(logits[negative]).float()
    )
    return 0.5 * (positive_loss + negative_loss)


def _oracle_distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise TrainingError("Oracle metric distribution is empty or non-finite")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


class OracleTemporalExperiment:
    """Train one Oracle-aligned residual model on one trajectory per window."""

    _temporal_prefixes = (
        "temporal_p16.",
        "temporal_p8.",
        "temporal_p4.",
        "temporal_point_delta.",
    )

    def __init__(
        self,
        *,
        protocol: AJAEProtocol,
        data_root: Path,
        output: Path,
        config: TrainConfig,
        mechanism_manifest: Path,
        experiment_protocol: Path,
        official_repository: Path,
        resume: bool = False,
    ) -> None:
        # Count source verification and model construction inside the experiment wall.
        self._run_wall_started = time.perf_counter()
        self._run_wall_prior_seconds = 0.0
        if not torch.cuda.is_available():
            raise TrainingError("Oracle temporal experiment requires CUDA")
        self.protocol = protocol
        self.data_root = data_root.expanduser().resolve(strict=True)
        self.output = output.expanduser().resolve()
        self.run_mode = "micro"
        self.resume = bool(resume)
        if self.output.exists():
            if not self.output.is_dir() or (any(self.output.iterdir()) and not resume):
                raise TrainingError("Oracle output directory must be absent or empty")
        else:
            self.output.mkdir(parents=True)
        self.config = config
        self.manifest_path = mechanism_manifest.expanduser().resolve(strict=True)
        self.experiment_protocol_path = experiment_protocol.expanduser().resolve(
            strict=True
        )
        self.experiment_protocol = self._load_experiment_protocol()
        self.official_repository = official_repository.expanduser().resolve(strict=True)
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(
            config.gpu_memory_fraction, self.device
        )
        torch.set_num_threads(CPU_THREADS)
        _seed_everything(config.seed)
        self.source = self._load_source()
        self.code_identity = training_code_identity(
            self.protocol, self.official_repository
        )
        self._configure_run_scope()
        self.progress: dict[str, object] | None = None
        self.training = STUSequence.open(
            self.data_root,
            protocol=protocol,
            partition="train",
            sequence_id=206,
            label_mode=LabelMode.REQUIRED,
        )
        self.validation = STUSequence.open(
            self.data_root,
            protocol=protocol,
            partition="train",
            sequence_id=201,
            label_mode=LabelMode.REQUIRED,
        )
        payload = torch.load(
            self.source["calibration_path"], map_location="cpu", weights_only=True
        )
        calibration = GeneratorCalibration.from_payload(payload)
        self.generator = CounterfactualGenerator(
            calibration.sensor,
            minimum_current_points=config.minimum_current_anomaly_points,
        )
        self.model = WindowDetectorPrototype.from_protocol(
            protocol, official_repository=self.official_repository
        ).to(self.device)
        if self.resume:
            self._verify_resume_identity()
        else:
            self._write_run()

    def _load_experiment_protocol(self) -> dict[str, object]:
        try:
            document = json.loads(
                self.experiment_protocol_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingError("Oracle temporal protocol is unreadable") from error
        if not isinstance(document, Mapping):
            raise TrainingError("Oracle temporal protocol must be an object")
        section_names = (
            "source_scope",
            "label_policy",
            "trajectory",
            "clean_select",
            "oracle_proposal",
            "direct_supervision",
            "explicit_null",
            "model",
            "training",
            "candidate_order_preflight",
            "gradient_audit",
            "micro_screen",
            "next_96_64",
            "validation",
            "cache",
            "recovery",
            "runtime_budget",
            "state_selection",
        )
        sections = {name: document.get(name) for name in section_names}
        if not all(isinstance(value, Mapping) for value in sections.values()):
            raise TrainingError("selector deconfounding protocol sections are incomplete")
        source_scope = sections["source_scope"]
        label_policy = sections["label_policy"]
        trajectory = sections["trajectory"]
        clean_select = sections["clean_select"]
        oracle_proposal = sections["oracle_proposal"]
        direct = sections["direct_supervision"]
        explicit_null = sections["explicit_null"]
        model = sections["model"]
        training = sections["training"]
        candidate_order = sections["candidate_order_preflight"]
        gradient_audit = sections["gradient_audit"]
        micro = sections["micro_screen"]
        continue_rule = micro.get("continue_rule")
        follow_up = sections["next_96_64"]
        cache = sections["cache"]
        recovery = sections["recovery"]
        runtime = sections["runtime_budget"]
        state_selection = sections["state_selection"]
        validation = sections["validation"]
        micro_train = tuple(micro.get("train_frames", ()))
        micro_order = tuple(micro.get("train_order", ()))
        micro_validation = tuple(micro.get("validation_frames", ()))
        audit_frames = tuple(gradient_audit.get("train_frames", ()))
        normal_safety_strata = validation.get("normal_safety_strata")
        order_tolerances = candidate_order.get("absolute_tolerances")
        expected_normal_safety_strata = {
            "stuff",
            "normal_instance",
            "observed_moving_normal_instance",
            "observed_static_normal_instance",
            "unresolved_normal_instance",
        }
        if (
            document.get("format") != ORACLE_TEMPORAL_PROTOCOL_FORMAT
            or document.get("status") != "exploratory_mechanism_only"
            or document.get("model_name")
            != "current-anchored factorized causal window encoder"
            or source_scope.get("manifest") != "results/oracle_source.json"
            or source_scope.get("manifest_format") != MECHANISM_MANIFEST_FORMAT
            or source_scope.get("manifest_is_an_exploratory_subset") is not True
            or source_scope.get("manifest_does_not_define_history_value_labels")
            is not True
            or source_scope.get("protocol_schema") != 28
            or source_scope.get("train_windows") != 24
            or source_scope.get("validation_windows") != 16
            or source_scope.get("all_selected_windows_have_four_uncorrupted_history_frames")
            is not True
            or tuple(source_scope.get("candidate_order_preflight_validation_frames", ()))
            != (28,)
            or source_scope.get(
                "candidate_order_preflight_frames_are_excluded_from_16_window_metrics"
            )
            is not True
            or source_scope.get("formal_training") is not False
            or label_policy.get("normal_206_native_raw_2_points") != 0
            or label_policy.get("normal_201_native_raw_2_points") != 0
            or tuple(trajectory.get("history_lengths", ())) != ORACLE_HISTORY_LENGTHS
            or trajectory.get("worlds_per_window") != 1
            or trajectory.get("static_rule")
            != "plans with radial_speed_mps=0 use zero radial and zero lateral velocity"
            or tuple(clean_select.get("real_slots_per_age", ())) != ("h_mix",)
            or clean_select.get("null_slots_per_age") != 1
            or clean_select.get("duplicate_real_candidates") is not False
            or tuple(oracle_proposal.get("real_slots_per_age", ()))
            != ("static", "object_motion")
            or oracle_proposal.get("null_slots_per_age") != 1
            or oracle_proposal.get("receives_clean_select_gradients") is not False
            or oracle_proposal.get("candidate_type_embedding") is not False
            or direct.get("used_only_by_arm") != "proposal_direct"
            or direct.get("truth_metadata_never_enters_q_k_v_context_or_classifier")
            is not True
            or direct.get("aggregation_softmax_scope")
            != "all real and per-age null candidates in the requested causal window"
            or direct.get("direct_softmax_scope")
            != (
                "independently within each history age over that age's real "
                "candidates and null, reusing the exact aggregation scores "
                "without new parameters"
            )
            or direct.get("match_formula")
            != (
                "L_match=-sum_(i,k) r_i log(sum_{j in P_(i,k)} "
                "a_direct_(i,j,k))/sum_(i,k) r_i over generated-object "
                "query-age pairs with any same-object candidate"
            )
            or direct.get("null_formula")
            != (
                "L_null=-sum_(i,k) r_i log(a_direct_(i,null,k))/sum_(i,k) "
                "r_i over generated-object query-age pairs with no same-object "
                "candidate but at least one valid competing real candidate; "
                "all-real-invalid cases are structural nulls and do not dilute "
                "the loss"
            )
            or tuple(direct.get("scales", ())) != ("p16", "p8", "p4")
            or float(direct.get("match_weight", math.nan)) != 1.0
            or float(direct.get("null_weight", math.nan)) != 1.0
            or float(direct.get("total_weight", math.nan)) != 1.0
            or explicit_null.get("one_candidate_per_scale_and_age") is not True
            or tuple(explicit_null.get("scales", ())) != ("p16", "p8", "p4")
            or float(explicit_null.get("feature_value", math.nan)) != 0.0
            or explicit_null.get("counts_as_real_history_support") is not False
            or model.get("free_point_gate") is not False
            or model.get("zero_history_content_implies_zero_correction") is not True
            or float(model.get("maximum_logit_correction", math.nan)) != 4.0
            or candidate_order.get("required_before_temporal_training") is not True
            or candidate_order.get("moving_validation_frame") != 28
            or tuple(candidate_order.get("evaluation_states", ()))
            != ("formal_temporal_initial", "deterministic_nonzero_model_copy")
            or tuple(candidate_order.get("orders", ()))
            != ("static_object", "object_static")
            or candidate_order.get("repetitions_per_order") != 2
            or float(candidate_order.get("repeat_noise_multiplier", math.nan)) != 4.0
            or not isinstance(candidate_order.get("nonzero_copy"), Mapping)
            or candidate_order["nonzero_copy"].get("seed") != 20260824
            or float(
                candidate_order["nonzero_copy"].get(
                    "standard_deviation", math.nan
                )
            )
            != 0.01
            or candidate_order["nonzero_copy"].get("formal_model_mutated")
            is not False
            or candidate_order["nonzero_copy"].get("optimizer_steps") != 0
            or not isinstance(order_tolerances, Mapping)
            or float(order_tolerances.get("point_and_scale", math.nan)) != 5.0e-6
            or float(order_tolerances.get("loss", math.nan)) != 1.0e-6
            or float(order_tolerances.get("parameter_gradient", math.nan)) != 1.0e-6
            or float(candidate_order.get("relative_tolerance", math.nan)) != 1.0e-5
            or float(training.get("safe_weight", math.nan)) != 1.0
            or training.get("optimizer") != "AdamW"
            or float(training.get("learning_rate", math.nan)) != 1.0e-4
            or float(training.get("weight_decay", math.nan)) != 1.0e-4
            or training.get("seed") != 20260813
            or float(training.get("magnitude_weight", math.nan)) != 0.1
            or float(training.get("smooth_l1_beta", math.nan)) != 1.0
            or float(training.get("gain_weight", math.nan)) != 0.0
            or training.get("rich_neutral_ordering") is not False
            or training.get("amp") is not False
            or tuple(training.get("arms", {}).keys()) != SELECTOR_ARMS
            or training.get("one_shared_current_head") is not True
            or training.get("identical_temporal_initial_state") is not True
            or training.get("independent_temporal_states_and_optimizers") is not True
            or training.get("select_proposal_joint_loss") is not False
            or state_selection.get("validation_based_pass_selection") is not False
            or tuple(micro.get("conditions", ()))
            != (
                "Current",
                "Clean-Select-4",
                "Proposal-Direct-4",
                "Proposal-Classification-4",
                "Null-4",
            )
            or tuple(micro.get("arms", ())) != SELECTOR_ARMS
            or not isinstance(continue_rule, Mapping)
            or continue_rule.get("per_arm_null_current_max_abs") != 0.0
            or continue_rule.get("per_arm_null_non_null_p4_support_points") != 0
            or continue_rule.get("moving_normal_sentinel_minimum_points") != 100
            or continue_rule.get("moving_normal_sentinel_minimum_windows") != 2
            or continue_rule.get(
                "proposal_direct_moving_normal_bce_not_worse_than_current"
            )
            is not True
            or continue_rule.get(
                "proposal_direct_moving_normal_bce_and_normal_up_not_worse_than_proposal_classification"
            )
            is not True
            or continue_rule.get(
                "direct_same_object_and_learnable_null_mass_improve_over_classification_at_all_scales"
            )
            is not True
            or micro.get("stage_a_passes") != 1
            or micro.get("stage_b_passes") != 1
            or tuple(micro.get("stage_b_history_lengths", ())) != (4,)
            or follow_up.get("implemented_in_this_protocol") is not False
            or follow_up.get("requires_passing_micro_screen") is not True
            or not isinstance(normal_safety_strata, Mapping)
            or set(normal_safety_strata) != expected_normal_safety_strata
            or _sha256_json(list(micro_train)) != micro.get("train_frames_sha256")
            or _sha256_json(list(micro_order)) != micro.get("train_order_sha256")
            or _sha256_json(list(micro_validation))
            != micro.get("validation_frames_sha256")
            or set(micro_order) != set(micro_train)
            or len(micro_train) != 24
            or len(micro_validation) != 16
            or _sha256_json(list(audit_frames)) != gradient_audit.get("frames_sha256")
            or len(audit_frames) != 8
            or not set(audit_frames).issubset(set(micro_train))
            or gradient_audit.get("report_format") != STAGE0_AUDIT_FORMAT
            or gradient_audit.get("audit_mode")
            != "zero_temporal_optimizer_updates_after_stage_a"
            or gradient_audit.get("optimizer_created") is not False
            or gradient_audit.get("optimizer_steps") != 0
            or gradient_audit.get("gradient_api") != "torch.autograd.grad"
            or gradient_audit.get("classification_and_direct_use_independent_forwards")
            is not True
            or tuple(gradient_audit.get("classification_terms", ()))
            != ("L_win", "L_safe", "0.1*L_mag")
            or tuple(gradient_audit.get("direct_terms", ()))
            != ("L_match", "L_null")
            or float(
                gradient_audit.get("overall_median_ratio_minimum", math.nan)
            )
            != 0.1
            or float(
                gradient_audit.get("overall_median_ratio_maximum", math.nan)
            )
            != 10.0
            or float(
                gradient_audit.get("per_scale_median_ratio_maximum", math.nan)
            )
            != 20.0
            or gradient_audit.get("undefined_zero_over_zero_json_value") is not None
            or gradient_audit.get(
                "temporal_point_delta_zero_over_zero_is_expected_at_initialization"
            )
            is not True
            or float(gradient_audit.get("lambda_dir", math.nan)) != 1.0
            or gradient_audit.get(
                "restore_formal_parameters_rng_mode_requires_grad_and_grad_buffers_in_finally"
            )
            is not True
            or gradient_audit.get("automatic_weight_change") is not False
            or tuple(gradient_audit.get("branches", ()))
            != (
                "temporal_p16",
                "temporal_p8",
                "temporal_p4",
                "temporal_point_delta",
            )
            or cache.get("persistent_render_or_feature_cache") is not False
            or cache.get("deterministic_rerender_from_frozen_plan") is not True
            or cache.get("window_local_fp32_materialization") is not True
            or cache.get("release_derived_window_after_use") is not True
            or cache.get("resume_recomputes_next_window_from_source") is not True
            or cache.get("precision") != "FP32"
            or recovery.get("window_level") is not True
            or recovery.get(
                "failure_runtime_saved_without_advancing_checkpoint_state"
            )
            is not True
            or recovery.get("stores_all_three_arm_states_and_optimizers") is not True
            or recovery.get("optimizer_retention_scope")
            != (
                "All three optimizer and RNG states are stored after every active "
                "stage-B window for exact recovery, then discarded after stage B "
                "is complete because validation performs no parameter updates; "
                "final model states remain stored."
            )
            or runtime.get("include_preparation_validation_and_final_save") is not True
            or runtime.get("include_source_verification_and_model_construction")
            is not True
            or runtime.get("accumulate_recovered_wall_seconds") is not True
            or runtime.get("projection_after_windows") != 8
            or float(runtime.get("window_quantile", math.nan)) != 0.9
            or runtime.get("include_first_window_in_projection") is not True
            or float(runtime.get("fixed_projection_overhead_seconds", math.nan))
            != 60.0
            or float(runtime.get("projection_contingency_multiplier", math.nan))
            != 1.2
            or float(runtime.get("maximum_wall_seconds", math.nan)) != 10800.0
            or float(runtime.get("final_result_write_reserve_seconds", math.nan))
            != 10.0
        ):
            raise TrainingError("selector deconfounding protocol changed its frozen design")
        return dict(document)

    def _configure_run_scope(self) -> None:
        """Select only the frozen 24/16 deconfounding screen."""

        section = self.experiment_protocol["micro_screen"]
        assert isinstance(section, Mapping)
        self.train_frames = tuple(int(value) for value in section["train_frames"])
        self.validation_frames = tuple(
            int(value) for value in section["validation_frames"]
        )
        self.train_orders = (tuple(int(value) for value in section["train_order"]),)
        self.current_passes = int(section["stage_a_passes"])
        self.temporal_passes = int(section["stage_b_passes"])
        self.temporal_lengths = tuple(
            int(value) for value in section["stage_b_history_lengths"]
        )
        if (
            len(self.train_orders) != self.current_passes
            or any(set(order) != set(self.train_frames) for order in self.train_orders)
            or self.temporal_passes != self.current_passes
        ):
            raise TrainingError("Oracle run scope has inconsistent fixed passes")

    def _verify_resume_identity(self) -> None:
        run_path = self.output / "run.json"
        progress_path = self.output / "progress.pt"
        if not run_path.is_file() or not progress_path.is_file():
            raise TrainingError("resume requires run.json and progress.pt")
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingError("resume run identity is unreadable") from error
        scope = run.get("scope") if isinstance(run, Mapping) else None
        if (
            not isinstance(run, Mapping)
            or run.get("format") != ORACLE_TEMPORAL_FORMAT
            or not isinstance(scope, Mapping)
            or scope.get("mode") != self.run_mode
            or run.get("manifest_sha256") != self.source["manifest_sha256"]
            or run.get("experiment_protocol_sha256")
            != _sha256_file(self.experiment_protocol_path)
            or run.get("code_identity") != self.code_identity
        ):
            raise TrainingError("resume identity differs from this invocation")
        progress = torch.load(progress_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(progress, Mapping)
            or progress.get("format") != ORACLE_TEMPORAL_PROGRESS_FORMAT
            or progress.get("mode") != self.run_mode
            or progress.get("manifest_sha256") != self.source["manifest_sha256"]
            or progress.get("experiment_protocol_sha256")
            != _sha256_file(self.experiment_protocol_path)
            or progress.get("code_identity") != self.code_identity
            or tuple(progress.get("train_frames", ())) != self.train_frames
            or tuple(tuple(value) for value in progress.get("train_orders", ()))
            != self.train_orders
        ):
            raise TrainingError("resume progress identity changed")
        current_state = progress.get("current_state")
        temporal_state = progress.get("temporal_state")
        if not isinstance(current_state, Mapping) or not isinstance(
            temporal_state, Mapping
        ):
            raise TrainingError("resume progress lacks model state")
        state = self.model.state_dict()
        expected = {name for name in state if name.startswith("point_anomaly_head.")}
        if set(current_state) != expected:
            raise TrainingError("resume current-head state is incomplete")
        expected_temporal = {
            name for name in state if name.startswith(self._temporal_prefixes)
        }
        if set(temporal_state) != expected_temporal:
            raise TrainingError("resume temporal state is incomplete")
        with torch.no_grad():
            for source in (current_state, temporal_state):
                for name, value in source.items():
                    if not isinstance(value, Tensor):
                        raise TrainingError("resume model state contains a non-tensor")
                    state[name].copy_(value.to(device=state[name].device))
        self.progress = dict(progress)
        rng = progress.get("rng")
        if not isinstance(rng, Mapping):
            raise TrainingError("resume progress lacks random state")
        self._restore_rng(rng)

    def _save_progress(
        self,
        stage: str,
        data: Mapping[str, object],
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        previous_data: dict[str, object] = {}
        if isinstance(self.progress, Mapping):
            value = self.progress.get("data")
            if isinstance(value, Mapping):
                previous_data.update(value)
        previous_data.update(data)
        previous_data["runtime_elapsed_wall_seconds"] = (
            self._observed_wall_seconds()
        )
        payload = {
            "format": ORACLE_TEMPORAL_PROGRESS_FORMAT,
            "mode": self.run_mode,
            "stage": stage,
            "manifest_sha256": self.source["manifest_sha256"],
            "experiment_protocol_sha256": _sha256_file(self.experiment_protocol_path),
            "code_identity": self.code_identity,
            "train_frames": list(self.train_frames),
            "train_orders": [list(order) for order in self.train_orders],
            "current_state": self._snapshot(self.model, ("point_anomaly_head.",)),
            "temporal_state": self._snapshot(self.model, self._temporal_prefixes),
            "current_head_sha256": _tensor_state_sha256(
                self._snapshot(self.model, ("point_anomaly_head.",))
            ),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "rng": self._rng_snapshot(),
            "data": previous_data,
        }
        _save_checkpoint(self.output / "progress.pt", payload)
        self.progress = payload

    def _progress_data(self) -> dict[str, object]:
        if not isinstance(self.progress, Mapping):
            return {}
        value = self.progress.get("data")
        return dict(value) if isinstance(value, Mapping) else {}

    def _progress_stage(self) -> str | None:
        if not isinstance(self.progress, Mapping):
            return None
        value = self.progress.get("stage")
        return value if isinstance(value, str) else None

    def _load_source(self) -> dict[str, object]:
        """Load the frozen source pool, raw identities, and truth trajectories."""

        try:
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingError("Oracle source manifest is unreadable") from error
        if not isinstance(document, Mapping):
            raise TrainingError("Oracle source manifest must be an object")
        payload = dict(document)
        declared_hash = payload.pop("payload_sha256", None)
        if document.get(
            "format"
        ) != MECHANISM_MANIFEST_FORMAT or declared_hash != _sha256_json(payload):
            raise TrainingError("Oracle source manifest identity is invalid")
        scope = document.get("scope")
        active = document.get("active_schema28_use")
        selection = document.get("selection")
        sources = document.get("sources")
        if not all(
            isinstance(value, Mapping)
            for value in (scope, active, selection, sources)
        ):
            raise TrainingError("Oracle source manifest sections are malformed")
        assert isinstance(scope, Mapping)
        assert isinstance(active, Mapping)
        assert isinstance(selection, Mapping)
        assert isinstance(sources, Mapping)
        if (
            scope.get("protocol_schema") != 28
            or scope.get("train_sequence") != 206
            or scope.get("validation_sequence") != 201
            or scope.get("train_windows") != 96
            or scope.get("validation_windows") != 64
            or scope.get("source_plan_epoch") != 0
            or scope.get("parameter_updates") != 0
            or scope.get("parameter_updates_scope")
            != "source-window and trajectory-plan selection only"
            or scope.get("selection_used_learned_model_or_baseline_outputs")
            is not False
        ):
            raise TrainingError(
                "Oracle source selection has the wrong scientific scope"
            )

        micro = self.experiment_protocol["micro_screen"]
        assert isinstance(micro, Mapping)
        if (
            active.get("role") != "deconfounded_selector_screen_only"
            or active.get("source_pool_is_inherited_96_64") is not True
            or tuple(active.get("active_train_frames", ()))
            != tuple(micro["train_frames"])
            or tuple(active.get("active_validation_frames", ()))
            != tuple(micro["validation_frames"])
            or tuple(active.get("preflight_only_validation_frames", ())) != (28,)
            or active.get("preflight_only_validation_frames_sha256")
            != _sha256_json([28])
            or active.get("preflight_frames_are_excluded_from_metrics") is not True
            or active.get("all_active_complete_windows_have_current_frame_at_least_8")
            is not True
        ):
            raise TrainingError("source pool does not bind the active 24/16 subset")

        def source_file(name: str) -> Path:
            value = sources.get(name)
            if not isinstance(value, Mapping):
                raise TrainingError(f"Oracle source {name!r} is missing")
            path_value = value.get("path")
            sha256 = value.get("sha256")
            if not isinstance(path_value, str) or not isinstance(sha256, str):
                raise TrainingError(f"Oracle source {name!r} is malformed")
            relative = Path(path_value)
            if relative.is_absolute():
                raise TrainingError(f"Oracle source {name!r} must be project-relative")
            path = (PROJECT_ROOT / relative).resolve(strict=True)
            try:
                path.relative_to(PROJECT_ROOT)
            except ValueError as error:
                raise TrainingError(
                    f"Oracle source {name!r} escapes the project"
                ) from error
            if _sha256_file(path) != sha256:
                raise TrainingError(f"Oracle source {name!r} changed")
            return path

        split_path = source_file("split")
        calibration_path = source_file("calibration")
        if split_path != self.protocol.path.resolve(strict=True):
            raise TrainingError("Oracle source manifest uses another protocol")

        raw_sequence_identity: dict[str, object] = {}
        for name, sequence in (
            ("raw_train_206", 206),
            ("raw_validation_201", 201),
        ):
            record = sources.get(name)
            if not isinstance(record, Mapping):
                raise TrainingError(f"Oracle source {name!r} is missing")
            root = record.get("root")
            identity = record.get("identity")
            if (
                root != f"train/{sequence}"
                or not isinstance(identity, Mapping)
            ):
                raise TrainingError(f"Oracle source {name!r} is malformed")
            observed = _source_tree_identity(self.data_root / str(root))
            if (
                observed.get("file_count") != identity.get("file_count")
                or observed.get("bytes") != identity.get("total_bytes")
                or observed.get("manifest_sha256")
                != identity.get("content_manifest_sha256")
            ):
                raise TrainingError(f"raw source tree changed for sequence {sequence}")
            raw_sequence_identity[str(sequence)] = dict(identity)

        def frames(name: str, expected: int, upper: int) -> tuple[int, ...]:
            values = selection.get(name)
            if not isinstance(values, list) or len(values) != expected:
                raise TrainingError(f"Oracle {name} must contain {expected} frames")
            result = tuple(values)
            if (
                any(type(value) is not int for value in result)
                or tuple(sorted(result)) != result
                or len(set(result)) != expected
                or result[0] < WINDOW_FRAMES - 1
                or result[-1] >= upper
            ):
                raise TrainingError(f"Oracle {name} is invalid")
            return result

        train_frames = frames("train_frames", 96, 449)
        validation_frames = frames("validation_frames", 64, 682)
        preflight_frames = tuple(active["preflight_only_validation_frames"])
        if (
            not set(preflight_frames).issubset(set(validation_frames))
            or set(preflight_frames)
            & {int(value) for value in micro["validation_frames"]}
        ):
            raise TrainingError("candidate preflight frame has the wrong source scope")
        raw_orders = selection.get("train_orders")
        if not isinstance(raw_orders, list) or len(raw_orders) != 3:
            raise TrainingError("Oracle training orders are missing")
        train_orders = tuple(tuple(order) for order in raw_orders)
        if any(
            len(order) != 96
            or any(type(frame) is not int for frame in order)
            or set(order) != set(train_frames)
            for order in train_orders
        ):
            raise TrainingError("Oracle training order is not a frame permutation")
        if tuple(selection.get("validation_order", ())) != validation_frames:
            raise TrainingError("Oracle validation order changed")

        raw_plans = selection.get("trajectory_plans")
        if not isinstance(raw_plans, Mapping):
            raise TrainingError("Oracle trajectory plans are missing")
        expected_keys = {
            _plan_key(role, 0, frame)
            for role, selected in (
                ("train", train_frames),
                ("validation", validation_frames),
            )
            for frame in selected
        }
        if set(raw_plans) != expected_keys:
            raise TrainingError("Oracle trajectory-plan membership changed")
        plans: dict[str, TrajectoryPlan] = {}
        for key, value in raw_plans.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                raise TrainingError("one Oracle trajectory plan is malformed")
            try:
                plans[key] = TrajectoryPlan(**dict(value))
            except (TypeError, ValueError) as error:
                raise TrainingError(
                    f"Oracle trajectory plan is invalid: {key}"
                ) from error

        return {
            "manifest_sha256": _sha256_file(self.manifest_path),
            "manifest_payload_sha256": declared_hash,
            "split_path": split_path,
            "calibration_path": calibration_path,
            "source_hashes": {
                "split": _sha256_file(split_path),
                "calibration": _sha256_file(calibration_path),
                "raw_train_206": raw_sequence_identity["206"][
                    "content_manifest_sha256"
                ],
                "raw_validation_201": raw_sequence_identity["201"][
                    "content_manifest_sha256"
                ],
            },
            "raw_sequence_identity": raw_sequence_identity,
            "plans": plans,
            "train_frames": train_frames,
            "validation_frames": validation_frames,
            "train_orders": train_orders,
        }

    def _write_run(self) -> None:
        _write_json(
            self.output / "run.json",
            {
                "format": ORACLE_TEMPORAL_FORMAT,
                "status": "initialized_no_parameter_updates",
                "scope": {
                    "mode": self.run_mode,
                    "train_sequence": 206,
                    "validation_sequence": 201,
                    "train_windows": len(self.train_frames),
                    "validation_windows": len(self.validation_frames),
                    "current_passes": self.current_passes,
                    "temporal_passes": self.temporal_passes,
                    "temporal_history_lengths": list(self.temporal_lengths),
                    "formal_training": False,
                    "learned_matcher": False,
                },
                "manifest": _project_relative_path(self.manifest_path),
                "manifest_sha256": self.source["manifest_sha256"],
                "experiment_protocol": _project_relative_path(
                    self.experiment_protocol_path
                ),
                "experiment_protocol_sha256": _sha256_file(
                    self.experiment_protocol_path
                ),
                "config": asdict(self.config),
                "code_identity": self.code_identity,
                "environment": {
                    "python": sys.version,
                    "executable": _project_relative_path(sys.executable),
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(self.device),
                },
            },
        )

    def _plan(self, role: str, frame: int) -> TrajectoryPlan:
        plans = self.source["plans"]
        assert isinstance(plans, Mapping)
        value = plans.get(_plan_key(role, 0, frame))
        if not isinstance(value, TrajectoryPlan):
            raise TrainingError(f"Oracle plan is missing for {role}:{frame}")
        return value

    def _realize(
        self, sequence: STUSequence, role: str, frame: int
    ) -> tuple[TrajectoryPlan, RenderedWindow]:
        """Deterministically rebuild one window without writing derived data."""

        plan = self._plan(role, frame)
        original = sequence.window(frame)
        rendered = _render_oracle_trajectory(original, plan, self.generator)
        return plan, rendered

    @staticmethod
    def _snapshot(
        model: WindowDetectorPrototype, prefixes: tuple[str, ...]
    ) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if name.startswith(prefixes)
        }

    def _restore_temporal(self, snapshot: Mapping[str, Tensor]) -> None:
        state = self.model.state_dict()
        if set(snapshot) != {
            name for name in state if name.startswith(self._temporal_prefixes)
        }:
            raise TrainingError("temporal initialization snapshot is incomplete")
        with torch.no_grad():
            for name, value in snapshot.items():
                state[name].copy_(value.to(device=state[name].device))

    def _set_trainable(self, parameters: Sequence[nn.Parameter]) -> list[nn.Parameter]:
        selected = list(parameters)
        identities = {id(parameter) for parameter in selected}
        self.model.requires_grad_(False)
        for parameter in self.model.parameters():
            if id(parameter) in identities:
                parameter.requires_grad_(True)
        if not selected or any(not parameter.requires_grad for parameter in selected):
            raise TrainingError("requested Oracle parameter stage was not enabled")
        return selected

    def _compile_current_features(
        self, sequence: STUSequence, role: str, frame: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compile one current window in memory and persist no derived tensors."""

        _, rendered = self._realize(sequence, role, frame)
        inputs = StaticInput.from_window(_current_counterfactual(rendered))
        anomaly, valid = _oracle_current_targets(rendered, self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.forward_current(inputs)
        features = prediction.features.detach().float()
        if (
            features.dtype != torch.float32
            or features.ndim != 2
            or features.shape[1] != 128
            or anomaly.dtype != torch.bool
            or valid.dtype != torch.bool
            or anomaly.shape != (features.shape[0],)
            or valid.shape != anomaly.shape
        ):
            raise TrainingError("compiled current feature tensors are malformed")
        return features, anomaly, valid

    def _compile_history_features(
        self,
        sequence: STUSequence,
        role: str,
        frame: int,
        *,
        include_validation_controls: bool,
        rendered: RenderedWindow | None = None,
    ) -> Mapping[str, object]:
        """Build one graph-free FP32 window package only in process memory."""

        if rendered is None:
            _, rendered = self._realize(sequence, role, frame)
        elif rendered.counterfactual.current_frame != frame:
            raise TrainingError("provided Oracle window does not match its frame")
        sampling = oracle_history_sampling(rendered)
        inputs = StaticInput.from_window(rendered.counterfactual)
        anomaly, valid = _oracle_current_targets(rendered, self.device)
        self.model.eval()
        with torch.no_grad():
            model_payload = self.model.prepare_frozen_history_cache(
                inputs,
                sampling,
                include_validation_controls=include_validation_controls,
            )
        payload = {
            "model": model_payload,
            "sampling": {
                "current_coordinates": torch.from_numpy(
                    np.array(sampling.current_coordinates, copy=True)
                ),
                "query_offsets": torch.from_numpy(
                    np.array(sampling.query_offsets, copy=True)
                ),
                "object_membership": torch.from_numpy(
                    np.array(sampling.object_membership, copy=True)
                ),
                "object_membership_by_age": [
                    torch.from_numpy(np.array(value, copy=True))
                    for value in sampling.object_membership_by_age or ()
                ],
            },
            "anomaly": anomaly.detach().cpu(),
            "valid": valid.detach().cpu(),
        }
        return payload

    def _cached_sampling_targets(
        self, payload: Mapping[str, object]
    ) -> tuple[Mapping[str, object], HistorySamplingOffsets, Tensor, Tensor]:
        model_payload = payload.get("model")
        sampling_payload = payload.get("sampling")
        anomaly = payload.get("anomaly")
        valid = payload.get("valid")
        if (
            not isinstance(model_payload, Mapping)
            or not isinstance(sampling_payload, Mapping)
            or not isinstance(anomaly, Tensor)
            or anomaly.dtype != torch.bool
            or not isinstance(valid, Tensor)
            or valid.dtype != torch.bool
        ):
            raise TrainingError("frozen history cache payload is malformed")

        def numpy(name: str, dtype: np.dtype) -> np.ndarray:
            value = sampling_payload.get(name)
            if not isinstance(value, Tensor):
                raise TrainingError(f"frozen history cache lacks sampling {name}")
            return np.ascontiguousarray(value.cpu().numpy(), dtype=dtype)

        by_age_payload = sampling_payload.get("object_membership_by_age")
        if not isinstance(by_age_payload, list) or not all(
            isinstance(value, Tensor) for value in by_age_payload
        ):
            raise TrainingError("frozen history cache lacks per-age object identity")
        sampling = HistorySamplingOffsets(
            current_coordinates=numpy("current_coordinates", np.dtype(np.float32)),
            query_offsets=numpy("query_offsets", np.dtype(np.float32)),
            object_membership=numpy("object_membership", np.dtype(np.bool_)),
            object_membership_by_age=tuple(
                np.ascontiguousarray(value.cpu().numpy(), dtype=np.bool_)
                for value in by_age_payload
            ),
        )
        return (
            model_payload,
            sampling,
            anomaly.to(self.device),
            valid.to(self.device),
        )

    def _train_current(self) -> list[dict[str, object]]:
        parameters = self._set_trainable(tuple(self.model.current_parameters()))
        optimizer = torch.optim.AdamW(
            parameters, lr=self.config.new_lr, weight_decay=self.config.weight_decay
        )
        progress_data = self._progress_data()
        stored_history = progress_data.get("current_training", [])
        if not isinstance(stored_history, list):
            raise TrainingError("stage-A recovery history is malformed")
        history = list(stored_history)
        if self._progress_stage() not in {None, "stage_a"}:
            if len(history) != self.current_passes:
                raise TrainingError("completed stage-A recovery history is incomplete")
            return history

        active = progress_data.get("stage_a_active")
        if active is not None and not isinstance(active, Mapping):
            raise TrainingError("stage-A active recovery state is malformed")
        if self._progress_stage() == "stage_a":
            optimizer_state = self.progress.get("optimizer") if self.progress else None
            if not isinstance(optimizer_state, Mapping):
                raise TrainingError("stage-A recovery lacks optimizer state")
            optimizer.load_state_dict(optimizer_state)
        timing_rows = progress_data.get("stage_a_timing", [])
        if not isinstance(timing_rows, list):
            raise TrainingError("stage-A timing recovery is malformed")
        timing_rows = list(timing_rows)
        self.model.train()
        for pass_index, order in enumerate(self.train_orders, 1):
            if pass_index <= len(history):
                continue
            if active is not None and int(active.get("pass_index", -1)) == pass_index:
                next_position = int(active.get("next_position", -1))
                raw_losses = active.get("losses")
                if (
                    next_position < 1
                    or next_position > len(order) + 1
                    or not isinstance(raw_losses, list)
                    or len(raw_losses) != next_position - 1
                ):
                    raise TrainingError("stage-A recovery cursor is inconsistent")
                losses = [float(value) for value in raw_losses]
                elapsed = float(active.get("elapsed_seconds", 0.0))
            else:
                next_position = 1
                losses = []
                elapsed = 0.0
            started = time.perf_counter()
            for position in range(next_position, len(order) + 1):
                self._enforce_hard_runtime_limit()
                frame = order[position - 1]
                window_started = time.perf_counter()
                torch.cuda.synchronize(self.device)
                preparation_started = time.perf_counter()
                features, anomaly, valid = self._compile_current_features(
                    self.training, "train", frame
                )
                torch.cuda.synchronize(self.device)
                preparation_seconds = time.perf_counter() - preparation_started
                update_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                logits = self.model.point_anomaly_head(features).squeeze(1)
                loss = _balanced_point_bce(logits, anomaly, valid)
                loss.backward()
                if not all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in parameters
                ):
                    raise TrainingError(
                        "current baseline produced non-finite gradients"
                    )
                self._enforce_hard_runtime_limit()
                optimizer.step()
                torch.cuda.synchronize(self.device)
                update_seconds = time.perf_counter() - update_started
                losses.append(float(loss.detach().cpu()))
                timing_rows.append(
                    {
                        "pass": pass_index,
                        "position": position,
                        "frame": frame,
                        "preparation_seconds": preparation_seconds,
                        "update_seconds": update_seconds,
                        "checkpoint_seconds": None,
                        "window_seconds_including_checkpoint": None,
                        "peak_gpu_bytes": int(
                            torch.cuda.max_memory_allocated(self.device)
                        ),
                    }
                )
                del logits, loss, features, anomaly, valid
                elapsed += time.perf_counter() - started
                started = time.perf_counter()
                checkpoint_started = time.perf_counter()
                self._save_progress(
                    "stage_a",
                    {
                        "current_training": history,
                        "stage_a_active": {
                            "pass_index": pass_index,
                            "next_position": position + 1,
                            "losses": losses,
                            "elapsed_seconds": elapsed,
                        },
                        "stage_a_timing": timing_rows,
                        "temporal_initial": self.temporal_initial_state,
                    },
                    optimizer=optimizer,
                )
                checkpoint_seconds = time.perf_counter() - checkpoint_started
                timing_rows[-1]["checkpoint_seconds"] = checkpoint_seconds
                timing_rows[-1]["window_seconds_including_checkpoint"] = (
                    time.perf_counter() - window_started
                )
                self._enforce_hard_runtime_limit()
                if position % 12 == 0:
                    torch.cuda.empty_cache()
                    print(
                        f"[Oracle current] pass={pass_index}/{self.current_passes}; "
                        f"window={position}/{len(order)}; loss={np.mean(losses):.6f}",
                        flush=True,
                    )
            elapsed += time.perf_counter() - started
            history.append(
                {
                    "pass": pass_index,
                    "loss": _oracle_distribution(losses),
                    "seconds": elapsed,
                    "updates": len(losses),
                }
            )
            active = None
            next_stage = (
                "stage_a_complete" if pass_index == self.current_passes else "stage_a"
            )
            self._save_progress(
                next_stage,
                {
                    "current_training": history,
                    "stage_a_active": None,
                    "stage_a_timing": timing_rows,
                    "temporal_initial": self.temporal_initial_state,
                },
                optimizer=None if next_stage == "stage_a_complete" else optimizer,
            )
        return history

    def _gradient_audit(self, parameters: Sequence[nn.Parameter]) -> dict[str, float]:
        totals = {prefix.rstrip("."): 0.0 for prefix in self._temporal_prefixes}
        for name, parameter in self.model.named_parameters():
            if id(parameter) not in {id(value) for value in parameters}:
                continue
            if parameter.grad is None:
                continue
            for prefix in self._temporal_prefixes:
                if name.startswith(prefix):
                    totals[prefix.rstrip(".")] += float(
                        parameter.grad.detach().float().square().sum().cpu()
                    )
                    break
        return {name: math.sqrt(value) for name, value in totals.items()}

    @staticmethod
    def _rng_snapshot() -> dict[str, object]:
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].copy()),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        }

    @staticmethod
    def _restore_rng(snapshot: Mapping[str, object]) -> None:
        numpy_state = snapshot.get("numpy")
        torch_state = snapshot.get("torch")
        cuda_state = snapshot.get("cuda")
        if (
            not isinstance(numpy_state, Mapping)
            or not isinstance(numpy_state.get("bit_generator"), str)
            or not isinstance(numpy_state.get("keys"), Tensor)
            or not isinstance(torch_state, Tensor)
            or not isinstance(cuda_state, list)
            or not all(isinstance(value, Tensor) for value in cuda_state)
        ):
            raise TrainingError("saved random state is malformed")
        random.setstate(snapshot["python"])  # type: ignore[arg-type]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.ascontiguousarray(
                    numpy_state["keys"].cpu().numpy(), dtype=np.uint32
                ),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)

    @staticmethod
    def _maximum_difference(left: Tensor, right: Tensor) -> float:
        if left.shape != right.shape:
            return math.inf
        return float((left.detach().float() - right.detach().float()).abs().max().cpu())

    def _stage0_invariance_checks(
        self, parameters: Sequence[nn.Parameter]
    ) -> dict[str, object]:
        """Check candidate permutation at formal and copied nonzero states."""

        tolerances = self.experiment_protocol["candidate_order_preflight"]
        assert isinstance(tolerances, Mapping)
        absolute = tolerances["absolute_tolerances"]
        assert isinstance(absolute, Mapping)
        atol_scale = float(absolute["point_and_scale"])
        atol_loss = float(absolute["loss"])
        atol_gradient = float(absolute["parameter_gradient"])
        rtol = float(tolerances["relative_tolerance"])
        repeat_multiplier = float(tolerances["repeat_noise_multiplier"])
        moving_frame = int(tolerances["moving_validation_frame"])

        self._enforce_hard_runtime_limit()
        moving_cache = self._compile_history_features(
            self.validation,
            "validation",
            moving_frame,
            include_validation_controls=True,
        )
        moving_model_cache, moving_sampling, anomaly, valid = (
            self._cached_sampling_targets(moving_cache)
        )
        self._enforce_hard_runtime_limit()

        def named_temporal(
            model: WindowDetectorPrototype,
        ) -> list[tuple[str, nn.Parameter]]:
            selected = {id(value) for value in model.temporal_parameters()}
            return [
                (name, value)
                for name, value in model.named_parameters()
                if id(value) in selected
            ]

        formal_named = named_temporal(self.model)
        if {id(value) for _, value in formal_named} != {id(value) for value in parameters}:
            raise TrainingError("candidate audit received another temporal parameter set")

        def candidate_order(
            model: WindowDetectorPrototype,
            named_parameters: Sequence[tuple[str, nn.Parameter]],
            order: str,
            rng: Mapping[str, object],
        ) -> dict[str, object]:
            self._enforce_hard_runtime_limit()
            self._restore_rng(rng)
            model.zero_grad(set_to_none=True)
            current, predictions = model.forward_cached_history_controls(
                moving_model_cache,
                sampling=moving_sampling,
                requests={
                    order: (
                        "oracle_proposal",
                        4,
                        "actual",
                        order,
                        "oracle_proposal",
                    ),
                },
            )
            prediction = predictions[order]
            classification = oracle_temporal_loss(
                prediction.logits,
                current.logits,
                anomaly,
                valid,
                safety_weight=1.0,
                magnitude_weight=0.1,
                smooth_l1_beta=1.0,
            ).total
            direct = proposal_match_null_loss(prediction).total
            parameter_values = tuple(value for _, value in named_parameters)
            gradients = torch.autograd.grad(
                classification + direct,
                parameter_values,
                allow_unused=True,
            )
            result = {
                "point_logits": prediction.logits.detach().clone(),
                "point_delta": (prediction.logits - current.logits).detach().clone(),
                "scale_residuals": tuple(
                    value.detach().clone() for value in prediction.scale_residuals
                ),
                "classification": classification.detach().clone(),
                "direct": direct.detach().clone(),
                "gradients": {
                    name: (
                        torch.zeros_like(parameter)
                        if gradient is None
                        else gradient.detach().clone()
                    )
                    for (name, parameter), gradient in zip(
                        named_parameters, gradients, strict=True
                    )
                },
            }
            del current, predictions, prediction, classification, direct, gradients
            self._enforce_hard_runtime_limit()
            return result

        def compare_state(
            model: WindowDetectorPrototype,
            named_parameters: Sequence[tuple[str, nn.Parameter]],
            *,
            require_nonzero: bool,
        ) -> dict[str, object]:
            common_rng = self._rng_snapshot()
            first = candidate_order(model, named_parameters, "static_object", common_rng)
            repeated = candidate_order(
                model, named_parameters, "static_object", common_rng
            )
            swapped = candidate_order(model, named_parameters, "object_static", common_rng)
            swapped_repeated = candidate_order(
                model, named_parameters, "object_static", common_rng
            )

            def compare(name: str, atol: float) -> dict[str, object]:
                return {
                    "name": name,
                    **_repeat_noise_relative_comparison(
                        first[name],
                        repeated[name],
                        swapped[name],
                        swapped_repeated[name],
                        absolute_tolerance=atol,
                        relative_tolerance=rtol,
                        repeat_noise_multiplier=repeat_multiplier,
                    ),
                }

            comparisons: dict[str, object] = {
                "point_logits": compare("point_logits", atol_scale),
                "classification_loss": compare("classification", atol_loss),
                "direct_loss": compare("direct", atol_loss),
            }
            for scale, index in zip(("p16", "p8", "p4"), range(3), strict=True):
                comparisons[f"{scale}_residual"] = {
                    "name": f"{scale}_residual",
                    **_repeat_noise_relative_comparison(
                        first["scale_residuals"][index],
                        repeated["scale_residuals"][index],
                        swapped["scale_residuals"][index],
                        swapped_repeated["scale_residuals"][index],
                        absolute_tolerance=atol_scale,
                        relative_tolerance=rtol,
                        repeat_noise_multiplier=repeat_multiplier,
                    ),
                }
            gradient_comparisons = {
                name: _repeat_noise_relative_comparison(
                    first["gradients"][name],
                    repeated["gradients"][name],
                    swapped["gradients"][name],
                    swapped_repeated["gradients"][name],
                    absolute_tolerance=atol_gradient,
                    relative_tolerance=rtol,
                    repeat_noise_multiplier=repeat_multiplier,
                )
                for name, _ in named_parameters
            }
            comparisons["parameter_gradients"] = {
                "parameters": gradient_comparisons,
                "maximum_repeat_noise": max(
                    float(value["repeat_noise"])
                    for value in gradient_comparisons.values()
                ),
                "maximum_swap_difference": max(
                    float(value["swap_difference"])
                    for value in gradient_comparisons.values()
                ),
                "passed": all(
                    bool(value["passed"]) for value in gradient_comparisons.values()
                ),
            }
            activation = {
                "point_delta_max_abs": float(
                    first["point_delta"].detach().float().abs().max().cpu()
                ),
                "scale_residual_max_abs_p16_p8_p4": [
                    float(value.detach().float().abs().max().cpu())
                    for value in first["scale_residuals"]
                ],
            }
            activation["passed"] = (
                not require_nonzero
                or (
                    float(activation["point_delta_max_abs"]) > 0.0
                    and all(
                        float(value) > 0.0
                        for value in activation[
                            "scale_residual_max_abs_p16_p8_p4"
                        ]
                    )
                )
            )
            passed = all(
                bool(value["passed"])
                for value in comparisons.values()
                if isinstance(value, Mapping)
            ) and bool(activation["passed"])
            return {
                "moving_validation_frame": moving_frame,
                "orders": ["static_object", "object_static"],
                "repetitions_per_order": 2,
                "comparison_rule": (
                    "repeat_noise <= absolute+relative base and swap_difference <= "
                    "base + repeat_noise_multiplier*repeat_noise"
                ),
                "repeat_noise_multiplier": repeat_multiplier,
                "activation": activation,
                "comparisons": comparisons,
                "passed": passed,
            }

        formal = compare_state(self.model, formal_named, require_nonzero=False)
        self._enforce_hard_runtime_limit()
        probe_spec = tolerances["nonzero_copy"]
        assert isinstance(probe_spec, Mapping)
        # Reconstruct sparse modules instead of deepcopying them: MinkowskiEngine
        # keeps required convolution metadata outside the tensor state.
        probe_model = WindowDetectorPrototype.from_protocol(
            self.protocol,
            official_repository=self.official_repository,
        ).to(self.device)
        try:
            self._enforce_hard_runtime_limit()
            probe_model.load_state_dict(self.model.state_dict(), strict=True)
            probe_model.requires_grad_(False)
            probe_named = named_temporal(probe_model)
            for _, parameter in probe_named:
                parameter.requires_grad_(True)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(probe_spec["seed"]))
            standard_deviation = float(probe_spec["standard_deviation"])
            maximum_perturbation = 0.0
            with torch.no_grad():
                for _, parameter in probe_named:
                    noise = torch.randn(
                        parameter.shape,
                        dtype=torch.float32,
                        device="cpu",
                        generator=generator,
                    ).to(device=parameter.device, dtype=parameter.dtype)
                    perturbation = standard_deviation * noise
                    parameter.add_(perturbation)
                    maximum_perturbation = max(
                        maximum_perturbation,
                        float(perturbation.detach().float().abs().max().cpu()),
                    )
            probe_model.eval()
            nonzero = compare_state(probe_model, probe_named, require_nonzero=True)
            nonzero["construction"] = {
                "independent_model_copy": True,
                "method": "fresh sparse model plus exact state_dict load",
                "seed": int(probe_spec["seed"]),
                "standard_deviation": standard_deviation,
                "maximum_abs_perturbation": maximum_perturbation,
                "optimizer_created": False,
                "optimizer_step_calls": 0,
            }
        finally:
            probe_model.zero_grad(set_to_none=True)
            del probe_model
            torch.cuda.empty_cache()
        self._enforce_hard_runtime_limit()

        self.model.zero_grad(set_to_none=True)
        self._enforce_hard_runtime_limit()
        with torch.no_grad():
            current, null_predictions = self.model.forward_cached_history_controls(
                moving_model_cache,
                sampling=moving_sampling,
                requests={
                    "Null-4": (
                        "oracle_select",
                        4,
                        "null",
                        "static_object",
                        "oracle_select",
                    )
                },
            )
        self._enforce_hard_runtime_limit()
        null_prediction = null_predictions["Null-4"]
        null_max = self._maximum_difference(null_prediction.logits, current.logits)
        null_support = int(null_prediction.point_history_support.sum().cpu())
        null_passed = torch.equal(null_prediction.logits, current.logits) and not bool(
            null_prediction.point_history_support.any()
        )

        static_frame = next(
            frame
            for frame in self.validation_frames
            if abs(self._plan("validation", frame).radial_speed_mps) <= 1.0e-12
        )
        self._enforce_hard_runtime_limit()
        static_cache = self._compile_history_features(
            self.validation,
            "validation",
            static_frame,
            include_validation_controls=True,
        )
        static_model_cache, static_sampling, _, _ = self._cached_sampling_targets(
            static_cache
        )
        self._enforce_hard_runtime_limit()
        with torch.no_grad():
            _, static_predictions = self.model.forward_cached_history_controls(
                static_model_cache,
                sampling=static_sampling,
                requests={
                    "Oracle": (
                        "oracle_select",
                        4,
                        "actual",
                        "static_object",
                        "oracle_select",
                    ),
                    "Fixed": (
                        "fixed_select",
                        4,
                        "actual",
                        "static_object",
                        "oracle_select",
                    ),
                    "Sham": (
                        "sham_select",
                        4,
                        "actual",
                        "static_object",
                        "oracle_select",
                    ),
                },
            )
        self._enforce_hard_runtime_limit()
        static_fixed = self._maximum_difference(
            static_predictions["Oracle"].logits, static_predictions["Fixed"].logits
        )
        static_sham = self._maximum_difference(
            static_predictions["Oracle"].logits, static_predictions["Sham"].logits
        )
        return {
            "formal_temporal_initial": formal,
            "deterministic_nonzero_model_copy": nonzero,
            "null_identity": {
                "point_max_abs": null_max,
                "non_null_p4_support_points": null_support,
                "passed": null_passed,
            },
            "static_identity": {
                "validation_frame": static_frame,
                "oracle_fixed_max_abs": static_fixed,
                "oracle_sham_max_abs": static_sham,
                "passed": static_fixed == 0.0 and static_sham == 0.0,
            },
            "passed": (
                bool(formal["passed"])
                and bool(nonzero["passed"])
                and null_passed
                and static_fixed == 0.0
                and static_sham == 0.0
            ),
        }

    def _loss_gradient_audit(self) -> dict[str, object]:
        """Measure formal temporal gradients without updating any formal state."""

        temporal_before = self._snapshot(self.model, self._temporal_prefixes)
        current_state_before = self._snapshot(self.model, ("point_anomaly_head.",))
        temporal_sha_before = _tensor_state_sha256(temporal_before)
        current_sha_before = _tensor_state_sha256(current_state_before)
        rng_before = self._rng_snapshot()
        mode_before = self.model.training
        requires_grad_before = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }
        gradients_before = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        report: dict[str, object] | None = None

        try:
            parameters = self._set_trainable(tuple(self.model.temporal_parameters()))
            parameter_ids = {id(value) for value in parameters}
            named_parameters = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if id(parameter) in parameter_ids
            ]
            parameter_values = tuple(parameter for _, parameter in named_parameters)
            audit_protocol = self.experiment_protocol["gradient_audit"]
            assert isinstance(audit_protocol, Mapping)
            audit_frames = tuple(
                int(value) for value in audit_protocol["train_frames"]
            )
            branch_names = tuple(prefix.rstrip(".") for prefix in self._temporal_prefixes)
            classification_union = {name: False for name in branch_names}
            direct_connectivity = {
                term: {name: False for name in branch_names}
                for term in ("L_match", "L_null")
            }

            def gradient_norms(
                gradients: Sequence[Tensor | None],
            ) -> dict[str, float]:
                squared = {name: 0.0 for name in branch_names}
                for (name, _), gradient in zip(
                    named_parameters, gradients, strict=True
                ):
                    if gradient is None:
                        continue
                    if not bool(torch.isfinite(gradient).all()):
                        raise TrainingError("stage-0 audit produced non-finite gradients")
                    for prefix, branch in zip(
                        self._temporal_prefixes, branch_names, strict=True
                    ):
                        if name.startswith(prefix):
                            squared[branch] += float(
                                gradient.detach().double().square().sum().cpu()
                            )
                            break
                result = {
                    name: math.sqrt(value) for name, value in squared.items()
                }
                result["selector"] = math.sqrt(
                    sum(
                        squared[name]
                        for name in (
                            "temporal_p16",
                            "temporal_p8",
                            "temporal_p4",
                        )
                    )
                )
                result["all"] = math.sqrt(sum(squared.values()))
                return result

            def strict_ratio(
                numerator: float, denominator: float
            ) -> tuple[float | None, str]:
                if not math.isfinite(numerator) or not math.isfinite(denominator):
                    return None, "non_finite"
                if denominator > 0.0:
                    return numerator / denominator, "finite"
                if numerator > 0.0:
                    return None, "positive_over_zero"
                return None, "both_zero"

            def objective_gradients(
                model_cache: Mapping[str, object],
                sampling: HistorySamplingOffsets,
                anomaly: Tensor,
                valid: Tensor,
                objective_name: str,
            ) -> dict[str, object]:
                self._enforce_hard_runtime_limit()
                self.model.zero_grad(set_to_none=True)
                current, predictions = self.model.forward_cached_history_controls(
                    model_cache,
                    sampling=sampling,
                    requests={
                        "Proposal": (
                            "oracle_proposal",
                            4,
                            "actual",
                            "static_object",
                            "oracle_proposal",
                        )
                    },
                )
                prediction = predictions["Proposal"]
                classification = oracle_temporal_loss(
                    prediction.logits,
                    current.logits,
                    anomaly,
                    valid,
                    safety_weight=1.0,
                    magnitude_weight=0.1,
                    smooth_l1_beta=1.0,
                )
                direct = proposal_match_null_loss(prediction)
                if objective_name == "classification":
                    value = classification.total
                    queries = int(valid.sum().item())
                elif objective_name == "direct":
                    value = direct.total
                    queries = sum(direct.match_queries_by_scale) + sum(
                        direct.null_queries_by_scale
                    )
                elif objective_name == "L_match":
                    value = direct.match
                    queries = sum(direct.match_queries_by_scale)
                elif objective_name == "L_null":
                    value = direct.null
                    queries = sum(direct.null_queries_by_scale)
                else:
                    raise TrainingError("unknown stage-0 gradient objective")
                gradients: tuple[Tensor | None, ...]
                if value.requires_grad:
                    gradients = torch.autograd.grad(
                        value, parameter_values, allow_unused=True
                    )
                else:
                    gradients = tuple(None for _ in parameter_values)
                result = {
                    "loss": float(value.detach().cpu()),
                    "queries": queries,
                    "branch_norms": gradient_norms(gradients),
                }
                del current, predictions, prediction, classification, direct, value
                self._enforce_hard_runtime_limit()
                return result

            rows: list[dict[str, object]] = []
            ratio_values = {
                name: []
                for name in (
                    "temporal_p16",
                    "temporal_p8",
                    "temporal_p4",
                    "selector",
                    "all",
                )
            }
            classification_norm_values = {name: [] for name in ratio_values}
            direct_norm_values = {name: [] for name in ratio_values}
            for frame in audit_frames:
                self._enforce_hard_runtime_limit()
                cache = self._compile_history_features(
                    self.training,
                    "train",
                    frame,
                    include_validation_controls=False,
                )
                model_cache, sampling, anomaly, valid = self._cached_sampling_targets(
                    cache
                )
                self._enforce_hard_runtime_limit()
                classification = objective_gradients(
                    model_cache, sampling, anomaly, valid, "classification"
                )
                direct = objective_gradients(
                    model_cache, sampling, anomaly, valid, "direct"
                )
                match = objective_gradients(
                    model_cache, sampling, anomaly, valid, "L_match"
                )
                null = objective_gradients(
                    model_cache, sampling, anomaly, valid, "L_null"
                )
                class_norms = classification["branch_norms"]
                direct_norms = direct["branch_norms"]
                assert isinstance(class_norms, Mapping) and isinstance(
                    direct_norms, Mapping
                )
                ratios: dict[str, float | None] = {}
                ratio_status: dict[str, str] = {}
                for name in (*branch_names, "selector", "all"):
                    value, status = strict_ratio(
                        float(direct_norms[name]), float(class_norms[name])
                    )
                    ratios[name] = value
                    ratio_status[name] = status
                    if name in ratio_values and value is not None:
                        ratio_values[name].append(value)
                        classification_norm_values[name].append(
                            float(class_norms[name])
                        )
                        direct_norm_values[name].append(float(direct_norms[name]))
                for name in branch_names:
                    classification_union[name] |= float(class_norms[name]) > 0.0
                for term_name, term in (("L_match", match), ("L_null", null)):
                    term_norms = term["branch_norms"]
                    assert isinstance(term_norms, Mapping)
                    if int(term["queries"]) > 0:
                        for name in branch_names:
                            direct_connectivity[term_name][name] |= (
                                float(term_norms[name]) > 0.0
                            )
                rows.append(
                    {
                        "frame": frame,
                        "classification": classification,
                        "direct": direct,
                        "L_match": match,
                        "L_null": null,
                        "direct_over_classification": ratios,
                        "ratio_status": ratio_status,
                    }
                )
                del cache, model_cache, sampling, anomaly, valid
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                self._enforce_hard_runtime_limit()

            required_ratio_names = (
                "temporal_p16",
                "temporal_p8",
                "temporal_p4",
                "all",
            )
            ratio_evaluable = all(
                len(ratio_values[name]) == len(audit_frames)
                for name in required_ratio_names
            )
            median_ratios = {
                name: (
                    float(np.median(np.asarray(values, dtype=np.float64)))
                    if values
                    else None
                )
                for name, values in ratio_values.items()
            }
            median_norms = {
                "classification": {
                    name: (
                        float(np.median(np.asarray(values, dtype=np.float64)))
                        if values
                        else None
                    )
                    for name, values in classification_norm_values.items()
                },
                "direct": {
                    name: (
                        float(np.median(np.asarray(values, dtype=np.float64)))
                        if values
                        else None
                    )
                    for name, values in direct_norm_values.items()
                },
            }
            overall_ratio = median_ratios["all"]
            ratio_gate = (
                ratio_evaluable
                and overall_ratio is not None
                and float(audit_protocol["overall_median_ratio_minimum"])
                <= overall_ratio
                <= float(audit_protocol["overall_median_ratio_maximum"])
                and all(
                    median_ratios[name] is not None
                    and float(median_ratios[name])
                    <= float(audit_protocol["per_scale_median_ratio_maximum"])
                    for name in (
                        "temporal_p16",
                        "temporal_p8",
                        "temporal_p4",
                    )
                )
            )
            scorer_branches = ("temporal_p16", "temporal_p8", "temporal_p4")
            connectivity_gate = all(
                classification_union[name] for name in scorer_branches
            ) and all(
                direct_connectivity[term][name]
                for term in ("L_match", "L_null")
                for name in scorer_branches
            )
            self._enforce_hard_runtime_limit()
            invariance = self._stage0_invariance_checks(parameters)
            self._enforce_hard_runtime_limit()
            passed = ratio_gate and connectivity_gate and bool(invariance["passed"])
            report = {
                "format": STAGE0_AUDIT_FORMAT,
                "passed": passed,
                "audit_frames": list(audit_frames),
                "rows": rows,
                "gradient_ratio": {
                    "definition": (
                        "per-window L2 norm of grad(L_match+L_null) divided by "
                        "the L2 norm of grad(L_win+L_safe+0.1*L_mag)"
                    ),
                    "median_norms": median_norms,
                    "median_direct_over_classification": median_ratios,
                    "all_required_ratios_evaluable": ratio_evaluable,
                    "overall_median_range": [
                        float(audit_protocol["overall_median_ratio_minimum"]),
                        float(audit_protocol["overall_median_ratio_maximum"]),
                    ],
                    "per_scale_median_maximum": float(
                        audit_protocol["per_scale_median_ratio_maximum"]
                    ),
                    "passed": ratio_gate,
                    "lambda_dir": 1.0,
                    "automatic_weight_change": False,
                },
                "classification_branch_connectivity": classification_union,
                "direct_supervision_branch_connectivity": direct_connectivity,
                "connectivity_passed": connectivity_gate,
                "invariance": invariance,
                "warnings": [
                    (
                        "temporal_point_delta is structurally 0/0 at the untouched "
                        "temporal initialization and is reported as JSON null"
                    )
                ],
                "zero_update": {
                    "scope": "stage-0 temporal audit after stage A",
                    "gradient_api": "torch.autograd.grad",
                    "optimizer_created": False,
                    "optimizer_step_calls": 0,
                    "formal_temporal_sha256_before": temporal_sha_before,
                    "current_head_sha256_before": current_sha_before,
                },
                "automatic_weight_change": False,
            }
        finally:
            self.model.zero_grad(set_to_none=True)
            state = self.model.state_dict()
            with torch.no_grad():
                for snapshot in (temporal_before, current_state_before):
                    for name, value in snapshot.items():
                        state[name].copy_(value.to(device=state[name].device))
            self._restore_rng(rng_before)
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(requires_grad_before[name])
                saved_gradient = gradients_before[name]
                parameter.grad = (
                    None
                    if saved_gradient is None
                    else saved_gradient.to(device=parameter.device).clone()
                )
            self.model.train(mode_before)

        temporal_sha_after = _tensor_state_sha256(
            self._snapshot(self.model, self._temporal_prefixes)
        )
        current_sha_after = _tensor_state_sha256(
            self._snapshot(self.model, ("point_anomaly_head.",))
        )
        rng_after = self._rng_snapshot()
        rng_restored = _nested_state_equal(rng_before, rng_after)
        requires_grad_restored = all(
            parameter.requires_grad == requires_grad_before[name]
            for name, parameter in self.model.named_parameters()
        )
        gradients_restored = all(
            (
                gradients_before[name] is None
                and parameter.grad is None
            )
            or (
                gradients_before[name] is not None
                and parameter.grad is not None
                and torch.equal(
                    gradients_before[name], parameter.grad.detach()
                )
            )
            for name, parameter in self.model.named_parameters()
        )
        mode_restored = self.model.training == mode_before
        state_restored = (
            temporal_sha_after == temporal_sha_before
            and current_sha_after == current_sha_before
            and rng_restored
            and requires_grad_restored
            and gradients_restored
            and mode_restored
        )
        if report is None:
            raise TrainingError("stage-0 audit produced no report")
        zero_update = report["zero_update"]
        assert isinstance(zero_update, dict)
        zero_update.update(
            {
                "formal_temporal_sha256_after": temporal_sha_after,
                "current_head_sha256_after": current_sha_after,
                "rng_restored_in_finally": rng_restored,
                "requires_grad_restored": requires_grad_restored,
                "gradient_buffers_restored": gradients_restored,
                "model_mode_restored": mode_restored,
                "formal_state_restored": state_restored,
            }
        )
        report["state_restored"] = state_restored
        report["passed"] = bool(report["passed"]) and state_restored
        _write_json(self.output / "stage0.json", report)
        if not bool(report["passed"]):
            raise TrainingError(
                "stage-0 zero-update numerical or gradient gate failed; "
                "see stage0.json for repeat-noise-relative diagnostics"
            )
        return report

    def _train_temporal(self) -> list[dict[str, object]]:
        """Train three independent selector arms from one identical initialization."""

        parameters = self._set_trainable(tuple(self.model.temporal_parameters()))
        optimizers = {
            arm: torch.optim.AdamW(
                parameters,
                lr=self.config.new_lr,
                weight_decay=self.config.weight_decay,
            )
            for arm in SELECTOR_ARMS
        }
        expected_names = set(self.temporal_initial_state)
        progress_data = self._progress_data()
        stored_history = progress_data.get("temporal_training", [])
        if not isinstance(stored_history, list):
            raise TrainingError("stage-B recovery history is malformed")
        history = list(stored_history)

        def checked_states(value: object) -> dict[str, dict[str, Tensor]]:
            if not isinstance(value, Mapping) or set(value) != set(SELECTOR_ARMS):
                raise TrainingError("stage-B recovery lacks all three arm states")
            result: dict[str, dict[str, Tensor]] = {}
            for arm in SELECTOR_ARMS:
                raw = value[arm]
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != expected_names
                    or not all(isinstance(tensor, Tensor) for tensor in raw.values())
                ):
                    raise TrainingError(f"stage-B state is malformed for {arm}")
                result[arm] = {
                    str(name): tensor.detach().cpu().clone()
                    for name, tensor in raw.items()
                }
            return result

        if self._progress_stage() not in {"stage0_complete", "stage_b"}:
            self.temporal_arm_states = checked_states(
                progress_data.get("temporal_arm_states")
            )
            if len(history) != self.temporal_passes:
                raise TrainingError("completed stage-B recovery history is incomplete")
            return history

        if self._progress_stage() == "stage_b":
            arm_states = checked_states(progress_data.get("temporal_arm_states"))
            optimizer_states = progress_data.get("temporal_arm_optimizers")
            arm_rng = progress_data.get("temporal_arm_rng")
            if (
                not isinstance(optimizer_states, Mapping)
                or set(optimizer_states) != set(SELECTOR_ARMS)
                or not isinstance(arm_rng, Mapping)
                or set(arm_rng) != set(SELECTOR_ARMS)
            ):
                raise TrainingError("stage-B recovery lacks independent optimizer/RNG state")
            for arm in SELECTOR_ARMS:
                state = optimizer_states[arm]
                rng = arm_rng[arm]
                if not isinstance(state, Mapping) or not isinstance(rng, Mapping):
                    raise TrainingError(f"stage-B recovery is malformed for {arm}")
                optimizers[arm].load_state_dict(state)
            active = progress_data.get("stage_b_active")
            if not isinstance(active, Mapping):
                raise TrainingError("stage-B recovery cursor is missing")
            next_position = int(active.get("next_position", -1))
            elapsed = float(active.get("elapsed_seconds", 0.0))
            raw_losses = active.get("losses")
            raw_components = active.get("components")
            raw_gradients = active.get("maximum_gradient_norm")
            if (
                next_position < 1
                or next_position > len(self.train_orders[0]) + 1
                or not isinstance(raw_losses, Mapping)
                or not isinstance(raw_components, Mapping)
                or not isinstance(raw_gradients, Mapping)
            ):
                raise TrainingError("stage-B recovery cursor is inconsistent")
            losses = {
                arm: [float(value) for value in raw_losses.get(arm, [])]
                for arm in SELECTOR_ARMS
            }
            if any(len(values) != next_position - 1 for values in losses.values()):
                raise TrainingError("stage-B arm update counts diverged")
            components = {
                arm: defaultdict(
                    list,
                    {
                        str(name): [float(value) for value in values]
                        for name, values in dict(raw_components.get(arm, {})).items()
                        if isinstance(values, list)
                    },
                )
                for arm in SELECTOR_ARMS
            }
            gradient_max = {
                arm: {
                    str(name): float(value)
                    for name, value in dict(raw_gradients.get(arm, {})).items()
                }
                for arm in SELECTOR_ARMS
            }
            arm_rng = {arm: arm_rng[arm] for arm in SELECTOR_ARMS}
        else:
            arm_states = {
                arm: {
                    name: value.detach().cpu().clone()
                    for name, value in self.temporal_initial_state.items()
                }
                for arm in SELECTOR_ARMS
            }
            initial_hashes = {
                arm: _tensor_state_sha256(state) for arm, state in arm_states.items()
            }
            if len(set(initial_hashes.values())) != 1:
                raise TrainingError("three temporal arms did not start byte-identically")
            empty_optimizer = optimizers[SELECTOR_ARMS[0]].state_dict()
            if any(
                optimizers[arm].state_dict() != empty_optimizer
                for arm in SELECTOR_ARMS[1:]
            ):
                raise TrainingError("three temporal optimizers differ at initialization")
            initial_rng = self._rng_snapshot()
            arm_rng = {
                arm: copy.deepcopy(initial_rng)
                for arm in SELECTOR_ARMS
            }
            next_position = 1
            elapsed = 0.0
            losses = {arm: [] for arm in SELECTOR_ARMS}
            components = {arm: defaultdict(list) for arm in SELECTOR_ARMS}
            gradient_max = {
                arm: {
                    prefix.rstrip("."): 0.0 for prefix in self._temporal_prefixes
                }
                for arm in SELECTOR_ARMS
            }

        timing_rows = progress_data.get("stage_b_timing", [])
        if not isinstance(timing_rows, list):
            raise TrainingError("stage-B timing recovery is malformed")
        timing_rows = list(timing_rows)
        order = self.train_orders[0]
        self.model.train()
        started = time.perf_counter()
        for position in range(next_position, len(order) + 1):
            self._enforce_hard_runtime_limit()
            frame = order[position - 1]
            window_started = time.perf_counter()
            torch.cuda.synchronize(self.device)
            preparation_started = time.perf_counter()
            cache = self._compile_history_features(
                self.training,
                "train",
                frame,
                include_validation_controls=False,
            )
            model_cache, sampling, anomaly, valid = self._cached_sampling_targets(cache)
            torch.cuda.synchronize(self.device)
            preparation_seconds = time.perf_counter() - preparation_started
            arm_seconds: dict[str, float] = {}
            for arm in SELECTOR_ARMS:
                self._enforce_hard_runtime_limit()
                self._restore_temporal(arm_states[arm])
                self._restore_rng(arm_rng[arm])
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                update_started = time.perf_counter()
                route = "oracle_select" if arm == "clean_select" else "oracle_proposal"
                current, predictions = self.model.forward_cached_history_controls(
                    model_cache,
                    sampling=sampling,
                    requests={
                        arm: (
                            route,
                            4,
                            "actual",
                            "static_object",
                            route,
                        )
                    },
                )
                prediction = predictions[arm]
                classification = oracle_temporal_loss(
                    prediction.logits,
                    current.logits,
                    anomaly,
                    valid,
                    safety_weight=1.0,
                    magnitude_weight=0.1,
                    smooth_l1_beta=1.0,
                )
                direct = (
                    proposal_match_null_loss(prediction)
                    if arm == "proposal_direct"
                    else None
                )
                loss = classification.total + (
                    direct.total if direct is not None else 0.0
                )
                loss.backward()
                if not all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in parameters
                ):
                    raise TrainingError(
                        f"temporal arm {arm} produced non-finite gradients"
                    )
                audit = self._gradient_audit(parameters)
                for name, value in audit.items():
                    gradient_max[arm][name] = max(
                        gradient_max[arm].get(name, 0.0), value
                    )
                self._enforce_hard_runtime_limit()
                optimizer.step()
                torch.cuda.synchronize(self.device)
                arm_seconds[arm] = time.perf_counter() - update_started
                arm_states[arm] = self._snapshot(self.model, self._temporal_prefixes)
                arm_rng[arm] = self._rng_snapshot()
                losses[arm].append(float(loss.detach().cpu()))
                for name in (
                    "window_bce",
                    "normal_safety",
                    "magnitude",
                    "anomaly_magnitude",
                    "normal_magnitude",
                ):
                    components[arm][name].append(
                        float(getattr(classification, name).detach().cpu())
                    )
                if direct is not None:
                    components[arm]["direct_total"].append(
                        float(direct.total.detach().cpu())
                    )
                    components[arm]["direct_match"].append(
                        float(direct.match.detach().cpu())
                    )
                    components[arm]["direct_null"].append(
                        float(direct.null.detach().cpu())
                    )
                    for scale, count in zip(
                        ("p16", "p8", "p4"),
                        direct.match_queries_by_scale,
                        strict=True,
                    ):
                        components[arm][f"{scale}_match_queries"].append(float(count))
                    for scale, count in zip(
                        ("p16", "p8", "p4"),
                        direct.null_queries_by_scale,
                        strict=True,
                    ):
                        components[arm][f"{scale}_null_queries"].append(float(count))
                    for scale, count in zip(
                        ("p16", "p8", "p4"),
                        direct.structural_null_queries_by_scale,
                        strict=True,
                    ):
                        components[arm][f"{scale}_structural_null_queries"].append(
                            float(count)
                        )
                del current, predictions, prediction, classification, direct, loss

            elapsed += time.perf_counter() - started
            started = time.perf_counter()
            if any(len(losses[arm]) != position for arm in SELECTOR_ARMS):
                raise TrainingError("three temporal arms received unequal exposure")
            timing_rows.append(
                {
                    "position": position,
                    "frame": frame,
                    "requested_K": 4,
                    "preparation_seconds": preparation_seconds,
                    "arm_update_seconds": arm_seconds,
                    "checkpoint_seconds": None,
                    "window_seconds_including_checkpoint": None,
                    "peak_gpu_bytes": int(
                        torch.cuda.max_memory_allocated(self.device)
                    ),
                }
            )
            self._restore_temporal(arm_states[SELECTOR_ARMS[-1]])
            checkpoint_started = time.perf_counter()
            self._save_progress(
                "stage_b",
                {
                    "temporal_training": history,
                    "temporal_arm_states": arm_states,
                    "temporal_arm_optimizers": {
                        arm: optimizers[arm].state_dict() for arm in SELECTOR_ARMS
                    },
                    "temporal_arm_rng": arm_rng,
                    "stage_b_active": {
                        "next_position": position + 1,
                        "losses": losses,
                        "components": {
                            arm: dict(values) for arm, values in components.items()
                        },
                        "maximum_gradient_norm": gradient_max,
                        "elapsed_seconds": elapsed,
                    },
                    "stage_b_timing": timing_rows,
                },
            )
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            timing_rows[-1]["checkpoint_seconds"] = checkpoint_seconds
            timing_rows[-1]["window_seconds_including_checkpoint"] = (
                time.perf_counter() - window_started
            )
            self._enforce_hard_runtime_limit()
            self._enforce_live_runtime_projection(
                stage="stage_b",
                completed_position=position,
                timing_rows=timing_rows,
            )
            del cache, model_cache, anomaly, valid
            if position % 8 == 0:
                torch.cuda.empty_cache()
                print(
                    f"[selector deconfounding] window={position}/{len(order)}; "
                    + "; ".join(
                        f"{arm}={np.mean(losses[arm]):.6f}"
                        for arm in SELECTOR_ARMS
                    ),
                    flush=True,
                )

        elapsed += time.perf_counter() - started
        if any(not losses[arm] for arm in SELECTOR_ARMS):
            raise TrainingError("one temporal arm received no updates")
        history.append(
            {
                "pass": 1,
                "seconds": elapsed,
                "updates_per_arm": len(order),
                "identical_window_order": list(order),
                "arms": {
                    arm: {
                        "loss": _oracle_distribution(losses[arm]),
                        "components": {
                            name: _oracle_distribution(values)
                            for name, values in components[arm].items()
                        },
                        "maximum_gradient_norm": gradient_max[arm],
                    }
                    for arm in SELECTOR_ARMS
                },
            }
        )
        self.temporal_arm_states = arm_states
        self._restore_temporal(arm_states[SELECTOR_ARMS[-1]])
        self._save_progress(
            "stage_b_complete",
            {
                "temporal_training": history,
                "temporal_arm_states": arm_states,
                "temporal_arm_optimizers": None,
                "temporal_arm_rng": None,
                "stage_b_active": None,
                "stage_b_timing": timing_rows,
            },
        )
        return history

    @staticmethod
    def _normal_safety_masks(
        rendered: RenderedWindow,
        anomaly: Tensor,
        valid: Tensor,
        device: torch.device,
    ) -> dict[str, Tensor]:
        """Partition current normal returns by raw instance and observed motion."""

        window = rendered.counterfactual
        if window.labels is None:
            raise TrainingError("normal safety strata require point labels")
        current_slice = window.members.current_slice
        current_instance = torch.from_numpy(
            np.array(
                window.labels.instance[current_slice],
                dtype=np.int64,
                order="C",
                copy=True,
            )
        ).to(device)
        current_group = torch.from_numpy(
            np.array(
                window.labels.group_key[current_slice],
                dtype=np.int64,
                order="C",
                copy=True,
            )
        ).to(device)
        if current_instance.shape != anomaly.shape:
            raise TrainingError("normal safety labels do not align with logits")
        normal = valid & ~anomaly
        normal_instance = normal & (current_instance > 0)
        masks = {
            "stuff": normal & (current_instance == 0),
            "normal_instance": normal_instance,
            "observed_moving_normal_instance": torch.zeros_like(normal),
            "observed_static_normal_instance": torch.zeros_like(normal),
            "unresolved_normal_instance": normal_instance.clone(),
        }

        semantic_value = window.labels.semantic_target
        if semantic_value is None:
            semantic = np.where(window.labels.semantic != np.uint16(0), 0, 255).astype(
                np.int64
            )
        else:
            semantic = semantic_value.astype(np.int64)
        coordinates = torch.from_numpy(
            np.array(
                window.members.coordinates_current,
                dtype=np.float32,
                order="C",
                copy=True,
            )
        ).to(device=device, dtype=torch.float32)
        ages = torch.from_numpy(
            np.array(
                window.members.frame_age,
                dtype=np.int64,
                order="C",
                copy=True,
            )
        ).to(device)
        group_key = torch.from_numpy(
            np.array(
                window.labels.group_key,
                dtype=np.int64,
                order="C",
                copy=True,
            )
        ).to(device)
        semantic_tensor = torch.from_numpy(
            np.array(semantic, dtype=np.int64, order="C", copy=True)
        ).to(device)
        displacement, displacement_valid = _observed_instance_displacements(
            coordinates, ages, group_key, semantic_tensor
        )
        for key in torch.unique(current_group[normal_instance], sorted=True):
            current_group_mask = normal_instance & (current_group == key)
            supported = displacement_valid & (group_key == key)
            if not bool(supported.any()):
                continue
            maximum_displacement = float(
                displacement[supported].norm(dim=1).max().cpu()
            )
            stratum = (
                "observed_moving_normal_instance"
                if maximum_displacement > INSTANCE_DISPLACEMENT_TOLERANCE_METRES
                else "observed_static_normal_instance"
            )
            masks[stratum] |= current_group_mask
            masks["unresolved_normal_instance"] &= ~current_group_mask
        return masks

    @staticmethod
    def _normal_safety_metrics(
        logits: Tensor,
        current_logits: Tensor,
        masks: Mapping[str, Tensor],
    ) -> dict[str, dict[str, float | int | None]]:
        delta = logits.float() - current_logits.detach().float()
        result: dict[str, dict[str, float | int | None]] = {}
        for name, mask in masks.items():
            count = int(mask.sum().cpu())
            if count == 0:
                result[name] = {
                    "points": 0,
                    "normal_bce": None,
                    "normal_up": None,
                    "abs_delta_median": None,
                    "abs_delta_q95": None,
                    "saturation_fraction": None,
                }
                continue
            selected_delta = delta[mask]
            absolute = selected_delta.abs()
            result[name] = {
                "points": count,
                "normal_bce": float(
                    torch.nn.functional.softplus(logits[mask].float()).mean().cpu()
                ),
                "normal_up": float(torch.relu(selected_delta).mean().cpu()),
                "abs_delta_median": float(absolute.median().cpu()),
                "abs_delta_q95": float(torch.quantile(absolute, 0.95).cpu()),
                "saturation_fraction": float((absolute >= 3.8).float().mean().cpu()),
            }
        return result

    @staticmethod
    def _condition_metrics(
        logits: Tensor,
        current_logits: Tensor,
        anomaly: Tensor,
        valid: Tensor,
    ) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        positive = valid & anomaly
        normal = valid & ~anomaly
        live = logits[valid].detach().float().cpu().numpy().astype(np.float64)
        truth = anomaly[valid].detach().cpu().numpy().astype(np.bool_)
        probability = 1.0 / (1.0 + np.exp(-np.clip(live, -80.0, 80.0)))
        delta = logits.float() - current_logits.detach().float()
        anomaly_loss = torch.nn.functional.softplus(-logits[positive].float()).mean()
        normal_loss = torch.nn.functional.softplus(logits[normal].float()).mean()
        normal_abs = delta[normal].abs()
        anomaly_abs = delta[positive].abs()
        metrics: dict[str, float | int] = {
            "balanced_bce": float((0.5 * (anomaly_loss + normal_loss)).cpu()),
            "ap": float(average_precision_score(truth, probability)),
            "auroc": float(roc_auc_score(truth, probability)),
            "anomaly_bce": float(anomaly_loss.cpu()),
            "normal_bce": float(normal_loss.cpu()),
            "normal_up": float(torch.relu(delta[normal]).mean().cpu()),
            "normal_abs_delta_median": float(normal_abs.median().cpu()),
            "normal_abs_delta_q95": float(torch.quantile(normal_abs, 0.95).cpu()),
            "normal_delta_above_0_5_fraction": float(
                (delta[normal] > 0.5).float().mean().cpu()
            ),
            "normal_delta_above_1_0_fraction": float(
                (delta[normal] > 1.0).float().mean().cpu()
            ),
            "normal_saturation_fraction": float(
                (normal_abs >= 3.8).float().mean().cpu()
            ),
            "anomaly_saturation_fraction": float(
                (anomaly_abs >= 3.8).float().mean().cpu()
            ),
            "anomaly_points": int(positive.sum().cpu()),
            "normal_points": int(normal.sum().cpu()),
        }
        return metrics, truth, live

    @staticmethod
    def _anomaly_support_metrics(
        logits: Tensor,
        current_logits: Tensor,
        anomaly: Tensor,
        valid: Tensor,
        support: Tensor,
    ) -> dict[str, object]:
        positive = valid & anomaly
        if support.shape != positive.shape or support.dtype != torch.bool:
            raise TrainingError("p4 support mask does not align with anomaly targets")
        delta = logits.float() - current_logits.detach().float()
        result: dict[str, object] = {}
        total = int(positive.sum().cpu())
        for name, mask in (
            ("supported", positive & support),
            ("unsupported", positive & ~support),
        ):
            count = int(mask.sum().cpu())
            if count == 0:
                result[name] = {
                    "points": 0,
                    "fraction_of_anomaly": 0.0,
                    "mean_logit_gain": None,
                    "median_logit_gain": None,
                    "positive_gain_fraction": None,
                    "anomaly_bce_improvement": None,
                    "maximum_absolute_correction": None,
                }
                continue
            selected = delta[mask]
            current_bce = torch.nn.functional.softplus(
                -current_logits[mask].detach().float()
            ).mean()
            window_bce = torch.nn.functional.softplus(-logits[mask].float()).mean()
            result[name] = {
                "points": count,
                "fraction_of_anomaly": count / total,
                "mean_logit_gain": float(selected.mean().cpu()),
                "median_logit_gain": float(selected.median().cpu()),
                "positive_gain_fraction": float((selected > 0).float().mean().cpu()),
                "anomaly_bce_improvement": float((current_bce - window_bce).cpu()),
                "maximum_absolute_correction": float(selected.abs().max().cpu()),
            }
        unsupported = positive & ~support
        if bool(unsupported.any()) and not torch.equal(
            logits[unsupported], current_logits[unsupported]
        ):
            raise TrainingError("p4-unsupported points received a temporal correction")
        return result

    @staticmethod
    def _pooled_metrics(
        truth: np.ndarray, logits: np.ndarray
    ) -> dict[str, float | int]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        positive = truth
        negative = ~truth
        return {
            "points": int(truth.size),
            "anomaly_points": int(positive.sum()),
            "anomaly_prevalence": float(positive.mean()),
            "balanced_bce": float(
                0.5
                * (
                    np.logaddexp(0.0, -logits[positive]).mean()
                    + np.logaddexp(0.0, logits[negative]).mean()
                )
            ),
            "ap": float(average_precision_score(truth, logits)),
            "auroc": float(roc_auc_score(truth, logits)),
        }

    @staticmethod
    def _summarize_normal_safety_strata(
        rows: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        if not rows:
            raise TrainingError("normal safety summary requires validation windows")
        first_conditions = rows[0].get("conditions")
        if not isinstance(first_conditions, Mapping):
            raise TrainingError("normal safety conditions are missing")
        summary: dict[str, object] = {}
        for condition in first_conditions:
            condition_summary: dict[str, object] = {}
            first_metrics = first_conditions[condition]
            if not isinstance(first_metrics, Mapping):
                raise TrainingError("normal safety condition is malformed")
            first_strata = first_metrics.get("normal_safety_by_stratum")
            if not isinstance(first_strata, Mapping):
                raise TrainingError("normal safety strata are missing")
            for stratum in first_strata:
                records: list[Mapping[str, object]] = []
                for row in rows:
                    conditions = row["conditions"]
                    assert isinstance(conditions, Mapping)
                    metrics = conditions[condition]
                    assert isinstance(metrics, Mapping)
                    strata = metrics["normal_safety_by_stratum"]
                    assert isinstance(strata, Mapping)
                    record = strata[stratum]
                    assert isinstance(record, Mapping)
                    if int(record["points"]) > 0:
                        records.append(record)
                points = sum(int(record["points"]) for record in records)
                if not records:
                    condition_summary[stratum] = {
                        "points": 0,
                        "windows_with_points": 0,
                        "point_weighted": None,
                        "window_balanced": None,
                    }
                    continue
                point_weighted = {
                    metric: float(
                        sum(
                            int(record["points"]) * float(record[metric])
                            for record in records
                        )
                        / points
                    )
                    for metric in (
                        "normal_bce",
                        "normal_up",
                        "saturation_fraction",
                    )
                }
                window_balanced = {
                    metric: _oracle_distribution(
                        [float(record[metric]) for record in records]
                    )
                    for metric in (
                        "normal_bce",
                        "normal_up",
                        "abs_delta_median",
                        "abs_delta_q95",
                        "saturation_fraction",
                    )
                }
                condition_summary[stratum] = {
                    "points": points,
                    "windows_with_points": len(records),
                    "point_weighted": point_weighted,
                    "window_balanced": window_balanced,
                }
            summary[str(condition)] = condition_summary
        return summary

    def _validate(self) -> dict[str, object]:
        """Evaluate the three final arm states on one shared frozen window cache."""

        progress_data = self._progress_data()
        completed = progress_data.get("validation_summary")
        if self._progress_stage() not in {"stage_b_complete", "validation"}:
            if not isinstance(completed, Mapping):
                raise TrainingError("completed validation recovery is missing")
            return dict(completed)
        stored_rows = progress_data.get("validation_rows", [])
        stored_fragments = progress_data.get("validation_fragments", [])
        timing_rows = progress_data.get("validation_timing", [])
        if (
            not isinstance(stored_rows, list)
            or not isinstance(stored_fragments, list)
            or not isinstance(timing_rows, list)
            or len(stored_rows) != len(stored_fragments)
        ):
            raise TrainingError("validation recovery state is malformed")
        rows = [dict(value) for value in stored_rows if isinstance(value, Mapping)]
        if len(rows) != len(stored_rows):
            raise TrainingError("validation recovery row is malformed")
        fragments = list(stored_fragments)
        timing_rows = list(timing_rows)
        pooled: dict[str, dict[str, dict[str, list[np.ndarray]]]] = defaultdict(
            lambda: defaultdict(lambda: {"truth": [], "logits": []})
        )
        for fragment in fragments:
            if not isinstance(fragment, Mapping):
                raise TrainingError("validation pooled recovery is malformed")
            truth = fragment.get("truth")
            logits = fragment.get("logits")
            subsets = fragment.get("subsets")
            if (
                not isinstance(truth, Tensor)
                or truth.dtype != torch.bool
                or not isinstance(logits, Mapping)
                or not isinstance(subsets, list)
            ):
                raise TrainingError("validation pooled recovery tensors are malformed")
            truth_array = truth.cpu().numpy().astype(np.bool_)
            for condition, values in logits.items():
                if not isinstance(condition, str) or not isinstance(values, Tensor):
                    raise TrainingError("validation pooled condition is malformed")
                live = values.cpu().numpy().astype(np.float64)
                for subset in subsets:
                    pooled[str(subset)][condition]["truth"].append(truth_array)
                    pooled[str(subset)][condition]["logits"].append(live)

        if set(self.temporal_arm_states) != set(SELECTOR_ARMS):
            raise TrainingError("validation lacks the three final temporal states")
        frames = self.validation_frames
        if len(rows) > len(frames) or [row.get("source_frame") for row in rows] != list(
            frames[: len(rows)]
        ):
            raise TrainingError("validation recovery order changed")
        self.model.eval()
        with torch.no_grad():
            for position in range(len(rows) + 1, len(frames) + 1):
                self._enforce_hard_runtime_limit()
                frame = frames[position - 1]
                window_started = time.perf_counter()
                plan, rendered = self._realize(self.validation, "validation", frame)
                parameters = rendered.proposal_parameters
                if not isinstance(parameters, Mapping):
                    raise TrainingError("trajectory metadata is missing")
                velocity = np.asarray(
                    parameters.get("velocity_world_mps"), dtype=np.float64
                )
                if velocity.shape != (3,) or not np.isfinite(velocity).all():
                    raise TrainingError("trajectory velocity is malformed")
                trajectory_speed = float(np.linalg.norm(velocity))
                moving = trajectory_speed > 1.0e-9
                torch.cuda.synchronize(self.device)
                preparation_started = time.perf_counter()
                cache = self._compile_history_features(
                    self.validation,
                    "validation",
                    frame,
                    include_validation_controls=False,
                    rendered=rendered,
                )
                model_cache, sampling, anomaly, valid = self._cached_sampling_targets(
                    cache
                )
                self._enforce_hard_runtime_limit()
                torch.cuda.synchronize(self.device)
                preparation_seconds = time.perf_counter() - preparation_started

                predictions: dict[str, object] = {}
                condition_logits: dict[str, Tensor] = {}
                reference_current: Tensor | None = None
                forward_seconds: dict[str, float] = {}
                null_names: dict[str, str] = {}
                for arm in SELECTOR_ARMS:
                    self._restore_temporal(self.temporal_arm_states[arm])
                    route = (
                        "oracle_select"
                        if arm == "clean_select"
                        else "oracle_proposal"
                    )
                    condition = SELECTOR_CONDITIONS[arm]
                    requests = {
                        condition: (
                            route,
                            4,
                            "actual",
                            "static_object",
                            route,
                        )
                    }
                    null_name = (
                        "Null-4" if arm == "proposal_direct" else f"{arm}-Null-4"
                    )
                    null_names[arm] = null_name
                    requests[null_name] = (
                        route,
                        4,
                        "null",
                        "static_object",
                        route,
                    )
                    torch.cuda.synchronize(self.device)
                    forward_started = time.perf_counter()
                    current, arm_predictions = (
                        self.model.forward_cached_history_controls(
                            model_cache,
                            sampling=sampling,
                            requests=requests,
                        )
                    )
                    torch.cuda.synchronize(self.device)
                    forward_seconds[arm] = time.perf_counter() - forward_started
                    if reference_current is None:
                        reference_current = current.logits
                        condition_logits["Current"] = reference_current
                    elif not torch.equal(reference_current, current.logits):
                        raise TrainingError(
                            "three arms changed the frozen current-frame logits"
                        )
                    for name, prediction in arm_predictions.items():
                        predictions[name] = prediction
                        condition_logits[name] = prediction.logits
                assert reference_current is not None
                null_identity: dict[str, dict[str, float | int]] = {}
                for arm, null_name in null_names.items():
                    null_prediction = predictions[null_name]
                    maximum = self._maximum_difference(
                        condition_logits[null_name], reference_current
                    )
                    support_points = int(
                        null_prediction.point_history_support.sum().cpu()
                    )
                    if not torch.equal(
                        condition_logits[null_name], reference_current
                    ):
                        raise TrainingError(
                            f"{arm} null is not a strict current identity"
                        )
                    if support_points != 0:
                        raise TrainingError(f"{arm} null was counted as p4 support")
                    null_identity[arm] = {
                        "current_max_abs": maximum,
                        "non_null_p4_support_points": support_points,
                    }

                normal_masks = self._normal_safety_masks(
                    rendered, anomaly, valid, self.device
                )
                subsets = [
                    "all",
                    "moving" if moving else "static",
                ]
                fragment_truth: np.ndarray | None = None
                fragment_logits: dict[str, Tensor] = {}
                condition_metrics: dict[str, object] = {}
                for name in (
                    "Current",
                    "Clean-Select-4",
                    "Proposal-Direct-4",
                    "Proposal-Classification-4",
                    "Null-4",
                ):
                    logits = condition_logits[name]
                    metrics, truth, live = self._condition_metrics(
                        logits, reference_current, anomaly, valid
                    )
                    if fragment_truth is None:
                        fragment_truth = truth
                    elif not np.array_equal(fragment_truth, truth):
                        raise TrainingError("validation conditions changed target points")
                    fragment_logits[name] = torch.from_numpy(live.copy())
                    metrics["normal_safety_by_stratum"] = self._normal_safety_metrics(
                        logits, reference_current, normal_masks
                    )
                    if name != "Current":
                        prediction = predictions[name]
                        metrics["point_history_support_fraction"] = float(
                            prediction.point_history_support.float().mean().cpu()
                        )
                        metrics["history_coverage_p16_p8_p4"] = (
                            prediction.history_coverage.float().cpu().tolist()
                        )
                        if name in {
                            "Clean-Select-4",
                            "Proposal-Direct-4",
                            "Proposal-Classification-4",
                        }:
                            metrics["anomaly_by_p4_support"] = (
                                self._anomaly_support_metrics(
                                    logits,
                                    reference_current,
                                    anomaly,
                                    valid,
                                    prediction.point_history_support,
                                )
                            )
                    condition_metrics[name] = metrics
                    for subset in subsets:
                        pooled[subset][name]["truth"].append(truth)
                        pooled[subset][name]["logits"].append(live)

                proposal_diagnostics: dict[str, object] = {}
                for proposal_condition in (
                    "Proposal-Direct-4",
                    "Proposal-Classification-4",
                ):
                    proposal_prediction = predictions[proposal_condition]
                    direct = proposal_match_null_loss(proposal_prediction)
                    match_scales: dict[str, object] = {}
                    assert proposal_prediction.match_mass_by_scale is not None
                    for (
                        scale,
                        mass,
                        match_loss,
                        null_loss,
                        match_count,
                        null_count,
                        structural_null_count,
                    ) in zip(
                        ("p16", "p8", "p4"),
                        proposal_prediction.match_mass_by_scale,
                        direct.match_by_scale,
                        direct.null_by_scale,
                        direct.match_queries_by_scale,
                        direct.null_queries_by_scale,
                        direct.structural_null_queries_by_scale,
                        strict=True,
                    ):
                        weights = mass.target_weight[:, None].expand_as(
                            mass.direct_same_object
                        )
                        eligible = weights > 0.0
                        match_target = eligible & mass.direct_has_same_object
                        null_target = (
                            eligible
                            & ~mass.direct_has_same_object
                            & mass.direct_real_valid
                        )

                        def weighted_mean(
                            value: Tensor, selected: Tensor
                        ) -> float | None:
                            if not bool(selected.any()):
                                return None
                            selected_weight = weights[selected].float()
                            return float(
                                (
                                    selected_weight * value[selected].float()
                                ).sum().div(selected_weight.sum()).cpu()
                            )

                        match_scales[scale] = {
                            "match_queries": match_count,
                            "learnable_null_queries": null_count,
                            "structural_null_queries": structural_null_count,
                            "same_object_attention_mass": weighted_mean(
                                mass.direct_same_object, match_target
                            ),
                            "null_attention_mass_on_null_targets": weighted_mean(
                                mass.direct_null, null_target
                            ),
                            "match_loss": float(match_loss.cpu()),
                            "null_loss": float(null_loss.cpu()),
                        }
                    proposal_diagnostics[proposal_condition] = {
                        "total": float(direct.total.cpu()),
                        "match": float(direct.match.cpu()),
                        "null": float(direct.null.cpu()),
                        "scales": match_scales,
                    }

                assert fragment_truth is not None
                row = {
                    "source_frame": frame,
                    "trajectory_speed_mps": trajectory_speed,
                    "planned_radial_speed_mps": plan.radial_speed_mps,
                    "motion_stratum": "moving" if moving else "static",
                    "available_history_frames": 4,
                    "conditions": condition_metrics,
                    "proposal_supervision_diagnostics": proposal_diagnostics,
                    "null_identity_by_arm": null_identity,
                }
                rows.append(row)
                fragments.append(
                    {
                        "source_frame": frame,
                        "subsets": subsets,
                        "truth": torch.from_numpy(fragment_truth.copy()),
                        "logits": fragment_logits,
                    }
                )
                timing_rows.append(
                    {
                        "position": position,
                        "frame": frame,
                        "preparation_seconds": preparation_seconds,
                        "arm_forward_seconds": forward_seconds,
                        "window_seconds_without_checkpoint": (
                            time.perf_counter() - window_started
                        ),
                        "checkpoint_seconds": None,
                        "window_seconds_including_checkpoint": None,
                        "peak_gpu_bytes": int(
                            torch.cuda.max_memory_allocated(self.device)
                        ),
                    }
                )
                checkpoint_started = time.perf_counter()
                self._save_progress(
                    "validation",
                    {
                        "validation_rows": rows,
                        "validation_fragments": fragments,
                        "validation_timing": timing_rows,
                        "validation_next_position": position + 1,
                    },
                )
                checkpoint_seconds = time.perf_counter() - checkpoint_started
                timing_rows[-1]["checkpoint_seconds"] = checkpoint_seconds
                timing_rows[-1]["window_seconds_including_checkpoint"] = (
                    time.perf_counter() - window_started
                )
                self._enforce_hard_runtime_limit()
                self._enforce_live_runtime_projection(
                    stage="validation",
                    completed_position=position,
                    timing_rows=timing_rows,
                )
                del cache, model_cache, anomaly, valid, predictions
                if position % 8 == 0:
                    torch.cuda.empty_cache()
                    print(
                        f"[selector validation] {position}/{len(frames)}",
                        flush=True,
                    )

        summary_started = time.perf_counter()
        pooled_summary = {
            subset: {
                condition: self._pooled_metrics(
                    np.concatenate(values["truth"]),
                    np.concatenate(values["logits"]),
                )
                for condition, values in conditions.items()
            }
            for subset, conditions in pooled.items()
        }
        identity_summary: dict[str, object] = {}
        for arm in SELECTOR_ARMS:
            records = [row["null_identity_by_arm"][arm] for row in rows]
            identity_summary[arm] = {
                "current_max_abs": _oracle_distribution(
                    [float(record["current_max_abs"]) for record in records]
                ),
                "non_null_p4_support_points": sum(
                    int(record["non_null_p4_support_points"])
                    for record in records
                ),
            }

        proposal_summary: dict[str, object] = {}
        for condition in (
            "Proposal-Direct-4",
            "Proposal-Classification-4",
        ):
            diagnostics = [
                row["proposal_supervision_diagnostics"][condition]
                for row in rows
            ]
            scale_summary: dict[str, object] = {}
            for scale in ("p16", "p8", "p4"):
                scale_rows = [record["scales"][scale] for record in diagnostics]
                same_values = [
                    float(record["same_object_attention_mass"])
                    for record in scale_rows
                    if record["same_object_attention_mass"] is not None
                ]
                null_values = [
                    float(record["null_attention_mass_on_null_targets"])
                    for record in scale_rows
                    if record["null_attention_mass_on_null_targets"] is not None
                ]
                scale_summary[scale] = {
                    "match_queries": sum(
                        int(record["match_queries"]) for record in scale_rows
                    ),
                    "learnable_null_queries": sum(
                        int(record["learnable_null_queries"])
                        for record in scale_rows
                    ),
                    "structural_null_queries": sum(
                        int(record["structural_null_queries"])
                        for record in scale_rows
                    ),
                    "same_object_attention_mass": (
                        _oracle_distribution(same_values) if same_values else None
                    ),
                    "null_attention_mass_on_null_targets": (
                        _oracle_distribution(null_values) if null_values else None
                    ),
                }
            proposal_summary[condition] = {
                "window_total": _oracle_distribution(
                    [float(record["total"]) for record in diagnostics]
                ),
                "window_match": _oracle_distribution(
                    [float(record["match"]) for record in diagnostics]
                ),
                "window_null": _oracle_distribution(
                    [float(record["null"]) for record in diagnostics]
                ),
                "scales": scale_summary,
            }

        direct_minus_classification: dict[str, object] = {
            term: _oracle_distribution(
                [
                    float(
                        row["proposal_supervision_diagnostics"][
                            "Proposal-Direct-4"
                        ][term]
                    )
                    - float(
                        row["proposal_supervision_diagnostics"][
                            "Proposal-Classification-4"
                        ][term]
                    )
                    for row in rows
                ]
            )
            for term in ("total", "match", "null")
        }
        scale_differences: dict[str, object] = {}
        for scale in ("p16", "p8", "p4"):
            metric_differences: dict[str, object] = {}
            for metric in (
                "same_object_attention_mass",
                "null_attention_mass_on_null_targets",
            ):
                values = []
                for row in rows:
                    diagnostics = row["proposal_supervision_diagnostics"]
                    left = diagnostics["Proposal-Direct-4"]["scales"][scale][
                        metric
                    ]
                    right = diagnostics["Proposal-Classification-4"]["scales"][
                        scale
                    ][metric]
                    if left is not None and right is not None:
                        values.append(float(left) - float(right))
                metric_differences[metric] = (
                    _oracle_distribution(values) if values else None
                )
            scale_differences[scale] = metric_differences
        direct_minus_classification["scales"] = scale_differences
        summary = {
            "windows": rows,
            "strata": {
                "manifest_windows": len(rows),
                "static_windows": sum(row["motion_stratum"] == "static" for row in rows),
                "moving_windows": sum(row["motion_stratum"] == "moving" for row in rows),
            },
            "pooled": pooled_summary,
            "comparisons": self._comparisons(rows),
            "normal_safety_strata": {
                "definitions": {
                    "minimum_current_and_historical_points_per_instance": (
                        MINIMUM_INSTANCE_CENTROID_POINTS
                    ),
                    "moving_displacement_threshold_metres": (
                        INSTANCE_DISPLACEMENT_TOLERANCE_METRES
                    ),
                    "motion_is_observed_centroid_proxy_not_ground_truth": True,
                },
                "summary": self._summarize_normal_safety_strata(rows),
            },
            "identity_controls": {"null_by_arm": identity_summary},
            "proposal_supervision_diagnostics": {
                "by_condition": proposal_summary,
                "proposal_direct_minus_classification": (
                    direct_minus_classification
                ),
            },
        }
        summary_seconds = time.perf_counter() - summary_started
        summary["summary_seconds"] = summary_seconds
        self._save_progress(
            "validation_complete",
            {
                "validation_summary": summary,
                "validation_rows": rows,
                "validation_fragments": None,
                "validation_timing": timing_rows,
                "validation_next_position": len(frames) + 1,
                "validation_summary_seconds": summary_seconds,
            },
        )
        return summary

    def _paired_summary(
        self,
        rows: Sequence[Mapping[str, object]],
        left: str,
        right: str,
        metric: str,
    ) -> dict[str, object]:
        values = np.asarray(
            [
                float(row["conditions"][left][metric])
                - float(row["conditions"][right][metric])
                for row in rows
            ],
            dtype=np.float64,
        )
        generator = np.random.default_rng(
            np.random.SeedSequence(
                (self.config.seed, sum(map(ord, left + right + metric)))
            )
        )
        repetitions = 100_000
        means = np.empty(repetitions, dtype=np.float64)
        for start in range(0, repetitions, 10_000):
            stop = min(start + 10_000, repetitions)
            indices = generator.integers(
                0, values.size, size=(stop - start, values.size)
            )
            means[start:stop] = values[indices].mean(axis=1)
        return {
            "windows": int(values.size),
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "positive_fraction": float(np.mean(values > 0.0)),
            "cluster_bootstrap_95_ci": [
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)),
            ],
            "bootstrap_repetitions": repetitions,
        }

    def _comparisons(self, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        all_rows = list(rows)
        moving = [row for row in rows if row["motion_stratum"] == "moving"]
        definitions = (
            (
                "clean_minus_current",
                all_rows,
                "Clean-Select-4",
                "Current",
            ),
            (
                "proposal_direct_minus_current",
                all_rows,
                "Proposal-Direct-4",
                "Current",
            ),
            (
                "proposal_direct_minus_classification",
                all_rows,
                "Proposal-Direct-4",
                "Proposal-Classification-4",
            ),
            (
                "proposal_direct_minus_clean",
                all_rows,
                "Proposal-Direct-4",
                "Clean-Select-4",
            ),
            (
                "proposal_classification_minus_clean",
                all_rows,
                "Proposal-Classification-4",
                "Clean-Select-4",
            ),
            (
                "proposal_direct_minus_classification_moving",
                moving,
                "Proposal-Direct-4",
                "Proposal-Classification-4",
            ),
            (
                "null_minus_current",
                all_rows,
                "Null-4",
                "Current",
            ),
        )
        return {
            name: {
                metric: self._paired_summary(selected, left, right, metric)
                for metric in ("balanced_bce", "ap", "auroc")
            }
            for name, selected, left, right in definitions
        }

    @staticmethod
    def _timing_quantile(
        rows: object,
        value,
        *,
        include_first: bool = False,
        quantile: float = 0.9,
    ) -> float:
        if not isinstance(rows, list):
            raise TrainingError("runtime timing rows are missing")
        values = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TrainingError("runtime timing row is malformed")
            if index == 0 and not include_first and len(rows) > 1:
                continue
            measured = float(value(row))
            if not math.isfinite(measured) or measured < 0.0:
                raise TrainingError("runtime timing contains an invalid duration")
            values.append(measured)
        if not values:
            raise TrainingError("runtime profile lacks the required observations")
        return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))

    def _enforce_hard_runtime_limit(self) -> None:
        """Stop only after a recoverable checkpoint if this run reaches three hours."""

        maximum = float(
            self.experiment_protocol["runtime_budget"]["maximum_wall_seconds"]
        )
        if self._observed_wall_seconds() >= maximum:
            raise TrainingError(
                f"the recoverable 24/16 run reached its {maximum:g}-second hard wall"
            )

    def _observed_wall_seconds(self) -> float:
        """Return command wall time accumulated across recoverable invocations."""

        started = getattr(self, "_run_wall_started", None)
        prior = float(getattr(self, "_run_wall_prior_seconds", 0.0))
        if started is None or not math.isfinite(prior) or prior < 0.0:
            raise TrainingError("runtime recovery lacks a valid cumulative wall time")
        current = time.perf_counter() - float(started)
        if not math.isfinite(current) or current < 0.0:
            raise TrainingError("runtime clock produced an invalid duration")
        return prior + current

    def _checkpoint_failure_runtime(self) -> None:
        """Persist elapsed wall time without advancing the last valid checkpoint."""

        if not isinstance(self.progress, Mapping):
            return
        progress_path = self.output / "progress.pt"
        if not progress_path.is_file():
            return
        payload = copy.deepcopy(dict(self.progress))
        data = payload.get("data")
        if not isinstance(data, Mapping):
            return
        updated_data = dict(data)
        updated_data["runtime_elapsed_wall_seconds"] = self._observed_wall_seconds()
        payload["data"] = updated_data
        _save_checkpoint(progress_path, payload)
        self.progress = payload

    def _enforce_live_runtime_projection(
        self,
        *,
        stage: str,
        completed_position: int,
        timing_rows: Sequence[Mapping[str, object]],
    ) -> None:
        """Project remaining work after eight windows and stop from a checkpoint."""

        runtime = self.experiment_protocol["runtime_budget"]
        projection_after = int(runtime["projection_after_windows"])
        if completed_position != projection_after:
            return
        if stage not in {"stage_b", "validation"}:
            raise TrainingError("live runtime projection received an invalid stage")

        def duration(row: Mapping[str, object]) -> float:
            measured = float(row.get("window_seconds_including_checkpoint") or 0.0)
            if measured:
                return measured
            component_name = (
                "arm_update_seconds" if stage == "stage_b" else "arm_forward_seconds"
            )
            components = row[component_name]
            if not isinstance(components, Mapping):
                raise TrainingError(f"{stage} runtime timing is malformed")
            return (
                float(row["preparation_seconds"])
                + sum(float(value) for value in components.values())
                + float(row.get("checkpoint_seconds") or 0.0)
            )

        per_window = self._timing_quantile(
            list(timing_rows),
            duration,
            include_first=bool(runtime["include_first_window_in_projection"]),
            quantile=float(runtime["window_quantile"]),
        )
        observed = self._observed_wall_seconds()
        if stage == "stage_b":
            remaining_windows = (
                len(self.train_frames) - completed_position
            ) + len(self.validation_frames)
        elif stage == "validation":
            remaining_windows = len(self.validation_frames) - completed_position
        projected = (
            observed
            + remaining_windows * per_window
            + float(runtime["fixed_projection_overhead_seconds"])
        )
        gated = float(runtime["projection_contingency_multiplier"]) * projected
        maximum = float(runtime["maximum_wall_seconds"])
        if gated > maximum:
            raise TrainingError(
                f"{stage} eight-window runtime projection exceeds {maximum:g} seconds"
            )

    def _micro_runtime_budget_gate(
        self,
        *,
        stage0_seconds: float,
        validation_summary_seconds: float,
    ) -> dict[str, object]:
        """Check the current 24/16 three-arm screen against its hard wall budget."""

        data = self._progress_data()
        runtime = self.experiment_protocol["runtime_budget"]
        include_first = bool(runtime["include_first_window_in_projection"])
        quantile = float(runtime["window_quantile"])
        current_rows = data.get("stage_a_timing")
        temporal_rows = data.get("stage_b_timing")
        validation_rows = data.get("validation_timing")
        current_window = self._timing_quantile(
            current_rows,
            lambda row: (
                float(row.get("window_seconds_including_checkpoint") or 0.0)
                or (
                    float(row["preparation_seconds"])
                    + float(row["update_seconds"])
                    + float(row.get("checkpoint_seconds") or 0.0)
                )
            ),
            include_first=include_first,
            quantile=quantile,
        )
        temporal_window = self._timing_quantile(
            temporal_rows,
            lambda row: (
                float(row.get("window_seconds_including_checkpoint") or 0.0)
                or (
                    float(row["preparation_seconds"])
                    + sum(
                        float(value)
                        for value in row["arm_update_seconds"].values()
                    )
                    + float(row.get("checkpoint_seconds") or 0.0)
                )
            ),
            include_first=include_first,
            quantile=quantile,
        )
        validation_window = self._timing_quantile(
            validation_rows,
            lambda row: (
                float(row.get("window_seconds_including_checkpoint") or 0.0)
                or (
                    float(row["preparation_seconds"])
                    + sum(
                        float(value)
                        for value in row["arm_forward_seconds"].values()
                    )
                    + float(row.get("checkpoint_seconds") or 0.0)
                )
            ),
            include_first=include_first,
            quantile=quantile,
        )
        components = {
            "stage0_audit": float(stage0_seconds),
            "stage_a_24_windows": len(self.train_frames) * current_window,
            "stage_b_three_arms_24_windows": len(self.train_frames) * temporal_window,
            "validation_three_arms_16_windows": (
                len(self.validation_frames) * validation_window
            ),
            "final_summary_and_save": max(10.0, float(validation_summary_seconds)),
        }
        projected = float(sum(components.values()))
        contingency = float(runtime["projection_contingency_multiplier"])
        gated = contingency * projected
        maximum = float(runtime["maximum_wall_seconds"])
        return {
            "applicable": True,
            "passed": gated <= maximum,
            "projection_scope": "current frozen 24/16 three-arm screen only",
            "quantile": quantile,
            "per_window_seconds": {
                "stage_a": current_window,
                "stage_b_all_three_arms": temporal_window,
                "validation_all_three_arms": validation_window,
            },
            "components_seconds": components,
            "projected_seconds_before_contingency": projected,
            "contingency_multiplier": contingency,
            "gated_projected_seconds": gated,
            "maximum_wall_seconds": maximum,
            "does_not_authorize_96_64": True,
        }

    def _micro_continue_gate(
        self, validation: Mapping[str, object]
    ) -> dict[str, object]:
        pooled = validation.get("pooled")
        rows = validation.get("windows")
        identities = validation.get("identity_controls")
        comparisons = validation.get("comparisons")
        safety = validation.get("normal_safety_strata")
        matching = validation.get("proposal_supervision_diagnostics")
        if (
            not isinstance(pooled, Mapping)
            or not isinstance(rows, list)
            or not isinstance(identities, Mapping)
            or not isinstance(comparisons, Mapping)
            or not isinstance(safety, Mapping)
            or not isinstance(matching, Mapping)
        ):
            raise TrainingError("micro validation lacks continue-gate inputs")
        all_metrics = pooled.get("all")
        paired = comparisons.get("proposal_direct_minus_classification")
        if not isinstance(all_metrics, Mapping) or not isinstance(paired, Mapping):
            raise TrainingError("micro validation lacks deconfounded comparisons")
        clean = all_metrics["Clean-Select-4"]
        direct = all_metrics["Proposal-Direct-4"]
        classification = all_metrics["Proposal-Classification-4"]
        for value in (clean, direct, classification):
            if not isinstance(value, Mapping):
                raise TrainingError("micro pooled metrics are malformed")
        bce_ci = paired["balanced_bce"]["cluster_bootstrap_95_ci"]
        ap_ci = paired["ap"]["cluster_bootstrap_95_ci"]
        auroc_ci = paired["auroc"]["cluster_bootstrap_95_ci"]
        checks = {
            "direct_better_bce": float(direct["balanced_bce"])
            < float(classification["balanced_bce"]),
            "direct_better_ap": float(direct["ap"])
            > float(classification["ap"]),
            "direct_better_auroc": float(direct["auroc"])
            > float(classification["auroc"]),
            "paired_bce_ci_excludes_zero": float(bce_ci[1]) < 0.0,
            "paired_ap_or_auroc_ci_excludes_zero": (
                float(ap_ci[0]) > 0.0 or float(auroc_ci[0]) > 0.0
            ),
        }
        for metric in ("balanced_bce", "ap", "auroc"):
            checks[f"direct_no_farther_from_clean_{metric}"] = abs(
                float(direct[metric]) - float(clean[metric])
            ) <= abs(float(classification[metric]) - float(clean[metric]))
        normal_points = 0
        saturated_points = 0.0
        for row in rows:
            if not isinstance(row, Mapping):
                raise TrainingError("micro validation row is malformed")
            conditions = row["conditions"]
            assert isinstance(conditions, Mapping)
            metrics = conditions["Proposal-Direct-4"]
            assert isinstance(metrics, Mapping)
            count = int(metrics["normal_points"])
            normal_points += count
            saturated_points += count * float(metrics["normal_saturation_fraction"])
        saturation = saturated_points / normal_points
        checks["normal_saturation_fraction_at_most_0_01"] = saturation <= 0.01
        null_by_arm = identities.get("null_by_arm")
        if not isinstance(null_by_arm, Mapping):
            raise TrainingError("micro per-arm Null identities are missing")
        for arm in SELECTOR_ARMS:
            identity = null_by_arm.get(arm)
            if not isinstance(identity, Mapping):
                raise TrainingError(f"micro Null identity is missing for {arm}")
            distribution = identity.get("current_max_abs")
            checks[f"{arm}_null_strict_current_identity"] = (
                isinstance(distribution, Mapping)
                and float(distribution["maximum"]) == 0.0
                and int(identity["non_null_p4_support_points"]) == 0
            )

        safety_summary = safety.get("summary")
        if not isinstance(safety_summary, Mapping):
            raise TrainingError("micro normal-safety summary is missing")
        moving_records = {
            condition: safety_summary[condition][
                "observed_moving_normal_instance"
            ]
            for condition in (
                "Current",
                "Proposal-Direct-4",
                "Proposal-Classification-4",
            )
        }
        direct_moving = moving_records["Proposal-Direct-4"]
        current_moving = moving_records["Current"]
        classification_moving = moving_records["Proposal-Classification-4"]
        moving_points = int(direct_moving["points"])
        moving_windows = int(direct_moving["windows_with_points"])
        checks["moving_normal_sentinel_has_at_least_100_points"] = (
            moving_points >= 100
        )
        checks["moving_normal_sentinel_has_at_least_2_windows"] = (
            moving_windows >= 2
        )
        if moving_points > 0:
            direct_weighted = direct_moving["point_weighted"]
            current_weighted = current_moving["point_weighted"]
            classification_weighted = classification_moving["point_weighted"]
            if not all(
                isinstance(value, Mapping)
                for value in (
                    direct_weighted,
                    current_weighted,
                    classification_weighted,
                )
            ):
                raise TrainingError("micro moving-normal metrics are malformed")
            checks["moving_normal_bce_not_worse_than_current"] = float(
                direct_weighted["normal_bce"]
            ) <= float(current_weighted["normal_bce"])
            checks["direct_moving_normal_bce_not_worse_than_classification"] = (
                float(direct_weighted["normal_bce"])
                <= float(classification_weighted["normal_bce"])
            )
            checks["direct_moving_normal_up_not_worse_than_classification"] = (
                float(direct_weighted["normal_up"])
                <= float(classification_weighted["normal_up"])
            )
        else:
            checks["moving_normal_bce_not_worse_than_current"] = False
            checks["direct_moving_normal_bce_not_worse_than_classification"] = False
            checks["direct_moving_normal_up_not_worse_than_classification"] = False

        differences = matching.get("proposal_direct_minus_classification")
        if not isinstance(differences, Mapping):
            raise TrainingError("micro B-C matching diagnostics are missing")
        scales = differences.get("scales")
        by_condition = matching.get("by_condition")
        if not isinstance(scales, Mapping) or not isinstance(by_condition, Mapping):
            raise TrainingError("micro matching scale diagnostics are missing")
        direct_matching = by_condition.get("Proposal-Direct-4")
        if not isinstance(direct_matching, Mapping):
            raise TrainingError("micro direct matching diagnostics are missing")
        direct_scales = direct_matching.get("scales")
        if not isinstance(direct_scales, Mapping):
            raise TrainingError("micro direct matching scales are missing")
        for scale in ("p16", "p8", "p4"):
            direct_scale = direct_scales[scale]
            difference = scales[scale]
            checks[f"{scale}_has_learnable_null_targets"] = (
                int(direct_scale["learnable_null_queries"]) > 0
            )
            same_difference = difference["same_object_attention_mass"]
            null_difference = difference["null_attention_mass_on_null_targets"]
            checks[f"{scale}_direct_improves_same_object_mass"] = (
                isinstance(same_difference, Mapping)
                and float(same_difference["mean"]) > 0.0
            )
            checks[f"{scale}_direct_improves_null_mass"] = (
                isinstance(null_difference, Mapping)
                and float(null_difference["mean"]) > 0.0
            )
        return {
            "applicable": True,
            "passed": all(checks.values()),
            "checks": checks,
            "normal_saturation_fraction": saturation,
            "moving_normal_sentinel": {
                "points": moving_points,
                "windows": moving_windows,
            },
            "decision_scope": (
                "permission for an independent 96/64 selector confirmation only; "
                "not permission for a learned matcher or formal training"
            ),
            "automatic_learned_matcher_decision": False,
        }

    def run(self) -> dict[str, object]:
        """Run the frozen 24/16 selector deconfounding screen."""

        progress_data = self._progress_data()
        prior_wall = float(progress_data.get("runtime_elapsed_wall_seconds", 0.0))
        if not math.isfinite(prior_wall) or prior_wall < 0.0:
            raise TrainingError("recovery has an invalid cumulative wall time")
        self._run_wall_prior_seconds = prior_wall
        self._enforce_hard_runtime_limit()
        stored_temporal_initial = progress_data.get("temporal_initial")
        if stored_temporal_initial is None:
            temporal_initial = self._snapshot(self.model, self._temporal_prefixes)
        elif isinstance(stored_temporal_initial, Mapping) and all(
            isinstance(value, Tensor) for value in stored_temporal_initial.values()
        ):
            temporal_initial = {
                str(name): value.detach().cpu().clone()
                for name, value in stored_temporal_initial.items()
            }
        else:
            raise TrainingError("recovery lacks a valid initial temporal state")
        if set(temporal_initial) != set(
            self._snapshot(self.model, self._temporal_prefixes)
        ):
            raise TrainingError("initial temporal recovery state is incomplete")
        self.temporal_initial_state = temporal_initial

        current_training = self._train_current()
        current_state = self._snapshot(self.model, ("point_anomaly_head.",))
        current_stage_a_sha256 = _tensor_state_sha256(current_state)
        progress_data = self._progress_data()
        stored_stage_a_sha256 = progress_data.get("stage_a_current_sha256")
        if (
            stored_stage_a_sha256 is not None
            and stored_stage_a_sha256 != current_stage_a_sha256
        ):
            raise TrainingError("recovered stage-A current head identity changed")
        current_stage_b_start_sha256 = _tensor_state_sha256(
            self._snapshot(self.model, ("point_anomaly_head.",))
        )
        if current_stage_b_start_sha256 != current_stage_a_sha256:
            raise TrainingError("stage-B current head differs from stage-A final state")

        progress_stage = self._progress_stage()
        progress_data = self._progress_data()
        stored_audit = progress_data.get("gradient_audit")
        completed_stages = {
            "stage0_complete",
            "stage_b",
            "stage_b_complete",
            "validation",
            "validation_complete",
            "complete",
        }
        if progress_stage in completed_stages:
            if (
                not isinstance(stored_audit, Mapping)
                or stored_audit.get("format") != STAGE0_AUDIT_FORMAT
                or stored_audit.get("passed") is not True
            ):
                raise TrainingError("stage-0 recovery report is missing")
            gradient_audit = dict(stored_audit)
            stage0_seconds = float(progress_data.get("stage0_seconds", 0.0))
        elif progress_stage == "stage_a_complete":
            audit_started = time.perf_counter()
            gradient_audit = self._loss_gradient_audit()
            stage0_seconds = time.perf_counter() - audit_started
            self._enforce_hard_runtime_limit()
            self._save_progress(
                "stage0_complete",
                {
                    "current_training": current_training,
                    "stage_a_current_sha256": current_stage_a_sha256,
                    "stage_b_start_current_sha256": current_stage_b_start_sha256,
                    "gradient_audit": gradient_audit,
                    "stage0_seconds": stage0_seconds,
                    "temporal_initial": temporal_initial,
                },
            )
            self._enforce_hard_runtime_limit()
        else:
            raise TrainingError(
                f"unexpected recovery stage before audit: {progress_stage}"
            )

        self._enforce_hard_runtime_limit()
        training_history = self._train_temporal()
        current_state_after_temporal = self._snapshot(
            self.model, ("point_anomaly_head.",)
        )
        if set(current_state_after_temporal) != set(current_state) or any(
            not torch.equal(current_state[name], current_state_after_temporal[name])
            for name in current_state
        ):
            raise TrainingError("temporal training changed the current-frame baseline")
        current_stage_b_end_sha256 = _tensor_state_sha256(current_state_after_temporal)
        if current_stage_b_end_sha256 != current_stage_b_start_sha256:
            raise TrainingError("stage-B current head identity changed")
        current_head_identity = {
            "algorithm": "sha256(name,dtype,shape,exact_tensor_bytes)",
            "stage_a_final": current_stage_a_sha256,
            "stage_b_start": current_stage_b_start_sha256,
            "stage_b_end": current_stage_b_end_sha256,
            "all_identical": True,
        }

        validation_started = time.perf_counter()
        validation = self._validate()
        observed_validation_wall = time.perf_counter() - validation_started
        progress_data = self._progress_data()
        validation_wall_seconds = float(
            progress_data.get("validation_wall_seconds", observed_validation_wall)
        )
        validation_summary_seconds = float(
            progress_data.get(
                "validation_summary_seconds",
                validation.get("summary_seconds", validation_wall_seconds),
            )
        )
        self._save_progress(
            "validation_complete",
            {
                "validation_summary": validation,
                "validation_wall_seconds": validation_wall_seconds,
                "validation_summary_seconds": validation_summary_seconds,
                "stage_b_end_current_sha256": current_stage_b_end_sha256,
            },
        )
        continue_gate = self._micro_continue_gate(validation)
        runtime_gate = self._micro_runtime_budget_gate(
            stage0_seconds=stage0_seconds,
            validation_summary_seconds=validation_summary_seconds,
        )

        temporal_initial_sha256 = _tensor_state_sha256(temporal_initial)
        temporal_arm_sha256 = {
            arm: _tensor_state_sha256(state)
            for arm, state in self.temporal_arm_states.items()
        }
        states_path = self.output / "states.pt"
        _save_checkpoint(
            states_path,
            {
                "format": ORACLE_TEMPORAL_STATE_FORMAT,
                "current": current_state,
                "temporal_initial": temporal_initial,
                "temporal_initial_sha256": temporal_initial_sha256,
                "temporal_arms": self.temporal_arm_states,
                "temporal_arm_sha256": temporal_arm_sha256,
                "current_head_identity": current_head_identity,
            },
        )
        observed_duration = self._observed_wall_seconds()
        maximum_wall = float(
            self.experiment_protocol["runtime_budget"]["maximum_wall_seconds"]
        )
        final_write_reserve = float(
            self.experiment_protocol["runtime_budget"][
                "final_result_write_reserve_seconds"
            ]
        )
        runtime_gate["observed_wall_seconds_before_result_write"] = (
            observed_duration
        )
        runtime_gate["final_result_write_reserve_seconds"] = final_write_reserve
        runtime_gate["observed_wall_within_limit"] = (
            observed_duration + final_write_reserve <= maximum_wall
        )
        runtime_gate["passed"] = bool(runtime_gate["passed"]) and bool(
            runtime_gate["observed_wall_within_limit"]
        )
        result = {
            "format": ORACLE_TEMPORAL_FORMAT,
            "material_passport": {
                "origin_skill": "academic-research-suite/experiment-agent",
                "origin_mode": "run",
                "verification_status": "UNVERIFIED",
                "version_label": "selector_deconfounding_v1",
            },
            "status": (
                "completed"
                if continue_gate["passed"] is True
                and runtime_gate["passed"] is True
                else "completed_micro_stop"
            ),
            "scope": {
                "mode": "24/16 selector deconfounding screen",
                "train_sequence": 206,
                "validation_sequence": 201,
                "train_windows": len(self.train_frames),
                "validation_windows": len(self.validation_frames),
                "current_passes": self.current_passes,
                "temporal_passes_per_arm": self.temporal_passes,
                "temporal_history_lengths": list(self.temporal_lengths),
                "formal_training": False,
                "learned_matcher": False,
                "truth_motion_candidates_supplied": True,
            },
            "source": {
                "mechanism_manifest": _project_relative_path(self.manifest_path),
                "mechanism_manifest_sha256": self.source["manifest_sha256"],
                "experiment_protocol": _project_relative_path(
                    self.experiment_protocol_path
                ),
                "experiment_protocol_sha256": _sha256_file(
                    self.experiment_protocol_path
                ),
                "source_hashes": self.source["source_hashes"],
                "raw_sequence_identity": self.source["raw_sequence_identity"],
                "code_identity": self.code_identity,
            },
            "design": {
                "one_physical_trajectory_per_window": True,
                "one_shared_frozen_current_head": True,
                "identical_temporal_initialization": True,
                "independent_temporal_states_and_optimizers": True,
                "identical_window_order_and_history_exposure": True,
                "arms": {
                    "clean_select": (
                        "one truth-constructed real candidate plus null per age; "
                        "classification objective only"
                    ),
                    "proposal_direct": (
                        "static, truth-motion, and null candidates; classification "
                        "plus direct same-object/null probability-mass supervision"
                    ),
                    "proposal_classification": (
                        "the same Proposal candidates with classification only"
                    ),
                },
                "proposal_arms_receive_clean_select_gradients": False,
                "truth_match_metadata_enters_classifier": False,
                "candidate_parameters_shared_across_real_slots": True,
                "candidate_type_embedding": False,
                "candidate_permutation_preflight_required": True,
                "explicit_null_value": 0.0,
                "explicit_null_counts_as_history_support": False,
                "current_baseline_unchanged_during_temporal_training": True,
                "maximum_temporal_logit_correction": 4.0,
                "classification_objective": (
                    "balanced_BCE + normal_safety + "
                    "0.1*class_balanced_SmoothL1(beta=1.0)"
                ),
                "direct_objective": (
                    "equal p16/p8/p4 mean of age-local same-object mass loss "
                    "plus learnable null competition loss"
                ),
                "free_gate": False,
                "gain_loss": False,
                "rich_neutral_ordering": False,
                "persistent_derived_cache": False,
                "window_local_fp32_materialization": True,
                "validation_based_state_selection": False,
            },
            "training": {
                "current": current_training,
                "temporal_three_arms": training_history,
            },
            "stage0": gradient_audit,
            "validation": validation,
            "states": {
                "path": _project_relative_path(states_path),
                "bytes": states_path.stat().st_size,
                "sha256": _sha256_file(states_path),
                "temporal_initial_sha256": temporal_initial_sha256,
                "temporal_arm_sha256": temporal_arm_sha256,
            },
            "current_head_identity": current_head_identity,
            "micro_continue_gate": continue_gate,
            "runtime_budget_gate": runtime_gate,
            "runtime_profile": {
                "stage_a_windows": self._progress_data().get(
                    "stage_a_timing", []
                ),
                "stage_b_windows": self._progress_data().get(
                    "stage_b_timing", []
                ),
                "validation_windows": self._progress_data().get(
                    "validation_timing", []
                ),
                "stage0_seconds": stage0_seconds,
                "validation_wall_seconds": validation_wall_seconds,
                "cuda_synchronized": True,
            },
            "automatic_learned_matcher_decision": False,
            "duration_seconds": observed_duration,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(self.device)),
        }
        # The first atomic write makes a complete result recoverable. The second
        # records its measured cost; a frozen reserve covers the second small write.
        _write_json(self.output / "result.json", result)
        observed_after_result_write = self._observed_wall_seconds()
        runtime_gate["observed_wall_seconds_after_initial_result_write"] = (
            observed_after_result_write
        )
        runtime_gate["observed_wall_within_limit"] = (
            observed_after_result_write + final_write_reserve <= maximum_wall
        )
        runtime_gate["passed"] = bool(runtime_gate["passed"]) and bool(
            runtime_gate["observed_wall_within_limit"]
        )
        result["status"] = (
            "completed"
            if continue_gate["passed"] is True and runtime_gate["passed"] is True
            else "completed_micro_stop"
        )
        result["duration_seconds"] = observed_after_result_write
        _write_json(self.output / "result.json", result)
        self._enforce_hard_runtime_limit()
        progress_path = self.output / "progress.pt"
        if progress_path.is_file():
            progress_path.unlink()
        return result

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 24/16 three-arm selector deconfounding screen "
            "on normal sequences 206 and 201."
        )
    )
    parser.add_argument("--split", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=DEFAULT_GPU_MEMORY_FRACTION,
        help="hard CUDA allocator fraction; values above 0.70 are rejected",
    )
    parser.add_argument(
        "--official-repository", type=Path, default=DEFAULT_STU_REPOSITORY
    )
    parser.add_argument(
        "--oracle-temporal-manifest",
        type=Path,
        default=PROJECT_ROOT / "results" / "oracle_source.json",
        help="frozen truth-trajectory source pool for the 24/16 selector screen",
    )
    parser.add_argument(
        "--oracle-temporal-protocol",
        type=Path,
        default=DEFAULT_ORACLE_TEMPORAL_PROTOCOL,
        help="frozen scientific definition for --oracle-temporal-manifest",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted identical Oracle run from progress.pt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    protocol = (
        load_protocol(arguments.split)
        if arguments.split is not None
        else load_protocol()
    )
    config = TrainConfig(
        new_lr=1.0e-4,
        weight_decay=1.0e-4,
        seed=20260813,
        minimum_current_anomaly_points=1,
        gpu_memory_fraction=arguments.gpu_memory_fraction,
    )
    experiment = OracleTemporalExperiment(
        protocol=protocol,
        data_root=arguments.data_root,
        output=arguments.output,
        config=config,
        mechanism_manifest=arguments.oracle_temporal_manifest,
        experiment_protocol=arguments.oracle_temporal_protocol,
        official_repository=arguments.official_repository,
        resume=arguments.resume,
    )
    try:
        result = experiment.run()
    except BaseException:
        try:
            experiment._checkpoint_failure_runtime()
        except Exception as checkpoint_error:
            print(
                f"warning: failed to persist exception wall time: {checkpoint_error}",
                file=sys.stderr,
                flush=True,
            )
        raise
    print(
        json.dumps(
            {
                "format": ORACLE_TEMPORAL_FORMAT,
                "status": result["status"],
                "output": _project_relative_path(
                    experiment.output / "result.json"
                ),
                "duration_seconds": result["duration_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
