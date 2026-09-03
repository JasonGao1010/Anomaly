#!/usr/bin/env python3
"""Schema-31 training over frozen five-scan WindowWorld banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .evaluate import (
        AJAEInference,
        DevelopmentClipResult,
        DevelopmentFusedAP,
        EvaluationIdentity,
        FUSION_SEMANTICS as DEVELOPMENT_FUSION_SEMANTICS,
        development_fused_ap,
    )
    from .model import (
        FrozenSTUPointEncoder,
        JointWindowPointTransformer,
        STUPointEncoding,
        stu_input_identity,
    )
    from .protocol import (
        AJAEProtocol,
        DevelopmentWorlds,
        ExperimentCondition,
        GroupingMode,
        load_development_worlds,
        load_protocol,
    )
    from .render import (
        DevelopmentClipWorld,
        RenderError,
        WindowEntityDescriptor,
        WorldGenerationReport,
        WorldSpec,
        source_observation_identity,
    )
    from .scene import assemble_window
except ImportError:  # Direct execution from src/.
    from evaluate import (
        AJAEInference,
        DevelopmentClipResult,
        DevelopmentFusedAP,
        EvaluationIdentity,
        FUSION_SEMANTICS as DEVELOPMENT_FUSION_SEMANTICS,
        development_fused_ap,
    )
    from model import (
        FrozenSTUPointEncoder,
        JointWindowPointTransformer,
        STUPointEncoding,
        stu_input_identity,
    )
    from protocol import (
        AJAEProtocol,
        DevelopmentWorlds,
        ExperimentCondition,
        GroupingMode,
        load_development_worlds,
        load_protocol,
    )
    from render import (
        DevelopmentClipWorld,
        RenderError,
        WindowEntityDescriptor,
        WorldGenerationReport,
        WorldSpec,
        source_observation_identity,
    )
    from scene import assemble_window


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 31
WINDOW_GROUPS = (0, 1, 2, 3, 4)
WINDOW_BANK_FORMAT = "ajae-window-train-bank-v1"
TRAIN_CHECKPOINT_FORMAT = "ajae-schema31-training-checkpoint-v1"
TRAIN_RESULT_FORMAT = "ajae-schema31-training-result-v1"
MANIFEST_NAME = "manifest.json"
_SHARD_ARRAYS = {
    "coordinates", "scan_group", "stu_features", "normal_evidence",
    "reliability_assign", "reliability_noobj", "intensity", "target",
    "valid", "source_frame", "source_slot", "source_ray",
}


class TrainingError(RuntimeError):
    """Report an invalid schema-31 bank, request, or optimization state."""


class TrainMode(str, Enum):
    """Mutually exclusive schema-31 training purposes."""

    TINY_OVERFIT = "tiny_overfit"
    PILOT = "pilot"
    FORMAL = "formal"


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _normalized_window_manifest(value: object) -> dict[str, object]:
    """Normalize the renderer's sole valid positive-infinity clearance sentinel."""

    plain = _plain_json(value)
    if not isinstance(plain, dict):
        raise TrainingError("window_manifest must be a JSON object")
    report = plain.get("report")
    if isinstance(report, dict) and isinstance(report.get("placements"), list):
        for placement in report["placements"]:
            if not isinstance(placement, dict):
                raise TrainingError("window report placement is malformed")
            proposals = placement.get("proposal_minimum_obstacle_sdf_m")
            if isinstance(proposals, list):
                for index, clearance in enumerate(proposals):
                    if (
                        isinstance(clearance, float)
                        and math.isinf(clearance)
                        and clearance > 0.0
                    ):
                        proposals[index] = None
            clearance = placement.get("minimum_obstacle_sdf_m")
            if (
                isinstance(clearance, float)
                and math.isinf(clearance)
                and clearance > 0.0
            ):
                placement["minimum_obstacle_sdf_m"] = None
    try:
        normalized = json.loads(_canonical_json(plain))
    except json.JSONDecodeError as error:  # Defensive; canonical JSON is internal.
        raise TrainingError("window_manifest cannot be normalized") from error
    if not isinstance(normalized, dict):
        raise TrainingError("window_manifest must remain a JSON object")
    return normalized


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _plain_json(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TrainingError("identity payload is not finite JSON") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    try:
        status = path.stat()
    except OSError as error:
        raise TrainingError(f"cannot stat frozen file: {path.name}") from error
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _identity_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TrainingError(f"{name} must be an integer >= {minimum}")
    return value


def _require_number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise TrainingError(f"{name} must be finite and >= {minimum}")
    return result


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TrainingError(f"{name} must be a JSON object with string keys")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], name: str
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise TrainingError(f"{name} keys differ; missing={missing}, extra={extra}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(
            _plain_json(payload), ensure_ascii=False, indent=2,
            sort_keys=True, allow_nan=False,
        ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_torch(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _seed_everything(seed: int) -> None:
    _require_int(seed, "seed")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _numpy_rng_record() -> dict[str, object]:
    generator, state, position, has_gauss, cached = np.random.get_state()
    return {
        "generator": generator,
        "state": state.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _restore_rng(payload: Mapping[str, object], device: torch.device) -> None:
    numpy_record = _require_mapping(
        payload.get("numpy_rng_state"), "checkpoint numpy_rng_state"
    )
    try:
        random.setstate(payload["python_rng_state"])  # type: ignore[arg-type]
        np.random.set_state((
            str(numpy_record["generator"]),
            np.asarray(numpy_record["state"], dtype=np.uint32),
            int(numpy_record["position"]),
            int(numpy_record["has_gauss"]),
            float(numpy_record["cached_gaussian"]),
        ))
        torch.set_rng_state(payload["torch_rng_state"].cpu())  # type: ignore[union-attr]
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise TrainingError("checkpoint RNG state is invalid") from error
    cuda_state = payload.get("cuda_rng_state")
    if device.type == "cuda":
        if (
            not isinstance(cuda_state, list)
            or len(cuda_state) != torch.cuda.device_count()
        ):
            raise TrainingError("checkpoint CUDA RNG state cannot be restored")
        torch.cuda.set_rng_state_all(cuda_state)
    elif cuda_state is not None:
        raise TrainingError("CPU checkpoint unexpectedly contains CUDA RNG state")


def _finite_tensor(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise TrainingError(f"{name} contains non-finite values")


def _cpu_float_tensor(value: object, name: str, shape: tuple[int, ...]) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise TrainingError(f"{name} must have shape {shape}")
    if not value.is_floating_point():
        raise TrainingError(f"{name} must be floating point")
    result = (
        value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    )
    _finite_tensor(result, name)
    return result


def _cpu_integer_tensor(value: object, name: str, count: int) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != (count,):
        raise TrainingError(f"{name} must have shape ({count},)")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise TrainingError(f"{name} must use an integer dtype")
    return value.detach().to(device="cpu", dtype=torch.long).contiguous().clone()


def _cpu_bool_tensor(value: object, name: str, count: int) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != (count,):
        raise TrainingError(f"{name} must have shape ({count},)")
    result = value.detach().to(device="cpu")
    if result.dtype != torch.bool:
        if result.is_floating_point() or result.is_complex() or not bool(
            ((result == 0) | (result == 1)).all()
        ):
            raise TrainingError(f"{name} must be boolean or an integer 0/1 tensor")
        result = result.bool()
    return result.contiguous().clone()


@dataclass(frozen=True, slots=True)
class WindowTrainingData:
    """All supervised returns from one symmetric five-scan WindowWorld."""

    coordinates: Tensor
    scan_group: Tensor
    stu_features: Tensor
    normal_evidence: Tensor
    reliability_assign: Tensor
    reliability_noobj: Tensor
    intensity: Tensor
    target: Tensor
    valid: Tensor
    source_frame: Tensor
    source_slot: Tensor
    source_ray: Tensor
    world_identity: str
    source_observation_identities: tuple[str, str, str, str, str]
    stu_input_identities: tuple[str, str, str, str, str]
    window_manifest: Mapping[str, object]
    window_identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, Tensor) or self.coordinates.ndim != 2:
            raise TrainingError("coordinates must be [N,3]")
        count = int(self.coordinates.shape[0])
        if count == 0 or self.coordinates.shape[1] != 3:
            raise TrainingError("coordinates must be a non-empty [N,3] tensor")
        floats = {
            "coordinates": (self.coordinates, (count, 3)),
            "stu_features": (self.stu_features, (count, 128)),
            "normal_evidence": (self.normal_evidence, (count, 19)),
            "reliability_assign": (self.reliability_assign, (count,)),
            "reliability_noobj": (self.reliability_noobj, (count,)),
            "intensity": (self.intensity, (count,)),
        }
        for name, (value, shape) in floats.items():
            object.__setattr__(self, name, _cpu_float_tensor(value, name, shape))
        for name in ("scan_group", "source_frame", "source_slot", "source_ray"):
            object.__setattr__(
                self, name, _cpu_integer_tensor(getattr(self, name), name, count)
            )
        object.__setattr__(self, "target", _cpu_bool_tensor(self.target, "target", count))
        object.__setattr__(self, "valid", _cpu_bool_tensor(self.valid, "valid", count))
        object.__setattr__(
            self, "world_identity", _require_digest(self.world_identity, "world_identity")
        )
        for name in ("source_observation_identities", "stu_input_identities"):
            values = tuple(getattr(self, name))
            if len(values) != 5:
                raise TrainingError(f"{name} must contain five ordered identities")
            object.__setattr__(
                self,
                name,
                tuple(
                    _require_digest(value, f"{name}[{index}]")
                    for index, value in enumerate(values)
                ),
            )
        if len(set(self.source_observation_identities)) != 5:
            raise TrainingError("five source observation identities must be unique")
        observed = tuple(np.unique(self.scan_group.numpy()).tolist())
        if observed != WINDOW_GROUPS:
            raise TrainingError("scan_group must contain every group 0..4")
        if bool((self.source_frame < 0).any()):
            raise TrainingError("source_frame cannot be negative")
        if bool((self.source_slot < 0).any()) or bool((self.source_ray < 0).any()):
            raise TrainingError("source_slot and source_ray cannot be negative")
        if bool((self.source_slot >= 128 * 1024).any()) or bool(
            (self.source_ray >= 128 * 1024).any()
        ):
            raise TrainingError(
                "source_slot and source_ray must lie in the OS1-128 domain"
            )
        frames: list[int] = []
        for group in WINDOW_GROUPS:
            member_frames = torch.unique(self.source_frame[self.scan_group == group])
            if member_frames.numel() != 1:
                raise TrainingError("each scan_group must identify exactly one source frame")
            frames.append(int(member_frames.item()))
        if tuple(frames) != tuple(range(frames[0], frames[0] + 5)):
            raise TrainingError("scan groups must map to five consecutive source frames")
        frame = self.source_frame.numpy()
        slot = self.source_slot.numpy()
        ray = self.source_ray.numpy()
        if np.unique(np.column_stack((frame, slot)), axis=0).shape[0] != count:
            raise TrainingError("(source_frame, source_slot) identities must be unique")
        if np.unique(np.column_stack((frame, ray)), axis=0).shape[0] != count:
            raise TrainingError("(source_frame, source_ray) identities must be unique")
        if not bool(self.valid.any()):
            raise TrainingError("a WindowWorld must contain at least one valid point")
        manifest = _normalized_window_manifest(self.window_manifest)
        identity = _validate_window_manifest(self, manifest)
        if self.window_identity and self.window_identity != identity:
            raise TrainingError(
                "WindowTrainingData.window_identity disagrees with its manifest"
            )
        object.__setattr__(self, "window_identity", identity)
        object.__setattr__(self, "window_manifest", _freeze_json(manifest))

    @property
    def point_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def five_source_frames(self) -> tuple[int, int, int, int, int]:
        frames = tuple(
            int(self.source_frame[self.scan_group == group][0].item())
            for group in WINDOW_GROUPS
        )
        return frames  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class WindowBankEntry:
    position: int
    shard: Path
    shard_sha256: str
    point_count: int
    world_identity: str
    window_identity: str
    source_frames: tuple[int, int, int, int, int]
    source_observation_identities: tuple[str, str, str, str, str]
    stu_input_identities: tuple[str, str, str, str, str]
    target_count: int
    valid_count: int
    anomaly_count: int
    normal_count: int
    ignored_count: int
    point_identity_sha256: str
    labels_sha256: str
    window_manifest: Mapping[str, object]


def protocol_bank_identity(protocol: object) -> str:
    """Hash only protocol rules that can change a frozen training-bank row."""

    if type(protocol) is not AJAEProtocol or protocol.schema_version != SCHEMA_VERSION:
        raise TrainingError("window training banks require schema 31")
    document = protocol.plain_document()
    data = _require_mapping(document["data"], "protocol.data")
    training = _require_mapping(document["training"], "protocol.training")
    evaluation = _require_mapping(document["evaluation"], "protocol.evaluation")
    # Model recipes and advancing run status do not alter this reusable artifact.
    payload = {
        "format": "ajae-schema31-window-train-bank-protocol-v1",
        "schema_version": SCHEMA_VERSION,
        "scientific_contract": document["scientific_contract"],
        "normal_training_data": data["normal_training"],
        "window": document["window"],
        "labels": document["labels"],
        "render": document["render"],
        "stu": document["stu"],
        "bank_source": {
            "source_partition": training["source_partition"],
            "source_sequence_id": training["source_sequence_id"],
            "bank": training["bank"],
        },
        # These fields define the valid mask stored in every shard.
        "validity_domain": {
            "domain": evaluation["domain"],
            "minimum_range_m_inclusive": evaluation[
                "minimum_range_m_inclusive"
            ],
            "maximum_range_m_inclusive": evaluation[
                "maximum_range_m_inclusive"
            ],
        },
    }
    return _identity_digest(payload)


def _load_training_protocol(path: Path | str) -> object:
    """Reject retired schemas before the full protocol loader is invoked."""

    resolved = Path(path).expanduser()
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingError("training protocol is unreadable") from error
    if not isinstance(document, Mapping) or document.get("schema_version") != 31:
        raise TrainingError("training requires schema 31; older schemas are retired")
    return load_protocol(resolved)


def _window_identity(
    *,
    world_identity: str,
    source_frames: Sequence[int],
    source_observation_identities: Sequence[str],
    renderer_identity: str,
    partition: str,
    sequence_id: int,
) -> str:
    frames = tuple(int(value) for value in source_frames)
    observations = tuple(source_observation_identities)
    if len(observations) != 5:
        raise TrainingError("window identity requires five source observations")
    return _identity_digest({
        "format": "ajae-window-world-v1",
        "world_identity": world_identity,
        "partition": partition,
        "sequence_id": sequence_id,
        "window_start": frames[0],
        "frame_ids": frames,
        "renderer_identity": renderer_identity,
        "source_observation_identities": observations,
    })


def _validate_window_manifest_record(
    manifest: Mapping[str, object],
    *,
    expected_world_identity: str,
    expected_source_frames: Sequence[int],
    expected_source_observation_identities: Sequence[str],
    expected_target_count: int,
) -> str:
    """Rebuild renderer objects and identities from one retained scientific record."""

    _require_exact_keys(manifest, {
        "format", "identity", "world_identity", "partition", "sequence_id",
        "window_start", "frame_ids", "renderer_identity", "world", "report",
        "source_observation_identities", "descriptors",
    }, "window_manifest")
    if manifest["format"] != "ajae-window-world-v1":
        raise TrainingError("window_manifest format is unsupported")
    world_identity = _require_digest(
        manifest["world_identity"], "window_manifest.world_identity"
    )
    if world_identity != expected_world_identity:
        raise TrainingError("window_manifest identifies a different WorldSpec")
    partition = manifest["partition"]
    if not isinstance(partition, str) or partition not in {"train", "val", "test"}:
        raise TrainingError("window_manifest partition is invalid")
    sequence_id = _require_int(
        manifest["sequence_id"], "window_manifest.sequence_id"
    )
    start = _require_int(
        manifest["window_start"], "window_manifest.window_start"
    )
    raw_frames = manifest["frame_ids"]
    if not isinstance(raw_frames, list) or len(raw_frames) != 5:
        raise TrainingError("window_manifest must contain five frame IDs")
    frames = tuple(
        _require_int(value, "window_manifest frame ID") for value in raw_frames
    )
    if frames != tuple(expected_source_frames):
        raise TrainingError("window_manifest frame IDs disagree with point rows")
    if start != frames[0]:
        raise TrainingError("window_manifest start disagrees with its frames")
    renderer = _require_digest(
        manifest["renderer_identity"], "window_manifest.renderer_identity"
    )
    raw_observations = manifest["source_observation_identities"]
    if not isinstance(raw_observations, list) or len(raw_observations) != 5:
        raise TrainingError("window_manifest must bind five source observations")
    observations = tuple(
        _require_digest(value, "source observation identity")
        for value in raw_observations
    )
    if observations != tuple(expected_source_observation_identities):
        raise TrainingError("window_manifest source observations changed")

    world_record = _require_mapping(
        manifest["world"], "window_manifest.world"
    )
    report_record = _require_mapping(
        manifest["report"], "window_manifest.report"
    )
    try:
        world = WorldSpec.from_dict(world_record)
        report_plain = json.loads(_canonical_json(report_record))
        for placement in report_plain.get("placements", []):
            proposals = placement.get("proposal_minimum_obstacle_sdf_m")
            if isinstance(proposals, list):
                placement["proposal_minimum_obstacle_sdf_m"] = [
                    math.inf if value is None else value for value in proposals
                ]
            if placement.get("minimum_obstacle_sdf_m") is None:
                placement["minimum_obstacle_sdf_m"] = math.inf
        report = WorldGenerationReport.from_dict(report_plain)
    except (TypeError, ValueError, KeyError, RenderError) as error:
        raise TrainingError(
            "window_manifest does not contain a complete WorldSpec/report"
        ) from error
    if world.identity != world_identity:
        raise TrainingError("WorldSpec content does not match world_identity")
    if (
        world.source_sequence_id != sequence_id
        or report.source_sequence_id != sequence_id
        or report.world_seed != world.seed
        or report.world_type != world.world_type
        or report.normal_count != world.normal_control_count
        or report.anomaly_count != world.anomaly_proxy_count
    ):
        raise TrainingError("WorldSpec and generation report identities differ")
    placements = {item.object_id: item.label for item in report.placements}
    objects = {item.object_id: item.label for item in world.objects}
    if len(report.placements) != len(placements) or placements != objects:
        raise TrainingError("generation report does not cover every WorldSpec object")

    raw_descriptors = manifest["descriptors"]
    if not isinstance(raw_descriptors, list):
        raise TrainingError("window_manifest.descriptors must be an array")
    descriptor_keys = {
        "object_id", "label", "visible_returns_by_scan",
        "spatial_voxels_by_scan", "joint_visible_return_count",
        "joint_spatial_voxel_count", "maximum_single_scan_spatial_voxel_count",
        "densification_gain", "duplicate_fraction", "median_distance_m",
        "occlusion_rate", "support_semantic_id", "visible_scan_count",
        "minimum_visible_return_height_m", "intensity_q05_median_q95",
        "beam_histogram",
    }
    descriptor_ids: list[int] = []
    anomaly_returns = 0
    for descriptor_index, raw_descriptor in enumerate(raw_descriptors):
        descriptor = _require_mapping(
            raw_descriptor,
            f"window_manifest.descriptors[{descriptor_index}]",
        )
        _require_exact_keys(
            descriptor,
            descriptor_keys,
            f"window_manifest.descriptors[{descriptor_index}]",
        )
        try:
            descriptor_value = WindowEntityDescriptor(**json.loads(
                _canonical_json(descriptor)
            ))
        except (TypeError, ValueError, KeyError, RenderError) as error:
            raise TrainingError("window density descriptor is invalid") from error
        object_id = descriptor_value.object_id
        label = descriptor_value.label
        if objects.get(object_id) != label:
            raise TrainingError("descriptor does not identify its WorldSpec object")
        returns_count = descriptor_value.joint_visible_return_count
        descriptor_ids.append(object_id)
        if label == "anomaly-proxy":
            anomaly_returns += returns_count
    if tuple(descriptor_ids) != tuple(sorted(objects)):
        raise TrainingError("descriptors must cover every WorldSpec object once")
    if expected_target_count != anomaly_returns:
        raise TrainingError("anomaly label count disagrees with proxy descriptors")

    expected = _window_identity(
        world_identity=world_identity,
        source_frames=frames,
        source_observation_identities=observations,
        renderer_identity=renderer,
        partition=partition,
        sequence_id=sequence_id,
    )
    declared = _require_digest(
        manifest["identity"], "window_manifest.identity"
    )
    if declared != expected:
        raise TrainingError("window_manifest identity does not match its inputs")
    return declared


def _validate_window_manifest(
    data: WindowTrainingData, manifest: Mapping[str, object]
) -> str:
    return _validate_window_manifest_record(
        manifest,
        expected_world_identity=data.world_identity,
        expected_source_frames=data.five_source_frames,
        expected_source_observation_identities=data.source_observation_identities,
        expected_target_count=int(data.target.sum().item()),
    )


def _tensor_identity_digest(
    fields: Sequence[tuple[str, Tensor]], *, boolean: bool = False
) -> str:
    digest = hashlib.sha256()
    for name, value in fields:
        array = value.detach().cpu().contiguous().numpy()
        if boolean:
            array = array.astype(np.uint8, copy=False)
        else:
            array = array.astype("<i8", copy=False)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _point_identity_digest(data: WindowTrainingData) -> str:
    return _tensor_identity_digest((
        ("scan_group", data.scan_group),
        ("source_frame", data.source_frame),
        ("source_slot", data.source_slot),
        ("source_ray", data.source_ray),
    ))


def _label_identity_digest(data: WindowTrainingData) -> str:
    return _tensor_identity_digest(
        (("target", data.target), ("valid", data.valid)), boolean=True
    )


def _label_counts(data: WindowTrainingData) -> tuple[int, int, int, int, int]:
    valid = data.valid
    return (
        int(data.target.sum().item()),
        int(valid.sum().item()),
        int((valid & data.target).sum().item()),
        int((valid & ~data.target).sum().item()),
        int((~valid).sum().item()),
    )


def window_training_data(
    window_world: object,
    stu_by_frame: Mapping[int, object],
    *,
    canonical_ray_by_slot: np.ndarray | Mapping[int, np.ndarray],
    ray_mapping_digest: str,
    protocol: object,
) -> WindowTrainingData:
    """Build one bank row from a real WindowWorld and audited frozen STU outputs."""

    if type(protocol) is not AJAEProtocol or protocol.schema_version != SCHEMA_VERSION:
        raise TrainingError("WindowTrainingData construction requires schema 31")
    if __package__:
        from .render import WindowWorld
    else:
        from render import WindowWorld
    if type(window_world) is not WindowWorld:
        raise TrainingError("window_world must be an authoritative WindowWorld")
    frame_ids = tuple(getattr(window_world, "frame_ids", ()))
    rendered = tuple(getattr(window_world, "rendered_frames", ()))
    if (
        len(frame_ids) != 5
        or len(rendered) != 5
        or tuple(getattr(item, "source").frame_id for item in rendered)
        != frame_ids
    ):
        raise TrainingError("window_world must contain five ordered rendered frames")
    if set(stu_by_frame) != set(frame_ids):
        raise TrainingError("STU outputs must identify exactly the five source frames")
    training = _require_mapping(
        getattr(protocol, "training", None), "protocol.training"
    )
    if (
        training["source_partition"] != "train"
        or training["source_sequence_id"] != 206
    ):
        raise TrainingError("training source must remain train/206")
    scene = assemble_window(
        protocol.normal_training,
        int(getattr(window_world, "window_start")),
        frame_ids,
        tuple(item.source for item in rendered),
        condition=ExperimentCondition.B3,
        canonical_ray_by_slot=canonical_ray_by_slot,
        ray_mapping_audited=True,
        ray_mapping_digest=ray_mapping_digest,
    )
    if scene.labels is None:
        raise TrainingError("training WindowWorld must retain rendered labels")
    reference_pose = getattr(window_world, "reference_pose", None)
    if reference_pose is None or not np.allclose(
        scene.reference_pose.world_from_window,
        reference_pose.world_from_window,
        rtol=1.0e-9,
        atol=1.0e-9,
    ):
        raise TrainingError("WindowWorld symmetric reference pose changed")

    stu_features: list[Tensor] = []
    normal_evidence: list[Tensor] = []
    reliability_assign: list[Tensor] = []
    reliability_noobj: list[Tensor] = []
    intensity: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    stu_inputs: list[str] = []
    for rendered_frame in rendered:
        source = rendered_frame.source
        encoding = stu_by_frame[source.frame_id]
        if type(encoding) is not STUPointEncoding:
            raise TrainingError(
                "STU evidence must be a FrozenSTUPointEncoder output"
            )
        real_slots = getattr(encoding, "real_slots", None)
        if not isinstance(real_slots, Tensor) or not np.array_equal(
            real_slots.detach().cpu().numpy(), source.real_slots
        ):
            raise TrainingError(
                f"STU evidence rows do not match frame {source.frame_id} real slots"
            )
        if encoding.input_identity != stu_input_identity(
            source.coordinates, source.features, source.real_slots
        ):
            raise TrainingError(
                f"STU evidence was not computed from rendered frame {source.frame_id}"
            )
        stu_inputs.append(encoding.input_identity)
        point_features = getattr(encoding, "point_features", None)
        evidence = getattr(encoding, "normal_evidence", None)
        assignment = getattr(encoding, "reliability_assign", None)
        no_object = getattr(encoding, "reliability_noobj", None)
        if not all(isinstance(value, Tensor) for value in (
            point_features, evidence, assignment, no_object
        )):
            raise TrainingError("STU evidence is missing a required frozen tensor")
        if any(value.requires_grad for value in (
            point_features, evidence, assignment, no_object
        )):
            raise TrainingError("STU evidence must be frozen before bank construction")
        stu_features.append(point_features.detach().cpu())
        normal_evidence.append(evidence.detach().cpu())
        reliability_assign.append(assignment.detach().cpu())
        reliability_noobj.append(no_object.detach().cpu())
        xyzi = source.xyzi[source.real_slots]
        intensity.append(xyzi[:, 3].astype(np.float32, copy=False))
        ranges.append(np.linalg.norm(
            xyzi[:, :3].astype(np.float32, copy=False), axis=1
        ))

    semantic = scene.labels.semantic
    point_ranges = np.concatenate(ranges)
    evaluation = getattr(protocol, "evaluation", None)
    minimum_range = float(getattr(evaluation, "minimum_range_m"))
    maximum_range = float(getattr(evaluation, "maximum_range_m"))
    target = semantic == np.uint16(2)
    valid = (
        (semantic != np.uint16(0))
        & (point_ranges >= minimum_range)
        & (point_ranges <= maximum_range)
    )
    manifest_method = getattr(window_world, "to_manifest", None)
    if not callable(manifest_method):
        raise TrainingError("window_world does not expose to_manifest()")
    world = getattr(window_world, "world", None)
    world_identity = getattr(world, "identity", None)
    source_observations = tuple(
        source_observation_identity(item.source) for item in rendered
    )
    manifest = manifest_method()
    if tuple(manifest.get("source_observation_identities", ())) != (
        source_observations
    ):
        raise TrainingError("WindowWorld manifest changed its rendered observations")
    return WindowTrainingData(
        coordinates=torch.from_numpy(scene.points.coordinates.copy()),
        scan_group=torch.from_numpy(scene.points.scan_group.copy()),
        stu_features=torch.cat(stu_features),
        normal_evidence=torch.cat(normal_evidence),
        reliability_assign=torch.cat(reliability_assign),
        reliability_noobj=torch.cat(reliability_noobj),
        intensity=torch.from_numpy(np.concatenate(intensity)),
        target=torch.from_numpy(target.copy()),
        valid=torch.from_numpy(valid.copy()),
        source_frame=torch.from_numpy(scene.points.source_frame.copy()),
        source_slot=torch.from_numpy(scene.points.source_slot.copy()),
        source_ray=torch.from_numpy(scene.points.source_ray.copy()),
        world_identity=world_identity,
        source_observation_identities=source_observations,  # type: ignore[arg-type]
        stu_input_identities=tuple(stu_inputs),  # type: ignore[arg-type]
        window_manifest=manifest,
    )


class WindowTrainingBank(Sequence[WindowTrainingData]):
    """A condition-independent, hash-bound view of frozen training shards."""

    def __init__(self, manifest: Path | str, *, protocol: object) -> None:
        requested = Path(manifest).expanduser()
        manifest_path = requested / MANIFEST_NAME if requested.is_dir() else requested
        try:
            self.manifest_path = manifest_path.resolve(strict=True)
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingError("window training bank manifest is unreadable") from error
        root = _require_mapping(document, "window training bank manifest")
        _require_exact_keys(root, {
            "format", "schema_version", "name", "protocol_identity",
            "bank_identity", "source_partition", "source_sequence_id",
            "renderer_identity", "stu_identity", "shared_by", "entry_count",
            "entries",
        }, "window training bank manifest")
        if root["format"] != WINDOW_BANK_FORMAT or root["schema_version"] != SCHEMA_VERSION:
            raise TrainingError("window training bank is not schema 31")
        training = _require_mapping(getattr(protocol, "training", None), "protocol.training")
        bank_spec = _require_mapping(training["bank"], "protocol.training.bank")
        if root["name"] != bank_spec["name"]:
            raise TrainingError("window training bank name differs from the protocol")
        expected_protocol = protocol_bank_identity(protocol)
        if root["protocol_identity"] != expected_protocol:
            raise TrainingError("window training bank protocol identity changed")
        identity_payload = dict(root)
        declared_bank_identity = _require_digest(
            identity_payload.pop("bank_identity"), "bank_identity"
        )
        if _identity_digest(identity_payload) != declared_bank_identity:
            raise TrainingError("window training bank manifest identity changed")
        partition = str(training["source_partition"])
        sequence_id = int(training["source_sequence_id"])
        if root["source_partition"] != partition or root["source_sequence_id"] != sequence_id:
            raise TrainingError("window training bank source differs from the protocol")
        renderer_identity = _require_digest(root["renderer_identity"], "renderer_identity")
        stu_identity = _require_digest(root["stu_identity"], "stu_identity")
        protocol_stu = _require_mapping(getattr(protocol, "stu", None), "protocol.stu")
        if stu_identity != protocol_stu["checkpoint_sha256"]:
            raise TrainingError("window training bank uses a different frozen STU")
        if root["shared_by"] != ["B1", "B2", "B3"]:
            raise TrainingError("one bank must be shared by B1, B2, and B3")
        raw_entries = root["entries"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise TrainingError("window training bank entries must be a non-empty array")
        if root["entry_count"] != len(raw_entries):
            raise TrainingError("window training bank entry_count is inconsistent")

        bank_root = self.manifest_path.parent.resolve()
        entries: list[WindowBankEntry] = []
        used_shards: set[Path] = set()
        used_windows: set[str] = set()
        used_worlds: set[str] = set()
        for position, raw_entry in enumerate(raw_entries):
            entry = _require_mapping(raw_entry, f"entries[{position}]")
            _require_exact_keys(entry, {
                "position", "shard", "shard_sha256", "point_count",
                "world_identity", "window_identity", "source_frames",
                "source_observation_identities", "stu_input_identities",
                "target_count", "valid_count", "anomaly_count", "normal_count",
                "ignored_count",
                "point_identity_sha256", "labels_sha256", "window_manifest",
            }, f"entries[{position}]")
            if entry["position"] != position:
                raise TrainingError("window training bank positions must be contiguous")
            raw_shard = entry["shard"]
            if not isinstance(raw_shard, str) or not raw_shard or Path(raw_shard).is_absolute():
                raise TrainingError("bank shard paths must be non-empty relative paths")
            try:
                shard = (bank_root / raw_shard).resolve(strict=True)
            except OSError as error:
                raise TrainingError("bank shard is missing") from error
            if bank_root != shard.parent and bank_root not in shard.parents:
                raise TrainingError("bank shard escapes its bank directory")
            if shard.suffix != ".npz" or shard in used_shards:
                raise TrainingError("bank shards must be unique .npz files")
            used_shards.add(shard)
            world_identity = _require_digest(entry["world_identity"], "world_identity")
            window_identity = _require_digest(entry["window_identity"], "window_identity")
            window_manifest = _normalized_window_manifest(entry["window_manifest"])
            if (
                window_manifest.get("partition") != partition
                or window_manifest.get("sequence_id") != sequence_id
                or window_manifest.get("renderer_identity") != renderer_identity
                or window_manifest.get("identity") != window_identity
                or window_manifest.get("world_identity") != world_identity
            ):
                raise TrainingError(
                    "bank entry identities disagree with its WindowWorld manifest"
                )
            raw_frames = entry["source_frames"]
            if not isinstance(raw_frames, list) or len(raw_frames) != 5:
                raise TrainingError("source_frames must contain five frame IDs")
            frames = tuple(_require_int(value, "source frame") for value in raw_frames)
            if frames != tuple(range(frames[0], frames[0] + 5)):
                raise TrainingError("source_frames must be five consecutive frames")
            raw_observations = entry["source_observation_identities"]
            raw_stu_inputs = entry["stu_input_identities"]
            if (
                not isinstance(raw_observations, list)
                or len(raw_observations) != 5
                or not isinstance(raw_stu_inputs, list)
                or len(raw_stu_inputs) != 5
            ):
                raise TrainingError("bank entry must bind five source/STU identities")
            observations = tuple(
                _require_digest(value, "source observation identity")
                for value in raw_observations
            )
            if len(set(observations)) != 5:
                raise TrainingError("bank entry source observations must be unique")
            stu_inputs = tuple(
                _require_digest(value, "STU input identity")
                for value in raw_stu_inputs
            )
            if window_manifest.get("source_observation_identities") != list(
                observations
            ):
                raise TrainingError("bank entry and WindowWorld observations differ")
            expected_window = _window_identity(
                world_identity=world_identity,
                source_frames=frames,
                source_observation_identities=observations,
                renderer_identity=renderer_identity,
                partition=partition,
                sequence_id=sequence_id,
            )
            if window_identity != expected_window or window_identity in used_windows:
                raise TrainingError("window identity is invalid or repeated")
            if world_identity in used_worlds:
                raise TrainingError("each bank entry must use a unique WorldSpec")
            used_windows.add(window_identity)
            used_worlds.add(world_identity)
            window_frames = getattr(protocol, "window_frame_ids", None)
            if not callable(window_frames) or tuple(
                window_frames(partition, sequence_id, frames[0])
            ) != frames:
                raise TrainingError("bank entry is not a legal protocol window")
            point_count = _require_int(
                entry["point_count"], "point_count", minimum=1
            )
            target_count = _require_int(entry["target_count"], "target_count")
            valid_count = _require_int(
                entry["valid_count"], "valid_count", minimum=1
            )
            anomaly_count = _require_int(
                entry["anomaly_count"], "anomaly_count"
            )
            normal_count = _require_int(entry["normal_count"], "normal_count")
            ignored_count = _require_int(entry["ignored_count"], "ignored_count")
            if (
                target_count > point_count
                or valid_count + ignored_count != point_count
                or anomaly_count + normal_count != valid_count
                or anomaly_count > target_count
            ):
                raise TrainingError("bank entry label counts are inconsistent")
            _validate_window_manifest_record(
                window_manifest,
                expected_world_identity=world_identity,
                expected_source_frames=frames,
                expected_source_observation_identities=observations,
                expected_target_count=target_count,
            )
            entries.append(WindowBankEntry(
                position=position,
                shard=shard,
                shard_sha256=_require_digest(entry["shard_sha256"], "shard_sha256"),
                point_count=point_count,
                world_identity=world_identity,
                window_identity=window_identity,
                source_frames=frames,  # type: ignore[arg-type]
                source_observation_identities=observations,  # type: ignore[arg-type]
                stu_input_identities=stu_inputs,  # type: ignore[arg-type]
                target_count=target_count,
                valid_count=valid_count,
                anomaly_count=anomaly_count,
                normal_count=normal_count,
                ignored_count=ignored_count,
                point_identity_sha256=_require_digest(
                    entry["point_identity_sha256"], "point_identity_sha256"
                ),
                labels_sha256=_require_digest(
                    entry["labels_sha256"], "labels_sha256"
                ),
                window_manifest=_freeze_json(window_manifest),  # type: ignore[arg-type]
            ))
        self.protocol_identity = expected_protocol
        self.bank_identity = declared_bank_identity
        self.renderer_identity = renderer_identity
        self.stu_identity = stu_identity
        self.source_partition = partition
        self.source_sequence_id = sequence_id
        self.entries = tuple(entries)
        self._verified_shards: dict[Path, tuple[int, int, int, int, int]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(
        self, index: int | slice
    ) -> WindowTrainingData | tuple[WindowTrainingData, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        entry = self.entries[index]
        fingerprint = _file_fingerprint(entry.shard)
        if self._verified_shards.get(entry.shard) != fingerprint:
            if _sha256_file(entry.shard) != entry.shard_sha256:
                raise TrainingError(f"training shard hash changed: {entry.shard.name}")
            if _file_fingerprint(entry.shard) != fingerprint:
                raise TrainingError(f"training shard changed while hashing: {entry.shard.name}")
            self._verified_shards[entry.shard] = fingerprint
        try:
            with np.load(entry.shard, allow_pickle=False) as archive:
                if set(archive.files) != _SHARD_ARRAYS:
                    raise TrainingError(f"training shard arrays differ: {entry.shard.name}")
                arrays = {name: np.array(archive[name], copy=True) for name in _SHARD_ARRAYS}
        except (OSError, ValueError) as error:
            raise TrainingError(f"cannot read training shard: {entry.shard.name}") from error
        if _file_fingerprint(entry.shard) != fingerprint:
            raise TrainingError(f"training shard changed while reading: {entry.shard.name}")

        def floating(name: str) -> Tensor:
            value = arrays[name]
            if value.dtype.kind != "f":
                raise TrainingError(f"{name} must be floating point in the shard")
            return torch.from_numpy(value.astype(np.float32, copy=False))

        def integer(name: str) -> Tensor:
            value = arrays[name]
            if value.dtype.kind not in "iu":
                raise TrainingError(f"{name} must be integer in the shard")
            return torch.from_numpy(value.astype(np.int64, copy=False))

        def boolean(name: str) -> Tensor:
            value = arrays[name]
            if value.dtype.kind not in "biu" or not np.all((value == 0) | (value == 1)):
                raise TrainingError(f"{name} must contain only boolean 0/1 values")
            return torch.from_numpy(value.astype(np.bool_, copy=False))

        data = WindowTrainingData(
            coordinates=floating("coordinates"),
            scan_group=integer("scan_group"),
            stu_features=floating("stu_features"),
            normal_evidence=floating("normal_evidence"),
            reliability_assign=floating("reliability_assign"),
            reliability_noobj=floating("reliability_noobj"),
            intensity=floating("intensity"),
            target=boolean("target"),
            valid=boolean("valid"),
            source_frame=integer("source_frame"),
            source_slot=integer("source_slot"),
            source_ray=integer("source_ray"),
            world_identity=entry.world_identity,
            source_observation_identities=entry.source_observation_identities,
            stu_input_identities=entry.stu_input_identities,
            window_manifest=entry.window_manifest,
            window_identity=entry.window_identity,
        )
        (
            target_count,
            valid_count,
            anomaly_count,
            normal_count,
            ignored_count,
        ) = _label_counts(data)
        if (
            data.point_count != entry.point_count
            or data.five_source_frames != entry.source_frames
            or target_count != entry.target_count
            or valid_count != entry.valid_count
            or anomaly_count != entry.anomaly_count
            or normal_count != entry.normal_count
            or ignored_count != entry.ignored_count
            or _point_identity_digest(data) != entry.point_identity_sha256
            or _label_identity_digest(data) != entry.labels_sha256
        ):
            raise TrainingError("training shard content disagrees with its manifest entry")
        return data

    def __iter__(self) -> Iterator[WindowTrainingData]:
        for position in range(len(self)):
            yield self[position]  # type: ignore[misc]


def load_window_train_bank(
    manifest: Path | str, *, protocol: object
) -> WindowTrainingBank:
    """Load the sole condition-independent schema-31 training bank."""

    return WindowTrainingBank(manifest, protocol=protocol)


def _window_arrays(window: WindowTrainingData) -> dict[str, np.ndarray]:
    return {
        "coordinates": window.coordinates.numpy(),
        "scan_group": window.scan_group.numpy().astype(np.int8, copy=False),
        "stu_features": window.stu_features.numpy(),
        "normal_evidence": window.normal_evidence.numpy(),
        "reliability_assign": window.reliability_assign.numpy(),
        "reliability_noobj": window.reliability_noobj.numpy(),
        "intensity": window.intensity.numpy(),
        "target": window.target.numpy(),
        "valid": window.valid.numpy(),
        "source_frame": window.source_frame.numpy().astype(np.int32, copy=False),
        "source_slot": window.source_slot.numpy().astype(np.int32, copy=False),
        "source_ray": window.source_ray.numpy().astype(np.int32, copy=False),
    }


def write_window_train_bank(
    destination: Path | str,
    windows: Iterable[WindowTrainingData],
    *,
    protocol: object,
    renderer_identity: str,
    stu_identity: str | None = None,
) -> WindowTrainingBank:
    """Atomically write shards plus complete normalized WindowWorld manifests."""

    final = Path(destination).expanduser().resolve()
    if final.exists():
        raise TrainingError("refusing to replace an existing frozen training bank")
    final.parent.mkdir(parents=True, exist_ok=True)
    renderer = _require_digest(renderer_identity, "renderer_identity")
    training = _require_mapping(getattr(protocol, "training", None), "protocol.training")
    bank_spec = _require_mapping(training["bank"], "protocol.training.bank")
    stu = _require_mapping(getattr(protocol, "stu", None), "protocol.stu")
    frozen_stu = _require_digest(
        stu["checkpoint_sha256"] if stu_identity is None else stu_identity,
        "stu_identity",
    )
    if frozen_stu != stu["checkpoint_sha256"]:
        raise TrainingError("writer STU identity differs from the protocol")
    partition = str(training["source_partition"])
    sequence_id = int(training["source_sequence_id"])
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        shard_directory = temporary / "shards"
        shard_directory.mkdir()
        entries: list[dict[str, object]] = []
        for position, window in enumerate(windows):
            if not isinstance(window, WindowTrainingData):
                raise TrainingError("bank writer accepts only WindowTrainingData entries")
            source_frames = window.five_source_frames
            expected_window = _window_identity(
                world_identity=window.world_identity,
                source_frames=source_frames,
                source_observation_identities=(
                    window.source_observation_identities
                ),
                renderer_identity=renderer,
                partition=partition,
                sequence_id=sequence_id,
            )
            if window.window_identity and window.window_identity != expected_window:
                raise TrainingError("WindowTrainingData.window_identity is inconsistent")
            if window.window_identity != expected_window:
                raise TrainingError(
                    "WindowTrainingData manifest uses another renderer or source"
                )
            window_manifest = _normalized_window_manifest(
                window.window_manifest
            )
            if (
                window_manifest["partition"] != partition
                or window_manifest["sequence_id"] != sequence_id
                or window_manifest["renderer_identity"] != renderer
            ):
                raise TrainingError(
                    "WindowTrainingData manifest differs from the bank identity"
                )
            relative = Path("shards") / f"{position:06d}.npz"
            shard = temporary / relative
            with shard.open("wb") as handle:
                np.savez_compressed(handle, **_window_arrays(window))
                handle.flush()
                os.fsync(handle.fileno())
            (
                target_count,
                valid_count,
                anomaly_count,
                normal_count,
                ignored_count,
            ) = _label_counts(window)
            entries.append({
                "position": position,
                "shard": relative.as_posix(),
                "shard_sha256": _sha256_file(shard),
                "point_count": window.point_count,
                "world_identity": window.world_identity,
                "window_identity": expected_window,
                "source_frames": list(source_frames),
                "source_observation_identities": list(
                    window.source_observation_identities
                ),
                "stu_input_identities": list(window.stu_input_identities),
                "target_count": target_count,
                "valid_count": valid_count,
                "anomaly_count": anomaly_count,
                "normal_count": normal_count,
                "ignored_count": ignored_count,
                "point_identity_sha256": _point_identity_digest(window),
                "labels_sha256": _label_identity_digest(window),
                "window_manifest": window_manifest,
            })
        if not entries:
            raise TrainingError("cannot write an empty window training bank")
        manifest: dict[str, object] = {
            "format": WINDOW_BANK_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "name": bank_spec["name"],
            "protocol_identity": protocol_bank_identity(protocol),
            "source_partition": partition,
            "source_sequence_id": sequence_id,
            "renderer_identity": renderer,
            "stu_identity": frozen_stu,
            "shared_by": ["B1", "B2", "B3"],
            "entry_count": len(entries),
            "entries": entries,
        }
        manifest["bank_identity"] = _identity_digest(manifest)
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True,
                allow_nan=False,
            ) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(shard_directory)
        _fsync_directory(temporary)
        WindowTrainingBank(manifest_path, protocol=protocol)
        os.replace(temporary, final)
        _fsync_directory(final.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return WindowTrainingBank(final, protocol=protocol)


@dataclass(slots=True)
class EffectiveBatchBCE:
    """Retain class sums until every WindowWorld in one optimizer step is known."""

    positive_loss_sum: Tensor | None = None
    negative_loss_sum: Tensor | None = None
    positive_count: int = 0
    negative_count: int = 0

    def add(self, logits: Tensor, target: Tensor, valid: Tensor) -> None:
        if logits.ndim != 1 or target.shape != logits.shape or valid.shape != logits.shape:
            raise TrainingError("logits, target, and valid must be aligned [N] tensors")
        if target.dtype != torch.bool or valid.dtype != torch.bool:
            raise TrainingError("target and valid must be boolean")
        if logits.device != target.device or logits.device != valid.device:
            raise TrainingError("loss tensors must share a device")
        _finite_tensor(logits, "training logits")
        raw = F.binary_cross_entropy_with_logits(
            logits, target.float(), reduction="none"
        )
        positive = valid & target
        negative = valid & ~target
        positive_count = int(positive.sum().item())
        negative_count = int(negative.sum().item())
        if positive_count:
            value = raw[positive].sum()
            self.positive_loss_sum = (
                value if self.positive_loss_sum is None
                else self.positive_loss_sum + value
            )
            self.positive_count += positive_count
        if negative_count:
            value = raw[negative].sum()
            self.negative_loss_sum = (
                value if self.negative_loss_sum is None
                else self.negative_loss_sum + value
            )
            self.negative_count += negative_count

    def loss(self) -> Tensor:
        if not self.positive_count and not self.negative_count:
            raise TrainingError("effective batch contains no valid targets")
        if self.positive_count and self.negative_count:
            assert self.positive_loss_sum is not None
            assert self.negative_loss_sum is not None
            return (
                0.5 * self.positive_loss_sum / self.positive_count
                + 0.5 * self.negative_loss_sum / self.negative_count
            )
        if self.positive_count:
            assert self.positive_loss_sum is not None
            return self.positive_loss_sum / self.positive_count
        assert self.negative_loss_sum is not None
        return self.negative_loss_sum / self.negative_count


def effective_batch_balanced_bce(
    batches: Iterable[tuple[Tensor, Tensor, Tensor]],
) -> Tensor:
    """Balance classes after aggregating every supplied WindowWorld micro-batch."""

    aggregate = EffectiveBatchBCE()
    for logits, target, valid in batches:
        aggregate.add(logits, target, valid)
    return aggregate.loss()


@dataclass(frozen=True, slots=True)
class _DeviceWindow:
    coordinates: Tensor
    scan_group: Tensor
    stu_features: Tensor
    normal_evidence: Tensor
    reliability_assign: Tensor
    reliability_noobj: Tensor
    intensity: Tensor
    target: Tensor
    valid: Tensor


def _to_device(window: WindowTrainingData, device: torch.device) -> _DeviceWindow:
    non_blocking = device.type == "cuda"

    def move(value: Tensor) -> Tensor:
        return value.to(device=device, non_blocking=non_blocking)

    return _DeviceWindow(
        move(window.coordinates),
        move(window.scan_group),
        move(window.stu_features),
        move(window.normal_evidence),
        move(window.reliability_assign),
        move(window.reliability_noobj),
        move(window.intensity),
        move(window.target),
        move(window.valid),
    )


def _forward_rows(
    model: nn.Module,
    window: _DeviceWindow,
    rows: Tensor,
    *,
    grouping_mode: GroupingMode,
    erase_group_identity: bool = False,
) -> Tensor:
    groups = window.scan_group[rows]
    if erase_group_identity:
        groups = torch.zeros_like(groups)
    logits = model(
        window.coordinates[rows],
        window.stu_features[rows],
        window.normal_evidence[rows],
        window.reliability_assign[rows],
        window.reliability_noobj[rows],
        window.intensity[rows],
        groups,
        grouping_mode=grouping_mode,
    )
    expected = int(rows.sum().item()) if rows.dtype == torch.bool else int(rows.numel())
    if not isinstance(logits, Tensor) or tuple(logits.shape) != (expected,):
        raise TrainingError("model must return one anomaly logit for every selected point")
    _finite_tensor(logits, "model logits")
    return logits


def _predict_window(
    model: nn.Module,
    window: WindowTrainingData,
    condition: ExperimentCondition | str,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Run the shared grouping semantics; callers enforce model authority."""

    selected = ExperimentCondition(condition)
    if not selected.trainable:
        raise TrainingError("B0 has no AJAE training forward path")
    if device is None:
        parameter = next(model.parameters(), None)
        resolved = torch.device("cpu" if parameter is None else parameter.device)
    else:
        resolved = torch.device(device)
    batch = _to_device(window, resolved)
    if selected is ExperimentCondition.B1:
        row_parts: list[Tensor] = []
        logit_parts: list[Tensor] = []
        for group in WINDOW_GROUPS:
            rows = torch.nonzero(
                batch.scan_group == group, as_tuple=False
            ).squeeze(1)
            row_parts.append(rows)
            logit_parts.append(_forward_rows(
                model,
                batch,
                rows,
                grouping_mode=GroupingMode.SINGLE,
                erase_group_identity=True,
            ))
        joined_rows = torch.cat(row_parts)
        order = torch.argsort(joined_rows)
        expected_rows = torch.arange(window.point_count, device=resolved)
        if not torch.equal(joined_rows[order], expected_rows):
            raise TrainingError("B1 scan forwards did not cover every original row once")
        return torch.cat(logit_parts)[order]
    rows = torch.arange(window.point_count, device=resolved)
    mode = (
        GroupingMode.PER_SCAN
        if selected is ExperimentCondition.B2
        else GroupingMode.JOINT
    )
    return _forward_rows(model, batch, rows, grouping_mode=mode)


def predict_window(
    model: JointWindowPointTransformer,
    window: WindowTrainingData,
    condition: ExperimentCondition | str,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Run the sole public schema-31 AJAE model path."""

    if type(model) is not JointWindowPointTransformer:
        raise TrainingError(
            "predict_window requires the authoritative JointWindowPointTransformer"
        )
    return _predict_window(model, window, condition, device=device)


def _predict_window_for_test(
    model: nn.Module,
    window: WindowTrainingData,
    condition: ExperimentCondition | str,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Exercise grouping behavior with a small explicit test double."""

    return _predict_window(model, window, condition, device=device)


def binary_average_precision(
    probabilities: Tensor, target: Tensor, valid: Tensor
) -> float:
    """Compute threshold-grouped point AP without making tie order informative."""

    if (
        probabilities.ndim != 1
        or target.shape != probabilities.shape
        or valid.shape != probabilities.shape
    ):
        raise TrainingError("AP inputs must be aligned [N] tensors")
    selected = valid.bool()
    score = probabilities[selected].detach().cpu().double()
    label = target[selected].detach().cpu().bool()
    if score.numel() == 0 or not bool(label.any()):
        return math.nan
    _finite_tensor(score, "AP probabilities")
    order = torch.argsort(score, descending=True, stable=True)
    score = score[order]
    label = label[order]
    true_positive = label.long().cumsum(0)
    false_positive = (~label).long().cumsum(0)
    final = torch.ones(1, dtype=torch.bool)
    ends = torch.nonzero(
        torch.cat((score[1:] != score[:-1], final))
    ).squeeze(1)
    precision = true_positive[ends].double() / (
        true_positive[ends] + false_positive[ends]
    )
    recall = true_positive[ends].double() / int(label.sum().item())
    previous = torch.cat((torch.zeros(1, dtype=torch.double), recall[:-1]))
    return float(((recall - previous) * precision).sum().item())


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    learning_rate: float
    gradient_accumulation: int
    weight_decay: float = 0.0
    scheduler: str = "constant"
    epochs: int = 1
    maximum_updates: int | None = None
    evaluation_interval_updates: int = 1
    gradient_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if _require_number(self.learning_rate, "learning_rate") <= 0.0:
            raise TrainingError("learning_rate must be positive")
        _require_int(
            self.gradient_accumulation, "gradient_accumulation", minimum=1
        )
        _require_number(self.weight_decay, "weight_decay")
        if self.scheduler not in {"constant", "five_percent_warmup_cosine"}:
            raise TrainingError("scheduler is not recognized")
        _require_int(self.epochs, "epochs", minimum=1)
        if self.maximum_updates is not None:
            _require_int(self.maximum_updates, "maximum_updates", minimum=1)
        _require_int(
            self.evaluation_interval_updates,
            "evaluation_interval_updates",
            minimum=1,
        )
        if self.gradient_clip_norm is not None and _require_number(
            self.gradient_clip_norm, "gradient_clip_norm"
        ) <= 0.0:
            raise TrainingError("gradient_clip_norm must be positive")


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    mode: TrainMode
    condition: ExperimentCondition
    seed: int
    window_count: int
    maximum_updates: int


def validate_training_request(
    protocol: object,
    mode: TrainMode | str,
    condition: ExperimentCondition | str,
    seed: int,
    config: TrainingConfig,
    *,
    bank_size: int,
) -> TrainingRequest:
    """Bind mode, seed, subset size, and search values to the protocol."""

    if type(protocol) is not AJAEProtocol or protocol.schema_version != SCHEMA_VERSION:
        raise TrainingError("training requests require the schema-31 protocol")
    selected_mode = TrainMode(mode)
    selected_condition = ExperimentCondition(condition)
    if not selected_condition.trainable:
        raise TrainingError("training condition must be one of B1, B2, or B3")
    seed = _require_int(seed, "seed")
    bank_size = _require_int(bank_size, "bank_size", minimum=1)
    training = _require_mapping(
        getattr(protocol, "training", None), "protocol.training"
    )
    status = _require_mapping(
        getattr(protocol, "status", None), "protocol.status"
    )
    state_machine = tuple(getattr(protocol, "state_machine", ()))
    node = status.get("current_node")
    if node not in state_machine:
        raise TrainingError("protocol current node is outside its state machine")
    if (
        selected_mode in {TrainMode.TINY_OVERFIT, TrainMode.PILOT}
        and node not in {"R04", "R05"}
    ):
        raise TrainingError(
            f"{selected_mode.value} training is permitted only during R04 or R05"
        )
    if tuple(training["modes"]) != tuple(item.value for item in TrainMode):
        raise TrainingError("protocol training modes differ from the implementation")
    pilot = _require_mapping(training["pilot"], "training.pilot")
    formal = _require_mapping(training["formal"], "training.formal")
    pilot_seeds = tuple(int(value) for value in pilot["seeds"])
    formal_seeds = tuple(int(value) for value in formal["seeds"])
    if set(pilot_seeds) & set(formal_seeds):
        raise TrainingError("pilot and formal seeds must be disjoint")
    if selected_mode is TrainMode.TINY_OVERFIT:
        if type(config) is not TrainingConfig:
            raise TrainingError("training config must be TrainingConfig")
        tiny = _require_mapping(
            training["tiny_overfit"], "training.tiny_overfit"
        )
        windows = int(tiny["windows"])
        maximum_updates = int(tiny["maximum_updates"])
        if config.maximum_updates not in {None, maximum_updates}:
            raise TrainingError(
                "tiny_overfit maximum_updates is fixed by the protocol"
            )
    elif selected_mode is TrainMode.PILOT:
        if type(config) is not TrainingConfig:
            raise TrainingError("training config must be TrainingConfig")
        if seed not in pilot_seeds:
            raise TrainingError(
                "pilot seed is outside the result-blind pilot seed set"
            )
        if config.learning_rate not in tuple(
            float(value) for value in pilot["learning_rates"]
        ):
            raise TrainingError("pilot learning_rate is outside the protocol grid")
        if config.gradient_accumulation not in tuple(
            int(value) for value in pilot["gradient_accumulation"]
        ):
            raise TrainingError(
                "pilot gradient_accumulation is outside the protocol grid"
            )
        if config.scheduler not in tuple(
            pilot["schedulers_after_learning_rate_selection"]
        ):
            raise TrainingError("pilot scheduler is outside the protocol grid")
        if config.weight_decay not in tuple(
            float(value)
            for value in pilot["weight_decay_after_scheduler_selection"]
        ):
            raise TrainingError("pilot weight_decay is outside the protocol grid")
        screens = tuple(int(value) for value in pilot["screen_updates"])
        if config.maximum_updates not in screens:
            raise TrainingError(
                "pilot maximum_updates must select a protocol screen"
            )
        windows = int(pilot["windows"])
        maximum_updates = int(config.maximum_updates)
    else:
        if seed not in formal_seeds:
            raise TrainingError(
                "formal seed is outside the frozen formal seed set"
            )
        if (
            status.get("formal_training_allowed") is not True
            or formal.get("recipe_status") != "frozen_result_blind_in_R05"
            or "R05" not in state_machine
            or node not in state_machine
            or state_machine.index(node) <= state_machine.index("R05")
        ):
            raise TrainingError(
                "formal training requires an enabled, completed R05 freeze"
            )
        if type(config) is not TrainingConfig:
            raise TrainingError("training config must be TrainingConfig")
        windows = bank_size
        steps_per_epoch = math.ceil(windows / config.gradient_accumulation)
        maximum_updates = (
            config.maximum_updates
            if config.maximum_updates is not None
            else config.epochs * steps_per_epoch
        )
    if bank_size < windows:
        raise TrainingError(
            f"{selected_mode.value} requires {windows} frozen windows, "
            f"bank has {bank_size}"
        )
    return TrainingRequest(
        selected_mode, selected_condition, seed, windows, maximum_updates
    )


@dataclass(frozen=True, slots=True)
class _DevelopmentSet:
    definitions: DevelopmentWorlds
    rendered_clips: tuple[DevelopmentClipWorld, ...]
    identity: str
    method_protocol_identity: str


def _load_development_set(
    path: Path | str,
    rendered_clips: Iterable[DevelopmentClipWorld],
    *,
    protocol: AJAEProtocol,
    renderer_identity: str,
) -> _DevelopmentSet:
    """Bind runtime clips to the exact validated 24+6 frozen definitions."""

    definitions = load_development_worlds(path, protocol=protocol)
    if (
        type(definitions) is not DevelopmentWorlds
        or not definitions.validated
        or definitions.protocol_identity != protocol.development_population_identity
    ):
        raise TrainingError("development worlds must be validated and frozen")
    expected_design = protocol.evaluation_document["synthetic_development"][
        "exact_clip_length_and_count"
    ]
    expected_counts = {
        "in_generator": int(expected_design["in_generator_clips"]),
        "torus_SDF": int(expected_design["held_out_torus_clips"]),
    }
    clips = tuple(definitions.clips)
    if (
        len(clips) != int(expected_design["total_clips"])
        or {
            mechanism: sum(item.mechanism == mechanism for item in clips)
            for mechanism in expected_counts
        }
        != expected_counts
        or any(
            len(item.frame_ids) != int(expected_design["frames_per_clip"])
            or len(item.windows) != int(expected_design["overlapping_windows_per_clip"])
            for item in clips
        )
    ):
        raise TrainingError("development definitions differ from the frozen 24+6 design")
    if any(item.renderer_identity != renderer_identity for item in clips):
        raise TrainingError("training and development data use different renderers")

    runtime = tuple(rendered_clips)
    if len(runtime) != len(clips) or any(
        type(item) is not DevelopmentClipWorld for item in runtime
    ):
        raise TrainingError(
            "development input must contain exactly 30 DevelopmentClipWorld values"
        )
    runtime_by_identity = {item.identity: item for item in runtime}
    if len(runtime_by_identity) != len(runtime) or set(runtime_by_identity) != {
        item.identity for item in clips
    }:
        raise TrainingError("rendered development clips differ from the frozen set")
    ordered: list[DevelopmentClipWorld] = []
    for definition in clips:
        rendered = runtime_by_identity[definition.identity]
        if (
            rendered.world.identity != definition.world_identity
            or rendered.clip_start != definition.clip_start
            or tuple(rendered.frame_ids) != tuple(definition.frame_ids)
            or rendered.renderer_identity != definition.renderer_identity
            or rendered.mechanism != definition.mechanism
            or tuple(rendered.source_observation_identities)
            != tuple(definition.source_observation_identities)
            or tuple(item.identity for item in rendered.windows)
            != tuple(item.identity for item in definition.windows)
        ):
            raise TrainingError(
                "rendered development clip does not match its frozen identity"
            )
        ordered.append(rendered)
    population_identity = _identity_digest({
        "format": "ajae-schema31-development-population-v1",
        "protocol_identity": definitions.protocol_identity,
        "sequence_id": definitions.sequence_id,
        "clips": [
            {
                "clip_identity": item.identity,
                "world_identity": item.world_identity,
                "mechanism": item.mechanism,
                "source_observation_identities": list(
                    item.source_observation_identities
                ),
                "window_identities": [window.identity for window in item.windows],
            }
            for item in clips
        ],
    })
    return _DevelopmentSet(
        definitions,
        tuple(ordered),
        population_identity,
        protocol.scientific_identity,
    )


def _development_record(
    evidence: DevelopmentFusedAP,
    condition: ExperimentCondition,
    development: _DevelopmentSet,
) -> dict[str, object]:
    """Select checkpoints on 24 in-generator clips; retain torus only as diagnosis."""

    if type(evidence) is not DevelopmentFusedAP:
        raise TrainingError("development result must be DevelopmentFusedAP")
    if ExperimentCondition(evidence.condition) is not condition:
        raise TrainingError("development result belongs to another condition")
    if evidence.fusion_semantics != DEVELOPMENT_FUSION_SEMANTICS:
        raise TrainingError(
            "development result does not use all-occurrence fusion"
        )
    identity = evidence.evaluation_identity
    method_protocol_identity = (
        development.method_protocol_identity
        if type(development) is _DevelopmentSet
        else getattr(identity, "protocol_identity", None)  # Private test seam.
    )
    if (
        type(identity) is not EvaluationIdentity
        or identity.test_fixture
        or identity.protocol_schema != SCHEMA_VERSION
        or identity.protocol_identity != method_protocol_identity
        or identity.condition != condition.value
        or identity.model_class != "JointWindowPointTransformer"
        or identity.stu_class != "FrozenSTUPointEncoder"
    ):
        raise TrainingError("development result has a non-authoritative method identity")
    clips = tuple(evidence.clips)
    definitions = tuple(development.definitions.clips)
    if len(clips) != 30 or len(definitions) != 30:
        raise TrainingError("development result must contain the frozen 30 clips")
    records: list[dict[str, object]] = []
    for index, (clip, definition) in enumerate(zip(clips, definitions, strict=True)):
        if type(clip) is not DevelopmentClipResult or (
            clip.clip_identity != definition.identity
            or clip.world_identity != definition.world_identity
            or tuple(clip.source_observation_identities)
            != tuple(definition.source_observation_identities)
            or clip.mechanism != definition.mechanism
            or clip.frame_count != len(definition.frame_ids)
            or clip.window_count != len(definition.windows)
        ):
            raise TrainingError("development result differs from the frozen clip set")
        average_precision = _require_number(
            clip.fused_point_ap, f"clips[{index}].fused_point_ap"
        )
        if average_precision > 1.0:
            raise TrainingError(
                "development fused point AP must lie in [0,1]"
            )
        unique_points = _require_int(
            clip.unique_point_count,
            f"clips[{index}].unique_point_count",
            minimum=1,
        )
        occurrences = _require_int(
            clip.occurrence_count,
            f"clips[{index}].occurrence_count",
            minimum=1,
        )
        if not unique_points <= occurrences <= 5 * unique_points:
            raise TrainingError(
                "development occurrence count is outside strata 1..5"
            )
        histogram = dict(clip.occurrence_histogram)
        if set(histogram) != {str(value) for value in range(1, 6)}:
            raise TrainingError("development result omits an occurrence stratum")
        records.append({
            "clip_identity": clip.clip_identity,
            "world_identity": clip.world_identity,
            "source_observation_identities": [
                _require_digest(value, "development source observation identity")
                for value in clip.source_observation_identities
            ],
            "mechanism": clip.mechanism,
            "fused_point_ap": average_precision,
            "unique_point_count": unique_points,
            "occurrence_count": occurrences,
            "occurrence_histogram": histogram,
        })

    def macro(mechanism: str | None) -> float:
        values = [
            float(item["fused_point_ap"])
            for item in records
            if mechanism is None or item["mechanism"] == mechanism
        ]
        if not values:
            raise TrainingError("development result has an empty mechanism stratum")
        return math.fsum(values) / len(values)

    in_generator = macro("in_generator")
    held_out = macro("torus_SDF")
    all_clips = macro(None)
    declared = (
        (evidence.in_generator_macro_fused_point_ap, in_generator),
        (evidence.macro_fused_point_ap, in_generator),
        (evidence.held_out_macro_fused_point_ap, held_out),
        (evidence.all_clips_macro_fused_point_ap, all_clips),
    )
    if any(
        not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected in declared
    ):
        raise TrainingError("development macro AP disagrees with per-clip AP")
    return {
        "fusion_semantics": DEVELOPMENT_FUSION_SEMANTICS,
        "selection_metric": "in_generator_macro_fused_point_ap",
        "in_generator_macro_fused_point_ap": in_generator,
        "held_out_torus_macro_fused_point_ap": held_out,
        "all_clip_macro_fused_point_ap": all_clips,
        "held_out_torus_role": "diagnostic_only_excluded_from_checkpoint_selection",
        "development_population_identity": development.identity,
        "evaluation_identity": identity.to_dict(),
        "clips": records,
    }


def _resolve_device(value: torch.device | str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise TrainingError("training device must be cpu or cuda")
    return device


def _scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    total_updates: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _step: 1.0
        )
    warmup = max(1, math.ceil(0.05 * total_updates))

    def multiplier(step: int) -> float:
        completed = min(step + 1, total_updates)
        if completed <= warmup:
            return completed / warmup
        progress = (completed - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _finite_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            raise TrainingError(f"non-finite gradient in {name}")


@dataclass(frozen=True, slots=True)
class _SubsetMetrics:
    loss: float
    average_precision: float


class WindowTrainer:
    """Optimize one shared model path over complete WindowWorld micro-batches."""

    def __init__(
        self,
        *,
        protocol: object,
        bank: WindowTrainingBank,
        model: JointWindowPointTransformer,
        request: TrainingRequest,
        config: TrainingConfig,
        device: torch.device | str,
        output_directory: Path | str,
        development: _DevelopmentSet | None = None,
        resume: bool = False,
    ) -> None:
        if type(protocol) is not AJAEProtocol or protocol.schema_version != SCHEMA_VERSION:
            raise TrainingError("WindowTrainer requires the schema-31 protocol")
        if type(model) is not JointWindowPointTransformer:
            raise TrainingError(
                "training requires the authoritative JointWindowPointTransformer"
            )
        if protocol_bank_identity(protocol) != bank.protocol_identity:
            raise TrainingError("training bank belongs to another bank protocol")
        self.protocol = protocol
        self.protocol_identity = _require_digest(
            protocol.scientific_identity, "protocol.scientific_identity"
        )
        self.bank = bank
        self.model = model
        self.request = request
        self.config = config
        self.device = _resolve_device(device)
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.last_path = self.output_directory / "last.pt"
        self.best_path = self.output_directory / "best.pt"
        self.result_path = self.output_directory / "result.json"
        if not resume and any(path.exists() for path in (
            self.last_path, self.best_path, self.result_path
        )):
            raise TrainingError(
                "output already contains a training run; use resume"
            )
        if request.mode is TrainMode.TINY_OVERFIT and development is not None:
            raise TrainingError(
                "tiny_overfit uses only its frozen training subset"
            )
        if request.mode is not TrainMode.TINY_OVERFIT and development is None:
            raise TrainingError(
                "pilot and formal training require the exact frozen development set"
            )
        self.development = development
        self.development_encoder = (
            None
            if development is None
            else FrozenSTUPointEncoder.from_protocol(protocol)
        )
        self.model.to(self.device)
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise TrainingError("model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = _scheduler(
            self.optimizer, config.scheduler, request.maximum_updates
        )
        self.epoch = 0
        self.next_entry = 0
        self.updates = 0
        self.windows_seen = 0
        self.best_development_ap: float | None = None
        self.last_development: dict[str, object] | None = None
        self.last_loss = math.nan
        self.last_evaluated_update = -1
        if resume:
            self._restore()

    def _checkpoint(self) -> dict[str, object]:
        return {
            "format": TRAIN_CHECKPOINT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "protocol_identity": self.protocol_identity,
            "bank_protocol_identity": self.bank.protocol_identity,
            "bank_identity": self.bank.bank_identity,
            "renderer_identity": self.bank.renderer_identity,
            "stu_identity": self.bank.stu_identity,
            "development_population_identity": (
                None if self.development is None else self.development.identity
            ),
            "mode": self.request.mode.value,
            "condition": self.request.condition.value,
            "grouping_mode": self.request.condition.grouping_mode.value,
            "seed": self.request.seed,
            "config": asdict(self.config),
            "window_count": self.request.window_count,
            "maximum_updates": self.request.maximum_updates,
            "device": str(self.device),
            "epoch": self.epoch,
            "next_entry": self.next_entry,
            "updates": self.updates,
            "windows_seen": self.windows_seen,
            "last_loss": self.last_loss,
            "best_development_ap": self.best_development_ap,
            "last_development": self.last_development,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": _numpy_rng_record(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if self.device.type == "cuda"
                else None
            ),
        }

    def _save_last(self) -> None:
        _atomic_torch(self.last_path, self._checkpoint())

    def _restore(self) -> None:
        if not self.last_path.is_file():
            raise TrainingError("resume requested but last.pt does not exist")
        try:
            payload = torch.load(
                self.last_path,
                map_location=self.device,
                weights_only=True,
            )
        except Exception as error:
            raise TrainingError("cannot safely load last.pt") from error
        if not isinstance(payload, Mapping):
            raise TrainingError("last.pt is not a training checkpoint")
        expected = {
            "format": TRAIN_CHECKPOINT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "protocol_identity": self.protocol_identity,
            "bank_protocol_identity": self.bank.protocol_identity,
            "bank_identity": self.bank.bank_identity,
            "renderer_identity": self.bank.renderer_identity,
            "stu_identity": self.bank.stu_identity,
            "development_population_identity": (
                None if self.development is None else self.development.identity
            ),
            "mode": self.request.mode.value,
            "condition": self.request.condition.value,
            "grouping_mode": self.request.condition.grouping_mode.value,
            "seed": self.request.seed,
            "config": asdict(self.config),
            "window_count": self.request.window_count,
            "maximum_updates": self.request.maximum_updates,
            "device": str(self.device),
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise TrainingError(f"resume checkpoint {name} changed")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.epoch = _require_int(payload["epoch"], "checkpoint epoch")
        self.next_entry = _require_int(
            payload["next_entry"], "checkpoint next_entry"
        )
        self.updates = _require_int(payload["updates"], "checkpoint updates")
        self.windows_seen = _require_int(
            payload["windows_seen"], "checkpoint windows_seen"
        )
        self.last_loss = float(payload["last_loss"])
        best = payload["best_development_ap"]
        self.best_development_ap = None if best is None else float(best)
        development = payload["last_development"]
        self.last_development = (
            None if development is None else dict(development)
        )
        _restore_rng(payload, self.device)
        if self.next_entry >= self.request.window_count:
            raise TrainingError(
                "checkpoint cursor lies outside the selected bank"
            )

    def _training_subset_metrics(self) -> _SubsetMetrics:
        was_training = self.model.training
        self.model.eval()
        aggregate = EffectiveBatchBCE()
        probabilities: list[Tensor] = []
        targets: list[Tensor] = []
        validity: list[Tensor] = []
        with torch.no_grad():
            for position in range(self.request.window_count):
                window = self.bank[position]
                logits = predict_window(
                    self.model,
                    window,
                    self.request.condition,
                    device=self.device,
                )
                target = window.target.to(self.device)
                valid = window.valid.to(self.device)
                aggregate.add(logits, target, valid)
                probabilities.append(logits.sigmoid().cpu())
                targets.append(window.target)
                validity.append(window.valid)
        if was_training:
            self.model.train()
        return _SubsetMetrics(
            float(aggregate.loss().item()),
            binary_average_precision(
                torch.cat(probabilities),
                torch.cat(targets),
                torch.cat(validity),
            ),
        )

    def _evaluate_development(self) -> None:
        if self.development is None or self.development_encoder is None:
            return
        was_training = self.model.training
        try:
            inference = AJAEInference(
                self.model,
                self.development_encoder,
                protocol=self.protocol,
                condition=self.request.condition,
                device=self.device,
            )
            with torch.no_grad():
                evidence = development_fused_ap(
                    inference,
                    self.development.definitions,
                    self.development.rendered_clips,
                    protocol=self.protocol,
                )
        finally:
            self.development_encoder.to("cpu")
            if was_training:
                self.model.train()
        record = _development_record(
            evidence, self.request.condition, self.development
        )
        score = float(record["in_generator_macro_fused_point_ap"])
        improved = (
            self.best_development_ap is None
            or score > self.best_development_ap
        )
        self.last_development = record
        self.last_evaluated_update = self.updates
        if improved:
            self.best_development_ap = score
            # Exact ties retain the earlier checkpoint until R05 freezes tie-breaks.
            _atomic_torch(self.best_path, self._checkpoint())

    def _one_update(self) -> None:
        stop = min(
            self.next_entry + self.config.gradient_accumulation,
            self.request.window_count,
        )
        self.optimizer.zero_grad(set_to_none=True)
        aggregate = EffectiveBatchBCE()
        for position in range(self.next_entry, stop):
            window = self.bank[position]
            logits = predict_window(
                self.model,
                window,
                self.request.condition,
                device=self.device,
            )
            aggregate.add(
                logits,
                window.target.to(self.device),
                window.valid.to(self.device),
            )
        loss = aggregate.loss()
        _finite_tensor(loss.reshape(1), "effective-batch loss")
        loss.backward()
        _finite_gradients(self.model)
        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        self.optimizer.step()
        self.scheduler.step()
        self.last_loss = float(loss.detach().item())
        self.updates += 1
        self.windows_seen += stop - self.next_entry
        self.next_entry = stop
        if self.next_entry == self.request.window_count:
            self.epoch += 1
            self.next_entry = 0

    def fit(self) -> dict[str, object]:
        self.model.train()
        tiny_metrics: _SubsetMetrics | None = None
        tiny_passed = False
        try:
            while self.updates < self.request.maximum_updates:
                self._one_update()
                if self.request.mode is TrainMode.TINY_OVERFIT:
                    tiny_metrics = self._training_subset_metrics()
                    rule = self.protocol.training["tiny_overfit"]["pass_any"]
                    tiny_passed = (
                        math.isfinite(tiny_metrics.average_precision)
                        and 100.0 * tiny_metrics.average_precision
                        >= float(rule["training_AP_minimum_percent"])
                    ) or tiny_metrics.loss < float(rule["loss_strictly_below"])
                evaluate_now = (
                    self.development is not None
                    and self.updates
                    % self.config.evaluation_interval_updates
                    == 0
                )
                if evaluate_now:
                    self._evaluate_development()
                if evaluate_now or self.next_entry == 0 or tiny_passed:
                    self._save_last()
                if tiny_passed:
                    break
        except KeyboardInterrupt:
            self._save_last()
            raise
        if (
            self.development is not None
            and self.last_evaluated_update != self.updates
        ):
            self._evaluate_development()
        self._save_last()
        status = (
            "passed"
            if tiny_passed
            else "failed"
            if self.request.mode is TrainMode.TINY_OVERFIT
            else "completed"
        )
        result: dict[str, object] = {
            "format": TRAIN_RESULT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "mode": self.request.mode.value,
            "condition": self.request.condition.value,
            "grouping_mode": self.request.condition.grouping_mode.value,
            "seed": self.request.seed,
            "protocol_identity": self.protocol_identity,
            "bank_protocol_identity": self.bank.protocol_identity,
            "bank_identity": self.bank.bank_identity,
            "window_count": self.request.window_count,
            "completed_epochs": self.epoch,
            "optimizer_updates": self.updates,
            "windows_seen": self.windows_seen,
            "last_effective_batch_loss": self.last_loss,
            "checkpoint_selection_metric": (
                "in_generator_macro_fused_point_ap"
            ),
            "best_development_in_generator_macro_fused_point_ap": (
                self.best_development_ap
            ),
            "development_population_identity": (
                None if self.development is None else self.development.identity
            ),
            "last_development": self.last_development,
        }
        if tiny_metrics is not None:
            result["tiny_overfit"] = {
                "training_loss": tiny_metrics.loss,
                "training_AP_percent": (
                    100.0 * tiny_metrics.average_precision
                    if math.isfinite(tiny_metrics.average_precision)
                    else None
                ),
                "pass_rule": _plain_json(
                    self.protocol.training["tiny_overfit"]["pass_any"]
                ),
            }
        _atomic_json(self.result_path, result)
        return result


def run_training(
    *,
    protocol_path: Path | str,
    bank_path: Path | str,
    output_directory: Path | str,
    mode: TrainMode | str,
    condition: ExperimentCondition | str,
    seed: int,
    device: torch.device | str,
    config: TrainingConfig,
    development_worlds_path: Path | str | None = None,
    rendered_development_clips: Iterable[DevelopmentClipWorld] | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Validate the request, construct the shared model class, and train it."""

    protocol = _load_training_protocol(protocol_path)
    selected_mode = TrainMode(mode)
    selected_condition = ExperimentCondition(condition)
    # Refuse invalid modes before opening a potentially large frozen bank.
    provisional_bank_size = {
        TrainMode.TINY_OVERFIT: 8,
        TrainMode.PILOT: 128,
        TrainMode.FORMAL: 1,
    }[selected_mode]
    validate_training_request(
        protocol,
        selected_mode,
        selected_condition,
        seed,
        config,
        bank_size=provisional_bank_size,
    )
    development_supplied = (
        development_worlds_path is not None
        and rendered_development_clips is not None
    )
    if (development_worlds_path is None) != (rendered_development_clips is None):
        raise TrainingError(
            "development definitions and rendered clips must be supplied together"
        )
    if selected_mode is TrainMode.TINY_OVERFIT and development_supplied:
        raise TrainingError("tiny_overfit cannot inspect the development set")
    if selected_mode is not TrainMode.TINY_OVERFIT and not development_supplied:
        raise TrainingError(
            "pilot and formal training require frozen development definitions and clips"
        )
    bank = load_window_train_bank(bank_path, protocol=protocol)
    request = validate_training_request(
        protocol,
        selected_mode,
        selected_condition,
        seed,
        config,
        bank_size=len(bank),
    )
    development = (
        None
        if development_worlds_path is None
        else _load_development_set(
            development_worlds_path,
            rendered_development_clips,  # type: ignore[arg-type]
            protocol=protocol,
            renderer_identity=bank.renderer_identity,
        )
    )
    _seed_everything(seed)
    model = JointWindowPointTransformer.from_protocol(protocol)
    trainer = WindowTrainer(
        protocol=protocol,
        bank=bank,
        model=model,
        request=request,
        config=config,
        device=device,
        output_directory=output_directory,
        development=development,
        resume=resume,
    )
    return trainer.fit()

def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train schema-31 AJAE from a frozen five-scan WindowWorld bank."
        )
    )
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "protocol.json"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[item.value for item in TrainMode],
    )
    parser.add_argument(
        "--condition", required=True, choices=["B1", "B2", "B3"]
    )
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument(
        "--gradient-accumulation", required=True, type=int
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--scheduler",
        choices=("constant", "five_percent_warmup_cosine"),
        default="constant",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--maximum-updates", type=int)
    parser.add_argument(
        "--evaluation-interval-updates", type=int, default=1
    )
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        epochs=args.epochs,
        maximum_updates=args.maximum_updates,
        evaluation_interval_updates=args.evaluation_interval_updates,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    result = run_training(
        protocol_path=args.protocol,
        bank_path=args.bank,
        output_directory=args.output,
        mode=args.mode,
        condition=args.condition,
        seed=args.seed,
        device=args.device,
        config=config,
        resume=args.resume,
    )
    print(json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True
    ))


if __name__ == "__main__":
    _main()
