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
# Required by deterministic CUDA matrix multiplication on supported PyTorch builds.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .evaluate import (
        AJAEInference,
        DevelopmentClipResult,
        DevelopmentFusedAP,
        EvaluationError,
        EvaluationIdentity,
        FUSION_SEMANTICS as DEVELOPMENT_FUSION_SEMANTICS,
        FormalGateVerdictRecord,
        development_fused_ap,
        official_metrics,
    )
    from .model import (
        FrozenSTUPointEncoder,
        JointWindowPointTransformer,
        STUPointEncoding,
        model_state_sha256,
        stu_input_identity,
    )
    from .protocol import (
        AJAEProtocol,
        DevelopmentWorlds,
        ExperimentCondition,
        GroupingMode,
        ProtocolError,
        R02_MATCHING_FEATURES,
        R02_SHORTCUT_SEED,
        R02_VALIDATION_KEYS,
        load_development_worlds,
        load_protocol,
        r02_audit_algorithm_identity,
    )
    from .render import (
        DevelopmentClipWorld,
        RenderError,
        WindowEntityDescriptor,
        WorldGenerationReport,
        WorldSpec,
        collect_observed_obstacle_index,
        extract_normal_template_library,
        load_qualified_support_pool,
        load_sensor_calibration,
        match_window_entities,
        render_development_clip_world,
        sample_development_clip_world,
        sample_window_world,
        save_development_worlds,
        source_observation_identity,
        window_matching_balance,
        window_shortcut_audit,
    )
    from .scene import (
        LabelMode,
        STUSequence,
        assemble_window,
        canonical_ray_mapping_digest,
    )
except ImportError:  # Direct execution from src/.
    from evaluate import (
        AJAEInference,
        DevelopmentClipResult,
        DevelopmentFusedAP,
        EvaluationError,
        EvaluationIdentity,
        FUSION_SEMANTICS as DEVELOPMENT_FUSION_SEMANTICS,
        FormalGateVerdictRecord,
        development_fused_ap,
        official_metrics,
    )
    from model import (
        FrozenSTUPointEncoder,
        JointWindowPointTransformer,
        STUPointEncoding,
        model_state_sha256,
        stu_input_identity,
    )
    from protocol import (
        AJAEProtocol,
        DevelopmentWorlds,
        ExperimentCondition,
        GroupingMode,
        ProtocolError,
        R02_MATCHING_FEATURES,
        R02_SHORTCUT_SEED,
        R02_VALIDATION_KEYS,
        load_development_worlds,
        load_protocol,
        r02_audit_algorithm_identity,
    )
    from render import (
        DevelopmentClipWorld,
        RenderError,
        WindowEntityDescriptor,
        WorldGenerationReport,
        WorldSpec,
        collect_observed_obstacle_index,
        extract_normal_template_library,
        load_qualified_support_pool,
        load_sensor_calibration,
        match_window_entities,
        render_development_clip_world,
        sample_development_clip_world,
        sample_window_world,
        save_development_worlds,
        source_observation_identity,
        window_matching_balance,
        window_shortcut_audit,
    )
    from scene import (
        LabelMode,
        STUSequence,
        assemble_window,
        canonical_ray_mapping_digest,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 31
WINDOW_GROUPS = (0, 1, 2, 3, 4)
WINDOW_BANK_FORMAT = "ajae-window-train-bank-v1"
TRAIN_CHECKPOINT_FORMAT = "ajae-schema31-training-checkpoint-v4"
TRAIN_RESULT_FORMAT = "ajae-schema31-training-result-v4"
R04_QUALIFICATION_FORMAT = "ajae-schema31-r04-training-qualification-v2"
DEVELOPMENT_METRIC_EVIDENCE_FORMAT = "ajae-schema31-development-metric-evidence-v1"
MANIFEST_NAME = "manifest.json"
_SHARD_ARRAYS = {
    "coordinates",
    "scan_group",
    "stu_features",
    "normal_evidence",
    "reliability_assign",
    "reliability_noobj",
    "intensity",
    "target",
    "valid",
    "source_frame",
    "source_slot",
    "source_ray",
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
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
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
            _plain_json(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
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
        not isinstance(value, str)
        or len(value) != 64
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
        handle.write(
            json.dumps(
                _plain_json(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
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
        np.random.set_state(
            (
                str(numpy_record["generator"]),
                np.asarray(numpy_record["state"], dtype=np.uint32),
                int(numpy_record["position"]),
                int(numpy_record["has_gauss"]),
                float(numpy_record["cached_gaussian"]),
            )
        )
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
    result = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
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
        if (
            result.is_floating_point()
            or result.is_complex()
            or not bool(((result == 0) | (result == 1)).all())
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
        object.__setattr__(
            self, "target", _cpu_bool_tensor(self.target, "target", count)
        )
        object.__setattr__(self, "valid", _cpu_bool_tensor(self.valid, "valid", count))
        object.__setattr__(
            self,
            "world_identity",
            _require_digest(self.world_identity, "world_identity"),
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
                raise TrainingError(
                    "each scan_group must identify exactly one source frame"
                )
            frames.append(int(member_frames.item()))
        if tuple(frames) != tuple(range(frames[0], frames[0] + 5)):
            raise TrainingError(
                "scan groups must map to five consecutive source frames"
            )
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
            "minimum_range_m_inclusive": evaluation["minimum_range_m_inclusive"],
            "maximum_range_m_inclusive": evaluation["maximum_range_m_inclusive"],
        },
    }
    return _identity_digest(payload)


def protocol_training_system_identity(protocol: object) -> str:
    """Hash fixed R04 mechanics while excluding sequential pilot decisions."""

    if type(protocol) is not AJAEProtocol or protocol.schema_version != SCHEMA_VERSION:
        raise TrainingError("training-system identity requires schema 31")
    document = protocol.plain_document()
    training = _require_mapping(document["training"], "protocol.training")
    pilot = dict(_require_mapping(training["pilot"], "protocol.training.pilot"))
    pilot.pop("frozen_stage_winners")
    formal = _require_mapping(training["formal"], "protocol.training.formal")
    stable_training = {
        name: training[name]
        for name in (
            "source_partition",
            "source_sequence_id",
            "bank",
            "micro_batch",
            "effective_batch",
            "epoch",
            "loss",
            "forced_partial_step_at_world_boundary",
            "modes",
            "tiny_overfit",
            "checkpoint_selection",
            "deterministic_algorithms",
        )
    }
    stable_training["pilot"] = pilot
    stable_training["formal_design"] = {
        name: formal[name]
        for name in (
            "seeds",
            "deployment_seed",
            "deployment_condition",
            "allowed_only_after",
        )
    }
    return _identity_digest(
        {
            "format": "ajae-schema31-training-system-protocol-v1",
            "schema_version": SCHEMA_VERSION,
            "scientific_contract": document["scientific_contract"],
            "data": document["data"],
            "window": document["window"],
            "labels": document["labels"],
            "render": document["render"],
            "stu": document["stu"],
            "model": document["model"],
            "experiments": document["experiments"],
            "training": stable_training,
            "evaluation": document["evaluation"],
        }
    )


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
    return _identity_digest(
        {
            "format": "ajae-window-world-v1",
            "world_identity": world_identity,
            "partition": partition,
            "sequence_id": sequence_id,
            "window_start": frames[0],
            "frame_ids": frames,
            "renderer_identity": renderer_identity,
            "source_observation_identities": observations,
        }
    )


def _validate_window_manifest_record(
    manifest: Mapping[str, object],
    *,
    expected_world_identity: str,
    expected_source_frames: Sequence[int],
    expected_source_observation_identities: Sequence[str],
    expected_target_count: int,
) -> str:
    """Rebuild renderer objects and identities from one retained scientific record."""

    _require_exact_keys(
        manifest,
        {
            "format",
            "identity",
            "world_identity",
            "partition",
            "sequence_id",
            "window_start",
            "frame_ids",
            "renderer_identity",
            "world",
            "report",
            "source_observation_identities",
            "descriptors",
        },
        "window_manifest",
    )
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
    sequence_id = _require_int(manifest["sequence_id"], "window_manifest.sequence_id")
    start = _require_int(manifest["window_start"], "window_manifest.window_start")
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

    world_record = _require_mapping(manifest["world"], "window_manifest.world")
    report_record = _require_mapping(manifest["report"], "window_manifest.report")
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
    if sequence_id == 206 and any(
        item.shape.to_dict().get("kind") == "held-out-torus-sdf"
        for item in world.objects
    ):
        raise TrainingError(
            "held-out torus geometry cannot enter the train/206 window bank"
        )
    placements = {item.object_id: item.label for item in report.placements}
    objects = {item.object_id: item.label for item in world.objects}
    if len(report.placements) != len(placements) or placements != objects:
        raise TrainingError("generation report does not cover every WorldSpec object")

    raw_descriptors = manifest["descriptors"]
    if not isinstance(raw_descriptors, list):
        raise TrainingError("window_manifest.descriptors must be an array")
    descriptor_keys = {
        "object_id",
        "label",
        "visible_returns_by_scan",
        "spatial_voxels_by_scan",
        "joint_visible_return_count",
        "joint_spatial_voxel_count",
        "maximum_single_scan_spatial_voxel_count",
        "densification_gain",
        "duplicate_fraction",
        "median_distance_m",
        "occlusion_rate",
        "support_semantic_id",
        "visible_scan_count",
        "minimum_visible_return_height_m",
        "intensity_q05_median_q95",
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
            descriptor_value = WindowEntityDescriptor(
                **json.loads(_canonical_json(descriptor))
            )
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
    declared = _require_digest(manifest["identity"], "window_manifest.identity")
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
    return _tensor_identity_digest(
        (
            ("scan_group", data.scan_group),
            ("source_frame", data.source_frame),
            ("source_slot", data.source_slot),
            ("source_ray", data.source_ray),
        )
    )


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
        or tuple(getattr(item, "source").frame_id for item in rendered) != frame_ids
    ):
        raise TrainingError("window_world must contain five ordered rendered frames")
    if set(stu_by_frame) != set(frame_ids):
        raise TrainingError("STU outputs must identify exactly the five source frames")
    training = _require_mapping(
        getattr(protocol, "training", None), "protocol.training"
    )
    if training["source_partition"] != "train" or training["source_sequence_id"] != 206:
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
            raise TrainingError("STU evidence must be a FrozenSTUPointEncoder output")
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
        if not all(
            isinstance(value, Tensor)
            for value in (point_features, evidence, assignment, no_object)
        ):
            raise TrainingError("STU evidence is missing a required frozen tensor")
        if any(
            value.requires_grad
            for value in (point_features, evidence, assignment, no_object)
        ):
            raise TrainingError("STU evidence must be frozen before bank construction")
        stu_features.append(point_features.detach().cpu())
        normal_evidence.append(evidence.detach().cpu())
        reliability_assign.append(assignment.detach().cpu())
        reliability_noobj.append(no_object.detach().cpu())
        xyzi = source.xyzi[source.real_slots]
        intensity.append(xyzi[:, 3].astype(np.float32, copy=False))
        ranges.append(
            np.linalg.norm(xyzi[:, :3].astype(np.float32, copy=False), axis=1)
        )

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


_BANK_PLAN_KEYS = {
    "position",
    "source_frames",
    "world_type",
    "root_seed",
    "observation_attempt_selection",
    "observation_attempt_stride",
    "maximum_observation_attempts",
}
_FORMAL_BANK_PREFIX_COUNTS = (8, 128, 445)


def _protocol_training_bank_plan(
    protocol: object,
) -> tuple[tuple[Mapping[str, object], ...], str]:
    """Validate the protocol-owned formal bank plan before consuming an artifact."""

    plan_method = getattr(protocol, "training_bank_plan", None)
    if not callable(plan_method):
        raise TrainingError("protocol does not expose its training bank plan")
    raw_plan = plan_method()
    if not isinstance(raw_plan, tuple) or len(raw_plan) != 445:
        raise TrainingError("formal training bank plan must contain 445 entries")
    plan: list[Mapping[str, object]] = []
    for position, raw_entry in enumerate(raw_plan):
        entry = _require_mapping(raw_entry, f"training bank plan[{position}]")
        _require_exact_keys(entry, _BANK_PLAN_KEYS, f"training bank plan[{position}]")
        if (
            _require_int(entry["position"], f"training bank plan[{position}].position")
            != position
        ):
            raise TrainingError("training bank plan positions must be contiguous")
        source_frames = entry["source_frames"]
        if not isinstance(source_frames, tuple) or len(source_frames) != 5:
            raise TrainingError("training bank plan source_frames must be a five-tuple")
        frames = tuple(
            _require_int(value, f"training bank plan[{position}] source frame")
            for value in source_frames
        )
        if frames != tuple(range(frames[0], frames[0] + 5)):
            raise TrainingError("training bank plan frames must be consecutive")
        if entry["world_type"] not in {
            "pure_normal",
            "control_only",
            "mixed",
            "anomaly_only",
        }:
            raise TrainingError("training bank plan world_type is invalid")
        _require_int(entry["root_seed"], "training bank plan root_seed")
        if (
            entry["observation_attempt_selection"]
            != "first_success_in_ascending_retry_index_no_manual_selection"
        ):
            raise TrainingError("training bank retry selection rule changed")
        _require_int(
            entry["observation_attempt_stride"],
            "training bank plan observation stride",
            minimum=1,
        )
        _require_int(
            entry["maximum_observation_attempts"],
            "training bank plan maximum attempts",
            minimum=1,
        )
        plan.append(entry)
    identity = _require_digest(
        getattr(protocol, "training_bank_plan_identity", None),
        "protocol training bank plan identity",
    )
    return tuple(plan), identity


def _bank_window_world(manifest: Mapping[str, object]) -> WorldSpec:
    """Recover the WorldSpec fields needed to enforce one bank-plan row."""

    try:
        world = WorldSpec.from_dict(
            _require_mapping(manifest.get("world"), "window_manifest.world")
        )
    except (TypeError, ValueError, KeyError, RenderError) as error:
        raise TrainingError("window_manifest contains an invalid WorldSpec") from error
    if world.source_sequence_id == 206 and any(
        item.shape.to_dict().get("kind") == "held-out-torus-sdf"
        for item in world.objects
    ):
        raise TrainingError(
            "held-out torus geometry cannot enter the train/206 window bank"
        )
    return world


def _validate_formal_bank_plan_row(
    position: int,
    source_frames: Sequence[int],
    world: WorldSpec,
    plan: Sequence[Mapping[str, object]],
) -> None:
    """Require a bank entry to be exactly the corresponding frozen plan row."""

    if position >= len(plan):
        raise TrainingError("window training bank exceeds its formal plan")
    expected = plan[position]
    if (
        expected["position"] != position
        or tuple(expected["source_frames"]) != tuple(source_frames)
        or expected["world_type"] != world.world_type
    ):
        raise TrainingError(
            f"bank entry {position} differs from its frozen generation plan"
        )
    root_seed = int(expected["root_seed"])
    stride = int(expected["observation_attempt_stride"])
    attempts = int(expected["maximum_observation_attempts"])
    delta = world.seed - root_seed
    if delta < 0 or delta % stride or delta // stride >= attempts:
        raise TrainingError(
            f"bank entry {position} WorldSpec seed is outside its retry stream"
        )
    if world.world_type == "pure_normal" and delta != 0:
        raise TrainingError(
            f"bank entry {position} skipped the necessarily successful root seed"
        )


def _test_bank_plan_identity(entries: Sequence[Mapping[str, object]]) -> str:
    """Content-address the explicit private-test plan without using formal identity."""

    plan_entries: list[dict[str, object]] = []
    for position, entry in enumerate(entries):
        manifest = _require_mapping(
            entry["window_manifest"], f"entries[{position}].window_manifest"
        )
        world = _bank_window_world(manifest)
        plan_entries.append(
            {
                "position": position,
                "source_frames": list(entry["source_frames"]),
                "world_type": world.world_type,
                "world_seed": world.seed,
            }
        )
    return _identity_digest(
        {
            "format": "ajae-schema31-test-window-bank-plan-v1",
            "entries": plan_entries,
        }
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
            raise TrainingError(
                "window training bank manifest is unreadable"
            ) from error
        root = _require_mapping(document, "window training bank manifest")
        _require_exact_keys(
            root,
            {
                "format",
                "schema_version",
                "name",
                "protocol_identity",
                "bank_identity",
                "source_partition",
                "source_sequence_id",
                "renderer_identity",
                "stu_identity",
                "shared_by",
                "entry_count",
                "plan_identity",
                "plan_prefix_count",
                "test_fixture",
                "entries",
            },
            "window training bank manifest",
        )
        if (
            root["format"] != WINDOW_BANK_FORMAT
            or root["schema_version"] != SCHEMA_VERSION
        ):
            raise TrainingError("window training bank is not schema 31")
        if type(root["test_fixture"]) is not bool:
            raise TrainingError("window training bank test_fixture must be boolean")
        test_fixture = bool(root["test_fixture"])
        plan_identity = _require_digest(root["plan_identity"], "plan_identity")
        plan_prefix_count = _require_int(
            root["plan_prefix_count"], "plan_prefix_count", minimum=1
        )
        training = _require_mapping(
            getattr(protocol, "training", None), "protocol.training"
        )
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
        if (
            root["source_partition"] != partition
            or root["source_sequence_id"] != sequence_id
        ):
            raise TrainingError("window training bank source differs from the protocol")
        renderer_identity = _require_digest(
            root["renderer_identity"], "renderer_identity"
        )
        if not test_fixture and renderer_identity != getattr(
            protocol, "renderer_identity", None
        ):
            raise TrainingError("window training bank uses a different renderer")
        stu_identity = _require_digest(root["stu_identity"], "stu_identity")
        protocol_stu = _require_mapping(getattr(protocol, "stu", None), "protocol.stu")
        if stu_identity != protocol_stu["checkpoint_sha256"]:
            raise TrainingError("window training bank uses a different frozen STU")
        if root["shared_by"] != ["B1", "B2", "B3"]:
            raise TrainingError("one bank must be shared by B1, B2, and B3")
        raw_entries = root["entries"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise TrainingError(
                "window training bank entries must be a non-empty array"
            )
        entry_count = _require_int(root["entry_count"], "entry_count", minimum=1)
        if entry_count != len(raw_entries):
            raise TrainingError("window training bank entry_count is inconsistent")
        if plan_prefix_count != len(raw_entries):
            raise TrainingError("bank plan_prefix_count differs from entry_count")
        formal_plan: tuple[Mapping[str, object], ...] = ()
        if not test_fixture:
            formal_plan, expected_plan_identity = _protocol_training_bank_plan(protocol)
            if plan_identity != expected_plan_identity:
                raise TrainingError("window training bank plan identity changed")
            if plan_prefix_count not in _FORMAL_BANK_PREFIX_COUNTS:
                raise TrainingError(
                    "formal bank must contain a fixed 8, 128, or 445-entry prefix"
                )

        bank_root = self.manifest_path.parent.resolve()
        entries: list[WindowBankEntry] = []
        used_shards: set[Path] = set()
        used_windows: set[str] = set()
        used_worlds: set[str] = set()
        for position, raw_entry in enumerate(raw_entries):
            entry = _require_mapping(raw_entry, f"entries[{position}]")
            _require_exact_keys(
                entry,
                {
                    "position",
                    "shard",
                    "shard_sha256",
                    "point_count",
                    "world_identity",
                    "window_identity",
                    "source_frames",
                    "source_observation_identities",
                    "stu_input_identities",
                    "target_count",
                    "valid_count",
                    "anomaly_count",
                    "normal_count",
                    "ignored_count",
                    "point_identity_sha256",
                    "labels_sha256",
                    "window_manifest",
                },
                f"entries[{position}]",
            )
            if entry["position"] != position:
                raise TrainingError("window training bank positions must be contiguous")
            raw_shard = entry["shard"]
            if (
                not isinstance(raw_shard, str)
                or not raw_shard
                or Path(raw_shard).is_absolute()
            ):
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
            window_identity = _require_digest(
                entry["window_identity"], "window_identity"
            )
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
            world = _bank_window_world(window_manifest)
            if not test_fixture:
                _validate_formal_bank_plan_row(position, frames, world, formal_plan)
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
                _require_digest(value, "STU input identity") for value in raw_stu_inputs
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
            if (
                not callable(window_frames)
                or tuple(window_frames(partition, sequence_id, frames[0])) != frames
            ):
                raise TrainingError("bank entry is not a legal protocol window")
            point_count = _require_int(entry["point_count"], "point_count", minimum=1)
            target_count = _require_int(entry["target_count"], "target_count")
            valid_count = _require_int(entry["valid_count"], "valid_count", minimum=1)
            anomaly_count = _require_int(entry["anomaly_count"], "anomaly_count")
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
            entries.append(
                WindowBankEntry(
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
                )
            )
        if test_fixture and plan_identity != _test_bank_plan_identity(raw_entries):
            raise TrainingError("test-fixture bank plan identity changed")
        self.protocol_identity = expected_protocol
        self.bank_identity = declared_bank_identity
        self.plan_identity = plan_identity
        self.plan_prefix_count = plan_prefix_count
        self.test_fixture = test_fixture
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
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        entry = self.entries[index]
        fingerprint = _file_fingerprint(entry.shard)
        if self._verified_shards.get(entry.shard) != fingerprint:
            if _sha256_file(entry.shard) != entry.shard_sha256:
                raise TrainingError(f"training shard hash changed: {entry.shard.name}")
            if _file_fingerprint(entry.shard) != fingerprint:
                raise TrainingError(
                    f"training shard changed while hashing: {entry.shard.name}"
                )
            self._verified_shards[entry.shard] = fingerprint
        try:
            with np.load(entry.shard, allow_pickle=False) as archive:
                if set(archive.files) != _SHARD_ARRAYS:
                    raise TrainingError(
                        f"training shard arrays differ: {entry.shard.name}"
                    )
                arrays = {
                    name: np.array(archive[name], copy=True) for name in _SHARD_ARRAYS
                }
        except (OSError, ValueError) as error:
            raise TrainingError(
                f"cannot read training shard: {entry.shard.name}"
            ) from error
        if _file_fingerprint(entry.shard) != fingerprint:
            raise TrainingError(
                f"training shard changed while reading: {entry.shard.name}"
            )

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
            raise TrainingError(
                "training shard content disagrees with its manifest entry"
            )
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


def _write_window_train_bank(
    destination: Path | str,
    windows: Iterable[WindowTrainingData],
    *,
    protocol: object,
    renderer_identity: str,
    stu_identity: str | None = None,
    test_fixture: bool,
) -> WindowTrainingBank:
    """Write either the protocol plan or an explicitly marked private fixture."""

    final = Path(destination).expanduser().resolve()
    if final.exists():
        raise TrainingError("refusing to replace an existing frozen training bank")
    final.parent.mkdir(parents=True, exist_ok=True)
    renderer = _require_digest(renderer_identity, "renderer_identity")
    if not test_fixture and renderer != getattr(protocol, "renderer_identity", None):
        raise TrainingError("writer renderer identity differs from the protocol")
    training = _require_mapping(
        getattr(protocol, "training", None), "protocol.training"
    )
    bank_spec = _require_mapping(training["bank"], "protocol.training.bank")
    formal_plan: tuple[Mapping[str, object], ...] = ()
    formal_plan_identity: str | None = None
    if not test_fixture:
        formal_plan, formal_plan_identity = _protocol_training_bank_plan(protocol)
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
                raise TrainingError(
                    "bank writer accepts only WindowTrainingData entries"
                )
            source_frames = window.five_source_frames
            expected_window = _window_identity(
                world_identity=window.world_identity,
                source_frames=source_frames,
                source_observation_identities=(window.source_observation_identities),
                renderer_identity=renderer,
                partition=partition,
                sequence_id=sequence_id,
            )
            if window.window_identity and window.window_identity != expected_window:
                raise TrainingError(
                    "WindowTrainingData.window_identity is inconsistent"
                )
            if window.window_identity != expected_window:
                raise TrainingError(
                    "WindowTrainingData manifest uses another renderer or source"
                )
            window_manifest = _normalized_window_manifest(window.window_manifest)
            if (
                window_manifest["partition"] != partition
                or window_manifest["sequence_id"] != sequence_id
                or window_manifest["renderer_identity"] != renderer
            ):
                raise TrainingError(
                    "WindowTrainingData manifest differs from the bank identity"
                )
            world = _bank_window_world(window_manifest)
            if not test_fixture:
                _validate_formal_bank_plan_row(
                    position, source_frames, world, formal_plan
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
            entries.append(
                {
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
                }
            )
        if not entries:
            raise TrainingError("cannot write an empty window training bank")
        if not test_fixture and len(entries) not in _FORMAL_BANK_PREFIX_COUNTS:
            raise TrainingError(
                "formal bank must contain a fixed 8, 128, or 445-entry prefix"
            )
        plan_identity = (
            _test_bank_plan_identity(entries) if test_fixture else formal_plan_identity
        )
        if plan_identity is None:
            raise AssertionError("formal bank plan identity was not resolved")
        manifest: dict[str, object] = {
            "format": WINDOW_BANK_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "name": bank_spec["name"],
            "protocol_identity": protocol_bank_identity(protocol),
            "plan_identity": plan_identity,
            "plan_prefix_count": len(entries),
            "test_fixture": test_fixture,
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
            handle.write(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
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


def _write_window_train_bank_for_test(
    destination: Path | str,
    windows: Iterable[WindowTrainingData],
    *,
    protocol: object,
    renderer_identity: str,
    stu_identity: str | None = None,
) -> WindowTrainingBank:
    """Write a visibly marked arbitrary-size bank for deterministic tests only."""

    return _write_window_train_bank(
        destination,
        windows,
        protocol=protocol,
        renderer_identity=renderer_identity,
        stu_identity=stu_identity,
        test_fixture=True,
    )


@dataclass(slots=True)
class EffectiveBatchBCE:
    """Retain class sums until every WindowWorld in one optimizer step is known."""

    positive_loss_sum: Tensor | None = None
    negative_loss_sum: Tensor | None = None
    positive_count: int = 0
    negative_count: int = 0

    def add(self, logits: Tensor, target: Tensor, valid: Tensor) -> None:
        if (
            logits.ndim != 1
            or target.shape != logits.shape
            or valid.shape != logits.shape
        ):
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
                value
                if self.positive_loss_sum is None
                else self.positive_loss_sum + value
            )
            self.positive_count += positive_count
        if negative_count:
            value = raw[negative].sum()
            self.negative_loss_sum = (
                value
                if self.negative_loss_sum is None
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
        raise TrainingError(
            "model must return one anomaly logit for every selected point"
        )
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
            rows = torch.nonzero(batch.scan_group == group, as_tuple=False).squeeze(1)
            row_parts.append(rows)
            logit_parts.append(
                _forward_rows(
                    model,
                    batch,
                    rows,
                    grouping_mode=GroupingMode.SINGLE,
                    erase_group_identity=True,
                )
            )
        joined_rows = torch.cat(row_parts)
        order = torch.argsort(joined_rows)
        expected_rows = torch.arange(window.point_count, device=resolved)
        if not torch.equal(joined_rows[order], expected_rows):
            raise TrainingError(
                "B1 scan forwards did not cover every original row once"
            )
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
    ends = torch.nonzero(torch.cat((score[1:] != score[:-1], final))).squeeze(1)
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
        _require_int(self.gradient_accumulation, "gradient_accumulation", minimum=1)
        if _require_number(self.weight_decay, "weight_decay") < 0.0:
            raise TrainingError("weight_decay cannot be negative")
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
        if (
            self.gradient_clip_norm is not None
            and _require_number(self.gradient_clip_norm, "gradient_clip_norm") <= 0.0
        ):
            raise TrainingError("gradient_clip_norm must be positive")


def _validate_result_request(
    protocol: AJAEProtocol,
    *,
    mode: TrainMode,
    condition: ExperimentCondition,
    seed: int,
    config: TrainingConfig,
    pilot_stage: object,
    authorization_node: object,
    window_count: object,
    maximum_updates: int,
    r04_identity: object,
    r05_identity: object,
    g2_identity: object,
) -> None:
    """Validate a completed run without depending on mutable R04 stage winners."""

    training = protocol.training
    if mode is TrainMode.TINY_OVERFIT:
        tiny = _require_mapping(training["tiny_overfit"], "training.tiny_overfit")
        if (
            authorization_node != "R04"
            or pilot_stage is not None
            or condition
            not in {
                ExperimentCondition.B1,
                ExperimentCondition.B2,
                ExperimentCondition.B3,
            }
            or seed != tiny["seed"]
            or dict(asdict(config)) != _plain_json(tiny["config"])
            or window_count != tiny["windows"]
            or maximum_updates != tiny["maximum_updates"]
            or r04_identity is not None
            or r05_identity is not None
            or g2_identity is not None
        ):
            raise TrainingError("tiny-overfit result differs from its frozen request")
        return

    if mode is TrainMode.PILOT:
        pilot = _require_mapping(training["pilot"], "training.pilot")
        fixed = _require_mapping(
            pilot["fixed_run_parameters"], "training.pilot.fixed_run_parameters"
        )
        if (
            authorization_node != "R04"
            or condition.value != pilot["condition"]
            or seed not in tuple(pilot["seeds"])
            or pilot_stage not in tuple(pilot["stage_order"])
            or window_count != pilot["windows"]
            or maximum_updates != config.maximum_updates
            or maximum_updates not in tuple(pilot["screen_updates"])
            or config.learning_rate not in tuple(pilot["learning_rates"])
            or config.gradient_accumulation not in tuple(pilot["gradient_accumulation"])
            or config.scheduler
            not in tuple(pilot["schedulers_after_learning_rate_selection"])
            or config.weight_decay
            not in tuple(pilot["weight_decay_after_scheduler_selection"])
            or any(
                getattr(config, name) != fixed[name]
                for name in (
                    "epochs",
                    "evaluation_interval_updates",
                    "gradient_clip_norm",
                )
            )
            or r04_identity is not None
            or r05_identity is not None
            or g2_identity is not None
        ):
            raise TrainingError("pilot result differs from the frozen generic plan")
        if pilot_stage == "learning_rate_and_batch":
            expected_seed = 1002 if maximum_updates == 600 else 1001
            if (
                seed != expected_seed
                or config.scheduler != "constant"
                or config.weight_decay != 0.0
            ):
                raise TrainingError("pilot learning-rate/batch result is mis-staged")
        elif pilot_stage == "scheduler":
            if seed != 1002 or maximum_updates != 600 or config.weight_decay != 0.0:
                raise TrainingError("pilot scheduler result is mis-staged")
        elif seed != 1002 or maximum_updates != 600:
            raise TrainingError("pilot weight-decay result is mis-staged")
        return

    formal = _require_mapping(training["formal"], "training.formal")
    recipe = _require_mapping(formal["recipe"], "training.formal.recipe")
    expected_updates = (
        int(config.maximum_updates)
        if config.maximum_updates is not None
        else int(config.epochs) * math.ceil(445 / int(config.gradient_accumulation))
    )
    expected_node = "G2" if condition is ExperimentCondition.B1 else "G3"
    if (
        condition
        not in {
            ExperimentCondition.B1,
            ExperimentCondition.B2,
            ExperimentCondition.B3,
        }
        or authorization_node != expected_node
        or pilot_stage is not None
        or seed not in tuple(formal["seeds"])
        or dict(asdict(config)) != _plain_json(recipe)
        or window_count != 445
        or maximum_updates != expected_updates
        or r04_identity != protocol.status["r04_training_qualification_identity"]
        or r05_identity != protocol.status["r05_freeze_identity"]
        or (
            g2_identity
            != (
                None
                if condition is ExperimentCondition.B1
                else protocol.status["g2_verdict_identity"]
            )
        )
    ):
        raise TrainingError("formal result differs from the R05-authorized request")


def _training_progress(
    *, window_count: int, accumulation: int, updates: int
) -> tuple[int, int, int]:
    """Return completed epochs, next entry, and exact windows consumed."""

    steps_per_epoch = math.ceil(window_count / accumulation)
    completed_epochs, residual_steps = divmod(updates, steps_per_epoch)
    residual_windows = min(residual_steps * accumulation, window_count)
    return (
        completed_epochs,
        residual_windows,
        completed_epochs * window_count + residual_windows,
    )


def _development_metric_arrays(
    evidence: DevelopmentFusedAP, development_identity: str
) -> dict[str, np.ndarray]:
    samples = evidence.raw_clip_samples
    if samples is None or tuple(samples) != tuple(
        item.clip_identity for item in evidence.clips
    ):
        raise TrainingError("development evaluation lacks raw point evidence")
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    offsets = [0]
    for identity in samples:
        label, score = samples[identity]
        label_array = np.asarray(label, dtype=np.bool_).reshape(-1)
        score_array = np.asarray(score, dtype=np.float64).reshape(-1)
        if label_array.shape != score_array.shape or label_array.size == 0:
            raise TrainingError("development raw labels and scores are invalid")
        labels.append(label_array)
        scores.append(score_array)
        offsets.append(offsets[-1] + label_array.size)
    return {
        "format": np.asarray(DEVELOPMENT_METRIC_EVIDENCE_FORMAT),
        "evaluation_identity": np.asarray(
            json.dumps(
                evidence.evaluation_identity.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "population_identity": np.asarray(development_identity),
        "clip_identities": np.asarray(tuple(samples)),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "labels": np.concatenate(labels),
        "scores": np.concatenate(scores),
    }


def _array_bundle_identity(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _save_development_metric_evidence(
    directory: Path,
    evidence: DevelopmentFusedAP,
    development_identity: str,
) -> dict[str, str]:
    arrays = _development_metric_arrays(evidence, development_identity)
    identity = _array_bundle_identity(arrays)
    path = directory / f"development-evidence-{identity}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as source:
                existing = {
                    name: np.asarray(source[name]).copy() for name in source.files
                }
        except (OSError, ValueError, TypeError) as error:
            raise TrainingError(
                "existing development metric evidence is unreadable"
            ) from error
        if set(existing) != set(arrays) or _array_bundle_identity(existing) != identity:
            raise TrainingError(
                "existing development metric evidence differs from its name"
            )
    else:
        temporary = path.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
    return {"path": str(path.resolve()), "file_sha256": _sha256_file(path)}


def _load_development_metric_evidence(
    value: object,
    *,
    result_directory: Path,
    evaluation_identity: EvaluationIdentity,
    development_identity: str,
    clip_identities: tuple[str, ...],
) -> dict[str, float]:
    reference = _require_mapping(value, "development metric evidence")
    _require_exact_keys(
        reference, {"path", "file_sha256"}, "development metric evidence"
    )
    raw_path = reference["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise TrainingError("development metric evidence path must be non-empty")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = result_directory / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise TrainingError("development metric evidence is unavailable") from error
    if _sha256_file(resolved) != _require_digest(
        reference["file_sha256"], "development metric evidence file identity"
    ):
        raise TrainingError("development metric evidence file identity changed")
    try:
        with np.load(resolved, allow_pickle=False) as source:
            expected = {
                "format",
                "evaluation_identity",
                "population_identity",
                "clip_identities",
                "offsets",
                "labels",
                "scores",
            }
            if set(source.files) != expected:
                raise TrainingError("development metric evidence schema changed")
            arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    except (OSError, ValueError, TypeError) as error:
        raise TrainingError("development metric evidence is unreadable") from error
    try:
        raw_identity = json.loads(str(arrays["evaluation_identity"].item()))
        stored_identity = EvaluationIdentity(**raw_identity)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise TrainingError("development evidence identity is invalid") from error
    identities = arrays["clip_identities"]
    offsets = arrays["offsets"]
    labels = arrays["labels"]
    scores = arrays["scores"]
    if (
        arrays["format"].shape != ()
        or str(arrays["format"].item()) != DEVELOPMENT_METRIC_EVIDENCE_FORMAT
        or arrays["population_identity"].shape != ()
        or str(arrays["population_identity"].item()) != development_identity
        or stored_identity != evaluation_identity
        or identities.dtype.kind != "U"
        or identities.tolist() != list(clip_identities)
        or offsets.dtype != np.dtype(np.int64)
        or offsets.shape != (len(clip_identities) + 1,)
        or offsets[0] != 0
        or offsets[-1] != labels.size
        or np.any(np.diff(offsets) <= 0)
        or labels.dtype != np.dtype(np.bool_)
        or labels.ndim != 1
        or scores.dtype != np.dtype(np.float64)
        or scores.shape != labels.shape
        or not np.isfinite(scores).all()
    ):
        raise TrainingError("development metric evidence arrays are invalid")
    return {
        identity: official_metrics(
            labels[int(offsets[index]) : int(offsets[index + 1])],
            scores[int(offsets[index]) : int(offsets[index + 1])],
        )["AP"]
        / 100.0
        for index, identity in enumerate(clip_identities)
    }


def _validate_development_summary(
    value: object,
    *,
    protocol: AJAEProtocol,
    condition: ExperimentCondition,
    protocol_identity: str,
    development_identity: str,
    development: DevelopmentWorlds,
    result_directory: Path,
    model_state_identity: str,
    name: str,
) -> tuple[float, float]:
    """Recompute the declared development macro AP and saturation summary."""

    record = _require_mapping(value, name)
    _require_exact_keys(
        record,
        {
            "fusion_semantics",
            "selection_metric",
            "macro_fused_point_ap",
            "probability_saturation_fraction",
            "scope",
            "development_population_identity",
            "evaluation_identity",
            "clips",
            "metric_evidence",
        },
        name,
    )
    if (
        record["fusion_semantics"] != DEVELOPMENT_FUSION_SEMANTICS
        or record["selection_metric"] != "macro_fused_point_ap"
        or record["scope"]
        != "24_in_generator_clips_only_held_out_torus_unopened_until_S01"
        or record["development_population_identity"] != development_identity
    ):
        raise TrainingError(f"{name} changed the frozen development semantics")
    raw_identity = _require_mapping(
        record["evaluation_identity"], f"{name}.evaluation_identity"
    )
    try:
        identity = EvaluationIdentity(**dict(raw_identity))
    except (TypeError, ValueError) as error:
        raise TrainingError(f"{name} has an invalid evaluation identity") from error
    if (
        identity.condition != condition.value
        or identity.protocol_identity != protocol_identity
        or identity.model_class != "JointWindowPointTransformer"
        or identity.model_state_sha256 != model_state_identity
        or identity.stu_class != "FrozenSTUPointEncoder"
        or identity.stu_checkpoint_sha256 != protocol.stu["checkpoint_sha256"]
        or identity.stu_model_state_sha256 != protocol.stu["model_state_tensor_sha256"]
        or identity.stu_source_manifest_sha256 != protocol.stu["source_manifest_sha256"]
        or identity.calibration_sha256 != protocol.render["calibration_sha256"]
        or identity.ray_mapping_sha256
        != protocol.render["ray_grid"]["canonical_sha256"]
        or identity.test_fixture
    ):
        raise TrainingError(f"{name} is not bound to its checkpoint model")
    raw_clips = record["clips"]
    definitions = tuple(development.clips)
    if (
        not isinstance(raw_clips, list)
        or len(raw_clips) != 24
        or len(definitions) != 24
    ):
        raise TrainingError(f"{name} must contain exactly 24 clip results")
    clip_ids: set[str] = set()
    world_ids: set[str] = set()
    scores: list[float] = []
    saturated = 0
    points = 0
    for index, (raw_clip, definition) in enumerate(
        zip(raw_clips, definitions, strict=True)
    ):
        clip = _require_mapping(raw_clip, f"{name}.clips[{index}]")
        _require_exact_keys(
            clip,
            {
                "clip_identity",
                "world_identity",
                "source_observation_identities",
                "mechanism",
                "fused_point_ap",
                "unique_point_count",
                "occurrence_count",
                "occurrence_histogram",
                "saturated_probability_count",
            },
            f"{name}.clips[{index}]",
        )
        clip_id = _require_digest(clip["clip_identity"], "development clip identity")
        world_id = _require_digest(clip["world_identity"], "development world identity")
        sources = clip["source_observation_identities"]
        if (
            clip["mechanism"] != "in_generator"
            or not isinstance(sources, list)
            or len(sources) != 9
        ):
            raise TrainingError(f"{name} clip population changed")
        for source in sources:
            _require_digest(source, "development source observation identity")
        if (
            clip_id != definition.identity
            or world_id != definition.world_identity
            or tuple(sources) != tuple(definition.source_observation_identities)
            or clip["mechanism"] != definition.mechanism
        ):
            raise TrainingError(f"{name} differs from the frozen development manifest")
        score = _require_number(clip["fused_point_ap"], "development clip AP")
        unique = _require_int(
            clip["unique_point_count"], "development unique points", minimum=1
        )
        occurrences = _require_int(
            clip["occurrence_count"], "development occurrence count", minimum=1
        )
        saturated_count = _require_int(
            clip["saturated_probability_count"],
            "development saturated probability count",
            minimum=0,
        )
        histogram = _require_mapping(
            clip["occurrence_histogram"], "development occurrence histogram"
        )
        if (
            not 0.0 <= score <= 1.0
            or clip_id in clip_ids
            or world_id in world_ids
            or not unique <= occurrences <= 5 * unique
            or saturated_count > unique
            or set(histogram) != {str(count) for count in range(1, 6)}
        ):
            raise TrainingError(f"{name} clip statistics are invalid")
        counts = {
            count: _require_int(
                histogram[str(count)], "development occurrence stratum", minimum=0
            )
            for count in range(1, 6)
        }
        if (
            sum(counts.values()) != unique
            or sum(count * value for count, value in counts.items()) != occurrences
        ):
            raise TrainingError(f"{name} occurrence histogram is inconsistent")
        clip_ids.add(clip_id)
        world_ids.add(world_id)
        scores.append(score)
        saturated += saturated_count
        points += unique
    macro = math.fsum(scores) / len(scores)
    recomputed = _load_development_metric_evidence(
        record["metric_evidence"],
        result_directory=result_directory,
        evaluation_identity=identity,
        development_identity=development_identity,
        clip_identities=tuple(item.identity for item in definitions),
    )
    observed = {
        str(clip["clip_identity"]): float(clip["fused_point_ap"])
        for clip in (_require_mapping(item, "development clip") for item in raw_clips)
    }
    if recomputed != observed:
        raise TrainingError(f"{name} clip AP does not reproduce from raw evidence")
    saturation = saturated / points
    if not math.isclose(
        _require_number(record["macro_fused_point_ap"], "development macro AP"),
        macro,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ) or not math.isclose(
        _require_number(
            record["probability_saturation_fraction"],
            "development saturation fraction",
        ),
        saturation,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise TrainingError(f"{name} aggregate statistics cannot be reproduced")
    return macro, saturation


@dataclass(frozen=True, slots=True)
class TrainingRunRecord:
    """Content-addressed result whose declared checkpoints are independently checked."""

    path: Path
    record_sha256: str
    payload: Mapping[str, object]

    @property
    def mode(self) -> TrainMode:
        return TrainMode(str(self.payload["mode"]))

    @property
    def condition(self) -> ExperimentCondition:
        return ExperimentCondition(str(self.payload["condition"]))

    @property
    def seed(self) -> int:
        return int(self.payload["seed"])

    @property
    def bank_identity(self) -> str:
        return str(self.payload["bank_identity"])

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        protocol: AJAEProtocol,
    ) -> "TrainingRunRecord":
        requested = Path(path).expanduser()
        result_path = requested / "result.json" if requested.is_dir() else requested
        try:
            resolved = result_path.resolve(strict=True)
            raw = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingError("training result is unreadable") from error
        payload = _require_mapping(raw, "training result")
        base_keys = {
            "format",
            "schema_version",
            "status",
            "mode",
            "pilot_stage",
            "condition",
            "grouping_mode",
            "seed",
            "authorization_node",
            "r03_qualification_identity",
            "r04_training_qualification_identity",
            "r05_freeze_identity",
            "g2_verdict_identity",
            "config",
            "protocol_identity",
            "training_system_identity",
            "bank_protocol_identity",
            "bank_identity",
            "bank_entry_count",
            "bank_plan_prefix_count",
            "initial_model_state_sha256",
            "window_count",
            "maximum_updates",
            "completed_epochs",
            "optimizer_updates",
            "windows_seen",
            "last_effective_batch_loss",
            "last_checkpoint_sha256",
            "last_model_state_sha256",
            "checkpoint_selection_metric",
            "best_development_macro_fused_point_ap",
            "best_checkpoint_sha256",
            "best_model_state_sha256",
            "best_evaluated_update",
            "best_development",
            "development_trace",
            "development_population_identity",
            "development_manifest",
            "last_development",
            "record_sha256",
        }
        try:
            mode = TrainMode(str(payload.get("mode")))
        except ValueError as error:
            raise TrainingError("training result mode is invalid") from error
        expected_keys = base_keys | (
            {"tiny_overfit"} if mode is TrainMode.TINY_OVERFIT else set()
        )
        _require_exact_keys(payload, expected_keys, "training result")
        if (
            payload["format"] != TRAIN_RESULT_FORMAT
            or payload["schema_version"] != SCHEMA_VERSION
        ):
            raise TrainingError("training result is not the current schema-31 format")
        unsigned = dict(payload)
        record_identity = _require_digest(
            unsigned.pop("record_sha256"), "training result identity"
        )
        if _identity_digest(unsigned) != record_identity:
            raise TrainingError("training result content hash changed")
        result_protocol_identity = _require_digest(
            payload["protocol_identity"], "training result protocol identity"
        )
        if payload["training_system_identity"] != protocol_training_system_identity(
            protocol
        ):
            raise TrainingError("training result uses another R04 training system")
        if payload["bank_protocol_identity"] != protocol_bank_identity(protocol):
            raise TrainingError("training result uses another bank protocol")
        frozen_bank = _require_digest(
            protocol.status.get("r04_training_bank_identity"),
            "status.r04_training_bank_identity",
        )
        if (
            _require_digest(payload["bank_identity"], "training result bank")
            != frozen_bank
        ):
            raise TrainingError("training result uses another frozen R04 bank")
        if (
            payload["bank_entry_count"] != 445
            or payload["bank_plan_prefix_count"] != 445
        ):
            raise TrainingError(
                "training result was not run from the complete R04 bank"
            )
        r03_identity = _require_digest(
            payload["r03_qualification_identity"],
            "training result R03 qualification",
        )
        if r03_identity != protocol.status["r03_qualification_identity"]:
            raise TrainingError("training result uses another R03 qualification")
        condition = ExperimentCondition(str(payload["condition"]))
        if (
            not condition.trainable
            or payload["grouping_mode"] != condition.grouping_mode.value
        ):
            raise TrainingError("training result condition or grouping mode is invalid")
        seed = _require_int(payload["seed"], "training result seed")
        config_value = _require_mapping(payload["config"], "training result config")
        try:
            config = TrainingConfig(**dict(config_value))
        except (TypeError, ValueError) as error:
            raise TrainingError("training result config is invalid") from error
        authorization_node = payload["authorization_node"]
        if authorization_node not in {"R04", "G2", "G3"}:
            raise TrainingError("training result authorization node is invalid")
        r04_value = payload["r04_training_qualification_identity"]
        r04_identity = (
            None
            if r04_value is None
            else _require_digest(r04_value, "training result R04 qualification")
        )
        r05_value = payload["r05_freeze_identity"]
        r05_identity = (
            None
            if r05_value is None
            else _require_digest(r05_value, "training result R05 freeze")
        )
        g2_value = payload["g2_verdict_identity"]
        g2_identity = (
            None
            if g2_value is None
            else _require_digest(g2_value, "training result G2 verdict")
        )
        initial_state = _require_digest(
            payload["initial_model_state_sha256"], "initial model state"
        )
        development_definitions: DevelopmentWorlds | None = None
        development_reference = payload["development_manifest"]
        if mode is TrainMode.TINY_OVERFIT:
            if development_reference is not None:
                raise TrainingError("tiny-overfit result cannot reference development")
        else:
            reference = _require_mapping(
                development_reference, "training result development manifest"
            )
            _require_exact_keys(
                reference,
                {"path", "file_sha256"},
                "training result development manifest",
            )
            reference_path = reference["path"]
            if not isinstance(reference_path, str) or not reference_path:
                raise TrainingError("development manifest path must be non-empty")
            candidate = Path(reference_path).expanduser()
            if not candidate.is_absolute():
                candidate = resolved.parent / candidate
            try:
                manifest_path = candidate.resolve(strict=True)
            except OSError as error:
                raise TrainingError("development manifest is unavailable") from error
            if _sha256_file(manifest_path) != _require_digest(
                reference["file_sha256"], "development manifest file identity"
            ):
                raise TrainingError("development manifest file identity changed")
            try:
                development_definitions = load_development_worlds(
                    manifest_path, protocol=protocol
                )
            except (OSError, ProtocolError) as error:
                raise TrainingError("development manifest is invalid") from error
            if (
                not development_definitions.validated
                or development_definitions.population_identity
                != payload["development_population_identity"]
            ):
                raise TrainingError(
                    "development manifest differs from the frozen population"
                )
        maximum_updates = _require_int(
            payload["maximum_updates"], "training result maximum updates", minimum=1
        )
        _require_number(payload["last_effective_batch_loss"], "last training loss")
        updates = _require_int(
            payload["optimizer_updates"], "optimizer updates", minimum=1
        )
        completed_epochs = _require_int(
            payload["completed_epochs"], "completed epochs", minimum=0
        )
        windows_seen = _require_int(payload["windows_seen"], "windows seen", minimum=1)
        if payload["checkpoint_selection_metric"] != "macro_fused_point_ap":
            raise TrainingError("training result checkpoint metric changed")
        _validate_result_request(
            protocol,
            mode=mode,
            condition=condition,
            seed=seed,
            config=config,
            pilot_stage=payload["pilot_stage"],
            authorization_node=authorization_node,
            window_count=payload["window_count"],
            maximum_updates=maximum_updates,
            r04_identity=r04_identity,
            r05_identity=r05_identity,
            g2_identity=g2_identity,
        )
        expected_epoch, expected_next_entry, expected_windows_seen = _training_progress(
            window_count=int(payload["window_count"]),
            accumulation=config.gradient_accumulation,
            updates=updates,
        )
        if completed_epochs != expected_epoch or windows_seen != expected_windows_seen:
            raise TrainingError(
                "training result epoch or exposure count is inconsistent"
            )

        def checkpoint(name: str, expected_sha256: object) -> Mapping[str, object]:
            checkpoint_path = resolved.parent / name
            if _sha256_file(checkpoint_path) != _require_digest(
                expected_sha256, f"{name} file identity"
            ):
                raise TrainingError(f"{name} file identity changed")
            try:
                loaded = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
            except Exception as error:
                raise TrainingError(f"cannot safely load {name}") from error
            checkpoint_payload = _require_mapping(loaded, name)
            expected = {
                "format": TRAIN_CHECKPOINT_FORMAT,
                "schema_version": SCHEMA_VERSION,
                "protocol_identity": payload["protocol_identity"],
                "training_system_identity": protocol_training_system_identity(protocol),
                "bank_protocol_identity": protocol_bank_identity(protocol),
                "bank_identity": frozen_bank,
                "renderer_identity": protocol.renderer_identity,
                "stu_identity": protocol.stu["checkpoint_sha256"],
                "mode": mode.value,
                "pilot_stage": payload["pilot_stage"],
                "condition": condition.value,
                "grouping_mode": condition.grouping_mode.value,
                "seed": seed,
                "authorization_node": payload["authorization_node"],
                "r03_qualification_identity": r03_identity,
                "r04_training_qualification_identity": payload[
                    "r04_training_qualification_identity"
                ],
                "r05_freeze_identity": payload["r05_freeze_identity"],
                "g2_verdict_identity": payload["g2_verdict_identity"],
                "initial_model_state_sha256": initial_state,
                "config": dict(config_value),
                "bank_entry_count": 445,
                "bank_plan_prefix_count": 445,
                "window_count": payload["window_count"],
                "maximum_updates": maximum_updates,
                "development_population_identity": payload[
                    "development_population_identity"
                ],
                "development_manifest": payload["development_manifest"],
            }
            _require_exact_keys(
                checkpoint_payload,
                {
                    *expected,
                    "renderer_identity",
                    "stu_identity",
                    "device",
                    "epoch",
                    "next_entry",
                    "updates",
                    "windows_seen",
                    "last_loss",
                    "best_development_ap",
                    "last_development",
                    "development_trace",
                    "model_state_sha256",
                    "model_state_dict",
                    "optimizer_state_dict",
                    "scheduler_state_dict",
                    "python_rng_state",
                    "numpy_rng_state",
                    "torch_rng_state",
                    "cuda_rng_state",
                },
                name,
            )
            for field, value in expected.items():
                if _plain_json(checkpoint_payload.get(field)) != _plain_json(value):
                    raise TrainingError(f"{name} {field} changed")
            checkpoint_updates = _require_int(
                checkpoint_payload.get("updates"), f"{name} updates", minimum=1
            )
            checkpoint_epoch, checkpoint_next, checkpoint_windows = _training_progress(
                window_count=int(payload["window_count"]),
                accumulation=config.gradient_accumulation,
                updates=checkpoint_updates,
            )
            if (
                checkpoint_payload.get("epoch") != checkpoint_epoch
                or checkpoint_payload.get("next_entry") != checkpoint_next
                or checkpoint_payload.get("windows_seen") != checkpoint_windows
            ):
                raise TrainingError(f"{name} training cursor is inconsistent")
            state = _require_mapping(
                checkpoint_payload.get("model_state_dict"), f"{name} model state"
            )
            state_identity = model_state_sha256(state)
            if checkpoint_payload.get("model_state_sha256") != state_identity:
                raise TrainingError(f"{name} model-state identity changed")
            return checkpoint_payload

        last = checkpoint("last.pt", payload["last_checkpoint_sha256"])
        if (
            last.get("model_state_sha256") != payload["last_model_state_sha256"]
            or last.get("updates") != updates
            or last.get("epoch") != payload["completed_epochs"]
            or last.get("next_entry") != expected_next_entry
            or last.get("windows_seen") != payload["windows_seen"]
            or last.get("last_loss") != payload["last_effective_batch_loss"]
            or last.get("best_development_ap")
            != payload["best_development_macro_fused_point_ap"]
            or _plain_json(last.get("development_trace"))
            != _plain_json(payload["development_trace"])
            or _plain_json(last.get("last_development"))
            != _plain_json(payload["last_development"])
        ):
            raise TrainingError("last checkpoint differs from the result summary")

        trace_raw = payload["development_trace"]
        if not isinstance(trace_raw, list):
            raise TrainingError("training result development trace must be an array")
        trace: list[Mapping[str, object]] = []
        previous_update = 0
        for position, item in enumerate(trace_raw):
            entry = _require_mapping(item, f"development trace[{position}]")
            _require_exact_keys(
                entry,
                {
                    "update",
                    "macro_fused_point_ap",
                    "probability_saturation_fraction",
                    "development_record_identity",
                },
                f"development trace[{position}]",
            )
            update = _require_int(
                entry["update"], "development trace update", minimum=1
            )
            score = _require_number(
                entry["macro_fused_point_ap"], "development trace score"
            )
            saturation = _require_number(
                entry["probability_saturation_fraction"],
                "development trace saturation fraction",
            )
            if (
                update <= previous_update
                or not 0.0 <= score <= 1.0
                or not 0.0 <= saturation <= 1.0
            ):
                raise TrainingError("development trace order or score is invalid")
            _require_digest(
                entry["development_record_identity"],
                "development trace record identity",
            )
            previous_update = update
            trace.append(entry)

        if mode is TrainMode.TINY_OVERFIT:
            if (
                payload["status"] not in {"passed", "failed"}
                or payload["pilot_stage"] is not None
                or trace
                or any(
                    payload[name] is not None
                    for name in (
                        "r05_freeze_identity",
                        "g2_verdict_identity",
                        "best_development_macro_fused_point_ap",
                        "best_checkpoint_sha256",
                        "best_model_state_sha256",
                        "best_evaluated_update",
                        "best_development",
                        "development_population_identity",
                        "development_manifest",
                        "last_development",
                    )
                )
            ):
                raise TrainingError("tiny-overfit result contains incompatible fields")
            tiny = _require_mapping(payload["tiny_overfit"], "tiny-overfit summary")
            _require_exact_keys(
                tiny,
                {"training_loss", "training_AP_percent", "pass_rule"},
                "tiny-overfit summary",
            )
            loss = _require_number(tiny["training_loss"], "tiny-overfit loss")
            ap_value = tiny["training_AP_percent"]
            ap = (
                None
                if ap_value is None
                else _require_number(ap_value, "tiny-overfit AP")
            )
            rule = _plain_json(protocol.training["tiny_overfit"]["pass_any"])
            if _plain_json(tiny["pass_rule"]) != rule:
                raise TrainingError("tiny-overfit pass rule changed")
            passed = (
                ap is not None and ap >= float(rule["training_AP_minimum_percent"])  # type: ignore[index]
            ) or loss < float(rule["loss_strictly_below"])  # type: ignore[index]
            if (payload["status"] == "passed") != passed:
                raise TrainingError("tiny-overfit decision disagrees with its metrics")
            if payload["status"] == "failed" and updates != maximum_updates:
                raise TrainingError(
                    "failed tiny-overfit stopped before its frozen limit"
                )
            if updates > maximum_updates:
                raise TrainingError("tiny-overfit exceeded its frozen update limit")
        else:
            expected_trace_updates = list(
                range(
                    config.evaluation_interval_updates,
                    maximum_updates + 1,
                    config.evaluation_interval_updates,
                )
            )
            if (
                not expected_trace_updates
                or expected_trace_updates[-1] != maximum_updates
            ):
                expected_trace_updates.append(maximum_updates)
            if (
                payload["status"] != "completed"
                or updates != maximum_updates
                or [int(item["update"]) for item in trace] != expected_trace_updates
            ):
                raise TrainingError("pilot/formal result is incomplete")
            best = checkpoint("best.pt", payload["best_checkpoint_sha256"])
            scores = [float(item["macro_fused_point_ap"]) for item in trace]
            maximum = max(scores)
            earliest = min(
                int(item["update"])
                for item in trace
                if float(item["macro_fused_point_ap"]) == maximum
            )
            best_record = _require_mapping(
                payload["best_development"], "best development record"
            )
            best_score, best_saturation = _validate_development_summary(
                best_record,
                protocol=protocol,
                condition=condition,
                protocol_identity=result_protocol_identity,
                development_identity=str(payload["development_population_identity"]),
                development=development_definitions,  # type: ignore[arg-type]
                result_directory=resolved.parent,
                model_state_identity=str(payload["best_model_state_sha256"]),
                name="best development record",
            )
            last_record = _require_mapping(
                payload["last_development"], "last development record"
            )
            last_score, last_saturation = _validate_development_summary(
                last_record,
                protocol=protocol,
                condition=condition,
                protocol_identity=result_protocol_identity,
                development_identity=str(payload["development_population_identity"]),
                development=development_definitions,  # type: ignore[arg-type]
                result_directory=resolved.parent,
                model_state_identity=str(payload["last_model_state_sha256"]),
                name="last development record",
            )
            best_trace = next(item for item in trace if int(item["update"]) == earliest)
            last_trace = trace[-1]
            if (
                payload["best_development_macro_fused_point_ap"] != maximum
                or payload["best_evaluated_update"] != earliest
                or best.get("updates") != earliest
                or best.get("best_development_ap") != maximum
                or best.get("model_state_sha256") != payload["best_model_state_sha256"]
                or _plain_json(best.get("last_development")) != _plain_json(best_record)
                or best_score != maximum
                or last_score != float(last_trace["macro_fused_point_ap"])
                or best_saturation
                != float(best_trace["probability_saturation_fraction"])
                or last_saturation
                != float(last_trace["probability_saturation_fraction"])
                or _identity_digest(best_record)
                != best_trace["development_record_identity"]
                or _identity_digest(last_record)
                != last_trace["development_record_identity"]
            ):
                raise TrainingError("best checkpoint selection cannot be reproduced")
            development_identity = _require_digest(
                payload["development_population_identity"],
                "training result development population",
            )
            if development_identity != protocol.status["r02_population_identity"]:
                raise TrainingError("training result uses another R02 population")
            if mode is TrainMode.PILOT:
                pass
            else:
                if result_protocol_identity != protocol.scientific_identity:
                    raise TrainingError("formal result differs from the R05 freeze")
        return cls(
            resolved,
            record_identity,
            _freeze_json(payload),  # type: ignore[arg-type]
        )


def _r04_run_reference(
    value: object,
    *,
    directory: Path,
    protocol: AJAEProtocol,
    name: str,
) -> TrainingRunRecord:
    reference = _require_mapping(value, name)
    _require_exact_keys(reference, {"path", "record_sha256"}, name)
    relative = reference["path"]
    if not isinstance(relative, str) or not relative:
        raise TrainingError(f"{name}.path must be a non-empty string")
    path = Path(relative)
    target = path if path.is_absolute() else directory / path
    record = TrainingRunRecord.load(target, protocol=protocol)
    if record.record_sha256 != _require_digest(
        reference["record_sha256"], f"{name}.record_sha256"
    ):
        raise TrainingError(f"{name} points to another training result")
    return record


def _r04_record_config(record: TrainingRunRecord) -> dict[str, object]:
    return dict(_require_mapping(record.payload["config"], "training result config"))


def _r04_record_score(record: TrainingRunRecord) -> float:
    return _require_number(
        record.payload["best_development_macro_fused_point_ap"],
        "pilot best development AP",
    )


@dataclass(frozen=True, slots=True)
class R04TrainingQualificationRecord:
    """Recompute the ordered tiny-overfit and pilot selection evidence."""

    path: Path
    record_sha256: str
    bank_identity: str
    winner: Mapping[str, object]

    @classmethod
    def load(
        cls, path: Path | str, *, protocol: AJAEProtocol
    ) -> "R04TrainingQualificationRecord":
        requested = Path(path).expanduser()
        try:
            resolved = requested.resolve(strict=True)
            raw = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingError("R04 qualification record is unreadable") from error
        root = _require_mapping(raw, "R04 qualification record")
        _require_exact_keys(
            root,
            {
                "format",
                "schema_version",
                "training_system_identity",
                "r02_population_identity",
                "r02_verdict_identity",
                "r03_qualification_identity",
                "bank",
                "tiny_overfit",
                "pilot",
                "frozen_stage_winners",
                "decision",
                "record_sha256",
            },
            "R04 qualification record",
        )
        unsigned = dict(root)
        identity = _require_digest(
            unsigned.pop("record_sha256"), "R04 qualification identity"
        )
        if _identity_digest(unsigned) != identity:
            raise TrainingError("R04 qualification content hash changed")
        status = protocol.status
        if (
            root["format"] != R04_QUALIFICATION_FORMAT
            or root["schema_version"] != SCHEMA_VERSION
            or root["training_system_identity"]
            != protocol_training_system_identity(protocol)
            or root["r02_population_identity"] != status["r02_population_identity"]
            or root["r02_verdict_identity"] != status["r02_verdict_identity"]
            or root["r03_qualification_identity"]
            != status["r03_qualification_identity"]
            or root["decision"] != "pass"
        ):
            raise TrainingError("R04 qualification does not authorize this protocol")
        if status["r04_training_qualification_identity"] not in {None, identity}:
            raise TrainingError("R04 qualification differs from the protocol identity")
        bank_reference = _require_mapping(root["bank"], "R04 bank reference")
        _require_exact_keys(
            bank_reference, {"path", "bank_identity"}, "R04 bank reference"
        )
        bank_path_value = bank_reference["path"]
        if not isinstance(bank_path_value, str) or not bank_path_value:
            raise TrainingError("R04 bank path must be a non-empty string")
        bank_path = Path(bank_path_value)
        bank = load_window_train_bank(
            bank_path if bank_path.is_absolute() else resolved.parent / bank_path,
            protocol=protocol,
        )
        bank_identity = _require_digest(
            bank_reference["bank_identity"], "R04 bank identity"
        )
        if (
            bank.test_fixture
            or len(bank) != 445
            or bank.plan_prefix_count != 445
            or bank.bank_identity != bank_identity
            or bank_identity != status["r04_training_bank_identity"]
        ):
            raise TrainingError("R04 qualification does not bind the complete bank")

        tiny = _require_mapping(root["tiny_overfit"], "R04 tiny-overfit records")
        _require_exact_keys(tiny, {"B1", "B2", "B3"}, "R04 tiny-overfit records")
        used: set[str] = set()
        tiny_records: list[TrainingRunRecord] = []
        tiny_spec = _require_mapping(
            protocol.training["tiny_overfit"], "training.tiny_overfit"
        )
        for condition in ("B1", "B2", "B3"):
            record = _r04_run_reference(
                tiny[condition],
                directory=resolved.parent,
                protocol=protocol,
                name=f"tiny_overfit.{condition}",
            )
            used.add(record.record_sha256)
            if (
                record.mode is not TrainMode.TINY_OVERFIT
                or record.condition.value != condition
                or record.payload["status"] != "passed"
                or record.payload["window_count"] != 8
                or record.seed != tiny_spec["seed"]
                or _r04_record_config(record) != _plain_json(tiny_spec["config"])
            ):
                raise TrainingError(f"tiny_overfit.{condition} did not pass")
            tiny_records.append(record)
        if (
            len(
                {
                    record.payload["initial_model_state_sha256"]
                    for record in tiny_records
                }
            )
            != 1
        ):
            raise TrainingError("tiny-overfit conditions did not share initialization")

        pilot = _require_mapping(root["pilot"], "R04 pilot")
        _require_exact_keys(
            pilot,
            {"learning_rate_and_batch", "scheduler", "weight_decay"},
            "R04 pilot",
        )
        pilot_spec = protocol.training["pilot"]
        fixed = dict(pilot_spec["fixed_run_parameters"])
        saturation_limit = float(
            pilot_spec["selection"]["maximum_complete_saturation_fraction_exclusive"]
        )

        def load_pilot_runs(
            value: object,
            *,
            name: str,
            stage: str,
            expected: Sequence[tuple[dict[str, object], int, int]],
            require_safe: bool = True,
        ) -> list[TrainingRunRecord]:
            if not isinstance(value, list) or len(value) != len(expected):
                raise TrainingError(f"{name} has the wrong run count")
            records: list[TrainingRunRecord] = []
            for position, (reference, (variables, seed, updates)) in enumerate(
                zip(value, expected, strict=True)
            ):
                record = _r04_run_reference(
                    reference,
                    directory=resolved.parent,
                    protocol=protocol,
                    name=f"{name}[{position}]",
                )
                config = _r04_record_config(record)
                expected_config = {
                    **fixed,
                    **variables,
                    "maximum_updates": updates,
                }
                saturation = _require_number(
                    _require_mapping(
                        record.payload["last_development"],
                        "pilot final development",
                    )["probability_saturation_fraction"],
                    "pilot probability saturation fraction",
                )
                if (
                    record.mode is not TrainMode.PILOT
                    or record.condition.value != pilot_spec["condition"]
                    or record.payload["pilot_stage"] != stage
                    or record.seed != seed
                    or record.payload["window_count"] != 128
                    or config != expected_config
                    or not 0.0 <= saturation <= 1.0
                    or (require_safe and saturation >= saturation_limit)
                    or record.record_sha256 in used
                ):
                    raise TrainingError(f"{name}[{position}] violates the pilot plan")
                used.add(record.record_sha256)
                records.append(record)
            return records

        lr_batch = _require_mapping(
            pilot["learning_rate_and_batch"], "R04 learning-rate/batch stage"
        )
        _require_exact_keys(
            lr_batch,
            {
                "screen_50",
                "eligible_after_50",
                "screen_200",
                "screen_600",
                "finalists",
                "winner",
            },
            "R04 learning-rate/batch stage",
        )
        grid = [
            {
                "learning_rate": float(learning_rate),
                "gradient_accumulation": int(accumulation),
                "scheduler": "constant",
                "weight_decay": 0.0,
            }
            for learning_rate in pilot_spec["learning_rates"]
            for accumulation in pilot_spec["gradient_accumulation"]
        ]
        screen_50 = load_pilot_runs(
            lr_batch["screen_50"],
            name="pilot.learning_rate_and_batch.screen_50",
            stage="learning_rate_and_batch",
            expected=[(item, 1001, 50) for item in grid],
            require_safe=False,
        )
        survivor_indices = [
            index
            for index, record in enumerate(screen_50)
            if _require_number(
                _require_mapping(
                    record.payload["last_development"],
                    "50-update pilot final development",
                )["probability_saturation_fraction"],
                "50-update pilot saturation fraction",
            )
            < saturation_limit
        ]
        minimum_survivors = int(pilot_spec["selection"]["screen_50_minimum_survivors"])
        eligible = [
            {
                "learning_rate": grid[index]["learning_rate"],
                "gradient_accumulation": grid[index]["gradient_accumulation"],
            }
            for index in survivor_indices
        ]
        if (
            len(survivor_indices) < minimum_survivors
            or _plain_json(lr_batch["eligible_after_50"]) != eligible
        ):
            raise TrainingError("R04 50-update eligibility cannot be reproduced")
        screen_200 = load_pilot_runs(
            lr_batch["screen_200"],
            name="pilot.learning_rate_and_batch.screen_200",
            stage="learning_rate_and_batch",
            expected=[(grid[index], 1001, 200) for index in survivor_indices],
        )
        if (
            len(
                {
                    record.payload["initial_model_state_sha256"]
                    for record in screen_50 + screen_200
                }
            )
            != 1
        ):
            raise TrainingError("pilot seed 1001 did not share one initialization")
        finalist_indices = sorted(
            survivor_indices,
            key=lambda index: (
                -_r04_record_score(screen_200[survivor_indices.index(index)]),
                index,
            ),
        )[: int(pilot_spec["selection"]["screen_200_finalist_count"])]
        finalists = [
            {
                "learning_rate": grid[index]["learning_rate"],
                "gradient_accumulation": grid[index]["gradient_accumulation"],
            }
            for index in finalist_indices
        ]
        if _plain_json(lr_batch["finalists"]) != finalists:
            raise TrainingError("R04 200-update finalists cannot be reproduced")
        screen_600_variables = [
            {
                **grid[index],
            }
            for index in finalist_indices
        ]
        screen_600 = load_pilot_runs(
            lr_batch["screen_600"],
            name="pilot.learning_rate_and_batch.screen_600",
            stage="learning_rate_and_batch",
            expected=[(item, 1002, 600) for item in screen_600_variables],
        )
        winning_index = max(
            range(len(screen_600)),
            key=lambda index: (
                _r04_record_score(screen_600[index]),
                -finalist_indices[index],
            ),
        )
        lr_winner = finalists[winning_index]
        if _plain_json(lr_batch["winner"]) != lr_winner:
            raise TrainingError("R04 learning-rate/batch winner cannot be reproduced")

        scheduler_stage = _require_mapping(pilot["scheduler"], "R04 scheduler stage")
        _require_exact_keys(scheduler_stage, {"runs", "winner"}, "R04 scheduler stage")
        scheduler_values = list(pilot_spec["schedulers_after_learning_rate_selection"])
        scheduler_runs = load_pilot_runs(
            scheduler_stage["runs"],
            name="pilot.scheduler.runs",
            stage="scheduler",
            expected=[
                (
                    {
                        **lr_winner,
                        "scheduler": scheduler,
                        "weight_decay": 0.0,
                    },
                    1002,
                    600,
                )
                for scheduler in scheduler_values
            ],
        )
        scheduler_index = max(
            range(len(scheduler_runs)),
            key=lambda index: (_r04_record_score(scheduler_runs[index]), -index),
        )
        scheduler_winner = scheduler_values[scheduler_index]
        if scheduler_stage["winner"] != scheduler_winner:
            raise TrainingError("R04 scheduler winner cannot be reproduced")

        decay_stage = _require_mapping(pilot["weight_decay"], "R04 weight-decay stage")
        _require_exact_keys(decay_stage, {"runs", "winner"}, "R04 weight-decay stage")
        decay_values = [
            float(value)
            for value in pilot_spec["weight_decay_after_scheduler_selection"]
        ]
        decay_runs = load_pilot_runs(
            decay_stage["runs"],
            name="pilot.weight_decay.runs",
            stage="weight_decay",
            expected=[
                (
                    {
                        **lr_winner,
                        "scheduler": scheduler_winner,
                        "weight_decay": decay,
                    },
                    1002,
                    600,
                )
                for decay in decay_values
            ],
        )
        if (
            len(
                {
                    record.payload["initial_model_state_sha256"]
                    for record in screen_600 + scheduler_runs + decay_runs
                }
            )
            != 1
        ):
            raise TrainingError("pilot seed 1002 did not share one initialization")
        decay_index = max(
            range(len(decay_runs)),
            key=lambda index: (_r04_record_score(decay_runs[index]), -index),
        )
        decay_winner = decay_values[decay_index]
        if decay_stage["winner"] != decay_winner:
            raise TrainingError("R04 weight-decay winner cannot be reproduced")
        winner = {
            **lr_winner,
            "scheduler": scheduler_winner,
            "weight_decay": decay_winner,
        }
        if (
            _plain_json(root["frozen_stage_winners"]) != winner
            or _plain_json(protocol.training["pilot"]["frozen_stage_winners"]) != winner
        ):
            raise TrainingError("R04 frozen pilot winners differ from recomputation")
        return cls(resolved, identity, bank_identity, MappingProxyType(winner))


def finalize_r04_training_qualification(
    *,
    protocol_path: Path | str,
    evidence_path: Path | str,
    destination: Path | str,
) -> R04TrainingQualificationRecord:
    """Build and validate the sole R04 verdict from content-addressed run references."""

    protocol = _load_training_protocol(protocol_path)
    if type(protocol) is not AJAEProtocol or protocol.status["current_node"] != "R04":
        raise TrainingError("R04 qualification can be finalized only during R04")
    if protocol.status["r04_training_qualification_identity"] is not None:
        raise TrainingError("R04 qualification is already bound by the protocol")
    try:
        evidence = json.loads(
            Path(evidence_path)
            .expanduser()
            .resolve(strict=True)
            .read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingError("R04 evidence draft is unreadable") from error
    draft = _require_mapping(evidence, "R04 evidence draft")
    _require_exact_keys(draft, {"bank", "tiny_overfit", "pilot"}, "R04 evidence draft")
    winners = _plain_json(protocol.training["pilot"]["frozen_stage_winners"])
    if not isinstance(winners, dict) or any(
        value is None for value in winners.values()
    ):
        raise TrainingError("all pilot-stage winners must be frozen before R04 verdict")
    record: dict[str, object] = {
        "format": R04_QUALIFICATION_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "training_system_identity": protocol_training_system_identity(protocol),
        "r02_population_identity": protocol.status["r02_population_identity"],
        "r02_verdict_identity": protocol.status["r02_verdict_identity"],
        "r03_qualification_identity": protocol.status["r03_qualification_identity"],
        "bank": _plain_json(draft["bank"]),
        "tiny_overfit": _plain_json(draft["tiny_overfit"]),
        "pilot": _plain_json(draft["pilot"]),
        "frozen_stage_winners": winners,
        "decision": "pass",
    }
    record["record_sha256"] = _identity_digest(record)
    final = Path(destination).expanduser().resolve()
    if final.exists():
        raise TrainingError("refusing to replace an existing R04 verdict")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_suffix(final.suffix + ".tmp")
    if temporary.exists():
        raise TrainingError("R04 verdict temporary path already exists")
    try:
        _atomic_json(temporary, record)
        R04TrainingQualificationRecord.load(temporary, protocol=protocol)
        os.replace(temporary, final)
        _fsync_directory(final.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return R04TrainingQualificationRecord.load(final, protocol=protocol)


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    mode: TrainMode
    condition: ExperimentCondition
    seed: int
    window_count: int
    maximum_updates: int
    pilot_stage: str | None = None


_TRAINING_AUTHORIZATION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _TrainingAuthorization:
    token: object
    node: str
    r03_qualification_identity: str
    r04_training_qualification_identity: str | None
    r05_freeze_identity: str | None
    g2_verdict_identity: str | None


def validate_training_request(
    protocol: object,
    mode: TrainMode | str,
    condition: ExperimentCondition | str,
    seed: int,
    config: TrainingConfig,
    *,
    bank_size: int,
    pilot_stage: str | None = None,
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
    status = _require_mapping(getattr(protocol, "status", None), "protocol.status")
    state_machine = tuple(getattr(protocol, "state_machine", ()))
    node = status.get("current_node")
    if node not in state_machine:
        raise TrainingError("protocol current node is outside its state machine")
    if selected_mode in {TrainMode.TINY_OVERFIT, TrainMode.PILOT} and node != "R04":
        raise TrainingError(
            f"{selected_mode.value} training is permitted only during R04"
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
        if pilot_stage is not None:
            raise TrainingError("tiny_overfit cannot declare a pilot stage")
        if type(config) is not TrainingConfig:
            raise TrainingError("training config must be TrainingConfig")
        tiny = _require_mapping(training["tiny_overfit"], "training.tiny_overfit")
        if seed != tiny["seed"] or dict(asdict(config)) != _plain_json(tiny["config"]):
            raise TrainingError("tiny_overfit seed or config differs from the protocol")
        windows = int(tiny["windows"])
        maximum_updates = int(tiny["maximum_updates"])
        if config.maximum_updates not in {None, maximum_updates}:
            raise TrainingError("tiny_overfit maximum_updates is fixed by the protocol")
    elif selected_mode is TrainMode.PILOT:
        if type(config) is not TrainingConfig:
            raise TrainingError("training config must be TrainingConfig")
        if selected_condition.value != pilot["condition"]:
            raise TrainingError("pilot selection is frozen to B3")
        if seed not in pilot_seeds:
            raise TrainingError("pilot seed is outside the result-blind pilot seed set")
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
            float(value) for value in pilot["weight_decay_after_scheduler_selection"]
        ):
            raise TrainingError("pilot weight_decay is outside the protocol grid")
        fixed = _require_mapping(
            pilot["fixed_run_parameters"], "training.pilot.fixed_run_parameters"
        )
        for name in (
            "epochs",
            "evaluation_interval_updates",
            "gradient_clip_norm",
        ):
            if getattr(config, name) != fixed[name]:
                raise TrainingError(f"pilot {name} differs from the frozen run value")
        screens = tuple(int(value) for value in pilot["screen_updates"])
        if config.maximum_updates not in screens:
            raise TrainingError("pilot maximum_updates must select a protocol screen")
        windows = int(pilot["windows"])
        maximum_updates = int(config.maximum_updates)
        stages = tuple(str(value) for value in pilot["stage_order"])
        if pilot_stage not in stages:
            raise TrainingError("pilot_stage must identify one ordered protocol stage")
        winners = _require_mapping(
            pilot["frozen_stage_winners"], "training.pilot.frozen_stage_winners"
        )
        if pilot_stage == "learning_rate_and_batch":
            if any(value is not None for value in winners.values()):
                raise TrainingError(
                    "learning-rate/batch screening requires no preselected pilot winner"
                )
            if config.scheduler != "constant" or config.weight_decay != 0.0:
                raise TrainingError(
                    "learning-rate/batch screening fixes constant scheduling and zero decay"
                )
            expected_seed = 1002 if maximum_updates == 600 else 1001
            if seed != expected_seed:
                raise TrainingError(
                    "learning-rate/batch screen seed is fixed by its update horizon"
                )
        elif pilot_stage == "scheduler":
            if (
                winners["learning_rate"] is None
                or winners["gradient_accumulation"] is None
                or winners["scheduler"] is not None
                or winners["weight_decay"] is not None
                or config.learning_rate != winners["learning_rate"]
                or config.gradient_accumulation != winners["gradient_accumulation"]
                or maximum_updates != 600
                or seed != 1002
                or config.weight_decay != 0.0
            ):
                raise TrainingError(
                    "scheduler screening requires the frozen learning-rate/batch winner"
                )
        elif (
            winners["learning_rate"] is None
            or winners["gradient_accumulation"] is None
            or winners["scheduler"] is None
            or winners["weight_decay"] is not None
            or config.learning_rate != winners["learning_rate"]
            or config.gradient_accumulation != winners["gradient_accumulation"]
            or config.scheduler != winners["scheduler"]
            or maximum_updates != 600
            or seed != 1002
        ):
            raise TrainingError(
                "weight-decay screening requires the frozen scheduler-stage winners"
            )
    else:
        if pilot_stage is not None:
            raise TrainingError("formal training cannot declare a pilot stage")
        if seed not in formal_seeds:
            raise TrainingError("formal seed is outside the frozen formal seed set")
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
        allowed_conditions = {
            "G2": {ExperimentCondition.B1},
            "G3": {ExperimentCondition.B2, ExperimentCondition.B3},
        }
        if selected_condition not in allowed_conditions.get(str(node), set()):
            raise TrainingError(
                "formal training is permitted only for B1 during G2 or B2/B3 during G3"
            )
        if type(config) is not TrainingConfig:
            raise TrainingError("training config must be TrainingConfig")
        recipe = formal.get("recipe")
        if not isinstance(recipe, Mapping) or dict(recipe) != asdict(config):
            raise TrainingError("formal config differs from the frozen R05 recipe")
        windows = int(training["bank"]["generation_plan"]["formal_entry_count"])
        if bank_size != windows:
            raise TrainingError("formal training requires the complete 445-window bank")
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
        selected_mode,
        selected_condition,
        seed,
        windows,
        maximum_updates,
        pilot_stage,
    )


@dataclass(frozen=True, slots=True)
class _DevelopmentSet:
    definitions: DevelopmentWorlds
    rendered_clips: tuple[DevelopmentClipWorld, ...]
    identity: str
    method_protocol_identity: str
    manifest_path: Path
    manifest_sha256: str


def _load_development_set(
    path: Path | str,
    rendered_clips: Iterable[DevelopmentClipWorld],
    *,
    protocol: AJAEProtocol,
    renderer_identity: str,
) -> _DevelopmentSet:
    """Bind runtime clips to the exact validated 24-clip development population."""

    try:
        manifest_path = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise TrainingError("development manifest is unavailable") from error
    definitions = load_development_worlds(manifest_path, protocol=protocol)
    if (
        type(definitions) is not DevelopmentWorlds
        or not definitions.validated
        or definitions.protocol_identity != protocol.development_population_identity
    ):
        raise TrainingError("development worlds must be validated and frozen")
    expected_design = protocol.evaluation_document["synthetic_development"][
        "exact_clip_length_and_count"
    ]
    clips = tuple(definitions.clips)
    if (
        len(clips) != int(expected_design["total_clips"])
        or len(clips) != int(expected_design["in_generator_clips"])
        or any(item.mechanism != "in_generator" for item in clips)
        or any(
            len(item.frame_ids) != int(expected_design["frames_per_clip"])
            or len(item.windows) != int(expected_design["overlapping_windows_per_clip"])
            for item in clips
        )
    ):
        raise TrainingError(
            "development definitions differ from the frozen 24-clip design"
        )
    if any(item.renderer_identity != renderer_identity for item in clips):
        raise TrainingError("training and development data use different renderers")

    runtime = tuple(rendered_clips)
    if len(runtime) != len(clips) or any(
        type(item) is not DevelopmentClipWorld for item in runtime
    ):
        raise TrainingError(
            "development input must contain exactly 24 DevelopmentClipWorld values"
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
    population_identity = _require_digest(
        definitions.population_identity, "development population identity"
    )
    return _DevelopmentSet(
        definitions,
        tuple(ordered),
        population_identity,
        protocol.scientific_identity,
        manifest_path,
        _sha256_file(manifest_path),
    )


def _development_record(
    evidence: DevelopmentFusedAP,
    condition: ExperimentCondition,
    development: _DevelopmentSet,
) -> dict[str, object]:
    """Select checkpoints on the 24 in-generator clips visible before S01."""

    if type(evidence) is not DevelopmentFusedAP:
        raise TrainingError("development result must be DevelopmentFusedAP")
    if ExperimentCondition(evidence.condition) is not condition:
        raise TrainingError("development result belongs to another condition")
    if evidence.fusion_semantics != DEVELOPMENT_FUSION_SEMANTICS:
        raise TrainingError("development result does not use all-occurrence fusion")
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
        raise TrainingError(
            "development result has a non-authoritative method identity"
        )
    clips = tuple(evidence.clips)
    definitions = tuple(development.definitions.clips)
    if len(clips) != 24 or len(definitions) != 24:
        raise TrainingError("development result must contain the frozen 24 clips")
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
            raise TrainingError("development fused point AP must lie in [0,1]")
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
            raise TrainingError("development occurrence count is outside strata 1..5")
        histogram = dict(clip.occurrence_histogram)
        if set(histogram) != {str(value) for value in range(1, 6)}:
            raise TrainingError("development result omits an occurrence stratum")
        records.append(
            {
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
                "saturated_probability_count": _require_int(
                    clip.saturated_probability_count,
                    f"clips[{index}].saturated_probability_count",
                    minimum=0,
                ),
            }
        )

    macro_ap = math.fsum(float(item["fused_point_ap"]) for item in records) / len(
        records
    )
    if not math.isclose(
        float(evidence.macro_fused_point_ap),
        macro_ap,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise TrainingError("development macro AP disagrees with per-clip AP")
    saturation_count = sum(int(item["saturated_probability_count"]) for item in records)
    total_points = sum(int(item["unique_point_count"]) for item in records)
    return {
        "fusion_semantics": DEVELOPMENT_FUSION_SEMANTICS,
        "selection_metric": "macro_fused_point_ap",
        "macro_fused_point_ap": macro_ap,
        "probability_saturation_fraction": saturation_count / total_points,
        "scope": "24_in_generator_clips_only_held_out_torus_unopened_until_S01",
        "development_population_identity": development.identity,
        "evaluation_identity": identity.to_dict(),
        "clips": records,
    }


def _verified_renderer_inputs(protocol: AJAEProtocol) -> tuple[object, object]:
    """Load the sole frozen ray/sensor payload after checking its file identity."""

    path = protocol.sensor_calibration_path()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise TrainingError("frozen renderer calibration is missing") from error
    if _sha256_file(resolved) != protocol.render["calibration_sha256"]:
        raise TrainingError("frozen renderer calibration hash changed")
    try:
        ray_grid, sensor = load_sensor_calibration(resolved)
    except (OSError, RuntimeError, RenderError) as error:
        raise TrainingError("frozen renderer calibration is invalid") from error
    return ray_grid, sensor


def _open_training_sequence(
    data_root: Path | str,
    protocol: AJAEProtocol,
    sequence_id: int,
) -> STUSequence:
    try:
        return STUSequence.open(
            data_root,
            protocol=protocol,
            partition="train",
            sequence_id=sequence_id,
            label_mode=LabelMode.REQUIRED,
        )
    except (OSError, ValueError, TypeError) as error:
        raise TrainingError(f"cannot open labelled train/{sequence_id}") from error


def _normal_templates(
    sequence: STUSequence, protocol: AJAEProtocol
) -> tuple[object, ...]:
    controls = protocol.render["normal_controls"]
    try:
        return extract_normal_template_library(
            (sequence.source_frame(frame_id) for frame_id in sequence.frame_ids),
            minimum_points=int(controls["minimum_template_points"]),
            maximum_templates_per_class=int(controls["maximum_templates_per_class"]),
        )
    except RenderError as error:
        raise TrainingError(
            "train/206 cannot supply the frozen normal templates"
        ) from error


def _report_from_definition(value: Mapping[str, object]) -> WorldGenerationReport:
    """Restore the renderer's sole JSON null representation of +infinity."""

    plain = json.loads(_canonical_json(value))
    for placement in plain.get("placements", []):
        proposals = placement.get("proposal_minimum_obstacle_sdf_m")
        if isinstance(proposals, list):
            placement["proposal_minimum_obstacle_sdf_m"] = [
                math.inf if item is None else item for item in proposals
            ]
        if placement.get("minimum_obstacle_sdf_m") is None:
            placement["minimum_obstacle_sdf_m"] = math.inf
    try:
        return WorldGenerationReport.from_dict(plain)
    except (TypeError, ValueError, KeyError, RenderError) as error:
        raise TrainingError("development generation report is invalid") from error


def _render_development_definitions(
    definitions: DevelopmentWorlds,
    *,
    data_root: Path | str,
    protocol: AJAEProtocol,
) -> tuple[DevelopmentClipWorld, ...]:
    """Deterministically rebuild and content-check the 24 visible development clips."""

    if len(definitions.clips) != 24 or any(
        item.mechanism != "in_generator" for item in definitions.clips
    ):
        raise TrainingError(
            "pre-S01 development definitions must contain 24 in-generator clips"
        )
    ray_grid, sensor = _verified_renderer_inputs(protocol)
    sequence = _open_training_sequence(data_root, protocol, 201)
    density_voxel_size = float(
        protocol.render["window_descriptors"]["density_voxel_size_m"]
    )
    rendered_clips: list[DevelopmentClipWorld] = []
    for definition in definitions.clips:
        try:
            world = WorldSpec.from_dict(definition.world)
            report = _report_from_definition(definition.report)
            sources = tuple(
                sequence.source_frame(frame_id) for frame_id in definition.frame_ids
            )
            rendered = render_development_clip_world(
                world,
                report,
                sources,
                ray_grid,
                sensor,
                renderer_identity=protocol.renderer_identity,
                density_voxel_size_m=density_voxel_size,
            )
        except (OSError, ValueError, TypeError, RenderError) as error:
            raise TrainingError(
                f"cannot rebuild development clip {definition.identity}"
            ) from error
        if (
            rendered.identity != definition.identity
            or tuple(rendered.source_observation_identities)
            != tuple(definition.source_observation_identities)
            or tuple(window.identity for window in rendered.windows)
            != tuple(window.identity for window in definition.windows)
            or tuple(
                tuple(descriptor.to_dict() for descriptor in window.descriptors)
                for window in rendered.windows
            )
            != tuple(
                tuple(_plain_json(descriptor) for descriptor in window.descriptors)
                for window in definition.windows
            )
        ):
            raise TrainingError(
                "rebuilt development clip differs from its frozen definition"
            )
        rendered_clips.append(rendered)
    return tuple(rendered_clips)


def render_frozen_development_clips(
    path: Path | str,
    *,
    data_root: Path | str,
    protocol: AJAEProtocol,
) -> tuple[DevelopmentClipWorld, ...]:
    """Rebuild only a successfully adjudicated R02 development population."""

    definitions = load_development_worlds(path, protocol=protocol)
    if not definitions.validated:
        raise TrainingError("development definitions have not passed R02")
    return _render_development_definitions(
        definitions,
        data_root=data_root,
        protocol=protocol,
    )


def generate_window_train_bank(
    *,
    protocol_path: Path | str,
    data_root: Path | str,
    destination: Path | str,
    entry_count: int,
    device: torch.device | str,
) -> WindowTrainingBank:
    """Generate one fixed 8/128/445 prefix using no caller-selected science values."""

    protocol = _load_training_protocol(protocol_path)
    assert isinstance(protocol, AJAEProtocol)
    count = _require_int(entry_count, "entry_count", minimum=1)
    allowed = tuple(
        int(value)
        for value in protocol.training["bank"]["generation_plan"][
            "allowed_prefix_counts"
        ]
    )
    if count not in allowed:
        raise TrainingError(
            "bank generation accepts only the frozen 8/128/445 prefixes"
        )
    node = str(protocol.status["current_node"])
    if node == "R02":
        if count != 8:
            raise TrainingError(
                "R02 may generate only the eight-window inspection prefix"
            )
    elif node == "R03":
        if count != 445:
            raise TrainingError(
                "R03 must generate the complete 445-window bank before R04"
            )
    else:
        raise TrainingError(
            "training-bank generation is allowed only for R02 inspection or R03 freeze"
        )
    ray_grid, sensor = _verified_renderer_inputs(protocol)
    sequence = _open_training_sequence(data_root, protocol, 206)
    try:
        support_pool = load_qualified_support_pool(
            protocol.support_pool_path(206), source_sequence_id=206
        )
    except (OSError, ValueError, RenderError) as error:
        raise TrainingError("train/206 support pool is invalid") from error
    templates = _normal_templates(sequence, protocol)
    resolved_device = _resolve_device(device)
    encoder = FrozenSTUPointEncoder.from_protocol(
        protocol, project_root=protocol.path.parent
    ).to(resolved_device)
    encoder.eval()
    plan = protocol.training_bank_plan()[:count]
    density_voxel_size = float(
        protocol.render["window_descriptors"]["density_voxel_size_m"]
    )
    ray_mapping = np.asarray(ray_grid.canonical_ray_by_slot, dtype=np.int32)
    ray_mapping_digest = canonical_ray_mapping_digest(ray_mapping)

    def windows() -> Iterator[WindowTrainingData]:
        for row in plan:
            sources = tuple(
                sequence.source_frame(int(frame_id))
                for frame_id in row["source_frames"]
            )
            try:
                obstacles = collect_observed_obstacle_index(
                    sources, source_sequence_id=206
                )
                window = sample_window_world(
                    templates,
                    support_pool,
                    obstacles,
                    sources,
                    ray_grid,
                    sensor,
                    str(row["world_type"]),
                    int(row["root_seed"]),
                    renderer_identity=protocol.renderer_identity,
                    density_voxel_size_m=density_voxel_size,
                    maximum_attempts=int(row["maximum_observation_attempts"]),
                )
                with torch.no_grad():
                    encoded = {
                        item.source.frame_id: encoder(
                            item.source.coordinates,
                            item.source.features,
                            item.source.real_slots,
                        )
                        for item in window.rendered_frames
                    }
                yield window_training_data(
                    window,
                    encoded,
                    canonical_ray_by_slot=ray_mapping,
                    ray_mapping_digest=ray_mapping_digest,
                    protocol=protocol,
                )
            except (OSError, ValueError, TypeError, RenderError) as error:
                raise TrainingError(
                    f"cannot generate bank plan position {row['position']}"
                ) from error

    return _write_window_train_bank(
        destination,
        windows(),
        protocol=protocol,
        renderer_identity=protocol.renderer_identity,
        test_fixture=False,
    )


def generate_development_worlds(
    *,
    protocol_path: Path | str,
    data_root: Path | str,
    destination: Path | str,
) -> DevelopmentWorlds:
    """Generate the sole 24-clip formal definitions after result-blind thresholds freeze."""

    protocol = _load_training_protocol(protocol_path)
    assert isinstance(protocol, AJAEProtocol)
    if protocol.status["current_node"] != "R02":
        raise TrainingError("formal development generation belongs only to R02")
    if protocol.r02_thresholds is None:
        raise TrainingError(
            "freeze result-blind R02 thresholds before formal generation"
        )
    ray_grid, sensor = _verified_renderer_inputs(protocol)
    training_sequence = _open_training_sequence(data_root, protocol, 206)
    development_sequence = _open_training_sequence(data_root, protocol, 201)
    try:
        support_pool = load_qualified_support_pool(
            protocol.support_pool_path(201), source_sequence_id=201
        )
    except (OSError, ValueError, RenderError) as error:
        raise TrainingError("train/201 support pool is invalid") from error
    templates = _normal_templates(training_sequence, protocol)
    density_voxel_size = float(
        protocol.render["window_descriptors"]["density_voxel_size_m"]
    )

    def clips() -> Iterator[DevelopmentClipWorld]:
        for row in protocol.development_clip_plan():
            sources = tuple(
                development_sequence.source_frame(int(frame_id))
                for frame_id in row["source_frames"]
            )
            try:
                obstacles = collect_observed_obstacle_index(
                    sources, source_sequence_id=201
                )
                yield sample_development_clip_world(
                    templates,
                    support_pool,
                    obstacles,
                    sources,
                    ray_grid,
                    sensor,
                    int(row["root_seed"]),
                    renderer_identity=protocol.renderer_identity,
                    density_voxel_size_m=density_voxel_size,
                    maximum_attempts=int(row["maximum_observation_attempts"]),
                )
            except (OSError, ValueError, TypeError, RenderError) as error:
                raise TrainingError(
                    f"cannot generate development plan position {row['position']}"
                ) from error

    save_development_worlds(
        destination,
        clips(),
        protocol_identity=protocol.development_population_identity,
        plan_identity=protocol.development_clip_plan_identity,
    )
    return load_development_worlds(destination, protocol=protocol)


def _sealed_summary(record: Mapping[str, object]) -> dict[str, object]:
    result = dict(record)
    result["artifact_sha256"] = _identity_digest(result)
    return result


def finalize_r02_development_worlds(
    *,
    protocol_path: Path | str,
    development_worlds_path: Path | str,
    data_root: Path | str,
    visual_review_path: Path | str,
) -> DevelopmentWorlds:
    """Recompute R02 evidence once and irreversibly record pass or stop."""

    protocol = _load_training_protocol(protocol_path)
    assert isinstance(protocol, AJAEProtocol)
    if protocol.status["current_node"] != "R02" or protocol.r02_thresholds is None:
        raise TrainingError("R02 finalization requires frozen thresholds at node R02")
    definitions = load_development_worlds(development_worlds_path, protocol=protocol)
    if definitions.status != "definitions_only_unvalidated":
        raise TrainingError(
            "R02 finalization accepts one unadjudicated formal population"
        )
    review_path = Path(visual_review_path).expanduser().resolve(strict=True)
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingError("R02 visual review record is unreadable") from error
    review = _require_mapping(review, "R02 visual review")
    _require_exact_keys(
        review,
        {
            "format",
            "population_identity",
            "reviewed_clip_identities",
            "reviewer",
            "reviewed_at_utc",
            "checklist",
            "findings",
        },
        "R02 visual review",
    )
    if review["format"] != "ajae-r02-visual-review-v1":
        raise TrainingError("R02 visual review format is unsupported")
    if review["population_identity"] != definitions.population_identity:
        raise TrainingError("visual review belongs to another development population")
    reviewed_ids = review["reviewed_clip_identities"]
    if reviewed_ids != [item.identity for item in definitions.clips]:
        raise TrainingError(
            "visual review does not cover the ordered 24-clip population"
        )
    for name in ("reviewer", "reviewed_at_utc", "findings"):
        if not isinstance(review[name], str) or not review[name].strip():
            raise TrainingError(f"R02 visual review {name} must be non-empty")
    checklist = _require_mapping(review["checklist"], "R02 visual checklist")
    checklist_keys = {
        "world_fixed_before_all_scans",
        "no_synthetic_point_completion",
        "no_bottom_return_insertion",
        "no_scan_duplication_or_copying",
        "placements_visually_plausible",
        "returns_and_occlusion_visually_plausible",
    }
    _require_exact_keys(checklist, checklist_keys, "R02 visual checklist")
    if any(type(value) is not bool for value in checklist.values()):
        raise TrainingError("R02 visual checklist values must be boolean")

    runtime = _render_development_definitions(
        definitions,
        data_root=data_root,
        protocol=protocol,
    )
    windows = tuple(window for clip in runtime for window in clip.windows)
    try:
        pairs = match_window_entities(windows)
        balance = window_matching_balance(pairs)
    except RenderError:
        pairs = ()
        balance = None
    try:
        shortcut = (
            None
            if balance is None
            else window_shortcut_audit(pairs, seed=R02_SHORTCUT_SEED)
        )
    except RenderError:
        shortcut = None
    thresholds = protocol.r02_thresholds
    assert thresholds is not None
    visual_pass = all(bool(value) for value in checklist.values())
    visual = _sealed_summary(
        {
            "reviewed_clip_count": len(reviewed_ids),
            "reviewed_clip_identities": reviewed_ids,
            "reviewer": review["reviewer"],
            "reviewed_at_utc": review["reviewed_at_utc"],
            "checklist": dict(checklist),
            "findings": review["findings"],
            "passed": visual_pass,
        }
    )
    matching_base = {
        "eligible_mechanism": "in_generator",
        "algorithm": "support_stratified_linear_sum_assignment_standardized_euclidean",
        "feature_names": list(R02_MATCHING_FEATURES),
        "exact_matching_stratum": "support_semantic_id",
    }
    if balance is None:
        matching_pass = False
        matching = _sealed_summary(
            {
                **matching_base,
                "status": "not_computable",
                "failure_code": "matching_not_computable",
                "passed": False,
            }
        )
    else:
        pair_payload = [item.to_dict() for item in pairs]
        imbalance = float(balance["maximum_absolute_standardized_mean_difference"])
        matching_pass = len(pairs) >= int(
            thresholds["minimum_matched_pairs"]
        ) and imbalance <= float(
            thresholds["maximum_absolute_standardized_mean_difference"]
        )
        matching = _sealed_summary(
            {
                **matching_base,
                "status": "computed",
                "matched_pairs_sha256": _identity_digest(pair_payload),
                "pair_count": len(pairs),
                "standardized_mean_difference": balance["standardized_mean_difference"],
                "support_semantic_counts": balance["support_semantic_counts"],
                "maximum_absolute_standardized_mean_difference": imbalance,
                "passed": matching_pass,
            }
        )
    proxy_descriptors = [
        descriptor.to_dict()
        for clip in runtime
        for window in clip.windows
        for descriptor in window.descriptors
        if descriptor.label == "anomaly-proxy"
    ]
    returns = sorted(
        float(item["joint_visible_return_count"]) for item in proxy_descriptors
    )
    gains = sorted(float(item["densification_gain"]) for item in proxy_descriptors)

    def median(values: Sequence[float]) -> float:
        middle = len(values) // 2
        return (
            float(values[middle])
            if len(values) % 2
            else 0.5 * float(values[middle - 1] + values[middle])
        )

    median_returns = median(returns)
    median_gain = median(gains)
    fraction_gain = sum(value > 1.0 for value in gains) / len(gains)
    density_pass = (
        median_returns
        >= float(thresholds["minimum_median_proxy_joint_visible_return_count"])
        and median_gain >= float(thresholds["minimum_median_proxy_densification_gain"])
        and fraction_gain
        >= float(thresholds["minimum_proxy_fraction_densification_gain_above_one"])
    )
    densification = _sealed_summary(
        {
            "eligible_mechanism": "in_generator",
            "descriptor_population_sha256": _identity_digest(proxy_descriptors),
            "proxy_window_entity_count": len(proxy_descriptors),
            "median_proxy_joint_visible_return_count": median_returns,
            "median_proxy_densification_gain": median_gain,
            "proxy_fraction_densification_gain_above_one": fraction_gain,
            "passed": density_pass,
        }
    )
    shortcut_base = {
        "eligible_mechanism": "in_generator",
        "algorithm": "standardized_logistic_regression",
        "feature_names": list(R02_MATCHING_FEATURES),
        "seed": R02_SHORTCUT_SEED,
        "split_unit": "world_identity",
    }
    if shortcut is None:
        shortcut_pass = False
        shortcut_summary = _sealed_summary(
            {
                **shortcut_base,
                "status": "not_computable",
                "failure_code": (
                    "matching_not_computable"
                    if balance is None
                    else "shortcut_not_computable"
                ),
                "passed": False,
            }
        )
    else:
        world_identities = sorted(
            {item.control_world_identity for item in pairs}
            | {item.proxy_world_identity for item in pairs}
        )
        train_worlds = [
            world_identities[int(index)] for index in shortcut["train_groups"]
        ]
        test_worlds = [
            world_identities[int(index)] for index in shortcut["test_groups"]
        ]
        shortcut_pass = float(shortcut["balanced_accuracy"]) <= float(
            thresholds["maximum_shortcut_balanced_accuracy"]
        ) and abs(float(shortcut["auroc"]) - 0.5) <= float(
            thresholds["maximum_shortcut_absolute_auroc_deviation_from_half"]
        )
        shortcut_summary = _sealed_summary(
            {
                **shortcut_base,
                "status": "computed",
                "train_world_identities": train_worlds,
                "test_world_identities": test_worlds,
                "train_samples": shortcut["train_samples"],
                "test_samples": shortcut["test_samples"],
                "standardized_coefficients": shortcut["standardized_coefficients"],
                "intercept": shortcut["intercept"],
                "balanced_accuracy": shortcut["balanced_accuracy"],
                "auroc": shortcut["auroc"],
                "passed": shortcut_pass,
            }
        )
    decisions = {
        "visual_review_passed": visual_pass,
        "descriptor_integrity_passed": True,
        "proxy_control_matching_passed": matching_pass,
        "densification_passed": density_pass,
        "shortcut_audit_passed": shortcut_pass,
    }
    if set(decisions) != set(R02_VALIDATION_KEYS):
        raise AssertionError("R02 component decisions differ from the protocol")
    verdict: dict[str, object] = {
        "format": "ajae-r02-scientific-verdict-v1",
        "development_protocol_identity": protocol.development_population_identity,
        "formal_population_identity": definitions.population_identity,
        "development_plan_identity": protocol.development_clip_plan_identity,
        "thresholds_identity": _identity_digest(thresholds),
        "audit_algorithm_identity": r02_audit_algorithm_identity(),
        "visual_review": visual,
        "descriptor_integrity": {
            "checked_window_count": len(windows),
            "checked_descriptor_count": sum(
                len(window.descriptors) for window in windows
            ),
            "passed": True,
        },
        "matching": matching,
        "densification": densification,
        "shortcut_audit": shortcut_summary,
        "component_decisions": decisions,
        "decision": "pass" if all(decisions.values()) else "fail",
    }
    verdict["record_sha256"] = _identity_digest(verdict)
    target = Path(development_worlds_path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingError(
            "development definitions changed during R02 finalization"
        ) from error
    if payload.get("status") != "definitions_only_unvalidated":
        raise TrainingError("development definitions were already adjudicated")
    payload["validation"] = decisions
    payload["scientific_verdict"] = verdict
    payload["status"] = (
        "validated_frozen"
        if verdict["decision"] == "pass"
        else "adjudicated_failed_R02"
    )
    temporary = target.with_suffix(target.suffix + ".r02.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return load_development_worlds(target, protocol=protocol)


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
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
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
        authorization: _TrainingAuthorization | None = None,
        resume: bool = False,
    ) -> None:
        if (
            type(protocol) is not AJAEProtocol
            or protocol.schema_version != SCHEMA_VERSION
        ):
            raise TrainingError("WindowTrainer requires the schema-31 protocol")
        if type(model) is not JointWindowPointTransformer:
            raise TrainingError(
                "training requires the authoritative JointWindowPointTransformer"
            )
        if protocol_bank_identity(protocol) != bank.protocol_identity:
            raise TrainingError("training bank belongs to another bank protocol")
        if (
            type(authorization) is not _TrainingAuthorization
            or authorization.token is not _TRAINING_AUTHORIZATION_TOKEN
            or authorization.node != protocol.status["current_node"]
        ):
            raise TrainingError("training lacks content-validated stage authorization")
        if type(request) is not TrainingRequest or type(config) is not TrainingConfig:
            raise TrainingError("trainer request or config is not authoritative")
        if bank.test_fixture or len(bank) != 445 or bank.plan_prefix_count != 445:
            raise TrainingError("trainer requires the complete formal R04 bank")
        frozen_bank = _require_digest(
            protocol.status["r04_training_bank_identity"],
            "status.r04_training_bank_identity",
        )
        if bank.bank_identity != frozen_bank:
            raise TrainingError("trainer bank differs from the frozen R04 bank")
        canonical_request = validate_training_request(
            protocol,
            request.mode,
            request.condition,
            request.seed,
            config,
            bank_size=len(bank),
            pilot_stage=request.pilot_stage,
        )
        if canonical_request != request:
            raise TrainingError(
                "trainer request differs from canonical protocol values"
            )
        r03_identity = _require_digest(
            authorization.r03_qualification_identity,
            "training authorization R03 qualification",
        )
        if r03_identity != protocol.status["r03_qualification_identity"]:
            raise TrainingError("trainer uses another R03 qualification")
        if request.mode is TrainMode.FORMAL:
            expected_g2 = (
                None
                if request.condition is ExperimentCondition.B1
                else protocol.status["g2_verdict_identity"]
            )
            if (
                authorization.r04_training_qualification_identity
                != protocol.status["r04_training_qualification_identity"]
                or authorization.r05_freeze_identity
                != protocol.status["r05_freeze_identity"]
                or authorization.g2_verdict_identity != expected_g2
            ):
                raise TrainingError(
                    "formal trainer lacks its bound R04/R05/G2 evidence"
                )
        elif (
            authorization.r04_training_qualification_identity is not None
            or authorization.r05_freeze_identity is not None
            or authorization.g2_verdict_identity is not None
        ):
            raise TrainingError("R04 qualification runs cannot claim later evidence")
        self.protocol = protocol
        self.protocol_identity = _require_digest(
            protocol.scientific_identity, "protocol.scientific_identity"
        )
        self.bank = bank
        self.model = model
        self.request = request
        self.config = config
        self.authorization = authorization
        self.device = _resolve_device(device)
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.last_path = self.output_directory / "last.pt"
        self.best_path = self.output_directory / "best.pt"
        self.result_path = self.output_directory / "result.json"
        if not resume and any(
            path.exists() for path in (self.last_path, self.best_path, self.result_path)
        ):
            raise TrainingError("output already contains a training run; use resume")
        if request.mode is TrainMode.TINY_OVERFIT and development is not None:
            raise TrainingError("tiny_overfit uses only its frozen training subset")
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
        self.initial_model_state_sha256 = model_state_sha256(model.state_dict())
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
        self.development_trace: list[dict[str, object]] = []
        self.last_loss = math.nan
        self.last_evaluated_update = -1
        if resume:
            self._restore()

    def _checkpoint(self) -> dict[str, object]:
        return {
            "format": TRAIN_CHECKPOINT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "protocol_identity": self.protocol_identity,
            "training_system_identity": protocol_training_system_identity(
                self.protocol
            ),
            "bank_protocol_identity": self.bank.protocol_identity,
            "bank_identity": self.bank.bank_identity,
            "renderer_identity": self.bank.renderer_identity,
            "stu_identity": self.bank.stu_identity,
            "development_population_identity": (
                None if self.development is None else self.development.identity
            ),
            "development_manifest": (
                None
                if self.development is None
                else {
                    "path": str(self.development.manifest_path),
                    "file_sha256": self.development.manifest_sha256,
                }
            ),
            "mode": self.request.mode.value,
            "pilot_stage": self.request.pilot_stage,
            "condition": self.request.condition.value,
            "grouping_mode": self.request.condition.grouping_mode.value,
            "seed": self.request.seed,
            "authorization_node": self.authorization.node,
            "r03_qualification_identity": (
                self.authorization.r03_qualification_identity
            ),
            "r04_training_qualification_identity": (
                self.authorization.r04_training_qualification_identity
            ),
            "r05_freeze_identity": self.authorization.r05_freeze_identity,
            "g2_verdict_identity": self.authorization.g2_verdict_identity,
            "initial_model_state_sha256": self.initial_model_state_sha256,
            "config": asdict(self.config),
            "bank_entry_count": len(self.bank),
            "bank_plan_prefix_count": self.bank.plan_prefix_count,
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
            "development_trace": self.development_trace,
            "model_state_sha256": model_state_sha256(self.model.state_dict()),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": _numpy_rng_record(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
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
            "training_system_identity": protocol_training_system_identity(
                self.protocol
            ),
            "bank_protocol_identity": self.bank.protocol_identity,
            "bank_identity": self.bank.bank_identity,
            "renderer_identity": self.bank.renderer_identity,
            "stu_identity": self.bank.stu_identity,
            "development_population_identity": (
                None if self.development is None else self.development.identity
            ),
            "development_manifest": (
                None
                if self.development is None
                else {
                    "path": str(self.development.manifest_path),
                    "file_sha256": self.development.manifest_sha256,
                }
            ),
            "mode": self.request.mode.value,
            "pilot_stage": self.request.pilot_stage,
            "condition": self.request.condition.value,
            "grouping_mode": self.request.condition.grouping_mode.value,
            "seed": self.request.seed,
            "authorization_node": self.authorization.node,
            "r03_qualification_identity": (
                self.authorization.r03_qualification_identity
            ),
            "r04_training_qualification_identity": (
                self.authorization.r04_training_qualification_identity
            ),
            "r05_freeze_identity": self.authorization.r05_freeze_identity,
            "g2_verdict_identity": self.authorization.g2_verdict_identity,
            "initial_model_state_sha256": self.initial_model_state_sha256,
            "config": asdict(self.config),
            "bank_entry_count": len(self.bank),
            "bank_plan_prefix_count": self.bank.plan_prefix_count,
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
        self.next_entry = _require_int(payload["next_entry"], "checkpoint next_entry")
        self.updates = _require_int(payload["updates"], "checkpoint updates")
        self.windows_seen = _require_int(
            payload["windows_seen"], "checkpoint windows_seen"
        )
        self.last_loss = float(payload["last_loss"])
        best = payload["best_development_ap"]
        self.best_development_ap = None if best is None else float(best)
        development = payload["last_development"]
        self.last_development = None if development is None else dict(development)
        trace = payload.get("development_trace")
        if not isinstance(trace, list) or any(
            not isinstance(item, Mapping) for item in trace
        ):
            raise TrainingError("checkpoint development trace is invalid")
        self.development_trace = [dict(item) for item in trace]
        self.last_evaluated_update = (
            -1
            if not self.development_trace
            else int(self.development_trace[-1]["update"])
        )
        if payload.get("model_state_sha256") != model_state_sha256(
            self.model.state_dict()
        ):
            raise TrainingError("checkpoint model-state identity changed")
        _restore_rng(payload, self.device)
        if self.next_entry >= self.request.window_count:
            raise TrainingError("checkpoint cursor lies outside the selected bank")

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
        record = _development_record(evidence, self.request.condition, self.development)
        record["metric_evidence"] = _save_development_metric_evidence(
            self.output_directory, evidence, self.development.identity
        )
        score = float(record["macro_fused_point_ap"])
        improved = self.best_development_ap is None or score > self.best_development_ap
        self.last_development = record
        self.last_evaluated_update = self.updates
        self.development_trace.append(
            {
                "update": self.updates,
                "macro_fused_point_ap": score,
                "probability_saturation_fraction": _require_number(
                    record["probability_saturation_fraction"],
                    "development probability saturation fraction",
                ),
                "development_record_identity": _identity_digest(record),
            }
        )
        if improved:
            self.best_development_ap = score
            # Strict improvement makes exact ties retain the earliest checkpoint.
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
                    and self.updates % self.config.evaluation_interval_updates == 0
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
        if self.development is not None and self.last_evaluated_update != self.updates:
            self._evaluate_development()
        self._save_last()
        status = (
            "passed"
            if tiny_passed
            else "failed"
            if self.request.mode is TrainMode.TINY_OVERFIT
            else "completed"
        )
        last_checkpoint_sha256 = _sha256_file(self.last_path)
        last_model_state_sha256 = model_state_sha256(self.model.state_dict())
        best_checkpoint_sha256: str | None = None
        best_model_state_sha256: str | None = None
        best_evaluated_update: int | None = None
        best_development: dict[str, object] | None = None
        if self.development is not None:
            if not self.best_path.is_file() or not self.development_trace:
                raise TrainingError(
                    "development training did not produce a best checkpoint"
                )
            try:
                best_payload = torch.load(
                    self.best_path, map_location="cpu", weights_only=True
                )
            except Exception as error:
                raise TrainingError("cannot safely load best.pt") from error
            if not isinstance(best_payload, Mapping):
                raise TrainingError("best.pt is not a training checkpoint")
            best_evaluated_update = _require_int(
                best_payload.get("updates"), "best checkpoint update", minimum=1
            )
            best_record = best_payload.get("last_development")
            if not isinstance(best_record, Mapping):
                raise TrainingError("best checkpoint lacks its development record")
            best_development = dict(best_record)
            scores = [
                _require_number(
                    item.get("macro_fused_point_ap"),
                    "development trace macro_fused_point_ap",
                )
                for item in self.development_trace
            ]
            maximum_score = max(scores)
            earliest_best_update = min(
                _require_int(item.get("update"), "development trace update", minimum=1)
                for item, score in zip(self.development_trace, scores, strict=True)
                if score == maximum_score
            )
            if (
                best_evaluated_update != earliest_best_update
                or _require_number(
                    best_development.get("macro_fused_point_ap"),
                    "best development macro_fused_point_ap",
                )
                != maximum_score
            ):
                raise TrainingError(
                    "best checkpoint violates strict-improvement earliest-tie selection"
                )
            best_model_state_sha256 = _require_digest(
                best_payload.get("model_state_sha256"),
                "best checkpoint model-state identity",
            )
            if best_model_state_sha256 != model_state_sha256(
                _require_mapping(
                    best_payload.get("model_state_dict"),
                    "best checkpoint model state",
                )
            ):
                raise TrainingError("best checkpoint model-state identity changed")
            best_checkpoint_sha256 = _sha256_file(self.best_path)
        result: dict[str, object] = {
            "format": TRAIN_RESULT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "mode": self.request.mode.value,
            "pilot_stage": self.request.pilot_stage,
            "condition": self.request.condition.value,
            "grouping_mode": self.request.condition.grouping_mode.value,
            "seed": self.request.seed,
            "authorization_node": self.authorization.node,
            "r03_qualification_identity": (
                self.authorization.r03_qualification_identity
            ),
            "r04_training_qualification_identity": (
                self.authorization.r04_training_qualification_identity
            ),
            "r05_freeze_identity": self.authorization.r05_freeze_identity,
            "g2_verdict_identity": self.authorization.g2_verdict_identity,
            "config": asdict(self.config),
            "protocol_identity": self.protocol_identity,
            "training_system_identity": protocol_training_system_identity(
                self.protocol
            ),
            "bank_protocol_identity": self.bank.protocol_identity,
            "bank_identity": self.bank.bank_identity,
            "bank_entry_count": len(self.bank),
            "bank_plan_prefix_count": self.bank.plan_prefix_count,
            "initial_model_state_sha256": self.initial_model_state_sha256,
            "window_count": self.request.window_count,
            "maximum_updates": self.request.maximum_updates,
            "completed_epochs": self.epoch,
            "optimizer_updates": self.updates,
            "windows_seen": self.windows_seen,
            "last_effective_batch_loss": self.last_loss,
            "last_checkpoint_sha256": last_checkpoint_sha256,
            "last_model_state_sha256": last_model_state_sha256,
            "checkpoint_selection_metric": "macro_fused_point_ap",
            "best_development_macro_fused_point_ap": (self.best_development_ap),
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "best_model_state_sha256": best_model_state_sha256,
            "best_evaluated_update": best_evaluated_update,
            "best_development": best_development,
            "development_trace": self.development_trace,
            "development_population_identity": (
                None if self.development is None else self.development.identity
            ),
            "development_manifest": (
                None
                if self.development is None
                else {
                    "path": str(self.development.manifest_path),
                    "file_sha256": self.development.manifest_sha256,
                }
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
        result["record_sha256"] = _identity_digest(result)
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
    pilot_stage: str | None = None,
    r03_qualification_path: Path | str | None = None,
    r04_qualification_path: Path | str | None = None,
    g2_verdict_path: Path | str | None = None,
    development_worlds_path: Path | str | None = None,
    rendered_development_clips: Iterable[DevelopmentClipWorld] | None = None,
    data_root: Path | str | None = None,
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
        TrainMode.FORMAL: 445,
    }[selected_mode]
    validate_training_request(
        protocol,
        selected_mode,
        selected_condition,
        seed,
        config,
        bank_size=provisional_bank_size,
        pilot_stage=pilot_stage,
    )
    r03_identity: str
    r04_identity: str | None = None
    g2_identity: str | None = None
    if selected_mode in {TrainMode.TINY_OVERFIT, TrainMode.PILOT}:
        if r03_qualification_path is None:
            raise TrainingError("R04 training requires the bound R03 qualification")
        try:
            from .qualify import QualificationError, R03QualificationRecord
        except ImportError:
            from qualify import QualificationError, R03QualificationRecord

        try:
            r03_record = R03QualificationRecord.load(
                r03_qualification_path, protocol=protocol
            )
        except QualificationError as error:
            raise TrainingError(
                "R03 qualification does not authorize R04 training"
            ) from error
        if protocol.status["r03_qualification_identity"] != r03_record.record_sha256:
            raise TrainingError("R03 qualification is not bound by the protocol")
        r03_identity = r03_record.record_sha256
        if r04_qualification_path is not None:
            raise TrainingError(
                "R04 qualification runs cannot consume their own verdict"
            )
        if g2_verdict_path is not None:
            raise TrainingError("R04 qualification runs cannot consume a G2 verdict")
    else:
        if r03_qualification_path is not None:
            raise TrainingError(
                "formal training is authorized by the completed R04 record"
            )
        if r04_qualification_path is None:
            raise TrainingError("formal training requires the bound R04 qualification")
        r04_record = R04TrainingQualificationRecord.load(
            r04_qualification_path, protocol=protocol
        )
        r04_identity = r04_record.record_sha256
        if r04_identity != protocol.status["r04_training_qualification_identity"]:
            raise TrainingError("R04 qualification is not bound by the protocol")
        r03_identity = _require_digest(
            protocol.status["r03_qualification_identity"],
            "status.r03_qualification_identity",
        )
        if protocol.status["current_node"] == "G3":
            if g2_verdict_path is None:
                raise TrainingError("G3 training requires the passed bound G2 verdict")
            try:
                g2_record = FormalGateVerdictRecord.load(
                    g2_verdict_path, protocol=protocol
                )
            except EvaluationError as error:
                raise TrainingError(
                    "G2 verdict does not authorize G3 training"
                ) from error
            g2_identity = g2_record.record_sha256
            if (
                g2_record.gate != "G2"
                or g2_record.decision != "pass"
                or g2_identity != protocol.status["g2_verdict_identity"]
            ):
                raise TrainingError("G3 training requires the passed bound G2 verdict")
        elif g2_verdict_path is not None:
            raise TrainingError("G2 training cannot consume an unfinished G2 verdict")
    if rendered_development_clips is None and development_worlds_path is not None:
        if data_root is None:
            raise TrainingError(
                "pilot and formal CLI use requires --data-root to rebuild development clips"
            )
        rendered_development_clips = render_frozen_development_clips(
            development_worlds_path,
            data_root=data_root,
            protocol=protocol,
        )
    development_supplied = (
        development_worlds_path is not None and rendered_development_clips is not None
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
    if bank.test_fixture:
        raise TrainingError("training cannot consume a private test-fixture bank")
    frozen_bank_identity = _require_digest(
        protocol.status.get("r04_training_bank_identity"),
        "status.r04_training_bank_identity",
    )
    if bank.bank_identity != frozen_bank_identity:
        raise TrainingError("training bank differs from the frozen R04 identity")
    if (
        selected_mode is TrainMode.FORMAL
        and bank.bank_identity != r04_record.bank_identity
    ):
        raise TrainingError("formal bank differs from the qualified R04 bank")
    if len(bank) != 445 or bank.plan_prefix_count != 445:
        raise TrainingError(
            "all training modes require the complete 445-window R04 bank"
        )
    request = validate_training_request(
        protocol,
        selected_mode,
        selected_condition,
        seed,
        config,
        bank_size=len(bank),
        pilot_stage=pilot_stage,
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
    if selected_mode is TrainMode.FORMAL:
        formal = _require_mapping(protocol.training["formal"], "training.formal")
        if formal["bank_identity"] != bank.bank_identity:
            raise TrainingError("formal bank differs from the frozen R05 identity")
        if development is None or (
            formal["development_population_identity"] != development.identity
        ):
            raise TrainingError(
                "formal development population differs from the frozen R05 identity"
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
        authorization=_TrainingAuthorization(
            _TRAINING_AUTHORIZATION_TOKEN,
            str(protocol.status["current_node"]),
            r03_identity,
            r04_identity,
            (
                _require_digest(
                    protocol.status["r05_freeze_identity"],
                    "status.r05_freeze_identity",
                )
                if selected_mode is TrainMode.FORMAL
                else None
            ),
            g2_identity,
        ),
        resume=resume,
    )
    return trainer.fit()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, adjudicate, or train the sole schema-31 AJAE route."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def protocol_argument(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--protocol", type=Path, default=PROJECT_ROOT / "protocol.json"
        )

    bank_command = commands.add_parser(
        "generate-bank", help="generate a frozen 8/128/445 training-bank prefix"
    )
    protocol_argument(bank_command)
    bank_command.add_argument("--data-root", required=True, type=Path)
    bank_command.add_argument("--output", required=True, type=Path)
    bank_command.add_argument("--entry-count", required=True, type=int)
    bank_command.add_argument("--device", required=True)

    development_command = commands.add_parser(
        "generate-development", help="generate the formal 24-clip R02 population"
    )
    protocol_argument(development_command)
    development_command.add_argument("--data-root", required=True, type=Path)
    development_command.add_argument("--output", required=True, type=Path)

    r02_command = commands.add_parser(
        "finalize-r02", help="recompute and irreversibly record the R02 verdict"
    )
    protocol_argument(r02_command)
    r02_command.add_argument("--development-worlds", required=True, type=Path)
    r02_command.add_argument("--data-root", required=True, type=Path)
    r02_command.add_argument("--visual-review", required=True, type=Path)

    r04_command = commands.add_parser(
        "finalize-r04", help="recompute the R04 training qualification verdict"
    )
    protocol_argument(r04_command)
    r04_command.add_argument("--evidence", required=True, type=Path)
    r04_command.add_argument("--output", required=True, type=Path)

    train_command = commands.add_parser(
        "train", help="run one authorized tiny-overfit, pilot, or formal job"
    )
    protocol_argument(train_command)
    train_command.add_argument(
        "--mode", required=True, choices=[item.value for item in TrainMode]
    )
    train_command.add_argument("--condition", required=True, choices=["B1", "B2", "B3"])
    train_command.add_argument("--bank", required=True, type=Path)
    train_command.add_argument("--seed", required=True, type=int)
    train_command.add_argument("--device", required=True)
    train_command.add_argument("--output", required=True, type=Path)
    train_command.add_argument("--learning-rate", required=True, type=float)
    train_command.add_argument("--gradient-accumulation", required=True, type=int)
    train_command.add_argument("--weight-decay", type=float, default=0.0)
    train_command.add_argument(
        "--scheduler",
        choices=("constant", "five_percent_warmup_cosine"),
        default="constant",
    )
    train_command.add_argument("--epochs", type=int, default=1)
    train_command.add_argument("--maximum-updates", type=int)
    train_command.add_argument("--evaluation-interval-updates", type=int, default=1)
    train_command.add_argument("--gradient-clip-norm", type=float)
    train_command.add_argument(
        "--pilot-stage",
        choices=("learning_rate_and_batch", "scheduler", "weight_decay"),
    )
    train_command.add_argument("--r03-qualification", type=Path)
    train_command.add_argument("--r04-qualification", type=Path)
    train_command.add_argument("--g2-verdict", type=Path)
    train_command.add_argument("--development-worlds", type=Path)
    train_command.add_argument("--data-root", type=Path)
    train_command.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.command == "generate-bank":
        bank = generate_window_train_bank(
            protocol_path=args.protocol,
            data_root=args.data_root,
            destination=args.output,
            entry_count=args.entry_count,
            device=args.device,
        )
        result: object = {
            "format": "ajae-schema31-bank-generation-summary-v1",
            "bank_identity": bank.bank_identity,
            "entry_count": len(bank),
            "plan_identity": bank.plan_identity,
            "test_fixture": bank.test_fixture,
        }
    elif args.command == "generate-development":
        development = generate_development_worlds(
            protocol_path=args.protocol,
            data_root=args.data_root,
            destination=args.output,
        )
        result = {
            "format": "ajae-schema31-development-generation-summary-v1",
            "status": development.status,
            "clip_count": len(development.clips),
            "plan_identity": development.plan_identity,
            "population_identity": development.population_identity,
        }
    elif args.command == "finalize-r02":
        development = finalize_r02_development_worlds(
            protocol_path=args.protocol,
            development_worlds_path=args.development_worlds,
            data_root=args.data_root,
            visual_review_path=args.visual_review,
        )
        result = {
            "format": "ajae-schema31-r02-finalization-summary-v1",
            "status": development.status,
            "population_identity": development.population_identity,
            "validation": dict(development.validation),
        }
    elif args.command == "finalize-r04":
        qualification = finalize_r04_training_qualification(
            protocol_path=args.protocol,
            evidence_path=args.evidence,
            destination=args.output,
        )
        result = {
            "format": "ajae-schema31-r04-finalization-summary-v1",
            "record_sha256": qualification.record_sha256,
            "bank_identity": qualification.bank_identity,
            "frozen_stage_winners": dict(qualification.winner),
        }
    else:
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
            pilot_stage=args.pilot_stage,
            r03_qualification_path=args.r03_qualification,
            r04_qualification_path=args.r04_qualification,
            g2_verdict_path=args.g2_verdict,
            development_worlds_path=args.development_worlds,
            data_root=args.data_root,
            resume=args.resume,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
