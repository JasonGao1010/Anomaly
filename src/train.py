#!/usr/bin/env python3
"""Train AJAE with the frozen four-world protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_FORMAT = "ajae-training-v2"
PROGRESS_FORMAT = "ajae-progress-v3"
MODEL_FORMAT = "ajae-model-v3"
RELATIVE_TIMES = (-2, -1, 0, 1, 2)
CAUSAL_TIMES = (-4, -3, -2, -1, 0)
WORLD_TYPES = ("pure_normal", "control_only", "mixed", "anomaly_only")
GATE1_EVIDENCE_KEYS = {
    "ray_slot_audit",
    "range_image_round_trip",
    "render_source_leakage",
    "beam_range_intensity",
}


class TrainingError(RuntimeError):
    """Report an invalid or failed AJAE optimization operation."""


def _seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_tensor(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise TrainingError(f"{name} contains non-finite values")


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_parent(path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_parent(path)


def _json_plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_plain(item) for item in value]
    return value


def _plain_json_object(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            _json_plain(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TrainingError(f"{name} must be a finite JSON object") from error
    if not isinstance(result, dict):
        raise TrainingError(f"{name} must be a JSON object")
    return result


def _canonical_json_object(name: str, value: Mapping[str, Any]) -> str:
    plain = _plain_json_object(name, value)
    return json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qualified_class(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _qualified_callable(value: Callable[..., object] | None) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


@dataclass(frozen=True, slots=True)
class ExperimentCondition:
    """One immutable B0--B5 scientific comparison condition."""

    name: str
    trainable: bool
    frame_offsets: tuple[int, ...]
    model_times: tuple[int, ...]
    cross_frame_enabled: bool
    supervised_times: tuple[int, ...]
    prediction_rule: str
    weights_from: str | None = None

    def __post_init__(self) -> None:
        if self.name not in {f"B{index}" for index in range(6)}:
            raise ValueError("condition name must be B0 through B5")
        if not self.frame_offsets or len(set(self.frame_offsets)) != len(
            self.frame_offsets
        ):
            raise ValueError("condition frame offsets must be non-empty and unique")
        if len(self.model_times) != len(self.frame_offsets) or len(
            set(self.model_times)
        ) != len(self.model_times):
            raise ValueError("model times must uniquely encode every physical frame")
        if not set(self.supervised_times).issubset(self.model_times):
            raise ValueError("supervised times must be present in the model input")
        if self.weights_from is not None and self.trainable:
            raise ValueError("a shared-weight condition cannot be trained independently")

    @property
    def frames(self) -> int:
        return len(self.frame_offsets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trainable": self.trainable,
            "frame_offsets": list(self.frame_offsets),
            "model_times": list(self.model_times),
            "cross_frame_enabled": self.cross_frame_enabled,
            "supervised_times": list(self.supervised_times),
            "prediction_rule": self.prediction_rule,
            "weights_from": self.weights_from,
        }


CONDITIONS: Mapping[str, ExperimentCondition] = MappingProxyType(
    {
        "B0": ExperimentCondition(
            "B0", False, (0,), (0,), False, (), "official_frozen_STU_MaxLogit"
        ),
        "B1": ExperimentCondition(
            "B1", True, (0,), (0,), False, (0,), "single_frame_point_prediction"
        ),
        "B2": ExperimentCondition(
            "B2",
            True,
            RELATIVE_TIMES,
            RELATIVE_TIMES,
            False,
            RELATIVE_TIMES,
            "center_q0_without_cross_frame_edges",
        ),
        "B3": ExperimentCondition(
            "B3",
            True,
            RELATIVE_TIMES,
            RELATIVE_TIMES,
            True,
            RELATIVE_TIMES,
            "center_q0_with_cross_frame_edges",
        ),
        "B4": ExperimentCondition(
            "B4",
            False,
            RELATIVE_TIMES,
            RELATIVE_TIMES,
            True,
            (),
            "equal_probability_mean_by_frame_ray",
            weights_from="B3",
        ),
        "B5": ExperimentCondition(
            "B5",
            True,
            CAUSAL_TIMES,
            RELATIVE_TIMES,
            True,
            RELATIVE_TIMES,
            "causal_current_frame_at_model_position_plus2",
        ),
    }
)


def experiment_condition(name: str) -> ExperimentCondition:
    try:
        return CONDITIONS[name]
    except KeyError as error:
        raise TrainingError("condition must be one of B0, B1, B2, B3, B4, B5") from error


def _world_cycle(probabilities: object) -> tuple[str, ...]:
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(WORLD_TYPES):
        raise TrainingError("training world probabilities must define all four types")
    values: list[float] = []
    fractions: list[Fraction] = []
    for kind in WORLD_TYPES:
        value = probabilities[kind]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingError(f"training world probability {kind} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise TrainingError(f"training world probability {kind} must be positive")
        values.append(number)
        fractions.append(Fraction(str(number)).limit_denominator(1000))
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise TrainingError("training world probabilities must sum to one")
    denominator = math.lcm(*(value.denominator for value in fractions))
    counts = [value.numerator * denominator // value.denominator for value in fractions]
    divisor = math.gcd(*counts)
    counts = [value // divisor for value in counts]
    if any(value < 1 for value in counts) or sum(counts) > 4096:
        raise TrainingError("training world probabilities need a simpler finite cycle")

    # Start with every world type, then realize any remaining frozen weight.
    cycle: list[str] = list(WORLD_TYPES)
    counts = [value - 1 for value in counts]
    total = sum(counts)
    if total == 0:
        return tuple(cycle)
    scores = [0] * len(WORLD_TYPES)
    for _ in range(total):
        scores = [score + count for score, count in zip(scores, counts, strict=True)]
        chosen = max(range(len(scores)), key=scores.__getitem__)
        scores[chosen] -= total
        cycle.append(WORLD_TYPES[chosen])
    return tuple(cycle)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Numerical choices read from the authoritative protocol."""

    seeds: tuple[int, ...] = (0, 1, 2)
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    micro_batch: int = 1
    gradient_accumulation: int = 8
    worlds_per_evaluation: int = 5
    patience: int = 4
    chunk_centers: int = 16
    cache_frames: int = 7
    world_type_cycle: tuple[str, ...] = WORLD_TYPES
    output_dir: str = "runs/ajae"

    def __post_init__(self) -> None:
        if (
            len(self.seeds) < 3
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
        ):
            raise ValueError("formal development requires at least three unique seeds")
        integers = {
            "micro_batch": (self.micro_batch, 1),
            "gradient_accumulation": (self.gradient_accumulation, 1),
            "worlds_per_evaluation": (self.worlds_per_evaluation, 1),
            "patience": (self.patience, 1),
            "chunk_centers": (self.chunk_centers, 1),
            "cache_frames": (self.cache_frames, 5),
        }
        for name, (value, minimum) in integers.items():
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.micro_batch != 1:
            raise ValueError("the scientific protocol fixes micro_batch=1")
        cycle = tuple(self.world_type_cycle)
        if not cycle or any(kind not in WORLD_TYPES for kind in cycle):
            raise ValueError("world_type_cycle contains an unknown world type")
        if not set(WORLD_TYPES).issubset(cycle):
            raise ValueError("every world type must have positive training proportion")
        output = Path(self.output_dir)
        if output.is_absolute() or ".." in output.parts:
            raise ValueError("output_dir must be a project-relative path")

    @classmethod
    def from_protocol(cls, protocol: object) -> TrainConfig:
        values = getattr(protocol, "training")
        if not isinstance(values, Mapping):
            try:
                values = asdict(values)
            except TypeError as error:
                raise TrainingError("protocol.training is not a mapping") from error
        if "seeds" not in values:
            raise TrainingError("protocol.training must freeze at least three seeds")
        selected: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            if name in values:
                selected[name] = values[name]
        selected["seeds"] = tuple(values["seeds"])
        if "world_type_probabilities" in values:
            selected["world_type_cycle"] = _world_cycle(
                values["world_type_probabilities"]
            )
        elif "world_type_cycle" in values:
            selected["world_type_cycle"] = tuple(values["world_type_cycle"])
        else:
            raise TrainingError("protocol.training must freeze world type proportions")
        return cls(**selected)


_FORMAL_PREFLIGHT_SEAL = object()


@dataclass(frozen=True, slots=True)
class FormalPreflightProof:
    """Opaque evidence that the exact formal inputs passed preflight."""

    _seal: object
    _protocol_json: str
    _development_json: str
    _selection_json: str
    _config_json: str

    @property
    def protocol_document(self) -> dict[str, Any]:
        return json.loads(self._protocol_json)

    @property
    def checkpoint_selection(self) -> dict[str, Any]:
        return json.loads(self._selection_json)

    @property
    def development_document_sha256(self) -> str:
        return hashlib.sha256(self._development_json.encode("utf-8")).hexdigest()

    @property
    def maximum_worlds(self) -> int:
        training = self.protocol_document.get("training")
        value = training.get("maximum_worlds") if isinstance(training, Mapping) else None
        if type(value) is not int:
            raise TrainingError(
                "protocol.training.maximum_worlds is not frozen before training"
            )
        return value


def _require_preflight_proof(
    proof: FormalPreflightProof,
    config: TrainConfig,
    *,
    protocol: object | None = None,
) -> None:
    if (
        type(proof) is not FormalPreflightProof
        or proof._seal is not _FORMAL_PREFLIGHT_SEAL
    ):
        raise TrainingError("formal training requires a valid preflight proof")
    if proof._config_json != _canonical_json_object(
        "training config", asdict(config)
    ):
        raise TrainingError("preflight proof uses a different training config")
    selection = proof.checkpoint_selection
    _checkpoint_selection_tolerance(selection)
    if proof.maximum_worlds < len(config.world_type_cycle):
        raise TrainingError(
            "protocol.training.maximum_worlds does not cover the world-type cycle"
        )
    protocol_development = proof.protocol_document.get("development")
    if (
        not isinstance(protocol_development, Mapping)
        or protocol_development.get("checkpoint_selection") != selection
    ):
        raise TrainingError("preflight proof contains inconsistent selection inputs")
    if protocol is not None:
        converter = getattr(protocol, "plain_document", None)
        if not callable(converter):
            raise TrainingError("protocol must expose plain_document()")
        document = converter()
        if not isinstance(document, Mapping) or proof._protocol_json != (
            _canonical_json_object("protocol", document)
        ):
            raise TrainingError("preflight proof uses a different protocol")


@dataclass(frozen=True, slots=True)
class DevelopmentWorldMetrics:
    """Metrics from exactly one in-generator development world."""

    world_id: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if type(self.world_id) is not int or self.world_id < 0:
            raise ValueError("development world_id must be a non-negative integer")
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("development world metrics must be a non-empty mapping")
        converted: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("development metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("development metrics must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("development metrics must be finite")
            converted[name] = number
        object.__setattr__(self, "metrics", MappingProxyType(converted))

    def to_dict(self) -> dict[str, Any]:
        return {"world_id": self.world_id, "metrics": dict(self.metrics)}


@dataclass(frozen=True, slots=True)
class DevelopmentEvidence:
    """Selection evidence that structurally excludes held-out worlds."""

    in_generator: tuple[DevelopmentWorldMetrics, ...]
    pure_normal: Mapping[str, float]

    def __post_init__(self) -> None:
        worlds = tuple(self.in_generator)
        if len(worlds) != 24 or any(
            not isinstance(item, DevelopmentWorldMetrics) for item in worlds
        ):
            raise ValueError("checkpoint selection requires exactly 24 world metrics")
        identifiers = [item.world_id for item in worlds]
        if tuple(sorted(identifiers)) != tuple(range(24)):
            raise ValueError("in-generator development world IDs must be exactly 0..23")
        if not isinstance(self.pure_normal, Mapping) or not self.pure_normal:
            raise ValueError("pure-normal statistics must be a non-empty mapping")
        normal: dict[str, float] = {}
        for name, value in self.pure_normal.items():
            if not isinstance(name, str) or not name:
                raise ValueError("pure-normal statistic names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("pure-normal statistics must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("pure-normal statistics must be finite")
            normal[name] = number
        object.__setattr__(self, "in_generator", worlds)
        object.__setattr__(self, "pure_normal", MappingProxyType(normal))

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_generator": [item.to_dict() for item in self.in_generator],
            "pure_normal": dict(self.pure_normal),
        }


def _checkpoint_selection_tolerance(rule: Mapping[str, Any]) -> float:
    expected = {
        "status": "frozen_before_training",
        "primary": "maximum macro mean of per-world AP over the 24 in-generator worlds",
        "first_tie_break": "lower pure-normal score q99.9",
        "second_tie_break": "earlier completed world index",
        "held_out_input_forbidden": True,
    }
    if set(rule) != {*expected, "tie_tolerance"} or any(
        rule.get(name) != value for name, value in expected.items()
    ):
        raise TrainingError("checkpoint selection rule is not the frozen schema-30 rule")
    tolerance = rule.get("tie_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TrainingError("checkpoint-selection tolerance must be numeric")
    result = float(tolerance)
    if not math.isfinite(result) or result <= 0.0:
        raise TrainingError("checkpoint-selection tolerance must be positive")
    return result


def checkpoint_selection_key(
    rule: Mapping[str, Any], evidence: DevelopmentEvidence
) -> tuple[float, float]:
    """Return the frozen AP/safety key; an exact tie keeps the earlier world."""

    _checkpoint_selection_tolerance(rule)
    ap = []
    for world in sorted(evidence.in_generator, key=lambda item: item.world_id):
        if "AP" not in world.metrics:
            raise TrainingError(f"development world {world.world_id} lacks AP")
        ap.append(float(world.metrics["AP"]))
    if "q99.9" not in evidence.pure_normal:
        raise TrainingError("pure-normal evidence lacks score q99.9")
    return float(np.mean(ap)), -float(evidence.pure_normal["q99.9"])


def balanced_bce_loss(logits: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
    """Average each present class, then give present classes equal weight."""

    if logits.ndim != 1 or targets.shape != logits.shape or valid.shape != logits.shape:
        raise TrainingError("logits, targets, and validity must be aligned vectors")
    if targets.dtype != torch.bool or valid.dtype != torch.bool:
        raise TypeError("targets and valid must be boolean")
    losses = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    terms = [
        losses[mask].mean()
        for mask in (valid & targets, valid & ~targets)
        if bool(mask.any())
    ]
    if not terms:
        raise TrainingError("window has no valid training point")
    return torch.stack(terms).mean()


@dataclass(frozen=True, slots=True)
class FrameCacheKey:
    """Bind one cached frame to every scientific input that can change it."""

    world_identity: str
    frame_identity: str
    renderer_generator_identity: str
    stu_identity: str

    def __post_init__(self) -> None:
        for name, value in (
            ("world_identity", self.world_identity),
            ("frame_identity", self.frame_identity),
            ("renderer_generator_identity", self.renderer_generator_identity),
            ("stu_identity", self.stu_identity),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


class FrameCache:
    """Bound rendered frames and frozen STU outputs to a small time block."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("frame cache capacity must be positive")
        self.capacity = capacity
        self.rendered: OrderedDict[FrameCacheKey, object] = OrderedDict()
        self.encoded: OrderedDict[FrameCacheKey, object] = OrderedDict()

    @staticmethod
    def _get(
        cache: OrderedDict[FrameCacheKey, object],
        key: FrameCacheKey,
        factory: Callable[[], object],
        capacity: int,
    ) -> object:
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        value = factory()
        cache[key] = value
        while len(cache) > capacity:
            cache.popitem(last=False)
        return value

    def rendered_frame(self, key: FrameCacheKey, factory: Callable[[], object]) -> object:
        return self._get(self.rendered, key, factory, self.capacity)

    def encoded_frame(self, key: FrameCacheKey, factory: Callable[[], object]) -> object:
        return self._get(self.encoded, key, factory, self.capacity)

    def clear(self) -> None:
        self.rendered.clear()
        self.encoded.clear()


@dataclass(slots=True)
class WindowTrainingData:
    coordinates: Tensor
    relative_times: Tensor
    stu_features: Tensor
    normal_evidence: Tensor
    assignment_reliability: Tensor
    no_object_reliability: Tensor
    intensity: Tensor
    targets: Tensor
    valid: Tensor


def _numpy_slots(value: object) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64)


def make_window_training_data(
    window: object,
    rendered: Sequence[object],
    encoded: Sequence[object],
    condition: ExperimentCondition,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    device: torch.device,
) -> WindowTrainingData:
    """Align every visible return, frozen STU field, and point target."""

    if len(rendered) != condition.frames or len(encoded) != condition.frames:
        raise TrainingError("rendered and encoded frames do not match the condition")
    feature_parts: list[Tensor] = []
    evidence_parts: list[Tensor] = []
    assignment_parts: list[Tensor] = []
    no_object_parts: list[Tensor] = []
    intensity_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    expected_points = 0
    for frame_view, encoding in zip(rendered, encoded, strict=True):
        source = getattr(frame_view, "source")
        slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
        count = slots.size
        expected_points += count
        if not np.array_equal(_numpy_slots(getattr(encoding, "real_slots")), slots):
            raise TrainingError("frozen STU output changed the visible-return order")
        point_features = getattr(encoding, "point_features")
        normal_evidence = getattr(encoding, "normal_evidence")
        assignment = getattr(encoding, "reliability_assign")
        no_object = getattr(encoding, "reliability_noobj")
        if point_features.shape != (count, 128):
            raise TrainingError("frozen STU point features must be [N,128]")
        if normal_evidence.shape != (count, 19):
            raise TrainingError("frozen STU normal evidence must be [N,19]")
        if assignment.shape != (count,) or no_object.shape != (count,):
            raise TrainingError("frozen STU reliability fields must be [N]")
        feature_parts.append(point_features)
        evidence_parts.append(normal_evidence)
        assignment_parts.append(assignment)
        no_object_parts.append(no_object)
        intensity_parts.append(np.asarray(source.xyzi[slots, 3], dtype=np.float32))

        packed = np.asarray(getattr(frame_view, "packed_labels"), dtype=np.uint32)
        normal_control_full = np.asarray(
            getattr(frame_view, "normal_control_mask"), dtype=np.bool_
        )
        anomaly_proxy_full = np.asarray(
            getattr(frame_view, "anomaly_proxy_mask"), dtype=np.bool_
        )
        slot_count = int(source.xyzi.shape[0])
        if (
            packed.shape != (slot_count,)
            or normal_control_full.shape != (slot_count,)
            or anomaly_proxy_full.shape != (slot_count,)
        ):
            raise TrainingError("render labels and generation masks must align with slots")
        normal_control = normal_control_full[slots]
        anomaly_proxy = anomaly_proxy_full[slots]
        if np.any(normal_control & anomaly_proxy):
            raise TrainingError("one return cannot be both a normal control and anomaly proxy")
        raw_semantic = packed[slots] & np.uint32(0xFFFF)
        inserted = normal_control | anomaly_proxy
        if np.any((raw_semantic == np.uint32(2)) & ~inserted):
            raise TrainingError("train/206 contains an unexpected native anomaly return")
        distance = np.linalg.norm(source.xyzi[slots, :3], axis=1)
        valid = inserted | (raw_semantic != np.uint32(0))
        valid &= distance >= minimum_range_m
        valid &= distance <= maximum_range_m
        target_parts.append(anomaly_proxy)
        valid_parts.append(valid)

    window_points = getattr(window, "points")
    if getattr(window_points, "ray_mapping_audited", None) is not True:
        raise TrainingError("formal training requires an audited slot-to-ray mapping")
    # Scene arrays are intentionally read-only; own the storage before tensor use.
    coordinates_np = np.asarray(window_points.coordinates_center).copy()
    times_np = np.asarray(window_points.relative_time)
    if coordinates_np.shape != (expected_points, 3) or times_np.shape != (
        expected_points,
    ):
        raise TrainingError("window geometry is not aligned with its visible returns")
    observed_times = tuple(sorted(int(value) for value in np.unique(times_np)))
    if observed_times != tuple(sorted(condition.model_times)):
        raise TrainingError("window model times do not match the experiment condition")
    model_times = times_np.astype(np.int64, copy=False)
    return WindowTrainingData(
        coordinates=torch.as_tensor(coordinates_np, device=device, dtype=torch.float32),
        relative_times=torch.as_tensor(model_times, device=device, dtype=torch.long),
        stu_features=torch.cat(feature_parts).to(device),
        normal_evidence=torch.cat(evidence_parts).to(device),
        assignment_reliability=torch.cat(assignment_parts).to(device),
        no_object_reliability=torch.cat(no_object_parts).to(device),
        intensity=torch.as_tensor(np.concatenate(intensity_parts), device=device),
        targets=torch.as_tensor(np.concatenate(target_parts), device=device),
        valid=torch.as_tensor(np.concatenate(valid_parts), device=device),
    )


def shuffled_center_blocks(
    legal_centers: Sequence[int], block_size: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Shuffle contiguous blocks while preserving time order within each block."""

    centers = tuple(int(value) for value in legal_centers)
    if not centers or any(right != left + 1 for left, right in zip(centers, centers[1:])):
        raise ValueError("legal centers must be one consecutive range")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    blocks = [
        centers[start : start + block_size]
        for start in range(0, len(centers), block_size)
    ]
    random.Random(seed).shuffle(blocks)
    return tuple(blocks)


def _derived_seed(training_seed: int, world_index: int, stream: int) -> int:
    value = (
        (training_seed + 1) * 1_000_003
        + (world_index + 1) * 9_176
        + stream * 2_654_435_761
    )
    return int(value % (2**63 - 1))


class AJAETrainer:
    """Optimize one condition and one seed over complete immutable worlds."""

    def __init__(
        self,
        *,
        model: nn.Module,
        encoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None,
        condition: ExperimentCondition,
        seed: int,
        legal_centers: Sequence[int],
        source_frame: Callable[[int], object],
        render_frame: Callable[[object, object], object],
        assemble_window: Callable[
            [ExperimentCondition, int, Sequence[object]], object
        ],
        config: TrainConfig,
        evaluation_range: tuple[float, float],
        run_dir: Path | str,
        scientific_identity: Mapping[str, Any],
        checkpoint_selection_rule: Mapping[str, Any],
        ray_mapping_digest: str,
        development_evaluator: Callable[
            [nn.Module, int, int, ExperimentCondition], DevelopmentEvidence
        ]
        | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        if not condition.trainable:
            if condition.name == "B4":
                raise TrainingError("B4 must reuse B3 weights and cannot be trained")
            raise TrainingError(f"{condition.name} is a non-trainable reference condition")
        if seed not in config.seeds:
            raise TrainingError("trainer seed is not one of the frozen formal seeds")
        if config.cache_frames < condition.frames:
            raise TrainingError("frame cache cannot hold one complete condition window")
        self.model = model
        self.encoder = encoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.condition = condition
        self.seed = seed
        self.legal_centers = tuple(int(value) for value in legal_centers)
        if not self.legal_centers or any(
            right != left + 1
            for left, right in zip(self.legal_centers, self.legal_centers[1:])
        ):
            raise TrainingError("legal centers must be one non-empty consecutive range")
        self.source_frame = source_frame
        self.render_frame_callback = render_frame
        self.assemble_window_callback = assemble_window
        self.config = config
        self.minimum_range_m, self.maximum_range_m = evaluation_range
        if not (
            math.isclose(self.minimum_range_m, 2.5, abs_tol=1.0e-12)
            and math.isclose(self.maximum_range_m, 50.0, abs_tol=1.0e-12)
        ):
            raise TrainingError("the main loss range must be the inclusive 2.5--50 m domain")
        self.run_dir = Path(run_dir)
        if not isinstance(ray_mapping_digest, str) or len(ray_mapping_digest) != 64:
            raise TrainingError(
                "trainer requires the digest of the loaded ray calibration"
            )
        self.ray_mapping_digest = ray_mapping_digest
        self.development_evaluator = development_evaluator
        self.selector_identity = _qualified_callable(checkpoint_selection_key)
        self.evaluator_identity = _qualified_callable(development_evaluator)
        self.device = torch.device(device)
        self.cache = FrameCache(config.cache_frames)
        self.scientific_identity = _plain_json_object(
            "scientific_identity", scientific_identity
        )
        if self.scientific_identity.get("ray_mapping_digest") != ray_mapping_digest:
            raise TrainingError(
                "scientific identity does not bind the loaded ray calibration"
            )
        self.selection_rule = _plain_json_object(
            "checkpoint_selection_rule", checkpoint_selection_rule
        )
        _checkpoint_selection_tolerance(self.selection_rule)
        if self.scientific_identity.get("checkpoint_selection") != self.selection_rule:
            raise TrainingError("scientific identity does not contain the active selection rule")
        protocol_identity = self.scientific_identity.get("protocol")
        protocol_development = (
            protocol_identity.get("development")
            if isinstance(protocol_identity, Mapping)
            else None
        )
        if (
            not isinstance(protocol_development, Mapping)
            or protocol_development.get("checkpoint_selection") != self.selection_rule
        ):
            raise TrainingError("scientific identity does not contain the complete protocol")

        self.world_index = 0
        self.update_index = 0
        self.accumulated_windows = 0
        self.best_key: tuple[float, ...] | None = None
        self.best_world = -1
        self.stale_evaluations = 0
        self.best_state: dict[str, Tensor] | None = None
        self.maximum_primary: float | None = None
        self.selection_candidates: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.commit_id = 0
        self.loaded_progress = False
        self.maximum_worlds: int | None = None
        self.stop_reason: str | None = None
        self.phase = "between_worlds"
        self.resume_world = 0
        self.active_world: dict[str, Any] | None = None
        self.active_world_kind: str | None = None
        self.active_world_seed: int | None = None
        self.block_order: tuple[tuple[int, ...], ...] | None = None
        self.next_block = 0
        self.next_window = 0
        self.world_loss_sum = 0.0
        self.world_window_count = 0
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        training_source = self.scientific_identity.get("training_source")
        if not isinstance(training_source, Mapping):
            raise TrainingError("scientific identity lacks the training source")
        self.training_source_identity = str(training_source.get("content_sha256", ""))
        self.renderer_generator_identity = str(
            self.scientific_identity.get("renderer_generator_sha256", "")
        )
        self.stu_identity = str(self.scientific_identity.get("stu_identity_sha256", ""))
        for name, value in (
            ("training source", self.training_source_identity),
            ("renderer/generator", self.renderer_generator_identity),
            ("STU", self.stu_identity),
        ):
            if len(value) != 64:
                raise TrainingError(f"scientific identity lacks a valid {name} digest")

    @staticmethod
    def _world_payload(world: object) -> dict[str, Any]:
        converter = getattr(world, "to_dict", None)
        if converter is None:
            raise TrainingError("a resumable world must implement to_dict()")
        value = converter()
        if not isinstance(value, Mapping):
            raise TrainingError("WorldSpec.to_dict() must return a mapping")
        return _plain_json_object("world specification", value)

    @staticmethod
    def _world_from_payload(payload: Mapping[str, Any]) -> object:
        try:
            from .render import WorldSpec
        except ImportError:  # pragma: no cover - direct script execution
            from render import WorldSpec  # type: ignore[no-redef]
        world = WorldSpec.from_dict(payload)
        if AJAETrainer._world_payload(world) != dict(payload):
            raise TrainingError("saved WorldSpec does not round-trip exactly")
        return world

    @staticmethod
    def _require_world(world: object, expected: str, expected_seed: int) -> None:
        actual = getattr(world, "world_type", None)
        if actual != expected:
            raise TrainingError(
                f"world factory returned type {actual!r}, expected {expected!r}"
            )
        if getattr(world, "seed", None) != expected_seed:
            raise TrainingError("world factory did not use the requested independent seed")
        if getattr(world, "source_sequence_id", None) != 206:
            raise TrainingError("training worlds must use the sole train/206 source")

    def _cache_key(self, world: object, frame_id: int) -> FrameCacheKey:
        world_identity = hashlib.sha256(
            _canonical_json_object("world specification", self._world_payload(world)).encode(
                "utf-8"
            )
        ).hexdigest()
        frame_identity = hashlib.sha256(
            f"{self.training_source_identity}:train:206:{frame_id}".encode("ascii")
        ).hexdigest()
        return FrameCacheKey(
            world_identity,
            frame_identity,
            self.renderer_generator_identity,
            self.stu_identity,
        )

    def _render(self, world: object, frame_id: int) -> object:
        key = self._cache_key(world, frame_id)
        return self.cache.rendered_frame(
            key,
            lambda: self.render_frame_callback(self.source_frame(frame_id), world),
        )

    def _encode(self, world: object, frame_id: int) -> object:
        key = self._cache_key(world, frame_id)

        def factory() -> object:
            rendered = self._render(world, frame_id)
            source = getattr(rendered, "source")
            with torch.no_grad():
                return self.encoder(
                    source.coordinates,
                    source.features,
                    source.real_slots,
                )

        return self.cache.encoded_frame(key, factory)

    def _window_data(self, world: object, center: int) -> WindowTrainingData:
        frame_ids = tuple(center + offset for offset in self.condition.frame_offsets)
        rendered = tuple(self._render(world, frame_id) for frame_id in frame_ids)
        sources = tuple(getattr(value, "source") for value in rendered)
        window = self.assemble_window_callback(self.condition, center, sources)
        if (
            getattr(getattr(window, "points"), "ray_mapping_digest", None)
            != self.ray_mapping_digest
        ):
            raise TrainingError("training window does not use the bound ray calibration")
        encoded = tuple(self._encode(world, frame_id) for frame_id in frame_ids)
        return make_window_training_data(
            window,
            rendered,
            encoded,
            self.condition,
            minimum_range_m=self.minimum_range_m,
            maximum_range_m=self.maximum_range_m,
            device=self.device,
        )

    def _step_window(self, world: object, center: int) -> float:
        batch = self._window_data(world, center)
        logits = self.model(
            batch.coordinates,
            batch.relative_times,
            batch.stu_features,
            batch.normal_evidence,
            batch.assignment_reliability,
            batch.no_object_reliability,
            batch.intensity,
            cross_frame_enabled=self.condition.cross_frame_enabled,
        )
        if logits.shape != batch.targets.shape:
            raise TrainingError("AJAE must return one logit for every supplied point")
        _finite_tensor("AJAE logits", logits)
        supervised = torch.zeros_like(batch.valid)
        for relative_time in self.condition.supervised_times:
            supervised |= batch.relative_times == relative_time
        loss = balanced_bce_loss(logits, batch.targets, batch.valid & supervised)
        (loss / self.config.gradient_accumulation).backward()
        self.update_index += 1
        self.accumulated_windows += 1
        return float(loss.detach().cpu())

    def _optimizer_step(self, *, partial: bool) -> None:
        if self.accumulated_windows < 1:
            raise TrainingError("cannot step an empty gradient accumulation group")
        if partial:
            scale = self.config.gradient_accumulation / self.accumulated_windows
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        elif self.accumulated_windows != self.config.gradient_accumulation:
            raise TrainingError("a full optimizer group has the wrong window count")
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulated_windows = 0
        if self.scheduler is not None:
            getattr(self.scheduler, "step")()

    @staticmethod
    def _next_cursor(
        blocks: tuple[tuple[int, ...], ...], block_index: int, window_index: int
    ) -> tuple[int, int]:
        if window_index + 1 < len(blocks[block_index]):
            return block_index, window_index + 1
        return block_index + 1, 0

    def _commit_window(self, next_block: int, next_window: int) -> None:
        self.next_block = next_block
        self.next_window = next_window
        assert self.block_order is not None
        self.phase = "world_complete" if next_block == len(self.block_order) else "windows"
        self.commit_id += 1
        self.save_progress()

    def train_world(
        self,
        world: object,
        world_index: int,
        world_kind: str,
        world_seed: int,
        *,
        blocks: Sequence[Sequence[int]] | None = None,
        start_block: int = 0,
        start_window: int = 0,
        loss_sum: float = 0.0,
        window_count: int = 0,
    ) -> float:
        self._require_world(world, world_kind, world_seed)
        self.world_index = int(world_index)
        self.resume_world = self.world_index
        self.active_world = self._world_payload(world)
        self.active_world_kind = world_kind
        self.active_world_seed = int(world_seed)
        self.model.train()
        self.cache.clear()
        new_world = blocks is None
        if new_world:
            if self.accumulated_windows != 0:
                raise TrainingError("a new world cannot inherit accumulated gradients")
            self.optimizer.zero_grad(set_to_none=True)
            self.block_order = shuffled_center_blocks(
                self.legal_centers,
                self.config.chunk_centers,
                _derived_seed(self.seed, world_index, 1),
            )
        else:
            self.block_order = tuple(
                tuple(int(center) for center in block) for block in blocks
            )
        flattened = tuple(center for block in self.block_order for center in block)
        if sorted(flattened) != list(self.legal_centers) or len(set(flattened)) != len(
            flattened
        ):
            raise TrainingError("saved block order does not cover every legal center once")
        if not 0 <= start_block <= len(self.block_order):
            raise TrainingError("saved block cursor lies outside the world")
        if start_block < len(self.block_order) and not 0 <= start_window < len(
            self.block_order[start_block]
        ):
            raise TrainingError("saved window cursor lies outside its block")
        if start_block == len(self.block_order) and start_window != 0:
            raise TrainingError("completed world cursor must have window index zero")
        expected_completed = sum(len(block) for block in self.block_order[:start_block])
        expected_completed += start_window
        if int(window_count) != expected_completed:
            raise TrainingError("saved window count does not match the exact cursor")
        expected_accumulated = expected_completed % self.config.gradient_accumulation
        if self.accumulated_windows != expected_accumulated:
            raise TrainingError("saved gradients do not match the exact window cursor")
        if not math.isfinite(float(loss_sum)):
            raise TrainingError("saved world loss is non-finite")
        self.world_loss_sum = float(loss_sum)
        self.world_window_count = int(window_count)
        self.next_block, self.next_window = start_block, start_window
        if start_block == len(self.block_order):
            self.phase = "world_complete"
            return self.world_loss_sum / self.world_window_count

        self.phase = "windows"
        if new_world:
            # Persist the complete WorldSpec and block order before the first window.
            self.commit_id += 1
            self.save_progress()
        for block_index in range(start_block, len(self.block_order)):
            block = self.block_order[block_index]
            first_window = start_window if block_index == start_block else 0
            for window_index in range(first_window, len(block)):
                self.world_loss_sum += self._step_window(world, block[window_index])
                self.world_window_count += 1
                next_block, next_window = self._next_cursor(
                    self.block_order, block_index, window_index
                )
                if self.accumulated_windows == self.config.gradient_accumulation:
                    self._optimizer_step(partial=False)
                elif next_block == len(self.block_order):
                    self._optimizer_step(partial=True)
                self._commit_window(next_block, next_window)
            self.cache.clear()
        if self.accumulated_windows:
            raise TrainingError("world completion left uncommitted gradients")
        if self.world_window_count != len(self.legal_centers):
            raise TrainingError("one world did not cover every legal window exactly once")
        return self.world_loss_sum / self.world_window_count

    def _progress_payload(self) -> dict[str, Any]:
        scheduler = {
            "present": self.scheduler is not None,
            "class": None if self.scheduler is None else _qualified_class(self.scheduler),
            "state": None
            if self.scheduler is None
            else getattr(self.scheduler, "state_dict")(),
        }
        return {
            "format": PROGRESS_FORMAT,
            "commit_id": self.commit_id,
            "phase": self.phase,
            "maximum_worlds": self.maximum_worlds,
            "stop_reason": self.stop_reason,
            "scientific_identity": self.scientific_identity,
            "training_condition": self.condition.to_dict(),
            "seed": self.seed,
            "config": asdict(self.config),
            "checkpoint_selector": self.selector_identity,
            "development_evaluator": self.evaluator_identity,
            "cursor": {
                "world_index": self.resume_world,
                "block_index": self.next_block,
                "window_index": self.next_window,
                "windows_completed": self.world_window_count,
            },
            "world": self.active_world,
            "world_kind": self.active_world_kind,
            "world_seed": self.active_world_seed,
            "blocks": None
            if self.block_order is None
            else [list(block) for block in self.block_order],
            "model": self.model.state_dict(),
            "optimizer": {
                "class": _qualified_class(self.optimizer),
                "state": self.optimizer.state_dict(),
            },
            "scheduler": scheduler,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "history": self.history,
            "current_world": {
                "loss_sum": self.world_loss_sum,
                "window_count": self.world_window_count,
            },
            "gradient_accumulation": {
                "windows": self.accumulated_windows,
                "gradients": {
                    name: parameter.grad.detach().cpu().clone()
                    for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None
                },
            },
            "update_index": self.update_index,
            "best": {
                "key": self.best_key,
                "world": self.best_world,
                "stale_evaluations": self.stale_evaluations,
                "state": self.best_state,
                "maximum_primary": self.maximum_primary,
                "candidates": self.selection_candidates,
            },
        }

    def save_progress(self) -> None:
        if not 0 <= self.accumulated_windows < self.config.gradient_accumulation:
            raise TrainingError("gradient accumulation count is outside its exact range")
        has_gradients = any(
            parameter.grad is not None for parameter in self.model.parameters()
        )
        if has_gradients != (self.accumulated_windows > 0):
            raise TrainingError("gradient buffers and accumulation count disagree")
        if self.phase != "windows" and self.accumulated_windows:
            raise TrainingError("only an in-progress world may retain gradients")
        _atomic_torch(self.run_dir / "progress.pt", self._progress_payload())

    def load_progress(self) -> int:
        path = self.run_dir / "progress.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("format") != PROGRESS_FORMAT:
            raise TrainingError("progress checkpoint has an unsupported format")
        if payload.get("config") != asdict(self.config):
            raise TrainingError("progress checkpoint uses a different runtime config")
        if payload.get("scientific_identity") != self.scientific_identity:
            raise TrainingError("progress checkpoint uses a different scientific identity")
        if payload.get("training_condition") != self.condition.to_dict():
            raise TrainingError("progress checkpoint uses a different B0--B5 condition")
        if payload.get("seed") != self.seed:
            raise TrainingError("progress checkpoint uses a different training seed")
        if payload.get("checkpoint_selector") != self.selector_identity:
            raise TrainingError("progress checkpoint uses a different selection function")
        if payload.get("development_evaluator") != self.evaluator_identity:
            raise TrainingError("progress checkpoint uses a different development evaluator")
        optimizer = payload.get("optimizer")
        if not isinstance(optimizer, Mapping) or optimizer.get("class") != _qualified_class(
            self.optimizer
        ):
            raise TrainingError("progress checkpoint uses a different optimizer class")
        scheduler = payload.get("scheduler")
        if not isinstance(scheduler, Mapping):
            raise TrainingError("progress checkpoint has an invalid scheduler record")
        saved_scheduler = scheduler.get("present") is True
        if saved_scheduler != (self.scheduler is not None):
            raise TrainingError("progress checkpoint scheduler presence differs")
        if self.scheduler is not None and scheduler.get("class") != _qualified_class(
            self.scheduler
        ):
            raise TrainingError("progress checkpoint uses a different scheduler class")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(optimizer["state"])
        if self.scheduler is not None:
            getattr(self.scheduler, "load_state_dict")(scheduler["state"])
        rng = payload.get("rng")
        if not isinstance(rng, Mapping):
            raise TrainingError("progress checkpoint has no exact RNG state")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        cuda_state = rng.get("cuda")
        if cuda_state is not None:
            if not torch.cuda.is_available():
                raise TrainingError("CUDA RNG cannot be restored without CUDA")
            if len(cuda_state) != torch.cuda.device_count():
                raise TrainingError("CUDA device count differs from the saved run")
            torch.cuda.set_rng_state_all(cuda_state)
        elif self.device.type == "cuda":
            raise TrainingError("CUDA training cannot resume from a CPU RNG record")
        best = payload.get("best")
        if not isinstance(best, Mapping):
            raise TrainingError("progress checkpoint has an invalid best-state record")
        raw_key = best.get("key")
        self.best_key = None if raw_key is None else tuple(float(value) for value in raw_key)
        if self.best_key is not None and not all(
            math.isfinite(value) for value in self.best_key
        ):
            raise TrainingError("saved checkpoint-selection key is non-finite")
        self.best_world = int(best["world"])
        self.stale_evaluations = int(best["stale_evaluations"])
        self.best_state = best["state"]
        raw_maximum_primary = best.get("maximum_primary")
        self.maximum_primary = (
            None if raw_maximum_primary is None else float(raw_maximum_primary)
        )
        if self.maximum_primary is not None and not math.isfinite(
            self.maximum_primary
        ):
            raise TrainingError("saved maximum development AP is non-finite")
        raw_candidates = best.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise TrainingError("saved checkpoint candidates are invalid")
        self.selection_candidates = []
        for candidate in raw_candidates:
            if not isinstance(candidate, Mapping):
                raise TrainingError("saved checkpoint candidate is invalid")
            candidate_key = tuple(float(value) for value in candidate.get("key", ()))
            candidate_state = candidate.get("state")
            if (
                len(candidate_key) != 2
                or not all(math.isfinite(value) for value in candidate_key)
                or type(candidate.get("world")) is not int
                or not isinstance(candidate_state, Mapping)
            ):
                raise TrainingError("saved checkpoint candidate is incomplete")
            self.selection_candidates.append(
                {
                    "key": candidate_key,
                    "world": int(candidate["world"]),
                    "state": dict(candidate_state),
                }
            )
        self.history = list(payload["history"])
        self.update_index = int(payload["update_index"])
        self.commit_id = int(payload["commit_id"])
        saved_maximum = payload.get("maximum_worlds")
        if saved_maximum is not None and (
            type(saved_maximum) is not int
            or saved_maximum < len(self.config.world_type_cycle)
        ):
            raise TrainingError("progress checkpoint has an invalid world limit")
        self.maximum_worlds = saved_maximum
        saved_reason = payload.get("stop_reason")
        if saved_reason not in {None, "maximum_worlds", "development_patience"}:
            raise TrainingError("progress checkpoint has an invalid stop reason")
        self.stop_reason = saved_reason
        self.phase = str(payload["phase"])
        if self.phase not in {
            "windows",
            "world_complete",
            "development_pending",
            "between_worlds",
            "finalizing",
            "budget_exhausted",
        }:
            raise TrainingError("progress checkpoint has an unknown phase")
        cursor = payload.get("cursor")
        if not isinstance(cursor, Mapping):
            raise TrainingError("progress checkpoint has no exact cursor")
        self.resume_world = int(cursor["world_index"])
        self.next_block = int(cursor["block_index"])
        self.next_window = int(cursor["window_index"])
        self.world_window_count = int(cursor["windows_completed"])
        current = payload.get("current_world")
        if not isinstance(current, Mapping) or int(current["window_count"]) != (
            self.world_window_count
        ):
            raise TrainingError("progress checkpoint has inconsistent window counts")
        self.world_loss_sum = float(current["loss_sum"])
        accumulation = payload.get("gradient_accumulation")
        if not isinstance(accumulation, Mapping):
            raise TrainingError("progress checkpoint has no gradient accumulation state")
        raw_accumulated = accumulation.get("windows")
        gradients = accumulation.get("gradients")
        if (
            type(raw_accumulated) is not int
            or not 0 <= raw_accumulated < self.config.gradient_accumulation
            or not isinstance(gradients, Mapping)
        ):
            raise TrainingError("progress checkpoint has invalid accumulated gradients")
        parameters = dict(self.model.named_parameters())
        if any(name not in parameters for name in gradients):
            raise TrainingError("progress checkpoint contains an unknown gradient")
        self.optimizer.zero_grad(set_to_none=True)
        for name, value in gradients.items():
            if not isinstance(value, Tensor) or value.shape != parameters[name].shape:
                raise TrainingError(f"saved gradient {name} has an invalid shape")
            _finite_tensor(f"saved gradient {name}", value)
            parameters[name].grad = value.to(
                device=parameters[name].device, dtype=parameters[name].dtype
            )
        self.accumulated_windows = raw_accumulated
        if bool(gradients) != (self.accumulated_windows > 0):
            raise TrainingError("saved gradient buffers and accumulation count disagree")
        if self.phase != "windows" and self.accumulated_windows:
            raise TrainingError("saved gradients exist outside an in-progress world")
        world = payload.get("world")
        self.active_world = None if world is None else dict(world)
        kind = payload.get("world_kind")
        if kind is not None and kind not in WORLD_TYPES:
            raise TrainingError("progress checkpoint has an unknown world type")
        self.active_world_kind = kind
        raw_world_seed = payload.get("world_seed")
        self.active_world_seed = (
            None if raw_world_seed is None else int(raw_world_seed)
        )
        if self.active_world is not None:
            expected_kind = self.config.world_type_cycle[
                self.resume_world % len(self.config.world_type_cycle)
            ]
            expected_seed = _derived_seed(self.seed, self.resume_world, 0)
            if (
                self.active_world_kind != expected_kind
                or self.active_world_seed != expected_seed
            ):
                raise TrainingError("saved world does not match the frozen world stream")
        blocks = payload.get("blocks")
        self.block_order = (
            None
            if blocks is None
            else tuple(tuple(int(center) for center in block) for block in blocks)
        )
        if self.phase == "windows" and self.accumulated_windows != (
            self.world_window_count % self.config.gradient_accumulation
        ):
            raise TrainingError("saved gradients do not match the saved window cursor")
        self.loaded_progress = True
        return self.resume_world

    def _development_update(self, world_index: int, training_loss: float) -> dict[str, Any]:
        if self.development_evaluator is None:
            raise TrainingError("formal training requires fixed 201 development evaluation")
        evidence = self.development_evaluator(
            self.model, world_index, self.seed, self.condition
        )
        if not isinstance(evidence, DevelopmentEvidence):
            raise TrainingError("development evaluator must return DevelopmentEvidence")
        raw_key = checkpoint_selection_key(self.selection_rule, evidence)
        if isinstance(raw_key, (str, bytes)):
            raise TrainingError("checkpoint selector must return a numeric sequence")
        key = tuple(float(value) for value in raw_key)
        if len(key) != 2 or not all(math.isfinite(value) for value in key):
            raise TrainingError("checkpoint selector returned an invalid ordering key")
        record: dict[str, Any] = {
            "world": world_index,
            "training": {"loss": float(training_loss)},
            "development": evidence.to_dict(),
            "selection_key": list(key),
        }
        tolerance = _checkpoint_selection_tolerance(self.selection_rule)
        previous_world = self.best_world
        state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        self.maximum_primary = (
            key[0]
            if self.maximum_primary is None
            else max(self.maximum_primary, key[0])
        )
        candidates = [
            *self.selection_candidates,
            {"key": key, "world": world_index, "state": state},
        ]
        candidates = [
            candidate
            for candidate in candidates
            if float(candidate["key"][0]) >= self.maximum_primary - tolerance
        ]

        # Retain the Pareto frontier needed if a later maximum narrows the band.
        retained: list[dict[str, Any]] = []
        for candidate in candidates:
            primary, safety = map(float, candidate["key"])
            candidate_world = int(candidate["world"])
            dominated = any(
                float(other["key"][0]) >= primary
                and (
                    float(other["key"][1]) > safety
                    or (
                        float(other["key"][1]) == safety
                        and int(other["world"]) < candidate_world
                    )
                )
                for other in candidates
                if other is not candidate
            )
            if not dominated:
                retained.append(candidate)
        if not retained:
            raise AssertionError("checkpoint candidate frontier cannot be empty")
        self.selection_candidates = retained
        selected = max(
            retained,
            key=lambda candidate: (
                float(candidate["key"][1]),
                -int(candidate["world"]),
            ),
        )
        self.best_key = tuple(map(float, selected["key"]))
        self.best_world = int(selected["world"])
        self.best_state = dict(selected["state"])
        if self.best_world != previous_world:
            self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1
        return record

    def _clear_active_world(self, next_world: int) -> None:
        self.resume_world = int(next_world)
        self.active_world = None
        self.active_world_kind = None
        self.active_world_seed = None
        self.block_order = None
        self.next_block = 0
        self.next_window = 0
        self.world_loss_sum = 0.0
        self.world_window_count = 0

    def _finalize(self) -> dict[str, Any]:
        if self.best_state is None or self.best_key is None:
            raise TrainingError("training ended before any fixed development evaluation")
        if self.stop_reason != "development_patience":
            raise TrainingError(
                "formal finalization requires the frozen development-patience rule"
            )
        self.model.load_state_dict(self.best_state, strict=True)
        if self.phase != "finalizing":
            self.phase = "finalizing"
            self.commit_id += 1
            self.save_progress()
        completion_id = self.commit_id
        model_payload = {
            "format": MODEL_FORMAT,
            "completion_id": completion_id,
            "model": self.model.state_dict(),
            "training_condition": self.condition.to_dict(),
            "condition": self.condition.to_dict(),
            "seed": self.seed,
            "best_world": self.best_world,
            "best_selection_key": self.best_key,
            "stop_reason": self.stop_reason,
            "maximum_worlds": self.maximum_worlds,
            "config": asdict(self.config),
            "scientific_identity": self.scientific_identity,
        }
        _atomic_torch(self.run_dir / "model.pt", model_payload)
        result = {
            "format": RUN_FORMAT,
            "status": "completed",
            "completion_id": completion_id,
            "training_condition": self.condition.to_dict(),
            "condition": self.condition.to_dict(),
            "seed": self.seed,
            "stop_reason": self.stop_reason,
            "maximum_worlds": self.maximum_worlds,
            "best_world": self.best_world,
            "best_selection_key": list(self.best_key),
            "scientific_identity": self.scientific_identity,
            "history": self.history,
        }
        _atomic_json(self.run_dir / "result.json", result)
        written = torch.load(
            self.run_dir / "model.pt", map_location="cpu", weights_only=True
        )
        verified = json.loads((self.run_dir / "result.json").read_text(encoding="utf-8"))
        shared = (
            written.get("format") == MODEL_FORMAT
            and written.get("completion_id") == verified.get("completion_id")
            and written.get("training_condition")
            == verified.get("training_condition")
            and written.get("condition") == verified.get("condition")
            and written.get("condition") == written.get("training_condition")
            and written.get("seed") == verified.get("seed")
            and list(written.get("best_selection_key", ()))
            == verified.get("best_selection_key")
            and written.get("scientific_identity")
            == verified.get("scientific_identity")
        )
        if not shared or verified.get("status") != "completed":
            raise TrainingError("final model and result do not form one completion")
        progress = self.run_dir / "progress.pt"
        progress.unlink()
        _fsync_parent(progress)
        return result

    def _record_budget_exhaustion(self) -> dict[str, Any]:
        """Preserve resumable state without publishing an unfinished model."""

        self.stop_reason = "maximum_worlds"
        self.phase = "budget_exhausted"
        self.commit_id += 1
        self.save_progress()
        result = {
            "format": RUN_FORMAT,
            "status": "budget_exhausted_unfinished",
            "training_condition": self.condition.to_dict(),
            "condition": self.condition.to_dict(),
            "seed": self.seed,
            "stop_reason": self.stop_reason,
            "maximum_worlds": self.maximum_worlds,
            "completed_worlds": self.resume_world,
            "best_world": self.best_world if self.best_state is not None else None,
            "best_selection_key": (
                None if self.best_key is None else list(self.best_key)
            ),
            "scientific_identity": self.scientific_identity,
            "history": self.history,
        }
        _atomic_json(self.run_dir / "result.json", result)
        return result

    def fit(
        self,
        world_factory: Callable[[str, int], object],
        *,
        maximum_worlds: int,
        start_world: int = 0,
    ) -> dict[str, Any]:
        if (
            type(maximum_worlds) is not int
            or maximum_worlds < len(self.config.world_type_cycle)
        ):
            raise ValueError("maximum_worlds must cover one complete world-type cycle")
        if self.loaded_progress and self.maximum_worlds not in {None, maximum_worlds}:
            raise TrainingError("maximum_worlds differs from the saved runtime limit")
        self.maximum_worlds = maximum_worlds
        if self.phase == "finalizing":
            return self._finalize()
        if self.phase == "budget_exhausted":
            raise TrainingError(
                "the frozen world budget was exhausted; this run is scientifically unfinished"
            )
        if maximum_worlds <= start_world:
            raise ValueError("maximum_worlds must exceed start_world")
        if self.phase != "between_worlds" and start_world != self.resume_world:
            raise TrainingError("start_world differs from the saved within-world cursor")
        if self.phase == "between_worlds":
            if self.loaded_progress and start_world not in {0, self.resume_world}:
                raise TrainingError("start_world differs from the saved world boundary")
            if self.loaded_progress:
                start_world = self.resume_world
            else:
                self.resume_world = start_world
        self.stop_reason = None
        world_index = self.resume_world
        while world_index < maximum_worlds:
            development_was_pending = self.phase == "development_pending"
            if self.phase in {"windows", "world_complete", "development_pending"}:
                if (
                    self.active_world is None
                    or self.active_world_kind is None
                    or self.active_world_seed is None
                    or self.block_order is None
                ):
                    raise TrainingError("saved within-world state is incomplete")
                world = self._world_from_payload(self.active_world)
                self._require_world(
                    world, self.active_world_kind, self.active_world_seed
                )
                if self.phase == "windows":
                    training_loss = self.train_world(
                        world,
                        world_index,
                        self.active_world_kind,
                        self.active_world_seed,
                        blocks=self.block_order,
                        start_block=self.next_block,
                        start_window=self.next_window,
                        loss_sum=self.world_loss_sum,
                        window_count=self.world_window_count,
                    )
                else:
                    if self.world_window_count != len(self.legal_centers):
                        raise TrainingError("completed world has an incomplete window count")
                    training_loss = self.world_loss_sum / self.world_window_count
            else:
                world_kind = self.config.world_type_cycle[
                    world_index % len(self.config.world_type_cycle)
                ]
                world_seed = _derived_seed(self.seed, world_index, 0)
                world = world_factory(world_kind, world_seed)
                self._require_world(world, world_kind, world_seed)
                training_loss = self.train_world(
                    world, world_index, world_kind, world_seed
                )

            completed = world_index + 1
            full_cycle_seen = completed >= len(self.config.world_type_cycle)
            evaluate_now = development_was_pending or (
                full_cycle_seen
                and (
                    completed % self.config.worlds_per_evaluation == 0
                    or completed == maximum_worlds
                )
            )
            if evaluate_now:
                if not development_was_pending:
                    self.phase = "development_pending"
                    self.save_progress()
                record = self._development_update(world_index, training_loss)
                if self.stale_evaluations >= self.config.patience:
                    self.stop_reason = "development_patience"
            else:
                record = {
                    "world": world_index,
                    "training": {"loss": float(training_loss)},
                }
            if not self.history or self.history[-1].get("world") != world_index:
                self.history.append(record)
            else:
                raise TrainingError("training history repeated a completed world")
            world_index += 1
            self._clear_active_world(world_index)
            if self.stop_reason == "development_patience":
                return self._finalize()
            if world_index >= maximum_worlds:
                return self._record_budget_exhaustion()
            self.phase = "between_worlds"
            self.commit_id += 1
            self.save_progress()
        return self._record_budget_exhaustion()


@dataclass(frozen=True, slots=True)
class FormalTrainingBuild:
    """Bound formal factories created from one preflighted protocol."""

    preflight: FormalPreflightProof
    config: TrainConfig
    condition: ExperimentCondition
    trainer_factory: Callable[[int, ExperimentCondition], AJAETrainer]
    world_factory: Callable[[str, int], object]
    maximum_worlds: int
    ray_mapping_digest: str


_AUTHORITATIVE_DEVELOPMENT_EVALUATOR: Callable[
    [nn.Module, int, int, ExperimentCondition], DevelopmentEvidence
] | None = None


def build_formal_training(
    protocol: object,
    *,
    preflight: FormalPreflightProof,
    data_root: Path | str | None,
    condition: ExperimentCondition,
    maximum_worlds: int,
    development_evaluator: Callable[
        [nn.Module, int, int, ExperimentCondition], DevelopmentEvidence
    ]
    | None,
    device: torch.device | str = "cuda",
) -> FormalTrainingBuild:
    """Bind the real train/206 sequence, renderer, STU, and AJAE optimizer."""

    config = TrainConfig.from_protocol(protocol)
    _require_preflight_proof(preflight, config, protocol=protocol)
    if not isinstance(condition, ExperimentCondition) or not condition.trainable:
        raise TrainingError("formal construction requires trainable B1, B2, B3, or B5")
    if type(maximum_worlds) is not int or maximum_worlds != preflight.maximum_worlds:
        raise TrainingError(
            "maximum_worlds must equal the protocol-frozen training limit"
        )
    authoritative_evaluator = _AUTHORITATIVE_DEVELOPMENT_EVALUATOR
    if authoritative_evaluator is None:
        raise TrainingError(
            "formal training cannot start: no reusable fixed-201 development "
            "evaluator is bound; checkpoint selection requires 24 in-generator "
            "per-world metrics plus independent pure-normal statistics"
        )
    if (
        development_evaluator is not None
        and development_evaluator is not authoritative_evaluator
    ):
        raise TrainingError(
            "formal training accepts only the repository-authoritative fixed-201 evaluator"
        )
    development_evaluator = authoritative_evaluator
    raw_evaluator_identity = getattr(
        development_evaluator, "scientific_identity", None
    )
    if not isinstance(raw_evaluator_identity, Mapping) or not raw_evaluator_identity:
        raise TrainingError(
            "the authoritative fixed-201 evaluator lacks a scientific identity"
        )
    evaluator_scientific_identity = _plain_json_object(
        "development evaluator identity", raw_evaluator_identity
    )
    if data_root is None:
        raise TrainingError("formal training requires the STU data root")
    try:
        runtime_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise TrainingError("formal training device is invalid") from error
    if runtime_device.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError("formal CUDA training requires an available CUDA device")
        if (
            runtime_device.index is not None
            and runtime_device.index >= torch.cuda.device_count()
        ):
            raise TrainingError("formal CUDA device index is unavailable")

    training = getattr(protocol, "training", None)
    if (
        not isinstance(training, Mapping)
        or training.get("source_partition") != "train"
        or training.get("source_sequence_id") != 206
    ):
        raise TrainingError("formal training source must be exactly train/206")

    try:
        from .model import (
            AJAEPointTransformer,
            FrozenSTUPointEncoder,
            stu_source_manifest,
            stu_weight_identity,
        )
        from .render import (
            PROCEDURAL_GENERATOR_SCHEMA,
            FROZEN_SENSOR_CALIBRATION_SHA256,
            build_coverage_control_context,
            canonical_normal_template_library_identity,
            extract_normal_template_library,
            collect_observed_obstacle_index,
            load_qualified_support_pool,
            load_sensor_calibration,
            precompute_coverage_control_support_streams,
            render_frame as render_counterfactual_frame,
            sample_training_world,
        )
        from .scene import (
            LabelMode,
            STUSequence,
            assemble_window as assemble_scene_window,
            canonical_ray_mapping_digest,
        )
    except ImportError:  # pragma: no cover - direct script execution
        from model import (
            AJAEPointTransformer,
            FrozenSTUPointEncoder,
            stu_source_manifest,
            stu_weight_identity,
        )
        from render import (  # type: ignore[no-redef]
            PROCEDURAL_GENERATOR_SCHEMA,
            FROZEN_SENSOR_CALIBRATION_SHA256,
            build_coverage_control_context,
            canonical_normal_template_library_identity,
            extract_normal_template_library,
            collect_observed_obstacle_index,
            load_qualified_support_pool,
            load_sensor_calibration,
            precompute_coverage_control_support_streams,
            render_frame as render_counterfactual_frame,
            sample_training_world,
        )
        from scene import (  # type: ignore[no-redef]
            LabelMode,
            STUSequence,
            assemble_window as assemble_scene_window,
            canonical_ray_mapping_digest,
        )

    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    calibration_path = getattr(protocol, "sensor_calibration_path")()
    ray_grid, sensor = load_sensor_calibration(calibration_path)
    calibration_sha256 = _sha256_file(calibration_path)
    if calibration_sha256 != FROZEN_SENSOR_CALIBRATION_SHA256:
        raise TrainingError("formal sensor calibration identity changed")
    render = getattr(protocol, "render", None)
    ray_spec = render.get("ray_grid") if isinstance(render, Mapping) else None
    required_provenance = {
        "protocol_schema": "30",
        "partition": "train",
        "sequence": "206",
        "frames": str(len(sequence.frame_ids)),
        "first_frame": str(sequence.frame_ids[0]),
        "last_frame": str(sequence.frame_ids[-1]),
    }
    provenance = dict(sensor.provenance)
    if (
        not isinstance(ray_spec, Mapping)
        or ray_grid.beam_count != ray_spec.get("beam_count")
        or ray_grid.columns != ray_spec.get("column_count")
        or ray_grid.slot_count != ray_grid.beam_count * ray_grid.columns
        or sensor.source_sequence_id != 206
        or ray_grid.calibration_frame_ids != sequence.frame_ids
        or any(provenance.get(key) != value for key, value in required_provenance.items())
    ):
        raise TrainingError("loaded sensor calibration does not match the protocol")
    canonical_ray_by_slot = np.asarray(
        ray_grid.beam_ids * ray_grid.columns + ray_grid.column_ids,
        dtype=np.int32,
    )
    ray_mapping_digest = canonical_ray_mapping_digest(canonical_ray_by_slot)
    canonical_ray_by_slot.setflags(write=False)

    training_source_digest = hashlib.sha256(b"AJAE-schema30-train-206\0")
    digested_frames: set[int] = set()

    def digest_training_frame(frame: object) -> None:
        frame_id = int(getattr(frame, "frame_id"))
        if frame_id in digested_frames:
            return
        labels = getattr(frame, "labels")
        if labels is None:
            raise TrainingError("train/206 content identity requires labels")
        training_source_digest.update(frame_id.to_bytes(8, "little"))
        for value in (
            getattr(frame, "xyzi"),
            getattr(frame, "lidar_pose"),
            getattr(labels, "packed"),
        ):
            array = np.ascontiguousarray(value)
            training_source_digest.update(array.dtype.str.encode("ascii"))
            training_source_digest.update(
                json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
            )
            training_source_digest.update(array.tobytes(order="C"))
        digested_frames.add(frame_id)

    training_frames = tuple(
        sequence.source_frame(frame_id) for frame_id in sequence.frame_ids
    )
    for frame in training_frames:
        if frame.slot_count != ray_grid.slot_count:
            raise TrainingError(
                "train/206 frame does not match the canonical ray grid"
            )
        digest_training_frame(frame)
    normal_templates = extract_normal_template_library(training_frames)
    canonical_normal_template_library_identity(normal_templates)
    if digested_frames != set(sequence.frame_ids):
        raise TrainingError("train/206 content identity is incomplete")
    training_source_sha256 = training_source_digest.hexdigest()
    normal_templates_sha256 = hashlib.sha256(
        _canonical_json_object(
            "normal template library",
            {"templates": [template.to_dict() for template in normal_templates]},
        ).encode("utf-8")
    ).hexdigest()
    project_root = Path(getattr(protocol, "path")).parent
    support_pool = load_qualified_support_pool(
        project_root / "runs/ajae/e21_v4_support_pool.npz"
    )
    obstacle_index = collect_observed_obstacle_index(training_frames)
    control_context = build_coverage_control_context(
        training_frames, support_pool, ray_grid, sensor
    )
    precompute_coverage_control_support_streams(
        control_context, normal_templates
    )

    def world_factory(world_type: str, seed: int) -> object:
        if world_type not in WORLD_TYPES:
            raise TrainingError("world factory received an unknown world type")
        if type(seed) is not int or seed < 0:
            raise TrainingError("world factory seed must be a non-negative integer")
        world, report = sample_training_world(
            normal_templates, support_pool, obstacle_index, world_type, seed,
            control_context=control_context,
        )
        report_dir = run_root / "world_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(report_dir / f"{seed}.json", report.to_dict())
        return world

    def render_training_frame(source: object, world: object) -> object:
        return render_counterfactual_frame(source, world, ray_grid, sensor)

    def assemble_training_window(
        selected: ExperimentCondition,
        center: int,
        sources: Sequence[object],
    ) -> object:
        if selected != condition:
            raise TrainingError("window factory received a different condition")
        return assemble_scene_window(
            sequence.spec,
            center,
            sources,
            condition=selected.name,
            canonical_ray_by_slot=canonical_ray_by_slot,
            ray_mapping_audited=True,
            ray_mapping_digest=ray_mapping_digest,
        )

    legal_centers = sequence.spec.legal_anchors(condition.frame_offsets)
    run_root = project_root / config.output_dir / condition.name
    stu_identity_payload = {
        "weights": stu_weight_identity(getattr(protocol, "checkpoint_path")(project_root)),
        "source_manifest_sha256": stu_source_manifest(
            getattr(protocol, "stu_repository_path")(project_root)
        )["manifest_sha256"],
    }
    renderer_generator_payload = {
        "generator_schema": PROCEDURAL_GENERATOR_SCHEMA,
        "render_source_sha256": _sha256_file(Path(__file__).with_name("render.py")),
    }
    renderer_generator_sha256 = hashlib.sha256(
        _canonical_json_object(
            "renderer/generator identity", renderer_generator_payload
        ).encode("utf-8")
    ).hexdigest()
    scientific_identity = {
        "protocol": preflight.protocol_document,
        "checkpoint_selection": preflight.checkpoint_selection,
        "development_worlds_sha256": preflight.development_document_sha256,
        "development_evaluator": {
            "callable": _qualified_callable(development_evaluator),
            "scientific_identity": evaluator_scientific_identity,
        },
        "training_source": {
            "partition": "train",
            "sequence_id": 206,
            "directory": str(sequence.sequence_dir),
            "content_sha256": training_source_sha256,
        },
        "renderer_generator": renderer_generator_payload,
        "renderer_generator_sha256": renderer_generator_sha256,
        "stu_identity_sha256": hashlib.sha256(
            _canonical_json_object("STU identity", stu_identity_payload).encode("utf-8")
        ).hexdigest(),
        "sensor_calibration": str(calibration_path),
        "calibration_sha256": calibration_sha256,
        "ray_mapping_digest": ray_mapping_digest,
        "world_support_rule": "centered_five_frames_at_seed_modulo_legal_center_count",
        "normal_template_count": len(normal_templates),
        "normal_template_library_sha256": normal_templates_sha256,
    }
    evaluation = getattr(protocol, "evaluation_spec")

    def trainer_factory(
        seed: int,
        selected: ExperimentCondition,
    ) -> AJAETrainer:
        if selected != condition:
            raise TrainingError("trainer factory received a different condition")
        if type(seed) is not int or seed not in config.seeds:
            raise TrainingError("trainer factory received a non-formal seed")
        model = AJAEPointTransformer.from_protocol(protocol).to(runtime_device)
        encoder = FrozenSTUPointEncoder.from_protocol(
            protocol,
            project_root=project_root,
        ).to(runtime_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return AJAETrainer(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            scheduler=None,
            condition=selected,
            seed=seed,
            legal_centers=legal_centers,
            source_frame=sequence.source_frame,
            render_frame=render_training_frame,
            assemble_window=assemble_training_window,
            config=config,
            evaluation_range=(
                float(evaluation.minimum_range_m),
                float(evaluation.maximum_range_m),
            ),
            run_dir=run_root / f"seed-{seed}",
            scientific_identity=scientific_identity,
            checkpoint_selection_rule=preflight.checkpoint_selection,
            ray_mapping_digest=ray_mapping_digest,
            development_evaluator=development_evaluator,
            device=runtime_device,
        )

    return FormalTrainingBuild(
        preflight=preflight,
        config=config,
        condition=condition,
        trainer_factory=trainer_factory,
        world_factory=world_factory,
        maximum_worlds=maximum_worlds,
        ray_mapping_digest=ray_mapping_digest,
    )


def train_all_seeds(
    config: TrainConfig,
    condition: ExperimentCondition,
    trainer_factory: Callable[[int, ExperimentCondition], AJAETrainer],
    world_factory: Callable[[str, int], object],
    *,
    preflight: FormalPreflightProof,
    maximum_worlds: int,
    resume: bool = False,
) -> dict[int, dict[str, Any]]:
    """Run every frozen seed independently and retain every seed result."""

    _require_preflight_proof(preflight, config)
    if not condition.trainable:
        raise TrainingError(f"{condition.name} cannot be trained independently")
    if type(maximum_worlds) is not int or maximum_worlds != preflight.maximum_worlds:
        raise TrainingError(
            "maximum_worlds must equal the protocol-frozen training limit"
        )
    if type(resume) is not bool:
        raise TrainingError("resume must be boolean")
    results: dict[int, dict[str, Any]] = {}
    model_refs: list[weakref.ReferenceType[nn.Module]] = []
    run_dirs: set[Path] = set()
    for seed in config.seeds:
        _seed_everything(seed)
        trainer = trainer_factory(seed, condition)
        if trainer.seed != seed or trainer.condition != condition or trainer.config != config:
            raise TrainingError("trainer factory changed the frozen seed or condition")
        if (
            trainer.scientific_identity.get("protocol")
            != preflight.protocol_document
            or trainer.scientific_identity.get("checkpoint_selection")
            != preflight.checkpoint_selection
        ):
            raise TrainingError("trainer factory changed the preflight identity")
        run_dir = trainer.run_dir.resolve()
        if any(reference() is trainer.model for reference in model_refs) or (
            run_dir in run_dirs
        ):
            raise TrainingError("each seed requires an independent model and run directory")
        model_refs.append(weakref.ref(trainer.model))
        run_dirs.add(run_dir)
        progress = trainer.run_dir / "progress.pt"
        if resume:
            if not progress.is_file():
                raise TrainingError(f"seed {seed} has no progress checkpoint to resume")
            start_world = trainer.load_progress()
        else:
            if progress.exists() or (trainer.run_dir / "model.pt").exists():
                raise TrainingError(f"seed {seed} run directory already contains state")
            start_world = 0
        seed_result = trainer.fit(
            world_factory,
            maximum_worlds=maximum_worlds,
            start_world=start_world,
        )
        results[seed] = seed_result
        if seed_result.get("status") != "completed":
            raise TrainingError(
                f"seed {seed} exhausted the frozen world budget; no formal model was published"
            )
    if set(results) != set(config.seeds):
        raise TrainingError("formal result omitted one or more frozen seeds")
    return results


def _contains_finite_statistic(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_finite_statistic(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_finite_statistic(item) for item in value)
    return False


def _loaded_development_document(development_worlds: object) -> dict[str, Any]:
    def record(item: object) -> dict[str, Any]:
        payload = {
            "world_id": getattr(item, "world_id"),
            "seed": getattr(item, "seed"),
            "center_frame": getattr(item, "center_frame"),
            "world": getattr(item, "world"),
            "difficulty": getattr(item, "difficulty"),
            "mechanism": getattr(item, "mechanism"),
        }
        plain = _json_plain(payload)
        if not isinstance(plain, dict):
            raise AssertionError("development record must be an object")
        return plain

    payload = {
        "format": getattr(development_worlds, "format"),
        "protocol_schema": getattr(development_worlds, "protocol_schema"),
        "sequence_id": getattr(development_worlds, "sequence_id"),
        "status": getattr(development_worlds, "status"),
        "validation": getattr(development_worlds, "validation"),
        "gate1": getattr(development_worlds, "gate1"),
        "in_generator": [
            record(item)
            for item in getattr(development_worlds, "in_generator", ())
        ],
        "generator_held_out": [
            record(item)
            for item in getattr(development_worlds, "generator_held_out", ())
        ],
    }
    plain = _json_plain(payload)
    if not isinstance(plain, dict):
        raise AssertionError("development document must be an object")
    return plain


def validate_formal_preflight(
    protocol: object,
    development_worlds: object,
    development_document: Mapping[str, Any],
) -> FormalPreflightProof:
    """Validate scientific prerequisites without creating run state."""

    errors: list[str] = []
    try:
        loaded_document = _loaded_development_document(development_worlds)
        if _canonical_json_object(
            "loaded development worlds", loaded_document
        ) != _canonical_json_object("development document", development_document):
            errors.append(
                "development document changed or differs from the loaded worlds"
            )
    except (AttributeError, TypeError, ValueError, TrainingError) as error:
        errors.append(f"development document identity is invalid: {error}")
    if getattr(development_worlds, "validated", False) is not True:
        errors.append("the 30 development worlds are not frozen and validated")
    in_generator = tuple(getattr(development_worlds, "in_generator", ()))
    held_out = tuple(getattr(development_worlds, "generator_held_out", ()))
    if len(in_generator) != 24 or len(held_out) != 6:
        errors.append("development worlds must contain exactly 24+6 definitions")
    else:
        try:
            from .render import WorldSpec
        except ImportError:  # pragma: no cover - direct script execution
            from render import WorldSpec  # type: ignore[no-redef]
        records = (*in_generator, *held_out)
        try:
            parsed = [WorldSpec.from_dict(item.world) for item in records]
        except (AttributeError, TypeError, ValueError) as error:
            errors.append(f"development WorldSpec parsing failed: {error}")
        else:
            identifiers = tuple(getattr(item, "world_id", None) for item in records)
            if identifiers != tuple(range(30)):
                errors.append("development world IDs must be exactly 0..29")
            if any(
                world.seed != getattr(item, "seed", None)
                or world.source_sequence_id != 201
                for item, world in zip(records, parsed, strict=True)
            ):
                errors.append("development WorldSpec identity does not match fixed 201")
            if any(world.world_type != "mixed" for world in parsed[:24]):
                errors.append("all 24 in-generator development worlds must be mixed")

    gate1 = (
        development_document.get("gate1")
        if isinstance(development_document, Mapping)
        else None
    )
    if not isinstance(gate1, Mapping) or gate1.get("status") != (
        "passed_with_real_evidence"
    ):
        errors.append("gate1 has not passed with real evidence")
    else:
        evidence = gate1.get("evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != GATE1_EVIDENCE_KEYS:
            errors.append("gate1 evidence does not contain the four required checks")
        else:
            for name in sorted(GATE1_EVIDENCE_KEYS):
                item = evidence[name]
                if not isinstance(item, Mapping) or not _contains_finite_statistic(item):
                    errors.append(f"gate1 evidence {name} lacks real statistics")

    development = getattr(protocol, "development", None)
    selection = (
        None
        if not isinstance(development, Mapping)
        else development.get("checkpoint_selection")
    )
    if not isinstance(selection, Mapping):
        errors.append("checkpoint selection is not frozen before training")
    else:
        try:
            _checkpoint_selection_tolerance(selection)
        except TrainingError as error:
            errors.append(str(error))
    fixed_world_evaluation = (
        None
        if not isinstance(development, Mapping)
        else development.get("fixed_world_evaluation")
    )
    if (
        not isinstance(fixed_world_evaluation, Mapping)
        or fixed_world_evaluation.get("status") != "frozen_before_training"
        or not isinstance(fixed_world_evaluation.get("scope"), Mapping)
    ):
        errors.append("fixed-201 development evaluation scope is not frozen")
    evaluation_document = getattr(protocol, "evaluation_document", None)
    comparison_frame_domain = (
        evaluation_document.get("comparison_frame_domain")
        if isinstance(evaluation_document, Mapping)
        else None
    )
    if (
        not isinstance(comparison_frame_domain, Mapping)
        or comparison_frame_domain.get("status") != "frozen_before_evaluation"
    ):
        errors.append("the common B0--B5 comparison frame domain is not frozen")
    decision_gates = getattr(protocol, "decision_gates", None)
    gate_criteria = (
        decision_gates.get("criteria")
        if isinstance(decision_gates, Mapping)
        else None
    )
    if (
        not isinstance(gate_criteria, Mapping)
        or gate_criteria.get("status") != "frozen_before_training"
        or any(
            not isinstance(gate_criteria.get(name), Mapping)
            for name in (
                "gate1",
                "gate2",
                "gate3",
                "gate4",
                "development_difficulty_coverage",
            )
        )
    ):
        errors.append("scientific decision-gate criteria are not frozen")
    config: TrainConfig | None = None
    try:
        config = TrainConfig.from_protocol(protocol)
    except (TypeError, ValueError, TrainingError) as error:
        errors.append(str(error))
    training_values = getattr(protocol, "training", None)
    frozen_maximum_worlds = (
        training_values.get("maximum_worlds")
        if isinstance(training_values, Mapping)
        else None
    )
    if type(frozen_maximum_worlds) is not int:
        errors.append("protocol.training.maximum_worlds is not frozen before training")
    elif config is not None and frozen_maximum_worlds < len(config.world_type_cycle):
        errors.append(
            "protocol.training.maximum_worlds does not cover the world-type cycle"
        )
    converter = getattr(protocol, "plain_document", None)
    protocol_document: Mapping[str, Any] | None = None
    if not callable(converter):
        errors.append("protocol does not expose plain_document()")
    else:
        converted = converter()
        if not isinstance(converted, Mapping):
            errors.append("protocol plain_document() is not a mapping")
        else:
            protocol_document = converted
    if errors:
        raise TrainingError("; ".join(errors))
    assert isinstance(selection, Mapping)
    assert config is not None and protocol_document is not None
    return FormalPreflightProof(
        _FORMAL_PREFLIGHT_SEAL,
        _canonical_json_object("protocol", protocol_document),
        _canonical_json_object("development document", development_document),
        _canonical_json_object("checkpoint selection", selection),
        _canonical_json_object("training config", asdict(config)),
    )


def run_formal_training(
    protocol_path: Path | str = Path("protocol.json"),
    *,
    development_path: Path | str | None = None,
    data_root: Path | str | None = None,
    condition_name: str = "B3",
    maximum_worlds: int,
    development_evaluator: Callable[
        [nn.Module, int, int, ExperimentCondition], DevelopmentEvidence
    ]
    | None = None,
    device: torch.device | str = "cuda",
    resume: bool = False,
) -> dict[int, dict[str, Any]]:
    """Run formal training only after validating the exact frozen inputs."""

    try:
        from .protocol import load_development_worlds, load_protocol
    except ImportError:  # pragma: no cover - direct script execution
        from protocol import load_development_worlds, load_protocol

    protocol = load_protocol(protocol_path)
    authoritative_development_path = (
        protocol.development_worlds_path().expanduser().resolve(strict=True)
    )
    selected_development_path = (
        authoritative_development_path
        if development_path is None
        else Path(development_path).expanduser().resolve(strict=True)
    )
    development_worlds = load_development_worlds(
        selected_development_path,
        protocol=protocol,
    )
    development_document = json.loads(
        Path(selected_development_path).read_text(encoding="utf-8")
    )
    preflight = validate_formal_preflight(
        protocol,
        development_worlds,
        development_document,
    )
    if selected_development_path != authoritative_development_path:
        raise TrainingError(
            "formal training must use the protocol-authoritative development worlds"
        )
    condition = experiment_condition(condition_name)
    build = build_formal_training(
        protocol,
        preflight=preflight,
        data_root=data_root,
        condition=condition,
        maximum_worlds=maximum_worlds,
        development_evaluator=development_evaluator,
        device=device,
    )
    return train_all_seeds(
        build.config,
        build.condition,
        build.trainer_factory,
        build.world_factory,
        preflight=build.preflight,
        maximum_worlds=build.maximum_worlds,
        resume=resume,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AJAE only when every formal prerequisite is bound"
    )
    parser.add_argument("--protocol", type=Path, default=Path("protocol.json"))
    parser.add_argument("--development", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), default="B3")
    parser.add_argument("--max-worlds", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        results = run_formal_training(
            args.protocol,
            development_path=args.development,
            data_root=args.data_root,
            condition_name=args.condition,
            maximum_worlds=args.max_worlds,
            development_evaluator=None,
            device=args.device,
            resume=args.resume,
        )
    except (OSError, TypeError, ValueError, TrainingError) as error:
        raise SystemExit(
            f"Formal training refused: {error}. No run state was written."
        ) from error
    print(
        json.dumps(
            {
                "status": "completed",
                "seeds": sorted(results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
