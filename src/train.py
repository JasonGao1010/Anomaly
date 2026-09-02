#!/usr/bin/env python3
"""Train AJAE with the frozen four-world protocol."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import random
import sys
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
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
_MALLOC_TRIM = getattr(ctypes.CDLL(None), "malloc_trim", None)
if _MALLOC_TRIM is not None:
    _MALLOC_TRIM.argtypes = (ctypes.c_size_t,)
    _MALLOC_TRIM.restype = ctypes.c_int


class TrainingError(RuntimeError):
    """Report an invalid or failed AJAE optimization operation."""


def _release_host_saved_tensors() -> None:
    """Return freed offload buffers to Linux after an exact backward pass."""

    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


@contextmanager
def _bounded_saved_tensor_offload(
    device: torch.device, *, budget_bytes: int = 2 * 1024**3
):
    """Offload saved CUDA tensors synchronously within a fixed host budget."""

    if device.type != "cuda":
        yield
        return
    offloaded_bytes = 0

    def pack(tensor: Tensor) -> tuple[Tensor, torch.device | None]:
        nonlocal offloaded_bytes
        if tensor.device.type != "cuda":
            return tensor, None
        tensor_bytes = tensor.numel() * tensor.element_size()
        if offloaded_bytes + tensor_bytes > budget_bytes:
            return tensor, None
        offloaded_bytes += tensor_bytes
        return tensor.detach().to(device="cpu", copy=True), tensor.device

    def unpack(payload: tuple[Tensor, torch.device | None]) -> Tensor:
        tensor, original_device = payload
        if original_device is None:
            return tensor
        return tensor.to(device=original_device)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def _seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Formal repeats must not depend on CUDA atomic reduction order.
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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


def _tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash a tensor state without depending on serialization metadata."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _array_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _atomic_npz(
    path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(
            json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
        ),
    )
    os.replace(temporary, path)
    _fsync_parent(path)


def _finite_world_report_document(report: object) -> dict[str, Any]:
    """Encode the valid no-obstacle +infinity sentinel as JSON null."""

    converter = getattr(report, "to_dict", None)
    if not callable(converter):
        raise TrainingError("world report does not expose to_dict()")
    document = converter()
    if not isinstance(document, dict) or not isinstance(
        document.get("placements"), list
    ):
        raise TrainingError("world report document is malformed")
    for placement in document["placements"]:
        if not isinstance(placement, dict):
            raise TrainingError("world report placement is malformed")
        proposals = placement.get("proposal_minimum_obstacle_sdf_m")
        minimum = placement.get("minimum_obstacle_sdf_m")
        if not isinstance(proposals, list):
            raise TrainingError("world report obstacle clearances are malformed")
        for index, value in enumerate(proposals):
            if isinstance(value, float) and math.isinf(value) and value > 0.0:
                proposals[index] = None
        if isinstance(minimum, float) and math.isinf(minimum) and minimum > 0.0:
            placement["minimum_obstacle_sdf_m"] = None
    return document


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


def _result_blind_budget_prefix_matches(
    current_identity: Mapping[str, Any],
    saved_identity: Mapping[str, Any],
    *,
    progress_sha256: str,
    payload: Mapping[str, Any],
) -> bool:
    """Accept only the one frozen 40-to-25-world prefix revision."""

    try:
        current = _plain_json_object("current scientific identity", current_identity)
        saved = _plain_json_object("saved scientific identity", saved_identity)
    except TrainingError:
        return False
    protocol = current.get("protocol")
    if not isinstance(protocol, dict):
        return False
    training = protocol.get("training")
    development = protocol.get("development")
    if not isinstance(training, dict) or not isinstance(development, dict):
        return False
    revision = training.get("result_blind_budget_revision")
    freeze = development.get("e63_freeze")
    shared = freeze.get("shared_training") if isinstance(freeze, dict) else None
    if not isinstance(revision, dict) or not isinstance(shared, dict):
        return False
    prefix = revision.get("checkpoint_prefix_reuse")
    if not isinstance(prefix, dict):
        return False
    cursor = payload.get("cursor")
    condition = payload.get("training_condition")
    history = payload.get("history")
    saved_limit = payload.get("maximum_worlds")
    development_count = (
        sum(isinstance(record, Mapping) and "development" in record for record in history)
        if isinstance(history, list)
        else -1
    )
    if (
        revision.get("version") != "E74-result-blind-budget-reduction-v1"
        or revision.get("status") != "frozen_before_result_exposure"
        or revision.get("previous_maximum_worlds") != 40
        or revision.get("maximum_worlds") != 25
        or training.get("maximum_worlds") != 25
        or shared.get("maximum_complete_worlds_per_seed") != 25
        or revision.get("development_metric_values_read") is not False
        or tuple(revision.get("scope_conditions", ())) != ("B1", "B2", "B3")
        or progress_sha256 != prefix.get("progress_sha256")
        or not isinstance(condition, Mapping)
        or condition.get("name") != prefix.get("condition")
        or payload.get("seed") != prefix.get("seed")
        or payload.get("phase") != prefix.get("phase")
        or cursor != prefix.get("cursor")
        or not isinstance(history, list)
        or len(history) != prefix.get("history_worlds")
        or development_count != prefix.get("completed_development_evaluations")
        or saved_limit != revision.get("previous_maximum_worlds")
        or hashlib.sha256(
            _canonical_json_object("saved scientific identity", saved).encode("utf-8")
        ).hexdigest()
        != prefix.get("scientific_identity_sha256")
    ):
        return False

    predecessor = _plain_json_object("predecessor scientific identity", current)
    predecessor_protocol = predecessor["protocol"]
    predecessor_training = predecessor_protocol["training"]
    predecessor_training.pop("result_blind_budget_revision", None)
    predecessor_training["maximum_worlds"] = 40
    predecessor_protocol["development"]["e63_freeze"]["shared_training"][
        "maximum_complete_worlds_per_seed"
    ] = 40
    return saved == predecessor


def _qualified_class(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _qualified_callable(value: Callable[..., object] | None) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", type(value).__module__)
    if module == "__main__":
        main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        module = getattr(main_spec, "name", module)
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
    selection = _plain_json_object("checkpoint selection", selection)
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
        if not worlds or len(worlds) > 24 or any(
            not isinstance(item, DevelopmentWorldMetrics) for item in worlds
        ):
            raise ValueError("checkpoint selection requires 1--24 world metrics")
        identifiers = [item.world_id for item in worlds]
        if (
            len(set(identifiers)) != len(identifiers)
            or any(world_id < 0 or world_id >= 24 for world_id in identifiers)
        ):
            raise ValueError("in-generator development world IDs must be a unique 0..23 subset")
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
        "primary": "maximum macro mean of per-world AP over the E63 common-domain eligible in-generator worlds",
        "first_tie_break": "lower development macro mean FPR95",
        "second_tie_break": "lower pure-normal cross-fit FPR",
        "third_tie_break": "earlier completed world index",
        "held_out_input_forbidden": True,
    }
    if set(rule) != {*expected, "tie_tolerance", "eligible_world_ids"} or any(
        rule.get(name) != value for name, value in expected.items()
    ):
        raise TrainingError("checkpoint selection rule is not the frozen E63-v2 rule")
    eligible = rule.get("eligible_world_ids")
    if (
        not isinstance(eligible, list)
        or not eligible
        or len(set(eligible)) != len(eligible)
        or any(type(world_id) is not int or world_id < 0 or world_id >= 24 for world_id in eligible)
    ):
        raise TrainingError("checkpoint selection lacks the E63 eligible-world identities")
    tolerance = rule.get("tie_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TrainingError("checkpoint-selection tolerance must be numeric")
    result = float(tolerance)
    if not math.isfinite(result) or result <= 0.0:
        raise TrainingError("checkpoint-selection tolerance must be positive")
    return result


def checkpoint_selection_key(
    rule: Mapping[str, Any], evidence: DevelopmentEvidence
) -> tuple[float, float, float]:
    """Return macro AP and the two frozen safety tie-break values."""

    _checkpoint_selection_tolerance(rule)
    eligible = tuple(int(world_id) for world_id in rule["eligible_world_ids"])
    if tuple(sorted(item.world_id for item in evidence.in_generator)) != tuple(
        sorted(eligible)
    ):
        raise TrainingError("development evidence does not match the E63 common domain")
    ap = []
    fpr95 = []
    for world in sorted(evidence.in_generator, key=lambda item: item.world_id):
        if "AP" not in world.metrics or "FPR95" not in world.metrics:
            raise TrainingError(
                f"development world {world.world_id} lacks AP or FPR95"
            )
        ap.append(float(world.metrics["AP"]))
        fpr95.append(float(world.metrics["FPR95"]))
    if "cross_fit_FPR" not in evidence.pure_normal:
        raise TrainingError("pure-normal evidence lacks cross-fit FPR")
    return (
        float(np.mean(ap)),
        -float(np.mean(fpr95)),
        -float(evidence.pure_normal["cross_fit_FPR"]),
    )


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
    """Shuffle ordered center blocks while preserving order within each block."""

    centers = tuple(int(value) for value in legal_centers)
    if not centers or any(right <= left for left, right in zip(centers, centers[1:])):
        raise ValueError("legal centers must be strictly increasing")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    blocks = [
        centers[start : start + block_size]
        for start in range(0, len(centers), block_size)
    ]
    random.Random(seed).shuffle(blocks)
    return tuple(blocks)


@dataclass(frozen=True, slots=True)
class TrainingSchedule:
    """Freeze center sampling, development evaluations, and stage pauses."""

    version: str
    center_stride: int
    phase_modulus: int
    windows_per_world: int
    development_worlds: tuple[int, ...]
    pause_after_worlds: tuple[int, ...]
    output_directory: str | None = None

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("training schedule version must be non-empty")
        if (
            type(self.center_stride) is not int
            or self.center_stride < 1
            or type(self.phase_modulus) is not int
            or self.phase_modulus != self.center_stride
            or type(self.windows_per_world) is not int
            or self.windows_per_world < 1
        ):
            raise ValueError("training center schedule is invalid")
        if (
            not self.development_worlds
            or tuple(sorted(set(self.development_worlds))) != self.development_worlds
            or tuple(sorted(set(self.pause_after_worlds))) != self.pause_after_worlds
        ):
            raise ValueError("training stage worlds are invalid")
        if self.output_directory is not None:
            if not self.output_directory:
                raise ValueError("schedule output directory must be non-empty")
            output = Path(self.output_directory)
            if output.is_absolute() or ".." in output.parts:
                raise ValueError("schedule output directory must be project-relative")

    def centers(
        self, legal_centers: Sequence[int], world_index: int
    ) -> tuple[int, ...]:
        centers = tuple(int(value) for value in legal_centers)
        if (
            not centers
            or any(right != left + 1 for left, right in zip(centers, centers[1:]))
            or type(world_index) is not int
            or world_index < 0
        ):
            raise TrainingError("training schedule requires one legal center range")
        phase = world_index % self.phase_modulus
        selected = centers[phase :: self.center_stride]
        if len(selected) != self.windows_per_world:
            raise TrainingError("training schedule changed its windows-per-world count")
        return selected


def training_schedule(
    protocol: object,
    condition: ExperimentCondition,
    config: TrainConfig,
    maximum_worlds: int,
) -> TrainingSchedule:
    """Resolve the active result-blind schedule from the authoritative protocol."""

    if condition.name == "B3":
        development = getattr(protocol, "development", None)
        exploration = (
            development.get("exploration_track")
            if isinstance(development, Mapping)
            else None
        )
        full = (
            exploration.get("full_ajae_x_freeze")
            if isinstance(exploration, Mapping)
            else None
        )
        fast = full.get("fast_viability") if isinstance(full, Mapping) else None
        if (
            isinstance(full, Mapping)
            and full.get("version") == "AJAE-full-first-v5-fast-viability"
            and isinstance(fast, Mapping)
        ):
            window = fast.get("window_schedule")
            stages = fast.get("stages")
            if not isinstance(window, Mapping) or not isinstance(stages, Mapping):
                raise TrainingError("fast viability schedule is incomplete")
            preview = stages.get("preview")
            screen = stages.get("screen")
            final = stages.get("final")
            if not all(isinstance(item, Mapping) for item in (preview, screen, final)):
                raise TrainingError("fast viability stages are incomplete")
            assert isinstance(preview, Mapping)
            assert isinstance(screen, Mapping)
            assert isinstance(final, Mapping)
            schedule = TrainingSchedule(
                version=str(fast.get("version", "")),
                center_stride=int(window.get("stride", 0)),
                phase_modulus=int(window.get("phase_modulus", 0)),
                windows_per_world=int(window.get("windows_per_world", 0)),
                development_worlds=(
                    int(screen.get("completed_worlds", 0)),
                    int(final.get("completed_worlds", 0)),
                ),
                pause_after_worlds=(
                    int(preview.get("completed_worlds", 0)),
                    int(screen.get("completed_worlds", 0)),
                ),
                output_directory=str(fast.get("output_directory", "")),
            )
            if schedule.development_worlds[-1] != maximum_worlds:
                raise TrainingError("fast viability final stage differs from the world limit")
            return schedule
    evaluation_worlds = tuple(
        range(config.worlds_per_evaluation, maximum_worlds + 1, config.worlds_per_evaluation)
    )
    if not evaluation_worlds or evaluation_worlds[-1] != maximum_worlds:
        evaluation_worlds = (*evaluation_worlds, maximum_worlds)
    return TrainingSchedule(
        version="E63-full-grid-v1",
        center_stride=1,
        phase_modulus=1,
        windows_per_world=445 if condition.name in {"B2", "B3"} else 449,
        development_worlds=evaluation_worlds,
        pause_after_worlds=(),
    )


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
        schedule: TrainingSchedule,
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
        self.schedule = schedule
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
        self.loaded_budget_revision = False
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
        # Preserve exact tensors on host memory until backward so dense formal
        # windows do not exceed WSL's GPU residency budget.
        with _bounded_saved_tensor_offload(self.device):
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
        loss_value = float(loss.detach().cpu())
        del batch, logits, loss, supervised
        if self.device.type == "cuda":
            _release_host_saved_tensors()
        return loss_value

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
        # Persist only completed deterministic blocks. A crash inside a block
        # resumes from its pre-block checkpoint and recomputes identical work.
        if next_window == 0:
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
        selected_centers = self.schedule.centers(self.legal_centers, world_index)
        new_world = blocks is None
        if new_world:
            if self.accumulated_windows != 0:
                raise TrainingError("a new world cannot inherit accumulated gradients")
            self.optimizer.zero_grad(set_to_none=True)
            self.block_order = shuffled_center_blocks(
                selected_centers,
                self.config.chunk_centers,
                _derived_seed(self.seed, world_index, 1),
            )
        else:
            self.block_order = tuple(
                tuple(int(center) for center in block) for block in blocks
            )
        flattened = tuple(center for block in self.block_order for center in block)
        if sorted(flattened) != list(selected_centers) or len(set(flattened)) != len(
            flattened
        ):
            raise TrainingError("saved block order does not cover every selected center once")
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
        if self.world_window_count != len(selected_centers):
            raise TrainingError("one world did not cover every selected window exactly once")
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
            "training_schedule": asdict(self.schedule),
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
        progress_sha256 = _sha256_file(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("format") != PROGRESS_FORMAT:
            raise TrainingError("progress checkpoint has an unsupported format")
        if payload.get("config") != asdict(self.config):
            raise TrainingError("progress checkpoint uses a different runtime config")
        if payload.get("training_schedule") != asdict(self.schedule):
            raise TrainingError("progress checkpoint uses a different training schedule")
        saved_identity = payload.get("scientific_identity")
        if saved_identity != self.scientific_identity:
            if not isinstance(saved_identity, Mapping) or not (
                _result_blind_budget_prefix_matches(
                    self.scientific_identity,
                    saved_identity,
                    progress_sha256=progress_sha256,
                    payload=payload,
                )
            ):
                raise TrainingError(
                    "progress checkpoint uses a different scientific identity"
                )
            self.loaded_budget_revision = True
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
        if self.best_key is not None and (
            len(self.best_key) != 3
            or not all(math.isfinite(value) for value in self.best_key)
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
                len(candidate_key) != 3
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
        if len(key) != 3 or not all(math.isfinite(value) for value in key):
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
            if (
                float(candidate["key"][0]) == self.maximum_primary
                or float(candidate["key"][0]) > self.maximum_primary - tolerance
            )
        ]
        if not candidates:
            raise AssertionError("checkpoint candidate set cannot be empty")
        self.selection_candidates = candidates
        selected = max(
            candidates,
            key=lambda candidate: (
                float(candidate["key"][1]),
                float(candidate["key"][2]),
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
        if self.stop_reason not in {"development_patience", "maximum_worlds"}:
            raise TrainingError(
                "formal finalization requires patience or the frozen world budget"
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
            if not (
                self.loaded_budget_revision
                and self.maximum_worlds == 40
                and maximum_worlds == 25
                and self.resume_world < maximum_worlds
            ):
                raise TrainingError("maximum_worlds differs from the saved runtime limit")
        self.maximum_worlds = maximum_worlds
        if self.phase == "finalizing":
            return self._finalize()
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
                    if self.world_window_count != len(
                        self.schedule.centers(self.legal_centers, world_index)
                    ):
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
                full_cycle_seen and completed in self.schedule.development_worlds
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
                self.stop_reason = "maximum_worlds"
                return self._finalize()
            self.phase = "between_worlds"
            self.commit_id += 1
            self.save_progress()
            if world_index in self.schedule.pause_after_worlds:
                return {
                    "format": RUN_FORMAT,
                    "status": "paused_for_external_evaluation",
                    "seed": self.seed,
                    "condition": self.condition.to_dict(),
                    "completed_worlds": world_index,
                    "maximum_worlds": maximum_worlds,
                    "training_schedule": asdict(self.schedule),
                    "progress_path": str(self.run_dir / "progress.pt"),
                }
        self.stop_reason = "maximum_worlds"
        return self._finalize()


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


class E63B1DevelopmentEvaluator:
    """Evaluate B1 on the frozen E57 worlds and E61 pure-normal set."""

    def __init__(
        self,
        *,
        protocol: object,
        project_root: Path,
        data_root: Path | str,
        device: torch.device,
        encoder: nn.Module,
        grid: object,
        sensor: object,
        canonical_by_slot: np.ndarray,
    ) -> None:
        try:
            from .scene import LabelMode, STUSequence
        except ImportError:  # pragma: no cover
            from scene import LabelMode, STUSequence  # type: ignore[no-redef]
        development = getattr(protocol, "development")
        freeze = development["e63_freeze"]
        e57_record = freeze["source_worlds"]
        e63_record = freeze["identity_artifact"]
        safety = development["safety_sets"]
        e57_path = project_root / e57_record["artifact"]
        e63_path = project_root / e63_record["path"]
        e61_path = project_root / "runs/ajae/e61_safety_identities.npz"
        if (
            _sha256_file(e57_path) != e57_record["artifact_sha256"]
            or _sha256_file(e63_path) != e63_record["artifact_sha256"]
            or _sha256_file(e61_path)
            != "8d3e08e0512dc70a75d2279cfb4515bc960bbfda4f35a872c4a76e9dad69d0e0"
        ):
            raise TrainingError("E63 development evaluator input identity changed")
        with np.load(e57_path, allow_pickle=False) as archive:
            self.world_id = np.asarray(archive["selected_world_id"], dtype=np.int16)
            self.center = np.asarray(
                archive["selected_center_frame"], dtype=np.int16
            )
            self.world_json = np.asarray(archive["selected_world_json"])
        with np.load(e63_path, allow_pickle=False) as archive:
            self.eligible = np.asarray(
                archive["common_domain_eligible"], dtype=np.bool_
            )
            self.fold = np.asarray(archive["safety_fold"])
        with np.load(e61_path, allow_pickle=False) as archive:
            self.pure_frame_id = np.asarray(
                archive["pure_frame_id"], dtype=np.int16
            )
            self.pure_mask_packed = np.asarray(
                archive["pure_canonical_mask_packed"], dtype=np.uint8
            )
            self.pure_count = np.asarray(
                archive["pure_point_count_by_frame"], dtype=np.int32
            )
        if (
            self.world_id.shape != (24,)
            or int(self.eligible.sum()) != 23
            or self.fold.shape != (24,)
            or int(self.pure_count.sum()) != int(safety["pure_normal"]["expected_points"])
        ):
            raise TrainingError("E63 development evaluator counts changed")
        self.protocol = protocol
        self.device = device
        self.encoder = encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.grid = grid
        self.sensor = sensor
        self.canonical_by_slot = np.asarray(canonical_by_slot, dtype=np.int32)
        self.sequence = STUSequence.open(
            data_root,
            protocol=protocol,
            partition="train",
            sequence_id=201,
            label_mode=LabelMode.REQUIRED,
        )
        self._development_inputs: list[dict[str, object]] | None = None
        self.scientific_identity = {
            "version": "E63-B1-fixed-development-evaluator-v1",
            "e57_artifact_sha256": e57_record["artifact_sha256"],
            "e63_artifact_sha256": e63_record["artifact_sha256"],
            "e61_artifact_sha256": "8d3e08e0512dc70a75d2279cfb4515bc960bbfda4f35a872c4a76e9dad69d0e0",
            "eligible_world_ids": self.world_id[self.eligible].astype(int).tolist(),
            "target": "q=0",
            "threshold_rule": "official first ROC threshold with TPR strictly greater than 0.95",
            "pure_normal_cross_fit": "mean FPR on the complete E61 pure-normal set under the two opposite-fold proxy thresholds",
            "threshold_comparison": "score strictly greater than threshold",
        }

    @staticmethod
    def _frame_seed(sequence_id: int, frame_id: int) -> int:
        payload = f"E53-STU-query-v1:train:{sequence_id}:{frame_id}".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
            2**63 - 1
        )

    def _encode(self, source: object) -> dict[str, Tensor]:
        seed = self._frame_seed(
            int(getattr(source, "sequence_id")), int(getattr(source, "frame_id"))
        )
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            encoding = self.encoder(
                getattr(source, "coordinates"),
                getattr(source, "features"),
                getattr(source, "real_slots"),
            )
        slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
        encoded_slots = getattr(encoding, "real_slots")
        if isinstance(encoded_slots, Tensor):
            encoded_slots = encoded_slots.detach().cpu().numpy()
        if not np.array_equal(np.asarray(encoded_slots, dtype=np.int64), slots):
            raise TrainingError("development STU output changed point order")
        return {
            "stu_features": getattr(encoding, "point_features").detach().cpu(),
            "normal_evidence": getattr(encoding, "normal_evidence").detach().cpu(),
            "assignment": getattr(encoding, "reliability_assign").detach().cpu(),
            "no_object": getattr(encoding, "reliability_noobj").detach().cpu(),
        }

    def _input(self, source: object) -> dict[str, object]:
        slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
        result: dict[str, object] = self._encode(source)
        xyzi = np.asarray(getattr(source, "xyzi"))[slots]
        result.update(
            coordinates=torch.as_tensor(xyzi[:, :3].copy()),
            intensity=torch.as_tensor(xyzi[:, 3].copy()),
            slots=slots,
        )
        return result

    def _scores(self, model: nn.Module, inputs: Mapping[str, object]) -> np.ndarray:
        coordinates = inputs["coordinates"]
        if not isinstance(coordinates, Tensor):
            raise TrainingError("development coordinates are invalid")
        count = int(coordinates.shape[0])
        with torch.no_grad():
            logits = model(
                coordinates.to(self.device),
                torch.zeros(count, dtype=torch.long, device=self.device),
                inputs["stu_features"].to(self.device),
                inputs["normal_evidence"].to(self.device),
                inputs["assignment"].to(self.device),
                inputs["no_object"].to(self.device),
                inputs["intensity"].to(self.device),
                cross_frame_enabled=False,
            )
        if logits.shape != (count,) or not bool(torch.isfinite(logits).all()):
            raise TrainingError("development model logits are invalid")
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    def _prepare_development(self) -> list[dict[str, object]]:
        if self._development_inputs is not None:
            return self._development_inputs
        try:
            from .render import WorldSpec, render_frame
        except ImportError:  # pragma: no cover
            from render import WorldSpec, render_frame  # type: ignore[no-redef]
        prepared: list[dict[str, object]] = []
        for row in np.flatnonzero(self.eligible):
            world = WorldSpec.from_dict(json.loads(str(self.world_json[row])))
            rendered = render_frame(
                self.sequence.source_frame(int(self.center[row])),
                world,
                self.grid,
                self.sensor,
            )
            source = rendered.source
            item = self._input(source)
            slots = np.asarray(item["slots"], dtype=np.int64)
            item.update(
                world_id=int(self.world_id[row]),
                fold=bytes(self.fold[row]).decode("ascii"),
                xyz=np.asarray(source.xyzi)[slots, :3].copy(),
                semantic=np.asarray(rendered.packed_labels, dtype=np.uint32)[slots]
                & np.uint32(0xFFFF),
                control=np.asarray(rendered.normal_control_mask, dtype=np.bool_)[slots],
                proxy=np.asarray(rendered.anomaly_proxy_mask, dtype=np.bool_)[slots],
            )
            prepared.append(item)
        self._development_inputs = prepared
        return prepared

    def __call__(
        self,
        model: nn.Module,
        world_index: int,
        seed: int,
        condition: ExperimentCondition,
    ) -> DevelopmentEvidence:
        del world_index, seed
        if condition.name != "B1":
            raise TrainingError("this evaluator instance is bound to B1")
        try:
            from .evaluate import PointMetricAccumulator
        except ImportError:  # pragma: no cover
            from evaluate import PointMetricAccumulator  # type: ignore[no-redef]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = model.training
        model.eval()
        try:
            metrics: list[DevelopmentWorldMetrics] = []
            fold_accumulator = {
                "A": PointMetricAccumulator(self.protocol),
                "B": PointMetricAccumulator(self.protocol),
            }
            for item in self._prepare_development():
                scores = self._scores(model, item)
                accumulator = PointMetricAccumulator(self.protocol)
                accumulator.update(item["xyz"], scores, item["semantic"])
                result = accumulator.compute()
                if result.get("accepted_frames") != 1:
                    raise TrainingError("one development world was not accepted")
                metrics.append(
                    DevelopmentWorldMetrics(
                        int(item["world_id"]),
                        {
                            "AP": float(result["AP"]),
                            "AUROC": float(result["AUROC"]),
                            "FPR95": float(result["FPR95"]),
                        },
                    )
                )
                fold_accumulator[str(item["fold"])].update(
                    item["xyz"], scores, item["semantic"]
                )
            thresholds = {
                fold: float(accumulator.compute()["threshold"])
                for fold, accumulator in fold_accumulator.items()
            }
            alarm = {"A": 0, "B": 0}
            total = 0
            for row, frame_id in enumerate(self.pure_frame_id.tolist()):
                expected = int(self.pure_count[row])
                if expected == 0:
                    continue
                source = self.sequence.source_frame(int(frame_id))
                inputs = self._input(source)
                scores = self._scores(model, inputs)
                slots = np.asarray(inputs["slots"], dtype=np.int64)
                full = np.zeros(int(getattr(source, "slot_count")), dtype=np.float32)
                full[slots] = scores
                canonical_mask = np.unpackbits(
                    self.pure_mask_packed[row], bitorder="little"
                )[: self.canonical_by_slot.size]
                selected_slots = np.flatnonzero(
                    canonical_mask[self.canonical_by_slot]
                )
                if selected_slots.size != expected:
                    raise TrainingError("E61 pure-normal mask count changed")
                selected_scores = full[selected_slots]
                alarm["A"] += int(np.count_nonzero(selected_scores > thresholds["A"]))
                alarm["B"] += int(np.count_nonzero(selected_scores > thresholds["B"]))
                total += expected
            if total != 48_828_507:
                raise TrainingError("pure-normal development count changed")
            cross_fit_fpr = 0.5 * (alarm["A"] + alarm["B"]) / total
            return DevelopmentEvidence(
                tuple(sorted(metrics, key=lambda item: item.world_id)),
                {
                    "cross_fit_FPR": float(cross_fit_fpr),
                    "threshold_A": thresholds["A"],
                    "threshold_B": thresholds["B"],
                    "pure_normal_points": float(total),
                },
            )
        finally:
            model.train(was_training)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


class E63B3DevelopmentEvaluator(E63B1DevelopmentEvaluator):
    """Evaluate B3 q=0 on the frozen common and P1-intersection domains."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        p1 = self.protocol.development["exploration_track"][
            "full_ajae_x_freeze"
        ]["f1_entry"]["p1_boundary_amendment"]
        pure = p1["pure_normal_q0"]
        frame_min, frame_max = map(int, pure["frame_range"])
        full = self.protocol.development["exploration_track"][
            "full_ajae_x_freeze"
        ]
        fast = full.get("fast_viability")
        if isinstance(fast, Mapping):
            lite = fast.get("lite_safety")
            if not isinstance(lite, Mapping):
                raise TrainingError("fast viability lite-safety identity is incomplete")
            selected = np.asarray(lite.get("pure_normal_frame_ids", ()), dtype=np.int16)
            keep = (
                np.isin(self.pure_frame_id, selected)
                & (self.pure_frame_id >= frame_min)
                & (self.pure_frame_id <= frame_max)
            )
            expected_frames = int(lite.get("q0_frame_count", 0))
            expected_points = int(lite.get("q0_point_count", 0))
            evaluator_version = "E63-B3-q0-development-evaluator-v5-fast-lite"
            pure_rule = "frozen E76-X-lite frames intersected with legal B3 q=0 centers"
        else:
            keep = (self.pure_frame_id >= frame_min) & (
                self.pure_frame_id <= frame_max
            )
            expected_frames = frame_max - frame_min + 1
            expected_points = int(pure["points"])
            evaluator_version = "E63-B3-q0-development-evaluator-full-first-v4"
            pure_rule = "P1 train/201 frame 6-679 q=0 intersection"
        self.pure_frame_id = self.pure_frame_id[keep]
        self.pure_mask_packed = self.pure_mask_packed[keep]
        self.pure_count = self.pure_count[keep]
        self.pure_expected = expected_points
        if (
            self.pure_frame_id.size != expected_frames
            or int(self.pure_count.sum()) != self.pure_expected
        ):
            raise TrainingError("P1 B3 pure-normal q=0 domain changed")
        try:
            from .scene import canonical_ray_mapping_digest
        except ImportError:  # pragma: no cover
            from scene import canonical_ray_mapping_digest  # type: ignore[no-redef]
        self.ray_mapping_digest = canonical_ray_mapping_digest(
            self.canonical_by_slot
        )
        self.condition = experiment_condition("B3")
        self.scientific_identity = {
            "version": evaluator_version,
            "e57_artifact_sha256": self.scientific_identity[
                "e57_artifact_sha256"
            ],
            "e63_artifact_sha256": self.scientific_identity[
                "e63_artifact_sha256"
            ],
            "e61_artifact_sha256": self.scientific_identity[
                "e61_artifact_sha256"
            ],
            "eligible_world_ids": self.world_id[self.eligible]
            .astype(int)
            .tolist(),
            "target": "B3 q=0 only",
            "threshold_rule": "official first ROC threshold with TPR strictly greater than 0.95",
            "pure_normal_cross_fit": (
                f"mean FPR on the {pure_rule} under the two opposite-fold "
                "proxy thresholds"
            ),
            "pure_normal_points": self.pure_expected,
            "threshold_comparison": "score strictly greater than threshold",
        }

    def _window_probabilities(
        self,
        model: nn.Module,
        sources: Sequence[object],
        center: int,
        *,
        input_cache: OrderedDict[int, dict[str, object]] | None = None,
    ) -> tuple[np.ndarray, ...]:
        try:
            from .scene import assemble_window
        except ImportError:  # pragma: no cover
            from scene import assemble_window  # type: ignore[no-redef]
        if len(sources) != 5:
            raise TrainingError("B3 development requires exactly five source frames")
        inputs: list[dict[str, object]] = []
        for source in sources:
            frame_id = int(getattr(source, "frame_id"))
            item = input_cache.get(frame_id) if input_cache is not None else None
            if item is None:
                item = self._input(source)
                if input_cache is not None:
                    input_cache[frame_id] = item
                    input_cache.move_to_end(frame_id)
                    while len(input_cache) > 7:
                        input_cache.popitem(last=False)
            inputs.append(item)
        window = assemble_window(
            self.sequence.spec,
            center,
            sources,
            condition="B3",
            canonical_ray_by_slot=self.canonical_by_slot,
            ray_mapping_audited=True,
            ray_mapping_digest=self.ray_mapping_digest,
        )
        coordinates = torch.as_tensor(
            np.asarray(window.points.coordinates_center).copy(),
            device=self.device,
            dtype=torch.float32,
        )
        times = torch.as_tensor(
            np.asarray(window.points.relative_time).copy(),
            device=self.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            logits = model(
                coordinates,
                times,
                torch.cat([item["stu_features"] for item in inputs]).to(
                    self.device
                ),
                torch.cat([item["normal_evidence"] for item in inputs]).to(
                    self.device
                ),
                torch.cat([item["assignment"] for item in inputs]).to(
                    self.device
                ),
                torch.cat([item["no_object"] for item in inputs]).to(
                    self.device
                ),
                torch.cat([item["intensity"] for item in inputs]).to(
                    self.device
                ),
                cross_frame_enabled=True,
            )
        counts = tuple(
            int(np.asarray(getattr(source, "real_slots")).size) for source in sources
        )
        if logits.shape != times.shape or sum(counts) != logits.numel() or not bool(
            torch.isfinite(logits).all()
        ):
            raise TrainingError("B3 development logits are invalid")
        probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        offsets = np.cumsum((0, *counts), dtype=np.int64)
        return tuple(
            probabilities[offsets[index] : offsets[index + 1]]
            for index in range(5)
        )

    def _window_scores(
        self,
        model: nn.Module,
        sources: Sequence[object],
        center: int,
        *,
        input_cache: OrderedDict[int, dict[str, object]] | None = None,
    ) -> np.ndarray:
        return self._window_probabilities(
            model, sources, center, input_cache=input_cache
        )[2]

    def sequence_b3_b4_scores(
        self,
        model: nn.Module,
        sequence: object,
        target_frames: Sequence[int],
    ) -> dict[int, tuple[np.ndarray | None, np.ndarray, np.ndarray]]:
        """Score frozen targets once per legal window and fuse B4 occurrences."""

        targets = tuple(sorted(set(int(value) for value in target_frames)))
        legal = tuple(
            int(value)
            for value in getattr(sequence, "spec").legal_anchors(RELATIVE_TIMES)
        )
        legal_set = frozenset(legal)
        target_set = frozenset(targets)
        centers = tuple(
            center
            for center in legal
            if any(abs(center - target) <= 2 for target in target_set)
        )
        sums: dict[int, np.ndarray] = {}
        counts: dict[int, np.ndarray] = {}
        direct: dict[int, np.ndarray] = {}
        cache: OrderedDict[int, dict[str, object]] = OrderedDict()
        for center in centers:
            sources = tuple(
                getattr(sequence, "source_frame")(center + offset)
                for offset in RELATIVE_TIMES
            )
            probabilities = self._window_probabilities(
                model, sources, center, input_cache=cache
            )
            for local, source in enumerate(sources):
                frame_id = int(getattr(source, "frame_id"))
                if frame_id not in target_set:
                    continue
                values = probabilities[local]
                if local == 2:
                    direct[frame_id] = values.copy()
                if frame_id not in sums:
                    sums[frame_id] = np.zeros(values.shape, dtype=np.float64)
                    counts[frame_id] = np.zeros(values.shape, dtype=np.uint8)
                if sums[frame_id].shape != values.shape:
                    raise TrainingError("one frame changed point identity across windows")
                sums[frame_id] += values
                counts[frame_id] += np.uint8(1)
        result: dict[int, tuple[np.ndarray | None, np.ndarray, np.ndarray]] = {}
        for frame_id in targets:
            if frame_id not in sums or not np.all(counts[frame_id] > 0):
                raise TrainingError("B4 target has no legal window occurrence")
            if (frame_id in legal_set) != (frame_id in direct):
                raise TrainingError("B3 q=0 target legality changed")
            result[frame_id] = (
                direct.get(frame_id),
                (sums[frame_id] / counts[frame_id]).astype(np.float32),
                counts[frame_id].copy(),
            )
        return result

    def development_b3_b4_scores(
        self,
        model: nn.Module,
        world_ids: Sequence[int],
    ) -> tuple[dict[str, object], ...]:
        """Evaluate B3 q=0 and B4 fusion on fixed rendered worlds."""

        try:
            from .render import WorldSpec, render_frame
        except ImportError:  # pragma: no cover
            from render import WorldSpec, render_frame  # type: ignore[no-redef]
        requested = tuple(int(value) for value in world_ids)
        rows = {int(self.world_id[row]): row for row in np.flatnonzero(self.eligible)}
        if len(set(requested)) != len(requested) or any(
            value not in rows for value in requested
        ):
            raise TrainingError("development world selection is invalid")
        output: list[dict[str, object]] = []
        for world_id in requested:
            row = rows[world_id]
            target = int(self.center[row])
            world = WorldSpec.from_dict(json.loads(str(self.world_json[row])))
            rendered = {
                frame_id: render_frame(
                    self.sequence.source_frame(frame_id), world, self.grid, self.sensor
                )
                for frame_id in range(target - 4, target + 5)
            }
            cache: OrderedDict[int, dict[str, object]] = OrderedDict()
            occurrences: list[np.ndarray] = []
            b3: np.ndarray | None = None
            for center in range(target - 2, target + 3):
                sources = tuple(
                    rendered[center + offset].source for offset in RELATIVE_TIMES
                )
                probabilities = self._window_probabilities(
                    model, sources, center, input_cache=cache
                )
                local = target - center + 2
                occurrences.append(probabilities[local])
                if center == target:
                    b3 = probabilities[2]
            if b3 is None or any(value.shape != b3.shape for value in occurrences):
                raise TrainingError("development B3/B4 point identity changed")
            target_render = rendered[target]
            source = target_render.source
            slots = np.asarray(source.real_slots, dtype=np.int64)
            output.append(
                {
                    "world_id": world_id,
                    "center_frame": target,
                    "slots": slots,
                    "canonical_ray": self.canonical_by_slot[slots],
                    "xyz": np.asarray(source.xyzi)[slots, :3],
                    "semantic": (
                        np.asarray(target_render.packed_labels, dtype=np.uint32)[slots]
                        & np.uint32(0xFFFF)
                    ),
                    "normal_control": np.asarray(
                        target_render.normal_control_mask, dtype=np.bool_
                    )[slots],
                    "B3": b3,
                    "B4": np.mean(
                        np.stack(occurrences, axis=0), axis=0, dtype=np.float64
                    ).astype(np.float32),
                    "B4_occurrence_count": np.full(b3.shape, 5, dtype=np.uint8),
                }
            )
        return tuple(output)

    def __call__(
        self,
        model: nn.Module,
        world_index: int,
        seed: int,
        condition: ExperimentCondition,
    ) -> DevelopmentEvidence:
        del world_index, seed
        if condition.name != "B3":
            raise TrainingError("this evaluator instance is bound to B3")
        try:
            from .evaluate import PointMetricAccumulator
            from .render import WorldSpec, render_frame
        except ImportError:  # pragma: no cover
            from evaluate import PointMetricAccumulator  # type: ignore[no-redef]
            from render import WorldSpec, render_frame  # type: ignore[no-redef]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = model.training
        model.eval()
        try:
            metrics: list[DevelopmentWorldMetrics] = []
            fold_accumulator = {
                "A": PointMetricAccumulator(self.protocol),
                "B": PointMetricAccumulator(self.protocol),
            }
            for row in np.flatnonzero(self.eligible):
                center = int(self.center[row])
                world = WorldSpec.from_dict(json.loads(str(self.world_json[row])))
                rendered = tuple(
                    render_frame(
                        self.sequence.source_frame(frame_id),
                        world,
                        self.grid,
                        self.sensor,
                    )
                    for frame_id in range(center - 2, center + 3)
                )
                scores = self._window_scores(
                    model, tuple(item.source for item in rendered), center
                )
                source = rendered[2].source
                slots = np.asarray(source.real_slots, dtype=np.int64)
                xyz = np.asarray(source.xyzi)[slots, :3]
                semantic = (
                    np.asarray(rendered[2].packed_labels, dtype=np.uint32)[slots]
                    & np.uint32(0xFFFF)
                )
                accumulator = PointMetricAccumulator(self.protocol)
                accumulator.update(xyz, scores, semantic)
                result = accumulator.compute()
                if result.get("accepted_frames") != 1:
                    raise TrainingError("one B3 development world was not accepted")
                metrics.append(
                    DevelopmentWorldMetrics(
                        int(self.world_id[row]),
                        {
                            "AP": float(result["AP"]),
                            "AUROC": float(result["AUROC"]),
                            "FPR95": float(result["FPR95"]),
                        },
                    )
                )
                fold_accumulator[bytes(self.fold[row]).decode("ascii")].update(
                    xyz, scores, semantic
                )
            thresholds = {
                fold: float(accumulator.compute()["threshold"])
                for fold, accumulator in fold_accumulator.items()
            }
            alarm = {"A": 0, "B": 0}
            total = 0
            cache: OrderedDict[int, dict[str, object]] = OrderedDict()
            for row, center in enumerate(self.pure_frame_id.tolist()):
                expected = int(self.pure_count[row])
                sources = tuple(
                    self.sequence.source_frame(frame_id)
                    for frame_id in range(center - 2, center + 3)
                )
                scores = self._window_scores(
                    model, sources, center, input_cache=cache
                )
                source = sources[2]
                slots = np.asarray(source.real_slots, dtype=np.int64)
                full = np.zeros(int(source.slot_count), dtype=np.float32)
                full[slots] = scores
                canonical_mask = np.unpackbits(
                    self.pure_mask_packed[row], bitorder="little"
                )[: self.canonical_by_slot.size]
                selected_slots = np.flatnonzero(
                    canonical_mask[self.canonical_by_slot]
                )
                if selected_slots.size != expected:
                    raise TrainingError("P1 pure-normal mask count changed")
                selected_scores = full[selected_slots]
                alarm["A"] += int(
                    np.count_nonzero(selected_scores > thresholds["A"])
                )
                alarm["B"] += int(
                    np.count_nonzero(selected_scores > thresholds["B"])
                )
                total += expected
            if total != self.pure_expected:
                raise TrainingError("P1 B3 pure-normal total changed")
            cross_fit_fpr = 0.5 * (alarm["A"] + alarm["B"]) / total
            return DevelopmentEvidence(
                tuple(sorted(metrics, key=lambda item: item.world_id)),
                {
                    "cross_fit_FPR": float(cross_fit_fpr),
                    "threshold_A": thresholds["A"],
                    "threshold_B": thresholds["B"],
                    "pure_normal_points": float(total),
                },
            )
        finally:
            model.train(was_training)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


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
    if development_evaluator is not None:
        raise TrainingError(
            "formal training constructs its identity-bound E63 evaluator internally"
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

    if condition.name not in {"B1", "B3"}:
        raise TrainingError(
            "the current formal evaluator implementation is qualified only for B1 and B3"
        )
    evaluator_encoder = FrozenSTUPointEncoder.from_protocol(
        protocol, project_root=project_root
    ).to(runtime_device)
    evaluator_type = (
        E63B1DevelopmentEvaluator
        if condition.name == "B1"
        else E63B3DevelopmentEvaluator
    )
    development_evaluator = evaluator_type(
        protocol=protocol,
        project_root=project_root,
        data_root=data_root,
        device=runtime_device,
        encoder=evaluator_encoder,
        grid=ray_grid,
        sensor=sensor,
        canonical_by_slot=canonical_ray_by_slot,
    )
    raw_evaluator_identity = development_evaluator.scientific_identity
    evaluator_scientific_identity = _plain_json_object(
        "development evaluator identity", raw_evaluator_identity
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
        _atomic_json(
            report_dir / f"{seed}.json", _finite_world_report_document(report)
        )
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
    schedule = training_schedule(protocol, condition, config, maximum_worlds)
    run_root = (
        project_root / schedule.output_directory
        if schedule.output_directory is not None
        else project_root / config.output_dir / condition.name
    )
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
        "training_schedule": asdict(schedule),
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
            schedule=schedule,
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


def run_b3_semantic_preflight(
    protocol_path: Path | str,
    *,
    data_root: Path | str,
    output_path: Path | str,
    maximum_worlds: int,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Exercise one frozen mixed B3 window through the formal optimizer path."""

    try:
        from .protocol import load_protocol
    except ImportError:  # pragma: no cover
        from protocol import load_protocol  # type: ignore[no-redef]
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    full = protocol.development["exploration_track"]["full_ajae_x_freeze"]
    p2 = full["f1_entry"]["p2_semantic_training_preflight"]
    if (
        protocol.development["exploration_track"]["current_node"]
        != "AJAE-F1-X_entry_P2"
        or full["status"] != "AJAE-F1-X_entry_P2_frozen"
        or p2["status"] != "frozen_before_execution"
    ):
        raise TrainingError("AJAE-F1-X P2 is not frozen as the current entry step")
    condition = experiment_condition("B3")
    preflight = validate_e63_formal_preflight(protocol)
    build = build_formal_training(
        protocol,
        preflight=preflight,
        data_root=data_root,
        condition=condition,
        maximum_worlds=maximum_worlds,
        development_evaluator=None,
        device=device,
    )
    seed = int(p2["seed"])
    world_index = int(p2["world_index"])
    world_kind = str(p2["world_type"])
    world_seed = _derived_seed(seed, world_index, 0)
    legal_centers = tuple(range(2, 447))
    blocks = shuffled_center_blocks(
        legal_centers,
        build.config.chunk_centers,
        _derived_seed(seed, world_index, 1),
    )
    center = int(blocks[0][0])
    if (
        build.config.world_type_cycle.index("mixed") != world_index
        or world_kind != "mixed"
        or world_seed != int(p2["world_seed"])
        or center != int(p2["center_frame"])
        or tuple(center + value for value in condition.frame_offsets)
        != tuple(p2["frame_ids"])
    ):
        raise TrainingError("AJAE-F1-X P2 frozen world/window identity changed")

    _seed_everything(seed)
    trainer = build.trainer_factory(seed, condition)
    world = build.world_factory(world_kind, world_seed)
    trainer._require_world(world, world_kind, world_seed)
    encoder_before = _tensor_state_sha256(trainer.encoder.state_dict())
    model_before = _tensor_state_sha256(trainer.model.state_dict())
    torch.cuda.reset_peak_memory_stats(trainer.device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    batch = trainer._window_data(world, center)
    rendered = tuple(
        trainer._render(world, center + offset)
        for offset in condition.frame_offsets
    )
    control_parts: list[np.ndarray] = []
    proxy_parts: list[np.ndarray] = []
    semantic_parts: list[np.ndarray] = []
    range_parts: list[np.ndarray] = []
    for item in rendered:
        source = item.source
        slots = np.asarray(source.real_slots, dtype=np.int64)
        control_parts.append(np.asarray(item.normal_control_mask, dtype=np.bool_)[slots])
        proxy_parts.append(np.asarray(item.anomaly_proxy_mask, dtype=np.bool_)[slots])
        semantic_parts.append(
            np.asarray(item.packed_labels, dtype=np.uint32)[slots]
            & np.uint32(0xFFFF)
        )
        range_parts.append(
            np.linalg.norm(np.asarray(source.xyzi)[slots, :3], axis=1)
        )
    control = np.concatenate(control_parts)
    proxy = np.concatenate(proxy_parts)
    semantic = np.concatenate(semantic_parts)
    distance = np.concatenate(range_parts)
    valid_expected = (control | proxy | (semantic != 0))
    valid_expected &= distance >= trainer.minimum_range_m
    valid_expected &= distance <= trainer.maximum_range_m
    target = batch.targets.detach().cpu().numpy()
    valid = batch.valid.detach().cpu().numpy()
    visible_counts = p2["identity_only_visible_counts"]
    label_error = int(
        np.any(control & proxy)
        or not np.array_equal(target, proxy)
        or not np.array_equal(valid, valid_expected)
        or np.any(target[control])
        or np.any(target[(~control & ~proxy) & valid_expected])
        or int(control.sum()) != int(visible_counts["normal_control"])
        or int(proxy.sum()) != int(visible_counts["anomaly_proxy"])
    )

    supervised = torch.zeros_like(batch.valid)
    for relative_time in condition.supervised_times:
        supervised |= batch.relative_times == relative_time
    observed_times, counts_by_time = torch.unique(
        batch.relative_times, sorted=True, return_counts=True
    )
    supervision_error = int(
        tuple(observed_times.detach().cpu().tolist()) != RELATIVE_TIMES
        or not bool(torch.equal(batch.valid & supervised, batch.valid))
        or bool(torch.any(counts_by_time == 0))
    )

    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        no_cross = trainer.model(
            batch.coordinates,
            batch.relative_times,
            batch.stu_features,
            batch.normal_evidence,
            batch.assignment_reliability,
            batch.no_object_reliability,
            batch.intensity,
            cross_frame_enabled=False,
        )
    # The control forward is retained only as logits. Release its cached CUDA
    # blocks before constructing the substantially larger five-frame graph.
    torch.cuda.empty_cache()
    logits = trainer.model(
        batch.coordinates,
        batch.relative_times,
        batch.stu_features,
        batch.normal_evidence,
        batch.assignment_reliability,
        batch.no_object_reliability,
        batch.intensity,
        cross_frame_enabled=True,
    )
    loss_mask = batch.valid & supervised
    loss = balanced_bce_loss(logits, batch.targets, loss_mask)
    element = F.binary_cross_entropy_with_logits(
        logits, batch.targets.to(logits.dtype), reduction="none"
    )
    expected_terms = [
        element[mask].mean()
        for mask in (
            loss_mask & batch.targets,
            loss_mask & ~batch.targets,
        )
        if bool(mask.any())
    ]
    expected_loss = torch.stack(expected_terms).mean()
    loss_error = int(
        not bool(torch.isfinite(loss))
        or not bool(torch.equal(loss, expected_loss))
    )
    q0 = batch.relative_times == 0
    q0_difference = torch.abs(logits.detach()[q0] - no_cross[q0])
    dependency_error = int(
        not bool(torch.isfinite(q0_difference).all())
        or not bool(torch.any(q0_difference > 0.0))
    )

    (loss / build.config.gradient_accumulation).backward()
    cross_gradient_norms = []
    for name, parameter in trainer.model.named_parameters():
        if ".cross_gate." in name and parameter.grad is not None:
            cross_gradient_norms.append(float(torch.linalg.vector_norm(parameter.grad)))
    temporal_gradient_error = int(
        not cross_gradient_norms
        or not all(math.isfinite(value) for value in cross_gradient_norms)
        or not any(value > 0.0 for value in cross_gradient_norms)
    )
    stu_gradient_error = int(
        any(parameter.requires_grad for parameter in trainer.encoder.parameters())
        or any(parameter.grad is not None for parameter in trainer.encoder.parameters())
    )
    trainer.accumulated_windows = 1
    trainer._optimizer_step(partial=True)
    model_after = _tensor_state_sha256(trainer.model.state_dict())
    encoder_after = _tensor_state_sha256(trainer.encoder.state_dict())
    stu_update_error = int(encoder_before != encoder_after)
    model_update_error = int(model_before == model_after)
    finished.record()
    torch.cuda.synchronize(trainer.device)

    arrays = {
        "frame_id": np.asarray(p2["frame_ids"], dtype=np.int16),
        "relative_time": observed_times.detach().cpu().numpy().astype(np.int8),
        "points_by_relative_time": counts_by_time.detach()
        .cpu()
        .numpy()
        .astype(np.int32),
        "valid_points_by_relative_time": np.asarray(
            [
                int(
                    torch.count_nonzero(
                        batch.valid & (batch.relative_times == relative_time)
                    )
                )
                for relative_time in RELATIVE_TIMES
            ],
            dtype=np.int32,
        ),
        "control_points": np.asarray(int(control.sum()), dtype=np.int32),
        "proxy_points": np.asarray(int(proxy.sum()), dtype=np.int32),
        "valid_real_normal_points": np.asarray(
            int(np.count_nonzero((~control & ~proxy) & valid_expected)),
            dtype=np.int32,
        ),
        "balanced_bce": np.asarray(float(loss.detach().cpu()), dtype=np.float64),
        "q0_changed_points": np.asarray(
            int(torch.count_nonzero(q0_difference > 0.0)), dtype=np.int32
        ),
        "q0_max_absolute_difference": np.asarray(
            float(q0_difference.max().detach().cpu()), dtype=np.float64
        ),
        "cross_gate_gradient_norm": np.asarray(
            cross_gradient_norms, dtype=np.float64
        ),
        "label_error_count": np.asarray(label_error, dtype=np.int16),
        "supervision_error_count": np.asarray(supervision_error, dtype=np.int16),
        "balanced_bce_error_count": np.asarray(loss_error, dtype=np.int16),
        "stu_gradient_error_count": np.asarray(stu_gradient_error, dtype=np.int16),
        "stu_update_error_count": np.asarray(stu_update_error, dtype=np.int16),
        "model_update_error_count": np.asarray(model_update_error, dtype=np.int16),
        "temporal_gradient_error_count": np.asarray(
            temporal_gradient_error, dtype=np.int16
        ),
        "q0_dependency_error_count": np.asarray(dependency_error, dtype=np.int16),
    }
    total_errors = sum(
        int(np.asarray(value).sum())
        for name, value in arrays.items()
        if name.endswith("error_count")
    )
    scientific_hash = _array_sha256(arrays)
    result = {
        "experiment": "AJAE-F1-X-entry-P2",
        "status": "PASS" if total_errors == 0 else "IMPLEMENTATION_DEFECT",
        "passed": total_errors == 0,
        "total_errors": total_errors,
        "seed": seed,
        "world_index": world_index,
        "world_type": world_kind,
        "world_seed": world_seed,
        "center_frame": center,
        "model_quality_evaluated": False,
        "protocol_sha256": _sha256_file(protocol_file),
        "stu_before_sha256": encoder_before,
        "stu_after_sha256": encoder_after,
        "model_before_sha256": model_before,
        "model_after_sha256": model_after,
        "scientific_array_sha256": scientific_hash,
        "gpu_peak_memory_bytes": int(
            torch.cuda.max_memory_allocated(trainer.device)
        ),
        "milliseconds": float(started.elapsed_time(finished)),
    }
    _atomic_npz(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def train_all_seeds(
    config: TrainConfig,
    condition: ExperimentCondition,
    trainer_factory: Callable[[int, ExperimentCondition], AJAETrainer],
    world_factory: Callable[[str, int], object],
    *,
    preflight: FormalPreflightProof,
    maximum_worlds: int,
    resume: bool = False,
    seeds: Sequence[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Run the requested frozen seeds independently and retain their results."""

    _require_preflight_proof(preflight, config)
    if not condition.trainable:
        raise TrainingError(f"{condition.name} cannot be trained independently")
    if type(maximum_worlds) is not int or maximum_worlds != preflight.maximum_worlds:
        raise TrainingError(
            "maximum_worlds must equal the protocol-frozen training limit"
        )
    if type(resume) is not bool:
        raise TrainingError("resume must be boolean")
    selected_seeds = tuple(config.seeds if seeds is None else seeds)
    if (
        not selected_seeds
        or len(set(selected_seeds)) != len(selected_seeds)
        or any(seed not in config.seeds for seed in selected_seeds)
    ):
        raise TrainingError("selected seeds must be a unique subset of frozen seeds")
    results: dict[int, dict[str, Any]] = {}
    model_refs: list[weakref.ReferenceType[nn.Module]] = []
    run_dirs: set[Path] = set()
    for seed in selected_seeds:
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
        model_path = trainer.run_dir / "model.pt"
        result_path = trainer.run_dir / "result.json"
        if resume:
            if progress.is_file():
                if model_path.exists() or result_path.exists():
                    raise TrainingError(
                        f"seed {seed} mixes progress and completed-run artifacts"
                    )
                start_world = trainer.load_progress()
            elif model_path.is_file() and result_path.is_file():
                saved_model = torch.load(
                    model_path, map_location="cpu", weights_only=True
                )
                saved_result = json.loads(result_path.read_text(encoding="utf-8"))
                shared_completion = (
                    saved_result.get("status") == "completed"
                    and saved_model.get("format") == MODEL_FORMAT
                    and saved_result.get("format") == RUN_FORMAT
                    and saved_model.get("completion_id")
                    == saved_result.get("completion_id")
                    and saved_model.get("seed") == seed
                    and saved_result.get("seed") == seed
                    and saved_model.get("config") == asdict(config)
                    and saved_model.get("condition") == condition.to_dict()
                    and saved_result.get("condition") == condition.to_dict()
                    and saved_model.get("scientific_identity")
                    == trainer.scientific_identity
                    and saved_result.get("scientific_identity")
                    == trainer.scientific_identity
                )
                if not shared_completion:
                    raise TrainingError(
                        f"seed {seed} completed artifacts do not match this run"
                    )
                results[seed] = saved_result
                continue
            elif model_path.exists() or result_path.exists():
                raise TrainingError(
                    f"seed {seed} has an incomplete completed-run artifact pair"
                )
            else:
                # A resumed multi-seed run may legitimately reach a seed that
                # had not started when the earlier process was interrupted.
                start_world = 0
        else:
            if progress.exists() or model_path.exists() or result_path.exists():
                raise TrainingError(f"seed {seed} run directory already contains state")
            start_world = 0
        seed_result = trainer.fit(
            world_factory,
            maximum_worlds=maximum_worlds,
            start_world=start_world,
        )
        results[seed] = seed_result
        if seed_result.get("status") == "paused_for_external_evaluation":
            return results
        if seed_result.get("status") != "completed":
            raise TrainingError(
                f"seed {seed} exhausted the frozen world budget; no formal model was published"
            )
    if set(results) != set(selected_seeds):
        raise TrainingError("formal result omitted one or more selected seeds")
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


def validate_e63_formal_preflight(protocol: object) -> FormalPreflightProof:
    """Bind formal training to E57/E63 instead of the retired dev.json."""

    development = getattr(protocol, "development", None)
    training = getattr(protocol, "training", None)
    decision_gates = getattr(protocol, "decision_gates", None)
    if not isinstance(development, Mapping) or not isinstance(training, Mapping):
        raise TrainingError("formal protocol lacks development or training rules")
    freeze = development.get("e63_freeze")
    selection = development.get("checkpoint_selection")
    fixed = development.get("fixed_world_evaluation")
    if (
        not isinstance(freeze, Mapping)
        or freeze.get("status") != "formal_pass"
        or not isinstance(selection, Mapping)
        or not isinstance(fixed, Mapping)
        or fixed.get("status") != "frozen_before_training"
    ):
        raise TrainingError("E63 formal identities are not complete")
    selection = _plain_json_object("checkpoint selection", selection)
    _checkpoint_selection_tolerance(selection)
    project_root = Path(getattr(protocol, "path")).parent
    source = freeze.get("source_worlds")
    identity = freeze.get("identity_artifact")
    smoke = training.get("e73_smoke")
    if (
        not isinstance(source, Mapping)
        or not isinstance(identity, Mapping)
        or not isinstance(smoke, Mapping)
        or smoke.get("status") != "formal_pass"
    ):
        raise TrainingError("E57, E63, or E73 formal identity is missing")
    e57_path = project_root / str(source["artifact"])
    e63_path = project_root / str(identity["path"])
    e73_result = smoke.get("result")
    if not isinstance(e73_result, Mapping):
        raise TrainingError("E73 formal result is missing")
    e73_path = project_root / str(e73_result["path"])
    for name, path, expected in (
        ("E57", e57_path, source.get("artifact_sha256")),
        ("E63", e63_path, identity.get("artifact_sha256")),
        ("E73", e73_path, e73_result.get("artifact_sha256")),
    ):
        if not path.is_file() or _sha256_file(path) != expected:
            raise TrainingError(f"{name} formal artifact identity changed")
    with np.load(e57_path, allow_pickle=False) as archive:
        world_ids = np.asarray(archive["selected_world_id"], dtype=np.int16)
        world_json = np.asarray(archive["selected_world_json"])
    with np.load(e63_path, allow_pickle=False) as archive:
        frozen_ids = np.asarray(archive["world_id"], dtype=np.int16)
        eligible = np.asarray(archive["common_domain_eligible"], dtype=np.bool_)
        folds = np.asarray(archive["safety_fold"])
    if (
        not np.array_equal(world_ids, np.arange(24, dtype=np.int16))
        or not np.array_equal(world_ids, frozen_ids)
        or world_json.shape != (24,)
        or eligible.shape != (24,)
        or int(eligible.sum()) != 23
        or world_ids[~eligible].tolist() != [5]
        or folds.shape != (24,)
        or int(np.count_nonzero(folds == b"A")) != 12
        or int(np.count_nonzero(folds == b"B")) != 12
    ):
        raise TrainingError("E57/E63 development identity arrays changed")
    criteria = (
        decision_gates.get("criteria")
        if isinstance(decision_gates, Mapping)
        else None
    )
    if not isinstance(criteria, Mapping) or criteria.get("status") != (
        "frozen_before_training"
    ):
        raise TrainingError("scientific decision criteria are not frozen")
    config = TrainConfig.from_protocol(protocol)
    converter = getattr(protocol, "plain_document", None)
    document = converter() if callable(converter) else None
    if not isinstance(document, Mapping):
        raise TrainingError("protocol does not expose its complete document")
    development_identity = {
        "version": "E63-v2-formal-training-input-v1",
        "e57_path": str(e57_path),
        "e57_sha256": source["artifact_sha256"],
        "e63_path": str(e63_path),
        "e63_sha256": identity["artifact_sha256"],
        "eligible_world_ids": world_ids[eligible].astype(int).tolist(),
        "safety_fold": [bytes(value).decode("ascii") for value in folds],
    }
    return FormalPreflightProof(
        _FORMAL_PREFLIGHT_SEAL,
        _canonical_json_object("protocol", document),
        _canonical_json_object("E63 development identity", development_identity),
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
    seed: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Run formal training only after validating the exact frozen inputs."""

    try:
        from .protocol import load_protocol
    except ImportError:  # pragma: no cover - direct script execution
        from protocol import load_protocol

    protocol = load_protocol(protocol_path)
    if development_path is not None:
        raise TrainingError(
            "formal training uses the E57/E63 artifacts; retired dev.json overrides are forbidden"
        )
    preflight = validate_e63_formal_preflight(protocol)
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
        seeds=None if seed is None else (seed,),
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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--semantic-preflight-output", type=Path)
    args = parser.parse_args()
    try:
        if args.semantic_preflight_output is not None:
            if (
                args.development is not None
                or args.resume
                or args.seed is not None
                or args.condition != "B3"
            ):
                raise TrainingError(
                    "semantic preflight requires B3 without development override or resume"
                )
            result = run_b3_semantic_preflight(
                args.protocol,
                data_root=args.data_root,
                output_path=args.semantic_preflight_output,
                maximum_worlds=args.max_worlds,
                device=args.device,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            if not result["passed"]:
                raise SystemExit(1)
            return
        results = run_formal_training(
            args.protocol,
            development_path=args.development,
            data_root=args.data_root,
            condition_name=args.condition,
            maximum_worlds=args.max_worlds,
            development_evaluator=None,
            device=args.device,
            resume=args.resume,
            seed=args.seed,
        )
    except (OSError, TypeError, ValueError, TrainingError) as error:
        raise SystemExit(
            f"Formal training refused: {error}. No run state was written."
        ) from error
    print(
        json.dumps(
            {
                "status": (
                    "paused_for_external_evaluation"
                    if any(
                        result.get("status") == "paused_for_external_evaluation"
                        for result in results.values()
                    )
                    else "completed"
                ),
                "seeds": sorted(results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
