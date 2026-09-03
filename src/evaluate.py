#!/usr/bin/env python3
"""Schema-31 window inference, point-identity fusion, and STU metrics."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_curve

try:
    from .protocol import ExperimentCondition
except ImportError:  # pragma: no cover - direct module execution
    from protocol import ExperimentCondition


FUSION_SEMANTICS = "all_occurrence_probability_mean_within_world"
B0_FUSION_SEMANTICS = "all_occurrence_raw_score_mean_within_world"
B0_FUSION_VALUE = "raw_finite_frozen_STU_official_MaxLogit_score"
AJAE_FUSION_VALUE = "sigmoid_of_each_anomaly_logit"
FUSION_REDUCTION = "equal_arithmetic_mean_over_every_legal_window_occurrence"
PREDICTION_COVERAGE_FORMAT = "ajae-schema31-prediction-coverage-v2"
METHOD_FREEZE_FORMAT = "ajae-schema31-method-freeze-v1"
METHOD_FREEZE_STATUS = "frozen_before_sealed_data_access"
MOVING_NORMAL_SEMANTICS = (252, 253, 254, 255, 256, 257, 258, 259)


class _EvaluationRange(Protocol):
    minimum_range_m: float
    maximum_range_m: float
    minimum_anomaly_points: int


class _Protocol(Protocol):
    evaluation: _EvaluationRange


class _SourceFrame(Protocol):
    xyzi: np.ndarray


class EvaluationError(ValueError):
    """Report an invalid prediction, identity, or evaluation quantity."""


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationError(f"{name} must be a lowercase SHA-256")
    return value


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        _plain_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fusion_value(condition: ExperimentCondition | str) -> str:
    selected = _condition(condition)
    return B0_FUSION_VALUE if selected is ExperimentCondition.B0 else AJAE_FUSION_VALUE


def _fusion_semantics(condition: ExperimentCondition | str) -> str:
    selected = _condition(condition)
    return (
        B0_FUSION_SEMANTICS if selected is ExperimentCondition.B0 else FUSION_SEMANTICS
    )


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    """Frozen scientific inputs that uniquely identify one inference method."""

    protocol_schema: int
    protocol_identity: str
    condition: str
    fusion_value: str
    model_class: str | None
    model_state_sha256: str | None
    stu_class: str
    stu_checkpoint_sha256: str
    stu_model_state_sha256: str
    stu_source_manifest_sha256: str
    calibration_sha256: str
    ray_mapping_sha256: str
    test_fixture: bool = False

    def __post_init__(self) -> None:
        selected = _condition(self.condition)
        if type(self.protocol_schema) is not int or self.protocol_schema != 31:
            raise EvaluationError("evaluation identity requires protocol schema 31")
        for name in (
            "protocol_identity",
            "stu_checkpoint_sha256",
            "stu_model_state_sha256",
            "stu_source_manifest_sha256",
            "calibration_sha256",
            "ray_mapping_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.fusion_value != _fusion_value(selected):
            raise EvaluationError("evaluation identity has the wrong score domain")
        if selected is ExperimentCondition.B0:
            if self.model_class is not None or self.model_state_sha256 is not None:
                raise EvaluationError("B0 identity cannot contain an AJAE model")
        elif (
            not isinstance(self.model_class, str)
            or not self.model_class
            or self.model_state_sha256 is None
        ):
            raise EvaluationError("trainable conditions require an AJAE model identity")
        if self.model_state_sha256 is not None:
            _sha256(self.model_state_sha256, "model_state_sha256")
        if not isinstance(self.stu_class, str) or not self.stu_class:
            raise EvaluationError("evaluation identity requires an STU class")
        if type(self.test_fixture) is not bool:
            raise TypeError("test_fixture must be boolean")
        object.__setattr__(self, "condition", selected.value)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_schema": self.protocol_schema,
            "protocol_identity": self.protocol_identity,
            "condition": self.condition,
            "fusion_value": self.fusion_value,
            "model_class": self.model_class,
            "model_state_sha256": self.model_state_sha256,
            "stu_class": self.stu_class,
            "stu_checkpoint_sha256": self.stu_checkpoint_sha256,
            "stu_model_state_sha256": self.stu_model_state_sha256,
            "stu_source_manifest_sha256": self.stu_source_manifest_sha256,
            "calibration_sha256": self.calibration_sha256,
            "ray_mapping_sha256": self.ray_mapping_sha256,
            "test_fixture": self.test_fixture,
        }


def _condition(value: ExperimentCondition | str) -> ExperimentCondition:
    try:
        return ExperimentCondition(value)
    except ValueError as error:
        raise EvaluationError("condition must be one of B0, B1, B2, or B3") from error


def _finite_vector(
    name: str, value: np.ndarray | Sequence[float], count: int | None = None
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or (count is not None and array.shape != (count,))
        or not np.issubdtype(array.dtype, np.floating)
        or not np.isfinite(array).all()
    ):
        expected = "[N]" if count is None else f"[{count}]"
        raise EvaluationError(f"{name} must be a finite floating vector {expected}")
    return array


def _integer_vector(
    name: str,
    value: int | np.ndarray | Sequence[int],
    count: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"{name} must use an integer dtype")
        array = np.full(count, int(array), dtype=np.int64)
    if (
        array.shape != (count,)
        or not np.issubdtype(array.dtype, np.integer)
        or np.any(array < minimum)
        or (maximum is not None and np.any(array > maximum))
    ):
        raise EvaluationError(f"{name} must be a non-negative integer vector [{count}]")
    return array.astype(np.int64, copy=False)


def _sigmoid(logits: np.ndarray | Sequence[float]) -> np.ndarray:
    values = _finite_vector("logits", logits).astype(np.float64, copy=False)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _evaluation_values(protocol: _Protocol | None) -> tuple[float, float, int]:
    if protocol is None:
        return 2.5, 50.0, 5
    evaluation = protocol.evaluation
    return (
        float(evaluation.minimum_range_m),
        float(evaluation.maximum_range_m),
        int(evaluation.minimum_anomaly_points),
    )


def _range_mask(
    protocol: _Protocol | None, source: _SourceFrame | np.ndarray
) -> np.ndarray:
    """Select points in the official inclusive range interval."""

    xyzi = np.asarray(source if isinstance(source, np.ndarray) else source.xyzi)
    if xyzi.ndim != 2 or xyzi.shape[1] < 3 or not np.isfinite(xyzi[:, :3]).all():
        raise EvaluationError("source coordinates must be finite [N,>=3]")
    minimum, maximum, _ = _evaluation_values(protocol)
    # The released evaluator takes the norm directly on float32 scan data.
    distance = np.linalg.norm(xyzi[:, :3].astype(np.float32, copy=False), axis=1)
    return (distance >= minimum) & (distance <= maximum)


def binary_targets(raw_semantic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map STU raw semantics to anomaly truth and label validity."""

    semantic = np.asarray(raw_semantic)
    if semantic.ndim != 1 or not np.issubdtype(semantic.dtype, np.integer):
        raise TypeError("raw_semantic must be an integer vector")
    return semantic == 2, semantic != 0


def official_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute released-STU-equivalent pooled AP, AUROC, and strict FPR95."""

    truth = np.asarray(labels)
    values = np.asarray(scores)
    if truth.dtype != np.bool_ or truth.ndim != 1 or values.shape != truth.shape:
        raise TypeError("point labels and scores must be aligned vectors")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise EvaluationError("point scores must be finite floating values")
    if truth.size == 0 or not bool(truth.any()) or bool(truth.all()):
        raise EvaluationError("point metrics require normal and anomaly points")
    false_positive, true_positive, thresholds = roc_curve(truth, values)
    candidates = np.flatnonzero(true_positive > 0.95)
    index = int(candidates[0]) if candidates.size else 0
    return {
        "AP": float(100.0 * average_precision_score(truth, values)),
        "AUROC": float(100.0 * auc(false_positive, true_positive)),
        "FPR95": float(100.0 * false_positive[index]),
        "threshold": float(thresholds[index]) if candidates.size else 0.0,
    }


# This is the same sole implementation retained for evaluator-qualification imports.
_point_metrics = official_metrics


def normal_alarm_threshold(scores: np.ndarray, alarm_rate: float) -> float:
    """Return the one-based normal-score order statistic used for calibration."""

    values = _finite_vector("normal calibration scores", scores)
    if values.size == 0:
        raise EvaluationError("normal calibration scores cannot be empty")
    if not 0.0 < alarm_rate < 1.0:
        raise EvaluationError("normal point alarm rate must lie strictly within (0,1)")
    rank = int(math.ceil((1.0 - alarm_rate) * values.size))
    return float(np.partition(values, rank - 1)[rank - 1])


def normal_score_statistics(
    frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    protocol: _Protocol | None = None,
    *,
    alarm_rate: float = 0.001,
) -> dict[str, float | int]:
    """Report pure-normal scores without applying the anomaly-frame gate."""

    collected: list[np.ndarray] = []
    frame_count = 0
    for points, scores, raw_semantic in frames:
        values = _finite_vector("normal scores", scores)
        anomaly, valid = binary_targets(raw_semantic)
        if values.shape != anomaly.shape:
            raise EvaluationError("pure-normal scores and labels are not aligned")
        valid &= _range_mask(protocol, points)
        if bool(np.any(anomaly[valid])):
            raise EvaluationError("pure-normal evaluation received an anomaly label")
        if np.any(valid):
            collected.append(values[valid].astype(np.float64, copy=True))
        frame_count += 1
    if not collected:
        raise EvaluationError("pure-normal evaluation contains no valid point")
    all_scores = np.concatenate(collected)
    threshold = normal_alarm_threshold(all_scores, alarm_rate)
    return {
        "frames": frame_count,
        "points": int(all_scores.size),
        "mean": float(np.mean(all_scores)),
        "standard_deviation": float(np.std(all_scores)),
        "median": float(np.quantile(all_scores, 0.5)),
        "p95": float(np.quantile(all_scores, 0.95)),
        "p99": float(np.quantile(all_scores, 0.99)),
        "q99.9": float(np.quantile(all_scores, 0.999)),
        "maximum": float(np.max(all_scores)),
        "target_alarm_rate": float(alarm_rate),
        "strict_threshold": threshold,
        "observed_strict_alarm_rate": float(np.mean(all_scores > threshold)),
    }


class PointMetricAccumulator:
    """Accumulate frames exactly as the official STU point evaluator does."""

    def __init__(self, protocol: _Protocol | None = None) -> None:
        self.protocol = protocol
        self._scores: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self.accepted_frames = 0
        self.skipped_frames = 0

    def update(
        self, points: np.ndarray, scores: np.ndarray, raw_semantic: np.ndarray
    ) -> bool:
        values = _finite_vector("scores", scores)
        anomaly, valid = binary_targets(raw_semantic)
        if values.shape != anomaly.shape:
            raise EvaluationError("prediction and label count mismatch")
        valid &= _range_mask(self.protocol, points)
        _, _, minimum_anomaly = _evaluation_values(self.protocol)
        if int(anomaly[valid].sum()) < minimum_anomaly:
            self.skipped_frames += 1
            return False
        self._scores.append(values[valid].astype(np.float64, copy=True))
        self._labels.append(anomaly[valid].astype(np.bool_, copy=True))
        self.accepted_frames += 1
        return True

    def compute(self) -> dict[str, float | int]:
        if not self._scores:
            return {
                "accepted_frames": self.accepted_frames,
                "skipped_frames": self.skipped_frames,
            }
        result: dict[str, float | int] = official_metrics(
            np.concatenate(self._labels), np.concatenate(self._scores)
        )
        result.update(
            accepted_frames=self.accepted_frames,
            skipped_frames=self.skipped_frames,
        )
        return result


class DevelopmentMetricAccumulator:
    """Pool all valid synthetic points without the public-frame anomaly gate."""

    def __init__(self, protocol: _Protocol | None = None) -> None:
        self.protocol = protocol
        self._scores: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self.frames = 0

    def update(
        self, points: np.ndarray, scores: np.ndarray, raw_semantic: np.ndarray
    ) -> None:
        values = _finite_vector("development scores", scores)
        anomaly, valid = binary_targets(raw_semantic)
        if values.shape != anomaly.shape:
            raise EvaluationError("development scores and labels are not aligned")
        valid &= _range_mask(self.protocol, points)
        if np.any(valid):
            self._scores.append(values[valid].astype(np.float64, copy=True))
            self._labels.append(anomaly[valid].astype(np.bool_, copy=True))
        self.frames += 1

    def compute(self) -> dict[str, float | int]:
        if not self._scores:
            raise EvaluationError("development evaluation contains no valid point")
        labels = np.concatenate(self._labels)
        scores = np.concatenate(self._scores)
        result: dict[str, float | int] = official_metrics(labels, scores)
        result.update(
            frames=self.frames,
            points=int(labels.size),
            anomaly_points=int(labels.sum()),
            normal_points=int((~labels).sum()),
        )
        return result


class OccurrenceMetricAccumulator:
    """Apply the official frame gate, then report point metrics by count 1--5."""

    def __init__(self, protocol: _Protocol | None = None) -> None:
        self.protocol = protocol
        self._scores = {count: [] for count in range(1, 6)}
        self._labels = {count: [] for count in range(1, 6)}
        self.accepted_frames = 0
        self.skipped_frames = 0

    def update(
        self,
        points: np.ndarray,
        scores: np.ndarray,
        raw_semantic: np.ndarray,
        occurrence_count: np.ndarray,
    ) -> bool:
        values = _finite_vector("scores", scores)
        anomaly, valid = binary_targets(raw_semantic)
        counts = np.asarray(occurrence_count)
        if (
            values.shape != anomaly.shape
            or counts.shape != anomaly.shape
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any((counts < 0) | (counts > 5))
        ):
            raise EvaluationError("scores, labels, and occurrence counts must align")
        valid &= _range_mask(self.protocol, points)
        _, _, minimum_anomaly = _evaluation_values(self.protocol)
        if int(anomaly[valid].sum()) < minimum_anomaly:
            self.skipped_frames += 1
            return False
        if np.any(valid & (counts == 0)):
            raise EvaluationError(
                "an evaluation-domain point lacks a window occurrence"
            )
        for count in range(1, 6):
            selected = valid & (counts == count)
            if np.any(selected):
                self._scores[count].append(values[selected].astype(np.float64))
                self._labels[count].append(anomaly[selected].astype(np.bool_))
        self.accepted_frames += 1
        return True

    def compute(self) -> dict[str, dict[str, float | int | None]]:
        result: dict[str, dict[str, float | int | None]] = {}
        for count in range(1, 6):
            if not self._scores[count]:
                result[str(count)] = {
                    "points": 0,
                    "normal_points": 0,
                    "anomaly_points": 0,
                    "AP": None,
                    "AUROC": None,
                    "FPR95": None,
                    "threshold": None,
                }
                continue
            scores = np.concatenate(self._scores[count])
            labels = np.concatenate(self._labels[count])
            record: dict[str, float | int | None] = {
                "points": int(labels.size),
                "normal_points": int((~labels).sum()),
                "anomaly_points": int(labels.sum()),
                "AP": None,
                "AUROC": None,
                "FPR95": None,
                "threshold": None,
            }
            if labels.any() and (~labels).any():
                record.update(official_metrics(labels, scores))
            result[str(count)] = record
        return result


class MovingNormalDiagnostic:
    """Report moving-normal safety without exposing semantics to the model."""

    def __init__(
        self,
        strict_threshold: float,
        protocol: _Protocol | None = None,
        moving_semantics: Iterable[int] = MOVING_NORMAL_SEMANTICS,
    ) -> None:
        if not math.isfinite(strict_threshold):
            raise EvaluationError("moving-normal threshold must be finite")
        self.strict_threshold = float(strict_threshold)
        self.protocol = protocol
        self.moving_semantics = np.asarray(tuple(moving_semantics), dtype=np.int64)
        if self.moving_semantics.size == 0:
            raise EvaluationError("moving semantic set cannot be empty")
        self._moving: list[np.ndarray] = []
        self._static: list[np.ndarray] = []

    def update(
        self, points: np.ndarray, scores: np.ndarray, raw_semantic: np.ndarray
    ) -> None:
        semantic = np.asarray(raw_semantic)
        values = _finite_vector("moving-normal scores", scores, semantic.size)
        if semantic.ndim != 1 or not np.issubdtype(semantic.dtype, np.integer):
            raise TypeError("moving-normal semantics must be an integer vector")
        valid = (semantic != 0) & (semantic != 2) & _range_mask(self.protocol, points)
        moving = valid & np.isin(semantic, self.moving_semantics)
        static = valid & ~np.isin(semantic, self.moving_semantics)
        if np.any(moving):
            self._moving.append(values[moving].astype(np.float64))
        if np.any(static):
            self._static.append(values[static].astype(np.float64))

    def compute(self) -> dict[str, float | int | None]:
        moving = np.concatenate(self._moving) if self._moving else np.empty(0)
        static = np.concatenate(self._static) if self._static else np.empty(0)
        moving_mean = float(moving.mean()) if moving.size else None
        static_mean = float(static.mean()) if static.size else None
        return {
            "strict_threshold": self.strict_threshold,
            "moving_points": int(moving.size),
            "moving_mean": moving_mean,
            "moving_false_positive_rate": (
                float(np.mean(moving > self.strict_threshold)) if moving.size else None
            ),
            "static_points": int(static.size),
            "static_mean": static_mean,
            "static_false_positive_rate": (
                float(np.mean(static > self.strict_threshold)) if static.size else None
            ),
            "moving_minus_static_mean": (
                moving_mean - static_mean
                if moving_mean is not None and static_mean is not None
                else None
            ),
        }


def _frozen(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FusedPointSet:
    """One namespace's deterministic frame-ray score table."""

    namespace: str
    source_frame: np.ndarray
    source_ray: np.ndarray
    source_slot: np.ndarray
    score: np.ndarray
    occurrence_count: np.ndarray
    fusion_value: str = AJAE_FUSION_VALUE
    partition: str | None = None
    sequence_id: int | None = None
    world_identity: str | None = None

    def __post_init__(self) -> None:
        count = int(np.asarray(self.source_frame).size)
        frames = _integer_vector("source_frame", self.source_frame, count)
        rays = _integer_vector("source_ray", self.source_ray, count, maximum=131071)
        slots = _integer_vector("source_slot", self.source_slot, count)
        scores = _finite_vector("fused score", self.score, count)
        occurrences = np.asarray(self.occurrence_count)
        if count == 0:
            raise EvaluationError("a fused point set cannot be empty")
        if self.fusion_value not in {B0_FUSION_VALUE, AJAE_FUSION_VALUE}:
            raise EvaluationError("fused score domain is unsupported")
        if self.fusion_value == AJAE_FUSION_VALUE and np.any(
            (scores < 0.0) | (scores > 1.0)
        ):
            raise EvaluationError("fused probabilities must lie in [0,1]")
        if (
            occurrences.shape != (count,)
            or not np.issubdtype(occurrences.dtype, np.integer)
            or np.any((occurrences < 1) | (occurrences > 5))
        ):
            raise EvaluationError("fused occurrence counts must lie in 1..5")
        pairs = np.rec.fromarrays((frames, rays), names=("frame", "ray"))
        if np.unique(pairs).size != count:
            raise EvaluationError("fused frame-ray identities must be unique")
        frame_slots = np.rec.fromarrays((frames, slots), names=("frame", "slot"))
        if np.unique(frame_slots).size != count:
            raise EvaluationError("fused source slots must be unique within each frame")
        if self.namespace == "real":
            if (
                self.partition not in {"train", "val", "test"}
                or type(self.sequence_id) is not int
                or self.sequence_id < 0
                or self.world_identity is not None
            ):
                raise EvaluationError(
                    "real fusion requires one partition/sequence scope"
                )
        elif self.namespace == "synthetic":
            if (
                not isinstance(self.world_identity, str)
                or not self.world_identity
                or self.partition is not None
                or self.sequence_id is not None
            ):
                raise EvaluationError("synthetic fusion requires one world identity")
        else:
            raise EvaluationError("fusion namespace must be real or synthetic")
        object.__setattr__(self, "source_frame", _frozen(frames.copy()))
        object.__setattr__(self, "source_ray", _frozen(rays.copy()))
        object.__setattr__(self, "source_slot", _frozen(slots.copy()))
        object.__setattr__(self, "score", _frozen(scores.astype(np.float64, copy=True)))
        object.__setattr__(
            self, "occurrence_count", _frozen(occurrences.astype(np.uint8, copy=True))
        )

    @property
    def unique_point_count(self) -> int:
        return int(self.source_frame.size)

    @property
    def total_occurrence_count(self) -> int:
        return int(self.occurrence_count.astype(np.int64).sum())

    @property
    def occurrence_histogram(self) -> dict[str, int]:
        return {
            str(count): int(np.sum(self.occurrence_count == count))
            for count in range(1, 6)
        }

    def frame_mask(self, frame_id: int) -> np.ndarray:
        return self.source_frame == int(frame_id)

    @property
    def probability(self) -> np.ndarray:
        """Expose AJAE probabilities while preventing B0 score-domain confusion."""

        if self.fusion_value != AJAE_FUSION_VALUE:
            raise EvaluationError("raw B0 MaxLogit scores are not probabilities")
        return self.score


@dataclass(slots=True)
class _FusionState:
    source_slot: int
    values: list[float]


class WindowScoreFusion:
    """Apply the declared score transform and fuse every legal occurrence."""

    def __init__(
        self, maximum_count: int = 5, *, fusion_value: str = AJAE_FUSION_VALUE
    ) -> None:
        if maximum_count != 5:
            raise EvaluationError(
                "schema 31 fixes the maximum occurrence count at five"
            )
        if fusion_value not in {B0_FUSION_VALUE, AJAE_FUSION_VALUE}:
            raise EvaluationError("fusion_value is not a schema-31 score domain")
        self.maximum_count = 5
        self.fusion_value = fusion_value
        self._states: dict[tuple[object, ...], _FusionState] = {}
        self._seen_occurrences: set[tuple[tuple[object, ...], int]] = set()

    @staticmethod
    def _real_prefix(partition: str, sequence_id: int) -> tuple[object, ...]:
        if partition not in {"train", "val", "test"}:
            raise EvaluationError("real fusion partition must be train, val, or test")
        if type(sequence_id) is not int or sequence_id < 0:
            raise EvaluationError("real fusion sequence_id must be non-negative")
        return ("real", partition, sequence_id)

    @staticmethod
    def _synthetic_prefix(world_identity: str) -> tuple[object, ...]:
        if not isinstance(world_identity, str) or not world_identity:
            raise EvaluationError(
                "synthetic fusion requires a non-empty world identity"
            )
        return ("synthetic", world_identity)

    def _add_values(
        self,
        prefix: tuple[object, ...],
        *,
        window_start: int,
        source_frame: int | np.ndarray | Sequence[int],
        source_ray: np.ndarray | Sequence[int],
        source_slot: np.ndarray | Sequence[int],
        values: np.ndarray | Sequence[float],
    ) -> None:
        scores = _finite_vector("occurrence scores", values)
        count = int(scores.size)
        if count == 0:
            raise EvaluationError("a window occurrence cannot be empty")
        if self.fusion_value == AJAE_FUSION_VALUE and np.any(
            (scores < 0.0) | (scores > 1.0)
        ):
            raise EvaluationError("fusion requires post-sigmoid probabilities in [0,1]")
        if type(window_start) is not int or window_start < 0:
            raise EvaluationError("window_start must be a non-negative integer")
        frames = _integer_vector("source_frame", source_frame, count)
        rays = _integer_vector("source_ray", source_ray, count, maximum=131071)
        slots = _integer_vector("source_slot", source_slot, count)
        pairs = np.rec.fromarrays((frames, rays), names=("frame", "ray"))
        if np.unique(pairs).size != count:
            raise EvaluationError("one window cannot repeat a frame-ray identity")
        frame_slots = np.rec.fromarrays((frames, slots), names=("frame", "slot"))
        if np.unique(frame_slots).size != count:
            raise EvaluationError("one window cannot map two rays to one source slot")
        for frame, ray, slot, score in zip(
            frames, rays, slots, scores.astype(np.float64), strict=True
        ):
            key = (*prefix, int(frame), int(ray))
            occurrence = (key, window_start)
            if occurrence in self._seen_occurrences:
                raise EvaluationError("one window occurrence was added more than once")
            state = self._states.get(key)
            if state is None:
                state = _FusionState(int(slot), [])
                self._states[key] = state
            elif state.source_slot != int(slot):
                raise EvaluationError("one frame-ray identity changed source slot")
            if len(state.values) >= self.maximum_count:
                raise EvaluationError("a frame-ray received more than five predictions")
            if (
                self.fusion_value == B0_FUSION_VALUE
                and state.values
                and float(score) != state.values[0]
            ):
                raise EvaluationError(
                    "frozen STU MaxLogit changed across occurrences of one point"
                )
            state.values.append(float(score))
            self._seen_occurrences.add(occurrence)

    def add_real(
        self,
        *,
        partition: str,
        sequence_id: int,
        window_start: int,
        source_frame: int | np.ndarray | Sequence[int],
        source_ray: np.ndarray | Sequence[int],
        source_slot: np.ndarray | Sequence[int],
        logits: np.ndarray | Sequence[float],
    ) -> None:
        if self.fusion_value != AJAE_FUSION_VALUE:
            raise EvaluationError("AJAE logits require the sigmoid fusion domain")
        self._add_values(
            self._real_prefix(partition, sequence_id),
            window_start=window_start,
            source_frame=source_frame,
            source_ray=source_ray,
            source_slot=source_slot,
            values=_sigmoid(logits),
        )

    def add_real_scores(
        self,
        *,
        partition: str,
        sequence_id: int,
        window_start: int,
        source_frame: int | np.ndarray | Sequence[int],
        source_ray: np.ndarray | Sequence[int],
        source_slot: np.ndarray | Sequence[int],
        scores: np.ndarray | Sequence[float],
    ) -> None:
        """Add unchanged official B0 scores in their native finite domain."""

        if self.fusion_value != B0_FUSION_VALUE:
            raise EvaluationError("raw scores require the B0 MaxLogit fusion domain")
        self._add_values(
            self._real_prefix(partition, sequence_id),
            window_start=window_start,
            source_frame=source_frame,
            source_ray=source_ray,
            source_slot=source_slot,
            values=scores,
        )

    def add_synthetic(
        self,
        *,
        world_identity: str,
        window_start: int,
        source_frame: int | np.ndarray | Sequence[int],
        source_ray: np.ndarray | Sequence[int],
        source_slot: np.ndarray | Sequence[int],
        logits: np.ndarray | Sequence[float],
    ) -> None:
        if self.fusion_value != AJAE_FUSION_VALUE:
            raise EvaluationError("AJAE logits require the sigmoid fusion domain")
        self._add_values(
            self._synthetic_prefix(world_identity),
            window_start=window_start,
            source_frame=source_frame,
            source_ray=source_ray,
            source_slot=source_slot,
            values=_sigmoid(logits),
        )

    def add_synthetic_scores(
        self,
        *,
        world_identity: str,
        window_start: int,
        source_frame: int | np.ndarray | Sequence[int],
        source_ray: np.ndarray | Sequence[int],
        source_slot: np.ndarray | Sequence[int],
        scores: np.ndarray | Sequence[float],
    ) -> None:
        if self.fusion_value != B0_FUSION_VALUE:
            raise EvaluationError("raw scores require the B0 MaxLogit fusion domain")
        self._add_values(
            self._synthetic_prefix(world_identity),
            window_start=window_start,
            source_frame=source_frame,
            source_ray=source_ray,
            source_slot=source_slot,
            values=scores,
        )

    def _finalize_prefix(self, prefix: tuple[object, ...]) -> FusedPointSet:
        selected = [
            (key, state)
            for key, state in self._states.items()
            if key[: len(prefix)] == prefix
        ]
        if not selected:
            raise EvaluationError("the requested fusion namespace has no predictions")
        selected.sort(key=lambda item: (int(item[0][-2]), int(item[0][-1])))
        frames = np.asarray([int(key[-2]) for key, _ in selected], dtype=np.int64)
        rays = np.asarray([int(key[-1]) for key, _ in selected], dtype=np.int64)
        slots = np.asarray([state.source_slot for _, state in selected], dtype=np.int64)
        occurrences = np.asarray(
            [len(state.values) for _, state in selected], dtype=np.uint8
        )
        scores = np.asarray(
            [math.fsum(state.values) / len(state.values) for _, state in selected],
            dtype=np.float64,
        )
        if self.fusion_value == B0_FUSION_VALUE:
            # Identical official scores make the mathematical mean an exact passthrough.
            scores = np.asarray(
                [state.values[0] for _, state in selected], dtype=np.float64
            )
        if prefix[0] == "real":
            return FusedPointSet(
                "real",
                frames,
                rays,
                slots,
                scores,
                occurrences,
                self.fusion_value,
                partition=str(prefix[1]),
                sequence_id=int(prefix[2]),
            )
        return FusedPointSet(
            "synthetic",
            frames,
            rays,
            slots,
            scores,
            occurrences,
            self.fusion_value,
            world_identity=str(prefix[1]),
        )

    def finalize_real(self, partition: str, sequence_id: int) -> FusedPointSet:
        return self._finalize_prefix(self._real_prefix(partition, sequence_id))

    def finalize_synthetic(self, world_identity: str) -> FusedPointSet:
        return self._finalize_prefix(self._synthetic_prefix(world_identity))

    def finalize(
        self,
        *,
        partition: str | None = None,
        sequence_id: int | None = None,
        world_identity: str | None = None,
    ) -> FusedPointSet:
        """Finalize exactly one real sequence or one synthetic clip world."""

        if world_identity is not None and partition is None and sequence_id is None:
            return self.finalize_synthetic(world_identity)
        if world_identity is None and partition is not None and sequence_id is not None:
            return self.finalize_real(partition, sequence_id)
        raise EvaluationError("finalize requires exactly one complete fusion scope")


class RollingCache:
    """A deterministic bounded cache for frozen per-frame STU evidence."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._values: OrderedDict[object, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: object, factory: Callable[[object], Any]) -> Any:
        if key in self._values:
            self.hits += 1
            value = self._values.pop(key)
            self._values[key] = value
            return value
        self.misses += 1
        value = factory(key)
        self._values[key] = value
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)
        return value

    def clear(self) -> None:
        self._values.clear()
        self.hits = 0
        self.misses = 0


@dataclass(frozen=True, slots=True)
class FramePrediction:
    """Fused scores restored to one original scan's complete slot order."""

    score: np.ndarray
    occurrence_count: np.ndarray
    fusion_value: str

    def __post_init__(self) -> None:
        scores = _finite_vector("frame score", self.score)
        occurrences = np.asarray(self.occurrence_count)
        if (
            occurrences.shape != scores.shape
            or not np.issubdtype(occurrences.dtype, np.integer)
            or np.any((occurrences < 0) | (occurrences > 5))
        ):
            raise EvaluationError("frame scores and occurrence counts are invalid")
        if self.fusion_value not in {B0_FUSION_VALUE, AJAE_FUSION_VALUE}:
            raise EvaluationError("frame score domain is unsupported")
        if self.fusion_value == AJAE_FUSION_VALUE and np.any(
            (scores < 0.0) | (scores > 1.0)
        ):
            raise EvaluationError("AJAE frame probabilities must lie in [0,1]")
        if np.any((occurrences == 0) & (scores != 0.0)):
            raise EvaluationError("uncovered file slots must retain score zero")
        object.__setattr__(self, "score", _frozen(scores.astype(np.float64, copy=True)))
        object.__setattr__(
            self, "occurrence_count", _frozen(occurrences.astype(np.uint8, copy=True))
        )

    @property
    def probability(self) -> np.ndarray:
        if self.fusion_value != AJAE_FUSION_VALUE:
            raise EvaluationError("raw B0 MaxLogit scores are not probabilities")
        return self.score


@dataclass(frozen=True, slots=True)
class SequencePrediction:
    """One condition's full covered sequence prediction and inference cost."""

    condition: str
    partition: str
    sequence_id: int
    window_starts: tuple[int, ...]
    frames: Mapping[int, FramePrediction]
    fused: FusedPointSet
    identity: EvaluationIdentity
    cost: Mapping[str, Any]

    def __post_init__(self) -> None:
        selected = _condition(self.condition)
        if (
            self.partition not in {"train", "val", "test"}
            or type(self.sequence_id) is not int
            or self.sequence_id < 0
            or self.fused.namespace != "real"
            or self.fused.partition != self.partition
            or self.fused.sequence_id != self.sequence_id
            or not isinstance(self.identity, EvaluationIdentity)
            or self.identity.condition != selected.value
            or self.identity.fusion_value != self.fused.fusion_value
        ):
            raise EvaluationError("sequence prediction has an inconsistent real scope")
        starts = tuple(self.window_starts)
        if (
            not starts
            or any(type(start) is not int or start < 0 for start in starts)
            or starts != tuple(sorted(set(starts)))
        ):
            raise EvaluationError("window starts must be non-empty, sorted, and unique")
        expected_frames = {
            frame_id for start in starts for frame_id in range(start, start + 5)
        }
        if set(self.frames) != expected_frames or any(
            type(frame_id) is not int
            or not isinstance(prediction, FramePrediction)
            or prediction.fusion_value != self.fused.fusion_value
            for frame_id, prediction in self.frames.items()
        ):
            raise EvaluationError("prediction frames must equal the legal-window union")
        if not isinstance(self.cost, Mapping):
            raise TypeError("prediction cost must be a mapping")
        object.__setattr__(self, "condition", selected.value)


def _protocol_slot_to_ray(
    protocol: object,
) -> tuple[Callable[[object], np.ndarray], str, str]:
    """Load the frozen OS1-128 file-slot to canonical-ray identity mapping."""

    try:
        from .render import load_sensor_calibration
        from .scene import canonical_ray_mapping_digest
    except ImportError:  # pragma: no cover - direct module execution
        from render import load_sensor_calibration
        from scene import canonical_ray_mapping_digest

    render = getattr(protocol, "render")
    grid_spec = render["ray_grid"]
    beams = int(grid_spec["beam_count"])
    columns = int(grid_spec["column_count"])
    if beams != 128 or columns != 1024:
        raise EvaluationError("evaluation requires the frozen OS1-128 ray grid")
    calibration_path = getattr(protocol, "sensor_calibration_path")()
    try:
        ray_grid, sensor = load_sensor_calibration(calibration_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise EvaluationError(
            "cannot load the authoritative ray calibration"
        ) from error
    if (
        ray_grid.beam_count != beams
        or ray_grid.columns != columns
        or sensor.source_sequence_id != 206
        or ray_grid.calibration_frame_ids != tuple(range(449))
    ):
        raise EvaluationError("runtime calibration does not match train/206 OS1-128")
    provenance = dict(sensor.provenance)
    if any(
        provenance.get(key) != value
        for key, value in {
            "partition": "train",
            "sequence": "206",
            "frames": "449",
            "first_frame": "0",
            "last_frame": "448",
        }.items()
    ):
        raise EvaluationError("sensor calibration lacks complete train/206 provenance")
    canonical_by_slot = np.asarray(ray_grid.canonical_ray_by_slot)
    if canonical_by_slot.dtype != np.int32 or canonical_by_slot.shape != (
        beams * columns,
    ):
        raise EvaluationError("calibration has an invalid canonical ray mapping")
    digest = canonical_ray_mapping_digest(canonical_by_slot)

    def mapping(source: object) -> np.ndarray:
        if int(getattr(source, "slot_count")) != canonical_by_slot.size:
            raise EvaluationError("source slots do not cover the canonical ray grid")
        return canonical_by_slot.copy()

    return mapping, digest, _sha256_file(calibration_path)


# Public spelling; the private spelling remains for qualification imports.
protocol_slot_to_ray = _protocol_slot_to_ray


class AJAEInference:
    """Run every legal five-scan window and fuse all visible-return logits."""

    def __init__(
        self,
        model: torch.nn.Module | None,
        encoder: torch.nn.Module,
        *,
        protocol: object,
        condition: ExperimentCondition | str,
        device: torch.device | str = "cpu",
        cache_frames: int = 10,
        time_budget_seconds: float | None = None,
    ) -> None:
        selected = _condition(condition)
        try:
            from .model import FrozenSTUPointEncoder, JointWindowPointTransformer
            from .protocol import AJAEProtocol
        except ImportError:  # pragma: no cover - direct module execution
            from model import FrozenSTUPointEncoder, JointWindowPointTransformer
            from protocol import AJAEProtocol

        if type(protocol) is not AJAEProtocol:
            raise TypeError("formal inference requires an AJAEProtocol")
        if type(encoder) is not FrozenSTUPointEncoder:
            raise TypeError("formal inference requires FrozenSTUPointEncoder")
        if selected is ExperimentCondition.B0 and model is not None:
            raise EvaluationError("B0 uses only frozen STU MaxLogit")
        if selected is not ExperimentCondition.B0 and type(model) is not (
            JointWindowPointTransformer
        ):
            raise TypeError("formal inference requires JointWindowPointTransformer")
        slot_to_ray, ray_mapping_digest, calibration_sha256 = _protocol_slot_to_ray(
            protocol
        )
        identity = self._formal_identity(
            protocol,
            selected,
            model,
            encoder,
            calibration_sha256=calibration_sha256,
            ray_mapping_digest=ray_mapping_digest,
        )
        self._initialize(
            model,
            encoder,
            protocol=protocol,
            condition=selected,
            slot_to_ray=slot_to_ray,
            ray_mapping_digest=ray_mapping_digest,
            calibration_sha256=calibration_sha256,
            identity=identity,
            device=device,
            cache_frames=cache_frames,
            time_budget_seconds=time_budget_seconds,
        )

    @staticmethod
    def _formal_identity(
        protocol: object,
        condition: ExperimentCondition,
        model: torch.nn.Module | None,
        encoder: torch.nn.Module,
        *,
        calibration_sha256: str,
        ray_mapping_digest: str,
    ) -> EvaluationIdentity:
        try:
            from .model import (
                STU_CHECKPOINT_SHA256,
                STU_MODEL_STATE_TENSOR_SHA256,
                FrozenSTUPointEncoder,
                JointWindowPointTransformer,
                model_state_sha256,
                stu_source_manifest,
            )
            from .protocol import AJAEProtocol
        except ImportError:  # pragma: no cover - direct module execution
            from model import (
                STU_CHECKPOINT_SHA256,
                STU_MODEL_STATE_TENSOR_SHA256,
                FrozenSTUPointEncoder,
                JointWindowPointTransformer,
                model_state_sha256,
                stu_source_manifest,
            )
            from protocol import AJAEProtocol

        if type(protocol) is not AJAEProtocol or protocol.schema_version != 31:
            raise EvaluationError(
                "formal inference requires the active schema-31 protocol"
            )
        if type(encoder) is not FrozenSTUPointEncoder:
            raise TypeError("formal inference requires FrozenSTUPointEncoder")
        evaluation = getattr(protocol, "evaluation_document")
        fusion_values = evaluation.get("fusion_values")
        expected_fusion_values = {
            "B0": {
                "input": B0_FUSION_VALUE,
                "per_occurrence_transform": "none",
                "repeated_window_mean": "identity_preserving",
            },
            "B1_B2_B3": {
                "input": "anomaly_logit",
                "per_occurrence_transform": "sigmoid",
            },
        }
        if (
            not isinstance(fusion_values, Mapping)
            or _plain_json(fusion_values) != expected_fusion_values
        ):
            raise EvaluationError(
                "protocol does not declare both schema-31 score domains"
            )
        if evaluation.get("fusion_reduction") != FUSION_REDUCTION:
            raise EvaluationError(
                "protocol changed the all-occurrence fusion reduction"
            )

        stu = getattr(protocol, "stu")
        checkpoint_sha256 = _sha256(
            stu.get("checkpoint_sha256"), "protocol STU checkpoint identity"
        )
        if checkpoint_sha256 != STU_CHECKPOINT_SHA256:
            raise EvaluationError(
                "protocol does not identify the official STU checkpoint"
            )
        if Path(encoder.official_repository).resolve() != Path(
            protocol.stu_repository_path()
        ).resolve() or not math.isclose(
            float(encoder.voxel_size), float(stu["voxel_size_m"]), abs_tol=1.0e-12
        ):
            raise EvaluationError(
                "runtime STU source/configuration differs from protocol"
            )
        stu_state = model_state_sha256(encoder.stu.state_dict())
        if stu_state != STU_MODEL_STATE_TENSOR_SHA256:
            raise EvaluationError(
                "runtime STU weights differ from the frozen official model"
            )
        source_manifest = stu_source_manifest(encoder.official_repository)
        source_identity = _sha256(
            source_manifest.get("manifest_sha256"), "STU source manifest identity"
        )

        model_state: str | None = None
        model_class: str | None = None
        if condition is not ExperimentCondition.B0:
            if type(model) is not JointWindowPointTransformer:
                raise TypeError("formal inference requires JointWindowPointTransformer")
            specification = getattr(protocol, "model")
            radius = specification["radius_neighbors"]
            pools = tuple(float(item.voxel_size) for item in model.pyramid.pools)
            blocks = tuple(model.pyramid.blocks)
            if (
                pools != tuple(float(value) for value in specification["voxel_sizes_m"])
                or tuple(float(item.radius) for item in blocks)
                != tuple(float(value) for value in radius["radii_m"])
                or tuple(int(item.neighborhood.k) for item in blocks)
                != tuple(int(value) for value in radius["maximum_neighbors"])
                or any(
                    int(item.heads) != int(specification["heads"]) for item in blocks
                )
                or int(model.upsample.k) != int(specification["upsample_neighbors"])
                or len(model.decoder_fusions) != len(pools) - 1
            ):
                raise EvaluationError("runtime AJAE architecture differs from protocol")
            model_class = "JointWindowPointTransformer"
            model_state = model_state_sha256(model.state_dict())

        return EvaluationIdentity(
            31,
            _sha256(protocol.scientific_identity, "protocol scientific identity"),
            condition.value,
            _fusion_value(condition),
            model_class,
            model_state,
            "FrozenSTUPointEncoder",
            checkpoint_sha256,
            stu_state,
            source_identity,
            _sha256(calibration_sha256, "calibration identity"),
            _sha256(ray_mapping_digest, "ray mapping identity"),
        )

    @classmethod
    def _for_test(
        cls,
        model: torch.nn.Module | None,
        encoder: torch.nn.Module,
        *,
        condition: ExperimentCondition | str,
        slot_to_ray: Callable[[object], np.ndarray],
        ray_mapping_digest: str = "0" * 64,
        device: torch.device | str = "cpu",
        cache_frames: int = 10,
        time_budget_seconds: float | None = None,
    ) -> AJAEInference:
        """Build an unmistakably test-only inference seam for small fixtures."""

        selected = _condition(condition)
        if selected is ExperimentCondition.B0 and model is not None:
            raise EvaluationError("B0 test inference cannot contain an AJAE model")
        if selected is not ExperimentCondition.B0 and model is None:
            raise EvaluationError("AJAE test inference requires a model fixture")
        identity = EvaluationIdentity(
            31,
            "0" * 64,
            selected.value,
            _fusion_value(selected),
            None if model is None else type(model).__name__,
            None if model is None else "0" * 64,
            type(encoder).__name__,
            "0" * 64,
            "0" * 64,
            "0" * 64,
            "0" * 64,
            _sha256(ray_mapping_digest, "test ray mapping identity"),
            True,
        )
        instance = cls.__new__(cls)
        instance._initialize(
            model,
            encoder,
            protocol=None,
            condition=selected,
            slot_to_ray=slot_to_ray,
            ray_mapping_digest=ray_mapping_digest,
            calibration_sha256="0" * 64,
            identity=identity,
            device=device,
            cache_frames=cache_frames,
            time_budget_seconds=time_budget_seconds,
        )
        return instance

    def _initialize(
        self,
        model: torch.nn.Module | None,
        encoder: torch.nn.Module,
        *,
        protocol: object | None,
        condition: ExperimentCondition,
        slot_to_ray: Callable[[object], np.ndarray],
        ray_mapping_digest: str,
        calibration_sha256: str,
        identity: EvaluationIdentity,
        device: torch.device | str,
        cache_frames: int,
        time_budget_seconds: float | None,
    ) -> None:
        if not callable(slot_to_ray):
            raise TypeError("slot_to_ray must be callable")
        _sha256(ray_mapping_digest, "ray_mapping_digest")
        _sha256(calibration_sha256, "calibration_sha256")
        if time_budget_seconds is not None and time_budget_seconds <= 0.0:
            raise EvaluationError("time budget must be positive")
        self.protocol = protocol
        self.condition = condition
        self.model = model
        self.encoder = encoder
        self.slot_to_ray = slot_to_ray
        self.ray_mapping_digest = ray_mapping_digest
        self.calibration_sha256 = calibration_sha256
        self.identity = identity
        self.device = torch.device(device)
        self.cache = RollingCache(cache_frames)
        self._cache_namespace: tuple[object, ...] = ("unscoped",)
        self.time_budget_seconds = time_budget_seconds
        if self.model is not None:
            self.model.to(self.device).eval()
        self.encoder.to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def _assert_components_unchanged(self) -> None:
        if self.identity.test_fixture:
            return
        modules = [self.encoder]
        if self.model is not None:
            modules.append(self.model)
        if any(layer.training for module in modules for layer in module.modules()):
            raise EvaluationError("formal inference modules left evaluation mode")
        current_mapping, mapping_digest, calibration_sha256 = _protocol_slot_to_ray(
            self.protocol
        )
        current = self._formal_identity(
            self.protocol,
            self.condition,
            self.model,
            self.encoder,
            calibration_sha256=calibration_sha256,
            ray_mapping_digest=mapping_digest,
        )
        if current != self.identity:
            raise EvaluationError("inference components changed after identity freeze")
        self.slot_to_ray = current_mapping

    def _encode(self, source: object) -> object:
        try:
            from .model import stu_input_identity
        except ImportError:  # pragma: no cover - direct module execution
            from model import stu_input_identity

        frame_id = int(getattr(source, "frame_id"))

        def factory(_: object) -> object:
            with torch.no_grad():
                return self.encoder(
                    getattr(source, "coordinates"),
                    getattr(source, "features"),
                    getattr(source, "real_slots"),
                )

        encoding = self.cache.get((*self._cache_namespace, frame_id), factory)
        encoded_slots = getattr(encoding, "real_slots")
        if isinstance(encoded_slots, torch.Tensor):
            encoded_slots = encoded_slots.detach().cpu().numpy()
        if not np.array_equal(
            np.asarray(encoded_slots, dtype=np.int64),
            np.asarray(getattr(source, "real_slots"), dtype=np.int64),
        ):
            raise EvaluationError("STU encoding changed visible-return order")
        expected_input = stu_input_identity(
            getattr(source, "coordinates"),
            getattr(source, "features"),
            getattr(source, "real_slots"),
        )
        if getattr(encoding, "input_identity", None) != expected_input:
            raise EvaluationError(
                "STU encoding belongs to different source-frame content"
            )
        return encoding

    def _source_ray_map(self, source: object) -> np.ndarray:
        mapping = np.asarray(self.slot_to_ray(source))
        count = int(getattr(source, "slot_count"))
        if (
            mapping.dtype != np.int32
            or mapping.shape != (count,)
            or np.any(mapping < 0)
            or np.unique(mapping).size != count
        ):
            raise EvaluationError(
                "slot_to_ray must return a one-to-one int32[slot] map"
            )
        return mapping

    def _audited_window(self, sequence: object, window_start: int) -> object:
        spec = getattr(sequence, "spec")
        frame_ids = tuple(spec.window_frame_ids(window_start))
        mappings = {
            frame_id: self._source_ray_map(sequence.source_frame(frame_id))
            for frame_id in frame_ids
        }
        window = sequence.window(
            window_start,
            condition=self.condition,
            canonical_ray_by_slot=mappings,
            ray_mapping_audited=True,
            ray_mapping_digest=self.ray_mapping_digest,
        )
        if getattr(getattr(window, "points"), "ray_mapping_audited", None) is not True:
            raise EvaluationError("scene window lacks an audited canonical ray mapping")
        return window

    def _window_content(self, window: object) -> tuple[list[object], list[object]]:
        frames = list(getattr(window, "frames"))
        sources = [getattr(frame, "source") for frame in frames]
        encodings = [self._encode(source) for source in sources]
        expected = int(getattr(getattr(window, "points"), "count"))
        if sum(int(getattr(source, "real_count")) for source in sources) != expected:
            raise EvaluationError("window/STU source point counts do not align")
        return sources, encodings

    def _baseline_scores(self, window: object) -> np.ndarray:
        _, encodings = self._window_content(window)
        values = torch.cat(
            [getattr(encoding, "maxlogit_score") for encoding in encodings], dim=0
        )
        if not bool(torch.isfinite(values).all()):
            raise EvaluationError("STU MaxLogit contains a non-finite value")
        return values.detach().cpu().numpy().astype(np.float64, copy=False)

    def _model_logits(self, window: object) -> np.ndarray:
        if self.model is None:
            raise EvaluationError("AJAE model is unavailable")
        sources, encodings = self._window_content(window)
        points = getattr(window, "points")

        def concatenate(name: str) -> torch.Tensor:
            return torch.cat([getattr(item, name) for item in encodings], dim=0).to(
                self.device
            )

        coordinates = torch.as_tensor(
            np.asarray(getattr(points, "coordinates")),
            dtype=torch.float32,
            device=self.device,
        )
        stu_features = concatenate("point_features")
        normal_evidence = concatenate("normal_evidence")
        reliability_assign = concatenate("reliability_assign")
        reliability_noobj = concatenate("reliability_noobj")
        intensity = torch.cat(
            [
                torch.as_tensor(
                    np.asarray(getattr(source, "xyzi"))[
                        np.asarray(getattr(source, "real_slots")), 3
                    ],
                    dtype=stu_features.dtype,
                    device=self.device,
                )
                for source in sources
            ],
            dim=0,
        )
        scan_group = torch.as_tensor(
            np.asarray(getattr(points, "scan_group")),
            dtype=torch.long,
            device=self.device,
        )
        if coordinates.shape[0] != stu_features.shape[0]:
            raise EvaluationError(
                "model content does not align with window point order"
            )

        if self.condition is ExperimentCondition.B1:
            chunks: list[torch.Tensor] = []
            offset = 0
            for source in sources:
                count = int(getattr(source, "real_count"))
                selected = slice(offset, offset + count)
                # Each member is an independent forward in the same window frame.
                logits = self.model(
                    coordinates[selected],
                    stu_features[selected],
                    normal_evidence[selected],
                    reliability_assign[selected],
                    reliability_noobj[selected],
                    intensity[selected],
                    torch.zeros(count, dtype=torch.long, device=self.device),
                    grouping_mode="single",
                )
                chunks.append(logits)
                offset += count
            if offset != coordinates.shape[0]:
                raise EvaluationError("B1 did not consume every window point")
            output = torch.cat(chunks, dim=0)
        else:
            grouping_mode = (
                "per_scan" if self.condition is ExperimentCondition.B2 else "joint"
            )
            output = self.model(
                coordinates,
                stu_features,
                normal_evidence,
                reliability_assign,
                reliability_noobj,
                intensity,
                scan_group,
                grouping_mode=grouping_mode,
            )
        if output.shape != (coordinates.shape[0],) or not bool(
            torch.isfinite(output).all()
        ):
            raise EvaluationError("model must return one finite logit for every point")
        return output.detach().cpu().numpy().astype(np.float64, copy=False)

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def predict_sequence(
        self,
        sequence: object,
        *,
        output_dir: Path | str | None = None,
    ) -> SequencePrediction:
        """Score every legal window and restore the covered union to file slots."""

        self._assert_components_unchanged()
        if output_dir is not None and self.identity.test_fixture:
            raise EvaluationError("test-only inference cannot write formal artifacts")
        self.cache.clear()
        spec = getattr(sequence, "spec")
        partition = str(getattr(spec, "partition"))
        sequence_id = int(getattr(spec, "sequence_id"))
        if not self.identity.test_fixture:
            sequence_protocol = getattr(sequence, "protocol", None)
            if (
                getattr(sequence_protocol, "scientific_identity", None)
                != self.identity.protocol_identity
                or getattr(self.protocol, "sequence")(partition, sequence_id) != spec
            ):
                raise EvaluationError("sequence does not belong to the frozen protocol")
        expected_starts = tuple(spec.legal_window_starts())
        starts = tuple(int(value) for value in getattr(sequence, "window_starts"))
        if starts != expected_starts or not starts:
            raise EvaluationError(
                "sequence must expose every legal window start exactly once"
            )
        self._cache_namespace = ("real", partition, sequence_id)
        fusion = WindowScoreFusion(fusion_value=self.identity.fusion_value)
        durations: list[float] = []
        processed_points = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._synchronize(self.device)
        total_started = time.perf_counter()
        with torch.inference_mode():
            for window_start in starts:
                self._synchronize(self.device)
                started = time.perf_counter()
                window = self._audited_window(sequence, window_start)
                points = getattr(window, "points")
                identity = {
                    "partition": partition,
                    "sequence_id": sequence_id,
                    "window_start": window_start,
                    "source_frame": np.asarray(getattr(points, "source_frame")),
                    "source_ray": np.asarray(getattr(points, "source_ray")),
                    "source_slot": np.asarray(getattr(points, "source_slot")),
                }
                if self.condition is ExperimentCondition.B0:
                    fusion.add_real_scores(
                        **identity, scores=self._baseline_scores(window)
                    )
                else:
                    fusion.add_real(**identity, logits=self._model_logits(window))
                processed_points += int(getattr(points, "count"))
                self._synchronize(self.device)
                durations.append(time.perf_counter() - started)
        self._synchronize(self.device)
        elapsed = time.perf_counter() - total_started
        fused = fusion.finalize_real(partition, sequence_id)
        self._assert_components_unchanged()

        expected_occurrence = Counter(
            frame_id for start in starts for frame_id in spec.window_frame_ids(start)
        )
        frames: dict[int, FramePrediction] = {}
        for frame_id in sorted(expected_occurrence):
            source = sequence.source_frame(frame_id)
            selected = fused.frame_mask(frame_id)
            slots = fused.source_slot[selected]
            rays = fused.source_ray[selected]
            real_slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
            if (
                slots.size != real_slots.size
                or np.unique(slots).size != slots.size
                or not np.array_equal(np.sort(slots), np.sort(real_slots))
            ):
                raise EvaluationError(
                    "fusion did not restore every visible return once"
                )
            mapping = self._source_ray_map(source)
            if not np.array_equal(mapping[slots], rays):
                raise EvaluationError("fused ray identity disagrees with source slots")
            counts = fused.occurrence_count[selected]
            if not np.all(counts == expected_occurrence[frame_id]):
                raise EvaluationError(
                    "point occurrence count disagrees with legal windows"
                )
            score = np.zeros(int(getattr(source, "slot_count")), dtype=np.float64)
            occurrence = np.zeros(score.size, dtype=np.uint8)
            score[slots] = fused.score[selected]
            occurrence[slots] = counts
            frames[frame_id] = FramePrediction(
                score, occurrence, self.identity.fusion_value
            )

        destination = None if output_dir is None else Path(output_dir)
        if destination is not None:
            for frame_id, prediction in frames.items():
                write_point_scores(
                    destination / f"{frame_id:06d}.txt",
                    prediction.score,
                    fusion_value=self.identity.fusion_value,
                )
            save_result(
                destination / "coverage.json",
                {
                    "format": PREDICTION_COVERAGE_FORMAT,
                    "condition": self.condition.value,
                    "partition": partition,
                    "sequence_id": sequence_id,
                    "window_starts": list(starts),
                    "frame_ids": list(frames),
                    "domain": "visible_returns_covered_by_at_least_one_legal_window",
                    "fusion_value": self.identity.fusion_value,
                    "fusion_reduction": FUSION_REDUCTION,
                    "evaluation_identity": self.identity.to_dict(),
                    "occurrence_histogram": fused.occurrence_histogram,
                    "padding_or_zero_fill_used": False,
                },
            )
        cache_total = self.cache.hits + self.cache.misses
        cost: dict[str, Any] = {
            "windows": len(starts),
            "covered_frames": len(frames),
            "unique_visible_points": fused.unique_point_count,
            "processed_window_points": processed_points,
            "latency_seconds": {
                "total": elapsed,
                "mean_per_window": float(np.mean(durations)),
                "p95_per_window": float(np.quantile(durations, 0.95)),
            },
            "throughput_window_points_per_second": processed_points
            / max(elapsed, 1.0e-12),
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            ),
            "stu_cache": {
                "capacity_frames": self.cache.capacity,
                "hits": self.cache.hits,
                "misses": self.cache.misses,
                "hit_rate": self.cache.hits / cache_total if cache_total else 0.0,
            },
            "time_budget_seconds": self.time_budget_seconds,
            "time_budget_exceeded": (
                elapsed > self.time_budget_seconds
                if self.time_budget_seconds is not None
                else False
            ),
        }
        return SequencePrediction(
            self.condition.value,
            partition,
            sequence_id,
            starts,
            frames,
            fused,
            self.identity,
            cost,
        )


def evaluate_sequence(
    sequence: object,
    prediction: SequencePrediction,
    protocol: _Protocol | None = None,
) -> dict[str, Any]:
    """Evaluate the complete covered frame-ray union and occurrence strata."""

    spec = getattr(sequence, "spec")
    if (
        prediction.partition != getattr(spec, "partition")
        or prediction.sequence_id != getattr(spec, "sequence_id")
        or prediction.window_starts != tuple(spec.legal_window_starts())
    ):
        raise EvaluationError("prediction identity does not match the source sequence")
    active_protocol = (
        protocol if protocol is not None else getattr(sequence, "protocol", None)
    )
    if (
        not prediction.identity.test_fixture
        and getattr(active_protocol, "scientific_identity", None)
        != prediction.identity.protocol_identity
    ):
        raise EvaluationError("evaluation protocol differs from prediction identity")
    pooled = PointMetricAccumulator(active_protocol)
    stratified = OccurrenceMetricAccumulator(active_protocol)
    covered_points = 0
    for frame_id, frame_prediction in prediction.frames.items():
        source = sequence.source_frame(frame_id)
        labels = getattr(source, "labels", None)
        if labels is None:
            raise EvaluationError("evaluation requires source semantic labels")
        xyzi = np.asarray(getattr(source, "xyzi"))
        semantic = np.asarray(getattr(labels, "semantic"))
        real_slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
        if not np.all(frame_prediction.occurrence_count[real_slots] >= 1):
            raise EvaluationError("a covered frame has an unscored visible return")
        covered_points += int(real_slots.size)
        pooled.update(xyzi, frame_prediction.score, semantic)
        stratified.update(
            xyzi,
            frame_prediction.score,
            semantic,
            frame_prediction.occurrence_count,
        )
    return {
        "condition": prediction.condition,
        "partition": prediction.partition,
        "sequence_id": prediction.sequence_id,
        "domain": "visible_returns_covered_by_at_least_one_legal_window",
        "fusion_value": prediction.identity.fusion_value,
        "fusion_reduction": FUSION_REDUCTION,
        "evaluation_identity": prediction.identity.to_dict(),
        "covered_frames": len(prediction.frames),
        "covered_visible_points": covered_points,
        "occurrence_histogram": prediction.fused.occurrence_histogram,
        "point_metrics": pooled.compute(),
        "occurrence_strata": stratified.compute(),
    }


@dataclass(frozen=True, slots=True)
class ConditionEvaluation:
    prediction: SequencePrediction
    metrics: Mapping[str, Any]


def evaluate_condition(
    inference: AJAEInference,
    sequence: object,
    *,
    output_dir: Path | str | None = None,
) -> ConditionEvaluation:
    """Run and evaluate one registered schema-31 condition."""

    prediction = inference.predict_sequence(sequence, output_dir=output_dir)
    return ConditionEvaluation(
        prediction,
        evaluate_sequence(sequence, prediction, protocol=inference.protocol),
    )


@dataclass(frozen=True, slots=True)
class DevelopmentClipResult:
    clip_identity: str
    world_identity: str
    source_observation_identities: tuple[str, ...]
    mechanism: str
    fused_point_ap: float
    unique_point_count: int
    occurrence_count: int
    occurrence_histogram: Mapping[str, int]
    frame_count: int
    window_count: int


@dataclass(frozen=True, slots=True)
class DevelopmentFusedAP:
    condition: str
    clips: tuple[DevelopmentClipResult, ...]
    evaluation_identity: EvaluationIdentity
    fusion_semantics: str

    def __post_init__(self) -> None:
        selected = _condition(self.condition)
        if (
            not isinstance(self.evaluation_identity, EvaluationIdentity)
            or self.evaluation_identity.test_fixture
            or self.evaluation_identity.condition != selected.value
            or self.fusion_semantics != _fusion_semantics(selected)
        ):
            raise EvaluationError("development fusion semantics changed")
        if (
            len(self.clips) != 30
            or len({item.clip_identity for item in self.clips}) != 30
            or len({item.world_identity for item in self.clips}) != 30
            or sum(item.mechanism == "in_generator" for item in self.clips) != 24
            or sum(item.mechanism == "torus_SDF" for item in self.clips) != 6
        ):
            raise EvaluationError(
                "development results must preserve the frozen 24+6 clips"
            )
        for item in self.clips:
            _sha256(item.clip_identity, "development clip identity")
            _sha256(item.world_identity, "development world identity")
            if len(item.source_observation_identities) != 9:
                raise EvaluationError("development result must bind nine observations")
            for identity in item.source_observation_identities:
                _sha256(identity, "development source observation identity")
            if (
                not math.isfinite(item.fused_point_ap)
                or not 0.0 <= item.fused_point_ap <= 1.0
                or item.unique_point_count < 1
                or item.occurrence_count < item.unique_point_count
                or item.occurrence_count > 5 * item.unique_point_count
                or item.frame_count != 9
                or item.window_count != 5
            ):
                raise EvaluationError("development clip result is invalid")
            if set(item.occurrence_histogram) != {str(count) for count in range(1, 6)}:
                raise EvaluationError("development clip lacks occurrence strata 1..5")
            histogram = {
                count: int(item.occurrence_histogram[str(count)])
                for count in range(1, 6)
            }
            if (
                any(value < 0 for value in histogram.values())
                or sum(histogram.values()) != item.unique_point_count
                or sum(count * value for count, value in histogram.items())
                != item.occurrence_count
            ):
                raise EvaluationError("development occurrence counts are inconsistent")

    @property
    def macro_fused_point_ap(self) -> float:
        """Return the frozen checkpoint metric over in-generator clips only."""

        return self.in_generator_macro_fused_point_ap

    @property
    def all_clips_macro_fused_point_ap(self) -> float:
        """Descriptive 30-clip mean; it is ineligible for model selection."""

        return float(np.mean([item.fused_point_ap for item in self.clips]))

    @property
    def in_generator_macro_fused_point_ap(self) -> float:
        """Checkpoint-selection metric over the 24 in-generator clips only."""

        return float(
            np.mean(
                [
                    item.fused_point_ap
                    for item in self.clips
                    if item.mechanism == "in_generator"
                ]
            )
        )

    @property
    def held_out_macro_fused_point_ap(self) -> float:
        """Diagnostic metric over the six held-out torus clips only."""

        return float(
            np.mean(
                [
                    item.fused_point_ap
                    for item in self.clips
                    if item.mechanism == "torus_SDF"
                ]
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "ajae-schema31-development-evaluation-v1",
            "condition": self.condition,
            "fusion_semantics": self.fusion_semantics,
            "fusion_value": self.evaluation_identity.fusion_value,
            "fusion_reduction": FUSION_REDUCTION,
            "evaluation_identity": self.evaluation_identity.to_dict(),
            "selection_mechanism": "in_generator",
            "selection_clip_count": 24,
            "macro_fused_point_ap": self.macro_fused_point_ap,
            "all_clips_macro_fused_point_ap": self.all_clips_macro_fused_point_ap,
            "held_out_macro_fused_point_ap": self.held_out_macro_fused_point_ap,
            "clips": [
                {
                    "clip_identity": item.clip_identity,
                    "world_identity": item.world_identity,
                    "source_observation_identities": list(
                        item.source_observation_identities
                    ),
                    "mechanism": item.mechanism,
                    "fused_point_ap": item.fused_point_ap,
                    "unique_point_count": item.unique_point_count,
                    "occurrence_count": item.occurrence_count,
                    "occurrence_histogram": dict(item.occurrence_histogram),
                    "frame_count": item.frame_count,
                    "window_count": item.window_count,
                }
                for item in self.clips
            ],
        }


def development_fused_ap(
    inference: AJAEInference,
    development_worlds: object,
    rendered_clips: Iterable[object],
    *,
    protocol: object,
) -> DevelopmentFusedAP:
    """Evaluate the exact frozen 30-by-9-by-5 development population."""

    try:
        from .protocol import AJAEProtocol, DevelopmentWorlds
        from .render import DevelopmentClipWorld, source_observation_identity
        from .scene import assemble_window
    except ImportError:  # pragma: no cover - direct module execution
        from protocol import AJAEProtocol, DevelopmentWorlds
        from render import DevelopmentClipWorld, source_observation_identity
        from scene import assemble_window

    if type(inference) is not AJAEInference:
        raise TypeError("development evaluation requires AJAEInference")
    if type(protocol) is not AJAEProtocol or type(development_worlds) is not (
        DevelopmentWorlds
    ):
        raise TypeError("development evaluation requires schema-31 protocol objects")
    inference._assert_components_unchanged()
    if (
        inference.identity.test_fixture
        or getattr(inference.protocol, "scientific_identity", None)
        != getattr(protocol, "scientific_identity", None)
        or (inference.identity.protocol_identity != protocol.scientific_identity)
    ):
        raise EvaluationError("development evaluation requires formal frozen inference")
    if (
        not development_worlds.validated
        or development_worlds.protocol_schema != 31
        or development_worlds.protocol_identity
        != protocol.development_population_identity
        or development_worlds.sequence_id != 201
    ):
        raise EvaluationError(
            "development definitions are not validated for this protocol"
        )
    definitions = tuple(development_worlds.clips)
    if (
        len(definitions) != 30
        or sum(item.mechanism == "in_generator" for item in definitions) != 24
        or sum(item.mechanism == "torus_SDF" for item in definitions) != 6
        or any(
            len(item.frame_ids) != 9 or len(item.windows) != 5 for item in definitions
        )
    ):
        raise EvaluationError("development population is not the frozen 30x9x5 design")

    runtime = tuple(rendered_clips)
    if len(runtime) != 30 or any(
        type(item) is not DevelopmentClipWorld for item in runtime
    ):
        raise EvaluationError(
            "rendered development input must contain exactly 30 clips"
        )
    runtime_by_identity = {item.identity: item for item in runtime}
    if len(runtime_by_identity) != 30 or set(runtime_by_identity) != {
        item.identity for item in definitions
    }:
        raise EvaluationError("rendered clips differ from the frozen clip identities")

    selected = inference.condition
    results: list[DevelopmentClipResult] = []
    for definition in definitions:
        clip = runtime_by_identity[definition.identity]
        if (
            clip.world.identity != definition.world_identity
            or clip.clip_start != definition.clip_start
            or tuple(clip.frame_ids) != tuple(definition.frame_ids)
            or clip.renderer_identity != definition.renderer_identity
            or clip.mechanism != definition.mechanism
            or tuple(clip.source_observation_identities)
            != tuple(definition.source_observation_identities)
            or _plain_json(clip.world.to_dict()) != _plain_json(definition.world)
            or _plain_json(clip.report.to_dict()) != _plain_json(definition.report)
        ):
            raise EvaluationError("rendered clip differs from its frozen WorldSpec")

        fusion = WindowScoreFusion(fusion_value=inference.identity.fusion_value)
        truth: dict[tuple[int, int], tuple[int, np.ndarray, int]] = {}
        observations: dict[int, tuple[str, object]] = {}
        inference.cache.clear()
        inference._cache_namespace = ("synthetic", definition.world_identity)
        for frozen_window, rendered_window in zip(
            definition.windows, clip.windows, strict=True
        ):
            if (
                rendered_window.identity != frozen_window.identity
                or rendered_window.window_start != frozen_window.window_start
                or tuple(rendered_window.frame_ids) != tuple(frozen_window.frame_ids)
                or tuple(rendered_window.source_observation_identities)
                != tuple(frozen_window.source_observation_identities)
                or _plain_json([item.to_dict() for item in rendered_window.descriptors])
                != _plain_json(frozen_window.descriptors)
            ):
                raise EvaluationError(
                    "rendered window differs from its frozen identity"
                )
            sources = tuple(item.source for item in rendered_window.rendered_frames)
            mappings: dict[int, np.ndarray] = {}
            for source in sources:
                frame_id = int(source.frame_id)
                observation_identity = source_observation_identity(source)
                definition_index = definition.frame_ids.index(frame_id)
                if (
                    observation_identity
                    != definition.source_observation_identities[definition_index]
                ):
                    raise EvaluationError(
                        "rendered source content differs from its frozen observation"
                    )
                previous = observations.get(frame_id)
                if previous is not None and previous[0] != observation_identity:
                    raise EvaluationError(
                        "one development frame changed across overlapping windows"
                    )
                observations[frame_id] = (observation_identity, source)
                mappings[frame_id] = inference._source_ray_map(source)
            window = assemble_window(
                protocol.development_sequence,
                rendered_window.window_start,
                rendered_window.frame_ids,
                sources,
                condition=selected,
                canonical_ray_by_slot=mappings,
                ray_mapping_audited=True,
                ray_mapping_digest=inference.ray_mapping_digest,
            )
            if not np.allclose(
                window.reference_pose.world_from_window,
                rendered_window.reference_pose.world_from_window,
                atol=1.0e-9,
                rtol=1.0e-9,
            ):
                raise EvaluationError(
                    "rendered and evaluated symmetric coordinates differ"
                )
            points = window.points
            labels = window.labels
            if labels is None or points.count != sum(
                source.real_count for source in sources
            ):
                raise EvaluationError(
                    "development window omits points or semantic labels"
                )
            identity = {
                "world_identity": definition.world_identity,
                "window_start": rendered_window.window_start,
                "source_frame": points.source_frame,
                "source_ray": points.source_ray,
                "source_slot": points.source_slot,
            }
            if selected is ExperimentCondition.B0:
                fusion.add_synthetic_scores(
                    **identity, scores=inference._baseline_scores(window)
                )
            else:
                fusion.add_synthetic(**identity, logits=inference._model_logits(window))

            source_by_frame = {int(item.frame_id): item for item in sources}
            for index in range(points.count):
                frame = int(points.source_frame[index])
                ray = int(points.source_ray[index])
                slot = int(points.source_slot[index])
                source = source_by_frame[frame]
                xyzi = np.asarray(source.xyzi[slot], dtype=np.float64)
                semantic = int(source.labels.semantic[slot])
                if semantic != int(labels.semantic[index]):
                    raise EvaluationError(
                        "window semantic is misaligned with its source slot"
                    )
                key = (frame, ray)
                previous = truth.get(key)
                if previous is not None and (
                    previous[0] != slot
                    or previous[2] != semantic
                    or not np.array_equal(previous[1], xyzi)
                ):
                    raise EvaluationError(
                        "one development point changed identity, coordinate, or semantic"
                    )
                truth[key] = (slot, xyzi.copy(), semantic)

        if set(observations) != set(definition.frame_ids):
            raise EvaluationError(
                "development clip does not expose all nine source frames"
            )
        fused = fusion.finalize_synthetic(definition.world_identity)
        fused_keys = tuple(
            zip(
                fused.source_frame.astype(int).tolist(),
                fused.source_ray.astype(int).tolist(),
                strict=True,
            )
        )
        expected_occurrence = Counter(
            frame_id for window in definition.windows for frame_id in window.frame_ids
        )
        expected_keys: set[tuple[int, int]] = set()
        for frame_id, (_, source) in observations.items():
            slots = np.asarray(source.real_slots, dtype=np.int64)
            rays = inference._source_ray_map(source)[slots]
            expected_keys.update((frame_id, int(ray)) for ray in rays)
        if set(fused_keys) != expected_keys or set(truth) != expected_keys:
            raise EvaluationError(
                "development fusion omitted or invented a source point"
            )
        for index, key in enumerate(fused_keys):
            slot, _, _ = truth[key]
            if (
                int(fused.source_slot[index]) != slot
                or int(fused.occurrence_count[index]) != expected_occurrence[key[0]]
            ):
                raise EvaluationError(
                    "development point occurrence identity is incomplete"
                )

        xyzi = np.stack([truth[key][1] for key in fused_keys])
        semantic = np.asarray([truth[key][2] for key in fused_keys], dtype=np.int64)
        anomaly, valid = binary_targets(semantic)
        valid &= _range_mask(protocol, xyzi)
        if not np.any(valid):
            raise EvaluationError("development clip has no internally valid point")
        labels = anomaly[valid]
        scores = fused.score[valid]
        if not labels.any() or labels.all():
            raise EvaluationError("each development clip needs both evaluation classes")
        results.append(
            DevelopmentClipResult(
                definition.identity,
                definition.world_identity,
                tuple(definition.source_observation_identities),
                definition.mechanism,
                float(average_precision_score(labels, scores)),
                int(valid.sum()),
                int(fused.occurrence_count[valid].astype(np.int64).sum()),
                {
                    str(count): int(np.sum(valid & (fused.occurrence_count == count)))
                    for count in range(1, 6)
                },
                len(definition.frame_ids),
                len(definition.windows),
            )
        )
    inference._assert_components_unchanged()
    return DevelopmentFusedAP(
        selected.value,
        tuple(results),
        inference.identity,
        _fusion_semantics(selected),
    )


@dataclass(frozen=True, slots=True)
class MethodFreezeRecord:
    """Validated, content-addressed authorization for sealed sequence access."""

    path: Path
    record_sha256: str
    evaluation_identity: EvaluationIdentity
    sealed_sequences: Mapping[str, tuple[int, ...]]

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        expected_identity: EvaluationIdentity,
        protocol: object,
    ) -> MethodFreezeRecord:
        if type(expected_identity) is not EvaluationIdentity:
            raise TypeError("expected_identity must be EvaluationIdentity")
        if expected_identity.test_fixture:
            raise EvaluationError("test fixtures cannot authorize sealed data")
        requested = Path(path).expanduser()
        try:
            resolved = requested.resolve(strict=True)
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationError("method-freeze record is unreadable") from error
        keys = {
            "format",
            "status",
            "evaluation_identity",
            "sealed_sequences",
            "record_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != keys:
            raise EvaluationError("method-freeze record has an invalid schema")
        unsigned = {key: payload[key] for key in keys - {"record_sha256"}}
        record_sha256 = _sha256(payload["record_sha256"], "method-freeze record")
        if record_sha256 != _json_sha256(unsigned):
            raise EvaluationError("method-freeze record content hash does not match")
        raw_identity = payload["evaluation_identity"]
        identity_fields = set(expected_identity.to_dict())
        if (
            not isinstance(raw_identity, Mapping)
            or set(raw_identity) != identity_fields
        ):
            raise EvaluationError("method-freeze record has an invalid method identity")
        try:
            record_identity = EvaluationIdentity(**dict(raw_identity))
        except (TypeError, ValueError) as error:
            raise EvaluationError(
                "method-freeze record has an invalid method identity"
            ) from error
        if (
            payload["format"] != METHOD_FREEZE_FORMAT
            or payload["status"] != METHOD_FREEZE_STATUS
            or record_identity != expected_identity
        ):
            raise EvaluationError("method-freeze record does not match this method")
        expected_sequences = {
            "val": list(getattr(protocol, "public_sequence_ids")),
            "test": list(getattr(protocol, "hidden_sequence_ids")),
        }
        raw_sequences = payload["sealed_sequences"]
        if (
            not isinstance(raw_sequences, Mapping)
            or set(raw_sequences) != {"val", "test"}
            or any(
                type(values) is not list
                or any(type(sequence_id) is not int for sequence_id in values)
                for values in raw_sequences.values()
            )
            or raw_sequences != expected_sequences
        ):
            raise EvaluationError(
                "method-freeze record does not bind the sealed population"
            )
        return cls(
            resolved,
            record_sha256,
            record_identity,
            {name: tuple(values) for name, values in expected_sequences.items()},
        )


def _require_sealed_node(protocol: object, partition: str) -> None:
    if partition not in {"val", "test"}:
        raise EvaluationError("sealed node checks apply only to val or test")
    status = getattr(protocol, "status", None)
    node = status.get("current_node") if isinstance(status, Mapping) else None
    allowed = {"V01", "T01"} if partition == "val" else {"T01"}
    if node not in allowed:
        raise EvaluationError(
            f"sealed {partition} data are unavailable at protocol node {node!r}"
        )


def open_sealed_sequence(
    data_root: Path | str,
    *,
    protocol: object,
    partition: str,
    sequence_id: int,
    condition: ExperimentCondition | str,
    label_mode: object,
    inference: AJAEInference | None = None,
    method_freeze_path: Path | str | None = None,
) -> object:
    """Open val/test only after the exact frozen method record is validated."""

    try:
        from .scene import STUSequence, _grant_sealed_sequence_access
    except ImportError:  # pragma: no cover - direct module execution
        from scene import STUSequence, _grant_sealed_sequence_access

    selected = _condition(condition)
    access = None
    if partition in {"val", "test"}:
        # A matching receipt identifies a method; the protocol node grants access.
        _require_sealed_node(protocol, partition)
        if (
            type(inference) is not AJAEInference
            or inference.protocol is not protocol
            or inference.condition is not selected
            or inference.identity.protocol_identity
            != getattr(protocol, "scientific_identity", None)
            or method_freeze_path is None
        ):
            raise EvaluationError(
                "sealed sequence access lacks the frozen method identity"
            )
        inference._assert_components_unchanged()
        # This record is validated before STUSequence can resolve any sealed data path.
        record = MethodFreezeRecord.load(
            method_freeze_path,
            expected_identity=inference.identity,
            protocol=protocol,
        )
        if sequence_id not in record.sealed_sequences[partition]:
            raise EvaluationError("sequence is outside the frozen sealed population")
        access = _grant_sealed_sequence_access(
            protocol, partition=partition, condition=selected.value
        )
    return STUSequence.open(
        data_root,
        protocol=protocol,
        partition=partition,
        sequence_id=sequence_id,
        label_mode=label_mode,
        sealed_access=access,
    )


def evaluate_frames(
    frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    protocol: _Protocol | None = None,
) -> dict[str, float | int]:
    """Evaluate an iterable of (xyz, scores, raw semantic) frames."""

    accumulator = PointMetricAccumulator(protocol)
    for points, scores, raw_semantic in frames:
        accumulator.update(points, scores, raw_semantic)
    return accumulator.compute()


def write_point_scores(
    path: Path | str,
    scores: np.ndarray,
    *,
    fusion_value: str,
) -> None:
    """Write one finite score per original slot at the text boundary only."""

    destination = Path(path)
    values = _finite_vector("scores", scores)
    if fusion_value not in {B0_FUSION_VALUE, AJAE_FUSION_VALUE}:
        raise EvaluationError("score output has an unsupported domain")
    if fusion_value == AJAE_FUSION_VALUE and np.any((values < 0.0) | (values > 1.0)):
        raise EvaluationError("point probabilities must lie in [0,1]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, values.astype(np.float64, copy=False), fmt="%.17g")


def load_prediction_coverage(
    directory: Path | str,
    *,
    condition: ExperimentCondition | str,
    partition: str,
    sequence_id: int,
    expected_window_starts: Iterable[int],
    expected_identity: EvaluationIdentity,
) -> Mapping[str, Any]:
    """Validate the schema-31 covered-union manifest before reading scores."""

    if type(expected_identity) is not EvaluationIdentity:
        raise TypeError("expected_identity must be EvaluationIdentity")
    path = Path(directory) / "coverage.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read prediction coverage: {path}") from error
    expected_keys = {
        "format",
        "condition",
        "partition",
        "sequence_id",
        "window_starts",
        "frame_ids",
        "domain",
        "fusion_value",
        "fusion_reduction",
        "evaluation_identity",
        "occurrence_histogram",
        "padding_or_zero_fill_used",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise EvaluationError("prediction coverage has an invalid schema")
    starts = tuple(expected_window_starts)
    if (
        payload["format"] != PREDICTION_COVERAGE_FORMAT
        or payload["condition"] != _condition(condition).value
        or payload["partition"] != partition
        or payload["sequence_id"] != sequence_id
        or tuple(payload["window_starts"]) != starts
        or payload["domain"] != "visible_returns_covered_by_at_least_one_legal_window"
        or payload["fusion_value"] != _fusion_value(condition)
        or payload["fusion_reduction"] != FUSION_REDUCTION
        or payload["evaluation_identity"] != expected_identity.to_dict()
        or expected_identity.condition != _condition(condition).value
        or payload["padding_or_zero_fill_used"] is not False
    ):
        raise EvaluationError("prediction coverage changed the schema-31 domain")
    expected_frames = sorted(
        {frame_id for start in starts for frame_id in range(start, start + 5)}
    )
    if payload["frame_ids"] != expected_frames:
        raise EvaluationError("coverage frame IDs do not equal the legal-window union")
    histogram = payload["occurrence_histogram"]
    if not isinstance(histogram, Mapping) or set(histogram) != {
        str(value) for value in range(1, 6)
    }:
        raise EvaluationError("coverage lacks occurrence strata 1..5")
    return payload


def save_result(path: Path | str, result: Mapping[str, Any]) -> None:
    """Atomically save a finite JSON evaluation artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
