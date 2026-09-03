#!/usr/bin/env python3
"""Schema-31 window inference, point-identity fusion, and STU metrics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from scipy.stats import t as student_t_distribution
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
# Kept for the narrow legacy loader used by the pre-M01 mechanical test.  Sealed
# access accepts only the multi-method v2 record below.
METHOD_FREEZE_FORMAT = "ajae-schema31-method-freeze-v1"
MULTI_METHOD_FREEZE_FORMAT = "ajae-schema31-method-freeze-v2"
METHOD_FREEZE_STATUS = "frozen_before_sealed_data_access"
FORMAL_EVALUATION_FORMAT = "ajae-schema31-formal-evaluation-v1"
FORMAL_METRIC_EVIDENCE_FORMAT = "ajae-schema31-formal-metric-evidence-v1"
NORMAL_SAFETY_EVIDENCE_FORMAT = "ajae-schema31-normal-safety-evidence-v1"
FORMAL_GATE_VERDICT_FORMAT = "ajae-schema31-formal-gate-verdict-v1"
S01_SHIFT_AUDIT_FORMAT = "ajae-schema31-s01-shift-audit-v1"
S01_CLIP_EVIDENCE_FORMAT = "ajae-schema31-s01-clip-evidence-v1"
PUBLIC_SEQUENCE_RESULT_FORMAT = "ajae-schema31-public-sequence-result-v1"
PUBLIC_METRIC_EVIDENCE_FORMAT = "ajae-schema31-public-metric-evidence-v1"
V01_VERDICT_FORMAT = "ajae-schema31-v01-verdict-v1"
PAIRED_STUDENT_T_METHOD = "paired_two_sided_student_t_v1"
METHOD_SELECTION_RULE = "lowest_frozen_formal_seed_result_blind"
METHOD_ROLES = ("B0_reference", "B1_reference", "frozen_final")
FROZEN_IMPLEMENTATION_FILES = (
    "src/protocol.py",
    "src/scene.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
)
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
        if stu_state != STU_MODEL_STATE_TENSOR_SHA256 or stu_state != _sha256(
            stu.get("model_state_tensor_sha256"),
            "protocol STU model-state identity",
        ):
            raise EvaluationError(
                "runtime STU weights differ from the frozen official model"
            )
        source_manifest = stu_source_manifest(encoder.official_repository)
        source_identity = _sha256(
            source_manifest.get("manifest_sha256"), "STU source manifest identity"
        )
        if source_identity != _sha256(
            stu.get("source_manifest_sha256"), "protocol STU source identity"
        ):
            raise EvaluationError("runtime STU source differs from the protocol")
        render = getattr(protocol, "render")
        ray_grid = render.get("ray_grid")
        if (
            not isinstance(ray_grid, Mapping)
            or _sha256(calibration_sha256, "calibration identity")
            != _sha256(
                render.get("calibration_sha256"), "protocol calibration identity"
            )
            or _sha256(ray_mapping_digest, "ray mapping identity")
            != _sha256(
                ray_grid.get("canonical_sha256"),
                "protocol canonical ray-mapping identity",
            )
        ):
            raise EvaluationError(
                "runtime calibration or ray mapping differs from the protocol"
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
    """One training-visible in-generator development clip result."""

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
    saturated_probability_count: int = 0

    def __post_init__(self) -> None:
        if self.mechanism != "in_generator":
            raise EvaluationError(
                "development results are restricted to in-generator clips"
            )


@dataclass(frozen=True, slots=True)
class DevelopmentFusedAP:
    """The only development metric visible before final method freeze."""

    condition: str
    clips: tuple[DevelopmentClipResult, ...]
    evaluation_identity: EvaluationIdentity
    fusion_semantics: str
    raw_clip_samples: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None

    def __post_init__(self) -> None:
        selected = _condition(self.condition)
        if (
            type(self.evaluation_identity) is not EvaluationIdentity
            or self.evaluation_identity.test_fixture
            or self.evaluation_identity.condition != selected.value
            or self.fusion_semantics != _fusion_semantics(selected)
        ):
            raise EvaluationError("development fusion semantics changed")
        if (
            type(self.clips) is not tuple
            or len(self.clips) != 24
            or any(
                type(item) is not DevelopmentClipResult
                or item.mechanism != "in_generator"
                for item in self.clips
            )
            or len({item.clip_identity for item in self.clips}) != 24
            or len({item.world_identity for item in self.clips}) != 24
        ):
            raise EvaluationError(
                "development results require the frozen 24 in-generator clips"
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
                or item.saturated_probability_count < 0
                or item.saturated_probability_count > item.unique_point_count
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
        if self.raw_clip_samples is not None:
            if tuple(self.raw_clip_samples) != tuple(
                item.clip_identity for item in self.clips
            ):
                raise EvaluationError("development raw samples changed clip order")
            for item in self.clips:
                labels, scores = self.raw_clip_samples[item.clip_identity]
                label = np.asarray(labels)
                score = np.asarray(scores)
                if (
                    label.dtype != np.dtype(np.bool_)
                    or label.ndim != 1
                    or score.dtype != np.dtype(np.float64)
                    or score.shape != label.shape
                    or label.size != item.unique_point_count
                    or not np.isfinite(score).all()
                    or not math.isclose(
                        official_metrics(label, score)["AP"] / 100.0,
                        item.fused_point_ap,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise EvaluationError("development raw samples do not reproduce AP")

    @property
    def macro_fused_point_ap(self) -> float:
        """Return the macro AP over the only development-visible 24 clips."""

        return math.fsum(item.fused_point_ap for item in self.clips) / len(self.clips)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "ajae-schema31-in-generator-development-evaluation-v1",
            "condition": self.condition,
            "fusion_semantics": self.fusion_semantics,
            "fusion_value": self.evaluation_identity.fusion_value,
            "fusion_reduction": FUSION_REDUCTION,
            "evaluation_identity": self.evaluation_identity.to_dict(),
            "mechanism": "in_generator",
            "clip_count": 24,
            "macro_fused_point_ap": self.macro_fused_point_ap,
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
                    "saturated_probability_count": item.saturated_probability_count,
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
    """Evaluate only the frozen 24 in-generator development clips."""

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
        len(definitions) != 24
        or any(item.mechanism != "in_generator" for item in definitions)
        or any(
            len(item.frame_ids) != 9 or len(item.windows) != 5 for item in definitions
        )
    ):
        raise EvaluationError(
            "development population is not the frozen 24x9x5 in-generator design"
        )

    runtime = tuple(rendered_clips)
    if len(runtime) != 24 or any(
        type(item) is not DevelopmentClipWorld or item.mechanism != "in_generator"
        for item in runtime
    ):
        raise EvaluationError(
            "development scoring accepts exactly 24 in-generator clips"
        )
    runtime_by_identity = {item.identity: item for item in runtime}
    if len(runtime_by_identity) != 24 or set(runtime_by_identity) != {
        item.identity for item in definitions
    }:
        raise EvaluationError("rendered clips differ from the frozen clip identities")

    selected = inference.condition
    results: list[DevelopmentClipResult] = []
    raw_samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
                (
                    0
                    if selected is ExperimentCondition.B0
                    else int(
                        np.count_nonzero((scores <= 1.0e-6) | (scores >= 1.0 - 1.0e-6))
                    )
                ),
            )
        )
        raw_samples[definition.identity] = (
            labels.astype(np.bool_, copy=True),
            scores.astype(np.float64, copy=True),
        )
    inference._assert_components_unchanged()
    return DevelopmentFusedAP(
        selected.value,
        tuple(results),
        inference.identity,
        _fusion_semantics(selected),
        raw_samples,
    )


def _record_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise EvaluationError(f"{name} must be a JSON object with string keys")
    return value


def _record_list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise EvaluationError(f"{name} must be a JSON array")
    return value


def _record_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise EvaluationError(f"{name} keys differ; missing={missing}, extra={extra}")


def _record_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationError(f"{name} must be finite")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _record_number(value, name)
    if not 0.0 <= result <= 1.0:
        raise EvaluationError(f"{name} must lie in [0,1]")
    return result


def _record_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvaluationError(f"{name} must be an integer >= {minimum}")
    return value


def _record_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{name} must be a non-empty string")
    return value


def _verified_record(
    value: object,
    *,
    name: str,
    keys: set[str],
    expected_format: str,
) -> dict[str, object]:
    record = _record_mapping(value, name)
    _record_keys(record, {*keys, "record_sha256"}, name)
    if record["format"] != expected_format:
        raise EvaluationError(f"{name} has the wrong format")
    supplied = _sha256(record["record_sha256"], f"{name} identity")
    unsigned = {key: record[key] for key in keys}
    if supplied != _json_sha256(unsigned):
        raise EvaluationError(f"{name} content hash does not match")
    plain = _plain_json(record)
    if not isinstance(plain, dict):  # pragma: no cover - guarded above
        raise AssertionError("record normalization did not return a dictionary")
    return plain


def _content_addressed(unsigned: Mapping[str, object]) -> dict[str, object]:
    if "record_sha256" in unsigned:
        raise EvaluationError("unsigned record cannot contain record_sha256")
    plain = _plain_json(unsigned)
    if not isinstance(plain, dict):
        raise EvaluationError("record must be a JSON object")
    return {**plain, "record_sha256": _json_sha256(plain)}


def _read_json_record(path: Path | str, name: str) -> tuple[Path, object]:
    requested = Path(path).expanduser()
    try:
        resolved = requested.resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{name} is unreadable") from error
    return resolved, value


def _protocol_root(protocol: object) -> Path:
    path = getattr(protocol, "path", None)
    if not isinstance(path, Path):
        raise EvaluationError("protocol does not expose its authoritative path")
    return path.resolve(strict=True).parent


def _resolve_protocol_record(protocol: object, value: object, name: str) -> Path:
    relative = _record_string(value, f"{name} path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise EvaluationError(f"{name} path must stay relative to the protocol root")
    root = _protocol_root(protocol)
    try:
        resolved = (root / path).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise EvaluationError(f"{name} path is unavailable") from error
    return resolved


def _implementation_manifest(protocol: object) -> dict[str, str]:
    root = _protocol_root(protocol)
    return {
        relative: _sha256_file(root / relative)
        for relative in FROZEN_IMPLEMENTATION_FILES
    }


def _dependency_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "torch": importlib.metadata.version("torch"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
    }


_EVALUATION_IDENTITY_KEYS = {
    "protocol_schema",
    "protocol_identity",
    "condition",
    "fusion_value",
    "model_class",
    "model_state_sha256",
    "stu_class",
    "stu_checkpoint_sha256",
    "stu_model_state_sha256",
    "stu_source_manifest_sha256",
    "calibration_sha256",
    "ray_mapping_sha256",
    "test_fixture",
}


def _evaluation_identity_record(value: object, name: str) -> EvaluationIdentity:
    record = _record_mapping(value, name)
    _record_keys(record, _EVALUATION_IDENTITY_KEYS, name)
    try:
        identity = EvaluationIdentity(**dict(record))
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"{name} is invalid") from error
    if identity.test_fixture:
        raise EvaluationError(f"{name} cannot be a test fixture")
    return identity


def _assert_frozen_evaluation_components(
    identity: EvaluationIdentity, protocol: object
) -> None:
    stu = _record_mapping(getattr(protocol, "stu", None), "protocol STU")
    render = _record_mapping(getattr(protocol, "render", None), "protocol render")
    ray_grid = _record_mapping(render.get("ray_grid"), "protocol ray grid")
    expected_model_class = (
        None if identity.condition == "B0" else "JointWindowPointTransformer"
    )
    if (
        identity.protocol_schema != 31
        or identity.protocol_identity != _protocol_identity(protocol)
        or identity.stu_class != "FrozenSTUPointEncoder"
        or identity.model_class != expected_model_class
        or identity.stu_checkpoint_sha256
        != _sha256(stu.get("checkpoint_sha256"), "protocol STU checkpoint")
        or identity.stu_model_state_sha256
        != _sha256(stu.get("model_state_tensor_sha256"), "protocol STU state")
        or identity.stu_source_manifest_sha256
        != _sha256(stu.get("source_manifest_sha256"), "protocol STU source")
        or identity.calibration_sha256
        != _sha256(render.get("calibration_sha256"), "protocol calibration")
        or identity.ray_mapping_sha256
        != _sha256(ray_grid.get("canonical_sha256"), "protocol ray mapping")
    ):
        raise EvaluationError(
            "evaluation identity differs from frozen protocol components"
        )
    if (identity.condition == "B0") != (identity.model_state_sha256 is None):
        raise EvaluationError("evaluation identity has the wrong model-state role")


def _artifact_destination(protocol: object, value: str, name: str) -> Path:
    relative = _record_string(value, f"{name} path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise EvaluationError(f"{name} path must stay relative to the protocol root")
    root = _protocol_root(protocol)
    destination = (root / path).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise EvaluationError(f"{name} path escapes the protocol root") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _npz_text(arrays: Mapping[str, np.ndarray], key: str, name: str) -> str:
    value = arrays[key]
    if value.shape != () or value.dtype.kind != "U":
        raise EvaluationError(f"{name}.{key} must be a Unicode scalar")
    return _record_string(value.item(), f"{name}.{key}")


def _npz_integer(arrays: Mapping[str, np.ndarray], key: str, name: str) -> int:
    value = arrays[key]
    if value.shape != () or value.dtype != np.dtype(np.int64):
        raise EvaluationError(f"{name}.{key} must be an int64 scalar")
    return int(value.item())


def _npz_float(arrays: Mapping[str, np.ndarray], key: str, name: str) -> float:
    value = arrays[key]
    if value.shape != () or value.dtype != np.dtype(np.float64):
        raise EvaluationError(f"{name}.{key} must be a float64 scalar")
    return _record_number(value.item(), f"{name}.{key}")


def _load_npz(
    path: Path,
    *,
    expected_keys: set[str],
    expected_sha256: object,
    name: str,
) -> tuple[dict[str, np.ndarray], str]:
    file_sha256 = _sha256(expected_sha256, f"{name} file identity")
    if _sha256_file(path) != file_sha256:
        raise EvaluationError(f"{name} file identity changed")
    try:
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != expected_keys:
                raise EvaluationError(f"{name} arrays have an invalid schema")
            arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    except (OSError, ValueError, TypeError) as error:
        raise EvaluationError(f"{name} cannot be safely loaded") from error
    return arrays, file_sha256


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        raise EvaluationError(f"refusing to replace evidence artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_file(path)


def _array_identity(name: str, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(name.encode("utf-8"))
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _identity_from_npz(
    arrays: Mapping[str, np.ndarray], key: str, name: str
) -> EvaluationIdentity:
    try:
        value = json.loads(_npz_text(arrays, key, name))
    except json.JSONDecodeError as error:
        raise EvaluationError(f"{name}.{key} is not valid JSON") from error
    return _evaluation_identity_record(value, f"{name} evaluation identity")


def _artifact_reference(protocol: object, value: object, name: str) -> tuple[Path, str]:
    reference = _record_mapping(value, name)
    _record_keys(reference, {"path", "file_sha256"}, name)
    path = _resolve_protocol_record(protocol, reference["path"], name)
    return path, _sha256(reference["file_sha256"], f"{name} file identity")


@dataclass(frozen=True, slots=True)
class NormalSafetyEvidence:
    """Raw normal labels and scores used to recompute a frozen-threshold FPR."""

    path: Path
    file_sha256: str
    evaluation_identity: EvaluationIdentity
    evaluation_set_identity: str
    threshold_rule_identity: str
    decision_threshold: float
    false_positive_rate: float

    @classmethod
    def load(
        cls,
        reference: object,
        *,
        protocol: object,
        expected_identity: EvaluationIdentity,
    ) -> NormalSafetyEvidence:
        path, file_sha256 = _artifact_reference(
            protocol, reference, "normal-safety evidence"
        )
        arrays, file_sha256 = _load_npz(
            path,
            expected_keys={
                "format",
                "protocol_identity",
                "evaluation_identity",
                "evaluation_set_identity",
                "threshold_rule_identity",
                "decision_threshold",
                "labels",
                "scores",
            },
            expected_sha256=file_sha256,
            name="normal-safety evidence",
        )
        if _npz_text(
            arrays, "format", "normal-safety evidence"
        ) != NORMAL_SAFETY_EVIDENCE_FORMAT or _npz_text(
            arrays, "protocol_identity", "normal-safety evidence"
        ) != _protocol_identity(protocol):
            raise EvaluationError("normal-safety evidence changed format or protocol")
        identity = _identity_from_npz(
            arrays, "evaluation_identity", "normal-safety evidence"
        )
        if identity != expected_identity:
            raise EvaluationError("normal-safety evidence changed its method")
        labels = arrays["labels"]
        scores = arrays["scores"]
        if (
            labels.dtype != np.dtype(np.bool_)
            or scores.dtype != np.dtype(np.float64)
            or labels.ndim != 1
            or scores.shape != labels.shape
            or labels.size == 0
            or bool(labels.any())
            or not np.isfinite(scores).all()
        ):
            raise EvaluationError("normal-safety raw evidence is invalid")
        if identity.condition != "B0" and np.any((scores < 0.0) | (scores > 1.0)):
            raise EvaluationError("learned normal-safety scores must lie in [0,1]")
        threshold = _npz_float(arrays, "decision_threshold", "normal-safety evidence")
        if identity.condition != "B0" and not 0.0 <= threshold <= 1.0:
            raise EvaluationError("learned decision threshold must lie in [0,1]")
        return cls(
            path,
            file_sha256,
            identity,
            _sha256(
                _npz_text(arrays, "evaluation_set_identity", "normal-safety evidence"),
                "normal-safety evaluation set",
            ),
            _sha256(
                _npz_text(arrays, "threshold_rule_identity", "normal-safety evidence"),
                "normal-safety threshold rule",
            ),
            threshold,
            float(np.count_nonzero(scores > threshold) / scores.size),
        )


def save_normal_safety_evidence(
    path: str,
    *,
    protocol: object,
    evaluation_identity: EvaluationIdentity,
    evaluation_set_identity: str,
    threshold_rule_identity: str,
    decision_threshold: float,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, str]:
    """Save raw normal-only evidence; the loader always recomputes its FPR."""

    _assert_frozen_evaluation_components(evaluation_identity, protocol)
    destination = _artifact_destination(protocol, path, "normal-safety evidence")
    arrays = {
        "format": np.asarray(NORMAL_SAFETY_EVIDENCE_FORMAT),
        "protocol_identity": np.asarray(_protocol_identity(protocol)),
        "evaluation_identity": np.asarray(
            json.dumps(
                evaluation_identity.to_dict(), sort_keys=True, separators=(",", ":")
            )
        ),
        "evaluation_set_identity": np.asarray(
            _sha256(evaluation_set_identity, "normal-safety evaluation set")
        ),
        "threshold_rule_identity": np.asarray(
            _sha256(threshold_rule_identity, "normal-safety threshold rule")
        ),
        "decision_threshold": np.asarray(
            _record_number(decision_threshold, "normal-safety threshold"),
            dtype=np.float64,
        ),
        "labels": np.asarray(labels, dtype=np.bool_),
        "scores": np.asarray(scores, dtype=np.float64),
    }
    reference = {"path": path, "file_sha256": _save_npz(destination, arrays)}
    NormalSafetyEvidence.load(
        reference, protocol=protocol, expected_identity=evaluation_identity
    )
    return reference


@dataclass(frozen=True, slots=True)
class FormalMetricEvidence:
    """Raw 24-clip labels/scores with independently recomputed clip AP."""

    path: Path
    file_sha256: str
    clip_ap: Mapping[str, float]

    @classmethod
    def load(
        cls,
        reference: object,
        *,
        protocol: object,
        expected_identity: EvaluationIdentity,
        expected_population_identity: str,
        expected_clip_identities: Sequence[str],
    ) -> FormalMetricEvidence:
        path, file_sha256 = _artifact_reference(
            protocol, reference, "formal metric evidence"
        )
        arrays, file_sha256 = _load_npz(
            path,
            expected_keys={
                "format",
                "protocol_identity",
                "evaluation_identity",
                "population_identity",
                "clip_identities",
                "offsets",
                "labels",
                "scores",
            },
            expected_sha256=file_sha256,
            name="formal metric evidence",
        )
        if (
            _npz_text(arrays, "format", "formal metric evidence")
            != FORMAL_METRIC_EVIDENCE_FORMAT
            or _npz_text(arrays, "protocol_identity", "formal metric evidence")
            != _protocol_identity(protocol)
            or _npz_text(arrays, "population_identity", "formal metric evidence")
            != expected_population_identity
            or _identity_from_npz(
                arrays, "evaluation_identity", "formal metric evidence"
            )
            != expected_identity
        ):
            raise EvaluationError("formal metric evidence changed its frozen inputs")
        identities = arrays["clip_identities"]
        offsets = arrays["offsets"]
        labels = arrays["labels"]
        scores = arrays["scores"]
        if (
            identities.dtype.kind != "U"
            or identities.ndim != 1
            or identities.tolist() != list(expected_clip_identities)
            or offsets.dtype != np.dtype(np.int64)
            or offsets.shape != (len(expected_clip_identities) + 1,)
            or offsets[0] != 0
            or offsets[-1] != labels.size
            or np.any(np.diff(offsets) <= 0)
            or labels.dtype != np.dtype(np.bool_)
            or scores.dtype != np.dtype(np.float64)
            or labels.ndim != 1
            or scores.shape != labels.shape
            or not np.isfinite(scores).all()
        ):
            raise EvaluationError("formal metric arrays are invalid")
        if expected_identity.condition != "B0" and np.any(
            (scores < 0.0) | (scores > 1.0)
        ):
            raise EvaluationError("learned formal scores must lie in [0,1]")
        clip_ap: dict[str, float] = {}
        for index, identity in enumerate(expected_clip_identities):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            clip_labels = labels[start:stop]
            if not bool(clip_labels.any()) or bool(clip_labels.all()):
                raise EvaluationError("formal clip evidence needs both point classes")
            clip_ap[identity] = (
                official_metrics(clip_labels, scores[start:stop])["AP"] / 100.0
            )
        return cls(path, file_sha256, clip_ap)


def save_formal_metric_evidence(
    path: str,
    *,
    protocol: object,
    development_evidence: DevelopmentFusedAP,
    population_identity: str,
) -> dict[str, str]:
    """Save the raw labels/scores from all 24 formal development clips."""

    if (
        type(development_evidence) is not DevelopmentFusedAP
        or development_evidence.raw_clip_samples is None
    ):
        raise EvaluationError("formal evidence requires raw development evaluation")
    evaluation_identity = development_evidence.evaluation_identity
    clip_samples = development_evidence.raw_clip_samples
    _assert_frozen_evaluation_components(evaluation_identity, protocol)
    identities = tuple(clip_samples)
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    offsets = [0]
    for identity in identities:
        raw_labels, raw_scores = clip_samples[identity]
        label = np.asarray(raw_labels, dtype=np.bool_).reshape(-1)
        score = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if label.shape != score.shape:
            raise EvaluationError("formal clip labels and scores are not aligned")
        labels.append(label)
        scores.append(score)
        offsets.append(offsets[-1] + label.size)
    destination = _artifact_destination(protocol, path, "formal metric evidence")
    arrays = {
        "format": np.asarray(FORMAL_METRIC_EVIDENCE_FORMAT),
        "protocol_identity": np.asarray(_protocol_identity(protocol)),
        "evaluation_identity": np.asarray(
            json.dumps(
                evaluation_identity.to_dict(), sort_keys=True, separators=(",", ":")
            )
        ),
        "population_identity": np.asarray(
            _sha256(population_identity, "formal development population")
        ),
        "clip_identities": np.asarray(identities),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "labels": np.concatenate(labels) if labels else np.empty(0, dtype=np.bool_),
        "scores": np.concatenate(scores) if scores else np.empty(0, dtype=np.float64),
    }
    reference = {"path": path, "file_sha256": _save_npz(destination, arrays)}
    return reference


def _protocol_identity(protocol: object) -> str:
    return _sha256(
        getattr(protocol, "scientific_identity", None),
        "protocol scientific identity",
    )


def _protocol_status(protocol: object) -> Mapping[str, object]:
    return _record_mapping(getattr(protocol, "status", None), "protocol status")


def _assert_record_lifecycle(
    protocol: object,
    *,
    milestone: str,
    status_field: str,
    record_sha256: str,
) -> None:
    states = tuple(
        _record_string(item, "protocol state")
        for item in _record_list(
            _plain_json(getattr(protocol, "state_machine", None)),
            "protocol state machine",
        )
    )
    node = _record_string(
        _protocol_status(protocol).get("current_node"), "protocol current node"
    )
    if (
        node not in states
        or milestone not in states
        or states.index(node) < states.index(milestone)
    ):
        raise EvaluationError(f"{milestone} record is unavailable at node {node}")
    bound = _protocol_status(protocol).get(status_field)
    if node == milestone:
        if bound is not None:
            raise EvaluationError(
                f"{milestone} current-node evidence must still be null"
            )
    elif bound != record_sha256:
        raise EvaluationError(
            f"{milestone} record is not the evidence bound by protocol status"
        )


def _protocol_gate(protocol: object, gate_name: str) -> Mapping[str, object]:
    gates = _record_mapping(
        getattr(protocol, "decision_gates", None), "protocol decision gates"
    )
    if gate_name not in gates:
        raise EvaluationError(f"protocol lacks decision gate {gate_name}")
    return _record_mapping(gates[gate_name], f"protocol {gate_name} gate")


def _frozen_gate_criteria(
    protocol: object, gate_name: str
) -> tuple[Mapping[str, object], str]:
    gate = _protocol_gate(protocol, gate_name)
    expected_status = "frozen_result_blind_in_R05"
    if gate.get("criteria_status") != expected_status:
        raise EvaluationError(f"{gate_name} criteria are not frozen")
    criteria = _record_mapping(gate.get("criteria"), f"{gate_name} criteria")
    identity = _sha256(gate.get("criteria_identity"), f"{gate_name} criteria")
    if identity != _json_sha256(criteria):
        raise EvaluationError(f"{gate_name} criteria identity does not match")
    interval = _record_mapping(
        criteria.get("confidence_interval"), f"{gate_name} confidence interval"
    )
    if interval.get("method") != PAIRED_STUDENT_T_METHOD:
        raise EvaluationError(f"{gate_name} must use {PAIRED_STUDENT_T_METHOD}")
    confidence = _record_number(
        interval.get("confidence_level"), f"{gate_name} confidence level"
    )
    if not 0.0 < confidence < 1.0:
        raise EvaluationError(f"{gate_name} confidence level must lie in (0,1)")
    return criteria, identity


@dataclass(frozen=True, slots=True)
class FormalEvaluationRecord:
    """A content-addressed formal result with AP recomputed from 24 clips."""

    payload: Mapping[str, object]
    record_sha256: str
    condition: str
    seed: int | None
    evaluation_identity: EvaluationIdentity
    development_ap: float
    normal_safety: Mapping[str, float]
    normal_safety_identity: str
    normal_safety_threshold_rule_identity: str
    decision_threshold: float
    training_result_identity: str | None
    initial_model_state_sha256: str | None

    @classmethod
    def from_mapping(cls, value: object, *, protocol: object) -> FormalEvaluationRecord:
        keys = {
            "format",
            "protocol_identity",
            "r05_freeze_identity",
            "condition",
            "seed",
            "evaluation_identity",
            "training_result_path",
            "training_result_identity",
            "initial_model_state_sha256",
            "development",
            "metric_evidence",
            "normal_safety",
            "normal_safety_evidence",
        }
        payload = _verified_record(
            value,
            name="formal evaluation record",
            keys=keys,
            expected_format=FORMAL_EVALUATION_FORMAT,
        )
        protocol_identity = _protocol_identity(protocol)
        if payload["protocol_identity"] != protocol_identity:
            raise EvaluationError("formal result belongs to another protocol")
        status = _protocol_status(protocol)
        r05_identity = _sha256(
            status.get("r05_freeze_identity"), "protocol R05 freeze identity"
        )
        if payload["r05_freeze_identity"] != r05_identity:
            raise EvaluationError("formal result belongs to another R05 freeze")

        condition = _condition(payload["condition"])
        seed = payload["seed"]
        training_path = payload["training_result_path"]
        training_identity = payload["training_result_identity"]
        initial_identity = payload["initial_model_state_sha256"]
        formal = _record_mapping(
            _record_mapping(getattr(protocol, "training", None), "training").get(
                "formal"
            ),
            "formal training protocol",
        )
        formal_seeds = tuple(
            _record_integer(item, "formal seed")
            for item in _record_list(_plain_json(formal.get("seeds")), "formal seeds")
        )
        if condition is ExperimentCondition.B0:
            if (
                seed is not None
                or training_path is not None
                or training_identity is not None
                or initial_identity is not None
            ):
                raise EvaluationError(
                    "B0 formal reference cannot carry a seed or training identity"
                )
        else:
            if type(seed) is not int or seed not in formal_seeds:
                raise EvaluationError("formal result seed is outside the frozen set")
            training_identity = _sha256(
                training_identity, "formal training-result identity"
            )
            initial_identity = _sha256(
                initial_identity, "formal initial-model identity"
            )
            training_result_path = _resolve_protocol_record(
                protocol, training_path, "formal training result"
            )

        identity = _evaluation_identity_record(
            payload["evaluation_identity"], "formal evaluation identity"
        )
        if (
            identity.protocol_identity != protocol_identity
            or identity.condition != condition.value
        ):
            raise EvaluationError(
                "formal evaluation identity changed method or protocol"
            )
        _assert_frozen_evaluation_components(identity, protocol)

        try:
            from .protocol import load_development_worlds
        except ImportError:  # pragma: no cover - direct module execution
            from protocol import load_development_worlds

        try:
            development_worlds = load_development_worlds(
                _protocol_root(protocol) / "dev.json", protocol=protocol
            )
        except (OSError, ValueError) as error:
            raise EvaluationError(
                "formal result cannot load the frozen development population"
            ) from error
        expected_clip_identities = tuple(
            str(item.identity) for item in development_worlds.clips
        )
        if len(expected_clip_identities) != 24:
            raise EvaluationError(
                "formal result requires the frozen 24-clip population"
            )

        development = _record_mapping(
            payload["development"], "formal development result"
        )
        _record_keys(
            development,
            {"population_identity", "clips", "macro_AP"},
            "formal development result",
        )
        population_identity = _sha256(
            development["population_identity"], "formal development population"
        )
        expected_population = _sha256(
            formal.get("development_population_identity"),
            "protocol formal development population",
        )
        if population_identity != expected_population:
            raise EvaluationError("formal result uses another development population")
        clips = _record_list(development["clips"], "formal development clips")
        if len(clips) != 24:
            raise EvaluationError(
                "formal result must contain exactly 24 clip AP values"
            )
        clip_identities: set[str] = set()
        clip_values: list[float] = []
        for index, item in enumerate(clips):
            clip = _record_mapping(item, f"formal development clip {index}")
            _record_keys(
                clip, {"clip_identity", "AP"}, f"formal development clip {index}"
            )
            clip_identity = _sha256(
                clip["clip_identity"], f"formal development clip {index} identity"
            )
            if clip_identity in clip_identities:
                raise EvaluationError("formal development clip identities repeat")
            clip_identities.add(clip_identity)
            clip_values.append(_unit_interval(clip["AP"], f"clip {index} AP"))
        if (
            tuple(
                str(_record_mapping(item, "formal development clip")["clip_identity"])
                for item in clips
            )
            != expected_clip_identities
        ):
            raise EvaluationError(
                "formal result clip identities or order differ from dev.json"
            )
        macro_ap = math.fsum(clip_values) / len(clip_values)
        reported_ap = _unit_interval(development["macro_AP"], "formal macro AP")
        if not math.isclose(macro_ap, reported_ap, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise EvaluationError(
                "formal macro AP disagrees with the 24 clip AP values"
            )
        metric_evidence = FormalMetricEvidence.load(
            payload["metric_evidence"],
            protocol=protocol,
            expected_identity=identity,
            expected_population_identity=population_identity,
            expected_clip_identities=expected_clip_identities,
        )
        if dict(metric_evidence.clip_ap) != {
            str(
                _record_mapping(item, "formal development clip")["clip_identity"]
            ): _unit_interval(
                _record_mapping(item, "formal development clip")["AP"],
                "formal development clip AP",
            )
            for item in clips
        }:
            raise EvaluationError("formal clip AP does not reproduce from raw evidence")

        if condition is not ExperimentCondition.B0:
            try:
                from .train import TrainingError, TrainingRunRecord
            except ImportError:  # pragma: no cover - direct module execution
                from train import TrainingError, TrainingRunRecord

            try:
                training_run = TrainingRunRecord.load(
                    training_result_path, protocol=protocol
                )
            except (OSError, ValueError, TrainingError) as error:
                raise EvaluationError(
                    "formal result cannot validate its training run"
                ) from error
            run_payload = _record_mapping(
                training_run.payload, "formal training-run payload"
            )
            best_development = _record_mapping(
                run_payload.get("best_development"), "formal best development"
            )
            run_clips = _record_list(
                _plain_json(best_development.get("clips")),
                "formal training best-development clips",
            )
            run_clip_values = {
                str(
                    _record_mapping(item, "training development clip")["clip_identity"]
                ): _unit_interval(
                    _record_mapping(item, "training development clip")[
                        "fused_point_ap"
                    ],
                    "training development clip AP",
                )
                for item in run_clips
            }
            supplied_clip_values = {
                str(
                    _record_mapping(item, "formal development clip")["clip_identity"]
                ): _unit_interval(
                    _record_mapping(item, "formal development clip")["AP"],
                    "formal development clip AP",
                )
                for item in clips
            }
            best_identity = _evaluation_identity_record(
                best_development.get("evaluation_identity"),
                "training best-development identity",
            )
            if (
                training_run.mode.value != "formal"
                or training_run.condition is not condition
                or training_run.seed != seed
                or training_run.record_sha256 != training_identity
                or run_payload.get("r05_freeze_identity") != r05_identity
                or run_payload.get("initial_model_state_sha256") != initial_identity
                or run_payload.get("best_model_state_sha256")
                != identity.model_state_sha256
                or run_payload.get("development_population_identity")
                != population_identity
                or best_identity != identity
                or run_clip_values != supplied_clip_values
                or not math.isclose(
                    _unit_interval(
                        best_development.get("macro_fused_point_ap"),
                        "training best macro AP",
                    ),
                    macro_ap,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ):
                raise EvaluationError(
                    "formal evaluation differs from its validated best training run"
                )

        safety = _record_mapping(payload["normal_safety"], "formal normal safety")
        _record_keys(
            safety,
            {
                "evaluation_set_identity",
                "threshold_rule_identity",
                "orientation",
                "statistics",
            },
            "formal normal safety",
        )
        safety_identity = _sha256(
            safety["evaluation_set_identity"], "normal-safety population identity"
        )
        threshold_rule_identity = _sha256(
            safety["threshold_rule_identity"],
            "normal-safety threshold-rule identity",
        )
        if safety["orientation"] != "higher_is_worse":
            raise EvaluationError("normal-safety statistics must use higher-is-worse")
        raw_statistics = _record_mapping(
            safety["statistics"], "normal-safety statistics"
        )
        if not raw_statistics:
            raise EvaluationError("normal-safety statistics cannot be empty")
        statistics = {
            name: _unit_interval(value, f"normal-safety statistic {name}")
            for name, value in raw_statistics.items()
        }
        safety_evidence = NormalSafetyEvidence.load(
            payload["normal_safety_evidence"],
            protocol=protocol,
            expected_identity=identity,
        )
        expected_statistics = {
            "normal_false_positive_rate_at_frozen_threshold_unit_interval": (
                safety_evidence.false_positive_rate
            )
        }
        if (
            safety_identity != safety_evidence.evaluation_set_identity
            or threshold_rule_identity != safety_evidence.threshold_rule_identity
            or statistics != expected_statistics
        ):
            raise EvaluationError(
                "formal normal safety does not reproduce from raw evidence"
            )
        return cls(
            payload,
            _sha256(payload["record_sha256"], "formal evaluation record"),
            condition.value,
            seed if type(seed) is int else None,
            identity,
            macro_ap,
            statistics,
            safety_identity,
            threshold_rule_identity,
            safety_evidence.decision_threshold,
            training_identity if isinstance(training_identity, str) else None,
            initial_identity if isinstance(initial_identity, str) else None,
        )

    @classmethod
    def load(cls, path: Path | str, *, protocol: object) -> FormalEvaluationRecord:
        _, value = _read_json_record(path, "formal evaluation record")
        return cls.from_mapping(value, protocol=protocol)


def make_formal_evaluation_record(
    *,
    protocol: object,
    condition: ExperimentCondition | str,
    seed: int | None,
    evaluation_identity: EvaluationIdentity,
    training_result_path: str | None,
    training_result_identity: str | None,
    initial_model_state_sha256: str | None,
    development_population_identity: str,
    metric_evidence: Mapping[str, str],
    normal_safety_evidence: Mapping[str, str],
) -> dict[str, object]:
    """Build and validate one formal evaluation record without reading sealed data."""

    selected = _condition(condition)
    try:
        from .protocol import load_development_worlds
    except ImportError:  # pragma: no cover - direct module execution
        from protocol import load_development_worlds

    development = load_development_worlds(
        _protocol_root(protocol) / "dev.json", protocol=protocol
    )
    expected_clips = tuple(str(item.identity) for item in development.clips)
    evidence = FormalMetricEvidence.load(
        metric_evidence,
        protocol=protocol,
        expected_identity=evaluation_identity,
        expected_population_identity=development_population_identity,
        expected_clip_identities=expected_clips,
    )
    safety_evidence = NormalSafetyEvidence.load(
        normal_safety_evidence,
        protocol=protocol,
        expected_identity=evaluation_identity,
    )
    clip_ap = evidence.clip_ap
    clips = [
        {"clip_identity": identity, "AP": value} for identity, value in clip_ap.items()
    ]
    if not clips:
        raise EvaluationError("formal evaluation requires clip AP values")
    unsigned = {
        "format": FORMAL_EVALUATION_FORMAT,
        "protocol_identity": _protocol_identity(protocol),
        "r05_freeze_identity": _protocol_status(protocol).get("r05_freeze_identity"),
        "condition": selected.value,
        "seed": seed,
        "evaluation_identity": evaluation_identity.to_dict(),
        "training_result_path": training_result_path,
        "training_result_identity": training_result_identity,
        "initial_model_state_sha256": initial_model_state_sha256,
        "development": {
            "population_identity": development_population_identity,
            "clips": clips,
            "macro_AP": math.fsum(float(value) for value in clip_ap.values())
            / len(clip_ap),
        },
        "metric_evidence": dict(metric_evidence),
        "normal_safety": {
            "evaluation_set_identity": safety_evidence.evaluation_set_identity,
            "threshold_rule_identity": safety_evidence.threshold_rule_identity,
            "orientation": "higher_is_worse",
            "statistics": {
                "normal_false_positive_rate_at_frozen_threshold_unit_interval": (
                    safety_evidence.false_positive_rate
                )
            },
        },
        "normal_safety_evidence": dict(normal_safety_evidence),
    }
    record = _content_addressed(unsigned)
    FormalEvaluationRecord.from_mapping(record, protocol=protocol)
    return record


def _paired_student_t(
    differences: Sequence[float], confidence_level: float
) -> tuple[float, float, float]:
    values = tuple(_record_number(item, "paired difference") for item in differences)
    if len(values) < 2:
        raise EvaluationError("paired Student t interval needs at least two units")
    mean = math.fsum(values) / len(values)
    squared = math.fsum((value - mean) ** 2 for value in values)
    standard_error = math.sqrt(squared / (len(values) - 1) / len(values))
    critical = float(
        student_t_distribution.ppf(
            0.5 + 0.5 * confidence_level,
            df=len(values) - 1,
        )
    )
    if not math.isfinite(critical):
        raise EvaluationError("paired Student t critical value is not finite")
    half_width = critical * standard_error
    return mean, mean - half_width, mean + half_width


def _result_common_identity(record: FormalEvaluationRecord) -> tuple[object, ...]:
    identity = record.evaluation_identity
    return (
        identity.protocol_identity,
        identity.stu_class,
        identity.stu_checkpoint_sha256,
        identity.stu_model_state_sha256,
        identity.stu_source_manifest_sha256,
        identity.calibration_sha256,
        identity.ray_mapping_sha256,
    )


def _indexed_formal_results(
    values: object, *, protocol: object
) -> dict[tuple[str, int | None], FormalEvaluationRecord]:
    records = _record_list(_plain_json(values), "formal gate results")
    indexed: dict[tuple[str, int | None], FormalEvaluationRecord] = {}
    for value in records:
        record = FormalEvaluationRecord.from_mapping(value, protocol=protocol)
        key = (record.condition, record.seed)
        if key in indexed:
            raise EvaluationError("formal gate repeats a condition/seed result")
        indexed[key] = record
    if not indexed:
        raise EvaluationError("formal gate has no results")
    first = next(iter(indexed.values()))
    for record in indexed.values():
        if (
            _result_common_identity(record) != _result_common_identity(first)
            or record.normal_safety_identity != first.normal_safety_identity
            or record.normal_safety_threshold_rule_identity
            != first.normal_safety_threshold_rule_identity
        ):
            raise EvaluationError(
                "formal gate results do not share method-independent inputs"
            )
    return indexed


def _comparison_result(
    *,
    unit_name: str,
    pairs: Sequence[tuple[int, float, float]],
    confidence_level: float,
    threshold: Mapping[str, object],
    v01: bool = False,
) -> dict[str, object]:
    differences = [candidate - reference for _, candidate, reference in pairs]
    mean, lower, upper = _paired_student_t(differences, confidence_level)
    mean_key = "minimum_mean_AP_difference" if v01 else "minimum_mean_difference"
    count_key = (
        "minimum_positive_sequence_count" if v01 else "minimum_positive_seed_count"
    )
    minimum_mean = _record_number(threshold.get(mean_key), mean_key)
    minimum_lower = _record_number(
        threshold.get("minimum_confidence_interval_lower_bound"),
        "minimum confidence-interval lower bound",
    )
    minimum_positive = _record_integer(threshold.get(count_key), count_key, minimum=0)
    positive = sum(value > 0.0 for value in differences)
    return {
        "paired_values": [
            {
                unit_name: unit,
                "candidate_AP": candidate,
                "reference_AP": reference,
                "difference": candidate - reference,
            }
            for unit, candidate, reference in pairs
        ],
        "mean_difference": mean,
        "confidence_interval": {
            "method": PAIRED_STUDENT_T_METHOD,
            "confidence_level": confidence_level,
            "lower_bound": lower,
            "upper_bound": upper,
        },
        "positive_count": positive,
        "passed": (
            mean >= minimum_mean
            and lower >= minimum_lower
            and positive >= minimum_positive
        ),
    }


def _normal_safety_result(
    *,
    unit_name: str,
    comparison_pairs: Mapping[
        str,
        Sequence[tuple[int, FormalEvaluationRecord, FormalEvaluationRecord]],
    ],
    criteria: Mapping[str, object],
) -> dict[str, object]:
    safety = _record_mapping(criteria.get("normal_safety"), "normal-safety criteria")
    evaluation_identity = _sha256(
        safety.get("evaluation_set_identity"), "normal-safety criteria population"
    )
    threshold_rule_identity = _sha256(
        safety.get("threshold_rule_identity"),
        "normal-safety criteria threshold rule",
    )
    statistics = tuple(
        _record_string(item, "normal-safety statistic name")
        for item in _record_list(
            _plain_json(safety.get("statistics")), "normal-safety statistic names"
        )
    )
    if not statistics or len(set(statistics)) != len(statistics):
        raise EvaluationError("normal-safety statistic names must be unique")
    maximum_allowed = _unit_interval(
        safety.get("maximum_allowed_signed_worsening"),
        "maximum allowed normal-safety worsening",
    )
    comparisons: dict[str, object] = {}
    all_passed = True
    for comparison, pairs in comparison_pairs.items():
        comparison_statistics: dict[str, object] = {}
        for statistic in statistics:
            values: list[dict[str, object]] = []
            worsening: list[float] = []
            for unit, candidate, reference in pairs:
                if (
                    candidate.normal_safety_identity != evaluation_identity
                    or reference.normal_safety_identity != evaluation_identity
                    or candidate.normal_safety_threshold_rule_identity
                    != threshold_rule_identity
                    or reference.normal_safety_threshold_rule_identity
                    != threshold_rule_identity
                    or statistic not in candidate.normal_safety
                    or statistic not in reference.normal_safety
                ):
                    raise EvaluationError(
                        "formal result does not match frozen normal-safety criteria"
                    )
                difference = (
                    candidate.normal_safety[statistic]
                    - reference.normal_safety[statistic]
                )
                worsening.append(difference)
                values.append(
                    {
                        unit_name: unit,
                        "candidate": candidate.normal_safety[statistic],
                        "reference": reference.normal_safety[statistic],
                        "signed_worsening": difference,
                    }
                )
            maximum = max(worsening)
            passed = maximum <= maximum_allowed
            all_passed &= passed
            comparison_statistics[statistic] = {
                "paired_values": values,
                "maximum_signed_worsening": maximum,
                "passed": passed,
            }
        comparisons[comparison] = comparison_statistics
    return {
        "evaluation_set_identity": evaluation_identity,
        "threshold_rule_identity": threshold_rule_identity,
        "orientation": "higher_is_worse",
        "comparisons": comparisons,
        "passed": all_passed,
    }


@dataclass(frozen=True, slots=True)
class FormalGateVerdictRecord:
    """A G2/G3 verdict whose statistics are reproduced from embedded results."""

    payload: Mapping[str, object]
    record_sha256: str
    gate: str
    decision: str
    formal_results: Mapping[tuple[str, int | None], FormalEvaluationRecord]
    g2_verdict: FormalGateVerdictRecord | None

    @classmethod
    def from_mapping(
        cls, value: object, *, protocol: object
    ) -> FormalGateVerdictRecord:
        keys = {
            "format",
            "gate",
            "protocol_identity",
            "r05_freeze_identity",
            "criteria_identity",
            "g2_verdict",
            "formal_results",
            "adjudication",
            "decision",
        }
        payload = _verified_record(
            value,
            name="formal gate verdict",
            keys=keys,
            expected_format=FORMAL_GATE_VERDICT_FORMAT,
        )
        gate = payload["gate"]
        if gate not in {"G2", "G3"}:
            raise EvaluationError("formal gate verdict must identify G2 or G3")
        if payload["protocol_identity"] != _protocol_identity(protocol):
            raise EvaluationError("formal gate verdict belongs to another protocol")
        r05_identity = _sha256(
            _protocol_status(protocol).get("r05_freeze_identity"),
            "protocol R05 freeze identity",
        )
        if payload["r05_freeze_identity"] != r05_identity:
            raise EvaluationError("formal gate verdict belongs to another R05 freeze")
        criteria, criteria_identity = _frozen_gate_criteria(protocol, str(gate))
        if payload["criteria_identity"] != criteria_identity:
            raise EvaluationError("formal gate verdict uses different criteria")

        g2_verdict: FormalGateVerdictRecord | None = None
        if gate == "G2":
            if payload["g2_verdict"] is not None:
                raise EvaluationError("G2 cannot contain a prerequisite G2 verdict")
        else:
            g2_verdict = cls.from_mapping(payload["g2_verdict"], protocol=protocol)
            if g2_verdict.gate != "G2" or g2_verdict.decision != "pass":
                raise EvaluationError("G3 requires a reproduced passed G2 verdict")
            if (
                _protocol_status(protocol).get("g2_verdict_identity")
                != g2_verdict.record_sha256
            ):
                raise EvaluationError("G3 is not bound to the protocol G2 decision")

        indexed = _indexed_formal_results(payload["formal_results"], protocol=protocol)
        seeds = (0, 1, 2)
        if gate == "G2":
            expected = {("B0", None), *(("B1", seed) for seed in seeds)}
        else:
            expected = {
                *(
                    (condition, seed)
                    for condition in ("B1", "B2", "B3")
                    for seed in seeds
                )
            }
        if set(indexed) != expected:
            raise EvaluationError(f"{gate} formal result population is incomplete")

        if gate == "G3":
            assert g2_verdict is not None
            for seed in seeds:
                if (
                    indexed[("B1", seed)].record_sha256
                    != g2_verdict.formal_results[("B1", seed)].record_sha256
                ):
                    raise EvaluationError("G3 changed a B1 result after G2")
                initial = {
                    indexed[(condition, seed)].initial_model_state_sha256
                    for condition in ("B1", "B2", "B3")
                }
                if len(initial) != 1:
                    raise EvaluationError(
                        "paired B1/B2/B3 runs do not share initialization"
                    )

        interval = _record_mapping(
            criteria.get("confidence_interval"), f"{gate} confidence interval"
        )
        confidence = _record_number(
            interval.get("confidence_level"), f"{gate} confidence level"
        )
        thresholds = _record_mapping(
            criteria.get("comparison_thresholds"), f"{gate} thresholds"
        )
        if gate == "G2":
            baseline = indexed[("B0", None)]
            comparison_records = {
                "B1_vs_B0": [(seed, indexed[("B1", seed)], baseline) for seed in seeds]
            }
        else:
            comparison_records = {
                "B3_vs_B1": [
                    (seed, indexed[("B3", seed)], indexed[("B1", seed)])
                    for seed in seeds
                ],
                "B3_vs_B2": [
                    (seed, indexed[("B3", seed)], indexed[("B2", seed)])
                    for seed in seeds
                ],
            }
        comparisons: dict[str, object] = {}
        for name, pairs in comparison_records.items():
            threshold = _record_mapping(
                thresholds.get(name), f"{gate} {name} threshold"
            )
            comparisons[name] = _comparison_result(
                unit_name="seed",
                pairs=[
                    (seed, candidate.development_ap, reference.development_ap)
                    for seed, candidate, reference in pairs
                ],
                confidence_level=confidence,
                threshold=threshold,
            )
        normal_safety = _normal_safety_result(
            unit_name="seed",
            comparison_pairs=comparison_records,
            criteria=criteria,
        )
        expected_adjudication = {
            "comparisons": comparisons,
            "normal_safety": normal_safety,
        }
        if _plain_json(payload["adjudication"]) != expected_adjudication:
            raise EvaluationError("formal gate adjudication does not reproduce")
        passed = all(
            bool(_record_mapping(item, "comparison result")["passed"])
            for item in comparisons.values()
        ) and bool(normal_safety["passed"])
        expected_decision = "pass" if passed else "fail"
        if payload["decision"] != expected_decision:
            raise EvaluationError("formal gate decision does not reproduce")
        record_sha256 = _sha256(payload["record_sha256"], "formal gate verdict")
        _assert_record_lifecycle(
            protocol,
            milestone=str(gate),
            status_field=f"{str(gate).lower()}_verdict_identity",
            record_sha256=record_sha256,
        )
        return cls(
            payload,
            record_sha256,
            str(gate),
            expected_decision,
            indexed,
            g2_verdict,
        )

    @classmethod
    def load(cls, path: Path | str, *, protocol: object) -> FormalGateVerdictRecord:
        _, value = _read_json_record(path, "formal gate verdict")
        return cls.from_mapping(value, protocol=protocol)


def adjudicate_formal_gate(
    *,
    protocol: object,
    gate: str,
    formal_results: Iterable[FormalEvaluationRecord | Mapping[str, object]],
    g2_verdict: FormalGateVerdictRecord | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create a G2/G3 record; its loader independently repeats this decision."""

    if gate not in {"G2", "G3"}:
        raise EvaluationError("formal gate must be G2 or G3")
    if _protocol_status(protocol).get("current_node") != gate:
        raise EvaluationError(f"{gate} can be adjudicated only at its protocol node")
    values = [
        dict(item.payload) if isinstance(item, FormalEvaluationRecord) else dict(item)
        for item in formal_results
    ]
    prerequisite = (
        None
        if g2_verdict is None
        else dict(g2_verdict.payload)
        if isinstance(g2_verdict, FormalGateVerdictRecord)
        else dict(g2_verdict)
    )
    criteria, criteria_identity = _frozen_gate_criteria(protocol, gate)
    provisional = {
        "format": FORMAL_GATE_VERDICT_FORMAT,
        "gate": gate,
        "protocol_identity": _protocol_identity(protocol),
        "r05_freeze_identity": _protocol_status(protocol).get("r05_freeze_identity"),
        "criteria_identity": criteria_identity,
        "g2_verdict": prerequisite,
        "formal_results": values,
        "adjudication": {},
        "decision": "fail",
    }
    # Use a temporary correctly hashed shell to reuse the sole adjudication logic.
    shell = _content_addressed(provisional)
    try:
        FormalGateVerdictRecord.from_mapping(shell, protocol=protocol)
    except EvaluationError as error:
        if "adjudication does not reproduce" not in str(error):
            raise

    indexed = _indexed_formal_results(values, protocol=protocol)
    seeds = (0, 1, 2)
    interval = _record_mapping(criteria["confidence_interval"], "confidence interval")
    confidence = _record_number(interval["confidence_level"], "confidence level")
    thresholds = _record_mapping(criteria["comparison_thresholds"], "thresholds")
    if gate == "G2":
        baseline = indexed[("B0", None)]
        comparison_records = {
            "B1_vs_B0": [(seed, indexed[("B1", seed)], baseline) for seed in seeds]
        }
    else:
        comparison_records = {
            "B3_vs_B1": [
                (seed, indexed[("B3", seed)], indexed[("B1", seed)]) for seed in seeds
            ],
            "B3_vs_B2": [
                (seed, indexed[("B3", seed)], indexed[("B2", seed)]) for seed in seeds
            ],
        }
    comparisons = {
        name: _comparison_result(
            unit_name="seed",
            pairs=[
                (seed, candidate.development_ap, reference.development_ap)
                for seed, candidate, reference in pairs
            ],
            confidence_level=confidence,
            threshold=_record_mapping(thresholds[name], f"{name} threshold"),
        )
        for name, pairs in comparison_records.items()
    }
    normal_safety = _normal_safety_result(
        unit_name="seed",
        comparison_pairs=comparison_records,
        criteria=criteria,
    )
    provisional["adjudication"] = {
        "comparisons": comparisons,
        "normal_safety": normal_safety,
    }
    provisional["decision"] = (
        "pass"
        if all(bool(item["passed"]) for item in comparisons.values())
        and bool(normal_safety["passed"])
        else "fail"
    )
    record = _content_addressed(provisional)
    FormalGateVerdictRecord.from_mapping(record, protocol=protocol)
    return record


def _diagnostic_group(value: object, name: str) -> dict[str, float]:
    group = _record_mapping(value, name)
    if not group:
        raise EvaluationError(f"{name} cannot be empty")
    return {key: _record_number(item, f"{name}.{key}") for key, item in group.items()}


def _selected_formal_records(
    verdict: FormalGateVerdictRecord,
) -> dict[str, FormalEvaluationRecord]:
    if verdict.gate != "G3" or verdict.decision != "pass" or verdict.g2_verdict is None:
        raise EvaluationError("method selection requires a passed G3 verdict")
    g2 = verdict.g2_verdict
    return {
        "B0_reference": g2.formal_results[("B0", None)],
        "B1_reference": verdict.formal_results[("B1", 0)],
        "frozen_final": verdict.formal_results[("B3", 0)],
    }


def _selected_formal_methods(
    verdict: FormalGateVerdictRecord,
) -> tuple[dict[str, EvaluationIdentity], dict[str, str]]:
    selected = _selected_formal_records(verdict)
    methods = {role: result.evaluation_identity for role, result in selected.items()}
    identities = {role: result.record_sha256 for role, result in selected.items()}
    return methods, identities


def _held_out_recipe_identity(protocol: object) -> str:
    evaluation = _record_mapping(
        getattr(protocol, "evaluation_document", None), "protocol evaluation"
    )
    render = _record_mapping(getattr(protocol, "render", None), "protocol render")
    proxies = _record_mapping(render.get("anomaly_proxies"), "anomaly proxies")
    return _json_sha256(
        {
            "evaluation": _record_mapping(
                evaluation.get("held_out_synthetic_shift"),
                "held-out evaluation recipe",
            ),
            "renderer": _record_mapping(
                proxies.get("held_out_recipe"), "held-out renderer recipe"
            ),
        }
    )


def _validated_s01_clip_manifest(
    value: object, *, protocol: object, plan: Mapping[str, object]
) -> tuple[str, str, dict[str, dict[str, float]]]:
    """Recompute clip/window identities and descriptor diagnostics."""

    try:
        from .render import (
            HeldOutTorusShape,
            WindowEntityDescriptor,
            WorldGenerationReport,
            WorldSpec,
        )
    except ImportError:  # pragma: no cover - direct module execution
        from render import (
            HeldOutTorusShape,
            WindowEntityDescriptor,
            WorldGenerationReport,
            WorldSpec,
        )

    manifest = _record_mapping(value, "S01 clip manifest")
    _record_keys(
        manifest,
        {
            "format",
            "identity",
            "world_identity",
            "clip_start",
            "frame_ids",
            "renderer_identity",
            "mechanism",
            "source_observation_identities",
            "world",
            "report",
            "windows",
        },
        "S01 clip manifest",
    )
    frames = tuple(
        _record_integer(item, "S01 source frame")
        for item in _record_list(manifest["frame_ids"], "S01 source frames")
    )
    expected_frames = tuple(int(item) for item in plan["source_frames"])  # type: ignore[arg-type]
    observations = tuple(
        _sha256(item, "S01 source observation")
        for item in _record_list(
            manifest["source_observation_identities"], "S01 observations"
        )
    )
    if (
        manifest["format"] != "ajae-development-clip-world-v1"
        or frames != expected_frames
        or manifest["clip_start"] != plan["clip_start"]
        or manifest["renderer_identity"] != getattr(protocol, "renderer_identity")
        or manifest["mechanism"] != "torus_SDF"
        or len(observations) != 9
        or len(set(observations)) != 9
    ):
        raise EvaluationError("S01 clip manifest differs from its frozen plan")
    try:
        world = WorldSpec.from_dict(_record_mapping(manifest["world"], "S01 world"))
        report = WorldGenerationReport.from_dict(
            _record_mapping(manifest["report"], "S01 report")
        )
    except (TypeError, ValueError, KeyError) as error:
        raise EvaluationError("S01 world or report is invalid") from error
    root_seed = int(plan["root_seed"])
    stride = int(plan["observation_attempt_stride"])
    attempts = int(plan["maximum_observation_attempts"])
    delta = world.seed - root_seed
    torus = tuple(
        item
        for item in world.objects
        if item.label == "anomaly-proxy" and isinstance(item.shape, HeldOutTorusShape)
    )
    if (
        world.identity != manifest["world_identity"]
        or world.source_sequence_id != 201
        or len(world.objects) != 1
        or len(torus) != 1
        or report.world_seed != world.seed
        or report.world_type != "anomaly_only"
        or report.normal_count != 0
        or report.anomaly_count != 1
        or delta < 0
        or delta % stride != 0
        or delta // stride >= attempts
    ):
        raise EvaluationError("S01 world does not reproduce the held-out recipe")
    expected_clip = _json_sha256(
        {
            "format": "ajae-development-clip-world-v1",
            "world_identity": world.identity,
            "clip_start": int(plan["clip_start"]),
            "frame_ids": frames,
            "renderer_identity": getattr(protocol, "renderer_identity"),
            "mechanism": "torus_SDF",
            "source_observation_identities": observations,
        }
    )
    if manifest["identity"] != expected_clip:
        raise EvaluationError("S01 clip identity does not reproduce")
    windows = _record_list(manifest["windows"], "S01 windows")
    if len(windows) != 5:
        raise EvaluationError("S01 clip must contain five overlapping windows")
    descriptors: list[WindowEntityDescriptor] = []
    for index, raw_window in enumerate(windows):
        window = _record_mapping(raw_window, f"S01 window {index}")
        _record_keys(
            window,
            {
                "identity",
                "window_start",
                "frame_ids",
                "source_observation_identities",
                "descriptors",
            },
            f"S01 window {index}",
        )
        window_frames = frames[index : index + 5]
        window_observations = observations[index : index + 5]
        expected_window = _json_sha256(
            {
                "format": "ajae-window-world-v1",
                "world_identity": world.identity,
                "partition": "train",
                "sequence_id": 201,
                "window_start": window_frames[0],
                "frame_ids": window_frames,
                "renderer_identity": getattr(protocol, "renderer_identity"),
                "source_observation_identities": window_observations,
            }
        )
        if (
            window["identity"] != expected_window
            or window["window_start"] != window_frames[0]
            or tuple(_record_list(window["frame_ids"], "S01 window frames"))
            != window_frames
            or tuple(
                _record_list(
                    window["source_observation_identities"],
                    "S01 window observations",
                )
            )
            != window_observations
        ):
            raise EvaluationError("S01 window identity does not reproduce")
        values = _record_list(window["descriptors"], "S01 window descriptors")
        if len(values) != 1:
            raise EvaluationError("S01 torus window must have one descriptor")
        try:
            descriptor = WindowEntityDescriptor(
                **dict(_record_mapping(values[0], "S01 descriptor"))
            )
        except (TypeError, ValueError) as error:
            raise EvaluationError("S01 descriptor is invalid") from error
        if descriptor.object_id != 1 or descriptor.label != "anomaly-proxy":
            raise EvaluationError("S01 descriptor does not identify the torus")
        descriptors.append(descriptor)
    density = [float(item.densification_gain) for item in descriptors]
    visibility = [float(item.visible_scan_count) for item in descriptors]
    height = [float(item.minimum_visible_return_height_m) for item in descriptors]
    diagnostics = {
        "joint_density": {
            "minimum_densification_gain": min(density),
            "mean_densification_gain": math.fsum(density) / len(density),
            "maximum_densification_gain": max(density),
        },
        "anomaly_visibility": {
            "minimum_visible_scan_count": min(visibility),
            "mean_visible_scan_count": math.fsum(visibility) / len(visibility),
            "minimum_visible_return_height_m": min(height),
        },
        "renderer_integrity": {
            "window_count": 5.0,
            "source_observation_count": 9.0,
            "shared_world_count": 1.0,
        },
    }
    return expected_clip, world.identity, diagnostics


@dataclass(frozen=True, slots=True)
class S01ClipEvidenceRecord:
    """Content-checked manifest for one generated held-out torus clip."""

    payload: Mapping[str, object]
    record_sha256: str
    clip_identity: str
    world_identity: str

    @classmethod
    def from_mapping(
        cls, value: object, *, protocol: object, plan_position: int
    ) -> S01ClipEvidenceRecord:
        payload = _verified_record(
            value,
            name=f"S01 clip evidence {plan_position}",
            keys={
                "format",
                "protocol_identity",
                "held_out_recipe_identity",
                "held_out_plan_identity",
                "plan",
                "clip_manifest",
                "diagnostics",
            },
            expected_format=S01_CLIP_EVIDENCE_FORMAT,
        )
        expected_plan = tuple(
            _plain_json(item)
            for item in getattr(protocol, "held_out_synthetic_shift_plan")()
        )
        index = plan_position - 24
        if (
            payload["protocol_identity"] != _protocol_identity(protocol)
            or payload["held_out_recipe_identity"]
            != _held_out_recipe_identity(protocol)
            or payload["held_out_plan_identity"]
            != getattr(protocol, "held_out_synthetic_shift_plan_identity", None)
            or len(expected_plan) != 6
            or index not in range(6)
            or payload["plan"] != expected_plan[index]
        ):
            raise EvaluationError("S01 clip evidence changed its protocol or plan")
        plan = _record_mapping(expected_plan[index], "S01 held-out plan")
        if (
            plan.get("position") != plan_position
            or plan.get("mechanism") != "torus_SDF"
        ):
            raise EvaluationError("S01 evidence is not a held-out torus clip")
        clip_identity, world_identity, expected_diagnostics = (
            _validated_s01_clip_manifest(
                payload["clip_manifest"], protocol=protocol, plan=plan
            )
        )
        observed = {
            name: _diagnostic_group(value, f"S01 diagnostics.{name}")
            for name, value in _record_mapping(
                payload["diagnostics"], "S01 diagnostics"
            ).items()
        }
        if observed != expected_diagnostics:
            raise EvaluationError("S01 diagnostics do not reproduce")
        return cls(
            payload,
            _sha256(payload["record_sha256"], "S01 clip evidence"),
            clip_identity,
            world_identity,
        )

    @classmethod
    def load(
        cls, path: Path | str, *, protocol: object, plan_position: int
    ) -> S01ClipEvidenceRecord:
        _, value = _read_json_record(path, "S01 clip evidence")
        return cls.from_mapping(value, protocol=protocol, plan_position=plan_position)


def _s01_clip_evidence_from_rendered(
    *, protocol: object, plan_position: int, clip: object
) -> dict[str, object]:
    """Build S01 evidence only from the authoritative rendered clip class."""

    try:
        from .render import DevelopmentClipWorld
        from .train import _normalized_window_manifest
    except ImportError:  # pragma: no cover - direct module execution
        from render import DevelopmentClipWorld
        from train import _normalized_window_manifest

    if _protocol_status(protocol).get("current_node") != "S01":
        raise EvaluationError("held-out clip evidence can be created only at S01")
    if type(clip) is not DevelopmentClipWorld or clip.mechanism != "torus_SDF":
        raise EvaluationError("S01 evidence requires a rendered held-out torus clip")
    plan = tuple(getattr(protocol, "held_out_synthetic_shift_plan")())
    index = plan_position - 24
    if len(plan) != 6 or index not in range(6):
        raise EvaluationError("S01 plan position must be one of 24..29")
    manifest = _normalized_window_manifest(clip.to_manifest())
    _, _, diagnostics = _validated_s01_clip_manifest(
        manifest, protocol=protocol, plan=_record_mapping(plan[index], "S01 plan")
    )
    record = _content_addressed(
        {
            "format": S01_CLIP_EVIDENCE_FORMAT,
            "protocol_identity": _protocol_identity(protocol),
            "held_out_recipe_identity": _held_out_recipe_identity(protocol),
            "held_out_plan_identity": getattr(
                protocol, "held_out_synthetic_shift_plan_identity", None
            ),
            "plan": _plain_json(plan[index]),
            "clip_manifest": manifest,
            "diagnostics": diagnostics,
        }
    )
    S01ClipEvidenceRecord.from_mapping(
        record, protocol=protocol, plan_position=plan_position
    )
    return record


def generate_s01_clip_evidence(
    *,
    protocol_path: Path | str,
    g3_verdict_path: Path | str,
    data_root: Path | str,
    plan_position: int,
    destination: Path | str,
) -> dict[str, str]:
    """Generate one authorized and replayable held-out plan position."""

    try:
        from .protocol import AJAEProtocol, load_protocol
        from .render import (
            RenderError,
            _sample_held_out_torus_clip_world,
            collect_observed_obstacle_index,
            load_qualified_support_pool,
        )
        from .train import (
            TrainingError,
            _open_training_sequence,
            _verified_renderer_inputs,
        )
    except ImportError:  # pragma: no cover - direct module execution
        from protocol import AJAEProtocol, load_protocol
        from render import (
            RenderError,
            _sample_held_out_torus_clip_world,
            collect_observed_obstacle_index,
            load_qualified_support_pool,
        )
        from train import (
            TrainingError,
            _open_training_sequence,
            _verified_renderer_inputs,
        )

    protocol = load_protocol(protocol_path)
    if type(protocol) is not AJAEProtocol or protocol.status["current_node"] != "S01":
        raise EvaluationError("held-out torus generation is authorized only at S01")
    g3 = FormalGateVerdictRecord.load(g3_verdict_path, protocol=protocol)
    if (
        g3.gate != "G3"
        or g3.decision != "pass"
        or g3.record_sha256 != protocol.status["g3_verdict_identity"]
    ):
        raise EvaluationError("S01 generation requires the bound passed G3 verdict")
    plan = tuple(protocol.held_out_synthetic_shift_plan())
    index = plan_position - 24
    if index not in range(6):
        raise EvaluationError("S01 plan position must be one of 24..29")
    row = plan[index]
    try:
        ray_grid, sensor = _verified_renderer_inputs(protocol)
        sequence = _open_training_sequence(data_root, protocol, 201)
        support_pool = load_qualified_support_pool(
            protocol.support_pool_path(201), source_sequence_id=201
        )
        sources = tuple(
            sequence.source_frame(int(frame_id)) for frame_id in row["source_frames"]
        )
        obstacles = collect_observed_obstacle_index(sources, source_sequence_id=201)
        clip = _sample_held_out_torus_clip_world(
            support_pool,
            obstacles,
            sources,
            ray_grid,
            sensor,
            int(row["root_seed"]),
            renderer_identity=protocol.renderer_identity,
            density_voxel_size_m=float(
                protocol.render["window_descriptors"]["density_voxel_size_m"]
            ),
            maximum_attempts=int(row["maximum_observation_attempts"]),
        )
    except (OSError, ValueError, TypeError, RenderError, TrainingError) as error:
        raise EvaluationError(
            f"cannot generate held-out S01 plan position {plan_position}"
        ) from error
    record = _s01_clip_evidence_from_rendered(
        protocol=protocol, plan_position=plan_position, clip=clip
    )
    root = _protocol_root(protocol)
    target = Path(destination).expanduser().resolve()
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as error:
        raise EvaluationError(
            "S01 evidence must stay under the protocol root"
        ) from error
    if target.exists():
        raise EvaluationError("refusing to replace existing S01 evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    loaded = S01ClipEvidenceRecord.load(
        target, protocol=protocol, plan_position=plan_position
    )
    return {"path": relative, "record_sha256": loaded.record_sha256}


@dataclass(frozen=True, slots=True)
class S01ShiftAuditRecord:
    """Six loadable held-out clips, with no post-G3 method selection."""

    payload: Mapping[str, object]
    record_sha256: str
    g3_verdict: FormalGateVerdictRecord
    method_roles: Mapping[str, EvaluationIdentity]
    formal_result_identities: Mapping[str, str]
    clip_evidence: tuple[S01ClipEvidenceRecord, ...]

    @classmethod
    def from_mapping(cls, value: object, *, protocol: object) -> S01ShiftAuditRecord:
        keys = {
            "format",
            "status",
            "protocol_identity",
            "g3_verdict",
            "held_out_recipe_identity",
            "held_out_plan_identity",
            "method_selection_rule",
            "method_roles",
            "formal_result_identities",
            "clip_evidence",
            "method_changed",
        }
        payload = _verified_record(
            value,
            name="S01 shift audit",
            keys=keys,
            expected_format=S01_SHIFT_AUDIT_FORMAT,
        )
        if (
            payload["status"] != "completed"
            or payload["method_changed"] is not False
            or payload["method_selection_rule"] != METHOD_SELECTION_RULE
        ):
            raise EvaluationError(
                "S01 must be complete without selecting or changing the method"
            )
        if payload["protocol_identity"] != _protocol_identity(protocol):
            raise EvaluationError("S01 audit belongs to another protocol")
        g3 = FormalGateVerdictRecord.from_mapping(
            payload["g3_verdict"], protocol=protocol
        )
        if g3.gate != "G3" or g3.decision != "pass":
            raise EvaluationError("S01 requires a reproduced passed G3 verdict")
        if _protocol_status(protocol).get("g3_verdict_identity") != g3.record_sha256:
            raise EvaluationError("S01 is not bound to the protocol G3 decision")
        expected_methods, expected_results = _selected_formal_methods(g3)
        raw_methods = _record_mapping(payload["method_roles"], "S01 method roles")
        _record_keys(raw_methods, set(METHOD_ROLES), "S01 method roles")
        methods = {
            role: _evaluation_identity_record(raw_methods[role], f"S01 {role}")
            for role in METHOD_ROLES
        }
        if methods != expected_methods:
            raise EvaluationError(
                "S01 method roles differ from the result-blind seed-zero selection"
            )
        raw_results = _record_mapping(
            payload["formal_result_identities"], "S01 formal result identities"
        )
        _record_keys(raw_results, set(METHOD_ROLES), "S01 result identities")
        result_identities = {
            role: _sha256(raw_results[role], f"S01 {role} formal result")
            for role in METHOD_ROLES
        }
        if result_identities != expected_results:
            raise EvaluationError("S01 changed the selected formal results")

        recipe_identity = _held_out_recipe_identity(protocol)
        plan_identity = _sha256(
            getattr(protocol, "held_out_synthetic_shift_plan_identity", None),
            "held-out shift plan identity",
        )
        if (
            payload["held_out_recipe_identity"] != recipe_identity
            or payload["held_out_plan_identity"] != plan_identity
        ):
            raise EvaluationError("S01 does not bind the held-out recipe and plan")
        references = _record_list(payload["clip_evidence"], "S01 clip evidence")
        if len(references) != 6:
            raise EvaluationError("S01 must reference exactly six held-out clips")
        loaded: list[S01ClipEvidenceRecord] = []
        paths: set[Path] = set()
        for index, value_item in enumerate(references):
            reference = _record_mapping(value_item, f"S01 clip reference {index}")
            _record_keys(
                reference, {"path", "record_sha256"}, f"S01 clip reference {index}"
            )
            path = _resolve_protocol_record(
                protocol, reference["path"], f"S01 clip evidence {index}"
            )
            if path in paths:
                raise EvaluationError("S01 clip evidence path repeats")
            paths.add(path)
            clip = S01ClipEvidenceRecord.load(
                path, protocol=protocol, plan_position=index + 24
            )
            if clip.record_sha256 != _sha256(
                reference["record_sha256"], f"S01 clip evidence {index} identity"
            ):
                raise EvaluationError("S01 clip reference points to different content")
            loaded.append(clip)
        if (
            len({item.clip_identity for item in loaded}) != 6
            or len({item.world_identity for item in loaded}) != 6
        ):
            raise EvaluationError("S01 clip or world identity repeats")
        record_sha256 = _sha256(payload["record_sha256"], "S01 shift audit")
        _assert_record_lifecycle(
            protocol,
            milestone="S01",
            status_field="s01_shift_audit_identity",
            record_sha256=record_sha256,
        )
        return cls(
            payload,
            record_sha256,
            g3,
            methods,
            result_identities,
            tuple(loaded),
        )

    @classmethod
    def load(cls, path: Path | str, *, protocol: object) -> S01ShiftAuditRecord:
        _, value = _read_json_record(path, "S01 shift audit")
        return cls.from_mapping(value, protocol=protocol)


def complete_s01_shift_audit(
    *,
    protocol: object,
    g3_verdict: FormalGateVerdictRecord | Mapping[str, object],
    clip_evidence: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Bind six saved clip records without permitting any post-G3 method change."""

    if _protocol_status(protocol).get("current_node") != "S01":
        raise EvaluationError("S01 can be completed only at its protocol node")
    g3 = (
        g3_verdict
        if isinstance(g3_verdict, FormalGateVerdictRecord)
        else FormalGateVerdictRecord.from_mapping(g3_verdict, protocol=protocol)
    )
    methods, results = _selected_formal_methods(g3)
    unsigned = {
        "format": S01_SHIFT_AUDIT_FORMAT,
        "status": "completed",
        "protocol_identity": _protocol_identity(protocol),
        "g3_verdict": dict(g3.payload),
        "held_out_recipe_identity": _held_out_recipe_identity(protocol),
        "held_out_plan_identity": getattr(
            protocol, "held_out_synthetic_shift_plan_identity", None
        ),
        "method_selection_rule": METHOD_SELECTION_RULE,
        "method_roles": {
            role: identity.to_dict() for role, identity in methods.items()
        },
        "formal_result_identities": results,
        "clip_evidence": [dict(item) for item in clip_evidence],
        "method_changed": False,
    }
    record = _content_addressed(unsigned)
    S01ShiftAuditRecord.from_mapping(record, protocol=protocol)
    return record


@dataclass(frozen=True, slots=True)
class MethodFreezeRecord:
    """Content-addressed authorization for all predeclared sealed-data methods."""

    path: Path | None
    payload: Mapping[str, object]
    record_sha256: str
    method_roles: Mapping[str, EvaluationIdentity]
    decision_thresholds: Mapping[str, float]
    sealed_sequences: Mapping[str, tuple[int, ...]]
    s01_audit: S01ShiftAuditRecord | None

    @property
    def evaluation_identity(self) -> EvaluationIdentity:
        """Return the final identity for callers that previously expected one method."""

        if "frozen_final" in self.method_roles:
            return self.method_roles["frozen_final"]
        return next(iter(self.method_roles.values()))

    @classmethod
    def _from_multi_mapping(
        cls,
        value: object,
        *,
        protocol: object,
        path: Path | None = None,
    ) -> MethodFreezeRecord:
        keys = {
            "format",
            "status",
            "protocol_identity",
            "s01_audit",
            "method_selection_rule",
            "method_roles",
            "implementation_manifest",
            "implementation_identity",
            "dependency_manifest",
            "dependency_identity",
            "v01_criteria_identity",
            "sealed_sequences",
        }
        payload = _verified_record(
            value,
            name="multi-method freeze record",
            keys=keys,
            expected_format=MULTI_METHOD_FREEZE_FORMAT,
        )
        if (
            payload["status"] != METHOD_FREEZE_STATUS
            or payload["protocol_identity"] != _protocol_identity(protocol)
            or payload["method_selection_rule"] != METHOD_SELECTION_RULE
        ):
            raise EvaluationError("method-freeze record changed its protocol or status")
        s01 = S01ShiftAuditRecord.from_mapping(payload["s01_audit"], protocol=protocol)
        if (
            _protocol_status(protocol).get("s01_shift_audit_identity")
            != s01.record_sha256
        ):
            raise EvaluationError("method-freeze record is not bound to protocol S01")
        expected_methods, expected_results = _selected_formal_methods(s01.g3_verdict)
        expected_records = _selected_formal_records(s01.g3_verdict)
        raw_methods = _record_mapping(payload["method_roles"], "method-freeze roles")
        _record_keys(raw_methods, set(METHOD_ROLES), "method-freeze roles")
        implementation = _record_mapping(
            payload["implementation_manifest"], "implementation manifest"
        )
        _record_keys(
            implementation,
            set(FROZEN_IMPLEMENTATION_FILES),
            "implementation manifest",
        )
        implementation_manifest = {
            name: _sha256(value, f"implementation file {name}")
            for name, value in implementation.items()
        }
        implementation_identity = _sha256(
            payload["implementation_identity"], "implementation manifest identity"
        )
        if implementation_manifest != _implementation_manifest(
            protocol
        ) or implementation_identity != _json_sha256(implementation_manifest):
            raise EvaluationError("method-freeze implementation no longer reproduces")
        dependency = _record_mapping(
            payload["dependency_manifest"], "dependency manifest"
        )
        expected_dependency = _dependency_manifest()
        _record_keys(dependency, set(expected_dependency), "dependency manifest")
        dependency_manifest = {
            name: _record_string(value, f"dependency {name}")
            for name, value in dependency.items()
        }
        dependency_identity = _sha256(
            payload["dependency_identity"], "dependency manifest identity"
        )
        if (
            dependency_manifest != expected_dependency
            or dependency_identity != _json_sha256(dependency_manifest)
        ):
            raise EvaluationError("method-freeze dependencies no longer reproduce")
        criteria, v01_identity = _frozen_gate_criteria(protocol, "V01")
        safety = _record_mapping(criteria.get("normal_safety"), "V01 normal safety")
        threshold_rule_identity = _sha256(
            safety.get("threshold_rule_identity"), "V01 threshold rule identity"
        )
        methods: dict[str, EvaluationIdentity] = {}
        thresholds: dict[str, float] = {}
        for role in METHOD_ROLES:
            method = _record_mapping(raw_methods[role], f"method role {role}")
            _record_keys(
                method,
                {
                    "evaluation_identity",
                    "formal_result_identity",
                    "decision_threshold",
                    "threshold_rule_identity",
                    "implementation_identity",
                    "dependency_identity",
                },
                f"method role {role}",
            )
            methods[role] = _evaluation_identity_record(
                method["evaluation_identity"], f"method role {role} identity"
            )
            if (
                _sha256(
                    method["formal_result_identity"],
                    f"method role {role} formal result",
                )
                != expected_results[role]
                or method["threshold_rule_identity"] != threshold_rule_identity
                or method["implementation_identity"] != implementation_identity
                or method["dependency_identity"] != dependency_identity
            ):
                raise EvaluationError(f"method role {role} changed its frozen evidence")
            threshold = _record_number(
                method["decision_threshold"], f"method role {role} threshold"
            )
            if role != "B0_reference" and not 0.0 <= threshold <= 1.0:
                raise EvaluationError("learned-method thresholds must lie in [0,1]")
            if not math.isclose(
                threshold,
                expected_records[role].decision_threshold,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise EvaluationError(
                    f"method role {role} threshold differs from formal evidence"
                )
            thresholds[role] = threshold
        if methods != expected_methods or methods != s01.method_roles:
            raise EvaluationError("method-freeze roles differ from the passed evidence")
        if (
            methods["B0_reference"].condition != "B0"
            or methods["B1_reference"].condition != "B1"
            or methods["frozen_final"].condition != "B3"
        ):
            raise EvaluationError("method-freeze role conditions changed")
        if expected_results != dict(s01.formal_result_identities):
            raise EvaluationError("method-freeze selected different formal results")
        if payload["v01_criteria_identity"] != v01_identity:
            raise EvaluationError("method-freeze record changed V01 criteria")
        expected_sequences = {
            "val": list(getattr(protocol, "public_sequence_ids")),
            "test": list(getattr(protocol, "hidden_sequence_ids")),
        }
        if payload["sealed_sequences"] != expected_sequences:
            raise EvaluationError(
                "method-freeze record does not bind the sealed population"
            )
        record_sha256 = _sha256(payload["record_sha256"], "multi-method freeze record")
        _assert_record_lifecycle(
            protocol,
            milestone="M01",
            status_field="m01_method_freeze_identity",
            record_sha256=record_sha256,
        )
        return cls(
            path,
            payload,
            record_sha256,
            methods,
            thresholds,
            {name: tuple(values) for name, values in expected_sequences.items()},
            s01,
        )

    @classmethod
    def _from_legacy_mapping(
        cls,
        value: object,
        *,
        expected_identity: EvaluationIdentity,
        protocol: object,
        path: Path | None,
    ) -> MethodFreezeRecord:
        keys = {
            "format",
            "status",
            "evaluation_identity",
            "sealed_sequences",
        }
        payload = _verified_record(
            value,
            name="legacy method-freeze record",
            keys=keys,
            expected_format=METHOD_FREEZE_FORMAT,
        )
        identity = _evaluation_identity_record(
            payload["evaluation_identity"], "legacy method identity"
        )
        if payload["status"] != METHOD_FREEZE_STATUS or identity != expected_identity:
            raise EvaluationError(
                "legacy method-freeze record does not match this method"
            )
        expected_sequences = {
            "val": list(getattr(protocol, "public_sequence_ids")),
            "test": list(getattr(protocol, "hidden_sequence_ids")),
        }
        if payload["sealed_sequences"] != expected_sequences:
            raise EvaluationError(
                "legacy method-freeze record does not bind the sealed population"
            )
        return cls(
            path,
            payload,
            _sha256(payload["record_sha256"], "legacy method-freeze record"),
            {"legacy": identity},
            {},
            {name: tuple(values) for name, values in expected_sequences.items()},
            None,
        )

    @classmethod
    def from_mapping(cls, value: object, *, protocol: object) -> MethodFreezeRecord:
        """Load only the multi-method scientific record from an embedded mapping."""

        return cls._from_multi_mapping(value, protocol=protocol)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        expected_identity: EvaluationIdentity,
        protocol: object,
        role: str | None = None,
        require_multi_method: bool = False,
    ) -> MethodFreezeRecord:
        if type(expected_identity) is not EvaluationIdentity:
            raise TypeError("expected_identity must be EvaluationIdentity")
        if expected_identity.test_fixture:
            raise EvaluationError("test fixtures cannot authorize sealed data")
        resolved, value = _read_json_record(path, "method-freeze record")
        raw = _record_mapping(value, "method-freeze record")
        if raw.get("format") == MULTI_METHOD_FREEZE_FORMAT:
            record = cls._from_multi_mapping(value, protocol=protocol, path=resolved)
            if role is None:
                matches = [
                    name
                    for name, identity in record.method_roles.items()
                    if identity == expected_identity
                ]
                if len(matches) != 1:
                    raise EvaluationError(
                        "method-freeze record does not uniquely identify this method"
                    )
                role = matches[0]
            if (
                role not in METHOD_ROLES
                or record.method_roles[role] != expected_identity
            ):
                raise EvaluationError("method-freeze role does not match this method")
            return record
        if require_multi_method:
            raise EvaluationError("sealed data require the multi-method M01 record")
        return cls._from_legacy_mapping(
            value,
            expected_identity=expected_identity,
            protocol=protocol,
            path=resolved,
        )


def freeze_method_record(
    *,
    protocol: object,
    s01_audit: S01ShiftAuditRecord | Mapping[str, object],
) -> dict[str, object]:
    """Freeze exact methods, thresholds, code, and dependencies before V01."""

    if _protocol_status(protocol).get("current_node") != "M01":
        raise EvaluationError("M01 can be frozen only at its protocol node")
    s01 = (
        s01_audit
        if isinstance(s01_audit, S01ShiftAuditRecord)
        else S01ShiftAuditRecord.from_mapping(s01_audit, protocol=protocol)
    )
    criteria, v01_identity = _frozen_gate_criteria(protocol, "V01")
    safety = _record_mapping(criteria.get("normal_safety"), "V01 normal safety")
    threshold_rule_identity = _sha256(
        safety.get("threshold_rule_identity"), "V01 threshold rule identity"
    )
    selected_records = _selected_formal_records(s01.g3_verdict)
    decision_thresholds = {
        role: selected_records[role].decision_threshold for role in METHOD_ROLES
    }
    implementation_manifest = _implementation_manifest(protocol)
    dependency_manifest = _dependency_manifest()
    implementation_identity = _json_sha256(implementation_manifest)
    dependency_identity = _json_sha256(dependency_manifest)
    unsigned = {
        "format": MULTI_METHOD_FREEZE_FORMAT,
        "status": METHOD_FREEZE_STATUS,
        "protocol_identity": _protocol_identity(protocol),
        "s01_audit": dict(s01.payload),
        "method_selection_rule": METHOD_SELECTION_RULE,
        "method_roles": {
            role: {
                "evaluation_identity": identity.to_dict(),
                "formal_result_identity": s01.formal_result_identities[role],
                "decision_threshold": decision_thresholds[role],
                "threshold_rule_identity": threshold_rule_identity,
                "implementation_identity": implementation_identity,
                "dependency_identity": dependency_identity,
            }
            for role, identity in s01.method_roles.items()
        },
        "implementation_manifest": implementation_manifest,
        "implementation_identity": implementation_identity,
        "dependency_manifest": dependency_manifest,
        "dependency_identity": dependency_identity,
        "v01_criteria_identity": v01_identity,
        "sealed_sequences": {
            "val": list(getattr(protocol, "public_sequence_ids")),
            "test": list(getattr(protocol, "hidden_sequence_ids")),
        },
    }
    record = _content_addressed(unsigned)
    MethodFreezeRecord.from_mapping(record, protocol=protocol)
    return record


@dataclass(frozen=True, slots=True)
class PublicMetricEvidence:
    """Raw covered public points used to reproduce one sequence's metrics."""

    path: Path
    file_sha256: str
    data_identity: str
    coverage_identity: str
    score_identity: str
    metrics_unit_interval: Mapping[str, float]

    @classmethod
    def load(
        cls,
        reference: object,
        *,
        protocol: object,
        expected_identity: EvaluationIdentity,
        expected_sequence_id: int,
    ) -> PublicMetricEvidence:
        path, file_sha256 = _artifact_reference(
            protocol, reference, "public metric evidence"
        )
        arrays, file_sha256 = _load_npz(
            path,
            expected_keys={
                "format",
                "protocol_identity",
                "evaluation_identity",
                "partition",
                "sequence_id",
                "source_frame",
                "source_ray",
                "source_slot",
                "occurrence_count",
                "coordinates",
                "raw_semantic",
                "scores",
            },
            expected_sha256=file_sha256,
            name="public metric evidence",
        )
        if (
            _npz_text(arrays, "format", "public metric evidence")
            != PUBLIC_METRIC_EVIDENCE_FORMAT
            or _npz_text(arrays, "protocol_identity", "public metric evidence")
            != _protocol_identity(protocol)
            or _identity_from_npz(
                arrays, "evaluation_identity", "public metric evidence"
            )
            != expected_identity
            or _npz_text(arrays, "partition", "public metric evidence") != "val"
            or _npz_integer(arrays, "sequence_id", "public metric evidence")
            != expected_sequence_id
        ):
            raise EvaluationError("public metric evidence changed its frozen inputs")
        frame = arrays["source_frame"]
        ray = arrays["source_ray"]
        slot = arrays["source_slot"]
        occurrence = arrays["occurrence_count"]
        coordinates = arrays["coordinates"]
        semantic = arrays["raw_semantic"]
        scores = arrays["scores"]
        count = scores.size
        if (
            count == 0
            or any(
                value.dtype != np.dtype(np.int64) or value.shape != (count,)
                for value in (frame, ray, slot, occurrence, semantic)
            )
            or coordinates.dtype != np.dtype(np.float32)
            or coordinates.shape != (count, 3)
            or scores.dtype != np.dtype(np.float64)
            or scores.shape != (count,)
            or not np.isfinite(coordinates).all()
            or not np.isfinite(scores).all()
            or np.any(frame < 0)
            or np.any(ray < 0)
            or np.any(slot < 0)
            or np.any((occurrence < 1) | (occurrence > 5))
        ):
            raise EvaluationError("public metric arrays are invalid")
        keys = np.stack((frame, ray), axis=1)
        if np.unique(keys, axis=0).shape[0] != count:
            raise EvaluationError("public metric evidence repeats a physical point")
        if expected_identity.condition != "B0" and np.any(
            (scores < 0.0) | (scores > 1.0)
        ):
            raise EvaluationError("learned public scores must lie in [0,1]")
        accumulator = PointMetricAccumulator(protocol)
        for frame_id in np.unique(frame):
            selected = frame == frame_id
            accumulator.update(
                coordinates[selected], scores[selected], semantic[selected]
            )
        computed = accumulator.compute()
        metrics = {
            name: _record_number(computed[name], f"computed public {name}") / 100.0
            for name in ("AP", "AUROC", "FPR95")
        }
        data_identity = _array_identity(
            "ajae-public-data-v1", frame, ray, slot, coordinates, semantic
        )
        coverage_identity = _array_identity(
            "ajae-public-coverage-v1", frame, ray, slot, occurrence
        )
        score_identity = _array_identity(
            "ajae-public-scores-v1", frame, ray, slot, scores
        )
        return cls(
            path,
            file_sha256,
            data_identity,
            coverage_identity,
            score_identity,
            metrics,
        )


def _save_public_metric_arrays(
    path: str,
    *,
    protocol: object,
    evaluation_identity: EvaluationIdentity,
    sequence_id: int,
    source_frame: np.ndarray,
    source_ray: np.ndarray,
    source_slot: np.ndarray,
    occurrence_count: np.ndarray,
    coordinates: np.ndarray,
    raw_semantic: np.ndarray,
    scores: np.ndarray,
) -> dict[str, str]:
    """Persist the exact covered public points before building a V01 result."""

    _assert_frozen_evaluation_components(evaluation_identity, protocol)
    if _protocol_status(protocol).get("current_node") != "V01":
        raise EvaluationError("public metric evidence can be saved only at V01")
    destination = _artifact_destination(protocol, path, "public metric evidence")
    arrays = {
        "format": np.asarray(PUBLIC_METRIC_EVIDENCE_FORMAT),
        "protocol_identity": np.asarray(_protocol_identity(protocol)),
        "evaluation_identity": np.asarray(
            json.dumps(
                evaluation_identity.to_dict(), sort_keys=True, separators=(",", ":")
            )
        ),
        "partition": np.asarray("val"),
        "sequence_id": np.asarray(sequence_id, dtype=np.int64),
        "source_frame": np.asarray(source_frame, dtype=np.int64),
        "source_ray": np.asarray(source_ray, dtype=np.int64),
        "source_slot": np.asarray(source_slot, dtype=np.int64),
        "occurrence_count": np.asarray(occurrence_count, dtype=np.int64),
        "coordinates": np.asarray(coordinates, dtype=np.float32),
        "raw_semantic": np.asarray(raw_semantic, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float64),
    }
    reference = {"path": path, "file_sha256": _save_npz(destination, arrays)}
    PublicMetricEvidence.load(
        reference,
        protocol=protocol,
        expected_identity=evaluation_identity,
        expected_sequence_id=sequence_id,
    )
    return reference


def save_public_metric_evidence(
    path: str,
    *,
    protocol: object,
    sequence: object,
    prediction: SequencePrediction,
) -> dict[str, str]:
    """Persist the complete covered union from one authorized V01 prediction."""

    spec = getattr(sequence, "spec", None)
    if (
        type(prediction) is not SequencePrediction
        or prediction.partition != "val"
        or getattr(spec, "partition", None) != "val"
        or prediction.sequence_id != getattr(spec, "sequence_id", None)
        or prediction.identity.test_fixture
        or prediction.identity.protocol_identity != _protocol_identity(protocol)
    ):
        raise EvaluationError("public evidence requires one formal V01 prediction")
    # This checks the full legal-window domain and label availability before export.
    evaluate_sequence(sequence, prediction, protocol=protocol)
    frames: list[np.ndarray] = []
    rays: list[np.ndarray] = []
    slots: list[np.ndarray] = []
    occurrences: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    semantics: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    fused = prediction.fused
    for frame_id in sorted(prediction.frames):
        source = sequence.source_frame(frame_id)
        labels = getattr(source, "labels", None)
        if labels is None:
            raise EvaluationError("public evidence requires semantic labels")
        real_slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
        selected = fused.frame_mask(frame_id)
        fused_slots = np.asarray(fused.source_slot[selected], dtype=np.int64)
        order = np.argsort(fused_slots)
        if not np.array_equal(fused_slots[order], np.sort(real_slots)):
            raise EvaluationError("public evidence point identities are incomplete")
        frame_prediction = prediction.frames[frame_id]
        frames.append(np.full(real_slots.size, frame_id, dtype=np.int64))
        rays.append(np.asarray(fused.source_ray[selected], dtype=np.int64)[order])
        slots.append(fused_slots[order])
        occurrences.append(
            np.asarray(frame_prediction.occurrence_count[real_slots], dtype=np.int64)
        )
        coordinates.append(
            np.asarray(getattr(source, "xyzi"), dtype=np.float32)[real_slots, :3]
        )
        semantics.append(
            np.asarray(getattr(labels, "semantic"), dtype=np.int64)[real_slots]
        )
        scores.append(np.asarray(frame_prediction.score[real_slots], dtype=np.float64))
    return _save_public_metric_arrays(
        path,
        protocol=protocol,
        evaluation_identity=prediction.identity,
        sequence_id=prediction.sequence_id,
        source_frame=np.concatenate(frames),
        source_ray=np.concatenate(rays),
        source_slot=np.concatenate(slots),
        occurrence_count=np.concatenate(occurrences),
        coordinates=np.concatenate(coordinates),
        raw_semantic=np.concatenate(semantics),
        scores=np.concatenate(scores),
    )


@dataclass(frozen=True, slots=True)
class PublicSequenceResultRecord:
    """One public-sequence result tied to an exact M01 method role."""

    payload: Mapping[str, object]
    record_sha256: str
    method_freeze_identity: str
    role: str
    evaluation_identity: EvaluationIdentity
    sequence_id: int
    data_identity: str
    average_precision: float
    decision_threshold: float
    normal_safety: Mapping[str, float]
    normal_safety_identity: str
    normal_safety_threshold_rule_identity: str

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        protocol: object,
        method_freeze: MethodFreezeRecord,
    ) -> PublicSequenceResultRecord:
        keys = {
            "format",
            "method_freeze_identity",
            "role",
            "evaluation_identity",
            "partition",
            "sequence_id",
            "data_identity",
            "coverage_identity",
            "score_identity",
            "metric_evidence",
            "point_metrics_unit_interval",
            "normal_safety",
            "normal_safety_evidence",
        }
        payload = _verified_record(
            value,
            name="public sequence result",
            keys=keys,
            expected_format=PUBLIC_SEQUENCE_RESULT_FORMAT,
        )
        if payload["method_freeze_identity"] != method_freeze.record_sha256:
            raise EvaluationError("public result belongs to another M01 freeze")
        role = payload["role"]
        if role not in METHOD_ROLES:
            raise EvaluationError("public result has an invalid method role")
        identity = _evaluation_identity_record(
            payload["evaluation_identity"], "public evaluation identity"
        )
        if identity != method_freeze.method_roles[str(role)]:
            raise EvaluationError("public result changed its frozen method")
        if payload["partition"] != "val":
            raise EvaluationError("V01 accepts only public validation results")
        sequence_id = _record_integer(payload["sequence_id"], "public sequence ID")
        if sequence_id not in tuple(getattr(protocol, "public_sequence_ids")):
            raise EvaluationError("public result sequence is outside V01")
        data_identity = _sha256(payload["data_identity"], "public sequence data")
        coverage_identity = _sha256(
            payload["coverage_identity"], "public prediction coverage"
        )
        score_identity = _sha256(payload["score_identity"], "public point scores")
        metric_evidence = PublicMetricEvidence.load(
            payload["metric_evidence"],
            protocol=protocol,
            expected_identity=identity,
            expected_sequence_id=sequence_id,
        )
        if (
            data_identity != metric_evidence.data_identity
            or coverage_identity != metric_evidence.coverage_identity
            or score_identity != metric_evidence.score_identity
        ):
            raise EvaluationError("public result identities do not match raw evidence")
        metrics = _record_mapping(
            payload["point_metrics_unit_interval"], "public point metrics"
        )
        _record_keys(metrics, {"AP", "AUROC", "FPR95"}, "public point metrics")
        average_precision = _unit_interval(metrics["AP"], "public sequence AP")
        _unit_interval(metrics["AUROC"], "public sequence AUROC")
        _unit_interval(metrics["FPR95"], "public sequence FPR95")
        if dict(metric_evidence.metrics_unit_interval) != {
            name: _unit_interval(metrics[name], f"public sequence {name}")
            for name in ("AP", "AUROC", "FPR95")
        }:
            raise EvaluationError("public metrics do not reproduce from raw evidence")
        safety = _record_mapping(payload["normal_safety"], "public normal safety")
        _record_keys(
            safety,
            {
                "evaluation_set_identity",
                "threshold_rule_identity",
                "decision_threshold",
                "orientation",
                "statistics",
            },
            "public normal safety",
        )
        safety_identity = _sha256(
            safety["evaluation_set_identity"], "public normal-safety population"
        )
        threshold_identity = _sha256(
            safety["threshold_rule_identity"], "public safety threshold rule"
        )
        decision_threshold = _record_number(
            safety["decision_threshold"], "public safety decision threshold"
        )
        if not math.isclose(
            decision_threshold,
            method_freeze.decision_thresholds[str(role)],
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise EvaluationError("public result changed its frozen threshold")
        criteria, _ = _frozen_gate_criteria(protocol, "V01")
        criteria_safety = _record_mapping(
            criteria.get("normal_safety"), "V01 normal-safety criteria"
        )
        if safety_identity != _sha256(
            criteria_safety.get("evaluation_set_identity"),
            "V01 normal-safety population",
        ) or threshold_identity != _sha256(
            criteria_safety.get("threshold_rule_identity"),
            "V01 normal-safety threshold rule",
        ):
            raise EvaluationError("public result changed normal-safety criteria")
        if safety["orientation"] != "higher_is_worse":
            raise EvaluationError("public normal safety must use higher-is-worse")
        raw_statistics = _record_mapping(
            safety["statistics"], "public normal-safety statistics"
        )
        if not raw_statistics:
            raise EvaluationError("public normal-safety statistics cannot be empty")
        statistics = {
            name: _unit_interval(value, f"public normal-safety statistic {name}")
            for name, value in raw_statistics.items()
        }
        safety_evidence = NormalSafetyEvidence.load(
            payload["normal_safety_evidence"],
            protocol=protocol,
            expected_identity=identity,
        )
        if (
            safety_identity != safety_evidence.evaluation_set_identity
            or threshold_identity != safety_evidence.threshold_rule_identity
            or not math.isclose(
                decision_threshold,
                safety_evidence.decision_threshold,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or statistics
            != {
                "normal_false_positive_rate_at_frozen_threshold_unit_interval": (
                    safety_evidence.false_positive_rate
                )
            }
        ):
            raise EvaluationError(
                "public normal safety does not reproduce from raw evidence"
            )
        return cls(
            payload,
            _sha256(payload["record_sha256"], "public sequence result"),
            method_freeze.record_sha256,
            str(role),
            identity,
            sequence_id,
            data_identity,
            average_precision,
            decision_threshold,
            statistics,
            safety_identity,
            threshold_identity,
        )


def make_public_sequence_result(
    *,
    protocol: object,
    method_freeze: MethodFreezeRecord | Mapping[str, object],
    role: str,
    sequence_id: int,
    metric_evidence: Mapping[str, str],
    normal_safety_evidence: Mapping[str, str],
) -> dict[str, object]:
    """Build one V01 input strictly from loadable raw evidence."""

    if _protocol_status(protocol).get("current_node") != "V01":
        raise EvaluationError("public sequence results can be created only at V01")
    freeze = (
        method_freeze
        if isinstance(method_freeze, MethodFreezeRecord)
        else MethodFreezeRecord.from_mapping(method_freeze, protocol=protocol)
    )
    if role not in METHOD_ROLES:
        raise EvaluationError("public result role is invalid")
    raw = PublicMetricEvidence.load(
        metric_evidence,
        protocol=protocol,
        expected_identity=freeze.method_roles[role],
        expected_sequence_id=sequence_id,
    )
    safety_raw = NormalSafetyEvidence.load(
        normal_safety_evidence,
        protocol=protocol,
        expected_identity=freeze.method_roles[role],
    )
    criteria, _ = _frozen_gate_criteria(protocol, "V01")
    criteria_safety = _record_mapping(
        criteria.get("normal_safety"), "V01 normal-safety criteria"
    )
    unsigned = {
        "format": PUBLIC_SEQUENCE_RESULT_FORMAT,
        "method_freeze_identity": freeze.record_sha256,
        "role": role,
        "evaluation_identity": freeze.method_roles[role].to_dict(),
        "partition": "val",
        "sequence_id": sequence_id,
        "data_identity": raw.data_identity,
        "coverage_identity": raw.coverage_identity,
        "score_identity": raw.score_identity,
        "metric_evidence": dict(metric_evidence),
        "point_metrics_unit_interval": dict(raw.metrics_unit_interval),
        "normal_safety": {
            "evaluation_set_identity": criteria_safety.get("evaluation_set_identity"),
            "threshold_rule_identity": criteria_safety.get("threshold_rule_identity"),
            "decision_threshold": freeze.decision_thresholds[role],
            "orientation": "higher_is_worse",
            "statistics": {
                "normal_false_positive_rate_at_frozen_threshold_unit_interval": (
                    safety_raw.false_positive_rate
                )
            },
        },
        "normal_safety_evidence": dict(normal_safety_evidence),
    }
    record = _content_addressed(unsigned)
    PublicSequenceResultRecord.from_mapping(
        record, protocol=protocol, method_freeze=freeze
    )
    return record


@dataclass(frozen=True, slots=True)
class V01VerdictRecord:
    """Reproduced 19-sequence public-validation decision bound to M01."""

    payload: Mapping[str, object]
    record_sha256: str
    method_freeze: MethodFreezeRecord
    decision: str
    sequence_results: Mapping[tuple[str, int], PublicSequenceResultRecord]

    @classmethod
    def from_mapping(cls, value: object, *, protocol: object) -> V01VerdictRecord:
        keys = {
            "format",
            "protocol_identity",
            "method_freeze",
            "criteria_identity",
            "sequence_results",
            "adjudication",
            "decision",
        }
        payload = _verified_record(
            value,
            name="V01 verdict",
            keys=keys,
            expected_format=V01_VERDICT_FORMAT,
        )
        if payload["protocol_identity"] != _protocol_identity(protocol):
            raise EvaluationError("V01 verdict belongs to another protocol")
        freeze = MethodFreezeRecord.from_mapping(
            payload["method_freeze"], protocol=protocol
        )
        status = _protocol_status(protocol)
        if status.get("m01_method_freeze_identity") != freeze.record_sha256:
            raise EvaluationError("V01 verdict is not bound to protocol M01")
        criteria, criteria_identity = _frozen_gate_criteria(protocol, "V01")
        if payload["criteria_identity"] != criteria_identity:
            raise EvaluationError("V01 verdict changed its frozen criteria")
        raw_results = _record_list(payload["sequence_results"], "V01 sequence results")
        indexed: dict[tuple[str, int], PublicSequenceResultRecord] = {}
        for value_item in raw_results:
            result = PublicSequenceResultRecord.from_mapping(
                value_item, protocol=protocol, method_freeze=freeze
            )
            key = (result.role, result.sequence_id)
            if key in indexed:
                raise EvaluationError("V01 repeats a role/sequence result")
            indexed[key] = result
        sequences = tuple(getattr(protocol, "public_sequence_ids"))
        expected = {(role, sequence) for role in METHOD_ROLES for sequence in sequences}
        if set(indexed) != expected:
            raise EvaluationError(
                "V01 requires exactly 19 results for every method role"
            )
        for sequence in sequences:
            if (
                len({indexed[(role, sequence)].data_identity for role in METHOD_ROLES})
                != 1
            ):
                raise EvaluationError(
                    "V01 methods did not evaluate identical sequence data"
                )

        interval = _record_mapping(
            criteria.get("confidence_interval"), "V01 confidence interval"
        )
        confidence = _record_number(
            interval.get("confidence_level"), "V01 confidence level"
        )
        thresholds = _record_mapping(
            criteria.get("comparison_thresholds"), "V01 thresholds"
        )
        comparison_records = {
            "frozen_final_vs_B0": [
                (
                    sequence,
                    indexed[("frozen_final", sequence)],
                    indexed[("B0_reference", sequence)],
                )
                for sequence in sequences
            ],
            "frozen_final_vs_B1": [
                (
                    sequence,
                    indexed[("frozen_final", sequence)],
                    indexed[("B1_reference", sequence)],
                )
                for sequence in sequences
            ],
        }
        comparisons = {
            name: _comparison_result(
                unit_name="sequence_id",
                pairs=[
                    (sequence, candidate.average_precision, reference.average_precision)
                    for sequence, candidate, reference in pairs
                ],
                confidence_level=confidence,
                threshold=_record_mapping(
                    thresholds.get(name), f"V01 {name} threshold"
                ),
                v01=True,
            )
            for name, pairs in comparison_records.items()
        }
        normal_safety = _normal_safety_result(
            unit_name="sequence_id",
            comparison_pairs=comparison_records,  # type: ignore[arg-type]
            criteria=criteria,
        )
        expected_adjudication = {
            "comparisons": comparisons,
            "normal_safety": normal_safety,
        }
        if _plain_json(payload["adjudication"]) != expected_adjudication:
            raise EvaluationError("V01 adjudication does not reproduce")
        passed = all(
            bool(_record_mapping(item, "V01 comparison")["passed"])
            for item in comparisons.values()
        ) and bool(normal_safety["passed"])
        expected_decision = "pass" if passed else "fail"
        if payload["decision"] != expected_decision:
            raise EvaluationError("V01 decision does not reproduce")
        record_sha256 = _sha256(payload["record_sha256"], "V01 verdict")
        _assert_record_lifecycle(
            protocol,
            milestone="V01",
            status_field="v01_verdict_identity",
            record_sha256=record_sha256,
        )
        return cls(
            payload,
            record_sha256,
            freeze,
            expected_decision,
            indexed,
        )

    @classmethod
    def load(cls, path: Path | str, *, protocol: object) -> V01VerdictRecord:
        _, value = _read_json_record(path, "V01 verdict")
        return cls.from_mapping(value, protocol=protocol)


def adjudicate_v01(
    *,
    protocol: object,
    method_freeze: MethodFreezeRecord | Mapping[str, object],
    sequence_results: Iterable[PublicSequenceResultRecord | Mapping[str, object]],
) -> dict[str, object]:
    """Build the sole V01 record and independently reproduce it before return."""

    if _protocol_status(protocol).get("current_node") != "V01":
        raise EvaluationError("V01 can be adjudicated only at its protocol node")
    freeze = (
        method_freeze
        if isinstance(method_freeze, MethodFreezeRecord)
        else MethodFreezeRecord.from_mapping(method_freeze, protocol=protocol)
    )
    values = [
        dict(item.payload)
        if isinstance(item, PublicSequenceResultRecord)
        else dict(item)
        for item in sequence_results
    ]
    criteria, criteria_identity = _frozen_gate_criteria(protocol, "V01")
    indexed: dict[tuple[str, int], PublicSequenceResultRecord] = {}
    for value_item in values:
        result = PublicSequenceResultRecord.from_mapping(
            value_item, protocol=protocol, method_freeze=freeze
        )
        key = (result.role, result.sequence_id)
        if key in indexed:
            raise EvaluationError("V01 repeats a role/sequence result")
        indexed[key] = result
    sequences = tuple(getattr(protocol, "public_sequence_ids"))
    expected = {(role, sequence) for role in METHOD_ROLES for sequence in sequences}
    if set(indexed) != expected:
        raise EvaluationError("V01 requires the complete 19-by-3 result population")
    for sequence in sequences:
        if len({indexed[(role, sequence)].data_identity for role in METHOD_ROLES}) != 1:
            raise EvaluationError("V01 role results use different sequence data")
    interval = _record_mapping(criteria["confidence_interval"], "V01 interval")
    confidence = _record_number(interval["confidence_level"], "V01 confidence")
    thresholds = _record_mapping(criteria["comparison_thresholds"], "V01 thresholds")
    comparison_records = {
        "frozen_final_vs_B0": [
            (
                sequence,
                indexed[("frozen_final", sequence)],
                indexed[("B0_reference", sequence)],
            )
            for sequence in sequences
        ],
        "frozen_final_vs_B1": [
            (
                sequence,
                indexed[("frozen_final", sequence)],
                indexed[("B1_reference", sequence)],
            )
            for sequence in sequences
        ],
    }
    comparisons = {
        name: _comparison_result(
            unit_name="sequence_id",
            pairs=[
                (sequence, candidate.average_precision, reference.average_precision)
                for sequence, candidate, reference in pairs
            ],
            confidence_level=confidence,
            threshold=_record_mapping(thresholds[name], f"V01 {name} threshold"),
            v01=True,
        )
        for name, pairs in comparison_records.items()
    }
    normal_safety = _normal_safety_result(
        unit_name="sequence_id",
        comparison_pairs=comparison_records,  # type: ignore[arg-type]
        criteria=criteria,
    )
    unsigned = {
        "format": V01_VERDICT_FORMAT,
        "protocol_identity": _protocol_identity(protocol),
        "method_freeze": dict(freeze.payload),
        "criteria_identity": criteria_identity,
        "sequence_results": values,
        "adjudication": {
            "comparisons": comparisons,
            "normal_safety": normal_safety,
        },
        "decision": (
            "pass"
            if all(bool(item["passed"]) for item in comparisons.values())
            and bool(normal_safety["passed"])
            else "fail"
        ),
    }
    record = _content_addressed(unsigned)
    V01VerdictRecord.from_mapping(record, protocol=protocol)
    return record


def _require_sealed_node(protocol: object, partition: str) -> None:
    if partition not in {"val", "test"}:
        raise EvaluationError("sealed node checks apply only to val or test")
    status = getattr(protocol, "status", None)
    node = status.get("current_node") if isinstance(status, Mapping) else None
    allowed = {"V01"} if partition == "val" else {"T01"}
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
    v01_verdict_path: Path | str | None = None,
) -> object:
    """Open val/test only after the exact frozen method record is validated."""

    try:
        from .scene import STUSequence, _grant_sealed_sequence_access
    except ImportError:  # pragma: no cover - direct module execution
        from scene import STUSequence, _grant_sealed_sequence_access

    selected = _condition(condition)
    access = None
    if partition in {"val", "test"}:
        _require_sealed_node(protocol, partition)
        if partition == "test" and selected is not ExperimentCondition.B3:
            raise EvaluationError("T01 permits only the frozen_final B3 method")
        role_by_condition = {
            ExperimentCondition.B0: "B0_reference",
            ExperimentCondition.B1: "B1_reference",
            ExperimentCondition.B3: "frozen_final",
        }
        role = role_by_condition.get(selected)
        if role is None:
            raise EvaluationError("sealed access permits only the three M01 roles")
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
            role=role,
            require_multi_method=True,
        )
        status = _protocol_status(protocol)
        if status.get("m01_method_freeze_identity") != record.record_sha256:
            raise EvaluationError(
                "method-freeze record is not bound by the protocol M01 decision"
            )
        if partition == "test":
            if v01_verdict_path is None:
                raise EvaluationError("T01 requires the saved V01 verdict")
            verdict = V01VerdictRecord.load(v01_verdict_path, protocol=protocol)
            if (
                status.get("v01_verdict_identity") != verdict.record_sha256
                or verdict.decision != "pass"
                or verdict.method_freeze.record_sha256 != record.record_sha256
            ):
                raise EvaluationError(
                    "T01 requires the bound passed V01 verdict for this M01 freeze"
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
