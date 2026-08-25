#!/usr/bin/env python3
"""AJAE window fusion, STU-compatible metrics, and prediction writing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from sklearn.cluster import DBSCAN
from sklearn.metrics import auc, average_precision_score, roc_curve


class _EvaluationRange(Protocol):
    minimum_range_m: float
    maximum_range_m: float
    minimum_anomaly_points: int
    normal_point_alarm_rate: float


class _Protocol(Protocol):
    evaluation: _EvaluationRange


class _SourceFrame(Protocol):
    xyzi: np.ndarray


class EvaluationError(ValueError):
    """Report malformed predictions or an undefined evaluation quantity."""


CONDITIONS = ("B0", "B1", "B2", "B3", "B4", "B5")
RELATIVE_TIMES = (-2, -1, 0, 1, 2)
MOVING_NORMAL_SEMANTICS = (252, 253, 254, 255, 256, 257, 258, 259)
METHOD_FREEZE_FORMAT = "ajae-method-freeze-v1"
PUBLIC_CONFIRMATION_FORMAT = "ajae-public-validation-confirmation-v1"
PUBLIC_RESULT_FORMAT = "ajae-public-validation-result-v1"
OFFICIAL_RESULT_FORMAT = "ajae-official-validation-result-v1"


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition(value: str) -> str:
    if value not in CONDITIONS:
        raise EvaluationError(f"condition must be one of {', '.join(CONDITIONS)}")
    return value


def _finite_vector(
    name: str, value: np.ndarray, count: int | None = None
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
    """Select points in the official inclusive 2.5--50 metre interval."""

    xyzi = np.asarray(source if isinstance(source, np.ndarray) else source.xyzi)
    if xyzi.ndim != 2 or xyzi.shape[1] < 3 or not np.isfinite(xyzi[:, :3]).all():
        raise EvaluationError("source coordinates must be finite [N,>=3]")
    minimum, maximum, _ = _evaluation_values(protocol)
    # The released evaluator computes the norm directly on float32 scan data.
    distance = np.linalg.norm(xyzi[:, :3].astype(np.float32, copy=False), axis=1)
    return (distance >= minimum) & (distance <= maximum)


def binary_targets(raw_semantic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map STU raw semantics to anomaly truth and validity."""

    semantic = np.asarray(raw_semantic)
    if semantic.ndim != 1 or not np.issubdtype(semantic.dtype, np.integer):
        raise TypeError("raw_semantic must be an integer vector")
    return semantic == 2, semantic != 0


def normal_alarm_threshold(scores: np.ndarray, alarm_rate: float) -> float:
    """Return the one-based normal-score order statistic used for calibration."""

    values = np.asarray(scores)
    if (
        values.ndim != 1
        or values.size < 1
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise EvaluationError("normal calibration scores must be a finite vector")
    if not 0.0 < alarm_rate < 1.0:
        raise EvaluationError("normal point alarm rate must lie strictly within (0,1)")
    rank = int(math.ceil((1.0 - alarm_rate) * values.size))
    return float(np.partition(values, rank - 1)[rank - 1])


def normal_score_statistics(
    frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    protocol: _Protocol | None = None,
) -> dict[str, float | int]:
    """Report pure-normal 201 scores without the official anomaly-frame gate."""

    collected: list[np.ndarray] = []
    frame_count = 0
    for points, scores, raw_semantic in frames:
        values = np.asarray(scores)
        anomaly, valid = binary_targets(raw_semantic)
        if values.shape != anomaly.shape or not np.isfinite(values).all():
            raise EvaluationError(
                "pure-normal scores and labels must be finite and aligned"
            )
        valid &= _range_mask(protocol, points)
        if bool(np.any(anomaly[valid])):
            raise EvaluationError("pure-normal evaluation received an anomaly label")
        if np.any(valid):
            collected.append(values[valid].astype(np.float64, copy=True))
        frame_count += 1
    if not collected:
        raise EvaluationError("pure-normal evaluation contains no valid point")
    all_scores = np.concatenate(collected)
    alarm_rate = (
        0.001
        if protocol is None
        else float(protocol.evaluation.normal_point_alarm_rate)
    )
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
        "target_alarm_rate": alarm_rate,
        "strict_threshold": threshold,
        "observed_strict_alarm_rate": float(np.mean(all_scores > threshold)),
    }


def _point_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute the official pooled AP, AUROC, and strict-recall FPR95."""

    truth = np.asarray(labels)
    values = np.asarray(scores)
    if truth.dtype != np.bool_ or truth.ndim != 1 or values.shape != truth.shape:
        raise TypeError("point labels and scores must be aligned vectors")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise EvaluationError("point scores must be finite floating values")
    if truth.size == 0 or not bool(truth.any()) or bool(truth.all()):
        raise EvaluationError("point metrics require both normal and anomaly points")
    false_positive, true_positive, thresholds = roc_curve(truth, values)
    candidates = np.flatnonzero(true_positive > 0.95)
    index = int(candidates[0]) if candidates.size else 0
    return {
        "AP": float(100.0 * average_precision_score(truth, values)),
        "AUROC": float(100.0 * auc(false_positive, true_positive)),
        "FPR95": float(100.0 * false_positive[index]),
        "threshold": float(thresholds[index]) if candidates.size else 0.0,
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
        xyz = np.asarray(points)
        values = np.asarray(scores)
        anomaly, valid = binary_targets(raw_semantic)
        if values.ndim != 1 or values.shape != anomaly.shape:
            raise EvaluationError("prediction and label count mismatch")
        if (
            not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise EvaluationError("scores must be a finite floating vector")
        valid &= _range_mask(self.protocol, xyz)
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
        metrics: dict[str, float | int] = _point_metrics(
            np.concatenate(self._labels), np.concatenate(self._scores)
        )
        metrics.update(
            accepted_frames=self.accepted_frames,
            skipped_frames=self.skipped_frames,
        )
        return metrics


class DevelopmentMetricAccumulator:
    """Pool all valid synthetic-development points after complete fusion."""

    def __init__(self, protocol: _Protocol | None = None) -> None:
        self.protocol = protocol
        self._scores: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self.frames = 0

    def update(
        self, points: np.ndarray, scores: np.ndarray, raw_semantic: np.ndarray
    ) -> None:
        values = np.asarray(scores)
        anomaly, valid = binary_targets(raw_semantic)
        if values.shape != anomaly.shape or not np.isfinite(values).all():
            raise EvaluationError(
                "development scores and labels must be finite and aligned"
            )
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
        metrics: dict[str, float | int] = _point_metrics(labels, scores)
        metrics.update(
            frames=self.frames,
            points=int(labels.size),
            anomaly_points=int(labels.sum()),
            normal_points=int((~labels).sum()),
        )
        return metrics


class PositionDiagnostic:
    """Measure score calibration separately at each centered-window position."""

    def __init__(self, protocol: _Protocol | None = None) -> None:
        self.protocol = protocol
        self._scores: dict[int, list[np.ndarray]] = {
            value: [] for value in RELATIVE_TIMES
        }
        self._labels: dict[int, list[np.ndarray]] = {
            value: [] for value in RELATIVE_TIMES
        }

    def update(
        self,
        relative_time: int,
        points: np.ndarray,
        scores: np.ndarray,
        raw_semantic: np.ndarray,
    ) -> None:
        if relative_time not in RELATIVE_TIMES:
            raise EvaluationError("position diagnostic requires q=-2,-1,0,1,2")
        anomaly, valid = binary_targets(raw_semantic)
        values = _finite_vector("position scores", scores, anomaly.size)
        valid &= _range_mask(self.protocol, points)
        if np.any(valid):
            self._scores[relative_time].append(values[valid].astype(np.float64))
            self._labels[relative_time].append(anomaly[valid].astype(np.bool_))

    def compute(self) -> dict[str, dict[str, float | int | None]]:
        result: dict[str, dict[str, float | int | None]] = {}
        for relative_time in RELATIVE_TIMES:
            if not self._scores[relative_time]:
                result[str(relative_time)] = {"points": 0, "AP": None}
                continue
            scores = np.concatenate(self._scores[relative_time])
            labels = np.concatenate(self._labels[relative_time])
            normal = scores[~labels]
            anomaly = scores[labels]
            result[str(relative_time)] = {
                "points": int(scores.size),
                "normal_points": int(normal.size),
                "anomaly_points": int(anomaly.size),
                "AP": (
                    float(100.0 * average_precision_score(labels, scores))
                    if normal.size and anomaly.size
                    else None
                ),
                "normal_mean": float(normal.mean()) if normal.size else None,
                "anomaly_mean": float(anomaly.mean()) if anomaly.size else None,
                "standard_deviation": float(scores.std()),
                "q05": float(np.quantile(scores, 0.05)),
                "q50": float(np.quantile(scores, 0.50)),
                "q95": float(np.quantile(scores, 0.95)),
            }
        return result


class MovingNormalDiagnostic:
    """Report raw moving-label safety without exposing labels to the model."""

    def __init__(
        self,
        strict_threshold: float,
        protocol: _Protocol | None = None,
        moving_semantics: Iterable[int] = MOVING_NORMAL_SEMANTICS,
    ) -> None:
        if not math.isfinite(strict_threshold):
            raise EvaluationError("moving diagnostic threshold must be finite")
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
            raise EvaluationError("moving-normal semantics must be an integer vector")
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


class ObjectScaleDiagnostic:
    """Measure synthetic-object coherence, boundary leakage, and visibility."""

    def __init__(
        self,
        protocol: _Protocol | None = None,
        *,
        background_radius_m: float = 0.5,
    ) -> None:
        if not math.isclose(background_radius_m, 0.5, abs_tol=1.0e-12):
            raise EvaluationError("object boundary radius is fixed at 0.5 m")
        self.protocol = protocol
        self.background_radius_m = float(background_radius_m)
        self.records: list[dict[str, float | int | None]] = []
        self._visibility_scores: dict[int, list[np.ndarray]] = {
            value: [] for value in range(1, 6)
        }
        self._visibility_labels: dict[int, list[np.ndarray]] = {
            value: [] for value in range(1, 6)
        }

    def update_window(
        self,
        *,
        world_id: int,
        window_id: int,
        points: np.ndarray,
        scores: np.ndarray,
        object_ids: np.ndarray,
        relative_times: np.ndarray,
        raw_semantic: np.ndarray,
    ) -> None:
        xyz = np.asarray(points)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise EvaluationError("object-scale points must be finite [N,3]")
        values = _finite_vector("object-scale scores", scores, xyz.shape[0])
        objects = np.asarray(object_ids)
        times = np.asarray(relative_times)
        semantic = np.asarray(raw_semantic)
        count = xyz.shape[0]
        if (
            objects.shape != (count,)
            or times.shape != (count,)
            or semantic.shape != (count,)
            or not np.issubdtype(objects.dtype, np.integer)
            or not np.issubdtype(times.dtype, np.integer)
            or not np.issubdtype(semantic.dtype, np.integer)
        ):
            raise EvaluationError("object-scale labels must be aligned integer vectors")
        valid = _range_mask(self.protocol, xyz)
        # Renderer background uses -1; zero is also accepted for exported fixtures.
        normal = valid & (objects <= 0) & (semantic != 0) & (semantic != 2)
        normal_points = xyz[normal]
        normal_scores = values[normal]
        for object_id in sorted(int(item) for item in np.unique(objects) if item > 0):
            entity = objects == object_id
            entity_semantics = np.unique(semantic[entity])
            if 2 not in entity_semantics:
                # Normal controls have internal IDs but are not anomaly objects.
                continue
            if np.any(entity & (semantic != 2)):
                raise EvaluationError("one generated object mixes normal and anomaly labels")
            selected = valid & entity
            if not np.any(selected):
                continue
            object_scores = values[selected].astype(np.float64)
            visibility = int(np.unique(times[selected]).size)
            if visibility not in self._visibility_scores:
                raise EvaluationError("object visibility must lie in V=1..5")
            nearby = np.zeros(normal_points.shape[0], dtype=np.bool_)
            if normal_points.size:
                distance, _ = cKDTree(xyz[selected]).query(
                    normal_points,
                    k=1,
                    distance_upper_bound=self.background_radius_m,
                    workers=1,
                )
                nearby = np.isfinite(distance) & (distance <= self.background_radius_m)
            background_scores = normal_scores[nearby].astype(np.float64)
            self.records.append(
                {
                    "world_id": int(world_id),
                    "window_id": int(window_id),
                    "object_id": object_id,
                    "visibility": visibility,
                    "object_points": int(object_scores.size),
                    "score_mean": float(object_scores.mean()),
                    "score_variance": float(object_scores.var()),
                    "nearby_normal_points": int(background_scores.size),
                    "nearby_normal_mean": (
                        float(background_scores.mean())
                        if background_scores.size
                        else None
                    ),
                    "object_minus_background_mean": (
                        float(object_scores.mean() - background_scores.mean())
                        if background_scores.size
                        else None
                    ),
                }
            )
            self._visibility_scores[visibility].append(
                np.concatenate((object_scores, background_scores))
            )
            self._visibility_labels[visibility].append(
                np.concatenate(
                    (
                        np.ones(object_scores.size, dtype=np.bool_),
                        np.zeros(background_scores.size, dtype=np.bool_),
                    )
                )
            )

    def compute(self) -> dict[str, Any]:
        strata: dict[str, dict[str, float | int | None]] = {}
        for visibility in range(1, 6):
            if not self._visibility_scores[visibility]:
                strata[str(visibility)] = {"objects": 0, "AP": None}
                continue
            scores = np.concatenate(self._visibility_scores[visibility])
            labels = np.concatenate(self._visibility_labels[visibility])
            records = [
                item for item in self.records if item["visibility"] == visibility
            ]
            strata[str(visibility)] = {
                "objects": len(records),
                "points": int(scores.size),
                "AP": (
                    float(100.0 * average_precision_score(labels, scores))
                    if labels.any() and (~labels).any()
                    else None
                ),
                "mean_object_variance": float(
                    np.mean([float(item["score_variance"]) for item in records])
                ),
                "mean_nearby_normal_score": (
                    float(
                        np.mean(
                            [
                                float(item["nearby_normal_mean"])
                                for item in records
                                if item["nearby_normal_mean"] is not None
                            ]
                        )
                    )
                    if any(item["nearby_normal_mean"] is not None for item in records)
                    else None
                ),
            }
        return {
            "background_radius_m": self.background_radius_m,
            "objects": self.records,
            "visibility": strata,
        }


class EvaluationLedger:
    """Preserve seed-, development-world-, and public-sequence-level evidence."""

    def __init__(self, condition: str) -> None:
        self.condition = _condition(condition)
        self.seeds: dict[str, Mapping[str, Any]] = {}
        self.development_worlds: dict[str, Mapping[str, Any]] = {}
        self.public_sequences: dict[str, Mapping[str, Any]] = {}
        self.pooled: Mapping[str, Any] | None = None
        self.cost: Mapping[str, Any] | None = None

    @staticmethod
    def _add(
        destination: dict[str, Mapping[str, Any]],
        identity: int,
        value: Mapping[str, Any],
        name: str,
    ) -> None:
        key = str(int(identity))
        if key in destination:
            raise EvaluationError(f"duplicate {name} result {key}")
        destination[key] = dict(value)

    def add_seed(self, seed: int, value: Mapping[str, Any]) -> None:
        self._add(self.seeds, seed, value, "seed")

    def add_development_world(self, world_id: int, value: Mapping[str, Any]) -> None:
        if not 0 <= int(world_id) < 24:
            raise EvaluationError("checkpoint-selection worlds must be IDs 0..23")
        self._add(self.development_worlds, world_id, value, "development world")

    def add_public_sequence(self, sequence_id: int, value: Mapping[str, Any]) -> None:
        self._add(self.public_sequences, sequence_id, value, "public sequence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "seeds": self.seeds,
            "development_worlds": self.development_worlds,
            "public_sequences": self.public_sequences,
            "pooled": None if self.pooled is None else dict(self.pooled),
            "cost": None if self.cost is None else dict(self.cost),
        }


def select_checkpoint(
    candidates: Iterable[Mapping[str, Any]],
    selection_key: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Any, ...]],
) -> Mapping[str, Any]:
    """Select from in-generator and pure-normal evidence only."""

    ranked: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    for candidate in candidates:
        if "generator_held_out" in candidate or "held_out" in candidate:
            raise EvaluationError("held-out generator evidence cannot enter selection")
        in_generator = candidate.get("in_generator")
        pure_normal = candidate.get("pure_normal")
        if not isinstance(in_generator, Mapping) or not isinstance(
            pure_normal, Mapping
        ):
            raise EvaluationError(
                "selection requires in-generator and pure-normal evidence"
            )
        ranked.append((selection_key(in_generator, pure_normal), candidate))
    if not ranked:
        raise EvaluationError("checkpoint selection received no candidate")
    return max(ranked, key=lambda item: item[0])[1]


@dataclass(slots=True)
class _FrameFusion:
    slot_to_ray: np.ndarray
    sorted_rays: np.ndarray
    sorted_slots: np.ndarray
    score_sum: np.ndarray
    score_count: np.ndarray


def _packed_rays(name: str, rays: np.ndarray, count: int | None = None) -> np.ndarray:
    values = np.asarray(rays)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype")
    if values.ndim == 1:
        if count is not None and values.shape != (count,):
            raise EvaluationError(f"{name} must contain {count} rays")
        if np.any(values < 0):
            raise EvaluationError(f"{name} cannot contain a negative ray")
        return values.astype(np.uint64, copy=False)
    if values.ndim != 2 or values.shape[1] != 2:
        raise EvaluationError(f"{name} must be [N] or [N,2]")
    if count is not None and values.shape[0] != count:
        raise EvaluationError(f"{name} must contain {count} rays")
    if np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max):
        raise EvaluationError(f"{name} components must be uint32-compatible")
    unsigned = values.astype(np.uint64, copy=False)
    return (unsigned[:, 0] << np.uint64(32)) | unsigned[:, 1]


class WindowScoreFusion:
    """Average probabilities by canonical frame-ray identity."""

    def __init__(self, maximum_count: int = 5) -> None:
        if maximum_count < 1:
            raise ValueError("maximum_count must be positive")
        self.maximum_count = maximum_count
        self._frames: dict[int, _FrameFusion] = {}

    def add(
        self,
        frame_id: int,
        canonical_ray_ids: np.ndarray,
        probabilities: np.ndarray,
        *,
        output_slots: np.ndarray,
        slot_to_ray: np.ndarray,
    ) -> None:
        mapping = _packed_rays("slot_to_ray", slot_to_ray)
        slot_count = int(mapping.size)
        if np.unique(mapping).size != slot_count:
            raise EvaluationError("slot_to_ray must map each slot to one unique ray")
        rays = _packed_rays("canonical_ray_ids", canonical_ray_ids)
        slots = np.asarray(output_slots)
        values = np.asarray(probabilities)
        if slots.ndim != 1 or not np.issubdtype(slots.dtype, np.integer):
            raise TypeError("output_slots must be an integer vector")
        if rays.shape != slots.shape:
            raise EvaluationError("canonical rays and output slots must align")
        if values.shape != rays.shape or not np.issubdtype(values.dtype, np.floating):
            raise TypeError("probabilities must be floating and aligned with rays")
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise EvaluationError("fusion requires finite post-sigmoid probabilities")
        if np.unique(rays).size != rays.size:
            raise EvaluationError("one window cannot repeat a frame-ray identity")
        if np.any(slots < 0) or np.any(slots >= slot_count):
            raise EvaluationError("output slot lies outside the source frame")
        frame = self._frames.get(int(frame_id))
        if frame is None:
            order = np.argsort(mapping, kind="stable")
            frame = _FrameFusion(
                mapping.copy(),
                mapping[order],
                order.astype(np.int64, copy=False),
                np.zeros(slot_count, dtype=np.float64),
                np.zeros(slot_count, dtype=np.uint8),
            )
            self._frames[int(frame_id)] = frame
        elif not np.array_equal(frame.slot_to_ray, mapping):
            raise EvaluationError("slot-to-ray mapping changed for an existing frame")
        positions = np.searchsorted(frame.sorted_rays, rays)
        found = positions < frame.sorted_rays.size
        found[found] &= frame.sorted_rays[positions[found]] == rays[found]
        if not np.all(found):
            raise EvaluationError("canonical ray is absent from slot_to_ray")
        resolved_slots = frame.sorted_slots[positions]
        if not np.array_equal(resolved_slots, slots.astype(np.int64, copy=False)):
            raise EvaluationError("output slots disagree with the explicit ray mapping")
        next_count = frame.score_count[slots].astype(np.int16) + 1
        if np.any(next_count > self.maximum_count):
            raise EvaluationError("a frame-ray received too many window predictions")
        frame.score_sum[slots] += values.astype(np.float64)
        frame.score_count[slots] = next_count.astype(np.uint8)

    def finalize(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            frame = self._frames[int(frame_id)]
        except KeyError as exc:
            raise EvaluationError(f"frame {frame_id} has no predictions") from exc
        scores = np.zeros(frame.score_sum.shape, dtype=np.float32)
        seen = frame.score_count > 0
        scores[seen] = (frame.score_sum[seen] / frame.score_count[seen]).astype(
            np.float32
        )
        return scores, frame.score_count.copy()

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._frames))


class RollingCache:
    """A small deterministic LRU cache for rendered frames or frozen features."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._values: OrderedDict[int, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: int, factory: Callable[[int], Any]) -> Any:
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
class SequencePrediction:
    """One condition's full sequence predictions and measured inference cost."""

    condition: str
    frames: Mapping[int, tuple[np.ndarray, np.ndarray]]
    cost: Mapping[str, Any]
    position_diagnostic: Mapping[str, Any] | None = None


class AJAEInference:
    """Run the exact B0--B5 inference rule selected for one checkpoint."""

    def __init__(
        self,
        model: nn.Module | None,
        encoder: nn.Module,
        *,
        condition: str,
        slot_to_ray: Callable[[object], np.ndarray],
        ray_mapping_digest: str,
        device: torch.device | str = "cuda",
        cache_frames: int = 7,
        time_budget_seconds: float | None = None,
        position_diagnostic: PositionDiagnostic | None = None,
    ) -> None:
        self.condition = _condition(condition)
        if self.condition != "B0" and model is None:
            raise EvaluationError(f"{self.condition} requires an AJAE model")
        if not callable(slot_to_ray):
            raise TypeError("slot_to_ray must be an explicit mapping callback")
        if (
            not isinstance(ray_mapping_digest, str)
            or len(ray_mapping_digest) != 64
        ):
            raise ValueError("ray_mapping_digest must identify the loaded calibration")
        if cache_frames < (5 if self.condition in {"B2", "B3", "B4", "B5"} else 1):
            raise ValueError(
                "inference cache is smaller than the selected input window"
            )
        if time_budget_seconds is not None and (
            not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0.0
        ):
            raise ValueError("time budget must be finite and positive")
        self.model = model
        self.encoder = encoder
        self.slot_to_ray = slot_to_ray
        self.ray_mapping_digest = ray_mapping_digest
        self.device = torch.device(device)
        self.cache = RollingCache(cache_frames)
        self.time_budget_seconds = time_budget_seconds
        self.position_diagnostic = position_diagnostic
        self._ray_maps: dict[int, np.ndarray] = {}
        if self.model is not None:
            self.model.to(self.device).eval()
        self.encoder.to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def _encode(self, source: object) -> object:
        frame_id = int(getattr(source, "frame_id"))

        def factory(_: int) -> object:
            with torch.no_grad():
                return self.encoder(
                    getattr(source, "coordinates"),
                    getattr(source, "features"),
                    getattr(source, "real_slots"),
                )

        encoding = self.cache.get(frame_id, factory)
        encoded_slots = getattr(encoding, "real_slots")
        if isinstance(encoded_slots, torch.Tensor):
            encoded_slots = encoded_slots.detach().cpu().numpy()
        if not np.array_equal(
            np.asarray(encoded_slots, dtype=np.int64),
            np.asarray(getattr(source, "real_slots"), dtype=np.int64),
        ):
            raise EvaluationError("STU encoding changed visible-return order")
        return encoding

    def _source_ray_map(self, source: object) -> np.ndarray:
        frame_id = int(getattr(source, "frame_id"))
        mapping = self._ray_maps.get(frame_id)
        if mapping is None:
            raw = np.asarray(self.slot_to_ray(source))
            _packed_rays(
                "slot_to_ray callback result", raw, int(getattr(source, "slot_count"))
            )
            mapping = raw.copy()
            self._ray_maps[frame_id] = mapping
        return mapping

    def _audited_window(self, sequence: object, anchor: int, condition: str) -> object:
        offsets = (-4, -3, -2, -1, 0) if condition == "B5" else RELATIVE_TIMES
        mappings = {
            frame_id: self._source_ray_map(sequence.source_frame(frame_id))
            for frame_id in (anchor + offset for offset in offsets)
        }
        window = sequence.window(
            anchor,
            condition=condition,
            canonical_ray_by_slot=mappings,
            ray_mapping_audited=True,
            ray_mapping_digest=self.ray_mapping_digest,
        )
        if getattr(getattr(window, "points"), "ray_mapping_audited", None) is not True:
            raise EvaluationError("scene window lacks an audited canonical ray mapping")
        return window

    @staticmethod
    def _condition_anchors(sequence: object, condition: str) -> tuple[int, ...]:
        offsets = (-4, -3, -2, -1, 0) if condition == "B5" else RELATIVE_TIMES
        frame_ids = frozenset(AJAEInference._all_frame_ids(sequence))
        excluded = frozenset(
            int(value)
            for value in getattr(
                getattr(sequence, "spec", object()), "excluded_source_frames", ()
            )
        )
        return tuple(
            anchor
            for anchor in sorted(frame_ids)
            if all(
                anchor + offset in frame_ids and anchor + offset not in excluded
                for offset in offsets
            )
        )

    @staticmethod
    def _all_frame_ids(sequence: object) -> tuple[int, ...]:
        direct = getattr(sequence, "frame_ids", None)
        if direct is not None:
            return tuple(int(value) for value in direct)
        spec = getattr(sequence, "spec")
        direct = getattr(spec, "frame_ids", None)
        if direct is not None:
            return tuple(int(value) for value in direct)
        span = getattr(spec, "frame_span", getattr(spec, "span", None))
        if span is not None:
            first = int(getattr(span, "first", getattr(span, "start", 0)))
            if hasattr(span, "stop"):
                return tuple(range(first, int(getattr(span, "stop"))))
            return tuple(range(first, int(getattr(span, "last")) + 1))
        first = int(getattr(spec, "first_frame"))
        last = int(getattr(spec, "last_frame"))
        return tuple(range(first, last + 1))

    @staticmethod
    def _usable_frame_ids(sequence: object) -> tuple[int, ...]:
        excluded = frozenset(
            int(value)
            for value in getattr(
                getattr(sequence, "spec", object()), "excluded_source_frames", ()
            )
        )
        return tuple(
            frame_id
            for frame_id in AJAEInference._all_frame_ids(sequence)
            if frame_id not in excluded
        )

    @staticmethod
    def _comparison_frame_ids(sequence: object) -> tuple[int, ...]:
        """Use one padding-free frame domain shared by every B0--B5 result."""

        centered = frozenset(AJAEInference._condition_anchors(sequence, "B3"))
        causal = frozenset(AJAEInference._condition_anchors(sequence, "B5"))
        frames = tuple(sorted(centered & causal))
        if not frames:
            raise EvaluationError(
                "sequence has no frame shared by complete centered and causal windows"
            )
        return frames

    def _model_probabilities(
        self,
        sources: list[object],
        coordinates: np.ndarray,
        relative_times: np.ndarray,
        *,
        cross_frame_enabled: bool,
    ) -> np.ndarray:
        if self.model is None:
            raise EvaluationError("AJAE model is unavailable")
        encodings = [self._encode(source) for source in sources]

        def concatenate(name: str) -> torch.Tensor:
            return torch.cat([getattr(item, name) for item in encodings], dim=0).to(
                self.device
            )

        intensity = torch.cat(
            [
                torch.tensor(
                    np.asarray(getattr(source, "xyzi"))[
                        np.asarray(getattr(source, "real_slots")), 3
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                for source in sources
            ]
        )
        logits = self.model(
            torch.tensor(coordinates, device=self.device, dtype=torch.float32),
            torch.tensor(relative_times, device=self.device, dtype=torch.long),
            concatenate("point_features"),
            concatenate("normal_evidence"),
            concatenate("reliability_assign"),
            concatenate("reliability_noobj"),
            intensity,
            cross_frame_enabled=cross_frame_enabled,
        )
        count = int(np.asarray(coordinates).shape[0])
        if logits.shape != (count,) or not bool(torch.isfinite(logits).all()):
            raise EvaluationError("model returned invalid point logits")
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    def _add_probabilities(
        self,
        fusion: WindowScoreFusion,
        source: object,
        probabilities: np.ndarray,
        canonical_rays: np.ndarray | None = None,
    ) -> None:
        slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
        mapping = self._source_ray_map(source)
        rays = mapping[slots] if canonical_rays is None else canonical_rays
        fusion.add(
            int(getattr(source, "frame_id")),
            rays,
            probabilities,
            output_slots=slots,
            slot_to_ray=mapping,
        )

    def _update_position(
        self, source: object, relative_time: int, probabilities: np.ndarray
    ) -> None:
        if self.position_diagnostic is None:
            return
        labels = getattr(source, "labels", None)
        if labels is None:
            return
        slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
        self.position_diagnostic.update(
            relative_time,
            np.asarray(getattr(source, "xyzi"))[slots, :3],
            probabilities,
            np.asarray(getattr(labels, "semantic"))[slots],
        )

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
        """Return full-slot scores/counts and optionally write official text files."""

        self.cache.clear()
        self._ray_maps.clear()
        fusion = WindowScoreFusion(maximum_count=5 if self.condition == "B4" else 1)
        direct: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        durations: list[float] = []
        scored_points = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._synchronize(self.device)
        started = time.perf_counter()

        def timed(call: Callable[[], int]) -> None:
            nonlocal scored_points
            self._synchronize(self.device)
            unit_started = time.perf_counter()
            scored_points += call()
            self._synchronize(self.device)
            durations.append(time.perf_counter() - unit_started)

        comparison_frames = self._comparison_frame_ids(sequence)
        comparison_set = frozenset(comparison_frames)
        with torch.inference_mode():
            if self.condition == "B0":
                for frame_id in comparison_frames:
                    source = sequence.source_frame(frame_id)

                    def score_b0(source: object = source) -> int:
                        encoding = self._encode(source)
                        values = (
                            getattr(encoding, "maxlogit_score")
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        slots = np.asarray(
                            getattr(source, "real_slots"), dtype=np.int64
                        )
                        scores = np.asarray(source.restore_real(values)).copy()
                        counts = np.zeros(
                            int(getattr(source, "slot_count")), dtype=np.uint8
                        )
                        counts[slots] = 1
                        direct[int(getattr(source, "frame_id"))] = scores, counts
                        self._update_position(source, 0, values)
                        return int(values.size)

                    timed(score_b0)
            elif self.condition == "B1":
                for frame_id in comparison_frames:
                    source = sequence.source_frame(frame_id)

                    def score_b1(source: object = source) -> int:
                        slots = np.asarray(
                            getattr(source, "real_slots"), dtype=np.int64
                        )
                        coordinates = np.asarray(getattr(source, "xyzi"))[slots, :3]
                        relative_times = np.zeros(slots.size, dtype=np.int8)
                        probabilities = self._model_probabilities(
                            [source],
                            coordinates,
                            relative_times,
                            cross_frame_enabled=False,
                        )
                        self._add_probabilities(fusion, source, probabilities)
                        self._update_position(source, 0, probabilities)
                        return int(probabilities.size)

                    timed(score_b1)
            elif self.condition in {"B2", "B3", "B4"}:
                centers = self._condition_anchors(sequence, self.condition)
                if not centers:
                    raise EvaluationError("sequence has no legal centered window")
                for center in centers:
                    if self.condition != "B4" and center not in comparison_set:
                        continue
                    window = self._audited_window(sequence, center, self.condition)

                    def score_centered(window: object = window) -> int:
                        frames = list(getattr(window, "frames"))
                        sources = [getattr(frame, "source") for frame in frames]
                        points = getattr(window, "points")
                        probabilities = self._model_probabilities(
                            sources,
                            np.asarray(getattr(points, "coordinates_center")),
                            np.asarray(getattr(points, "relative_time")),
                            cross_frame_enabled=self.condition != "B2",
                        )
                        selected = range(5) if self.condition == "B4" else (2,)
                        for local_frame, source in enumerate(sources):
                            point_slice = points.frame_slice(local_frame)
                            frame_probabilities = probabilities[point_slice]
                            self._update_position(
                                source, RELATIVE_TIMES[local_frame], frame_probabilities
                            )
                            if local_frame in selected:
                                canonical_rays = getattr(points, "source_ray", None)
                                rays = (
                                    None
                                    if canonical_rays is None
                                    else np.asarray(canonical_rays)[point_slice]
                                )
                                self._add_probabilities(
                                    fusion, source, frame_probabilities, rays
                                )
                        return int(probabilities.size)

                    timed(score_centered)
            else:
                currents = tuple(
                    frame_id
                    for frame_id in self._condition_anchors(sequence, "B5")
                    if frame_id in comparison_set
                )
                if not currents:
                    raise EvaluationError("sequence has no legal causal window")
                for current in currents:
                    window = self._audited_window(sequence, current, "B5")

                    def score_causal(window: object = window) -> int:
                        frames = list(getattr(window, "frames"))
                        sources = [getattr(frame, "source") for frame in frames]
                        points = getattr(window, "points")
                        probabilities = self._model_probabilities(
                            sources,
                            np.asarray(getattr(points, "coordinates_center")),
                            np.asarray(getattr(points, "relative_time")),
                            cross_frame_enabled=True,
                        )
                        current_probabilities: np.ndarray | None = None
                        for local_frame, (model_time, source) in enumerate(
                            zip(RELATIVE_TIMES, sources, strict=True)
                        ):
                            point_slice = points.frame_slice(local_frame)
                            frame_probabilities = probabilities[point_slice]
                            self._update_position(
                                source, model_time, frame_probabilities
                            )
                            if model_time == 2:
                                current_probabilities = frame_probabilities
                        assert current_probabilities is not None
                        self._add_probabilities(
                            fusion,
                            sources[-1],
                            current_probabilities,
                            np.asarray(getattr(points, "source_ray"))[
                                points.frame_slice(4)
                            ],
                        )
                        return int(probabilities.size)

                    timed(score_causal)

        self._synchronize(self.device)
        elapsed = time.perf_counter() - started

        destination = None if output_dir is None else Path(output_dir)
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for frame_id in comparison_frames:
            source = sequence.source_frame(frame_id)
            if frame_id in direct:
                scores, counts = direct[frame_id]
            elif frame_id in fusion.frame_ids:
                scores, counts = fusion.finalize(frame_id)
            else:
                raise EvaluationError(
                    f"{self.condition} produced no score on comparison frame {frame_id}"
                )
            # Empty rays always retain the required zero score.
            scores[np.asarray(source.zero_slot_mask)] = 0.0
            counts[np.asarray(source.zero_slot_mask)] = 0
            result[frame_id] = scores, counts
            if destination is not None:
                write_point_scores(destination / f"{frame_id:06d}.txt", scores)
        if destination is not None:
            save_result(
                destination / "coverage.json",
                {
                    "format": "ajae-prediction-coverage-v1",
                    "condition": self.condition,
                    "frame_domain": (
                        "intersection_of_complete_centered_q0_and_"
                        "complete_causal_current_frames"
                    ),
                    "frame_ids": list(comparison_frames),
                    "padding_or_zero_fill_used": False,
                },
            )
        cache_total = self.cache.hits + self.cache.misses
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        cost: dict[str, Any] = {
            "latency_seconds": {
                "total": elapsed,
                "mean_per_unit": float(np.mean(durations)) if durations else 0.0,
                "p95_per_unit": (
                    float(np.quantile(durations, 0.95)) if durations else 0.0
                ),
                "units": len(durations),
            },
            "peak_memory_bytes": peak_memory,
            "throughput_points_per_second": scored_points / max(elapsed, 1.0e-12),
            "scored_points": scored_points,
            "stu_cache": {
                "capacity_frames": self.cache.capacity,
                "hits": self.cache.hits,
                "misses": self.cache.misses,
                "hit_rate": self.cache.hits / cache_total if cache_total else 0.0,
            },
            "temporal_input_frames": 1 if self.condition in {"B0", "B1"} else 5,
            "comparison_frames": len(comparison_frames),
            "time_budget_seconds": self.time_budget_seconds,
            "time_budget_exceeded": (
                elapsed > self.time_budget_seconds
                if self.time_budget_seconds is not None
                else False
            ),
        }
        diagnostic = (
            self.position_diagnostic.compute()
            if self.position_diagnostic is not None
            else None
        )
        return SequencePrediction(self.condition, result, cost, diagnostic)


def write_point_scores(path: Path | str, scores: np.ndarray) -> None:
    """Write one finite confidence per original file slot for official tools."""

    destination = Path(path)
    values = np.asarray(scores)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
        raise TypeError("scores must be a floating vector")
    if not np.isfinite(values).all():
        raise EvaluationError("scores must be finite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, values.astype(np.float32), fmt="%.9g")


def load_prediction_coverage(
    directory: Path | str,
    *,
    condition: str,
    expected_frame_ids: Iterable[int],
) -> tuple[int, ...]:
    """Validate the no-padding frame manifest before reading any score file."""

    path = Path(directory) / "coverage.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read prediction coverage: {path}") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "format",
        "condition",
        "frame_domain",
        "frame_ids",
        "padding_or_zero_fill_used",
    }:
        raise EvaluationError("prediction coverage has an invalid schema")
    if (
        payload["format"] != "ajae-prediction-coverage-v1"
        or payload["condition"] != _condition(condition)
        or payload["frame_domain"]
        != "intersection_of_complete_centered_q0_and_complete_causal_current_frames"
        or payload["padding_or_zero_fill_used"] is not False
    ):
        raise EvaluationError("prediction coverage changed the frozen frame domain")
    raw_frames = payload["frame_ids"]
    if not isinstance(raw_frames, list) or any(
        type(value) is not int for value in raw_frames
    ):
        raise EvaluationError("prediction coverage frame_ids must be an integer list")
    frames = tuple(raw_frames)
    expected = tuple(int(value) for value in expected_frame_ids)
    if frames != expected:
        raise EvaluationError("prediction coverage does not match the source sequence")
    return frames


def write_packed_labels(path: Path | str, packed: np.ndarray) -> None:
    """Write one little-endian packed label per original file slot."""

    destination = Path(path)
    values = np.asarray(packed)
    if values.dtype != np.uint32 or values.ndim != 1:
        raise TypeError("packed labels must be uint32[N]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    values.astype("<u4", copy=False).tofile(destination)


def scores_to_packed_instances(
    points: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    eps: float = 1.0,
    min_samples: int = 1,
    protocol: _Protocol | None = None,
) -> np.ndarray:
    """Cluster one frame only and pack semantic/instance IDs as uint32."""

    xyz = np.asarray(points)
    values = np.asarray(scores)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or values.shape != (xyz.shape[0],):
        raise EvaluationError("points and scores are not aligned")
    if not math.isfinite(float(threshold)) or eps <= 0.0 or min_samples < 1:
        raise EvaluationError("invalid clustering parameters")
    # The protocol clusters every thresholded point first; the official object
    # evaluator applies its range mask only after instance construction.
    selected = values > threshold
    semantic = np.zeros(values.size, dtype=np.uint32)
    instance = np.zeros(values.size, dtype=np.uint32)
    indices = np.flatnonzero(selected)
    if indices.size:
        clusters = DBSCAN(
            eps=float(eps), min_samples=int(min_samples), n_jobs=-1, leaf_size=100
        ).fit_predict(xyz[indices, :3])
        retained = clusters >= 0
        semantic[indices[retained]] = 1
        instance[indices[retained]] = clusters[retained].astype(np.uint32) + 1
    return semantic | (instance << np.uint32(16))


class ObjectMetricAccumulator:
    """Reproduce the official single-anomaly-class UQ/PQ bookkeeping."""

    def __init__(self, protocol: _Protocol | None = None) -> None:
        self.protocol = protocol
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.accepted_frames = 0

    def update(
        self,
        points: np.ndarray,
        packed_prediction: np.ndarray,
        packed_target: np.ndarray,
    ) -> bool:
        prediction = np.asarray(packed_prediction, dtype=np.uint32)
        target = np.asarray(packed_target, dtype=np.uint32)
        if prediction.ndim != 1 or target.shape != prediction.shape:
            raise EvaluationError("packed prediction and target must be aligned")
        pred_sem = prediction & np.uint32(0xFFFF)
        pred_inst = prediction >> np.uint32(16)
        raw_sem = target & np.uint32(0xFFFF)
        raw_inst = target >> np.uint32(16)
        anomaly, valid = binary_targets(raw_sem)
        valid &= _range_mask(self.protocol, np.asarray(points))
        _, _, minimum = _evaluation_values(self.protocol)
        if int(anomaly[valid].sum()) < minimum:
            return False
        self.accepted_frames += 1
        pred_sem = pred_sem[valid]
        pred_inst = pred_inst[valid] + 1
        gt_sem = anomaly[valid].astype(np.uint32)
        gt_inst = raw_inst[valid] + 1

        pred_ids, pred_area = np.unique(pred_inst[pred_sem == 1], return_counts=True)
        gt_ids, gt_area = np.unique(gt_inst[gt_sem == 1], return_counts=True)
        pred_index = {int(value): index for index, value in enumerate(pred_ids)}
        gt_index = {int(value): index for index, value in enumerate(gt_ids)}
        matched_pred = np.zeros(pred_ids.size, dtype=bool)
        matched_gt = np.zeros(gt_ids.size, dtype=bool)
        overlap = (pred_sem == 1) & (gt_sem == 1)
        if np.any(overlap):
            pairs = pred_inst[overlap].astype(np.uint64) + (
                np.uint64(2**32) * gt_inst[overlap].astype(np.uint64)
            )
            unique_pairs, intersections = np.unique(pairs, return_counts=True)
            pair_gt = unique_pairs // np.uint64(2**32)
            pair_pred = unique_pairs % np.uint64(2**32)
            gt_areas = np.asarray([gt_area[gt_index[int(x)]] for x in pair_gt])
            pred_areas = np.asarray([pred_area[pred_index[int(x)]] for x in pair_pred])
            pair_iou = intersections / (gt_areas + pred_areas - intersections)
            matches = pair_iou > 0.5
            self.tp += int(matches.sum())
            self.iou += float(pair_iou[matches].sum())
            for value in pair_gt[matches]:
                matched_gt[gt_index[int(value)]] = True
            for value in pair_pred[matches]:
                matched_pred[pred_index[int(value)]] = True
        self.fn += int(np.sum((gt_area >= minimum) & ~matched_gt))
        self.fp += int(np.sum((pred_area >= minimum) & ~matched_pred))
        return True

    def compute(self) -> dict[str, float | int]:
        sq = self.iou / max(float(self.tp), 1.0e-15)
        recall = self.tp / max(float(self.tp + self.fn), 1.0e-15)
        rq = self.tp / max(float(self.tp + 0.5 * self.fp + 0.5 * self.fn), 1.0e-15)
        return {
            "SQ": 100.0 * sq,
            "RecallQ": 100.0 * recall,
            "UQ": 100.0 * sq * recall,
            "RQ": 100.0 * rq,
            "PQ": 100.0 * sq * rq,
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "accepted_frames": self.accepted_frames,
        }


def evaluate_frames(
    frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    protocol: _Protocol | None = None,
) -> dict[str, float | int]:
    """Evaluate an iterable of (xyz, scores, raw semantic) frames."""

    accumulator = PointMetricAccumulator(protocol)
    for points, scores, raw_semantic in frames:
        accumulator.update(points, scores, raw_semantic)
    return accumulator.compute()


def save_result(path: Path | str, result: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _load_json_record(path: Path | str, expected_format: str) -> Mapping[str, Any]:
    try:
        source = Path(path).expanduser().resolve(strict=True)
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid JSON record: {path}") from error
    if not isinstance(value, Mapping) or value.get("format") != expected_format:
        raise EvaluationError(f"unsupported record format: {source}")
    return value


def _validate_method_freeze(
    path: Path | str, protocol: object, condition: str
) -> Mapping[str, Any]:
    try:
        from .protocol import load_development_worlds
    except ImportError:  # pragma: no cover - direct script execution
        from protocol import load_development_worlds

    record = _load_json_record(path, METHOD_FREEZE_FORMAT)
    protocol_path = getattr(protocol, "path")
    calibration_path = getattr(protocol, "sensor_calibration_path")()
    development_path = getattr(protocol, "development_worlds_path")()
    selection = getattr(protocol, "development")["checkpoint_selection"]
    evaluation = getattr(protocol, "evaluation")
    evaluation_document = getattr(protocol, "evaluation_document")
    frame_domain = evaluation_document["comparison_frame_domain"]
    development_worlds = load_development_worlds(
        development_path,
        protocol=protocol,
    )
    fixed_world_evaluation = getattr(protocol, "development")[
        "fixed_world_evaluation"
    ]
    gate_criteria = getattr(protocol, "decision_gates")["criteria"]
    maximum_worlds = getattr(protocol, "training").get("maximum_worlds")
    try:
        threshold = float(record.get("object_score_threshold"))
    except (TypeError, ValueError):
        threshold = math.nan
    pure_normal = record.get("pure_normal_201")
    development_sequence = getattr(protocol, "development_sequence")
    development_span = getattr(development_sequence, "span")
    if development_span is None:
        raise EvaluationError("development sequence needs a fixed source span")
    excluded_development_frames = set(
        getattr(development_sequence, "excluded_source_frames")
    )
    usable_development_frames = set(
        range(int(development_span.start), int(development_span.stop))
    ) - excluded_development_frames
    centered_frames = set(development_sequence.center_frames())
    causal_frames = {
        frame_id
        for frame_id in usable_development_frames
        if all(
            frame_id + offset in usable_development_frames
            for offset in (-4, -3, -2, -1, 0)
        )
    }
    expected_pure_normal_frames = len(centered_frames & causal_frames)
    pure_normal_valid = False
    if isinstance(pure_normal, Mapping):
        try:
            pure_threshold = float(pure_normal.get("strict_threshold"))
            target_alarm = float(pure_normal.get("target_alarm_rate"))
            observed_alarm = float(pure_normal.get("observed_strict_alarm_rate"))
            normal_points = int(pure_normal.get("points"))
            normal_frames = int(pure_normal.get("frames"))
        except (TypeError, ValueError):
            pass
        else:
            pure_normal_valid = (
                math.isfinite(pure_threshold)
                and math.isclose(pure_threshold, threshold, rel_tol=0.0, abs_tol=0.0)
                and math.isclose(
                    target_alarm,
                    float(getattr(protocol, "evaluation").normal_point_alarm_rate),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and 0.0 <= observed_alarm <= target_alarm
                and normal_points > 0
                and normal_frames == expected_pure_normal_frames
            )
    if (
        record.get("protocol_schema") != int(getattr(protocol, "schema_version"))
        or record.get("condition") != condition
        or record.get("frozen") is not True
        or record.get("protocol_sha256") != _sha256_file(protocol_path)
        or record.get("calibration_sha256") != _sha256_file(calibration_path)
        or record.get("development_worlds_sha256")
        != _sha256_file(development_path)
        or record.get("comparison_frame_domain")
        != "intersection_of_complete_centered_q0_and_complete_causal_current_frames"
        or record.get("padding_or_zero_fill_used") is not False
        or frame_domain.get("status") != "frozen_before_evaluation"
        or selection["status"] != "frozen_before_training"
        or fixed_world_evaluation.get("status") != "frozen_before_training"
        or not isinstance(fixed_world_evaluation.get("scope"), Mapping)
        or gate_criteria.get("status") != "frozen_before_training"
        or any(
            not isinstance(gate_criteria.get(name), Mapping)
            for name in ("gate1", "gate2", "gate3", "gate4")
        )
        or type(maximum_worlds) is not int
        or development_worlds.validated is not True
        or record.get("checkpoint_selection") != dict(selection)
        or not 0.0 <= threshold <= 1.0
        or record.get("dbscan_eps_m") != evaluation.dbscan_eps_m
        or record.get("dbscan_min_samples") != evaluation.dbscan_min_samples
        or record.get("score_fusion")
        != "equal_mean_of_probabilities_by_frame_and_canonical_ray"
        or not pure_normal_valid
    ):
        raise EvaluationError(
            "method-freeze record does not bind this protocol, renderer, and frame domain"
        )
    return record


def _condition_paths(values: Iterable[str], *, name: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        condition, separator, raw_path = str(value).partition("=")
        if not separator or condition not in CONDITIONS or not raw_path:
            raise EvaluationError(
                f"each {name} must use CONDITION=/path/to/record.json"
            )
        if condition in paths:
            raise EvaluationError(f"duplicate {name} for {condition}")
        try:
            paths[condition] = Path(raw_path).expanduser().resolve(strict=True)
        except OSError as error:
            raise EvaluationError(
                f"{name} path does not exist for {condition}"
            ) from error
    return paths


def _finite_json_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationError(f"{name} must be finite")
    return result


def _nonnegative_json_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise EvaluationError(f"{name} must be a non-negative integer")
    return value


def _validate_public_point_metrics(
    value: object,
    *,
    expected_frames: int,
    name: str,
) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{name} point metrics must be a mapping")
    accepted = _nonnegative_json_integer(
        value.get("accepted_frames"), f"{name}.accepted_frames"
    )
    skipped = _nonnegative_json_integer(
        value.get("skipped_frames"), f"{name}.skipped_frames"
    )
    if accepted < 1 or accepted + skipped != expected_frames:
        raise EvaluationError(f"{name} does not cover its complete comparison domain")
    for metric in ("AP", "AUROC", "FPR95"):
        number = _finite_json_number(value.get(metric), f"{name}.{metric}")
        if not 0.0 <= number <= 100.0:
            raise EvaluationError(f"{name}.{metric} lies outside [0,100]")
    threshold = _finite_json_number(value.get("threshold"), f"{name}.threshold")
    if not 0.0 <= threshold <= 1.0:
        raise EvaluationError(f"{name}.threshold lies outside [0,1]")
    return accepted, skipped


def _validate_public_moving_metrics(
    value: object,
    *,
    threshold: float,
    name: str,
) -> tuple[int, int, float | None, float | None, float | None, float | None]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{name} moving-normal metrics must be a mapping")
    recorded_threshold = _finite_json_number(
        value.get("strict_threshold"), f"{name}.strict_threshold"
    )
    if not math.isclose(recorded_threshold, threshold, rel_tol=0.0, abs_tol=0.0):
        raise EvaluationError(f"{name} uses a different normal-alarm threshold")
    moving_points = _nonnegative_json_integer(
        value.get("moving_points"), f"{name}.moving_points"
    )
    static_points = _nonnegative_json_integer(
        value.get("static_points"), f"{name}.static_points"
    )

    def optional_rate(field: str, count: int) -> float | None:
        raw = value.get(field)
        if count == 0:
            if raw is not None:
                raise EvaluationError(f"{name}.{field} must be null for zero points")
            return None
        result = _finite_json_number(raw, f"{name}.{field}")
        if not 0.0 <= result <= 1.0:
            raise EvaluationError(f"{name}.{field} lies outside [0,1]")
        return result

    moving_mean = optional_rate("moving_mean", moving_points)
    moving_fpr = optional_rate("moving_false_positive_rate", moving_points)
    static_mean = optional_rate("static_mean", static_points)
    static_fpr = optional_rate("static_false_positive_rate", static_points)
    difference = value.get("moving_minus_static_mean")
    if moving_mean is None or static_mean is None:
        if difference is not None:
            raise EvaluationError(
                f"{name}.moving_minus_static_mean must be null without both classes"
            )
    elif not math.isclose(
        _finite_json_number(difference, f"{name}.moving_minus_static_mean"),
        moving_mean - static_mean,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise EvaluationError(f"{name} moving/static means are inconsistent")
    return (
        moving_points,
        static_points,
        moving_mean,
        moving_fpr,
        static_mean,
        static_fpr,
    )


def _validate_public_result(
    path: Path,
    *,
    protocol: object,
    condition: str,
) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid public validation result: {path}") from error
    if not isinstance(value, Mapping):
        raise EvaluationError(f"public validation result is not a mapping: {path}")
    expected_sequences = {
        str(sequence_id) for sequence_id in getattr(protocol, "public_sequence_ids")
    }
    public_sequences = value.get("public_sequences")
    pooled = value.get("pooled")
    if (
        value.get("format") != PUBLIC_RESULT_FORMAT
        or value.get("protocol_schema") != int(getattr(protocol, "schema_version"))
        or value.get("protocol_sha256")
        != _sha256_file(getattr(protocol, "path"))
        or value.get("condition") != condition
        or not isinstance(public_sequences, Mapping)
        or set(public_sequences) != expected_sequences
        or not isinstance(pooled, Mapping)
        or not isinstance(value.get("method_freeze"), Mapping)
    ):
        raise EvaluationError(
            f"public validation result for {condition} is incomplete or unbound"
        )
    method_freeze = value["method_freeze"]
    threshold = _finite_json_number(
        method_freeze.get("object_score_threshold"),
        f"{condition}.method_freeze.object_score_threshold",
    )
    specifications = {
        str(spec.sequence_id): spec for spec in getattr(protocol, "public_validation")
    }
    accepted_total = 0
    skipped_total = 0
    moving_records: list[
        tuple[int, int, float | None, float | None, float | None, float | None]
    ] = []
    for sequence_id, result in public_sequences.items():
        if not isinstance(result, Mapping):
            raise EvaluationError(f"public sequence {sequence_id} is not a mapping")
        specification = specifications[sequence_id]
        expected_frames = len(
            set(specification.legal_anchors(RELATIVE_TIMES))
            & set(specification.legal_anchors((-4, -3, -2, -1, 0)))
        )
        if result.get("comparison_frame_count") != expected_frames:
            raise EvaluationError(
                f"public sequence {sequence_id} has incomplete frame coverage"
            )
        accepted, skipped = _validate_public_point_metrics(
            result.get("point"),
            expected_frames=expected_frames,
            name=f"public sequence {sequence_id}",
        )
        accepted_total += accepted
        skipped_total += skipped
        moving_records.append(
            _validate_public_moving_metrics(
                result.get("moving_normal"),
                threshold=threshold,
                name=f"public sequence {sequence_id}",
            )
        )
    pooled_accepted, pooled_skipped = _validate_public_point_metrics(
        pooled.get("point"),
        expected_frames=accepted_total + skipped_total,
        name="pooled public result",
    )
    if (pooled_accepted, pooled_skipped) != (accepted_total, skipped_total):
        raise EvaluationError("pooled point frame counts are inconsistent")
    pooled_moving = _validate_public_moving_metrics(
        pooled.get("moving_normal"),
        threshold=threshold,
        name="pooled public result",
    )
    if pooled_moving[0] < 1 or pooled_moving[1] < 1:
        raise EvaluationError("pooled public result lacks moving or static normal points")
    if pooled_moving[:2] != (
        sum(record[0] for record in moving_records),
        sum(record[1] for record in moving_records),
    ):
        raise EvaluationError("pooled moving-normal point counts are inconsistent")
    for count_index, value_index in ((0, 2), (0, 3), (1, 4), (1, 5)):
        count = sum(record[count_index] for record in moving_records)
        numerator = sum(
            record[count_index] * float(record[value_index])
            for record in moving_records
            if record[value_index] is not None
        )
        expected = numerator / count
        if not math.isclose(
            float(pooled_moving[value_index]),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise EvaluationError("pooled moving-normal statistics are inconsistent")
    return value


def _validate_official_public_result(
    path: Path,
    *,
    protocol: object,
    condition: str,
    method_freeze: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid official validation result: {path}") from error
    if not isinstance(value, Mapping):
        raise EvaluationError("official validation result must be a mapping")
    coverage = value.get("comparison_frame_domain")
    point = value.get("point")
    if (
        value.get("format") != OFFICIAL_RESULT_FORMAT
        or value.get("protocol_schema") != int(getattr(protocol, "schema_version"))
        or value.get("protocol_sha256")
        != _sha256_file(getattr(protocol, "path"))
        or value.get("condition") != condition
        or value.get("method_freeze") != dict(method_freeze)
        or not isinstance(coverage, Mapping)
        or set(coverage)
        != {str(item) for item in getattr(protocol, "public_sequence_ids")}
        or not isinstance(point, Mapping)
        or not point
    ):
        raise EvaluationError(
            f"official public result for {condition} is incomplete or unbound"
        )
    specifications = {
        str(spec.sequence_id): spec for spec in getattr(protocol, "public_validation")
    }
    for sequence_id, frame_ids in coverage.items():
        if not isinstance(frame_ids, list) or any(type(item) is not int for item in frame_ids):
            raise EvaluationError("official coverage must contain integer frame lists")
        specification = specifications[sequence_id]
        expected = tuple(
            sorted(
                set(specification.legal_anchors(RELATIVE_TIMES))
                & set(specification.legal_anchors((-4, -3, -2, -1, 0)))
            )
        )
        if tuple(frame_ids) != expected:
            raise EvaluationError(
                f"official public coverage differs for sequence {sequence_id}"
            )
    return value


def _validate_public_confirmation(
    path: Path | str,
    protocol: object,
    condition: str,
    *,
    method_freeze_path: Path | str,
    public_method_freezes: Iterable[str],
    public_results: Iterable[str],
    public_official_results: Iterable[str],
) -> Mapping[str, Any]:
    record = _load_json_record(path, PUBLIC_CONFIRMATION_FORMAT)
    raw_sequence_ids = record.get("sequence_ids")
    if not isinstance(raw_sequence_ids, list) or any(
        type(value) is not int for value in raw_sequence_ids
    ):
        raise EvaluationError("public confirmation sequence_ids must be integers")
    sequence_ids = tuple(raw_sequence_ids)
    result_paths = _condition_paths(public_results, name="public result")
    method_paths = _condition_paths(
        public_method_freezes, name="public method freeze"
    )
    official_paths = _condition_paths(
        public_official_results, name="official public result"
    )
    required_conditions = {"B0", "B1", condition}
    result_hashes = record.get("public_result_sha256_by_condition")
    method_hashes = record.get("method_freeze_sha256_by_condition")
    official_hashes = record.get("official_result_sha256_by_condition")
    moving_safety = record.get("moving_normal_safety_by_condition")
    verdict = record.get("scientific_verdict")
    gate_criteria = getattr(protocol, "decision_gates")["criteria"]
    if (
        record.get("protocol_schema") != int(getattr(protocol, "schema_version"))
        or record.get("protocol_sha256")
        != _sha256_file(getattr(protocol, "path"))
        or record.get("condition") != condition
        or record.get("completed") is not True
        or record.get("supports_main_claim") is not True
        or sequence_ids != tuple(getattr(protocol, "public_sequence_ids"))
        or record.get("method_freeze_sha256")
        != _sha256_file(method_freeze_path)
        or gate_criteria.get("status") != "frozen_before_training"
        or not isinstance(gate_criteria.get("gate4"), Mapping)
        or not isinstance(result_hashes, Mapping)
        or not required_conditions.issubset(result_hashes)
        or not isinstance(method_hashes, Mapping)
        or not required_conditions.issubset(method_hashes)
        or not isinstance(official_hashes, Mapping)
        or not required_conditions.issubset(official_hashes)
        or not isinstance(moving_safety, Mapping)
        or any(moving_safety.get(name) is not True for name in required_conditions)
        or not isinstance(verdict, Mapping)
        or verdict.get("passed") is not True
        or verdict.get("decided_before_hidden") is not True
        or not isinstance(verdict.get("criterion"), str)
        or not str(verdict.get("criterion")).strip()
        or not isinstance(verdict.get("judgment"), str)
        or not str(verdict.get("judgment")).strip()
        or not required_conditions.issubset(result_paths)
        or not required_conditions.issubset(method_paths)
        or not required_conditions.issubset(official_paths)
    ):
        raise EvaluationError(
            "hidden evaluation requires identity-bound public evidence and an explicit verdict"
        )
    for public_condition in sorted(required_conditions):
        public_method_path = method_paths[public_condition]
        public_method = _validate_method_freeze(
            public_method_path, protocol, public_condition
        )
        if method_hashes.get(public_condition) != _sha256_file(public_method_path):
            raise EvaluationError(
                f"method-freeze hash does not match {public_condition} evidence"
            )
        result_path = result_paths[public_condition]
        result = _validate_public_result(
            result_path,
            protocol=protocol,
            condition=public_condition,
        )
        if result_hashes.get(public_condition) != _sha256_file(result_path):
            raise EvaluationError(
                f"public result hash does not match {public_condition} evidence"
            )
        if result.get("method_freeze") != dict(public_method):
            raise EvaluationError(
                f"{public_condition} public result uses a different method freeze"
            )
        official_path = official_paths[public_condition]
        _validate_official_public_result(
            official_path,
            protocol=protocol,
            condition=public_condition,
            method_freeze=public_method,
        )
        if official_hashes.get(public_condition) != _sha256_file(official_path):
            raise EvaluationError(
                f"official-result hash does not match {public_condition} evidence"
            )
    if _sha256_file(method_paths[condition]) != _sha256_file(method_freeze_path):
        raise EvaluationError("the active method freeze differs from public validation")
    return record


def _load_prediction_model(
    protocol: object,
    path: Path,
    device: torch.device,
    condition: str,
    *,
    calibration_digest: str,
    ray_mapping_digest: str,
) -> nn.Module:
    try:
        from .model import AJAEPointTransformer
    except ImportError:  # pragma: no cover - direct script execution
        from model import AJAEPointTransformer
    payload = torch.load(
        path.expanduser().resolve(strict=True),
        map_location=device,
        weights_only=True,
    )
    if not isinstance(payload, Mapping) or payload.get("format") != "ajae-model-v3":
        raise EvaluationError("model checkpoint has an unsupported format")
    identity = payload.get("scientific_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("protocol") != protocol.plain_document()
        or identity.get("calibration_sha256") != calibration_digest
        or identity.get("ray_mapping_digest") != ray_mapping_digest
    ):
        raise EvaluationError(
            "model checkpoint does not match the complete active protocol"
        )
    expected_training_condition = "B3" if condition == "B4" else condition
    training_condition = payload.get("training_condition")
    if (
        not isinstance(training_condition, Mapping)
        or training_condition.get("name") != expected_training_condition
    ):
        raise EvaluationError(
            f"{condition} requires a {expected_training_condition} training checkpoint"
        )
    expected_geometry = {
        "B1": {
            "frame_offsets": [0],
            "model_times": [0],
            "cross_frame_enabled": False,
            "supervised_times": [0],
        },
        "B2": {
            "frame_offsets": list(RELATIVE_TIMES),
            "model_times": list(RELATIVE_TIMES),
            "cross_frame_enabled": False,
            "supervised_times": list(RELATIVE_TIMES),
        },
        "B3": {
            "frame_offsets": list(RELATIVE_TIMES),
            "model_times": list(RELATIVE_TIMES),
            "cross_frame_enabled": True,
            "supervised_times": list(RELATIVE_TIMES),
        },
        "B5": {
            "frame_offsets": [-4, -3, -2, -1, 0],
            "model_times": list(RELATIVE_TIMES),
            "cross_frame_enabled": True,
            "supervised_times": list(RELATIVE_TIMES),
        },
    }[expected_training_condition]
    if any(
        training_condition.get(key) != value for key, value in expected_geometry.items()
    ):
        raise EvaluationError("checkpoint training geometry is inconsistent")
    recorded_condition = payload.get("condition")
    if (
        not isinstance(recorded_condition, Mapping)
        or recorded_condition.get("name") != expected_training_condition
        or recorded_condition != training_condition
    ):
        raise EvaluationError("checkpoint condition record is inconsistent")
    model = AJAEPointTransformer.from_protocol(protocol).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def _protocol_slot_to_ray(
    protocol: object,
) -> tuple[Callable[[object], np.ndarray], str, str]:
    try:
        from .render import load_sensor_calibration
        from .scene import canonical_ray_mapping_digest
    except ImportError:  # pragma: no cover - direct script execution
        from render import load_sensor_calibration
        from scene import canonical_ray_mapping_digest

    ray_grid = getattr(protocol, "render")["ray_grid"]
    beams = int(ray_grid["beam_count"])
    columns = int(ray_grid["column_count"])
    if beams != 128 or columns != 1024:
        raise EvaluationError("evaluation requires the frozen OS1-128 ray grid")
    calibration_path = getattr(protocol, "sensor_calibration_path")()
    try:
        calibrated_grid, sensor = load_sensor_calibration(calibration_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise EvaluationError(
            "prediction requires the authoritative schema-30 ray/sensor calibration"
        ) from error
    if (
        calibrated_grid.beam_count != beams
        or calibrated_grid.columns != columns
        or sensor.source_sequence_id != 206
        or calibrated_grid.calibration_frame_ids != tuple(range(449))
    ):
        raise EvaluationError("the runtime calibration does not match train/206 OS1-128")
    provenance = dict(sensor.provenance)
    required_provenance = {
        "protocol_schema": "30",
        "partition": "train",
        "sequence": "206",
        "frames": "449",
        "first_frame": "0",
        "last_frame": "448",
    }
    if any(provenance.get(key) != value for key, value in required_provenance.items()):
        raise EvaluationError("sensor calibration lacks complete train/206 provenance")
    canonical_by_slot = (
        calibrated_grid.beam_ids * columns + calibrated_grid.column_ids
    ).astype(np.int32)
    digest = canonical_ray_mapping_digest(canonical_by_slot)

    def mapping(source: object) -> np.ndarray:
        count = int(getattr(source, "slot_count"))
        if count != beams * columns:
            raise EvaluationError("source slots do not cover the canonical ray grid")
        return canonical_by_slot.copy()

    return mapping, digest, _sha256_file(calibration_path)


def _predict_command(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from .model import FrozenSTUPointEncoder
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence, _grant_sealed_sequence_access
    except ImportError:  # pragma: no cover - direct script execution
        from model import FrozenSTUPointEncoder
        from protocol import load_protocol
        from scene import LabelMode, STUSequence, _grant_sealed_sequence_access
    protocol = load_protocol(args.protocol)
    condition = _condition(args.condition)
    method_freeze = _validate_method_freeze(
        args.method_freeze_record, protocol, condition
    )
    sealed_access = _grant_sealed_sequence_access(
        protocol,
        partition=args.partition,
        condition=condition,
    )
    public_confirmation: Mapping[str, Any] | None = None
    if args.partition == "test":
        if (
            args.public_confirmation_record is None
            or not args.public_result
            or not args.public_method_freeze
            or not args.public_official_result
        ):
            raise EvaluationError(
                "hidden prediction requires --public-confirmation-record and "
                "identity-bound public, official, and method-freeze records"
            )
        public_confirmation = _validate_public_confirmation(
            args.public_confirmation_record,
            protocol,
            condition,
            method_freeze_path=args.method_freeze_record,
            public_method_freezes=args.public_method_freeze,
            public_results=args.public_result,
            public_official_results=args.public_official_result,
        )
    device = torch.device(args.device)
    slot_to_ray, ray_mapping_digest, calibration_digest = _protocol_slot_to_ray(
        protocol
    )
    if method_freeze.get("calibration_sha256") != calibration_digest:
        raise EvaluationError("method-freeze record does not bind the active calibration")
    model = None
    if condition != "B0":
        if args.model is None:
            raise EvaluationError(f"{condition} requires --model")
        if method_freeze.get("model_sha256") != _sha256_file(args.model):
            raise EvaluationError("method-freeze record does not bind the model checkpoint")
        model = _load_prediction_model(
            protocol,
            args.model,
            device,
            condition,
            calibration_digest=calibration_digest,
            ray_mapping_digest=ray_mapping_digest,
        )
    elif method_freeze.get("model_sha256") is not None:
        raise EvaluationError("B0 method-freeze record must not bind an AJAE model")
    encoder = FrozenSTUPointEncoder.from_protocol(protocol).to(device)
    inference = AJAEInference(
        model,
        encoder,
        condition=condition,
        slot_to_ray=slot_to_ray,
        ray_mapping_digest=ray_mapping_digest,
        device=device,
        cache_frames=int(protocol.training["cache_frames"]),
        time_budget_seconds=args.time_budget_seconds,
    )
    sequence_ids = tuple(args.sequence or ())
    if not sequence_ids:
        sequence_ids = (
            protocol.public_sequence_ids
            if args.partition == "val"
            else protocol.hidden_sequence_ids
        )
    records: list[dict[str, Any]] = []
    for sequence_id in sequence_ids:
        sequence = STUSequence.open(
            args.data_root,
            protocol=protocol,
            partition=args.partition,
            sequence_id=sequence_id,
            label_mode=LabelMode.FORBIDDEN,
            sealed_access=sealed_access,
        )
        output = args.output / condition / str(sequence_id)
        predictions = inference.predict_sequence(sequence, output_dir=output)
        records.append(
            {
                "sequence": sequence_id,
                "frames": len(predictions.frames),
                "output": str(output),
                "cost": dict(predictions.cost),
                "position_diagnostic": predictions.position_diagnostic,
            }
        )
    result = {
        "condition": condition,
        "partition": args.partition,
        "sequences": records,
        "method_freeze": dict(method_freeze),
        "public_confirmation": (
            None if public_confirmation is None else dict(public_confirmation)
        ),
    }
    save_result(args.output / condition / "prediction_result.json", result)
    return result


def _metrics_command(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence, _grant_sealed_sequence_access
    except ImportError:  # pragma: no cover - direct script execution
        from protocol import load_protocol
        from scene import LabelMode, STUSequence, _grant_sealed_sequence_access
    protocol = load_protocol(args.protocol)
    condition = _condition(args.condition)
    method_freeze = _validate_method_freeze(
        args.method_freeze_record, protocol, condition
    )
    sealed_access = _grant_sealed_sequence_access(
        protocol,
        partition="val",
        condition=condition,
    )
    point = PointMetricAccumulator(protocol)
    threshold = float(method_freeze["object_score_threshold"])
    moving = MovingNormalDiagnostic(threshold, protocol)
    objects = ObjectMetricAccumulator(protocol) if args.instances is not None else None
    ledger = EvaluationLedger(condition)
    for sequence_id in protocol.public_sequence_ids:
        sequence_point = PointMetricAccumulator(protocol)
        sequence_moving = MovingNormalDiagnostic(threshold, protocol)
        sequence_objects = (
            ObjectMetricAccumulator(protocol) if args.instances is not None else None
        )
        sequence = STUSequence.open(
            args.data_root,
            protocol=protocol,
            partition="val",
            sequence_id=sequence_id,
            label_mode=LabelMode.REQUIRED,
            sealed_access=sealed_access,
        )
        prediction_dir = args.predictions / condition / str(sequence_id)
        frame_ids = load_prediction_coverage(
            prediction_dir,
            condition=condition,
            expected_frame_ids=AJAEInference._comparison_frame_ids(sequence),
        )
        for frame_id in frame_ids:
            source = sequence.source_frame(frame_id)
            score_path = prediction_dir / f"{frame_id:06d}.txt"
            scores = np.loadtxt(score_path, dtype=np.float32, ndmin=1)
            assert source.labels is not None
            point.update(source.xyzi[:, :3], scores, source.labels.semantic)
            sequence_point.update(source.xyzi[:, :3], scores, source.labels.semantic)
            moving.update(source.xyzi[:, :3], scores, source.labels.semantic)
            sequence_moving.update(
                source.xyzi[:, :3], scores, source.labels.semantic
            )
            if objects is not None:
                packed = np.fromfile(
                    args.instances
                    / condition
                    / str(sequence_id)
                    / f"{frame_id:06d}.label",
                    dtype="<u4",
                ).astype(np.uint32, copy=False)
                objects.update(source.xyzi[:, :3], packed, source.labels.packed)
                assert sequence_objects is not None
                sequence_objects.update(
                    source.xyzi[:, :3], packed, source.labels.packed
                )
        sequence_result: dict[str, Any] = {
            "point": sequence_point.compute(),
            "moving_normal": sequence_moving.compute(),
            "comparison_frame_count": len(frame_ids),
        }
        if sequence_objects is not None:
            sequence_result["object"] = sequence_objects.compute()
        ledger.add_public_sequence(sequence_id, sequence_result)
    pooled: dict[str, Any] = {
        "point": point.compute(),
        "moving_normal": moving.compute(),
    }
    if objects is not None:
        pooled["object"] = objects.compute()
    ledger.pooled = pooled
    result = ledger.to_dict()
    result["format"] = PUBLIC_RESULT_FORMAT
    result["protocol_schema"] = int(protocol.schema_version)
    result["protocol_sha256"] = _sha256_file(protocol.path)
    result["method_freeze"] = dict(method_freeze)
    if args.output is not None:
        save_result(args.output, result)
    return result


def _instances_command(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence, _grant_sealed_sequence_access
    except ImportError:  # pragma: no cover - direct script execution
        from protocol import load_protocol
        from scene import LabelMode, STUSequence, _grant_sealed_sequence_access
    protocol = load_protocol(args.protocol)
    condition = _condition(args.condition)
    method_freeze = _validate_method_freeze(
        args.method_freeze_record, protocol, condition
    )
    sealed_access = _grant_sealed_sequence_access(
        protocol,
        partition=args.partition,
        condition=condition,
    )
    if not math.isclose(
        float(args.threshold),
        float(method_freeze["object_score_threshold"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise EvaluationError("instance threshold differs from the frozen method")
    public_confirmation: Mapping[str, Any] | None = None
    if args.partition == "test":
        if (
            args.public_confirmation_record is None
            or not args.public_result
            or not args.public_method_freeze
            or not args.public_official_result
        ):
            raise EvaluationError(
                "hidden instance prediction requires --public-confirmation-record and "
                "identity-bound public, official, and method-freeze records"
            )
        public_confirmation = _validate_public_confirmation(
            args.public_confirmation_record,
            protocol,
            condition,
            method_freeze_path=args.method_freeze_record,
            public_method_freezes=args.public_method_freeze,
            public_results=args.public_result,
            public_official_results=args.public_official_result,
        )
    sequence_ids = tuple(args.sequence or ())
    if not sequence_ids:
        sequence_ids = (
            protocol.public_sequence_ids
            if args.partition == "val"
            else protocol.hidden_sequence_ids
        )
    records = []
    for sequence_id in sequence_ids:
        sequence = STUSequence.open(
            args.data_root,
            protocol=protocol,
            partition=args.partition,
            sequence_id=sequence_id,
            label_mode=LabelMode.FORBIDDEN,
            sealed_access=sealed_access,
        )
        destination = args.output / condition / str(sequence_id)
        prediction_dir = args.predictions / condition / str(sequence_id)
        frame_ids = load_prediction_coverage(
            prediction_dir,
            condition=condition,
            expected_frame_ids=AJAEInference._comparison_frame_ids(sequence),
        )
        for frame_id in frame_ids:
            source = sequence.source_frame(frame_id)
            scores = np.loadtxt(
                prediction_dir / f"{frame_id:06d}.txt",
                dtype=np.float32,
                ndmin=1,
            )
            packed = scores_to_packed_instances(
                source.xyzi[:, :3],
                scores,
                args.threshold,
                eps=protocol.evaluation.dbscan_eps_m,
                min_samples=protocol.evaluation.dbscan_min_samples,
                protocol=protocol,
            )
            write_packed_labels(destination / f"{frame_id:06d}.label", packed)
        records.append(
            {
                "sequence": sequence_id,
                "frames": len(frame_ids),
                "output": str(destination),
            }
        )
    result = {
        "condition": condition,
        "partition": args.partition,
        "sequences": records,
        "method_freeze": dict(method_freeze),
        "public_confirmation": (
            None if public_confirmation is None else dict(public_confirmation)
        ),
    }
    save_result(args.output / condition / "instance_result.json", result)
    return result


def _build_official_filtered_view(
    root: Path,
    *,
    protocol: object,
    condition: str,
    data_dir: Path,
    predictions: Path,
    instances: Path | None,
) -> tuple[Path, Path, Path | None, dict[str, list[int]]]:
    """Expose only real predictions on the frozen common frame domain."""

    filtered_data = root / "data"
    filtered_predictions = root / "predictions"
    filtered_instances = None if instances is None else root / "instances"
    coverage: dict[str, list[int]] = {}
    for sequence_id in getattr(protocol, "public_sequence_ids"):
        spec = getattr(protocol, "sequence")("val", sequence_id)
        centered = frozenset(spec.legal_anchors(RELATIVE_TIMES))
        causal = frozenset(spec.legal_anchors((-4, -3, -2, -1, 0)))
        expected = tuple(sorted(centered & causal))
        source_prediction = predictions / str(sequence_id)
        frames = load_prediction_coverage(
            source_prediction,
            condition=condition,
            expected_frame_ids=expected,
        )
        coverage[str(sequence_id)] = list(frames)
        source_sequence = data_dir / str(sequence_id)
        data_sequence = filtered_data / str(sequence_id)
        prediction_sequence = filtered_predictions / str(sequence_id)
        (data_sequence / "velodyne").mkdir(parents=True)
        (data_sequence / "labels").mkdir()
        prediction_sequence.mkdir(parents=True)
        instance_sequence = None
        if filtered_instances is not None:
            instance_sequence = filtered_instances / str(sequence_id)
            instance_sequence.mkdir(parents=True)
        for frame_id in frames:
            stem = f"{frame_id:06d}"
            links = (
                (
                    source_sequence / "velodyne" / f"{stem}.bin",
                    data_sequence / "velodyne" / f"{stem}.bin",
                ),
                (
                    source_sequence / "labels" / f"{stem}.label",
                    data_sequence / "labels" / f"{stem}.label",
                ),
                (
                    source_prediction / f"{stem}.txt",
                    prediction_sequence / f"{stem}.txt",
                ),
            )
            for source, destination in links:
                destination.symlink_to(source.resolve(strict=True))
            if instance_sequence is not None and instances is not None:
                (instance_sequence / f"{stem}.label").symlink_to(
                    (instances / str(sequence_id) / f"{stem}.label").resolve(
                        strict=True
                    )
                )
    return filtered_data, filtered_predictions, filtered_instances, coverage


def _official_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run the released STU scripts over complete prediction directories."""

    try:
        from .protocol import load_protocol
    except ImportError:  # pragma: no cover - direct script execution
        from protocol import load_protocol
    protocol = load_protocol(args.protocol)
    condition = _condition(args.condition)
    method_freeze = _validate_method_freeze(
        args.method_freeze_record, protocol, condition
    )
    official_root = protocol.stu_repository_path().parent
    point_script = official_root / "compute_point_level_ood.py"
    object_script = official_root / "compute_object_level_ood.py"
    for path in (point_script, object_script):
        if not path.is_file():
            raise EvaluationError(f"released STU evaluator is missing: {path}")
    data_dir = args.data_root.expanduser().resolve(strict=True) / "val"
    predictions = (args.predictions / condition).expanduser().resolve(strict=True)
    destination = (args.output / condition).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    instances = (
        None
        if args.instances is None
        else (args.instances / condition).expanduser().resolve(strict=True)
    )
    point_output = destination / "official_point.json"
    with tempfile.TemporaryDirectory(prefix="ajae-official-") as temporary:
        filtered_data, filtered_predictions, filtered_instances, coverage = (
            _build_official_filtered_view(
                Path(temporary),
                protocol=protocol,
                condition=condition,
                data_dir=data_dir,
                predictions=predictions,
                instances=instances,
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(point_script),
                "--data-dir",
                str(filtered_data),
                "--pred-dir",
                str(filtered_predictions),
                "--output",
                str(point_output),
            ],
            cwd=official_root,
            check=True,
        )
        result: dict[str, Any] = {
            "format": OFFICIAL_RESULT_FORMAT,
            "protocol_schema": int(protocol.schema_version),
            "protocol_sha256": _sha256_file(protocol.path),
            "condition": condition,
            "method_freeze": dict(method_freeze),
            "comparison_frame_domain": coverage,
            "point": json.loads(point_output.read_text(encoding="utf-8")),
        }
        if filtered_instances is not None:
            object_output = destination / "official_object.json"
            subprocess.run(
                [
                    sys.executable,
                    str(object_script),
                    "--data-dir",
                    str(filtered_data),
                    "--instance-dir",
                    str(filtered_instances),
                    "--output",
                    str(object_output),
                    "--min-points",
                    str(protocol.evaluation.minimum_anomaly_points),
                ],
                cwd=official_root,
                check=True,
            )
            result["object"] = json.loads(
                object_output.read_text(encoding="utf-8")
            )
    save_result(destination / "official_result.json", result)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="AJAE inference and official evaluation"
    )
    parser.add_argument("--protocol", type=Path, default=Path("protocol.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="run one frozen B0--B5 condition")
    predict.add_argument("--data-root", type=Path, required=True)
    predict.add_argument("--partition", choices=("val", "test"), required=True)
    predict.add_argument("--condition", choices=CONDITIONS, required=True)
    predict.add_argument(
        "--sequence",
        type=int,
        action="append",
        help="sequence ID; repeat as needed, or omit to process the full protocol split",
    )
    predict.add_argument("--model", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--method-freeze-record", type=Path, required=True)
    predict.add_argument("--public-confirmation-record", type=Path)
    predict.add_argument(
        "--public-method-freeze",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="method freeze used by each public B0/B1/active-condition result",
    )
    predict.add_argument(
        "--public-result",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="validated public result; hidden prediction requires B0, B1, and the active condition",
    )
    predict.add_argument(
        "--public-official-result",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="released-STU evaluator result for public B0/B1/active conditions",
    )
    predict.add_argument("--time-budget-seconds", type=float)
    predict.set_defaults(handler=_predict_command)

    metrics = subparsers.add_parser("metrics", help="compute public validation metrics")
    metrics.add_argument("--data-root", type=Path, required=True)
    metrics.add_argument("--condition", choices=CONDITIONS, required=True)
    metrics.add_argument("--predictions", type=Path, required=True)
    metrics.add_argument("--instances", type=Path)
    metrics.add_argument("--output", type=Path)
    metrics.add_argument("--method-freeze-record", type=Path, required=True)
    metrics.set_defaults(handler=_metrics_command)

    instances = subparsers.add_parser(
        "instances", help="make per-frame DBSCAN instances"
    )
    instances.add_argument("--data-root", type=Path, required=True)
    instances.add_argument("--partition", choices=("val", "test"), required=True)
    instances.add_argument("--condition", choices=CONDITIONS, required=True)
    instances.add_argument(
        "--sequence",
        type=int,
        action="append",
        help="sequence ID; repeat as needed, or omit to process the full protocol split",
    )
    instances.add_argument("--predictions", type=Path, required=True)
    instances.add_argument("--threshold", type=float, required=True)
    instances.add_argument("--output", type=Path, required=True)
    instances.add_argument("--method-freeze-record", type=Path, required=True)
    instances.add_argument("--public-confirmation-record", type=Path)
    instances.add_argument(
        "--public-method-freeze",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="method freeze used by each public B0/B1/active-condition result",
    )
    instances.add_argument(
        "--public-result",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="validated public result; hidden instances require B0, B1, and the active condition",
    )
    instances.add_argument(
        "--public-official-result",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="released-STU evaluator result for public B0/B1/active conditions",
    )
    instances.set_defaults(handler=_instances_command)

    official = subparsers.add_parser(
        "official", help="recompute final validation metrics with released STU scripts"
    )
    official.add_argument("--data-root", type=Path, required=True)
    official.add_argument("--condition", choices=CONDITIONS, required=True)
    official.add_argument("--predictions", type=Path, required=True)
    official.add_argument("--instances", type=Path)
    official.add_argument("--output", type=Path, required=True)
    official.add_argument("--method-freeze-record", type=Path, required=True)
    official.set_defaults(handler=_official_command)

    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
