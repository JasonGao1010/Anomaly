#!/usr/bin/env python3
"""Pure point-level metrics shared by AJAE mechanism and final evaluation.

This module deliberately contains no model loader or prediction writer. The
current Oracle experiment has not frozen a deployable matcher, so retaining the
retired inference path would falsely imply that final STU evaluation is ready.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from sklearn.metrics import auc, average_precision_score, roc_curve


class _EvaluationRange(Protocol):
    minimum_range_m: float
    maximum_range_m: float


class _Protocol(Protocol):
    evaluation: _EvaluationRange


class _SourceFrame(Protocol):
    xyzi: np.ndarray


class EvaluationError(ValueError):
    """Report an undefined metric or a malformed point population."""


def _range_mask(protocol: _Protocol, source: _SourceFrame) -> np.ndarray:
    """Select points inside the inclusive official STU distance interval."""

    xyzi = np.asarray(source.xyzi)
    if xyzi.ndim != 2 or xyzi.shape[1] < 3 or not np.isfinite(xyzi[:, :3]).all():
        raise EvaluationError("source coordinates must be finite [N,>=3]")
    distance = np.linalg.norm(xyzi[:, :3].astype(np.float64), axis=1)
    return (distance >= protocol.evaluation.minimum_range_m) & (
        distance <= protocol.evaluation.maximum_range_m
    )


def normal_alarm_threshold(scores: np.ndarray, alarm_rate: float) -> float:
    """Return the exact one-based normal-score order statistic used by STU."""

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


def balanced_binary_cross_entropy(labels: np.ndarray, logits: np.ndarray) -> float:
    """Average positive and negative logistic losses with equal class weight."""

    truth = np.asarray(labels)
    values = np.asarray(logits)
    if truth.dtype != np.bool_ or truth.ndim != 1 or values.shape != truth.shape:
        raise TypeError("labels must be bool[N] and logits must be aligned")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise EvaluationError("point logits must be finite floating values")
    if not bool(truth.any()) or bool(truth.all()):
        raise EvaluationError("balanced BCE requires both point classes")
    # log(1 + exp(x)) - y*x is stable through logaddexp.
    loss = np.logaddexp(0.0, values.astype(np.float64)) - truth * values
    return float(0.5 * (loss[truth].mean() + loss[~truth].mean()))


def _point_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute official percentage AP, AUROC and FPR95 for one pooled population."""

    truth = np.asarray(labels)
    values = np.asarray(scores)
    if truth.dtype != np.bool_ or truth.ndim != 1 or values.shape != truth.shape:
        raise TypeError("point labels and scores must be aligned one-dimensional arrays")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise EvaluationError("point scores must be finite floating values")
    if truth.size == 0 or not bool(truth.any()) or bool(truth.all()):
        raise EvaluationError("point metrics require both normal and anomaly points")
    average_precision = average_precision_score(truth, values)
    false_positive, true_positive, thresholds = roc_curve(truth, values)
    area = auc(false_positive, true_positive)
    candidates = np.flatnonzero(true_positive > 0.95)
    if candidates.size == 0:
        raise EvaluationError("FPR95 is undefined because recall never exceeds 95%")
    index = int(candidates[0])
    return {
        "AP": float(average_precision * 100.0),
        "AUROC": float(area * 100.0),
        "FPR95": float(false_positive[index] * 100.0),
        "FPR95_threshold": float(thresholds[index]),
    }


def mechanism_point_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """Report the three classification quantities used by the Oracle experiment."""

    values = np.asarray(logits)
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise EvaluationError("point logits must be finite floating values")
    probabilities = np.empty(values.shape, dtype=np.float64)
    positive = values >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    official = _point_metrics(np.asarray(labels), probabilities)
    return {
        "balanced_BCE": balanced_binary_cross_entropy(labels, values),
        "AP": official["AP"],
        "AUROC": official["AUROC"],
    }
