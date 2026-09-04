#!/usr/bin/env python3
"""Run the three pre-training AJAE feasibility experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_curve

try:
    from .model import FrozenSTUPointEncoder, STUPointEncoding
    from .protocol import AJAEProtocol, load_protocol
    from .render import (
        PlacementError,
        collect_observed_obstacle_index,
        load_qualified_support_pool,
        load_sensor_calibration,
        sample_development_clip_world,
    )
    from .scene import (
        LabelMode,
        SceneWindow,
        STUSequence,
        assemble_window,
        canonical_ray_mapping_digest,
    )
except ImportError:  # Direct script execution.
    from model import FrozenSTUPointEncoder, STUPointEncoding
    from protocol import AJAEProtocol, load_protocol
    from render import (
        PlacementError,
        collect_observed_obstacle_index,
        load_qualified_support_pool,
        load_sensor_calibration,
        sample_development_clip_world,
    )
    from scene import (
        LabelMode,
        SceneWindow,
        STUSequence,
        assemble_window,
        canonical_ray_mapping_digest,
    )


class EvaluationError(ValueError):
    """Report a feasibility input or metric contradiction."""


@dataclass(frozen=True, slots=True)
class WindowSTUInputs:
    """Exact single- and dense-STU inputs for one current frame."""

    single_coordinates: np.ndarray
    single_features: np.ndarray
    single_real_slots: np.ndarray
    dense_coordinates: np.ndarray
    dense_features: np.ndarray
    dense_current_rows: np.ndarray

    def __post_init__(self) -> None:
        if self.single_coordinates.ndim != 2 or self.single_coordinates.shape[1] != 3:
            raise EvaluationError("single STU coordinates must be [S,3]")
        if self.single_features.shape != (self.single_coordinates.shape[0], 2):
            raise EvaluationError("single STU features must be [S,2]")
        if self.dense_coordinates.ndim != 2 or self.dense_coordinates.shape[1] != 3:
            raise EvaluationError("dense STU coordinates must be [N,3]")
        if self.dense_features.shape != (self.dense_coordinates.shape[0], 2):
            raise EvaluationError("dense STU features must be [N,2]")
        if self.dense_current_rows.shape != (self.single_real_slots.size,):
            raise EvaluationError(
                "dense current rows do not match current real returns"
            )


@dataclass(frozen=True, slots=True)
class PairedScores:
    """All dense outputs plus the paired current-point comparison view."""

    single_score: np.ndarray
    single_class: np.ndarray
    dense_all_score: np.ndarray
    dense_all_class: np.ndarray
    current_rows: np.ndarray

    def __post_init__(self) -> None:
        current_count = self.single_score.size
        if (
            self.single_score.shape != (current_count,)
            or self.single_class.shape != (current_count,)
            or self.dense_all_score.ndim != 1
            or self.dense_all_class.shape != self.dense_all_score.shape
            or self.current_rows.shape != (current_count,)
            or np.any(self.current_rows < 0)
            or np.any(self.current_rows >= self.dense_all_score.size)
        ):
            raise EvaluationError("paired STU outputs have inconsistent point rows")

    @property
    def dense_current_score(self) -> np.ndarray:
        return self.dense_all_score[self.current_rows]

    @property
    def dense_current_class(self) -> np.ndarray:
        return self.dense_all_class[self.current_rows]


def window_stu_inputs(window: SceneWindow) -> WindowSTUInputs:
    """Build a pseudo-scan without exposing scan identity to frozen STU."""

    current = window.current_frame.source
    frames = window.frames
    intensity = np.concatenate(
        [frame.source.xyzi[frame.source.real_slots, 3] for frame in frames]
    ).astype(np.float32, copy=False)
    if intensity.shape != (window.points.count,):
        raise EvaluationError("dense intensity rows do not match aligned points")

    current_xyz = np.asarray(window.points.coordinates, dtype=np.float64)
    pose = window.current_pose.world_from_current
    dense_world = current_xyz @ pose[:3, :3].T + pose[:3, 3]
    center = dense_world.mean(axis=0)
    distance = np.linalg.norm(dense_world - center, axis=1).astype(np.float32)
    dense_features = np.column_stack((intensity, distance)).astype(np.float32)
    dense_rows = np.flatnonzero(window.current_mask).astype(np.int64)

    # The latest member must be geometrically unchanged by T_t<-t.
    current_native = current.xyzi[current.real_slots, :3]
    if not np.allclose(
        window.points.coordinates[dense_rows], current_native, atol=1.0e-5, rtol=1.0e-6
    ):
        raise EvaluationError(
            "latest-scan points changed during current-frame alignment"
        )
    return WindowSTUInputs(
        single_coordinates=np.asarray(current.coordinates),
        single_features=np.asarray(current.features),
        single_real_slots=np.asarray(current.real_slots, dtype=np.int64),
        dense_coordinates=dense_world,
        dense_features=dense_features,
        dense_current_rows=dense_rows,
    )


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _classes(encoding: STUPointEncoding) -> np.ndarray:
    classes = _numpy(encoding.normal_class).astype(np.uint8)
    official_reference = _numpy(encoding.normal_evidence.argmax(dim=1)).astype(np.uint8)
    if not np.array_equal(classes, official_reference):
        raise EvaluationError(
            "normal_class differs from the official assigned-query semantic output"
        )
    return classes


def score_window(
    encoder: FrozenSTUPointEncoder,
    window: SceneWindow,
) -> PairedScores:
    """Run both inputs while retaining every dense-STU point prediction."""

    inputs = window_stu_inputs(window)
    single = encoder(
        inputs.single_coordinates,
        inputs.single_features,
        inputs.single_real_slots,
    )
    dense = encoder(inputs.dense_coordinates, inputs.dense_features)
    rows = torch.as_tensor(
        inputs.dense_current_rows, dtype=torch.long, device=dense.maxlogit_score.device
    )
    return PairedScores(
        single_score=_numpy(single.maxlogit_score).astype(np.float64),
        single_class=_classes(single),
        dense_all_score=_numpy(dense.maxlogit_score).astype(np.float64),
        dense_all_class=_classes(dense),
        current_rows=_numpy(rows).astype(np.int64),
    )


def _voxel_keys(points: np.ndarray, size: float) -> np.ndarray:
    return np.floor(np.asarray(points, dtype=np.float64) / float(size)).astype(np.int64)


def geometry_record(
    window: SceneWindow, voxel_sizes: Iterable[float]
) -> dict[str, object]:
    """Measure added physical observations without invoking STU."""

    current = np.asarray(
        window.points.coordinates[window.current_mask], dtype=np.float64
    )
    dense = np.asarray(window.points.coordinates, dtype=np.float64)
    record: dict[str, object] = {
        "window_start": window.window_start,
        "current_frame": window.current_frame_id,
        "single_visible_returns": int(current.shape[0]),
        "dense_visible_returns": int(dense.shape[0]),
        "visible_return_ratio": float(dense.shape[0] / current.shape[0]),
        "voxels": {},
    }
    voxel_records: dict[str, object] = {}
    for size in voxel_sizes:
        single_keys = np.unique(_voxel_keys(current, size), axis=0)
        dense_keys = np.unique(_voxel_keys(dense, size), axis=0)
        single_set = {tuple(row) for row in single_keys.tolist()}
        new_count = sum(tuple(row) not in single_set for row in dense_keys.tolist())
        voxel_records[f"{float(size):.2f}"] = {
            "single_unique_voxels": int(single_keys.shape[0]),
            "dense_unique_voxels": int(dense_keys.shape[0]),
            "unique_voxel_ratio": float(dense_keys.shape[0] / single_keys.shape[0]),
            "new_voxels_not_present_in_current_scan": int(new_count),
        }
    record["voxels"] = voxel_records
    return record


def _write_ply(path: Path, window: SceneWindow) -> None:
    points = np.asarray(window.points.coordinates, dtype=np.float32)
    groups = np.asarray(window.points.scan_group, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {points.shape[0]}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar scan_age\nend_header\n")
        for point, group in zip(points, groups, strict=True):
            stream.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {4 - int(group)}\n"
            )
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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
    os.replace(temporary, path)


def _open_normal_201(data_root: Path, protocol: AJAEProtocol) -> STUSequence:
    return STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )


def run_f1_geometry(
    data_root: Path,
    output_dir: Path,
    *,
    protocol: AJAEProtocol,
) -> dict[str, object]:
    sequence = _open_normal_201(data_root, protocol)
    spec = protocol.normal_development
    settings = protocol.feasibility["F1_geometry"]
    sizes = tuple(float(value) for value in settings["voxel_sizes_m"])
    ply_frames = frozenset(int(value) for value in settings["ply_current_frames"])
    records: list[dict[str, object]] = []
    for start in spec.legal_window_starts():
        window = sequence.window(start)
        records.append(geometry_record(window, sizes))
        if window.current_frame_id in ply_frames:
            _write_ply(
                output_dir / f"dense_current_{window.current_frame_id:06d}.ply", window
            )
    ratios = np.asarray(
        [record["visible_return_ratio"] for record in records], dtype=np.float64
    )
    voxel_summary: dict[str, object] = {}
    for size in sizes:
        key = f"{size:.2f}"
        voxel_ratios = np.asarray(
            [record["voxels"][key]["unique_voxel_ratio"] for record in records],
            dtype=np.float64,
        )
        new_voxels = np.asarray(
            [
                record["voxels"][key]["new_voxels_not_present_in_current_scan"]
                for record in records
            ],
            dtype=np.int64,
        )
        voxel_summary[key] = {
            "unique_voxel_ratio_median": float(np.median(voxel_ratios)),
            "unique_voxel_ratio_q05_q95": np.quantile(
                voxel_ratios, (0.05, 0.95)
            ).tolist(),
            "new_voxels_median": float(np.median(new_voxels)),
        }
    summary = {
        "format": "ajae-schema33-F1-geometry-v1",
        "protocol_identity": protocol.scientific_identity,
        "status": "completed_descriptive",
        "window_count": len(records),
        "visible_return_ratio_median": float(np.median(ratios)),
        "visible_return_ratio_q05_q95": np.quantile(ratios, (0.05, 0.95)).tolist(),
        "voxel_summary": voxel_summary,
        "records": records,
    }
    _atomic_json(output_dir / "F1_geometry.json", summary)
    return summary


def _miou(target: np.ndarray, prediction: np.ndarray) -> float:
    values: list[float] = []
    for label in np.unique(target):
        union = np.count_nonzero((target == label) | (prediction == label))
        if union:
            values.append(
                np.count_nonzero((target == label) & (prediction == label)) / union
            )
    return float(np.mean(values)) if values else math.nan


@dataclass(frozen=True, slots=True)
class F2PointMasks:
    """Separate anomaly-task normal points from 19-class semantic points."""

    normal_anomaly: np.ndarray
    semantic_class: np.ndarray
    semantic_target: np.ndarray


def f2_point_masks(window: SceneWindow, protocol: AJAEProtocol) -> F2PointMasks:
    labels = window.current_frame.source.labels
    if labels is None or labels.semantic_target is None:
        raise EvaluationError("normal stability requires mapped train/201 labels")
    current = window.current_frame.source
    slots = current.real_slots
    semantic = np.asarray(labels.semantic[slots], dtype=np.uint16)
    target = np.asarray(labels.semantic_target[slots], dtype=np.uint8)
    ranges = np.linalg.norm(current.xyzi[slots, :3], axis=1)
    in_range = protocol.evaluation.range_mask(ranges)
    normal_anomaly = in_range & (semantic != np.uint16(0)) & (semantic != np.uint16(2))
    semantic_class = in_range & (target != np.uint8(255))
    return F2PointMasks(normal_anomaly, semantic_class, target)


def run_f2_normal_stability(
    data_root: Path,
    output_dir: Path,
    *,
    protocol: AJAEProtocol,
    device: str,
    encoder: FrozenSTUPointEncoder | None = None,
) -> dict[str, object]:
    sequence = _open_normal_201(data_root, protocol)
    settings = protocol.feasibility["F2_normal_stability"]
    model = (
        FrozenSTUPointEncoder.from_protocol(protocol) if encoder is None else encoder
    )
    model.to(torch.device(device)).eval()
    single_scores: list[np.ndarray] = []
    dense_scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    single_classes: list[np.ndarray] = []
    dense_classes: list[np.ndarray] = []
    frame_records: list[dict[str, object]] = []
    precheck_frames = frozenset(
        int(value) for value in settings["official_normal_class_precheck_frames"]
    )
    precheck_points = 0
    for current_frame in settings["current_frames"]:
        window = sequence.window(int(current_frame) - 4)
        paired = score_window(model, window)
        masks = f2_point_masks(window, protocol)
        if not np.any(masks.normal_anomaly) or not np.any(masks.semantic_class):
            raise EvaluationError(
                f"current frame {current_frame} has an empty F2 evaluation mask"
            )
        single_scores.append(paired.single_score[masks.normal_anomaly])
        dense_scores.append(paired.dense_current_score[masks.normal_anomaly])
        targets.append(masks.semantic_target[masks.semantic_class])
        single_classes.append(paired.single_class[masks.semantic_class])
        dense_classes.append(paired.dense_current_class[masks.semantic_class])
        if int(current_frame) in precheck_frames:
            precheck_points += int(np.count_nonzero(masks.semantic_class))
        frame_records.append(
            {
                "current_frame": int(current_frame),
                "normal_anomaly_points": int(np.count_nonzero(masks.normal_anomaly)),
                "semantic_class_points": int(np.count_nonzero(masks.semantic_class)),
                "single_MaxLogit_median": float(
                    np.median(paired.single_score[masks.normal_anomaly])
                ),
                "dense_MaxLogit_median": float(
                    np.median(paired.dense_current_score[masks.normal_anomaly])
                ),
            }
        )
    single_score = np.concatenate(single_scores)
    dense_score = np.concatenate(dense_scores)
    target = np.concatenate(targets)
    single_class = np.concatenate(single_classes)
    dense_class = np.concatenate(dense_classes)
    median_shift = float(np.median(dense_score) - np.median(single_score))
    q95_shift = float(np.quantile(dense_score, 0.95) - np.quantile(single_score, 0.95))
    single_accuracy = float(np.mean(single_class == target))
    dense_accuracy = float(np.mean(dense_class == target))
    accuracy_drop = single_accuracy - dense_accuracy
    stable = (
        median_shift <= float(settings["maximum_median_MaxLogit_increase"])
        and q95_shift <= float(settings["maximum_q95_MaxLogit_increase"])
        and accuracy_drop <= float(settings["maximum_normal_accuracy_drop"])
    )
    summary = {
        "format": "ajae-schema33-F2-normal-stability-v1",
        "protocol_identity": protocol.scientific_identity,
        "status": "passed" if stable else "failed",
        "frame_count": len(frame_records),
        "normal_anomaly_point_count": int(single_score.size),
        "semantic_class_point_count": int(target.size),
        "official_normal_class_precheck": {
            "status": "passed",
            "frames": sorted(precheck_frames),
            "point_count": precheck_points,
            "reference": "released_STU_trainer_formula_checked_inside_encoder",
        },
        "median_MaxLogit_increase": median_shift,
        "q95_MaxLogit_increase": q95_shift,
        "single_normal_accuracy": single_accuracy,
        "dense_normal_accuracy": dense_accuracy,
        "normal_accuracy_drop": accuracy_drop,
        "single_normal_mIoU": _miou(target, single_class),
        "dense_normal_mIoU": _miou(target, dense_class),
        "frames": frame_records,
    }
    _atomic_json(output_dir / "F2_normal_stability.json", summary)
    return summary


def official_point_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Reproduce STU's released point-level AP, AUROC, and FPR95."""

    truth = np.asarray(labels, dtype=np.bool_)
    value = np.asarray(scores, dtype=np.float64)
    if truth.shape != value.shape or truth.ndim != 1 or not np.isfinite(value).all():
        raise EvaluationError("point-metric inputs must be aligned finite vectors")
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        raise EvaluationError("point metrics require at least one anomaly point")
    if positives == truth.size:
        raise EvaluationError("point metrics require at least one normal point")
    fpr, tpr, thresholds = roc_curve(truth, value)
    eligible = np.flatnonzero(tpr > 0.95)
    if eligible.size == 0:
        raise EvaluationError("official FPR95 threshold was not reached")
    index = int(eligible[0])
    return {
        "AP": float(average_precision_score(truth, value) * 100.0),
        "AUROC": float(auc(fpr, tpr) * 100.0),
        "FPR95": float(fpr[index] * 100.0),
        "threshold": float(thresholds[index]),
    }


def _paired_bootstrap(
    values: np.ndarray, repetitions: int, seed: int
) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise EvaluationError("paired bootstrap requires finite world differences")
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[sample].mean(axis=1)
    return tuple(map(float, np.quantile(means, (0.025, 0.975))))


def _f3_world_record(
    *,
    phase: str,
    world_index: int,
    source_start: int,
    root_seed: int,
    sequence: STUSequence,
    model: FrozenSTUPointEncoder,
    protocol: AJAEProtocol,
    ray_grid: object,
    sensor: object,
    ray_digest: str,
    support_pool: object,
) -> dict[str, object]:
    settings = protocol.feasibility["F3_proxy_signal"]
    length = int(settings["frames_per_sequence"])
    sources = tuple(
        sequence.source_frame(frame)
        for frame in range(source_start, source_start + length)
    )
    obstacles = collect_observed_obstacle_index(sources, source_sequence_id=201)
    maximum_attempts = int(
        protocol.render["F3_world_retry"]["maximum_placement_attempts_per_root_seed"]
    )
    base = {
        "phase": phase,
        "world_index": world_index,
        "source_start": source_start,
        "root_seed": root_seed,
    }
    try:
        clip = sample_development_clip_world(
            support_pool,
            obstacles,
            sources,
            ray_grid,
            sensor,
            root_seed,
            renderer_identity=protocol.scientific_identity,
            maximum_attempts=maximum_attempts,
        )
    except PlacementError as error:
        return {
            **base,
            "status": "physical_world_generation_failed",
            "reason": str(error),
            "seed_substituted": False,
        }

    world_record = {
        "world_identity": clip.world.identity,
        "world": clip.world.to_dict(),
        "generation_report": clip.report.to_dict(),
    }

    labels: list[np.ndarray] = []
    single_scores: list[np.ndarray] = []
    dense_scores: list[np.ndarray] = []
    for rendered_window in clip.windows:
        window = assemble_window(
            sequence.spec,
            rendered_window.window_start,
            rendered_window.frame_ids,
            tuple(item.source for item in rendered_window.rendered_frames),
            canonical_ray_by_slot=ray_grid.canonical_ray_by_slot,
            ray_mapping_audited=True,
            ray_mapping_digest=ray_digest,
        )
        current = window.current_frame.source
        if current.labels is None:
            raise EvaluationError("proxy scoring requires rendered labels")
        slots = current.real_slots
        semantic = current.labels.semantic[slots]
        ranges = np.linalg.norm(current.xyzi[slots, :3], axis=1)
        valid = (semantic != 0) & protocol.evaluation.range_mask(ranges)
        frame_truth = semantic[valid] == np.uint16(2)
        if np.count_nonzero(frame_truth) < protocol.evaluation.minimum_anomaly_points:
            continue
        paired = score_window(model, window)
        labels.append(frame_truth)
        single_scores.append(paired.single_score[valid])
        dense_scores.append(paired.dense_current_score[valid])
    if not labels:
        return {
            **base,
            "status": "unevaluable_no_current_frame_with_five_anomaly_points",
            **world_record,
            "candidate_current_frames": len(clip.windows),
            "evaluated_current_frames": 0,
            "seed_substituted": False,
        }

    truth = np.concatenate(labels)
    single = official_point_metrics(truth, np.concatenate(single_scores))
    dense = official_point_metrics(truth, np.concatenate(dense_scores))
    return {
        **base,
        "status": "evaluated",
        **world_record,
        "candidate_current_frames": len(clip.windows),
        "evaluated_current_frames": len(labels),
        "normal_points": int(truth.size - np.count_nonzero(truth)),
        "anomaly_points": int(np.count_nonzero(truth)),
        "single": single,
        "dense": dense,
        "delta": {
            name: dense[name] - single[name] for name in ("AP", "AUROC", "FPR95")
        },
        "seed_substituted": False,
    }


def _f3_phase_summary(
    records: Sequence[Mapping[str, object]], *, repetitions: int, seed: int
) -> dict[str, object]:
    status_counts = {
        status: sum(item["status"] == status for item in records)
        for status in sorted({str(item["status"]) for item in records})
    }
    evaluated = [item for item in records if item["status"] == "evaluated"]
    if not evaluated:
        return {
            "planned_worlds": len(records),
            "evaluable_worlds": 0,
            "status_counts": status_counts,
            "positive_delta_AP_worlds": 0,
            "mean_delta_AP": None,
            "median_delta_AP": None,
            "paired_world_bootstrap_95_interval": None,
        }
    differences = np.asarray(
        [item["delta"]["AP"] for item in evaluated], dtype=np.float64
    )
    interval = _paired_bootstrap(differences, repetitions, seed)
    return {
        "planned_worlds": len(records),
        "evaluable_worlds": len(evaluated),
        "status_counts": status_counts,
        "positive_delta_AP_worlds": int(np.count_nonzero(differences > 0.0)),
        "mean_delta_AP": float(np.mean(differences)),
        "median_delta_AP": float(np.median(differences)),
        "paired_world_bootstrap_95_interval": list(interval),
    }


def f3_screen_action(summary: Mapping[str, object]) -> str:
    """Apply the frozen eight-world rule without hiding unevaluable worlds."""

    complete = summary["evaluable_worlds"] == summary["planned_worlds"]
    interval = summary["paired_world_bootstrap_95_interval"]
    if complete and interval is not None and interval[0] > 0.0:
        return "support"
    if (
        complete
        and summary["mean_delta_AP"] is not None
        and summary["mean_delta_AP"] <= 0.0
        and summary["median_delta_AP"] <= 0.0
    ):
        return "reject"
    return "extend"


def run_f3_proxy_signal(
    data_root: Path,
    output_dir: Path,
    *,
    protocol: AJAEProtocol,
    device: str,
    encoder: FrozenSTUPointEncoder | None = None,
) -> dict[str, object]:
    sequence = _open_normal_201(data_root, protocol)
    settings = protocol.feasibility["F3_proxy_signal"]
    model = (
        FrozenSTUPointEncoder.from_protocol(protocol) if encoder is None else encoder
    )
    model.to(torch.device(device)).eval()
    protocol.verify_official_point_evaluator()
    calibration_path = protocol.verify_sensor_calibration()
    ray_grid, sensor = load_sensor_calibration(calibration_path)
    ray_digest = canonical_ray_mapping_digest(ray_grid.canonical_ray_by_slot)
    support_pool = load_qualified_support_pool(
        protocol.verify_support_pool(201), source_sequence_id=201
    )
    repetitions = int(settings["bootstrap_repetitions"])
    bootstrap_seed = int(settings["bootstrap_seed"])

    def execute_plan(name: str, offset: int) -> list[dict[str, object]]:
        plan = settings[name]
        return [
            _f3_world_record(
                phase=name,
                world_index=offset + index,
                source_start=int(start),
                root_seed=int(seed),
                sequence=sequence,
                model=model,
                protocol=protocol,
                ray_grid=ray_grid,
                sensor=sensor,
                ray_digest=ray_digest,
                support_pool=support_pool,
            )
            for index, (start, seed) in enumerate(
                zip(plan["source_starts"], plan["world_root_seeds"], strict=True)
            )
        ]

    screen_records = execute_plan("screen", 0)
    screen = _f3_phase_summary(
        screen_records, repetitions=repetitions, seed=bootstrap_seed
    )
    screen_action = f3_screen_action(screen)
    if screen_action == "support":
        status = "screen_supported_pending_F2"
        extension_records: list[dict[str, object]] = []
    elif screen_action == "reject":
        status = "screen_rejected_enter_F4"
        extension_records = []
    else:
        extension_records = execute_plan(
            "extension_if_screen_is_inconclusive", len(screen_records)
        )
        final = _f3_phase_summary(
            [*screen_records, *extension_records],
            repetitions=repetitions,
            seed=bootstrap_seed,
        )
        final_interval = final["paired_world_bootstrap_95_interval"]
        status = (
            "final_supported_pending_F2"
            if final_interval is not None and final_interval[0] > 0.0
            else "final_not_supported_enter_F4"
        )
    records = [*screen_records, *extension_records]
    summary = {
        "format": "ajae-schema33-F3-proxy-signal-v2",
        "protocol_identity": protocol.scientific_identity,
        "status": status,
        "metric_scale": "percent",
        "screen": screen,
        "final": (
            _f3_phase_summary(records, repetitions=repetitions, seed=bootstrap_seed)
            if extension_records
            else None
        ),
        "worlds": records,
        "direct_dense_STU_requires_separate_F2_pass": True,
    }
    _atomic_json(output_dir / "F3_proxy_signal.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AJAE schema-33 feasibility experiments"
    )
    parser.add_argument("experiment", choices=("F1", "F2", "F3"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ajae/schema33"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser


def require_experiment_stage(protocol: AJAEProtocol, experiment: str) -> None:
    """Prevent running a feasibility experiment outside its active stage."""

    stage = str(protocol.status["current_stage"])
    if experiment not in {"F1", "F2", "F3"} or stage != experiment:
        raise EvaluationError(
            f"protocol stage {stage} does not authorize experiment {experiment}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = (
        load_protocol() if args.protocol is None else load_protocol(args.protocol)
    )
    require_experiment_stage(protocol, args.experiment)
    if args.experiment == "F1":
        result = run_f1_geometry(args.data_root, args.output_dir, protocol=protocol)
    elif args.experiment == "F2":
        result = run_f2_normal_stability(
            args.data_root, args.output_dir, protocol=protocol, device=args.device
        )
    else:
        result = run_f3_proxy_signal(
            args.data_root, args.output_dir, protocol=protocol, device=args.device
        )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"records", "frames", "worlds"}
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
