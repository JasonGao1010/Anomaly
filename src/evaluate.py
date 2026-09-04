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

try:
    from .model import FrozenSTUPointEncoder, STUPointEncoding
    from .protocol import AJAEProtocol, load_protocol
    from .render import (
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
    """Frozen-STU outputs on exactly the same current-frame returns."""

    single_score: np.ndarray
    dense_score: np.ndarray
    single_class: np.ndarray
    dense_class: np.ndarray


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
    return _numpy(encoding.normal_class).astype(np.uint8)


def score_window(
    encoder: FrozenSTUPointEncoder,
    window: SceneWindow,
) -> PairedScores:
    """Compare frozen STU while reading only the current scan in both inputs."""

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
        dense_score=_numpy(dense.maxlogit_score.index_select(0, rows)).astype(
            np.float64
        ),
        single_class=_classes(single),
        dense_class=_numpy(dense.normal_class.index_select(0, rows)).astype(np.uint8),
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
    summary = {
        "format": "ajae-schema33-F1-geometry-v1",
        "protocol_identity": protocol.scientific_identity,
        "status": "completed_descriptive",
        "window_count": len(records),
        "visible_return_ratio_median": float(np.median(ratios)),
        "visible_return_ratio_q05_q95": np.quantile(ratios, (0.05, 0.95)).tolist(),
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


def _normal_target(window: SceneWindow) -> np.ndarray:
    labels = window.current_frame.source.labels
    if labels is None or labels.semantic_target is None:
        raise EvaluationError("normal stability requires mapped train/201 labels")
    return np.asarray(
        labels.semantic_target[window.current_frame.source.real_slots], dtype=np.uint8
    )


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
    for current_frame in settings["current_frames"]:
        window = sequence.window(int(current_frame) - 4)
        paired = score_window(model, window)
        target = _normal_target(window)
        valid = target != np.uint8(255)
        if not np.any(valid):
            raise EvaluationError(
                f"current frame {current_frame} has no valid normal labels"
            )
        single_scores.append(paired.single_score[valid])
        dense_scores.append(paired.dense_score[valid])
        targets.append(target[valid])
        single_classes.append(paired.single_class[valid])
        dense_classes.append(paired.dense_class[valid])
        frame_records.append(
            {
                "current_frame": int(current_frame),
                "valid_points": int(np.count_nonzero(valid)),
                "single_MaxLogit_median": float(np.median(paired.single_score[valid])),
                "dense_MaxLogit_median": float(np.median(paired.dense_score[valid])),
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
        "point_count": int(target.size),
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


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.bool_)
    value = np.asarray(scores, dtype=np.float64)
    if truth.shape != value.shape or truth.ndim != 1 or not np.isfinite(value).all():
        raise EvaluationError("AP inputs must be aligned finite vectors")
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        raise EvaluationError("AP requires at least one anomaly point")
    if positives == truth.size:
        raise EvaluationError("AP requires at least one normal point")
    order = np.argsort(-value, kind="stable")
    ranked = truth[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(np.sum(precision[ranked]) / positives)


def _paired_bootstrap(
    values: np.ndarray, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[sample].mean(axis=1)
    return tuple(map(float, np.quantile(means, (0.025, 0.975))))


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
    ray_grid, sensor = load_sensor_calibration(protocol.sensor_calibration_path())
    ray_digest = canonical_ray_mapping_digest(ray_grid.canonical_ray_by_slot)
    support_pool = load_qualified_support_pool(
        protocol.support_pool_path(201), source_sequence_id=201
    )
    world_records: list[dict[str, object]] = []
    for world_index, start_value in enumerate(settings["source_starts"]):
        start = int(start_value)
        length = int(settings["frames_per_sequence"])
        sources = tuple(
            sequence.source_frame(frame) for frame in range(start, start + length)
        )
        obstacles = collect_observed_obstacle_index(sources, source_sequence_id=201)
        clip = sample_development_clip_world(
            support_pool,
            obstacles,
            sources,
            ray_grid,
            sensor,
            33_000 + world_index,
            renderer_identity=protocol.scientific_identity,
        )
        labels: list[np.ndarray] = []
        single_scores: list[np.ndarray] = []
        dense_scores: list[np.ndarray] = []
        candidate_frames = 0
        for rendered_window in clip.windows:
            candidate_frames += 1
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
            if (
                np.count_nonzero(frame_truth)
                < protocol.evaluation.minimum_anomaly_points
            ):
                continue
            paired = score_window(model, window)
            labels.append(frame_truth)
            single_scores.append(paired.single_score[valid])
            dense_scores.append(paired.dense_score[valid])
        if not labels:
            raise EvaluationError(
                f"F3 world {world_index} has no eligible current frame"
            )
        truth = np.concatenate(labels)
        single = average_precision(truth, np.concatenate(single_scores))
        dense = average_precision(truth, np.concatenate(dense_scores))
        world_records.append(
            {
                "world_index": world_index,
                "source_start": start,
                "world_identity": clip.world.identity,
                "candidate_current_frames": candidate_frames,
                "evaluated_current_frames": len(labels),
                "anomaly_points": int(np.count_nonzero(truth)),
                "single_AP": single,
                "dense_AP": dense,
                "delta_AP": dense - single,
            }
        )
    differences = np.asarray(
        [item["delta_AP"] for item in world_records], dtype=np.float64
    )
    interval = _paired_bootstrap(
        differences,
        int(settings["bootstrap_repetitions"]),
        int(settings["bootstrap_seed"]),
    )
    summary = {
        "format": "ajae-schema33-F3-proxy-signal-v1",
        "protocol_identity": protocol.scientific_identity,
        "status": "positive" if interval[0] > 0.0 else "not_positive",
        "world_count": len(world_records),
        "mean_delta_AP": float(np.mean(differences)),
        "paired_world_bootstrap_95_interval": list(interval),
        "worlds": world_records,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = (
        load_protocol() if args.protocol is None else load_protocol(args.protocol)
    )
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
