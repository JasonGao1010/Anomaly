#!/usr/bin/env python3
"""Mechanical qualification for the compact schema-33 feasibility path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

try:
    from .evaluate import score_window, window_stu_inputs
    from .model import (
        MASK_DIM,
        NUM_NORMAL_CLASSES,
        NUM_QUERIES,
        STUPointEncoding,
        assigned_stu_evidence,
        official_stu_semantic_class,
        official_stu_sparse_quantize,
        stu_input_identity,
    )
    from .protocol import FrameSpan, SequenceSpec, load_protocol
    from .scene import PointLabels, assemble_window, make_source_frame
except ImportError:  # Direct script execution.
    from evaluate import score_window, window_stu_inputs
    from model import (
        MASK_DIM,
        NUM_NORMAL_CLASSES,
        NUM_QUERIES,
        STUPointEncoding,
        assigned_stu_evidence,
        official_stu_semantic_class,
        official_stu_sparse_quantize,
        stu_input_identity,
    )
    from protocol import FrameSpan, SequenceSpec, load_protocol
    from scene import PointLabels, assemble_window, make_source_frame


class QualificationError(AssertionError):
    """Report a failed schema-33 semantic invariant."""


def _source(frame_id: int, pose_x: float, point_x: float) -> object:
    xyzi = np.asarray(
        [[point_x, 0.0, 0.0, 0.5], [point_x + 1.0, 0.0, 0.0, 0.7]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = pose_x
    packed = np.asarray([40, 40], dtype=np.uint32)
    labels = PointLabels(
        packed=packed,
        semantic=packed.astype(np.uint16),
        instance=np.zeros(2, dtype=np.uint16),
        semantic_target=np.asarray([8, 8], dtype=np.uint8),
    )
    return make_source_frame(
        frame_id,
        xyzi,
        pose,
        labels,
        partition="train",
        sequence_id=201,
    )


def _window() -> object:
    spec = SequenceSpec("train", 201, "fixture", True, FrameSpan(0, 5))
    sources = tuple(_source(index, float(index), 10.0) for index in range(5))
    return assemble_window(spec, 0, tuple(range(5)), sources)


def _latest_frame_alignment() -> dict[str, object]:
    window = _window()
    latest = window.current_frame.source
    rows = window.current_mask
    error = float(
        np.max(
            np.abs(window.points.coordinates[rows] - latest.xyzi[latest.real_slots, :3])
        )
    )
    if error != 0.0:
        raise QualificationError("latest scan did not remain in its native coordinates")
    return {"current_frame": window.current_frame_id, "maximum_error": error}


def _past_frame_alignment() -> dict[str, object]:
    window = _window()
    # Every fixture point has world x = 10 + source pose; current pose is x = 4.
    expected_first_x = 6.0
    observed_first_x = float(window.points.coordinates[0, 0])
    if observed_first_x != expected_first_x:
        raise QualificationError("past scan was not transformed by T_current<-source")
    return {"expected_first_x": expected_first_x, "observed_first_x": observed_first_x}


def _paired_input_rows() -> dict[str, object]:
    inputs = window_stu_inputs(_window())
    if inputs.dense_coordinates.shape[0] != 5 * inputs.single_real_slots.size:
        raise QualificationError("dense pseudo-scan does not contain all five scans")
    if inputs.dense_current_rows.size != inputs.single_real_slots.size:
        raise QualificationError(
            "single and dense outputs do not address the same points"
        )
    return {
        "single_points": int(inputs.single_real_slots.size),
        "dense_points": int(inputs.dense_coordinates.shape[0]),
        "scored_dense_rows": int(inputs.dense_current_rows.size),
    }


def _online_uniqueness() -> dict[str, object]:
    protocol = load_protocol()
    starts = protocol.normal_development.legal_window_starts()
    current = tuple(start + 4 for start in starts)
    if len(current) != len(set(current)) or len(current) != 546:
        raise QualificationError(
            "online windows must produce each development time once"
        )
    return {
        "windows": len(starts),
        "first_current": current[0],
        "last_current": current[-1],
    }


def _route_is_pretraining() -> dict[str, object]:
    protocol = load_protocol()
    if protocol.status["training_allowed"] or set(protocol.methods) != {
        "single_stu",
        "dense_stu",
    }:
        raise QualificationError("schema 33 retained a trainable comparison condition")
    return {"training_allowed": False, "methods": sorted(protocol.methods)}


class _DeterministicEncoder:
    def __call__(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        real_slots: np.ndarray | None = None,
    ) -> STUPointEncoding:
        rows = (
            np.arange(coordinates.shape[0], dtype=np.int64)
            if real_slots is None
            else np.asarray(real_slots, dtype=np.int64)
        )
        count = rows.size
        score = torch.from_numpy(
            np.asarray(features[rows, 0] + features[rows, 1], dtype=np.float32)
        )
        evidence = torch.zeros((count, NUM_NORMAL_CLASSES), dtype=torch.float32)
        evidence[:, 8] = 1.0
        return STUPointEncoding(
            point_features=torch.zeros((count, MASK_DIM)),
            assigned_query=torch.zeros(count, dtype=torch.long),
            normal_evidence=evidence,
            reliability_assign=torch.ones(count),
            reliability_noobj=torch.zeros(count),
            maxlogit_score=score,
            normal_class=torch.full((count,), 8, dtype=torch.long),
            inverse_map=torch.arange(count, dtype=torch.long),
            real_slots=torch.as_tensor(rows, dtype=torch.long),
            input_identity=stu_input_identity(coordinates, features, rows),
        )


def _single_scan_degeneracy() -> dict[str, object]:
    spec = SequenceSpec("train", 201, "fixture", True, FrameSpan(0, 5))
    sources = []
    for frame in range(5):
        xyzi = (
            np.zeros((1, 4), dtype=np.float32)
            if frame < 4
            else np.asarray(((10.0, 0.0, 0.0, 0.5),), dtype=np.float32)
        )
        packed = np.asarray((40,), dtype=np.uint32)
        labels = PointLabels(
            packed=packed,
            semantic=packed.astype(np.uint16),
            instance=np.zeros(1, dtype=np.uint16),
            semantic_target=np.asarray((8,), dtype=np.uint8),
        )
        sources.append(
            make_source_frame(
                frame,
                xyzi,
                np.eye(4, dtype=np.float64),
                labels,
                partition="train",
                sequence_id=201,
            )
        )
    window = assemble_window(spec, 0, tuple(range(5)), tuple(sources))
    inputs = window_stu_inputs(window)
    if not (
        np.array_equal(inputs.single_coordinates, inputs.dense_coordinates)
        and np.array_equal(inputs.single_features, inputs.dense_features)
    ):
        raise QualificationError(
            "dense input does not reduce exactly to the single scan"
        )
    scores = score_window(_DeterministicEncoder(), window)  # type: ignore[arg-type]
    if not (
        np.array_equal(scores.single_score, scores.dense_current_score)
        and np.array_equal(scores.single_class, scores.dense_current_class)
    ):
        raise QualificationError("single-scan degeneration changed its predictions")
    return {"input_points": int(inputs.dense_coordinates.shape[0]), "exact": True}


def _shared_voxel_current_recovery() -> dict[str, object]:
    spec = SequenceSpec("train", 201, "fixture", True, FrameSpan(0, 5))
    sources = tuple(_source(frame, 0.0, 1.001 + 0.001 * frame) for frame in range(5))
    window = assemble_window(spec, 0, tuple(range(5)), sources)
    inputs = window_stu_inputs(window)
    _, _, _, inverse = official_stu_sparse_quantize(
        inputs.dense_coordinates, inputs.dense_features
    )
    inverse_map = np.asarray(inverse, dtype=np.int64)
    current = int(inputs.dense_current_rows[0])
    if inverse_map.shape != (inputs.dense_coordinates.shape[0],) or not np.any(
        inverse_map[:current] == inverse_map[current]
    ):
        raise QualificationError(
            "official voxel inverse map did not recover a shared-voxel current point"
        )
    return {
        "dense_points": int(inverse_map.size),
        "current_row": current,
        "shared_voxel_row": int(inverse_map[current]),
    }


def _official_semantic_equivalence() -> dict[str, object]:
    logits = torch.full((NUM_QUERIES, NUM_NORMAL_CLASSES + 1), -20.0)
    logits[:, -1] = 20.0
    logits[0, 0], logits[0, -1] = 5.0, 0.0
    logits[1, 1], logits[1, -1] = 4.0, 0.0
    masks = torch.full((1, NUM_QUERIES), -20.0)
    masks[0, 0], masks[0, 1] = 3.0, 1.0
    evidence = assigned_stu_evidence(logits, masks)
    reference = official_stu_semantic_class(logits, masks)
    if not torch.equal(evidence.normal_class, reference):
        raise QualificationError("normal_class differs from official query semantics")
    return {"checked_voxels": 1, "class": int(reference[0])}


def run_schema33_qualification() -> dict[str, object]:
    checks: tuple[tuple[str, Callable[[], dict[str, object]]], ...] = (
        ("latest_scan_is_the_coordinate_frame", _latest_frame_alignment),
        ("past_scans_use_current_from_source_transform", _past_frame_alignment),
        ("single_and_dense_score_the_same_current_points", _paired_input_rows),
        ("one_online_output_per_current_frame", _online_uniqueness),
        ("active_route_contains_no_trainable_model", _route_is_pretraining),
        ("dense_degenerates_exactly_to_single_scan", _single_scan_degeneracy),
        ("shared_voxel_inverse_recovers_current_point", _shared_voxel_current_recovery),
        (
            "normal_class_matches_official_query_semantics",
            _official_semantic_equivalence,
        ),
    )
    results = []
    for name, check in checks:
        results.append({"name": name, "passed": True, "details": check()})
    return {
        "format": "ajae-schema33-qualification-v1",
        "mechanical": {"passed": True, "check_count": len(results), "checks": results},
        "scientific_status": "pending_real_F1_F2_F3_execution",
        "performance_claim_available": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify AJAE schema-33 mechanics")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_schema33_qualification()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
