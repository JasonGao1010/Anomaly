#!/usr/bin/env python3
"""Mechanical qualification for the compact schema-33 feasibility path."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

try:
    from .evaluate import score_window, window_stu_inputs
    from .model import (
        FrozenSTUPointEncoder,
        MASK_DIM,
        NUM_NORMAL_CLASSES,
        NUM_QUERIES,
        STUPointEncoding,
        assigned_stu_evidence,
        official_stu_semantic_class,
        official_stu_sparse_quantize,
        stu_input_identity,
    )
    from .protocol import AJAEProtocol, FrameSpan, SequenceSpec, load_protocol
    from .scene import LabelMode, PointLabels, STUSequence, assemble_window, make_source_frame
except ImportError:  # Direct script execution.
    from evaluate import score_window, window_stu_inputs
    from model import (
        FrozenSTUPointEncoder,
        MASK_DIM,
        NUM_NORMAL_CLASSES,
        NUM_QUERIES,
        STUPointEncoding,
        assigned_stu_evidence,
        official_stu_semantic_class,
        official_stu_sparse_quantize,
        stu_input_identity,
    )
    from protocol import AJAEProtocol, FrameSpan, SequenceSpec, load_protocol
    from scene import LabelMode, PointLabels, STUSequence, assemble_window, make_source_frame


class QualificationError(AssertionError):
    """Report a failed schema-33 semantic invariant."""


def _module_from_file(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise QualificationError(f"cannot load official STU module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_lidar_module(path: Path) -> object:
    """Load inference-only STU data code without its unused augmentation package."""

    if importlib.util.find_spec("volumentations") is not None:
        return _module_from_file("_ajae_official_stu_lidar", path)
    sys.modules["volumentations"] = types.ModuleType("volumentations")
    try:
        return _module_from_file("_ajae_official_stu_lidar", path)
    finally:
        sys.modules.pop("volumentations", None)


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


def _official_single_frame_equivalence(
    data_root: Path,
    frame_ids: Sequence[int],
    *,
    protocol: AJAEProtocol,
    device: str,
) -> dict[str, object]:
    """Compare official sweep=1 loading and collation with the AJAE bridge."""

    frames = tuple(int(value) for value in frame_ids)
    if not 1 <= len(frames) <= 3 or len(set(frames)) != len(frames):
        raise QualificationError("real equivalence requires one to three unique frames")
    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    repository = protocol.stu_repository_path()
    encoder = FrozenSTUPointEncoder.from_protocol(protocol)
    encoder.to(torch.device(device)).eval()
    lidar_module = _official_lidar_module(repository / "datasets/lidar.py")
    utils_module = _module_from_file(
        "_ajae_official_stu_dataset_utils", repository / "datasets/utils.py"
    )
    lidar_class = lidar_module.LidarDataset
    dataset = lidar_class.__new__(lidar_class)
    dataset.mode = "validation"
    dataset.add_distance = True
    dataset.ignore_label = 255
    dataset.instance_population = 0
    dataset.sweep = 1
    dataset.config = dataset._load_yaml(repository / "conf/semantic-kitti.yaml")
    dataset.label_info = dataset._select_correct_labels(
        dataset.config["learning_ignore"]
    )
    collate = utils_module.VoxelizeCollate(ignore_label=255, voxel_size=0.05)
    me = importlib.import_module("MinkowskiEngine")
    records: list[dict[str, object]] = []

    for frame_id in frames:
        source = sequence.source_frame(frame_id)
        scan_path = sequence.sequence_dir / "velodyne" / f"{frame_id:06d}.bin"
        label_path = sequence.sequence_dir / "labels" / f"{frame_id:06d}.label"
        dataset.data = [[{
            "filepath": str(scan_path),
            "label_filepath": str(label_path),
            "scene": 201,
            "pose": source.lidar_pose.tolist(),
        }]]
        official_sample = dataset[0]
        official_coordinates = np.asarray(official_sample["coordinates"])
        official_features = np.asarray(official_sample["features"][:, 4:], dtype=np.float32)
        np.testing.assert_array_equal(official_coordinates, source.coordinates)
        np.testing.assert_array_equal(official_features, source.features)
        official_data, _ = collate([official_sample])
        ajae_coordinates, ajae_features, ajae_unique, ajae_inverse = (
            official_stu_sparse_quantize(
                source.coordinates,
                source.features,
                official_repository=repository,
            )
        )
        if not isinstance(ajae_features, torch.Tensor):
            ajae_features = torch.from_numpy(np.asarray(ajae_features))
        ajae_collated_coordinates, ajae_collated_features = me.utils.sparse_collate(
            [ajae_coordinates], [ajae_features.float()]
        )
        ajae_unique_np = np.asarray(ajae_unique, dtype=np.int64)
        ajae_raw_coordinates = torch.from_numpy(
            np.column_stack(
                (
                    source.coordinates[ajae_unique_np],
                    np.zeros(ajae_unique_np.size, dtype=np.float64),
                )
            )
        ).float()
        for name, official_value, ajae_value in (
            ("sparse coordinates", official_data.coordinates, ajae_collated_coordinates),
            ("sparse features", official_data.features, ajae_collated_features),
            ("raw coordinates", official_data.raw_coordinates, ajae_raw_coordinates),
        ):
            if not torch.equal(official_value.cpu(), ajae_value.cpu()):
                raise QualificationError(f"AJAE {name} differs from official sweep=1")
        if not np.array_equal(
            np.asarray(official_data.inverse_maps[0]), np.asarray(ajae_inverse)
        ):
            raise QualificationError("AJAE full inverse map differs from official sweep=1")
        sparse = me.SparseTensor(
            coordinates=official_data.coordinates,
            features=official_data.features,
            device=encoder.device,
        )
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(encoder.device)
            if encoder.device.type == "cuda"
            else None
        )
        with torch.no_grad():
            output = encoder.stu(
                sparse,
                raw_coordinates=official_data.raw_coordinates,
                is_eval=True,
            )
        logits = encoder._single_prediction(output["pred_logits"], "pred_logits")
        masks = encoder._single_prediction(output["pred_masks"], "pred_masks")
        official_evidence = assigned_stu_evidence(logits, masks)
        full_inverse = torch.as_tensor(
            official_data.inverse_maps[0], dtype=torch.long, device=encoder.device
        )
        real_slots = torch.as_tensor(
            np.asarray(source.real_slots).copy(),
            dtype=torch.long,
            device=encoder.device,
        )
        official_inverse = full_inverse[real_slots]
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, encoder.device)
        observed = encoder(source.coordinates, source.features, source.real_slots)
        if not torch.equal(observed.inverse_map, official_inverse):
            raise QualificationError("AJAE inverse map differs from official sweep=1")
        official_score = official_evidence.maxlogit_score[official_inverse]
        if not torch.allclose(
            observed.maxlogit_score, official_score, atol=1.0e-6, rtol=1.0e-6
        ):
            maximum_error = float(
                torch.max(torch.abs(observed.maxlogit_score - official_score)).item()
            )
            raise QualificationError(
                f"AJAE MaxLogit differs from official sweep=1 by {maximum_error}"
            )
        official_class = official_evidence.normal_class[official_inverse]
        if not torch.equal(observed.normal_class, official_class):
            mismatches = int(torch.count_nonzero(observed.normal_class != official_class))
            raise QualificationError(
                f"AJAE normal class differs from official sweep=1 at {mismatches} points"
            )
        records.append(
            {
                "frame_id": frame_id,
                "file_slots": int(source.slot_count),
                "real_returns": int(source.real_count),
                "sparse_voxels": int(masks.shape[0]),
                "maximum_MaxLogit_absolute_error": float(
                    torch.max(
                        torch.abs(observed.maxlogit_score - official_score)
                    ).item()
                ),
                "normal_class_exact": True,
                "inverse_map_exact": True,
            }
        )
    return {
        "passed": True,
        "official_path": "LidarDataset(sweep=1)->VoxelizeCollate->official_model",
        "ajae_path": "STUSequence.SourceFrame->FrozenSTUPointEncoder",
        "frames": records,
    }


def run_schema33_qualification(
    *,
    data_root: Path | None = None,
    real_frame_ids: Sequence[int] = (8, 198, 387),
    device: str = "cpu",
) -> dict[str, object]:
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
    result: dict[str, object] = {
        "format": "ajae-schema33-qualification-v1",
        "mechanical": {"passed": True, "check_count": len(results), "checks": results},
        "scientific_status": "pending_real_F1_F2_F3_execution",
        "performance_claim_available": False,
    }
    if data_root is not None:
        protocol = load_protocol()
        result["real_single_frame_equivalence"] = _official_single_frame_equivalence(
            data_root,
            real_frame_ids,
            protocol=protocol,
            device=device,
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify AJAE schema-33 mechanics")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--real-frames", default="8,198,387")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        real_frames = tuple(int(value) for value in args.real_frames.split(","))
    except ValueError as error:
        raise QualificationError("--real-frames must be comma-separated integers") from error
    result = run_schema33_qualification(
        data_root=args.data_root,
        real_frame_ids=real_frames,
        device=args.device,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
