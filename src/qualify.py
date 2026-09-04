#!/usr/bin/env python3
"""Mechanical qualification for the compact schema-33 feasibility path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import types
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

try:
    from .evaluate import (
        require_clean_implementation,
        score_window,
        window_stu_inputs,
    )
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
    from evaluate import (
        require_clean_implementation,
        score_window,
        window_stu_inputs,
    )
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


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _restore_rng(
    cpu_state: torch.Tensor,
    cuda_state: torch.Tensor | None,
    device: torch.device,
) -> None:
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _official_projection(
    encoder: FrozenSTUPointEncoder,
    official_data: object,
    selected_rows: np.ndarray,
) -> dict[str, np.ndarray | int]:
    me = importlib.import_module("MinkowskiEngine")
    sparse = me.SparseTensor(
        coordinates=official_data.coordinates,
        features=official_data.features,
        device=encoder.device,
    )
    with torch.no_grad():
        output = encoder.stu(
            sparse,
            raw_coordinates=official_data.raw_coordinates,
            is_eval=True,
        )
    logits = encoder._single_prediction(output["pred_logits"], "pred_logits")
    masks = encoder._single_prediction(output["pred_masks"], "pred_masks")
    evidence = assigned_stu_evidence(logits, masks)
    full_inverse = torch.as_tensor(
        official_data.inverse_maps[0], dtype=torch.long, device=encoder.device
    )
    rows = torch.as_tensor(selected_rows, dtype=torch.long, device=encoder.device)
    inverse = full_inverse[rows]
    return {
        "score": evidence.maxlogit_score[inverse].detach().cpu().numpy(),
        "normal_class": evidence.normal_class[inverse].detach().cpu().numpy(),
        "inverse_map": inverse.detach().cpu().numpy(),
        "sparse_voxels": int(masks.shape[0]),
    }


def _ajae_projection(
    encoder: FrozenSTUPointEncoder,
    coordinates: np.ndarray,
    features: np.ndarray,
    selected_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    encoding = encoder(coordinates, features, selected_rows)
    result = {
        "score": encoding.maxlogit_score.detach().cpu().numpy(),
        "normal_class": encoding.normal_class.detach().cpu().numpy(),
        "inverse_map": encoding.inverse_map.detach().cpu().numpy(),
    }
    del encoding
    gc.collect()
    if encoder.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _maximum_absolute_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _equivalence_case(
    *,
    sequence: STUSequence,
    encoder: FrozenSTUPointEncoder,
    dataset: object,
    collate: object,
    frame_ids: tuple[int, ...],
    coordinates: np.ndarray,
    features: np.ndarray,
    selected_rows: np.ndarray,
    current_rows: np.ndarray,
    repeat_runs: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    repository = encoder.official_repository
    sources = tuple(sequence.source_frame(frame_id) for frame_id in frame_ids)
    dataset.sweep = len(frame_ids)
    dataset.data = [[
        {
            "filepath": str(
                sequence.sequence_dir / "velodyne" / f"{source.frame_id:06d}.bin"
            ),
            "label_filepath": str(
                sequence.sequence_dir / "labels" / f"{source.frame_id:06d}.label"
            ),
            "scene": 201,
            "pose": source.lidar_pose.tolist(),
        }
        for source in sources
    ]]
    official_sample = dataset[0]
    official_coordinates = np.asarray(official_sample["coordinates"])
    official_features = np.asarray(
        official_sample["features"][:, 4:], dtype=np.float32
    )
    if not np.array_equal(official_coordinates, coordinates):
        error = _maximum_absolute_error(official_coordinates, coordinates)
        raise QualificationError(
            f"AJAE coordinates differ from official sweep={len(frame_ids)} by {error}"
        )
    if not np.array_equal(official_features, features):
        error = _maximum_absolute_error(official_features, features)
        raise QualificationError(
            f"AJAE features differ from official sweep={len(frame_ids)} by {error}"
        )

    official_data, _ = collate([official_sample])
    sparse_c, sparse_f, unique, inverse = official_stu_sparse_quantize(
        coordinates,
        features,
        official_repository=repository,
    )
    me = importlib.import_module("MinkowskiEngine")
    if not isinstance(sparse_f, torch.Tensor):
        sparse_f = torch.from_numpy(np.asarray(sparse_f))
    collated_c, collated_f = me.utils.sparse_collate(
        [sparse_c], [sparse_f.float()]
    )
    unique_np = np.asarray(unique, dtype=np.int64)
    raw_spatial = torch.from_numpy(coordinates[unique_np]).float()
    comparisons = (
        ("sparse coordinates", official_data.coordinates, collated_c),
        ("sparse features", official_data.features, collated_f),
        ("raw spatial coordinates", official_data.raw_coordinates[:, :3], raw_spatial),
    )
    for name, official_value, ajae_value in comparisons:
        if not torch.equal(official_value.cpu(), ajae_value.cpu()):
            raise QualificationError(
                f"AJAE {name} differs from official sweep={len(frame_ids)}"
            )
    if not np.array_equal(
        np.asarray(official_data.inverse_maps[0]), np.asarray(inverse)
    ):
        raise QualificationError(
            f"AJAE full inverse map differs from official sweep={len(frame_ids)}"
        )

    cpu_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state(encoder.device)
        if encoder.device.type == "cuda"
        else None
    )
    official = _official_projection(encoder, official_data, selected_rows)
    observed_runs: list[dict[str, np.ndarray]] = []
    for _ in range(repeat_runs):
        _restore_rng(cpu_state, cuda_state, encoder.device)
        observed_runs.append(
            _ajae_projection(encoder, coordinates, features, selected_rows)
        )
    observed = observed_runs[0]
    official_score = np.asarray(official["score"])
    official_class = np.asarray(official["normal_class"])
    official_inverse = np.asarray(official["inverse_map"])
    score_error = _maximum_absolute_error(observed["score"], official_score)
    class_mismatches = int(
        np.count_nonzero(observed["normal_class"] != official_class)
    )
    if not np.allclose(
        observed["score"],
        official_score,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise QualificationError(
            f"AJAE MaxLogit differs from official sweep={len(frame_ids)} by {score_error}"
        )
    if class_mismatches or not np.array_equal(
        observed["inverse_map"], official_inverse
    ):
        raise QualificationError(
            f"AJAE classes or selected inverse map differ from official sweep={len(frame_ids)}"
        )

    repeat_score_error = 0.0
    repeat_class_mismatches = 0
    repeat_inverse_exact = True
    for repeated in observed_runs[1:]:
        repeat_score_error = max(
            repeat_score_error,
            _maximum_absolute_error(repeated["score"], observed["score"]),
        )
        repeat_class_mismatches += int(
            np.count_nonzero(repeated["normal_class"] != observed["normal_class"])
        )
        repeat_inverse_exact &= np.array_equal(
            repeated["inverse_map"], observed["inverse_map"]
        )
    if repeat_score_error != 0.0 or repeat_class_mismatches or not repeat_inverse_exact:
        raise QualificationError(
            f"AJAE sweep={len(frame_ids)} repeatability is not exact"
        )

    current_score_error = _maximum_absolute_error(
        observed["score"][current_rows], official_score[current_rows]
    )
    current_class_mismatches = int(
        np.count_nonzero(
            observed["normal_class"][current_rows]
            != official_class[current_rows]
        )
    )
    sparse_voxels = int(official["sparse_voxels"])
    del official_data, official, observed_runs
    gc.collect()
    return {
        "passed": True,
        "frame_ids": list(frame_ids),
        "file_slots": int(coordinates.shape[0]),
        "real_returns": int(selected_rows.size),
        "current_real_returns": int(current_rows.size),
        "sparse_voxels": sparse_voxels,
        "input_coordinates_exact": True,
        "input_features_exact": True,
        "sparse_coordinates_exact": True,
        "sparse_features_exact": True,
        "raw_spatial_coordinates_exact": True,
        "full_inverse_map_exact": True,
        "official_vs_AJAE_max_abs_MaxLogit_error": score_error,
        "official_vs_AJAE_class_mismatches": class_mismatches,
        "selected_inverse_map_exact": True,
        "AJAE_repeat_max_abs_MaxLogit_error": repeat_score_error,
        "AJAE_repeat_class_mismatches": repeat_class_mismatches,
        "AJAE_repeat_inverse_map_exact": repeat_inverse_exact,
        "current_view_max_abs_MaxLogit_error": current_score_error,
        "current_view_class_mismatches": current_class_mismatches,
        "official_time_ids_present": len(frame_ids) > 1,
        "time_ids_used_by_Mask4Former3D": False,
    }


def _official_real_equivalence(
    data_root: Path,
    *,
    protocol: AJAEProtocol,
    device: str,
) -> dict[str, object]:
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
    dataset = lidar_module.LidarDataset.__new__(lidar_module.LidarDataset)
    dataset.mode = "validation"
    dataset.add_distance = True
    dataset.ignore_label = 255
    dataset.instance_population = 0
    dataset.config = dataset._load_yaml(repository / "conf/semantic-kitti.yaml")
    dataset.label_info = dataset._select_correct_labels(
        dataset.config["learning_ignore"]
    )
    collate = utils_module.VoxelizeCollate(ignore_label=255, voxel_size=0.05)
    settings = protocol.stu["F0_qualification"]
    repeat_runs = int(settings["AJAE_repeat_runs"])
    absolute_tolerance = float(settings["MaxLogit_absolute_tolerance"])
    relative_tolerance = float(settings["MaxLogit_relative_tolerance"])

    single_records = []
    for value in settings["single_frame_ids"]:
        frame_id = int(value)
        source = sequence.source_frame(frame_id)
        selected = source.real_slots.astype(np.int64)
        single_records.append(
            _equivalence_case(
                sequence=sequence,
                encoder=encoder,
                dataset=dataset,
                collate=collate,
                frame_ids=(frame_id,),
                coordinates=source.coordinates,
                features=source.features,
                selected_rows=selected,
                current_rows=np.arange(selected.size, dtype=np.int64),
                repeat_runs=repeat_runs,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        )

    five_scan_records = []
    for value in settings["five_scan_window_starts"]:
        start = int(value)
        inputs = window_stu_inputs(sequence.window(start))
        five_scan_records.append(
            _equivalence_case(
                sequence=sequence,
                encoder=encoder,
                dataset=dataset,
                collate=collate,
                frame_ids=tuple(range(start, start + 5)),
                coordinates=inputs.dense_coordinates,
                features=inputs.dense_features,
                selected_rows=inputs.dense_real_slots,
                current_rows=inputs.dense_current_rows,
                repeat_runs=repeat_runs,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        )
    return {
        "passed": True,
        "device": str(encoder.device),
        "AJAE_repeat_runs": repeat_runs,
        "official_path": "LidarDataset(sweep)->VoxelizeCollate->official_model",
        "AJAE_path": "STUSequence/SceneWindow->FrozenSTUPointEncoder",
        "single_scan": single_records,
        "five_scan": five_scan_records,
    }


def run_schema33_qualification(
    *,
    data_root: Path | None = None,
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
        "format": "ajae-schema33-F0-qualification-v2",
        "mechanical": {"passed": True, "check_count": len(results), "checks": results},
        "scientific_status": "pending_real_F1_F2_F3_execution",
        "performance_claim_available": False,
        "F0_verdict": "not_run_without_real_data",
    }
    if data_root is not None:
        protocol = load_protocol()
        implementation = require_clean_implementation(protocol)
        implementation["source_files_sha256"] = {
            **implementation["source_files_sha256"],
            "src/qualify.py": _file_sha256(Path(__file__).resolve()),
        }
        result.update(
            {
                "contract_identity": protocol.contract_identity,
                "protocol_file_sha256": protocol.execution_identity,
                "implementation_identity": implementation,
                "environment": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "torch_CUDA_build": torch.version.cuda,
                    "MinkowskiEngine": importlib.metadata.version(
                        "MinkowskiEngine"
                    ),
                    "PyTorch3D": importlib.metadata.version("pytorch3d"),
                    "logical_CPUs": os.cpu_count(),
                    "torch_threads": torch.get_num_threads(),
                    "execution_device": str(torch.device(device)),
                },
            }
        )
        result["real_equivalence"] = _official_real_equivalence(
            data_root,
            protocol=protocol,
            device=device,
        )
        result["F0_verdict"] = "passed"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify AJAE schema-33 mechanics")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_schema33_qualification(
        data_root=args.data_root,
        device=args.device,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
