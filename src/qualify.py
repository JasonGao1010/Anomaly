#!/usr/bin/env python3
"""Execute the frozen post-Gate-1 mechanical qualifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .model import (
    DEFAULT_STU_REPOSITORY,
    MASK_DIM,
    FrozenSTUPointEncoder,
    stu_source_manifest,
    stu_weight_identity,
)
from .protocol import load_protocol
from .scene import (
    LabelMode,
    STUSequence,
    official_stu_coordinates,
    official_stu_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE5_FRAME_NAMESPACE = "E50-phase5-frame-v1"
PHASE5_FRAMES = {
    206: (14, 41, 98, 125, 189, 199, 272, 304, 329, 347, 378, 385, 386, 387, 407, 409),
    201: (16, 67, 176, 239, 245, 289, 337, 344, 416, 417, 423, 474, 479, 496, 524, 670),
}
FROZEN_STU_SOURCE_MANIFEST_SHA256 = (
    "f0cead4f5e721262f9f1c26231d116406bb4fb0a43139f22e3706be89b914891"
)
E50_ARTIFACT_SHA256 = (
    "2c2d8507df0f9e4c9984118e59c6d65a8f13835590fee5b51bed02c282c5671a"
)


class QualificationError(ValueError):
    """Report an invalid qualification input, identity, or output."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
    digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def phase5_frame_ids(protocol: object, sequence_id: int) -> tuple[int, ...]:
    """Select frames only from frozen identities, never model outputs."""

    spec = protocol.sequence("train", sequence_id)
    ranked = sorted(
        (
            hashlib.sha256(
                f"{PHASE5_FRAME_NAMESPACE}:train:{sequence_id}:{frame_id}".encode(
                    "ascii"
                )
            ).digest(),
            int(frame_id),
        )
        for frame_id in spec.center_frames()
    )
    return tuple(sorted(frame_id for _, frame_id in ranked[:16]))


def independent_sparse_quantize(
    coordinates: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute first-occurrence sparse rows without MinkowskiEngine."""

    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise QualificationError("quantization coordinates must be nonempty [N,3]")
    if not np.isfinite(points).all() or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise QualificationError("quantization input must be finite with positive size")
    rows = np.floor(points / voxel_size).astype(np.int64)
    row_by_key: dict[tuple[int, int, int], int] = {}
    unique_indices: list[int] = []
    inverse = np.empty(points.shape[0], dtype=np.int64)
    for point_index, row in enumerate(rows):
        key = (int(row[0]), int(row[1]), int(row[2]))
        sparse_index = row_by_key.get(key)
        if sparse_index is None:
            sparse_index = len(unique_indices)
            row_by_key[key] = sparse_index
            unique_indices.append(point_index)
        inverse[point_index] = sparse_index
    unique = np.asarray(unique_indices, dtype=np.int64)
    return rows[unique], unique, inverse


def _save(path: Path, arrays: Mapping[str, np.ndarray], result: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
    )
    os.replace(temporary, path)


def run_e50(
    data_root: Path | str,
    protocol_path: Path | str,
    output_path: Path | str,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    """Verify the official frozen STU 128-channel point-feature interface."""

    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    selected: list[tuple[int, int]] = []
    for sequence_id in (206, 201):
        frame_ids = phase5_frame_ids(protocol, sequence_id)
        if frame_ids != PHASE5_FRAMES[sequence_id]:
            raise QualificationError("E50 frozen frame identity changed")
        selected.extend((sequence_id, frame_id) for frame_id in frame_ids)

    source_identity = stu_source_manifest(DEFAULT_STU_REPOSITORY)
    if source_identity["manifest_sha256"] != FROZEN_STU_SOURCE_MANIFEST_SHA256:
        raise QualificationError("E50 official STU source identity changed")
    checkpoint = PROJECT_ROOT / "weights" / "59p6pq_ens1.ckpt"
    weight_identity = stu_weight_identity(checkpoint)
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise QualificationError("E50 requested CUDA but CUDA is unavailable")

    sequences = {
        sequence_id: STUSequence.open(
            data_root,
            protocol=protocol,
            partition="train",
            sequence_id=sequence_id,
            label_mode=LabelMode.FORBIDDEN,
        )
        for sequence_id in (206, 201)
    }
    encoder = FrozenSTUPointEncoder.from_protocol(protocol).to(runtime_device)
    encoder.eval()
    if encoder.training or encoder.stu.training or any(
        parameter.requires_grad for parameter in encoder.stu.parameters()
    ):
        raise QualificationError("E50 STU is not frozen in evaluation mode")

    sequence_array = np.asarray([item[0] for item in selected], dtype=np.int16)
    frame_array = np.asarray([item[1] for item in selected], dtype=np.int16)
    real_count = np.zeros((2, len(selected)), dtype=np.int32)
    voxel_count = np.zeros_like(real_count)
    feature_hash = np.empty((2, len(selected)), dtype="S64")
    seconds = np.zeros((2, len(selected)), dtype=np.float64)
    finite_errors = np.zeros((2, len(selected)), dtype=np.int32)
    shape_errors = np.zeros_like(finite_errors)
    gradient_errors = np.zeros_like(finite_errors)

    started = time.monotonic()
    for repetition in range(2):
        for index, (sequence_id, frame_id) in enumerate(selected):
            frame = sequences[sequence_id].source_frame(frame_id)
            coordinates = official_stu_coordinates(frame.xyzi, frame.lidar_pose)
            features = official_stu_features(frame.xyzi, frame.lidar_pose)
            item_started = time.monotonic()
            encoding = encoder(coordinates, features, frame.real_slots)
            if runtime_device.type == "cuda":
                torch.cuda.synchronize(runtime_device)
            seconds[repetition, index] = time.monotonic() - item_started
            real_count[repetition, index] = frame.real_count
            voxel_count[repetition, index] = int(encoding.inverse_map.max().item()) + 1
            shape_errors[repetition, index] = int(
                encoding.point_features.shape != (frame.real_count, MASK_DIM)
            )
            finite_errors[repetition, index] = int(
                not bool(torch.isfinite(encoding.point_features).all())
            )
            gradient_errors[repetition, index] = int(
                encoding.point_features.requires_grad
            )
            feature_hash[repetition, index] = _tensor_hash(encoding.point_features)
            del encoding
    elapsed = time.monotonic() - started

    identity_errors = int(
        not np.array_equal(real_count[0], real_count[1])
        or not np.array_equal(voxel_count[0], voxel_count[1])
    )
    reproduction_errors = int(np.count_nonzero(feature_hash[0] != feature_hash[1]))
    hard_errors = int(
        finite_errors.sum()
        + shape_errors.sum()
        + gradient_errors.sum()
        + identity_errors
        + reproduction_errors
    )
    passed = hard_errors == 0
    arrays = {
        "sequence_id": sequence_array,
        "frame_id": frame_array,
        "real_count": real_count,
        "voxel_count": voxel_count,
        "point_feature_sha256": feature_hash,
        "frame_seconds": seconds,
        "finite_errors": finite_errors,
        "shape_errors": shape_errors,
        "gradient_errors": gradient_errors,
    }
    result: dict[str, object] = {
        "experiment": "E50",
        "passed": passed,
        "failure_classification": None if passed else "stu_point_feature_interface_failure",
        "frame_namespace": PHASE5_FRAME_NAMESPACE,
        "frame_ids": {
            str(sequence_id): list(PHASE5_FRAMES[sequence_id])
            for sequence_id in (206, 201)
        },
        "frames": len(selected),
        "formal_repetitions": 2,
        "point_feature_width": MASK_DIM,
        "official_source_manifest_sha256": source_identity["manifest_sha256"],
        "official_source_file_count": source_identity["file_count"],
        "weight_identity": weight_identity,
        "protocol_sha256": _sha256(protocol_file),
        "device": str(runtime_device),
        "total_real_returns": int(real_count[0].sum()),
        "total_sparse_voxels": int(voxel_count[0].sum()),
        "finite_errors": int(finite_errors.sum()),
        "shape_errors": int(shape_errors.sum()),
        "gradient_errors": int(gradient_errors.sum()),
        "identity_errors": identity_errors,
        "reproduction_errors": reproduction_errors,
        "elapsed_seconds": elapsed,
        "scientific_array_sha256": _array_hash(arrays),
        "claim_limit": (
            "E50 qualifies only the frozen official STU 128-channel point-feature "
            "interface on the selected real frames."
        ),
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def run_e51(
    data_root: Path | str,
    protocol_path: Path | str,
    e50_path: Path | str,
    output_path: Path | str,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    """Verify exact sparse-voxel inverse recovery for every real return."""

    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    e50_file = Path(e50_path).expanduser().resolve(strict=True)
    if _sha256(e50_file) != E50_ARTIFACT_SHA256:
        raise QualificationError("E51 input is not the frozen E50 PASS artifact")
    with np.load(e50_file, allow_pickle=False) as e50:
        e50_metadata = json.loads(str(e50["metadata_json"]))
        if not e50_metadata.get("passed") or e50_metadata.get("experiment") != "E50":
            raise QualificationError("E51 input does not declare E50 PASS")

    selected: list[tuple[int, int]] = []
    for sequence_id in (206, 201):
        frame_ids = phase5_frame_ids(protocol, sequence_id)
        if frame_ids != PHASE5_FRAMES[sequence_id]:
            raise QualificationError("E51 frozen frame identity changed")
        selected.extend((sequence_id, frame_id) for frame_id in frame_ids)

    source_identity = stu_source_manifest(DEFAULT_STU_REPOSITORY)
    if source_identity["manifest_sha256"] != FROZEN_STU_SOURCE_MANIFEST_SHA256:
        raise QualificationError("E51 official STU source identity changed")
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise QualificationError("E51 requested CUDA but CUDA is unavailable")
    sequences = {
        sequence_id: STUSequence.open(
            data_root,
            protocol=protocol,
            partition="train",
            sequence_id=sequence_id,
            label_mode=LabelMode.FORBIDDEN,
        )
        for sequence_id in (206, 201)
    }
    encoder = FrozenSTUPointEncoder.from_protocol(protocol).to(runtime_device)
    encoder.eval()

    # This fixture locks negative bins and first-occurrence duplicate ordering.
    fixture = np.asarray(
        [[0.11, 0.0, 0.0], [0.01, 0.0, 0.0], [0.12, 0.0, 0.0],
         [-0.01, 0.0, 0.0], [-0.06, 0.0, 0.0]],
        dtype=np.float64,
    )
    fixture_rows, fixture_unique, fixture_inverse = independent_sparse_quantize(
        fixture, 0.05
    )
    if not (
        np.array_equal(fixture_rows[:, 0], np.asarray([2, 0, -1, -2]))
        and np.array_equal(fixture_unique, np.asarray([0, 1, 3, 4]))
        and np.array_equal(fixture_inverse, np.asarray([0, 1, 0, 2, 3]))
    ):
        raise QualificationError("E51 analytic quantization fixture failed")

    try:
        import MinkowskiEngine as me
    except ImportError as error:
        raise QualificationError("E51 requires the official MinkowskiEngine") from error

    count = len(selected)
    sequence_array = np.asarray([item[0] for item in selected], dtype=np.int16)
    frame_array = np.asarray([item[1] for item in selected], dtype=np.int16)
    real_count = np.zeros((2, count), dtype=np.int32)
    zero_slot_count = np.zeros_like(real_count)
    voxel_count = np.zeros_like(real_count)
    range_errors = np.zeros_like(real_count)
    recovery_errors = np.zeros_like(real_count)
    slot_errors = np.zeros_like(real_count)
    zero_exclusion_errors = np.zeros_like(real_count)
    quantized_coordinate_errors = np.zeros_like(real_count)
    unique_index_errors = np.zeros_like(real_count)
    inverse_errors = np.zeros_like(real_count)
    mapping_hash = np.empty((2, count), dtype="S64")
    seconds = np.zeros((2, count), dtype=np.float64)

    started = time.monotonic()
    for repetition in range(2):
        for index, (sequence_id, frame_id) in enumerate(selected):
            frame = sequences[sequence_id].source_frame(frame_id)
            coordinates = official_stu_coordinates(frame.xyzi, frame.lidar_pose)
            features = official_stu_features(frame.xyzi, frame.lidar_pose)
            point_coordinates = coordinates[frame.real_slots]
            expected_rows, expected_unique, expected_inverse = (
                independent_sparse_quantize(point_coordinates, encoder.voxel_size)
            )
            observed_rows, observed_unique, observed_inverse = me.utils.sparse_quantize(
                coordinates=point_coordinates,
                return_index=True,
                return_inverse=True,
                quantization_size=encoder.voxel_size,
            )
            observed_rows = np.asarray(observed_rows, dtype=np.int64)
            observed_unique = np.asarray(observed_unique, dtype=np.int64)
            observed_inverse = np.asarray(observed_inverse, dtype=np.int64)

            item_started = time.monotonic()
            encoding = encoder(coordinates, features, frame.real_slots)
            if runtime_device.type == "cuda":
                torch.cuda.synchronize(runtime_device)
            seconds[repetition, index] = time.monotonic() - item_started
            actual_inverse = encoding.inverse_map.detach().cpu().numpy()
            actual_slots = encoding.real_slots.detach().cpu().numpy()

            real_count[repetition, index] = frame.real_count
            zero_slot_count[repetition, index] = int(frame.zero_slot_mask.sum())
            voxel_count[repetition, index] = expected_rows.shape[0]
            range_errors[repetition, index] = int(
                actual_inverse.shape != (frame.real_count,)
                or actual_inverse.size == 0
                or int(actual_inverse.min()) < 0
                or int(actual_inverse.max()) >= expected_rows.shape[0]
            )
            if range_errors[repetition, index] == 0:
                recovery_errors[repetition, index] = int(
                    np.count_nonzero(expected_rows[actual_inverse] != np.floor(
                        point_coordinates / encoder.voxel_size
                    ).astype(np.int64))
                )
            slot_errors[repetition, index] = int(
                np.count_nonzero(actual_slots != frame.real_slots)
                if actual_slots.shape == frame.real_slots.shape
                else frame.slot_count
            )
            if actual_slots.size:
                zero_exclusion_errors[repetition, index] = int(
                    np.count_nonzero(frame.zero_slot_mask[actual_slots])
                )
            quantized_coordinate_errors[repetition, index] = int(
                np.count_nonzero(observed_rows != expected_rows)
                if observed_rows.shape == expected_rows.shape
                else max(observed_rows.size, expected_rows.size)
            )
            unique_index_errors[repetition, index] = int(
                np.count_nonzero(observed_unique != expected_unique)
                if observed_unique.shape == expected_unique.shape
                else max(observed_unique.size, expected_unique.size)
            )
            inverse_errors[repetition, index] = int(
                np.count_nonzero(observed_inverse != expected_inverse)
                + np.count_nonzero(actual_inverse != expected_inverse)
                if observed_inverse.shape == expected_inverse.shape
                and actual_inverse.shape == expected_inverse.shape
                else max(observed_inverse.size, actual_inverse.size, expected_inverse.size)
            )
            mapping_hash[repetition, index] = _array_hash(
                {
                    "quantized_coordinates": expected_rows,
                    "unique_indices": expected_unique,
                    "inverse_map": actual_inverse,
                    "real_slots": actual_slots,
                }
            )
            del encoding
    elapsed = time.monotonic() - started

    reproduction_errors = int(np.count_nonzero(mapping_hash[0] != mapping_hash[1]))
    count_reproduction_errors = int(
        not np.array_equal(real_count[0], real_count[1])
        or not np.array_equal(zero_slot_count[0], zero_slot_count[1])
        or not np.array_equal(voxel_count[0], voxel_count[1])
    )
    hard_error_arrays = (
        range_errors,
        recovery_errors,
        slot_errors,
        zero_exclusion_errors,
        quantized_coordinate_errors,
        unique_index_errors,
        inverse_errors,
    )
    hard_errors = int(
        sum(int(value.sum()) for value in hard_error_arrays)
        + reproduction_errors
        + count_reproduction_errors
    )
    arrays = {
        "sequence_id": sequence_array,
        "frame_id": frame_array,
        "real_count": real_count,
        "zero_slot_count": zero_slot_count,
        "voxel_count": voxel_count,
        "range_errors": range_errors,
        "recovery_errors": recovery_errors,
        "slot_errors": slot_errors,
        "zero_exclusion_errors": zero_exclusion_errors,
        "quantized_coordinate_errors": quantized_coordinate_errors,
        "unique_index_errors": unique_index_errors,
        "inverse_errors": inverse_errors,
        "mapping_sha256": mapping_hash,
        "frame_seconds": seconds,
    }
    result: dict[str, object] = {
        "experiment": "E51",
        "passed": hard_errors == 0,
        "failure_classification": None
        if hard_errors == 0
        else "sparse_voxel_inverse_mapping_failure",
        "frames": count,
        "formal_repetitions": 2,
        "e50_artifact_sha256": E50_ARTIFACT_SHA256,
        "protocol_sha256": _sha256(protocol_file),
        "official_source_manifest_sha256": source_identity["manifest_sha256"],
        "device": str(runtime_device),
        "total_file_slots": int((real_count[0] + zero_slot_count[0]).sum()),
        "total_real_returns": int(real_count[0].sum()),
        "total_zero_slots_excluded": int(zero_slot_count[0].sum()),
        "total_sparse_voxels": int(voxel_count[0].sum()),
        "range_errors": int(range_errors.sum()),
        "recovery_errors": int(recovery_errors.sum()),
        "slot_errors": int(slot_errors.sum()),
        "zero_exclusion_errors": int(zero_exclusion_errors.sum()),
        "quantized_coordinate_errors": int(quantized_coordinate_errors.sum()),
        "unique_index_errors": int(unique_index_errors.sum()),
        "inverse_errors": int(inverse_errors.sum()),
        "reproduction_errors": reproduction_errors + count_reproduction_errors,
        "elapsed_seconds": elapsed,
        "scientific_array_sha256": _array_hash(arrays),
        "claim_limit": (
            "E51 qualifies only exact sparse-row recovery and zero-slot exclusion "
            "for the frozen E50 real returns."
        ),
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    e50 = commands.add_parser("e50")
    e50.add_argument("--data-root", type=Path, required=True)
    e50.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e50.add_argument("--output", type=Path, required=True)
    e50.add_argument("--device", default="cuda")
    e51 = commands.add_parser("e51")
    e51.add_argument("--data-root", type=Path, required=True)
    e51.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e51.add_argument("--e50", type=Path, required=True)
    e51.add_argument("--output", type=Path, required=True)
    e51.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "e50":
        result = run_e50(
            args.data_root, args.protocol, args.output, device=args.device
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e51":
        result = run_e51(
            args.data_root,
            args.protocol,
            args.e50,
            args.output,
            device=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
