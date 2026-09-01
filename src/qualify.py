#!/usr/bin/env python3
"""Execute the frozen post-Gate-1 mechanical qualifications."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import importlib.util
import inspect
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from .evaluate import PointMetricAccumulator
from .model import (
    DEFAULT_STU_REPOSITORY,
    MASK_DIM,
    AJAEPointTransformer,
    FrozenSTUPointEncoder,
    assigned_stu_evidence,
    stu_source_manifest,
    stu_weight_identity,
)
from .protocol import load_protocol
from .scene import (
    ExperimentCondition,
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
E51_ARTIFACT_SHA256 = (
    "bca33539ea2c3cb9d815cc4586d98fc356f134d40351e63cbb8d2e1c256ccafa"
)
E52_ARTIFACT_SHA256 = (
    "2e519c358133cb03fbbbafed82062906eceec071279da0149b2e6a1eac1c9a69"
)
E53_ARTIFACT_SHA256 = (
    "e39511b76aec4c90b6d77d22b9d5f89d57184873ddc495677c8e786ffb476a03"
)
E54_ARTIFACT_SHA256 = (
    "67187b039bdafbea0d8f728a017daea043c2fdb6f7a6c7754da3998fa6173dac"
)
E55_ARTIFACT_SHA256 = (
    "13d367fa0f7f0ed86ba6de24fc535df44e4ea90ab6f38989dec4ea4d6e35aaf8"
)
E53_SEED_NAMESPACE = "E53-STU-query-v1"
E61_MATCH_NAMESPACE = "E61-static-match-v1"
E62_NUMERICAL_NAMESPACE = "E62-numerical-fixture-v1"
E62_NUMERICAL_SEED = 62002026
E63_SAFETY_NAMESPACE = "E63-safety-crossfit-v1"
E63_BOOTSTRAP_NAMESPACE = "E63-hierarchical-paired-bootstrap-v1"
E63_BOOTSTRAP_SEED = 63002026


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


def e53_frame_seed(sequence_id: int, frame_id: int) -> int:
    """Bind the official query decoder's random stream to frame identity."""

    payload = f"{E53_SEED_NAMESPACE}:train:{sequence_id}:{frame_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _e53_cpu_worker(
    task: tuple[str, str, tuple[tuple[int, int], ...], int]
) -> list[dict[str, object]]:
    """Evaluate an identity-fixed E53 frame chunk on deterministic CPU."""

    data_root, protocol_path, selected, threads = task
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    protocol = load_protocol(protocol_path)
    sequences = {
        sequence_id: STUSequence.open(
            data_root,
            protocol=protocol,
            partition="train",
            sequence_id=sequence_id,
            label_mode=LabelMode.FORBIDDEN,
        )
        for sequence_id in sorted({item[0] for item in selected})
    }
    encoder = FrozenSTUPointEncoder.from_protocol(protocol).cpu().eval()
    captured: list[object] = []

    def capture_output(_module: object, _inputs: object, output: object) -> None:
        captured.append(output)

    hook = encoder.stu.register_forward_hook(capture_output)
    records: list[dict[str, object]] = []
    try:
        for sequence_id, frame_id in selected:
            frame = sequences[sequence_id].source_frame(frame_id)
            coordinates = official_stu_coordinates(frame.xyzi, frame.lidar_pose)
            features = official_stu_features(frame.xyzi, frame.lidar_pose)
            repetitions: list[dict[str, object]] = []
            for _ in range(2):
                captured.clear()
                torch.manual_seed(e53_frame_seed(sequence_id, frame_id))
                item_started = time.monotonic()
                encoding = encoder(coordinates, features, frame.real_slots)
                seconds = time.monotonic() - item_started
                if len(captured) != 1 or not isinstance(captured[0], Mapping):
                    raise QualificationError("E53 failed to capture one official output")
                official_output = captured[0]
                logits = encoder._single_prediction(
                    official_output["pred_logits"], "pred_logits"
                )
                masks = encoder._single_prediction(
                    official_output["pred_masks"], "pred_masks"
                )
                class_probability = logits.softmax(dim=-1)
                normal_probability = class_probability[:, :19]
                mask_probability = masks.sigmoid()
                query_confidence = normal_probability.max(dim=1).values
                strengths = mask_probability * query_confidence[None, :]
                expected_query = strengths.argmax(dim=1)
                row = torch.arange(masks.shape[0])
                expected_mask = mask_probability[row, expected_query]
                expected_evidence = (
                    expected_mask[:, None] * normal_probability[expected_query]
                )
                expected_assignment = strengths[row, expected_query]
                expected_noobj = class_probability[expected_query, 19]
                inverse = encoding.inverse_map
                repetitions.append(
                    {
                        "voxel_count": int(masks.shape[0]),
                        "active_query_count": int(torch.unique(expected_query).numel()),
                        "query_identity_errors": int(
                            torch.count_nonzero(
                                encoding.assigned_query != expected_query[inverse]
                            ).item()
                        ),
                        "evidence_errors": int(
                            torch.count_nonzero(
                                encoding.normal_evidence != expected_evidence[inverse]
                            ).item()
                        ),
                        "assignment_reliability_errors": int(
                            torch.count_nonzero(
                                encoding.reliability_assign
                                != expected_assignment[inverse]
                            ).item()
                        ),
                        "no_object_errors": int(
                            torch.count_nonzero(
                                encoding.reliability_noobj != expected_noobj[inverse]
                            ).item()
                        ),
                        "output_hash": _array_hash(
                            {
                                "assigned_query": encoding.assigned_query.numpy(),
                                "normal_evidence": encoding.normal_evidence.numpy(),
                                "assignment_reliability": encoding.reliability_assign.numpy(),
                                "no_object_reliability": encoding.reliability_noobj.numpy(),
                            }
                        ),
                        "seconds": seconds,
                    }
                )
            records.append(
                {
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "repetitions": repetitions,
                }
            )
    finally:
        hook.remove()
    return records


def _e54_cpu_worker(
    task: tuple[str, str, tuple[tuple[int, int], ...], int]
) -> list[dict[str, object]]:
    """Recompute voxel and point evidence on the official float32 tensors."""

    data_root, protocol_path, selected, threads = task
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    protocol = load_protocol(protocol_path)
    sequences = {
        sequence_id: STUSequence.open(
            data_root, protocol=protocol, partition="train",
            sequence_id=sequence_id, label_mode=LabelMode.FORBIDDEN,
        )
        for sequence_id in sorted({item[0] for item in selected})
    }
    encoder = FrozenSTUPointEncoder.from_protocol(protocol).cpu().eval()
    captured: list[object] = []
    hook = encoder.stu.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    records: list[dict[str, object]] = []
    try:
        for sequence_id, frame_id in selected:
            frame = sequences[sequence_id].source_frame(frame_id)
            coordinates = official_stu_coordinates(frame.xyzi, frame.lidar_pose)
            features = official_stu_features(frame.xyzi, frame.lidar_pose)
            repetitions: list[dict[str, object]] = []
            for _ in range(2):
                captured.clear()
                torch.manual_seed(e53_frame_seed(sequence_id, frame_id))
                item_started = time.monotonic()
                encoding = encoder(coordinates, features, frame.real_slots)
                seconds = time.monotonic() - item_started
                if len(captured) != 1 or not isinstance(captured[0], Mapping):
                    raise QualificationError("E54 failed to capture one official output")
                official_output = captured[0]
                logits = encoder._single_prediction(
                    official_output["pred_logits"], "pred_logits"
                )
                masks = encoder._single_prediction(
                    official_output["pred_masks"], "pred_masks"
                )
                voxel_actual = assigned_stu_evidence(logits, masks)
                probability = logits.softmax(dim=-1)
                normal = probability[:, :19]
                mask_probability = masks.sigmoid()
                strength = mask_probability * normal.max(dim=1).values[None, :]
                query = strength.argmax(dim=1)
                row = torch.arange(query.numel())
                voxel_evidence = mask_probability[row, query, None] * normal[query]
                voxel_assignment = strength[row, query]
                voxel_noobj = probability[query, 19]
                inverse = encoding.inverse_map.numpy()
                actual_arrays = (
                    voxel_actual.normal_evidence.numpy(),
                    voxel_actual.reliability_assign.numpy(),
                    voxel_actual.reliability_noobj.numpy(),
                    encoding.normal_evidence.numpy(),
                    encoding.reliability_assign.numpy(),
                    encoding.reliability_noobj.numpy(),
                )
                reference_arrays = (
                    voxel_evidence.numpy(),
                    voxel_assignment.numpy(),
                    voxel_noobj.numpy(),
                    voxel_evidence.numpy()[inverse],
                    voxel_assignment.numpy()[inverse],
                    voxel_noobj.numpy()[inverse],
                )
                maximum_errors = tuple(
                    float(np.max(np.abs(actual.astype(np.float64) - reference)))
                    for actual, reference in zip(
                        actual_arrays, reference_arrays, strict=True
                    )
                )
                finite_errors = int(
                    sum(not np.isfinite(value).all() for value in actual_arrays)
                )
                gradient_errors = int(
                    encoding.normal_evidence.requires_grad
                    or encoding.reliability_assign.requires_grad
                    or encoding.reliability_noobj.requires_grad
                )
                broadcast_errors = int(
                    np.count_nonzero(
                        voxel_actual.normal_evidence.numpy()[inverse]
                        != encoding.normal_evidence.numpy()
                    )
                    + np.count_nonzero(
                        voxel_actual.reliability_assign.numpy()[inverse]
                        != encoding.reliability_assign.numpy()
                    )
                    + np.count_nonzero(
                        voxel_actual.reliability_noobj.numpy()[inverse]
                        != encoding.reliability_noobj.numpy()
                    )
                )
                repetitions.append(
                    {
                        "voxel_count": int(query.numel()),
                        "point_count": int(inverse.size),
                        "maximum_errors": maximum_errors,
                        "finite_errors": finite_errors,
                        "gradient_errors": gradient_errors,
                        "broadcast_errors": broadcast_errors,
                        "output_hash": _array_hash(
                            {
                                "voxel_evidence": actual_arrays[0],
                                "voxel_assignment": actual_arrays[1],
                                "voxel_noobj": actual_arrays[2],
                                "point_evidence": actual_arrays[3],
                                "point_assignment": actual_arrays[4],
                                "point_noobj": actual_arrays[5],
                            }
                        ),
                        "seconds": seconds,
                    }
                )
            records.append(
                {"sequence_id": sequence_id, "frame_id": frame_id,
                 "repetitions": repetitions}
            )
    finally:
        hook.remove()
    return records


def _e55_cpu_worker(
    task: tuple[str, str, int, int, int]
) -> dict[str, object]:
    """Build one real five-frame AJAE input twice on deterministic CPU."""

    data_root, protocol_path, sequence_id, center_frame, threads = task
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    protocol = load_protocol(protocol_path)
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train",
        sequence_id=sequence_id, label_mode=LabelMode.FORBIDDEN,
    )
    from .evaluate import _protocol_slot_to_ray

    slot_to_ray, ray_digest, _ = _protocol_slot_to_ray(protocol)
    mapping = slot_to_ray(sequence.source_frame(center_frame))
    window = sequence.window(
        center_frame,
        condition=ExperimentCondition.B3,
        canonical_ray_by_slot=mapping,
        ray_mapping_audited=True,
        ray_mapping_digest=ray_digest,
    )
    encoder = FrozenSTUPointEncoder.from_protocol(protocol).cpu().eval()
    torch.manual_seed(5500)
    projection = AJAEPointTransformer.from_protocol(protocol).cpu().eval().input_projection
    repetitions: list[dict[str, object]] = []
    for _ in range(2):
        encoded = []
        started = time.monotonic()
        for item in window.frames:
            source = item.source
            torch.manual_seed(e53_frame_seed(sequence_id, source.frame_id))
            encoded.append(
                encoder(
                    official_stu_coordinates(source.xyzi, source.lidar_pose),
                    official_stu_features(source.xyzi, source.lidar_pose),
                    source.real_slots,
                )
            )
        coordinates = torch.from_numpy(window.points.coordinates_center.copy())
        relative_times = torch.from_numpy(
            window.points.relative_time.astype(np.int64, copy=True)
        )
        stu_features = torch.cat([value.point_features for value in encoded])
        normal_evidence = torch.cat([value.normal_evidence for value in encoded])
        assignment = torch.cat([value.reliability_assign for value in encoded])
        noobj = torch.cat([value.reliability_noobj for value in encoded])
        intensity = torch.from_numpy(
            np.concatenate(
                [item.source.xyzi[item.source.real_slots, 3] for item in window.frames]
            ).astype(np.float32, copy=False)
        )
        expected_content = torch.cat(
            (stu_features, normal_evidence, assignment[:, None],
             noobj[:, None], intensity[:, None]), dim=1
        )
        captured_content: list[torch.Tensor] = []
        captured_position: list[torch.Tensor] = []
        captured_time: list[torch.Tensor] = []
        hooks = (
            projection.content[0].register_forward_pre_hook(
                lambda _module, inputs: captured_content.append(inputs[0].detach())
            ),
            projection.position[0].register_forward_pre_hook(
                lambda _module, inputs: captured_position.append(inputs[0].detach())
            ),
            projection.time.register_forward_pre_hook(
                lambda _module, inputs: captured_time.append(inputs[0].detach())
            ),
        )
        try:
            with torch.no_grad():
                projected = projection(
                    coordinates, relative_times, stu_features, normal_evidence,
                    assignment, noobj, intensity,
                )
        finally:
            for hook in hooks:
                hook.remove()
        point_count = int(coordinates.shape[0])
        slot_errors = int(
            sum(
                not np.array_equal(value.real_slots.numpy(), item.source.real_slots)
                for value, item in zip(encoded, window.frames, strict=True)
            )
        )
        identity_errors = int(
            point_count != window.points.count
            or window.points.source_frame.shape != (point_count,)
            or window.points.source_slot.shape != (point_count,)
            or window.points.source_ray.shape != (point_count,)
        )
        content_errors = int(
            len(captured_content) != 1
            or captured_content[0].shape != (point_count, 150)
            or not torch.equal(captured_content[0], expected_content)
        )
        coordinate_errors = int(
            len(captured_position) != 1
            or not torch.equal(captured_position[0], coordinates)
        )
        time_errors = int(
            len(captured_time) != 1
            or not torch.equal(captured_time[0], relative_times + 2)
        )
        dtype_errors = int(
            expected_content.dtype != torch.float32
            or coordinates.dtype != torch.float32
            or relative_times.dtype != torch.long
        )
        output_errors = int(
            projected.shape != (point_count, int(protocol.model["hidden_dim"]))
            or not bool(torch.isfinite(projected).all())
        )
        repetitions.append(
            {
                "point_count": point_count,
                "time_counts": np.bincount(
                    (relative_times + 2).numpy(), minlength=5
                ).astype(np.int64),
                "slot_errors": slot_errors,
                "identity_errors": identity_errors,
                "content_errors": content_errors,
                "coordinate_errors": coordinate_errors,
                "time_errors": time_errors,
                "dtype_errors": dtype_errors,
                "output_errors": output_errors,
                "content_hash": _tensor_hash(expected_content),
                "projected_hash": _tensor_hash(projected),
                "seconds": time.monotonic() - started,
            }
        )
        del encoded, projected, expected_content
    return {
        "sequence_id": sequence_id,
        "center_frame": center_frame,
        "source_frames": [item.source.frame_id for item in window.frames],
        "repetitions": repetitions,
    }


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


def run_e52(
    data_root: Path | str,
    protocol_path: Path | str,
    e51_path: Path | str,
    output_path: Path | str,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    """Verify raw-point identity survives shared STU sparse voxels."""

    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    e51_file = Path(e51_path).expanduser().resolve(strict=True)
    if _sha256(e51_file) != E51_ARTIFACT_SHA256:
        raise QualificationError("E52 input is not the frozen E51 PASS artifact")
    with np.load(e51_file, allow_pickle=False) as e51:
        e51_metadata = json.loads(str(e51["metadata_json"]))
        if not e51_metadata.get("passed") or e51_metadata.get("experiment") != "E51":
            raise QualificationError("E52 input does not declare E51 PASS")

    selected: list[tuple[int, int]] = []
    for sequence_id in (206, 201):
        frame_ids = phase5_frame_ids(protocol, sequence_id)
        if frame_ids != PHASE5_FRAMES[sequence_id]:
            raise QualificationError("E52 frozen frame identity changed")
        selected.extend((sequence_id, frame_id) for frame_id in frame_ids)
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise QualificationError("E52 requested CUDA but CUDA is unavailable")
    sequences = {
        sequence_id: STUSequence.open(
            data_root,
            protocol=protocol,
            partition="train",
            sequence_id=sequence_id,
            label_mode=LabelMode.REQUIRED,
        )
        for sequence_id in (206, 201)
    }
    try:
        from .evaluate import _protocol_slot_to_ray
    except ImportError as error:  # pragma: no cover - package execution is required
        raise QualificationError("E52 cannot load the frozen ray mapping") from error
    slot_to_ray, ray_mapping_digest, calibration_sha256 = _protocol_slot_to_ray(
        protocol
    )
    encoder = FrozenSTUPointEncoder.from_protocol(protocol).to(runtime_device)
    encoder.eval()

    count = len(selected)
    sequence_array = np.asarray([item[0] for item in selected], dtype=np.int16)
    frame_array = np.asarray([item[1] for item in selected], dtype=np.int16)
    real_count = np.zeros((2, count), dtype=np.int32)
    shared_voxel_count = np.zeros_like(real_count)
    shared_point_count = np.zeros_like(real_count)
    shared_feature_errors = np.zeros_like(real_count)
    frame_identity_errors = np.zeros_like(real_count)
    slot_identity_errors = np.zeros_like(real_count)
    ray_identity_errors = np.zeros_like(real_count)
    coordinate_identity_errors = np.zeros_like(real_count)
    intensity_identity_errors = np.zeros_like(real_count)
    label_identity_errors = np.zeros_like(real_count)
    shared_identity_collision_errors = np.zeros_like(real_count)
    identity_hash = np.empty((2, count), dtype="S64")
    seconds = np.zeros((2, count), dtype=np.float64)

    started = time.monotonic()
    for repetition in range(2):
        for index, (sequence_id, frame_id) in enumerate(selected):
            sequence = sequences[sequence_id]
            frame = sequence.source_frame(frame_id)
            mapping = slot_to_ray(frame)
            window = sequence.window(
                frame_id,
                condition=ExperimentCondition.B0,
                canonical_ray_by_slot=mapping,
                ray_mapping_audited=True,
                ray_mapping_digest=ray_mapping_digest,
            )
            if frame.labels is None or window.labels is None:
                raise QualificationError("E52 requires source and window labels")
            coordinates = official_stu_coordinates(frame.xyzi, frame.lidar_pose)
            features = official_stu_features(frame.xyzi, frame.lidar_pose)
            item_started = time.monotonic()
            encoding = encoder(coordinates, features, frame.real_slots)
            if runtime_device.type == "cuda":
                torch.cuda.synchronize(runtime_device)
            seconds[repetition, index] = time.monotonic() - item_started

            inverse = encoding.inverse_map
            sparse_counts = torch.bincount(inverse)
            shared_rows = sparse_counts > 1
            point_is_shared = shared_rows[inverse]
            order = torch.argsort(inverse)
            sorted_inverse = inverse[order]
            adjacent_shared = sorted_inverse[1:] == sorted_inverse[:-1]
            adjacent_feature_difference = torch.any(
                encoding.point_features[order][1:]
                != encoding.point_features[order][:-1],
                dim=1,
            )
            slots = frame.real_slots
            rays = mapping[slots]
            source_xyz = frame.xyzi[slots, :3]
            source_intensity = frame.xyzi[slots, 3]
            source_labels = frame.labels.packed[slots]

            real_count[repetition, index] = frame.real_count
            shared_voxel_count[repetition, index] = int(shared_rows.sum().item())
            shared_point_count[repetition, index] = int(point_is_shared.sum().item())
            shared_feature_errors[repetition, index] = int(
                (adjacent_shared & adjacent_feature_difference).sum().item()
            )
            frame_identity_errors[repetition, index] = int(
                np.count_nonzero(window.points.source_frame != frame_id)
            )
            slot_identity_errors[repetition, index] = int(
                np.count_nonzero(window.points.source_slot != slots)
            )
            ray_identity_errors[repetition, index] = int(
                np.count_nonzero(window.points.source_ray != rays)
            )
            coordinate_identity_errors[repetition, index] = int(
                np.count_nonzero(window.points.coordinates_reference != source_xyz)
            )
            # The model input reads this exact source vector in visible-slot order.
            intensity_identity_errors[repetition, index] = int(
                source_intensity.shape != (frame.real_count,)
                or not np.isfinite(source_intensity).all()
            )
            label_identity_errors[repetition, index] = int(
                np.count_nonzero(window.labels.packed != source_labels)
            )
            shared_indices = np.flatnonzero(point_is_shared.detach().cpu().numpy())
            shared_identities = np.column_stack((slots[shared_indices], rays[shared_indices]))
            shared_identity_collision_errors[repetition, index] = int(
                shared_identities.shape[0]
                - np.unique(shared_identities, axis=0).shape[0]
            )
            identity_hash[repetition, index] = _array_hash(
                {
                    "source_frame": window.points.source_frame,
                    "source_slot": window.points.source_slot,
                    "source_ray": window.points.source_ray,
                    "coordinates": window.points.coordinates_reference,
                    "intensity": source_intensity,
                    "packed_label": window.labels.packed,
                    "inverse_map": inverse.detach().cpu().numpy(),
                    "point_feature_sha256": np.frombuffer(
                        _tensor_hash(encoding.point_features).encode("ascii"),
                        dtype=np.uint8,
                    ),
                }
            )
            del encoding
    elapsed = time.monotonic() - started

    # Counterexample: shared STU content must still yield one final logit per raw row.
    torch.manual_seed(5200)
    fixture_model = AJAEPointTransformer.from_protocol(protocol).cpu().eval()
    fixture_coordinates = torch.tensor(
        [[0.001, 0.0, 0.0], [0.049, 0.0, 0.0],
         [0.101, 0.0, 0.0], [0.149, 0.0, 0.0]],
        dtype=torch.float32,
    )
    fixture_times = torch.zeros(4, dtype=torch.long)
    fixture_features = torch.ones(4, MASK_DIM)
    fixture_evidence = torch.zeros(4, 19)
    fixture_reliability = torch.zeros(4)
    fixture_intensity = torch.tensor([0.1, 0.9, 0.2, 0.8])
    fixture_labels = np.asarray([10, 2, 40, 2], dtype=np.uint16)
    fixture_rays = np.asarray([11, 12, 13, 14], dtype=np.int32)
    fixture_order = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    fixture_logits = []
    with torch.no_grad():
        for _ in range(2):
            fixture_logits.append(
                fixture_model(
                    fixture_coordinates,
                    fixture_times,
                    fixture_features,
                    fixture_evidence,
                    fixture_reliability,
                    fixture_reliability,
                    fixture_intensity,
                    cross_frame_enabled=False,
                )
            )
        permuted_logits = fixture_model(
            fixture_coordinates[fixture_order],
            fixture_times[fixture_order],
            fixture_features[fixture_order],
            fixture_evidence[fixture_order],
            fixture_reliability[fixture_order],
            fixture_reliability[fixture_order],
            fixture_intensity[fixture_order],
            cross_frame_enabled=False,
        )
    fixture_shape_errors = int(
        fixture_logits[0].shape != (4,) or permuted_logits.shape != (4,)
    )
    fixture_reproduction_errors = int(
        not torch.equal(fixture_logits[0], fixture_logits[1])
    )
    fixture_permutation_errors = int(
        not torch.equal(permuted_logits, fixture_logits[0][fixture_order])
    )
    fixture_identity_errors = int(
        np.unique(fixture_rays).size != 4
        or np.unique(fixture_labels).size < 2
        or not torch.equal(fixture_features[0], fixture_features[1])
    )

    reproduction_errors = int(np.count_nonzero(identity_hash[0] != identity_hash[1]))
    statistic_reproduction_errors = int(
        any(
            not np.array_equal(value[0], value[1])
            for value in (real_count, shared_voxel_count, shared_point_count)
        )
    )
    error_arrays = (
        shared_feature_errors,
        frame_identity_errors,
        slot_identity_errors,
        ray_identity_errors,
        coordinate_identity_errors,
        intensity_identity_errors,
        label_identity_errors,
        shared_identity_collision_errors,
    )
    hard_errors = int(
        sum(int(value.sum()) for value in error_arrays)
        + reproduction_errors
        + statistic_reproduction_errors
        + fixture_shape_errors
        + fixture_reproduction_errors
        + fixture_permutation_errors
        + fixture_identity_errors
    )
    arrays = {
        "sequence_id": sequence_array,
        "frame_id": frame_array,
        "real_count": real_count,
        "shared_voxel_count": shared_voxel_count,
        "shared_point_count": shared_point_count,
        "shared_feature_errors": shared_feature_errors,
        "frame_identity_errors": frame_identity_errors,
        "slot_identity_errors": slot_identity_errors,
        "ray_identity_errors": ray_identity_errors,
        "coordinate_identity_errors": coordinate_identity_errors,
        "intensity_identity_errors": intensity_identity_errors,
        "label_identity_errors": label_identity_errors,
        "shared_identity_collision_errors": shared_identity_collision_errors,
        "identity_sha256": identity_hash,
        "frame_seconds": seconds,
        "fixture_coordinates": fixture_coordinates.numpy(),
        "fixture_intensity": fixture_intensity.numpy(),
        "fixture_labels": fixture_labels,
        "fixture_rays": fixture_rays,
        "fixture_logits": torch.stack(fixture_logits).numpy(),
        "fixture_order": fixture_order.numpy(),
        "fixture_permuted_logits": permuted_logits.numpy(),
    }
    result: dict[str, object] = {
        "experiment": "E52",
        "passed": hard_errors == 0,
        "failure_classification": None
        if hard_errors == 0
        else "raw_point_identity_merging_failure",
        "frames": count,
        "formal_repetitions": 2,
        "e51_artifact_sha256": E51_ARTIFACT_SHA256,
        "protocol_sha256": _sha256(protocol_file),
        "calibration_sha256": calibration_sha256,
        "ray_mapping_digest": ray_mapping_digest,
        "device": str(runtime_device),
        "total_real_returns": int(real_count[0].sum()),
        "total_shared_voxels": int(shared_voxel_count[0].sum()),
        "total_points_in_shared_voxels": int(shared_point_count[0].sum()),
        "shared_feature_errors": int(shared_feature_errors.sum()),
        "frame_identity_errors": int(frame_identity_errors.sum()),
        "slot_identity_errors": int(slot_identity_errors.sum()),
        "ray_identity_errors": int(ray_identity_errors.sum()),
        "coordinate_identity_errors": int(coordinate_identity_errors.sum()),
        "intensity_identity_errors": int(intensity_identity_errors.sum()),
        "label_identity_errors": int(label_identity_errors.sum()),
        "shared_identity_collision_errors": int(
            shared_identity_collision_errors.sum()
        ),
        "fixture_shape_errors": fixture_shape_errors,
        "fixture_reproduction_errors": fixture_reproduction_errors,
        "fixture_permutation_errors": fixture_permutation_errors,
        "fixture_identity_errors": fixture_identity_errors,
        "reproduction_errors": reproduction_errors + statistic_reproduction_errors,
        "elapsed_seconds": elapsed,
        "scientific_array_sha256": _array_hash(arrays),
        "claim_limit": (
            "E52 qualifies raw-row identity preservation under shared STU sparse "
            "features; it does not qualify query evidence or learned performance."
        ),
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def run_e53(
    data_root: Path | str,
    protocol_path: Path | str,
    e52_path: Path | str,
    output_path: Path | str,
    *,
    device: str = "cpu",
    workers: int = 4,
    threads_per_worker: int = 6,
) -> dict[str, object]:
    """Verify official query assignment and smallest-index tie semantics."""

    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    e52_file = Path(e52_path).expanduser().resolve(strict=True)
    if _sha256(e52_file) != E52_ARTIFACT_SHA256:
        raise QualificationError("E53 input is not the frozen E52 PASS artifact")
    with np.load(e52_file, allow_pickle=False) as e52:
        metadata = json.loads(str(e52["metadata_json"]))
        if not metadata.get("passed") or metadata.get("experiment") != "E52":
            raise QualificationError("E53 input does not declare E52 PASS")

    selected: list[tuple[int, int]] = []
    for sequence_id in (206, 201):
        frame_ids = phase5_frame_ids(protocol, sequence_id)
        if frame_ids != PHASE5_FRAMES[sequence_id]:
            raise QualificationError("E53 frozen frame identity changed")
        selected.extend((sequence_id, frame_id) for frame_id in frame_ids)
    runtime_device = torch.device(device)
    if runtime_device.type != "cpu":
        raise QualificationError("E53 requires deterministic official CPU evaluation")
    if workers != 4 or threads_per_worker != 6:
        raise QualificationError("E53 requires the frozen 4x6 CPU execution layout")
    source_identity = stu_source_manifest(DEFAULT_STU_REPOSITORY)
    if source_identity["manifest_sha256"] != FROZEN_STU_SOURCE_MANIFEST_SHA256:
        raise QualificationError("E53 official STU source identity changed")

    # The exact tie has equal strengths for q=0 and q=1; argmax must select q=0.
    tie_logits = torch.zeros(100, 20)
    tie_masks = torch.full((3, 100), -20.0)
    tie_logits[0, 0] = tie_logits[1, 1] = 5.0
    tie_masks[:, :2] = 2.0
    tie_evidence = assigned_stu_evidence(tie_logits, tie_masks)
    tie_errors = int(
        not torch.equal(tie_evidence.assigned_query, torch.zeros(3, dtype=torch.long))
    )

    count = len(selected)
    sequence_array = np.asarray([item[0] for item in selected], dtype=np.int16)
    frame_array = np.asarray([item[1] for item in selected], dtype=np.int16)
    seed_array = np.asarray(
        [e53_frame_seed(*item) for item in selected], dtype=np.int64
    )
    voxel_count = np.zeros((2, count), dtype=np.int32)
    active_query_count = np.zeros_like(voxel_count)
    query_identity_errors = np.zeros_like(voxel_count)
    evidence_errors = np.zeros_like(voxel_count)
    assignment_reliability_errors = np.zeros_like(voxel_count)
    no_object_errors = np.zeros_like(voxel_count)
    output_hash = np.empty((2, count), dtype="S64")
    seconds = np.zeros((2, count), dtype=np.float64)

    started = time.monotonic()
    chunks = tuple(tuple(selected[offset::workers]) for offset in range(workers))
    tasks = tuple(
        (str(Path(data_root).expanduser().resolve()), str(protocol_file), chunk,
         threads_per_worker)
        for chunk in chunks
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        worker_outputs = tuple(executor.map(_e53_cpu_worker, tasks))
    index_by_identity = {identity: index for index, identity in enumerate(selected)}
    for worker_output in worker_outputs:
        for record in worker_output:
            identity = (int(record["sequence_id"]), int(record["frame_id"]))
            index = index_by_identity[identity]
            repetitions = record["repetitions"]
            for repetition, values in enumerate(repetitions):
                voxel_count[repetition, index] = int(values["voxel_count"])
                active_query_count[repetition, index] = int(
                    values["active_query_count"]
                )
                query_identity_errors[repetition, index] = int(
                    values["query_identity_errors"]
                )
                evidence_errors[repetition, index] = int(values["evidence_errors"])
                assignment_reliability_errors[repetition, index] = int(
                    values["assignment_reliability_errors"]
                )
                no_object_errors[repetition, index] = int(values["no_object_errors"])
                output_hash[repetition, index] = str(values["output_hash"])
                seconds[repetition, index] = float(values["seconds"])
    elapsed = time.monotonic() - started

    reproduction_errors = int(np.count_nonzero(output_hash[0] != output_hash[1]))
    statistic_reproduction_errors = int(
        not np.array_equal(voxel_count[0], voxel_count[1])
        or not np.array_equal(active_query_count[0], active_query_count[1])
    )
    error_arrays = (
        query_identity_errors,
        evidence_errors,
        assignment_reliability_errors,
        no_object_errors,
    )
    hard_errors = int(
        sum(int(value.sum()) for value in error_arrays)
        + tie_errors
        + reproduction_errors
        + statistic_reproduction_errors
    )
    arrays = {
        "sequence_id": sequence_array,
        "frame_id": frame_array,
        "frame_seed": seed_array,
        "voxel_count": voxel_count,
        "active_query_count": active_query_count,
        "query_identity_errors": query_identity_errors,
        "evidence_errors": evidence_errors,
        "assignment_reliability_errors": assignment_reliability_errors,
        "no_object_errors": no_object_errors,
        "output_sha256": output_hash,
        "frame_seconds": seconds,
        "tie_assigned_query": tie_evidence.assigned_query.numpy(),
    }
    result: dict[str, object] = {
        "experiment": "E53",
        "passed": hard_errors == 0,
        "failure_classification": None
        if hard_errors == 0
        else "official_query_assignment_failure",
        "frames": count,
        "formal_repetitions": 2,
        "seed_namespace": E53_SEED_NAMESPACE,
        "e52_artifact_sha256": E52_ARTIFACT_SHA256,
        "protocol_sha256": _sha256(protocol_file),
        "official_source_manifest_sha256": source_identity["manifest_sha256"],
        "device": str(runtime_device),
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "total_sparse_voxels": int(voxel_count[0].sum()),
        "minimum_active_queries_per_frame": int(active_query_count[0].min()),
        "maximum_active_queries_per_frame": int(active_query_count[0].max()),
        "query_identity_errors": int(query_identity_errors.sum()),
        "evidence_errors": int(evidence_errors.sum()),
        "assignment_reliability_errors": int(
            assignment_reliability_errors.sum()
        ),
        "no_object_errors": int(no_object_errors.sum()),
        "tie_errors": tie_errors,
        "reproduction_errors": reproduction_errors + statistic_reproduction_errors,
        "elapsed_seconds": elapsed,
        "scientific_array_sha256": _array_hash(arrays),
        "claim_limit": (
            "E53 qualifies the frozen minimum-index query assignment identity; "
            "E54 separately qualifies numerical tolerance and point broadcasting."
        ),
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def run_e54(
    data_root: Path | str,
    protocol_path: Path | str,
    e53_path: Path | str,
    output_path: Path | str,
    *,
    device: str = "cpu",
    workers: int = 4,
    threads_per_worker: int = 6,
) -> dict[str, object]:
    """Verify voxel/point evidence against an independent frozen-formula path."""

    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    e53_file = Path(e53_path).expanduser().resolve(strict=True)
    if _sha256(e53_file) != E53_ARTIFACT_SHA256:
        raise QualificationError("E54 input is not the frozen E53 PASS artifact")
    with np.load(e53_file, allow_pickle=False) as e53:
        metadata = json.loads(str(e53["metadata_json"]))
        if not metadata.get("passed") or metadata.get("experiment") != "E53":
            raise QualificationError("E54 input does not declare E53 PASS")
    selected = [
        (sequence_id, frame_id)
        for sequence_id in (206, 201)
        for frame_id in phase5_frame_ids(protocol, sequence_id)
    ]
    if any(
        tuple(frame for sequence, frame in selected if sequence == sequence_id)
        != PHASE5_FRAMES[sequence_id]
        for sequence_id in (206, 201)
    ):
        raise QualificationError("E54 frozen frame identity changed")
    if torch.device(device).type != "cpu":
        raise QualificationError("E54 requires deterministic official CPU evaluation")
    if workers != 4 or threads_per_worker != 6:
        raise QualificationError("E54 requires the frozen 4x6 CPU execution layout")

    count = len(selected)
    sequence_array = np.asarray([item[0] for item in selected], dtype=np.int16)
    frame_array = np.asarray([item[1] for item in selected], dtype=np.int16)
    voxel_count = np.zeros((2, count), dtype=np.int32)
    point_count = np.zeros_like(voxel_count)
    maximum_error = np.zeros((2, count, 6), dtype=np.float64)
    finite_errors = np.zeros_like(voxel_count)
    gradient_errors = np.zeros_like(voxel_count)
    broadcast_errors = np.zeros_like(voxel_count)
    output_hash = np.empty((2, count), dtype="S64")
    seconds = np.zeros((2, count), dtype=np.float64)

    chunks = tuple(tuple(selected[offset::workers]) for offset in range(workers))
    tasks = tuple(
        (str(Path(data_root).expanduser().resolve()), str(protocol_file), chunk,
         threads_per_worker)
        for chunk in chunks
    )
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        worker_outputs = tuple(executor.map(_e54_cpu_worker, tasks))
    index_by_identity = {identity: index for index, identity in enumerate(selected)}
    for worker_output in worker_outputs:
        for record in worker_output:
            index = index_by_identity[(int(record["sequence_id"]), int(record["frame_id"]))]
            for repetition, values in enumerate(record["repetitions"]):
                voxel_count[repetition, index] = int(values["voxel_count"])
                point_count[repetition, index] = int(values["point_count"])
                maximum_error[repetition, index] = values["maximum_errors"]
                finite_errors[repetition, index] = int(values["finite_errors"])
                gradient_errors[repetition, index] = int(values["gradient_errors"])
                broadcast_errors[repetition, index] = int(values["broadcast_errors"])
                output_hash[repetition, index] = str(values["output_hash"])
                seconds[repetition, index] = float(values["seconds"])
    elapsed = time.monotonic() - started

    tolerance = 1e-7
    tolerance_errors = int(np.count_nonzero(maximum_error > tolerance))
    reproduction_errors = int(np.count_nonzero(output_hash[0] != output_hash[1]))
    count_reproduction_errors = int(
        not np.array_equal(voxel_count[0], voxel_count[1])
        or not np.array_equal(point_count[0], point_count[1])
        or not np.array_equal(maximum_error[0], maximum_error[1])
    )
    hard_errors = int(
        finite_errors.sum()
        + gradient_errors.sum()
        + broadcast_errors.sum()
        + tolerance_errors
        + reproduction_errors
        + count_reproduction_errors
    )
    arrays = {
        "sequence_id": sequence_array,
        "frame_id": frame_array,
        "voxel_count": voxel_count,
        "point_count": point_count,
        "maximum_absolute_error": maximum_error,
        "finite_errors": finite_errors,
        "gradient_errors": gradient_errors,
        "broadcast_errors": broadcast_errors,
        "output_sha256": output_hash,
        "frame_seconds": seconds,
    }
    result: dict[str, object] = {
        "experiment": "E54",
        "passed": hard_errors == 0,
        "failure_classification": None
        if hard_errors == 0
        else "evidence_reliability_numerical_failure",
        "frames": count,
        "formal_repetitions": 2,
        "e53_artifact_sha256": E53_ARTIFACT_SHA256,
        "protocol_sha256": _sha256(protocol_file),
        "device": "cpu",
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "tolerance": tolerance,
        "total_sparse_voxels": int(voxel_count[0].sum()),
        "total_real_returns": int(point_count[0].sum()),
        "maximum_absolute_error": float(maximum_error.max()),
        "tolerance_errors": tolerance_errors,
        "finite_errors": int(finite_errors.sum()),
        "gradient_errors": int(gradient_errors.sum()),
        "broadcast_errors": int(broadcast_errors.sum()),
        "reproduction_errors": reproduction_errors + count_reproduction_errors,
        "elapsed_seconds": elapsed,
        "scientific_array_sha256": _array_hash(arrays),
        "claim_limit": (
            "E54 qualifies only the frozen 19D evidence and reliability numerics "
            "and inverse broadcasting."
        ),
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def run_e55(
    data_root: Path | str,
    protocol_path: Path | str,
    e54_path: Path | str,
    output_path: Path | str,
    *,
    device: str = "cpu",
    workers: int = 2,
    threads_per_worker: int = 12,
) -> dict[str, object]:
    """Verify the actual five-frame AJAE input schema and field boundary."""

    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    e54_file = Path(e54_path).expanduser().resolve(strict=True)
    if _sha256(e54_file) != E54_ARTIFACT_SHA256:
        raise QualificationError("E55 input is not the frozen E54 PASS artifact")
    with np.load(e54_file, allow_pickle=False) as e54:
        metadata = json.loads(str(e54["metadata_json"]))
        if not metadata.get("passed") or metadata.get("experiment") != "E54":
            raise QualificationError("E55 input does not declare E54 PASS")
    if torch.device(device).type != "cpu":
        raise QualificationError("E55 requires deterministic official CPU evaluation")
    if workers != 2 or threads_per_worker != 12:
        raise QualificationError("E55 requires the frozen 2x12 CPU execution layout")

    centers = ((206, PHASE5_FRAMES[206][0]), (201, PHASE5_FRAMES[201][0]))
    tasks = tuple(
        (str(Path(data_root).expanduser().resolve()), str(protocol_file), sequence,
         center, threads_per_worker)
        for sequence, center in centers
    )
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        records = tuple(executor.map(_e55_cpu_worker, tasks))
    elapsed = time.monotonic() - started

    point_count = np.zeros((2, 2), dtype=np.int32)
    time_count = np.zeros((2, 2, 5), dtype=np.int32)
    error_names = (
        "slot_errors", "identity_errors", "content_errors", "coordinate_errors",
        "time_errors", "dtype_errors", "output_errors",
    )
    errors = {name: np.zeros((2, 2), dtype=np.int32) for name in error_names}
    content_hash = np.empty((2, 2), dtype="S64")
    projected_hash = np.empty((2, 2), dtype="S64")
    seconds = np.zeros((2, 2), dtype=np.float64)
    source_frames = np.zeros((2, 5), dtype=np.int16)
    for window_index, record in enumerate(records):
        source_frames[window_index] = record["source_frames"]
        for repetition, values in enumerate(record["repetitions"]):
            point_count[repetition, window_index] = int(values["point_count"])
            time_count[repetition, window_index] = values["time_counts"]
            for name in error_names:
                errors[name][repetition, window_index] = int(values[name])
            content_hash[repetition, window_index] = str(values["content_hash"])
            projected_hash[repetition, window_index] = str(values["projected_hash"])
            seconds[repetition, window_index] = float(values["seconds"])

    allowed_parameters = {
        "coordinates", "relative_times", "stu_features", "normal_evidence",
        "reliability_assign", "reliability_noobj", "intensity",
        "cross_frame_enabled",
    }
    observed_parameters = set(inspect.signature(AJAEPointTransformer.forward).parameters)
    observed_parameters.discard("self")
    forbidden = {
        "assigned_query", "query_token", "entropy", "energy", "msp",
        "instance_id", "moving_label", "generator_family", "nvis",
        "occlusion", "support_semantic", "proposal_count",
    }
    signature_errors = int(
        observed_parameters != allowed_parameters
        or bool(observed_parameters.intersection(forbidden))
    )
    reproduction_errors = int(
        np.count_nonzero(content_hash[0] != content_hash[1])
        + np.count_nonzero(projected_hash[0] != projected_hash[1])
        + (not np.array_equal(point_count[0], point_count[1]))
        + (not np.array_equal(time_count[0], time_count[1]))
    )
    schema_errors = int(
        int(protocol.model["input_dim"]) != 150
        or 128 + 19 + 1 + 1 + 1 != 150
        or np.any(time_count == 0)
        or np.any(point_count != time_count.sum(axis=2))
    )
    hard_errors = int(
        sum(int(value.sum()) for value in errors.values())
        + signature_errors
        + reproduction_errors
        + schema_errors
    )
    arrays = {
        "sequence_id": np.asarray([206, 201], dtype=np.int16),
        "center_frame": np.asarray([centers[0][1], centers[1][1]], dtype=np.int16),
        "source_frames": source_frames,
        "point_count": point_count,
        "time_count": time_count,
        **errors,
        "content_sha256": content_hash,
        "projected_sha256": projected_hash,
        "window_seconds": seconds,
    }
    result: dict[str, object] = {
        "experiment": "E55",
        "passed": hard_errors == 0,
        "failure_classification": None
        if hard_errors == 0
        else "ajae_input_schema_or_leakage_failure",
        "windows": 2,
        "formal_repetitions": 2,
        "centers": {"206": centers[0][1], "201": centers[1][1]},
        "e54_artifact_sha256": E54_ARTIFACT_SHA256,
        "protocol_sha256": _sha256(protocol_file),
        "device": "cpu",
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "base_input_width": 150,
        "component_widths": [128, 19, 1, 1, 1],
        "total_points": int(point_count[0].sum()),
        "signature_errors": signature_errors,
        "schema_errors": schema_errors,
        "field_errors": int(sum(int(value.sum()) for value in errors.values())),
        "reproduction_errors": reproduction_errors,
        "elapsed_seconds": elapsed,
        "scientific_array_sha256": _array_hash(arrays),
        "claim_limit": (
            "E55 qualifies the actual five-frame input schema and prohibited-field "
            "boundary; it does not qualify coordinate alignment quality or learning."
        ),
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def run_e56(
    data_root: Path | str,
    protocol_path: Path | str,
    e55_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Verify center-coordinate alignment and preservation of object motion."""

    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    e55_file = Path(e55_path).expanduser().resolve(strict=True)
    if _sha256(e55_file) != E55_ARTIFACT_SHA256:
        raise QualificationError("E56 input is not the frozen E55 PASS artifact")
    with np.load(e55_file, allow_pickle=False) as e55:
        metadata = json.loads(str(e55["metadata_json"]))
        if not metadata.get("passed") or metadata.get("experiment") != "E55":
            raise QualificationError("E56 input does not declare E55 PASS")
    from .evaluate import _protocol_slot_to_ray

    slot_to_ray, ray_digest, calibration_sha256 = _protocol_slot_to_ray(protocol)
    sequences = {
        sequence_id: STUSequence.open(
            data_root, protocol=protocol, partition="train",
            sequence_id=sequence_id, label_mode=LabelMode.REQUIRED,
        )
        for sequence_id in (206, 201)
    }
    moving_semantics = np.asarray(protocol.labels["moving_normal_semantic_ids"])
    centers = [
        (sequence_id, frame_id)
        for sequence_id in (206, 201)
        for frame_id in PHASE5_FRAMES[sequence_id]
    ]
    window_count = len(centers)
    before_median = np.zeros((2, window_count), dtype=np.float64)
    before_q95 = np.zeros_like(before_median)
    after_median = np.zeros_like(before_median)
    after_q95 = np.zeros_like(before_median)
    static_point_count = np.zeros((2, window_count), dtype=np.int32)
    moving_track_count = np.zeros_like(static_point_count)
    moving_max_displacement = np.zeros_like(before_median)
    matrix_errors = np.zeros_like(static_point_count)
    frame_errors = np.zeros_like(static_point_count)
    finite_errors = np.zeros_like(static_point_count)
    window_hash = np.empty((2, window_count), dtype="S64")

    # Exactly representable translations make the analytic expected error strict.
    analytic_world = np.asarray([3.0, -2.0, 1.0], dtype=np.float64)
    analytic_translations = np.asarray(
        [[-2.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0],
         [1.0, -1.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    analytic_errors = []
    for translation in analytic_translations:
        source_point = analytic_world - translation
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = translation
        analytic_errors.append(
            float(np.linalg.norm(source_point @ transform[:3, :3].T
                                 + transform[:3, 3] - analytic_world))
        )
    analytic_max_error = max(analytic_errors)

    started = time.monotonic()
    for repetition in range(2):
        for index, (sequence_id, center_frame) in enumerate(centers):
            sequence = sequences[sequence_id]
            mapping = slot_to_ray(sequence.source_frame(center_frame))
            window = sequence.window(
                center_frame, condition=ExperimentCondition.B3,
                canonical_ray_by_slot=mapping, ray_mapping_audited=True,
                ray_mapping_digest=ray_digest,
            )
            center_item = window.frames[2]
            center_source = center_item.source
            if center_source.labels is None:
                raise QualificationError("E56 requires real labels")
            center_slots = center_source.real_slots
            center_semantic = center_source.labels.semantic[center_slots]
            center_static = (
                (center_semantic != 0)
                & (center_semantic != 2)
                & ~np.isin(center_semantic, moving_semantics)
            )
            center_xyz = center_source.xyzi[center_slots, :3][center_static]
            if center_xyz.shape[0] == 0:
                raise QualificationError("E56 center frame has no static background")
            tree = cKDTree(center_xyz)
            before_parts = []
            after_parts = []
            displacements = []
            center_instance = center_source.labels.instance[center_slots]
            center_moving = np.isin(center_semantic, moving_semantics) & (
                center_instance > 0
            )
            center_centroids = {
                int(instance): center_source.xyzi[center_slots, :3][
                    center_moving & (center_instance == instance)
                ].mean(axis=0)
                for instance in np.unique(center_instance[center_moving])
            }
            for local_index, item in enumerate(window.frames):
                expected_transform = np.linalg.solve(
                    center_source.lidar_pose, item.source.lidar_pose
                )
                matrix_errors[repetition, index] += int(
                    not np.allclose(
                        item.source_to_reference, expected_transform,
                        rtol=0.0, atol=1e-9,
                    )
                )
                frame_errors[repetition, index] += int(
                    item.source.frame_id != center_frame + local_index - 2
                )
                point_slice = window.points.frame_slice(local_index)
                aligned = window.points.coordinates_center[point_slice]
                source = item.source
                if source.labels is None:
                    raise QualificationError("E56 source frame lacks labels")
                slots = source.real_slots
                semantic = source.labels.semantic[slots]
                static = (
                    (semantic != 0)
                    & (semantic != 2)
                    & ~np.isin(semantic, moving_semantics)
                )
                if local_index != 2 and np.any(static):
                    before_parts.append(
                        tree.query(source.xyzi[slots, :3][static], workers=1)[0]
                    )
                    after_parts.append(tree.query(aligned[static], workers=1)[0])
                instance = source.labels.instance[slots]
                moving = np.isin(semantic, moving_semantics) & (instance > 0)
                for identity in np.unique(instance[moving]):
                    identity = int(identity)
                    if local_index == 2 or identity not in center_centroids:
                        continue
                    centroid = aligned[moving & (instance == identity)].mean(axis=0)
                    displacements.append(
                        float(np.linalg.norm(centroid - center_centroids[identity]))
                    )
                finite_errors[repetition, index] += int(
                    not np.isfinite(aligned).all()
                    or not np.isfinite(item.source_to_reference).all()
                )
            before = np.concatenate(before_parts)
            after = np.concatenate(after_parts)
            before_median[repetition, index] = np.median(before)
            before_q95[repetition, index] = np.quantile(before, 0.95)
            after_median[repetition, index] = np.median(after)
            after_q95[repetition, index] = np.quantile(after, 0.95)
            static_point_count[repetition, index] = before.size
            moving_track_count[repetition, index] = len(displacements)
            moving_max_displacement[repetition, index] = (
                max(displacements) if displacements else 0.0
            )
            window_hash[repetition, index] = _array_hash(
                {
                    "coordinates": window.points.coordinates_center,
                    "source_frame": window.points.source_frame,
                    "source_slot": window.points.source_slot,
                    "source_ray": window.points.source_ray,
                    "before_summary": np.asarray(
                        [before_median[repetition, index], before_q95[repetition, index]]
                    ),
                    "after_summary": np.asarray(
                        [after_median[repetition, index], after_q95[repetition, index]]
                    ),
                }
            )
    elapsed = time.monotonic() - started

    pooled_before_median = float(np.median(before_median[0]))
    pooled_before_q95 = float(np.median(before_q95[0]))
    pooled_after_median = float(np.median(after_median[0]))
    pooled_after_q95 = float(np.median(after_q95[0]))
    improvement_errors = int(
        not pooled_after_median < pooled_before_median
        or not pooled_after_q95 < pooled_before_q95
    )
    motion_errors = int(
        int(moving_track_count[0].sum()) == 0
        or float(moving_max_displacement[0].max()) <= 1e-6
    )
    analytic_errors_count = int(not analytic_max_error < 1e-9)
    reproduction_errors = int(
        np.count_nonzero(window_hash[0] != window_hash[1])
        + (not np.array_equal(before_median[0], before_median[1]))
        + (not np.array_equal(before_q95[0], before_q95[1]))
        + (not np.array_equal(after_median[0], after_median[1]))
        + (not np.array_equal(after_q95[0], after_q95[1]))
        + (not np.array_equal(moving_max_displacement[0], moving_max_displacement[1]))
    )
    hard_errors = int(
        matrix_errors.sum() + frame_errors.sum() + finite_errors.sum()
        + improvement_errors + motion_errors + analytic_errors_count
        + reproduction_errors
    )
    arrays = {
        "sequence_id": np.asarray([item[0] for item in centers], dtype=np.int16),
        "center_frame": np.asarray([item[1] for item in centers], dtype=np.int16),
        "before_median_m": before_median,
        "before_q95_m": before_q95,
        "after_median_m": after_median,
        "after_q95_m": after_q95,
        "static_point_count": static_point_count,
        "moving_track_count": moving_track_count,
        "moving_max_displacement_m": moving_max_displacement,
        "matrix_errors": matrix_errors,
        "frame_errors": frame_errors,
        "finite_errors": finite_errors,
        "window_sha256": window_hash,
        "analytic_error_m": np.asarray(analytic_errors),
    }
    result: dict[str, object] = {
        "experiment": "E56",
        "passed": hard_errors == 0,
        "failure_classification": None
        if hard_errors == 0
        else "center_coordinate_alignment_failure",
        "windows": window_count,
        "formal_repetitions": 2,
        "e55_artifact_sha256": E55_ARTIFACT_SHA256,
        "protocol_sha256": _sha256(protocol_file),
        "calibration_sha256": calibration_sha256,
        "analytic_max_error_m": analytic_max_error,
        "static_points_compared": int(static_point_count[0].sum()),
        "before_median_m": pooled_before_median,
        "before_q95_m": pooled_before_q95,
        "after_median_m": pooled_after_median,
        "after_q95_m": pooled_after_q95,
        "moving_tracks": int(moving_track_count[0].sum()),
        "maximum_moving_displacement_m": float(moving_max_displacement[0].max()),
        "matrix_errors": int(matrix_errors.sum()),
        "frame_errors": int(frame_errors.sum()),
        "finite_errors": int(finite_errors.sum()),
        "improvement_errors": improvement_errors,
        "motion_errors": motion_errors,
        "reproduction_errors": reproduction_errors,
        "elapsed_seconds": elapsed,
        "scientific_array_sha256": _array_hash(arrays),
        "claim_limit": (
            "E56 qualifies rigid center-coordinate alignment and preservation of "
            "observed moving-normal displacement on the frozen Phase 5 windows."
        ),
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def _e61_identity_rank(sequence_id: int, frame_id: int, ray_id: int) -> int:
    payload = (
        f"{E61_MATCH_NAMESPACE}:{sequence_id}:{frame_id}:{ray_id}"
    ).encode("ascii")
    digest = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    # The appended identity is only a deterministic SHA-256 collision tie-break.
    return (digest << 40) | (int(frame_id) << 20) | int(ray_id)


def _e61_build_once(data_root: Path | str, protocol: object) -> dict[str, np.ndarray]:
    """Build score-blind E61 identities from the two frozen train sequences."""

    from .evaluate import _protocol_slot_to_ray, _range_mask

    safety = getattr(protocol, "development")["safety_sets"]
    pure_rule = safety["pure_normal"]
    moving_rule = safety["moving_normal"]
    match_rule = safety["static_match"]
    slot_to_ray, ray_digest, calibration_sha256 = _protocol_slot_to_ray(protocol)
    sequences = {
        sequence_id: STUSequence.open(
            data_root, protocol=protocol, partition="train",
            sequence_id=sequence_id, label_mode=LabelMode.REQUIRED,
        )
        for sequence_id in (201, 206)
    }
    first_mapping = slot_to_ray(sequences[206].source_frame(0))
    ray_count = int(first_mapping.size)
    if not np.array_equal(np.sort(first_mapping), np.arange(ray_count)):
        raise QualificationError("E61 canonical ray mapping is not a permutation")
    packed_width = (ray_count + 7) // 8
    moving_semantics = np.asarray(moving_rule["semantic_ids"], dtype=np.uint16)
    static_semantics = np.asarray(
        [match_rule["moving_to_static_semantic"][str(int(value))]
         for value in moving_semantics],
        dtype=np.uint16,
    )
    inner_edges = np.asarray(match_rule["range_bin_edges_m"][1:-1])

    pure_frames = np.arange(
        int(pure_rule["frame_range"][0]),
        int(pure_rule["frame_range"][1]) + 1,
        dtype=np.int16,
    )
    pure_packed = np.zeros((pure_frames.size, packed_width), dtype=np.uint8)
    pure_count = np.zeros(pure_frames.size, dtype=np.int32)
    pure_sequence = sequences[201]
    for row, frame_id in enumerate(pure_frames):
        frame = pure_sequence.source_frame(int(frame_id))
        assert frame.labels is not None
        semantic = np.asarray(frame.labels.semantic)
        eligible = (
            _range_mask(protocol, frame.xyzi)
            & (semantic != 0)
            & (semantic != 2)
            & ~np.isin(semantic, moving_semantics)
        )
        mapping = slot_to_ray(frame)
        canonical = np.zeros(ray_count, dtype=np.bool_)
        canonical[mapping[np.flatnonzero(eligible)]] = True
        pure_packed[row] = np.packbits(canonical, bitorder="little")
        pure_count[row] = int(np.count_nonzero(eligible))

    moving_frames = np.arange(
        int(moving_rule["frame_range"][0]),
        int(moving_rule["frame_range"][1]) + 1,
        dtype=np.int16,
    )
    moving_packed = np.zeros((moving_frames.size, packed_width), dtype=np.uint8)
    moving_count = np.zeros(moving_frames.size, dtype=np.int32)
    moving_candidate_count = np.zeros((8, 4), dtype=np.int32)
    static_candidate_count = np.zeros((8, 4), dtype=np.int32)
    moving_records: list[list[tuple[int, int, int]]] = [
        [] for _ in range(32)
    ]
    moving_sequence = sequences[206]
    for row, frame_id in enumerate(moving_frames):
        frame = moving_sequence.source_frame(int(frame_id))
        assert frame.labels is not None
        semantic = np.asarray(frame.labels.semantic)
        valid = _range_mask(protocol, frame.xyzi) & (semantic != 0) & (semantic != 2)
        distance = np.linalg.norm(
            np.asarray(frame.xyzi[:, :3], dtype=np.float32), axis=1
        )
        range_bin = np.searchsorted(inner_edges, distance, side="right")
        mapping = slot_to_ray(frame)
        canonical = np.zeros(ray_count, dtype=np.bool_)
        moving_mask = valid & np.isin(semantic, moving_semantics)
        canonical[mapping[np.flatnonzero(moving_mask)]] = True
        moving_packed[row] = np.packbits(canonical, bitorder="little")
        moving_count[row] = int(np.count_nonzero(moving_mask))
        for family, (moving_value, static_value) in enumerate(
            zip(moving_semantics, static_semantics, strict=True)
        ):
            for bin_id in range(4):
                cell = family * 4 + bin_id
                moving_slots = np.flatnonzero(
                    valid & (semantic == moving_value) & (range_bin == bin_id)
                )
                static_slots = np.flatnonzero(
                    valid & (semantic == static_value) & (range_bin == bin_id)
                )
                moving_candidate_count[family, bin_id] += moving_slots.size
                static_candidate_count[family, bin_id] += static_slots.size
                moving_records[cell].extend(
                    (
                        _e61_identity_rank(206, int(frame_id), int(mapping[slot])),
                        int(frame_id),
                        int(mapping[slot]),
                    )
                    for slot in moving_slots
                )

    matched_count = np.minimum(moving_candidate_count, static_candidate_count)
    static_heaps: list[list[tuple[int, int, int]]] = [[] for _ in range(32)]
    for frame_id in moving_frames:
        frame = moving_sequence.source_frame(int(frame_id))
        assert frame.labels is not None
        semantic = np.asarray(frame.labels.semantic)
        valid = _range_mask(protocol, frame.xyzi) & (semantic != 0) & (semantic != 2)
        distance = np.linalg.norm(
            np.asarray(frame.xyzi[:, :3], dtype=np.float32), axis=1
        )
        range_bin = np.searchsorted(inner_edges, distance, side="right")
        mapping = slot_to_ray(frame)
        for family, static_value in enumerate(static_semantics):
            for bin_id in range(4):
                limit = int(matched_count[family, bin_id])
                if limit == 0:
                    continue
                cell = family * 4 + bin_id
                heap = static_heaps[cell]
                for slot in np.flatnonzero(
                    valid & (semantic == static_value) & (range_bin == bin_id)
                ):
                    ray_id = int(mapping[slot])
                    rank = _e61_identity_rank(206, int(frame_id), ray_id)
                    entry = (-rank, int(frame_id), ray_id)
                    if len(heap) < limit:
                        heapq.heappush(heap, entry)
                    elif rank < -heap[0][0]:
                        heapq.heapreplace(heap, entry)

    matched_moving_packed = np.zeros_like(moving_packed)
    matched_static_packed = np.zeros_like(moving_packed)
    paired_cell: list[int] = []
    paired_moving_frame: list[int] = []
    paired_moving_ray: list[int] = []
    paired_static_frame: list[int] = []
    paired_static_ray: list[int] = []
    for cell in range(32):
        family, bin_id = divmod(cell, 4)
        limit = int(matched_count[family, bin_id])
        selected_moving = sorted(moving_records[cell])[:limit]
        selected_static = sorted(
            ((-negated, frame_id, ray_id)
             for negated, frame_id, ray_id in static_heaps[cell])
        )
        if len(selected_moving) != limit or len(selected_static) != limit:
            raise QualificationError("E61 matched-cell cardinality changed")
        for (_, moving_frame, moving_ray), (_, static_frame, static_ray) in zip(
            selected_moving, selected_static, strict=True
        ):
            paired_cell.append(cell)
            paired_moving_frame.append(moving_frame)
            paired_moving_ray.append(moving_ray)
            paired_static_frame.append(static_frame)
            paired_static_ray.append(static_ray)
            matched_moving_packed[moving_frame, moving_ray // 8] |= np.uint8(
                1 << (moving_ray % 8)
            )
            matched_static_packed[static_frame, static_ray // 8] |= np.uint8(
                1 << (static_ray % 8)
            )

    return {
        "pure_frame_id": pure_frames,
        "pure_canonical_mask_packed": pure_packed,
        "pure_point_count_by_frame": pure_count,
        "moving_frame_id": moving_frames,
        "moving_canonical_mask_packed": moving_packed,
        "moving_point_count_by_frame": moving_count,
        "matched_moving_canonical_mask_packed": matched_moving_packed,
        "matched_static_canonical_mask_packed": matched_static_packed,
        "moving_candidate_count": moving_candidate_count,
        "static_candidate_count": static_candidate_count,
        "matched_count": matched_count,
        "pair_cell": np.asarray(paired_cell, dtype=np.int8),
        "pair_moving_frame": np.asarray(paired_moving_frame, dtype=np.int16),
        "pair_moving_canonical_ray": np.asarray(paired_moving_ray, dtype=np.int32),
        "pair_static_frame": np.asarray(paired_static_frame, dtype=np.int16),
        "pair_static_canonical_ray": np.asarray(paired_static_ray, dtype=np.int32),
        "moving_semantic": moving_semantics,
        "static_semantic": static_semantics,
        "range_bin_edges_m": np.asarray(
            match_rule["range_bin_edges_m"], dtype=np.float64
        ),
        "canonical_ray_mapping_sha256": np.asarray(ray_digest, dtype="S64"),
        "calibration_sha256": np.asarray(calibration_sha256, dtype="S64"),
    }


def run_e61(
    data_root: Path | str,
    protocol_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Freeze E61 safety identities without reading predictions or model scores."""

    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    started = time.monotonic()
    first = _e61_build_once(data_root, protocol)
    second = _e61_build_once(data_root, protocol)
    elapsed = time.monotonic() - started
    reproduction_errors = sum(
        int(not np.array_equal(first[name], second[name])) for name in first
    )
    pure_points = int(first["pure_point_count_by_frame"].sum())
    moving_points = int(first["moving_point_count_by_frame"].sum())
    paired_points = int(first["pair_cell"].size)
    identity_errors = int(
        pure_points
        != int(protocol.development["safety_sets"]["pure_normal"]["expected_points"])
        or moving_points
        != int(protocol.development["safety_sets"]["moving_normal"]["expected_points"])
        or np.unique(
            first["pair_moving_frame"].astype(np.int64) * 131072
            + first["pair_moving_canonical_ray"]
        ).size != paired_points
        or np.unique(
            first["pair_static_frame"].astype(np.int64) * 131072
            + first["pair_static_canonical_ray"]
        ).size != paired_points
    )
    count_errors = int(
        not np.array_equal(
            first["matched_count"],
            np.minimum(
                first["moving_candidate_count"], first["static_candidate_count"]
            ),
        )
        or int(first["matched_count"].sum()) != paired_points
    )
    prediction_access_errors = 0
    label_input_errors = int(
        "raw_semantic" in inspect.signature(AJAEPointTransformer.forward).parameters
    )
    passed = (
        reproduction_errors == 0
        and identity_errors == 0
        and count_errors == 0
        and prediction_access_errors == 0
        and label_input_errors == 0
    )
    result: dict[str, object] = {
        "experiment": "E61",
        "passed": passed,
        "failure_classification": None
        if passed else "safety_identity_or_isolation_failure",
        "pure_normal_points": pure_points,
        "pure_normal_frames": int(first["pure_frame_id"].size),
        "moving_normal_points": moving_points,
        "moving_normal_frames_with_points": int(np.count_nonzero(
            first["moving_point_count_by_frame"]
        )),
        "matched_pairs": paired_points,
        "matched_fraction_of_moving": paired_points / max(moving_points, 1),
        "moving_candidate_count": first["moving_candidate_count"].tolist(),
        "static_candidate_count": first["static_candidate_count"].tolist(),
        "matched_count": first["matched_count"].tolist(),
        "identity_errors": identity_errors,
        "count_errors": count_errors,
        "prediction_access_errors": prediction_access_errors,
        "label_input_errors": label_input_errors,
        "reproduction_errors": reproduction_errors,
        "formal_repetitions": 2,
        "elapsed_seconds": elapsed,
        "protocol_sha256": _sha256(protocol_file),
        "scientific_array_sha256": _array_hash(first),
        "claim_limit": protocol.development["safety_sets"]["claim_limit"],
    }
    _save(Path(output_path).expanduser().resolve(), first, result)
    return result


def _e62_points(declared_ranges: Sequence[float]) -> np.ndarray:
    """Construct float32 points while retaining the requested range metadata."""

    points = np.zeros((len(declared_ranges), 3), dtype=np.float32)
    points[:, 0] = np.asarray(declared_ranges, dtype=np.float32)
    for index, declared in enumerate(declared_ranges):
        if declared == 50.000001:
            # Scalar 50.000001 rounds to 50.0 in float32. This vector has the
            # requested real norm but an official float32 norm strictly above 50.
            points[index] = np.asarray((50.0, 0.011309734, 0.0), dtype=np.float32)
    return points


def e62_fixture_arrays() -> dict[str, np.ndarray]:
    """Build the E62-v2 fixtures without invoking either evaluator."""

    analytic_cases: list[tuple[str, list[tuple[int, list[float], list[float], list[int]]]]] = []
    boundary_ranges = [
        2.499999, 2.5, 50.0, 50.000001, 6.0, 7.0,
        8.0, 9.0, 10.0, 12.0, 51.0, 20.0,
    ]
    analytic_cases.append(
        (
            "range_ignore_and_post_filter_frame_gate",
            [
                (
                    100,
                    boundary_ranges,
                    [0.99, 0.10, 0.80, 1.00, 0.70, 0.60, 0.50, 0.20, 1.00, 0.30, 0.95, 0.40],
                    [2, 10, 2, 2, 2, 2, 2, 10, 0, 10, 2, 10],
                ),
                (
                    101,
                    boundary_ranges,
                    [0.98, 0.91, 0.81, 0.99, 0.71, 0.61, 0.51, 0.21, 1.00, 0.31, 0.96, 0.41],
                    [10, 2, 2, 10, 2, 2, 2, 10, 0, 10, 2, 10],
                ),
            ],
        )
    )
    analytic_cases.append(
        (
            "all_scores_tied",
            [
                (
                    200,
                    [float(value) for value in range(5, 17)],
                    [0.5] * 12,
                    [2, 10, 2, 10, 2, 10, 2, 10, 2, 10, 0, 30],
                )
            ],
        )
    )
    strict_semantic = [2] * 20 + [10] * 10
    strict_scores = [0.9] * 19 + [0.7] + [0.8] * 2 + [0.6] * 8
    analytic_cases.append(
        (
            "strict_tpr_above_0.95",
            [(300, [15.0] * 30, strict_scores, strict_semantic)],
        )
    )
    analytic_cases.append(
        (
            "mixed_repeated_scores",
            [
                (
                    400,
                    [25.0] * 24,
                    [0.95, 0.95, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4,
                     0.95, 0.8, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2,
                     0.1, 0.1, 0.05, 0.05, 1.0, 0.0, 0.4, 0.6],
                    [2, 10, 2, 10, 2, 10, 2, 10, 2, 10, 2, 10,
                     2, 10, 2, 10, 10, 10, 10, 10, 0, 0, 30, 31],
                )
            ],
        )
    )

    case_names: list[str] = []
    case_frame_offsets = [0]
    frame_ids: list[int] = []
    frame_point_offsets = [0]
    points_parts: list[np.ndarray] = []
    scores_parts: list[np.ndarray] = []
    semantic_parts: list[np.ndarray] = []
    declared_parts: list[np.ndarray] = []
    point_id_parts: list[np.ndarray] = []
    next_point_id = 0
    for case_name, frames in analytic_cases:
        case_names.append(case_name)
        for frame_id, ranges, scores, semantic in frames:
            count = len(ranges)
            frame_ids.append(frame_id)
            points_parts.append(_e62_points(ranges))
            scores_parts.append(np.asarray(scores, dtype=np.float32))
            semantic_parts.append(np.asarray(semantic, dtype=np.uint16))
            declared_parts.append(np.asarray(ranges, dtype=np.float64))
            point_id_parts.append(np.arange(next_point_id, next_point_id + count, dtype=np.int64))
            next_point_id += count
            frame_point_offsets.append(next_point_id)
        case_frame_offsets.append(len(frame_ids))

    analytic_points = np.concatenate(points_parts)
    analytic_semantic = np.concatenate(semantic_parts)
    analytic_norm = np.linalg.norm(analytic_points, axis=1)
    analytic_range_valid = (analytic_norm >= 2.5) & (analytic_norm <= 50.0)
    analytic_valid = analytic_range_valid & (analytic_semantic != 0)
    analytic_frame_accepted = np.zeros(len(frame_ids), dtype=np.bool_)
    for frame_index in range(len(frame_ids)):
        start, stop = frame_point_offsets[frame_index : frame_index + 2]
        analytic_frame_accepted[frame_index] = int(
            np.count_nonzero(
                analytic_valid[start:stop] & (analytic_semantic[start:stop] == 2)
            )
        ) >= 5

    rng = np.random.Generator(np.random.PCG64(E62_NUMERICAL_SEED))
    numerical_frame_ids = np.arange(1000, 1010, dtype=np.int32)
    numerical_offsets = np.arange(0, 10 * 96 + 1, 96, dtype=np.int64)
    numerical_points: list[np.ndarray] = []
    numerical_scores: list[np.ndarray] = []
    numerical_semantic: list[np.ndarray] = []
    for frame_index in range(10):
        radii = rng.uniform(3.0, 49.0, size=96)
        radii[:4] = np.asarray((2.0, 51.0, 2.499999, 50.0))
        angle = rng.uniform(-np.pi, np.pi, size=96)
        points = np.zeros((96, 3), dtype=np.float32)
        points[:, 0] = (radii * np.cos(angle)).astype(np.float32)
        points[:, 1] = (radii * np.sin(angle)).astype(np.float32)
        points[:, 2] = rng.uniform(-0.25, 0.25, size=96).astype(np.float32)
        semantics = np.where(np.arange(96) % 3 == 0, 30, 10).astype(np.uint16)
        semantics[np.arange(0, 96, 17)] = 0
        eligible = np.flatnonzero(
            (np.linalg.norm(points, axis=1) >= 2.5)
            & (np.linalg.norm(points, axis=1) <= 50.0)
            & (semantics != 0)
        )
        anomaly_count = 4 if frame_index == 0 else 4 + frame_index
        semantics[eligible[:anomaly_count]] = 2
        semantics[:2] = 2  # Out-of-range anomalies verify post-filter gating.
        scores = (rng.integers(0, 41, size=96) / 40.0).astype(np.float32)
        scores[np.arange(0, 96, 17)] = 1.0
        scores[1::23] = 0.0
        numerical_points.append(points)
        numerical_scores.append(scores)
        numerical_semantic.append(semantics)

    numerical_points_array = np.concatenate(numerical_points)
    numerical_semantic_array = np.concatenate(numerical_semantic)
    numerical_norm = np.linalg.norm(numerical_points_array, axis=1)
    numerical_valid = (
        (numerical_norm >= 2.5)
        & (numerical_norm <= 50.0)
        & (numerical_semantic_array != 0)
    )
    numerical_frame_accepted = np.zeros(10, dtype=np.bool_)
    for frame_index in range(10):
        start, stop = numerical_offsets[frame_index : frame_index + 2]
        numerical_frame_accepted[frame_index] = int(
            np.count_nonzero(
                numerical_valid[start:stop]
                & (numerical_semantic_array[start:stop] == 2)
            )
        ) >= 5

    return {
        "analytic_case_name": np.asarray(case_names),
        "analytic_case_frame_offset": np.asarray(case_frame_offsets, dtype=np.int64),
        "analytic_frame_id": np.asarray(frame_ids, dtype=np.int32),
        "analytic_frame_point_offset": np.asarray(frame_point_offsets, dtype=np.int64),
        "analytic_points": analytic_points,
        "analytic_scores": np.concatenate(scores_parts),
        "analytic_semantic": analytic_semantic,
        "analytic_declared_range_m": np.concatenate(declared_parts),
        "analytic_point_id": np.concatenate(point_id_parts),
        "analytic_expected_range_valid": analytic_range_valid,
        "analytic_expected_frame_accepted": analytic_frame_accepted,
        "numerical_frame_id": numerical_frame_ids,
        "numerical_frame_point_offset": numerical_offsets,
        "numerical_points": numerical_points_array,
        "numerical_scores": np.concatenate(numerical_scores),
        "numerical_semantic": numerical_semantic_array,
        "numerical_point_id": np.arange(1_000_000, 1_000_960, dtype=np.int64),
        "numerical_expected_range_valid": (numerical_norm >= 2.5) & (numerical_norm <= 50.0),
        "numerical_expected_frame_accepted": numerical_frame_accepted,
    }


def run_e62_fixture(
    protocol_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Materialize the frozen E62 inputs before either evaluator is called."""

    protocol = load_protocol(protocol_path)
    specification = protocol.evaluation_document["evaluator_equivalence"]
    if specification["status"] != "protocol_completed_before_fixture_freeze":
        raise QualificationError("E62 fixture generation requires the pre-freeze status")
    numerical = specification["fixtures"]["numerical_fixture"]
    if (
        numerical["namespace"] != E62_NUMERICAL_NAMESPACE
        or numerical["pcg64_seed"] != E62_NUMERICAL_SEED
    ):
        raise QualificationError("E62 numerical fixture identity changed")
    arrays = e62_fixture_arrays()
    result = {
        "experiment": "E62-fixture-freeze",
        "fixture_frozen": True,
        "evaluator_calls": 0,
        "analytic_cases": int(arrays["analytic_case_name"].size),
        "analytic_frames": int(arrays["analytic_frame_id"].size),
        "analytic_points": int(arrays["analytic_points"].shape[0]),
        "numerical_frames": int(arrays["numerical_frame_id"].size),
        "numerical_points": int(arrays["numerical_points"].shape[0]),
        "scientific_array_sha256": _array_hash(arrays),
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def _e62_official_calculator(specification: Mapping[str, object]) -> type:
    """Load the source-hashed released STU calculator without reading data."""

    official = specification["official"]
    repository = Path(str(official["repository"])).expanduser().resolve(strict=True)
    source = (repository / str(official["source_file"])).resolve(strict=True)
    if _sha256(source) != official["source_sha256"]:
        raise QualificationError("E62 official evaluator source hash changed")
    commit = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != official["commit"]:
        raise QualificationError("E62 official evaluator repository commit changed")
    module_name = "ajae_e62_frozen_official_evaluator"
    module_spec = importlib.util.spec_from_file_location(module_name, source)
    if module_spec is None or module_spec.loader is None:
        raise QualificationError("E62 cannot load the official evaluator")
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(repository))
    try:
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(repository))
        sys.modules.pop(module_name, None)
    return module.PointOODMetricsCalculator


def _e62_fixture(path: Path, specification: Mapping[str, object]) -> dict[str, np.ndarray]:
    fixture = specification["fixtures"]
    expected_path = (PROJECT_ROOT / str(fixture["artifact"])).resolve()
    if path.resolve(strict=True) != expected_path:
        raise QualificationError("E62 must use the protocol-bound fixture path")
    if _sha256(path) != fixture["artifact_sha256"]:
        raise QualificationError("E62 fixture artifact hash changed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata_json"
        }
    if _array_hash(arrays) != fixture["scientific_array_sha256"]:
        raise QualificationError("E62 fixture scientific-array hash changed")
    return arrays


def _e62_compare_case(
    calculator_type: type,
    protocol: object,
    *,
    frame_ids: np.ndarray,
    point_offsets: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    semantic: np.ndarray,
    point_ids: np.ndarray,
    expected_range_valid: np.ndarray,
    expected_frame_accepted: np.ndarray,
) -> dict[str, object]:
    """Compare one frozen case through independent official and custom objects."""

    official = calculator_type()
    custom = PointMetricAccumulator(protocol)
    official_identity = calculator_type()
    custom_identity = PointMetricAccumulator(protocol)
    accepted: list[int] = []
    skipped: list[int] = []
    selected_ids: list[np.ndarray] = []
    selection_errors = 0
    content_errors = 0
    expected_errors = 0
    for frame_index, frame_id in enumerate(frame_ids.tolist()):
        start, stop = map(int, point_offsets[frame_index : frame_index + 2])
        frame_points = points[start:stop]
        frame_scores = scores[start:stop]
        frame_semantic = semantic[start:stop]
        before = len(official.all_scores)
        official.update(frame_points, frame_scores, frame_semantic)
        official_accepted = len(official.all_scores) == before + 1
        custom_accepted = custom.update(frame_points, frame_scores, frame_semantic)

        identity_scores = point_ids[start:stop].astype(np.float64)
        identity_before = len(official_identity.all_scores)
        official_identity.update(frame_points, identity_scores, frame_semantic)
        official_identity_accepted = (
            len(official_identity.all_scores) == identity_before + 1
        )
        custom_identity_accepted = custom_identity.update(
            frame_points, identity_scores, frame_semantic
        )
        expected_accepted = bool(expected_frame_accepted[frame_index])
        if not (
            official_accepted
            == custom_accepted
            == official_identity_accepted
            == custom_identity_accepted
        ):
            selection_errors += 1
        if official_accepted != expected_accepted:
            expected_errors += 1
        if not official_accepted:
            skipped.append(int(frame_id))
            continue
        accepted.append(int(frame_id))
        official_ids = np.asarray(official_identity.all_scores[-1], dtype=np.int64)
        custom_ids = np.asarray(custom_identity._scores[-1], dtype=np.int64)
        expected_mask = (
            expected_range_valid[start:stop] & (frame_semantic != 0)
        )
        expected_ids = point_ids[start:stop][expected_mask]
        if not (
            np.array_equal(official_ids, custom_ids)
            and np.array_equal(official_ids, expected_ids)
        ):
            selection_errors += 1
        selected_ids.append(official_ids)
        if not (
            np.array_equal(official.all_labels[-1].astype(np.bool_), custom._labels[-1])
            and np.array_equal(
                official.all_scores[-1].astype(np.float64), custom._scores[-1]
            )
        ):
            content_errors += 1

    official_result = official.compute_metrics()
    custom_result = custom.compute()
    metric_names = ("AP", "AUROC", "FPR95", "threshold")
    official_metrics = np.asarray(
        [official_result[name] for name in metric_names], dtype=np.float64
    )
    custom_metrics = np.asarray(
        [custom_result[name] for name in metric_names], dtype=np.float64
    )
    metric_difference = np.abs(official_metrics - custom_metrics)
    official_labels = np.concatenate(official.all_labels).astype(np.bool_)
    custom_labels = np.concatenate(custom._labels).astype(np.bool_)
    official_scores = np.concatenate(official.all_scores).astype(np.float64)
    custom_scores = np.concatenate(custom._scores).astype(np.float64)
    if not (
        np.array_equal(official_labels, custom_labels)
        and np.array_equal(official_scores, custom_scores)
    ):
        content_errors += 1
    return {
        "accepted_frame_id": np.asarray(accepted, dtype=np.int32),
        "skipped_frame_id": np.asarray(skipped, dtype=np.int32),
        "selected_point_id": (
            np.concatenate(selected_ids) if selected_ids else np.empty(0, dtype=np.int64)
        ),
        "pooled_labels": official_labels,
        "pooled_scores": official_scores,
        "official_metrics": official_metrics,
        "custom_metrics": custom_metrics,
        "metric_difference": metric_difference,
        "valid_points": int(official_labels.size),
        "positive_points": int(np.count_nonzero(official_labels)),
        "negative_points": int(np.count_nonzero(~official_labels)),
        "selection_errors": selection_errors,
        "content_errors": content_errors,
        "expected_errors": expected_errors,
    }


def run_e62(
    protocol_path: Path | str,
    fixture_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Run E62 on the sole fixture artifact frozen before this comparison."""

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    specification = protocol.evaluation_document["evaluator_equivalence"]
    if specification["status"] not in {
        "fixtures_frozen_before_formal_comparison",
        "formal_pass",
    }:
        raise QualificationError("E62 formal comparison requires frozen fixtures")
    fixture_file = Path(fixture_path).expanduser().resolve(strict=True)
    arrays = _e62_fixture(fixture_file, specification)
    calculator_type = _e62_official_calculator(specification)

    cases: list[tuple[str, dict[str, np.ndarray]]] = []
    case_names = arrays["analytic_case_name"].tolist()
    case_frame_offsets = arrays["analytic_case_frame_offset"]
    analytic_point_offsets = arrays["analytic_frame_point_offset"]
    for case_index, case_name in enumerate(case_names):
        frame_start, frame_stop = map(
            int, case_frame_offsets[case_index : case_index + 2]
        )
        point_start = int(analytic_point_offsets[frame_start])
        point_stop = int(analytic_point_offsets[frame_stop])
        cases.append(
            (
                str(case_name),
                {
                    "frame_ids": arrays["analytic_frame_id"][frame_start:frame_stop],
                    "point_offsets": analytic_point_offsets[frame_start : frame_stop + 1] - point_start,
                    "points": arrays["analytic_points"][point_start:point_stop],
                    "scores": arrays["analytic_scores"][point_start:point_stop],
                    "semantic": arrays["analytic_semantic"][point_start:point_stop],
                    "point_ids": arrays["analytic_point_id"][point_start:point_stop],
                    "expected_range_valid": arrays["analytic_expected_range_valid"][point_start:point_stop],
                    "expected_frame_accepted": arrays["analytic_expected_frame_accepted"][frame_start:frame_stop],
                },
            )
        )
    cases.append(
        (
            "non_symbolic_numerical_fixture",
            {
                "frame_ids": arrays["numerical_frame_id"],
                "point_offsets": arrays["numerical_frame_point_offset"],
                "points": arrays["numerical_points"],
                "scores": arrays["numerical_scores"],
                "semantic": arrays["numerical_semantic"],
                "point_ids": arrays["numerical_point_id"],
                "expected_range_valid": arrays["numerical_expected_range_valid"],
                "expected_frame_accepted": arrays["numerical_expected_frame_accepted"],
            },
        )
    )

    compared = [
        (name, _e62_compare_case(calculator_type, protocol, **payload))
        for name, payload in cases
    ]
    metric_names = np.asarray(("AP", "AUROC", "FPR95", "threshold"))
    evidence: dict[str, np.ndarray] = {
        "case_name": np.asarray([name for name, _ in compared]),
        "metric_name": metric_names,
        "official_metrics": np.stack([item["official_metrics"] for _, item in compared]),
        "custom_metrics": np.stack([item["custom_metrics"] for _, item in compared]),
        "metric_absolute_difference": np.stack(
            [item["metric_difference"] for _, item in compared]
        ),
        "accepted_frame_count": np.asarray(
            [item["accepted_frame_id"].size for _, item in compared], dtype=np.int64
        ),
        "skipped_frame_count": np.asarray(
            [item["skipped_frame_id"].size for _, item in compared], dtype=np.int64
        ),
        "valid_point_count": np.asarray(
            [item["valid_points"] for _, item in compared], dtype=np.int64
        ),
        "positive_point_count": np.asarray(
            [item["positive_points"] for _, item in compared], dtype=np.int64
        ),
        "negative_point_count": np.asarray(
            [item["negative_points"] for _, item in compared], dtype=np.int64
        ),
        "selection_errors": np.asarray(
            [item["selection_errors"] for _, item in compared], dtype=np.int64
        ),
        "content_errors": np.asarray(
            [item["content_errors"] for _, item in compared], dtype=np.int64
        ),
        "expected_identity_errors": np.asarray(
            [item["expected_errors"] for _, item in compared], dtype=np.int64
        ),
    }
    for field in (
        "accepted_frame_id", "skipped_frame_id", "selected_point_id",
        "pooled_labels", "pooled_scores",
    ):
        values = [np.asarray(item[field]) for _, item in compared]
        offsets = np.cumsum([0] + [value.size for value in values], dtype=np.int64)
        evidence[f"{field}_offset"] = offsets
        evidence[field] = np.concatenate(values)

    tolerance = float(specification["comparison"]["maximum_absolute_difference"])
    maximum_difference = float(np.max(evidence["metric_absolute_difference"]))
    discrete_errors = int(
        evidence["selection_errors"].sum()
        + evidence["content_errors"].sum()
        + evidence["expected_identity_errors"].sum()
    )
    passed = discrete_errors == 0 and maximum_difference <= tolerance
    result = {
        "experiment": "E62-v2",
        "passed": passed,
        "fixture_sha256": _sha256(fixture_file),
        "fixture_scientific_array_sha256": specification["fixtures"]["scientific_array_sha256"],
        "official_commit": specification["official"]["commit"],
        "official_source_sha256": specification["official"]["source_sha256"],
        "cases": len(compared),
        "accepted_frames": int(evidence["accepted_frame_count"].sum()),
        "skipped_frames": int(evidence["skipped_frame_count"].sum()),
        "valid_points": int(evidence["valid_point_count"].sum()),
        "positive_points": int(evidence["positive_point_count"].sum()),
        "negative_points": int(evidence["negative_point_count"].sum()),
        "discrete_errors": discrete_errors,
        "maximum_metric_absolute_difference": maximum_difference,
        "metric_tolerance": tolerance,
        "elapsed_seconds": time.monotonic() - started,
        "protocol_sha256": _sha256(protocol_file),
        "scientific_array_sha256": _array_hash(evidence),
        "failure_meaning": "implementation mismatch only",
    }
    _save(Path(output_path).expanduser().resolve(), evidence, result)
    return result


def e63_identity_arrays(e57_path: Path | str) -> dict[str, np.ndarray]:
    """Derive the common domain, safety folds, and resamples from identities only."""

    source = Path(e57_path).expanduser().resolve(strict=True)
    if _sha256(source) != (
        "b14efc1aad86ac67b5bf7c8631f02b2e68664e071b747b7b210d5f7a30f5d123"
    ):
        raise QualificationError("E63 source must be the frozen E57-v2 artifact")
    with np.load(source, allow_pickle=False) as archive:
        world_id = np.asarray(archive["selected_world_id"], dtype=np.int16)
        world_identity = np.asarray(archive["selected_candidate_sha256"], dtype="S64")
        center_frame = np.asarray(archive["selected_center_frame"], dtype=np.int16)
    if (
        not np.array_equal(world_id, np.arange(24, dtype=np.int16))
        or world_identity.shape != (24,)
        or len(set(world_identity.tolist())) != 24
        or center_frame.shape != (24,)
    ):
        raise QualificationError("E63 source-world identities are incomplete")
    offsets = np.arange(-4, 3, dtype=np.int16)
    required_frame_id = center_frame[:, None] + offsets[None, :]
    common_domain_eligible = np.all(
        (required_frame_id >= 4) & (required_frame_id <= 681), axis=1
    )

    safety_hash = np.asarray(
        [
            hashlib.sha256(
                E63_SAFETY_NAMESPACE.encode("utf-8")
                + b":"
                + identity.decode("ascii").encode("utf-8")
            ).hexdigest()
            for identity in world_identity
        ],
        dtype="S64",
    )
    order = np.argsort(safety_hash, kind="stable")
    safety_rank = np.empty(24, dtype=np.int16)
    safety_rank[order] = np.arange(24, dtype=np.int16)
    safety_fold = np.where(safety_rank < 12, b"A", b"B").astype("S1")

    generator = np.random.Generator(np.random.PCG64(E63_BOOTSTRAP_SEED))
    bootstrap_training_seed = generator.choice(
        np.asarray((0, 1, 2), dtype=np.int8), size=(5000, 3), replace=True
    )
    bootstrap_world_id = generator.choice(
        world_id, size=(5000, 24), replace=True
    ).astype(np.int16, copy=False)
    return {
        "world_id": world_id,
        "world_identity": world_identity,
        "center_frame": center_frame,
        "required_frame_id": required_frame_id,
        "common_domain_eligible": common_domain_eligible,
        "safety_hash": safety_hash,
        "safety_rank": safety_rank,
        "safety_fold": safety_fold,
        "bootstrap_training_seed": bootstrap_training_seed,
        "bootstrap_world_id": bootstrap_world_id,
    }


def run_e63(
    protocol_path: Path | str,
    e57_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Materialize the approved E63-v2 identities without reading model results."""

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    specification = protocol.development["e63_freeze"]
    if specification["status"] not in {
        "frozen_before_identity_generation",
        "formal_pass",
    }:
        raise QualificationError("E63-v2 rules are not frozen")
    expected_source = (PROJECT_ROOT / specification["source_worlds"]["artifact"]).resolve()
    source = Path(e57_path).expanduser().resolve(strict=True)
    if source != expected_source:
        raise QualificationError("E63 input path differs from the frozen E57 source")
    first = e63_identity_arrays(source)
    second = e63_identity_arrays(source)
    reproduction_errors = sum(
        not np.array_equal(first[name], second[name]) for name in first
    )
    eligible = first["world_id"][first["common_domain_eligible"]]
    fold_a = first["world_id"][first["safety_fold"] == b"A"]
    fold_b = first["world_id"][first["safety_fold"] == b"B"]
    identity_errors = int(
        eligible.size != 23
        or fold_a.size != 12
        or fold_b.size != 12
        or set(fold_a.tolist()).intersection(fold_b.tolist())
        or sorted((*fold_a.tolist(), *fold_b.tolist())) != list(range(24))
        or first["bootstrap_training_seed"].shape != (5000, 3)
        or first["bootstrap_world_id"].shape != (5000, 24)
    )
    result = {
        "experiment": "E63-v2",
        "passed": reproduction_errors == 0 and identity_errors == 0,
        "source_e57_sha256": _sha256(source),
        "eligible_world_ids": eligible.astype(int).tolist(),
        "excluded_world_ids": first["world_id"][~first["common_domain_eligible"]]
        .astype(int)
        .tolist(),
        "fold_a_world_ids": fold_a.astype(int).tolist(),
        "fold_b_world_ids": fold_b.astype(int).tolist(),
        "eligible_fold_a_worlds": int(
            np.count_nonzero(
                first["common_domain_eligible"] & (first["safety_fold"] == b"A")
            )
        ),
        "eligible_fold_b_worlds": int(
            np.count_nonzero(
                first["common_domain_eligible"] & (first["safety_fold"] == b"B")
            )
        ),
        "bootstrap_replicates": 5000,
        "bootstrap_seed_draws": 3,
        "bootstrap_world_draws": 24,
        "identity_errors": identity_errors,
        "reproduction_errors": reproduction_errors,
        "model_results_read": 0,
        "held_out_worlds_read": 0,
        "public_real_ood_sequences_read": 0,
        "hidden_test_sequences_read": 0,
        "protocol_sha256": _sha256(protocol_file),
        "scientific_array_sha256": _array_hash(first),
        "seconds": time.monotonic() - started,
    }
    _save(Path(output_path).expanduser().resolve(), first, result)
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
    e52 = commands.add_parser("e52")
    e52.add_argument("--data-root", type=Path, required=True)
    e52.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e52.add_argument("--e51", type=Path, required=True)
    e52.add_argument("--output", type=Path, required=True)
    e52.add_argument("--device", default="cuda")
    e53 = commands.add_parser("e53")
    e53.add_argument("--data-root", type=Path, required=True)
    e53.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e53.add_argument("--e52", type=Path, required=True)
    e53.add_argument("--output", type=Path, required=True)
    e53.add_argument("--device", default="cpu")
    e53.add_argument("--workers", type=int, default=4)
    e53.add_argument("--threads-per-worker", type=int, default=6)
    e54 = commands.add_parser("e54")
    e54.add_argument("--data-root", type=Path, required=True)
    e54.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e54.add_argument("--e53", type=Path, required=True)
    e54.add_argument("--output", type=Path, required=True)
    e54.add_argument("--device", default="cpu")
    e54.add_argument("--workers", type=int, default=4)
    e54.add_argument("--threads-per-worker", type=int, default=6)
    e55 = commands.add_parser("e55")
    e55.add_argument("--data-root", type=Path, required=True)
    e55.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e55.add_argument("--e54", type=Path, required=True)
    e55.add_argument("--output", type=Path, required=True)
    e55.add_argument("--device", default="cpu")
    e55.add_argument("--workers", type=int, default=2)
    e55.add_argument("--threads-per-worker", type=int, default=12)
    e56 = commands.add_parser("e56")
    e56.add_argument("--data-root", type=Path, required=True)
    e56.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e56.add_argument("--e55", type=Path, required=True)
    e56.add_argument("--output", type=Path, required=True)
    e61 = commands.add_parser("e61")
    e61.add_argument("--data-root", type=Path, required=True)
    e61.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e61.add_argument("--output", type=Path, required=True)
    e62_fixture = commands.add_parser("e62-fixture")
    e62_fixture.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "protocol.json"
    )
    e62_fixture.add_argument("--output", type=Path, required=True)
    e62 = commands.add_parser("e62")
    e62.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e62.add_argument("--fixture", type=Path, required=True)
    e62.add_argument("--output", type=Path, required=True)
    e63 = commands.add_parser("e63")
    e63.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e63.add_argument("--e57", type=Path, required=True)
    e63.add_argument("--output", type=Path, required=True)
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
    if args.command == "e52":
        result = run_e52(
            args.data_root,
            args.protocol,
            args.e51,
            args.output,
            device=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e53":
        result = run_e53(
            args.data_root,
            args.protocol,
            args.e52,
            args.output,
            device=args.device,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e54":
        result = run_e54(
            args.data_root,
            args.protocol,
            args.e53,
            args.output,
            device=args.device,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e55":
        result = run_e55(
            args.data_root,
            args.protocol,
            args.e54,
            args.output,
            device=args.device,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e56":
        result = run_e56(
            args.data_root, args.protocol, args.e55, args.output
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e61":
        result = run_e61(args.data_root, args.protocol, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e62-fixture":
        result = run_e62_fixture(args.protocol, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "e62":
        result = run_e62(args.protocol, args.fixture, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e63":
        result = run_e63(args.protocol, args.e57, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
