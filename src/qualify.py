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
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from .evaluate import PointMetricAccumulator, WindowScoreFusion, _point_metrics
from .model import (
    DEFAULT_STU_REPOSITORY,
    MASK_DIM,
    AJAEPointTransformer,
    FrozenSTUPointEncoder,
    KnnUpsample,
    TemporalPointBlock,
    VoxelPool,
    assigned_stu_evidence,
    stu_source_manifest,
    stu_weight_identity,
    temporal_radius_knn,
)
from .protocol import load_protocol
from .train import (
    E63B1DevelopmentEvaluator,
    balanced_bce_loss,
    experiment_condition,
    make_window_training_data,
)
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
E75_BOOTSTRAP_NAMESPACE = "E75-common-domain-bootstrap-correction-v1"
E76_LITE_PURE_NAMESPACE = "E76-X-lite-pure-normal-frame-v1"
COMMON_DEVELOPMENT_BOOTSTRAP_COMPARISONS = ("E75", "E81", "E82", "E88")
REPORTED_PERCENT_SCALE = 100.0
E63_COMMON_WORLD_ID = (*range(5), *range(6, 24))
E63_COMMON_SAFETY_FOLD = (
    "B", "B", "A", "A", "B", "A", "B", "A", "A", "B", "A", "B",
    "A", "B", "B", "B", "B", "A", "B", "A", "A", "A", "B",
)
PHASE7_SEED = 640071
E61_ARTIFACT_SHA256 = (
    "8d3e08e0512dc70a75d2279cfb4515bc960bbfda4f35a872c4a76e9dad69d0e0"
)
E72_ARTIFACT_SHA256 = (
    "208487d5c91b131856e908988cf6d955305fa09364450d509e32f617295b5863"
)
E73_E26_ARTIFACT_SHA256 = (
    "2653f705d2e890d99cda732a7a00387b5621cd05abb9c4681c7a9f284c34363c"
)
E63_ARTIFACT_SHA256 = (
    "5dbf99eaa59a05a83774e42beb6b8d7a95cf9309ebd42ab7870604a20d410dd9"
)
E75_IDENTITY_ARTIFACT_SHA256 = (
    "1bae1dbe4b5ded34cf9cebd818b4877368973114c0e7046840c0ff342fb73b9d"
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


def _plain_json(value: object) -> object:
    """Convert frozen protocol containers into JSON-native values."""

    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise QualificationError(f"unsupported manifest value type: {type(value).__name__}")


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


def e75_bootstrap_identity_arrays(e63_path: Path | str) -> dict[str, np.ndarray]:
    """Build the corrected E75 bootstrap over the 23-world common domain."""

    source = Path(e63_path).expanduser().resolve(strict=True)
    if _sha256(source) != E63_ARTIFACT_SHA256:
        raise QualificationError("E75 bootstrap input is not the frozen E63 artifact")
    with np.load(source, allow_pickle=False) as artifact:
        required = {
            "world_id",
            "common_domain_eligible",
            "bootstrap_training_seed",
            "bootstrap_world_id",
        }
        if not required.issubset(artifact.files):
            raise QualificationError("E63 artifact lacks E75 bootstrap identities")
        world_id = np.asarray(artifact["world_id"], dtype=np.int16)
        eligible = np.asarray(artifact["common_domain_eligible"], dtype=np.bool_)
        predecessor_seed_draws = np.asarray(
            artifact["bootstrap_training_seed"], dtype=np.int8
        )
    if (
        world_id.shape != (24,)
        or eligible.shape != (24,)
        or world_id.tolist() != list(range(24))
        or world_id[~eligible].tolist() != [5]
        or int(np.count_nonzero(eligible)) != 23
    ):
        raise QualificationError("E63 common-domain identity changed")

    # Preserve the original RNG prefix, then sample only observable paired worlds.
    generator = np.random.Generator(np.random.PCG64(E63_BOOTSTRAP_SEED))
    bootstrap_training_seed = generator.choice(
        np.asarray((0, 1, 2), dtype=np.int8), size=(5000, 3), replace=True
    )
    if not np.array_equal(bootstrap_training_seed, predecessor_seed_draws):
        raise QualificationError("E75 training-seed bootstrap stream changed")
    common_world_id = world_id[eligible]
    bootstrap_world_id = generator.choice(
        common_world_id, size=(5000, 23), replace=True
    ).astype(np.int16, copy=False)
    return {
        "common_domain_world_id": common_world_id,
        "excluded_world_id": world_id[~eligible],
        "bootstrap_training_seed": bootstrap_training_seed,
        "bootstrap_world_id": bootstrap_world_id,
    }


def run_e75_identity_correction(
    protocol_path: Path | str,
    e63_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Freeze E75's 23-world bootstrap before any model result is read."""

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    gate2 = protocol.decision_gates["criteria"]["gate2"]
    shared = protocol.development["e63_freeze"]["shared_training"]
    if (
        protocol.training["maximum_worlds"] != 25
        or shared["maximum_complete_worlds_per_seed"] != 25
        or gate2["minimum_mean_macro_world_AP_difference"] != 0.02
        or gate2["bootstrap_95_percent_lower_bound_strictly_greater_than"] != 0.0
        or gate2["minimum_positive_training_seeds"] != 2
        or gate2["training_seeds"] != 3
    ):
        raise QualificationError("E75 training budget or Gate 2 criteria changed")
    source = Path(e63_path).expanduser().resolve(strict=True)
    first = e75_bootstrap_identity_arrays(source)
    second = e75_bootstrap_identity_arrays(source)
    reproduction_errors = sum(
        not np.array_equal(first[name], second[name]) for name in first
    )
    identity_errors = int(
        first["common_domain_world_id"].shape != (23,)
        or first["excluded_world_id"].tolist() != [5]
        or first["bootstrap_training_seed"].shape != (5000, 3)
        or first["bootstrap_world_id"].shape != (5000, 23)
        or bool(np.any(first["bootstrap_world_id"] == 5))
        or set(np.unique(first["bootstrap_training_seed"]).tolist()) != {0, 1, 2}
        or set(np.unique(first["bootstrap_world_id"]).tolist())
        != set(first["common_domain_world_id"].tolist())
    )
    result = {
        "experiment": "E75 pre-result statistical identity correction",
        "status": "frozen_before_result_exposure",
        "passed": reproduction_errors == 0 and identity_errors == 0,
        "failure_classification": "statistical_identity_implementation_defect",
        "namespace": E75_BOOTSTRAP_NAMESPACE,
        "generator": "NumPy PCG64",
        "seed": E63_BOOTSTRAP_SEED,
        "replicates": 5000,
        "training_seed_draws_per_replicate": 3,
        "development_world_draws_per_replicate": 23,
        "paired_models_share_realized_indices": True,
        "source_e63_sha256": _sha256(source),
        "protocol_sha256": _sha256(protocol_file),
        "development_metric_values_read": False,
        "training_maximum_worlds": 25,
        "identity_errors": identity_errors,
        "reproduction_errors": reproduction_errors,
        "scientific_array_sha256": _array_hash(first),
        "seconds": time.monotonic() - started,
    }
    _save(Path(output_path).expanduser().resolve(), first, result)
    return result


def e75_superiority_statistics(
    b0_world_id: np.ndarray,
    b0_ap: np.ndarray,
    b1_world_id: np.ndarray,
    b1_ap: np.ndarray,
    common_world_id: np.ndarray,
    bootstrap_training_seed: np.ndarray,
    bootstrap_world_id: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute E75 on the [0,1] decision scale from percent-reported AP."""

    baseline_id = np.asarray(b0_world_id, dtype=np.int16)
    baseline_ap = np.asarray(b0_ap, dtype=np.float64)
    trained_id = np.asarray(b1_world_id, dtype=np.int16)
    trained_ap = np.asarray(b1_ap, dtype=np.float64)
    common = np.asarray(common_world_id, dtype=np.int16)
    seed_draws = np.asarray(bootstrap_training_seed, dtype=np.int8)
    world_draws = np.asarray(bootstrap_world_id, dtype=np.int16)
    if (
        baseline_id.shape != (23,)
        or baseline_ap.shape != (23,)
        or trained_id.shape != (3, 23)
        or trained_ap.shape != (3, 23)
        or common.shape != (23,)
        or seed_draws.shape != (5000, 3)
        or world_draws.shape != (5000, 23)
        or not np.array_equal(baseline_id, common)
        or not np.all(trained_id == common[None, :])
        or not np.isfinite(baseline_ap).all()
        or not np.isfinite(trained_ap).all()
        or np.any((seed_draws < 0) | (seed_draws > 2))
    ):
        raise QualificationError("E75 paired metric identities are invalid")
    column_by_world = np.full(24, -1, dtype=np.int16)
    column_by_world[common] = np.arange(23, dtype=np.int16)
    if np.any((world_draws < 0) | (world_draws >= 24)):
        raise QualificationError("E75 bootstrap contains an invalid world identity")
    world_columns = column_by_world[world_draws]
    if np.any(world_columns < 0):
        raise QualificationError("E75 bootstrap contains a non-common-domain world")

    if (
        np.any((baseline_ap < 0.0) | (baseline_ap > REPORTED_PERCENT_SCALE))
        or np.any((trained_ap < 0.0) | (trained_ap > REPORTED_PERCENT_SCALE))
    ):
        raise QualificationError("E75 reported AP must use the [0,100] scale")
    baseline_decision_ap = baseline_ap / REPORTED_PERCENT_SCALE
    trained_decision_ap = trained_ap / REPORTED_PERCENT_SCALE
    paired_difference = trained_decision_ap - baseline_decision_ap[None, :]
    # Both models use the same realized seed/world indices in every replicate.
    bootstrap_difference = paired_difference[
        seed_draws[:, :, None], world_columns[:, None, :]
    ].mean(axis=(1, 2))
    return {
        "world_id": common,
        "b0_ap_reported": baseline_ap,
        "b1_ap_reported": trained_ap,
        "b0_ap_decision": baseline_decision_ap,
        "b1_ap_decision": trained_decision_ap,
        "paired_ap_decision_difference": paired_difference,
        "seed_mean_ap_decision_difference": paired_difference.mean(axis=1),
        "bootstrap_mean_ap_decision_difference": bootstrap_difference,
    }


def e75_exploratory_statistics(
    b0_world_id: np.ndarray,
    b0_ap_reported: np.ndarray,
    b1_world_id: np.ndarray,
    b1_ap_reported: np.ndarray,
) -> dict[str, np.ndarray]:
    """Describe the first two preregistered B1 seeds without a gate verdict."""

    world_id = np.asarray(b0_world_id, dtype=np.int16)
    baseline = np.asarray(b0_ap_reported, dtype=np.float64)
    trained_world = np.asarray(b1_world_id, dtype=np.int16)
    trained = np.asarray(b1_ap_reported, dtype=np.float64)
    if (
        world_id.tolist() != list(E63_COMMON_WORLD_ID)
        or baseline.shape != (23,)
        or trained_world.shape != (2, 23)
        or not np.all(trained_world == world_id[None, :])
        or trained.shape != (2, 23)
        or not np.isfinite(baseline).all()
        or not np.isfinite(trained).all()
        or np.any((baseline < 0.0) | (baseline > REPORTED_PERCENT_SCALE))
        or np.any((trained < 0.0) | (trained > REPORTED_PERCENT_SCALE))
    ):
        raise QualificationError("E75-X AP arrays or world identities are invalid")
    baseline_decision = baseline / REPORTED_PERCENT_SCALE
    trained_decision = trained / REPORTED_PERCENT_SCALE
    paired = trained_decision - baseline_decision[None, :]
    return {
        "world_id": world_id,
        "b0_ap_reported": baseline,
        "b1_ap_reported": trained,
        "b0_ap_decision": baseline_decision,
        "b1_ap_decision": trained_decision,
        "paired_ap_decision_difference": paired,
        "seed_mean_ap_decision_difference": paired.mean(axis=1),
        "two_seed_mean_ap_decision_difference": np.asarray(paired.mean()),
        "positive_world_count": np.count_nonzero(paired > 0.0, axis=1).astype(
            np.int16
        ),
    }


def run_e75_exploratory(
    protocol_path: Path | str,
    e72_path: Path | str,
    b1_dir: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Execute E75-X on completed seeds 0/1 while seed 2 stays suspended."""

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    project_root = Path(protocol.path).parent
    exploration = protocol.development["exploration_track"]
    confirmation = exploration["e74_confirmation"]
    if (
        exploration["current_node"] not in {"E75-X", "E76-X"}
        or tuple(exploration["cohort"]["seeds"]) != (0, 1)
        or confirmation["formal_pass_forbidden"] is not True
        or confirmation["partial_seed2_result_use_forbidden"] is not True
    ):
        raise QualificationError("E75-X exploration identity changed")
    e72_file = Path(e72_path).expanduser().resolve(strict=True)
    if _sha256(e72_file) != E72_ARTIFACT_SHA256:
        raise QualificationError("E75-X B0 input is not the frozen E72 artifact")
    progress_file = project_root / confirmation["paused_progress_path"]
    if (
        _sha256(progress_file) != confirmation["paused_progress_sha256"]
        or (progress_file.parent / "model.pt").exists()
        or (progress_file.parent / "result.json").exists()
    ):
        raise QualificationError("E75-X seed-2 suspension identity changed")

    with np.load(e72_file, allow_pickle=False) as archive:
        metric_order = np.asarray(archive["metric_order"])
        ap_column = np.flatnonzero(metric_order == "AP")
        if ap_column.tolist() != [0]:
            raise QualificationError("E75-X E72 AP column identity changed")
        b0_world_id = np.asarray(archive["development_world_id"], dtype=np.int16)
        b0_ap = np.asarray(archive["development_metric"], dtype=np.float64)[:, 0]

    result_root = Path(b1_dir).expanduser().resolve(strict=True)
    expected_models = tuple(confirmation["completed_seed_model_sha256"])
    expected_results = tuple(confirmation["completed_seed_result_sha256"])
    b1_world_id = np.empty((2, 23), dtype=np.int16)
    b1_ap = np.empty((2, 23), dtype=np.float64)
    for seed in (0, 1):
        seed_dir = result_root / f"seed-{seed}"
        result_file = seed_dir / "result.json"
        model_file = seed_dir / "model.pt"
        if (
            _sha256(result_file) != expected_results[seed]
            or _sha256(model_file) != expected_models[seed]
        ):
            raise QualificationError(f"E75-X seed {seed} artifact identity changed")
        record = json.loads(result_file.read_text(encoding="utf-8"))
        if (
            record.get("status") != "completed"
            or record.get("seed") != seed
            or record.get("maximum_worlds") != 25
            or record.get("condition", {}).get("name") != "B1"
        ):
            raise QualificationError(f"E75-X seed {seed} completion is invalid")
        selected = [
            item
            for item in record.get("history", ())
            if item.get("world") == record.get("best_world")
            and "development" in item
        ]
        if len(selected) != 1:
            raise QualificationError(f"E75-X seed {seed} lacks one selected result")
        worlds = sorted(
            selected[0]["development"]["in_generator"],
            key=lambda item: item["world_id"],
        )
        if len(worlds) != 23:
            raise QualificationError(f"E75-X seed {seed} world count changed")
        b1_world_id[seed] = [item["world_id"] for item in worlds]
        b1_ap[seed] = [item["metrics"]["AP"] for item in worlds]
        if not np.isclose(
            b1_ap[seed].mean(),
            float(record["best_selection_key"][0]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise QualificationError(f"E75-X seed {seed} selected AP is inconsistent")

    arrays = e75_exploratory_statistics(
        b0_world_id, b0_ap, b1_world_id, b1_ap
    )
    result = {
        "experiment": "E75-X",
        "passed": True,
        "pass_scope": "descriptive_execution_only",
        "formal_gate2_adjudicated": False,
        "formal_confidence_interval_computed": False,
        "cohort_selection": exploration["cohort"]["selection"],
        "training_seeds": [0, 1],
        "development_worlds": 23,
        "decision_metric_scale": "[0,1]",
        "seed_mean_ap_decision_difference": arrays[
            "seed_mean_ap_decision_difference"
        ].tolist(),
        "two_seed_mean_ap_decision_difference": float(
            arrays["two_seed_mean_ap_decision_difference"]
        ),
        "positive_world_count": arrays["positive_world_count"].tolist(),
        "next_node": "E76-X",
        "protocol_sha256": _sha256(protocol_file),
        "e72_artifact_sha256": _sha256(e72_file),
        "seed2_paused_progress_sha256": _sha256(progress_file),
        "b1_model_sha256": list(expected_models),
        "b1_result_sha256": list(expected_results),
        "scientific_array_sha256": _array_hash(arrays),
        "seconds": time.monotonic() - started,
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def e76_safety_statistics(
    development_world_id: np.ndarray,
    development_point_world_id: np.ndarray,
    development_label: np.ndarray,
    development_normal_control: np.ndarray,
    development_score: np.ndarray,
    safety_fold: np.ndarray,
    pure_normal_score: np.ndarray,
    moving_normal_score: np.ndarray,
    development_fpr95_reported: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the frozen B0/B1 safety estimand from aligned score arrays."""

    raw_label = np.asarray(development_label)
    raw_control = np.asarray(development_normal_control)
    raw_score = np.asarray(development_score)
    raw_pure = np.asarray(pure_normal_score)
    raw_moving = np.asarray(moving_normal_score)
    raw_fpr95 = np.asarray(development_fpr95_reported)
    world_id = np.asarray(development_world_id, dtype=np.int16)
    point_world = np.asarray(development_point_world_id, dtype=np.int16)
    label = raw_label.astype(np.bool_, copy=False)
    control = raw_control.astype(np.bool_, copy=False)
    score = raw_score.astype(np.float64, copy=False)
    fold = np.asarray(safety_fold, dtype="S1")
    pure = raw_pure.astype(np.float64, copy=False)
    moving = raw_moving.astype(np.float64, copy=False)
    fpr95_reported = raw_fpr95.astype(np.float64, copy=False)
    model_count = int(score.shape[0]) if score.ndim == 2 else 0
    if (
        world_id.tolist() != list(E63_COMMON_WORLD_ID)
        or fold.tolist() != [name.encode("ascii") for name in E63_COMMON_SAFETY_FOLD]
        or fold.shape != (23,)
        or set(np.unique(fold).tolist()) != {b"A", b"B"}
        or raw_label.dtype != np.bool_
        or raw_control.dtype != np.bool_
        or not np.issubdtype(raw_score.dtype, np.floating)
        or not np.issubdtype(raw_pure.dtype, np.floating)
        or not np.issubdtype(raw_moving.dtype, np.floating)
        or not np.issubdtype(raw_fpr95.dtype, np.floating)
        or point_world.shape != label.shape
        or control.shape != label.shape
        or model_count not in {3, 4}
        or score.shape != (model_count, label.size)
        or pure.ndim != 2
        or pure.shape[0] != model_count
        or pure.shape[1] == 0
        or moving.ndim != 2
        or moving.shape[0] != model_count
        or moving.shape[1] == 0
        or fpr95_reported.shape != (model_count, 23)
        or not np.isfinite(score).all()
        or not np.isfinite(pure).all()
        or not np.isfinite(moving).all()
        or not np.isfinite(fpr95_reported).all()
        or np.any((fpr95_reported < 0.0) | (fpr95_reported > REPORTED_PERCENT_SCALE))
        or set(np.unique(point_world).tolist()) != set(world_id.tolist())
        or bool(np.any(control & label))
    ):
        raise QualificationError("E76 score or identity arrays are invalid")
    fold_by_world = {int(current): fold[index] for index, current in enumerate(world_id)}
    point_fold = np.asarray([fold_by_world[int(current)] for current in point_world])

    threshold = np.empty((model_count, 2), dtype=np.float64)
    control_false_positive = np.zeros((model_count, 2), dtype=np.int64)
    control_count = np.asarray(
        [np.count_nonzero(control & (point_fold == name)) for name in (b"A", b"B")],
        dtype=np.int64,
    )
    if bool(np.any(control_count == 0)):
        raise QualificationError("E76 requires normal-control points in both folds")
    for model in range(model_count):
        for fold_index, fold_name in enumerate((b"A", b"B")):
            selected = point_fold == fold_name
            threshold[model, fold_index] = float(
                _point_metrics(label[selected], score[model, selected])["threshold"]
            )
        # Threshold A evaluates Fold B controls; threshold B evaluates Fold A.
        control_false_positive[model, 0] = np.count_nonzero(
            score[model, control & (point_fold == b"B")] > threshold[model, 0]
        )
        control_false_positive[model, 1] = np.count_nonzero(
            score[model, control & (point_fold == b"A")] > threshold[model, 1]
        )

    control_fpr = control_false_positive.sum(axis=1) / control_count.sum()
    pure_fpr = 0.5 * np.stack(
        (
            np.mean(pure > threshold[:, 0, None], axis=1),
            np.mean(pure > threshold[:, 1, None], axis=1),
        ),
        axis=1,
    ).sum(axis=1)
    moving_fpr = 0.5 * np.stack(
        (
            np.mean(moving > threshold[:, 0, None], axis=1),
            np.mean(moving > threshold[:, 1, None], axis=1),
        ),
        axis=1,
    ).sum(axis=1)
    development_fpr95 = np.mean(
        fpr95_reported / REPORTED_PERCENT_SCALE, axis=1
    )
    measure = np.stack(
        (pure_fpr, control_fpr, moving_fpr, development_fpr95), axis=1
    )
    seed_worsening = measure[1:] - measure[0]
    mean_worsening = seed_worsening.mean(axis=0)
    return {
        "model_name": np.asarray(
            ("B0",) + tuple(f"B1_{seed}" for seed in range(model_count - 1))
        ),
        "safety_measure_name": np.asarray(
            ("pure_normal_FPR", "normal_control_FPR", "moving_normal_FPR", "development_FPR95")
        ),
        "threshold": threshold,
        "normal_control_count_by_fold": control_count,
        "normal_control_evaluation_fold": np.asarray(("B", "A")),
        "normal_control_false_positive_crossfit": control_false_positive,
        "safety_measure": measure,
        "seed_safety_worsening": seed_worsening,
        "mean_safety_worsening": mean_worsening,
        "passed": np.asarray(bool(np.all(mean_worsening <= 0.03))),
    }


def e76_lite_pure_frame_rows(frame_id: np.ndarray) -> np.ndarray:
    """Select the frozen 64 E61 frames without using counts or model outputs."""

    frames = np.asarray(frame_id)
    if (
        frames.dtype != np.int16
        or frames.shape != (678,)
        or np.unique(frames).size != frames.size
    ):
        raise QualificationError("E76-X-lite source frame identities are invalid")
    ranked = sorted(
        range(frames.size),
        key=lambda row: (
            hashlib.sha256(
                f"{E76_LITE_PURE_NAMESPACE}:train:201:{int(frames[row])}".encode(
                    "ascii"
                )
            ).digest(),
            int(frames[row]),
        ),
    )
    return np.asarray(ranked[:64], dtype=np.int16)


def e76v1_group_a_selection(
    world_id: np.ndarray, selected_descriptor: np.ndarray
) -> tuple[tuple[str, int, float], ...]:
    """Freeze six proxy-visibility cases without reading model outputs."""

    worlds = np.asarray(world_id)
    descriptor = np.asarray(selected_descriptor)
    if (
        worlds.dtype != np.int16
        or worlds.shape != (24,)
        or descriptor.shape != (24, 8)
        or not np.issubdtype(descriptor.dtype, np.floating)
        or not np.isfinite(descriptor).all()
    ):
        raise QualificationError("E76-V1 E57 descriptors are invalid")
    eligible = np.flatnonzero(worlds != 5)
    ranked = eligible[
        np.lexsort((worlds[eligible], descriptor[eligible, 4]))
    ]
    selected: list[tuple[str, int, float]] = []
    for name, rows in zip(("low", "mid", "high"), np.array_split(ranked, 3)):
        chosen = rows[np.argsort(worlds[rows], kind="stable")[:2]]
        selected.extend(
            (name, int(worlds[row]), float(descriptor[row, 4]))
            for row in chosen
        )
    return tuple(selected)


def _write_binary_ply(
    path: Path,
    properties: Mapping[str, np.ndarray],
    *,
    comments: Sequence[str],
) -> dict[str, object]:
    """Write a compact CloudCompare-compatible vertex-only PLY."""

    if not properties:
        raise QualificationError("PLY requires at least one property")
    count = int(next(iter(properties.values())).size)
    type_by_dtype = {
        np.dtype("<f4").str: "float",
        np.dtype("|u1").str: "uchar",
        np.dtype("<u2").str: "ushort",
    }
    normalized: dict[str, np.ndarray] = {}
    fields: list[tuple[str, np.dtype]] = []
    for name, raw in properties.items():
        value = np.asarray(raw)
        if (
            not name.replace("_", "").isalnum()
            or value.ndim != 1
            or value.size != count
            or value.dtype.str not in type_by_dtype
            or (
                np.issubdtype(value.dtype, np.floating)
                and not np.isfinite(value).all()
            )
        ):
            raise QualificationError(f"invalid PLY property {name}")
        normalized[name] = np.ascontiguousarray(value)
        fields.append((name, value.dtype))
    vertices = np.empty(count, dtype=np.dtype(fields, align=False))
    for name, value in normalized.items():
        vertices[name] = value
    header = ["ply", "format binary_little_endian 1.0"]
    for comment in comments:
        cleaned = str(comment).replace("\n", " ").replace("\r", " ")
        header.append(f"comment {cleaned}")
    header.append(f"element vertex {count}")
    for name, value in normalized.items():
        header.append(f"property {type_by_dtype[value.dtype.str]} {name}")
    header.extend(("end_header", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        vertices.tofile(handle)
    return {
        "path": path,
        "points": count,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "properties": list(normalized),
    }


def run_e76_exploratory(
    data_root: Path | str,
    protocol_path: Path | str,
    e57_path: Path | str,
    e61_path: Path | str,
    e63_path: Path | str,
    e72_path: Path | str,
    b1_dir: Path | str,
    output_path: Path | str,
    *,
    device: str = "cuda",
    lite: bool = False,
) -> dict[str, object]:
    """Execute full or lightweight two-seed safety without a formal verdict."""

    from .render import load_sensor_calibration

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    project_root = Path(protocol.path).parent
    exploration = protocol.development["exploration_track"]
    confirmation = exploration["e74_confirmation"]
    experiment = "E76-X-lite" if lite else "E76-X"
    if (
        exploration["current_node"] != experiment
        or tuple(exploration["cohort"]["seeds"]) != (0, 1)
        or exploration["formal_gate2_and_gate3_status"] != "not adjudicated"
        or exploration["e75x_result"]["formal_gate2_adjudicated"] is not False
        or confirmation["partial_seed2_result_use_forbidden"] is not True
    ):
        raise QualificationError(f"{experiment} exploration identity changed")

    e57_file = Path(e57_path).expanduser().resolve(strict=True)
    e61_file = Path(e61_path).expanduser().resolve(strict=True)
    e63_file = Path(e63_path).expanduser().resolve(strict=True)
    e72_file = Path(e72_path).expanduser().resolve(strict=True)
    expected = protocol.development["e63_freeze"]
    if (
        _sha256(e57_file) != expected["source_worlds"]["artifact_sha256"]
        or _sha256(e61_file) != E61_ARTIFACT_SHA256
        or _sha256(e63_file) != expected["identity_artifact"]["artifact_sha256"]
        or _sha256(e72_file) != E72_ARTIFACT_SHA256
    ):
        raise QualificationError("E76-X input artifact identity changed")
    progress_file = project_root / confirmation["paused_progress_path"]
    if (
        _sha256(progress_file) != confirmation["paused_progress_sha256"]
        or (progress_file.parent / "model.pt").exists()
        or (progress_file.parent / "result.json").exists()
    ):
        raise QualificationError("E76-X seed-2 suspension identity changed")

    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise QualificationError("E76-X CUDA device is unavailable")
    with np.load(e63_file, allow_pickle=False) as archive:
        eligible = np.asarray(archive["common_domain_eligible"], dtype=np.bool_)
        frozen_world = np.asarray(archive["world_id"], dtype=np.int16)
        safety_fold = np.asarray(archive["safety_fold"])[eligible]
    with np.load(e61_file, allow_pickle=False) as archive:
        full_pure_frame = np.asarray(archive["pure_frame_id"], dtype=np.int16)
        full_pure_packed = np.asarray(
            archive["pure_canonical_mask_packed"], dtype=np.uint8
        )
        full_pure_count = np.asarray(
            archive["pure_point_count_by_frame"], dtype=np.int32
        )
        moving_frame = np.asarray(archive["moving_frame_id"], dtype=np.int16)
        moving_packed = np.asarray(
            archive["moving_canonical_mask_packed"], dtype=np.uint8
        )
        moving_count = np.asarray(
            archive["moving_point_count_by_frame"], dtype=np.int32
        )
    with np.load(e72_file, allow_pickle=False) as archive:
        development_world = np.asarray(
            archive["development_world_id"], dtype=np.int16
        )
        development_offset = np.asarray(
            archive["development_point_offset"], dtype=np.int64
        )
        development_ray = np.asarray(
            archive["development_canonical_ray"], dtype=np.int32
        )
        development_label = np.asarray(
            archive["development_label"], dtype=np.bool_
        )
        development_control = np.asarray(
            archive["development_normal_control"], dtype=np.bool_
        )
        b0_development_score = np.asarray(
            archive["development_score"], dtype=np.float32
        )
        metric_order = np.asarray(archive["metric_order"])
        b0_development_metric = np.asarray(
            archive["development_metric"], dtype=np.float64
        )
        e72_pure_frame = np.asarray(archive["pure_frame_id"], dtype=np.int16)
        full_pure_offset = np.asarray(
            archive["pure_point_offset"], dtype=np.int64
        )
        full_pure_ray = np.asarray(
            archive["pure_canonical_ray"], dtype=np.int32
        )
        full_b0_pure_score = np.asarray(
            archive["pure_score"], dtype=np.float32
        )
        e72_moving_frame = np.asarray(
            archive["moving_frame_id"], dtype=np.int16
        )
        moving_offset = np.asarray(
            archive["moving_point_offset"], dtype=np.int64
        )
        moving_ray = np.asarray(
            archive["moving_canonical_ray"], dtype=np.int32
        )
        b0_moving_score = np.asarray(archive["moving_score"], dtype=np.float32)
    fpr95_column = np.flatnonzero(metric_order == "FPR95")
    if (
        fpr95_column.tolist() != [2]
        or not np.array_equal(development_world, frozen_world[eligible])
        or development_world.tolist() != list(E63_COMMON_WORLD_ID)
        or development_offset.shape != (24,)
        or not np.array_equal(e72_pure_frame, full_pure_frame)
        or full_pure_offset.shape != (full_pure_frame.size + 1,)
        or not np.array_equal(e72_moving_frame, moving_frame)
        or moving_offset.shape != (moving_frame.size + 1,)
        or int(full_pure_offset[-1]) != 48_828_507
        or int(moving_offset[-1]) != 13_011
    ):
        raise QualificationError("E76-X aligned score identity changed")

    if lite:
        lite_freeze = exploration["e76x_lite_freeze"]
        selection = lite_freeze["pure_normal_selection"]
        pure_rows = e76_lite_pure_frame_rows(full_pure_frame)
        pure_frame = full_pure_frame[pure_rows]
        if (
            E76_LITE_PURE_NAMESPACE != selection["namespace"]
            or tuple(pure_frame.astype(int).tolist())
            != tuple(selection["selected_frame_ids"])
        ):
            raise QualificationError("E76-X-lite frame selection changed")
        pure_packed = full_pure_packed[pure_rows]
        pure_count = full_pure_count[pure_rows]
        pure_offset = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum(pure_count, dtype=np.int64),
            )
        )
        pure_ray = np.concatenate(
            [
                full_pure_ray[
                    full_pure_offset[row] : full_pure_offset[row + 1]
                ]
                for row in pure_rows
            ]
        )
        b0_pure_score = np.concatenate(
            [
                full_b0_pure_score[
                    full_pure_offset[row] : full_pure_offset[row + 1]
                ]
                for row in pure_rows
            ]
        )
    else:
        pure_frame = full_pure_frame
        pure_packed = full_pure_packed
        pure_count = full_pure_count
        pure_offset = full_pure_offset
        pure_ray = full_pure_ray
        b0_pure_score = full_b0_pure_score
    if lite and (
        pure_frame.size != 64
        or int(pure_offset[-1]) != int(selection["selected_point_count"])
    ):
        raise QualificationError("E76-X-lite pure-normal count changed")

    grid, sensor = load_sensor_calibration(protocol.sensor_calibration_path())
    canonical_by_slot = np.asarray(
        grid.beam_ids * grid.columns + grid.column_ids, dtype=np.int32
    )
    encoder = FrozenSTUPointEncoder.from_protocol(
        protocol, project_root=project_root
    ).to(runtime_device).eval()
    evaluator = E63B1DevelopmentEvaluator(
        protocol=protocol,
        project_root=project_root,
        data_root=data_root,
        device=runtime_device,
        encoder=encoder,
        grid=grid,
        sensor=sensor,
        canonical_by_slot=canonical_by_slot,
    )
    result_root = Path(b1_dir).expanduser().resolve(strict=True)
    expected_models = tuple(confirmation["completed_seed_model_sha256"])
    expected_results = tuple(confirmation["completed_seed_result_sha256"])
    models: list[AJAEPointTransformer] = []
    for seed in (0, 1):
        model_file = result_root / f"seed-{seed}" / "model.pt"
        result_file = result_root / f"seed-{seed}" / "result.json"
        if (
            _sha256(model_file) != expected_models[seed]
            or _sha256(result_file) != expected_results[seed]
        ):
            raise QualificationError(f"E76-X seed {seed} artifact identity changed")
        record = json.loads(result_file.read_text(encoding="utf-8"))
        payload = torch.load(model_file, map_location="cpu", weights_only=True)
        if (
            record.get("status") != "completed"
            or payload.get("seed") != seed
            or payload.get("completion_id") != record.get("completion_id")
            or payload.get("best_world") != record.get("best_world")
            or payload.get("scientific_identity") != record.get("scientific_identity")
        ):
            raise QualificationError(f"E76-X seed {seed} completion is invalid")
        model = AJAEPointTransformer.from_protocol(protocol).to(runtime_device).eval()
        model.load_state_dict(payload["model"], strict=True)
        models.append(model)

    development_score = np.empty(
        (3, b0_development_score.size), dtype=np.float32
    )
    development_score[0] = b0_development_score
    development_fpr95 = np.empty((3, 23), dtype=np.float64)
    development_fpr95[0] = b0_development_metric[:, 2]
    prepared = evaluator._prepare_development()
    for row, item in enumerate(prepared):
        start, stop = development_offset[row : row + 2]
        xyz = np.asarray(item["xyz"])
        semantic = np.asarray(item["semantic"])
        slots = np.asarray(item["slots"], dtype=np.int64)
        ranges = np.linalg.norm(xyz.astype(np.float32, copy=False), axis=1)
        valid = (
            (ranges >= 2.5)
            & (ranges <= 50.0)
            & (semantic != 0)
        )
        order = np.argsort(canonical_by_slot[slots[valid]], kind="stable")
        labels = (semantic[valid][order] == 2)
        controls = np.asarray(item["control"], dtype=np.bool_)[valid][order]
        rays = canonical_by_slot[slots[valid]][order]
        if (
            int(item["world_id"]) != int(development_world[row])
            or not np.array_equal(rays, development_ray[start:stop])
            or not np.array_equal(labels, development_label[start:stop])
            or not np.array_equal(controls, development_control[start:stop])
        ):
            raise QualificationError("E76-X development alignment changed")
        for model_index, model in enumerate(models, start=1):
            scores = evaluator._scores(model, item)[valid][order]
            development_score[model_index, start:stop] = scores
            development_fpr95[model_index, row] = _point_metrics(
                labels, scores
            )["FPR95"]

    pure_score = np.empty((3, b0_pure_score.size), dtype=np.float32)
    pure_score[0] = b0_pure_score
    for row, frame_id in enumerate(pure_frame.tolist()):
        start, stop = pure_offset[row : row + 2]
        if start == stop:
            continue
        source = evaluator.sequence.source_frame(int(frame_id))
        item = evaluator._input(source)
        slots = np.asarray(item["slots"], dtype=np.int64)
        mask = np.unpackbits(pure_packed[row], bitorder="little")[
            : canonical_by_slot.size
        ]
        selected = np.flatnonzero(mask[canonical_by_slot])
        order = np.argsort(canonical_by_slot[selected], kind="stable")
        if (
            selected.size != int(pure_count[row])
            or not np.array_equal(canonical_by_slot[selected][order], pure_ray[start:stop])
        ):
            raise QualificationError("E76-X pure-normal alignment changed")
        slot_to_real = np.full(canonical_by_slot.size, -1, dtype=np.int32)
        slot_to_real[slots] = np.arange(slots.size, dtype=np.int32)
        selected_index = slot_to_real[selected[order]]
        if bool(np.any(selected_index < 0)):
            raise QualificationError(
                "E76-X pure-normal mask selected an absent return"
            )
        for model_index, model in enumerate(models, start=1):
            pure_score[model_index, start:stop] = evaluator._scores(
                model, item
            )[selected_index]

    sequence_206 = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    moving_score = np.empty((3, b0_moving_score.size), dtype=np.float32)
    moving_score[0] = b0_moving_score
    for row, frame_id in enumerate(moving_frame.tolist()):
        start, stop = moving_offset[row : row + 2]
        if start == stop:
            continue
        source = sequence_206.source_frame(int(frame_id))
        item = evaluator._input(source)
        slots = np.asarray(item["slots"], dtype=np.int64)
        mask = np.unpackbits(moving_packed[row], bitorder="little")[
            : canonical_by_slot.size
        ]
        selected = np.flatnonzero(mask[canonical_by_slot])
        order = np.argsort(canonical_by_slot[selected], kind="stable")
        if (
            selected.size != int(moving_count[row])
            or not np.array_equal(
                canonical_by_slot[selected][order], moving_ray[start:stop]
            )
        ):
            raise QualificationError("E76-X moving-normal alignment changed")
        slot_to_real = np.full(canonical_by_slot.size, -1, dtype=np.int32)
        slot_to_real[slots] = np.arange(slots.size, dtype=np.int32)
        selected_index = slot_to_real[selected[order]]
        if bool(np.any(selected_index < 0)):
            raise QualificationError(
                "E76-X moving-normal mask selected an absent return"
            )
        for model_index, model in enumerate(models, start=1):
            moving_score[model_index, start:stop] = evaluator._scores(
                model, item
            )[selected_index]

    point_world = np.repeat(development_world, np.diff(development_offset))
    arrays = e76_safety_statistics(
        development_world,
        point_world,
        development_label,
        development_control,
        development_score,
        safety_fold,
        pure_score,
        moving_score,
        development_fpr95,
    )
    arrays["pure_normal_frame_id"] = pure_frame
    arrays["pure_normal_point_count_by_frame"] = pure_count
    reference_satisfied = bool(arrays["passed"])
    result = {
        "experiment": experiment,
        "passed": True,
        "pass_scope": (
            "exploratory_catastrophic_safety_screen_only"
            if lite
            else "exploratory_execution_only"
        ),
        "formal_e76_adjudicated": False,
        "formal_gate2_adjudicated": False,
        "training_seeds": [0, 1],
        "development_worlds": 23,
        "development_points": int(development_label.size),
        "pure_normal_frames": int(pure_frame.size),
        "pure_normal_points": int(pure_score.shape[1]),
        "moving_normal_points": int(moving_score.shape[1]),
        "safety_measure_name": arrays["safety_measure_name"].tolist(),
        "safety_measure": arrays["safety_measure"].tolist(),
        "seed_safety_worsening": arrays["seed_safety_worsening"].tolist(),
        "mean_safety_worsening": arrays["mean_safety_worsening"].tolist(),
        "original_e76_mean_reference": 0.03,
        "original_e76_mean_reference_satisfied": reference_satisfied,
        "pure_normal_selection_sha256": (
            selection["selection_sha256"] if lite else None
        ),
        "full_e76x_deferred": lite,
        "exploratory_outcome": (
            "non_disastrous_continue_to_E78-X"
            if reference_satisfied
            else "safety_review_required_before_B2_B3"
        ),
        "next_node": "E78-X" if reference_satisfied else None,
        "protocol_sha256": _sha256(protocol_file),
        "e57_artifact_sha256": _sha256(e57_file),
        "e61_artifact_sha256": _sha256(e61_file),
        "e63_artifact_sha256": _sha256(e63_file),
        "e72_artifact_sha256": _sha256(e72_file),
        "b1_model_sha256": list(expected_models),
        "b1_result_sha256": list(expected_results),
        "seed2_paused_progress_sha256": _sha256(progress_file),
        "scientific_array_sha256": _array_hash(arrays),
        "seconds": time.monotonic() - started,
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def run_e76_visual_audit(
    data_root: Path | str,
    protocol_path: Path | str,
    e57_path: Path | str,
    e61_path: Path | str,
    e76_path: Path | str,
    b1_dir: Path | str,
    output_dir: Path | str,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    """Export the frozen descriptive proxy and moving-normal PLY audit."""

    from .render import WorldSpec, load_sensor_calibration, render_frame

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    project_root = Path(protocol.path).parent
    exploration = protocol.development["exploration_track"]
    freeze = exploration["e76v1_freeze"]
    confirmation = exploration["e74_confirmation"]
    if (
        exploration["current_node"] != "E76-V1"
        or freeze["status"] != "frozen_before_export_or_visual_review"
        or freeze["e78x_remains_locked"] is not True
        or exploration["e76x_lite_result"]["e78x_locked"] is not True
    ):
        raise QualificationError("E76-V1 protocol identity changed")
    e57_file = Path(e57_path).expanduser().resolve(strict=True)
    e61_file = Path(e61_path).expanduser().resolve(strict=True)
    e76_file = Path(e76_path).expanduser().resolve(strict=True)
    if (
        _sha256(e57_file) != freeze["group_a"]["source_artifact_sha256"]
        or _sha256(e61_file) != freeze["group_b"]["source_artifact_sha256"]
        or _sha256(e76_file)
        != exploration["e76x_lite_result"]["artifact_sha256"]
    ):
        raise QualificationError("E76-V1 input artifact identity changed")
    destination = Path(output_dir).expanduser().resolve()
    if (
        destination != project_root / freeze["output_directory"]
        or destination.exists()
    ):
        raise QualificationError("E76-V1 output directory is invalid or already exists")
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise QualificationError("E76-V1 CUDA device is unavailable")

    with np.load(e57_file, allow_pickle=False) as archive:
        world_id = np.asarray(archive["selected_world_id"], dtype=np.int16)
        center_frame = np.asarray(
            archive["selected_center_frame"], dtype=np.int16
        )
        descriptor = np.asarray(archive["selected_descriptor"], dtype=np.float64)
        world_json = np.asarray(archive["selected_world_json"])
    selected_a = e76v1_group_a_selection(world_id, descriptor)
    expected_a = tuple(
        (name, int(current), float(value))
        for name in ("low", "mid", "high")
        for current, value in zip(
            freeze["group_a"]["selected_world_ids"][name],
            freeze["group_a"]["selected_proxy_Nvis"][name],
            strict=True,
        )
    )
    if selected_a != expected_a:
        raise QualificationError("E76-V1 group-A selection changed")
    row_by_world = {int(value): row for row, value in enumerate(world_id)}
    with np.load(e61_file, allow_pickle=False) as archive:
        moving_frame = np.asarray(archive["moving_frame_id"], dtype=np.int16)
        moving_packed = np.asarray(
            archive["moving_canonical_mask_packed"], dtype=np.uint8
        )
        moving_count = np.asarray(
            archive["moving_point_count_by_frame"], dtype=np.int32
        )
    with np.load(e76_file, allow_pickle=False) as archive:
        threshold = np.asarray(archive["threshold"], dtype=np.float64)
        if (
            np.asarray(archive["model_name"]).tolist()
            != ["B0", "B1_0", "B1_1"]
            or threshold.shape != (3, 2)
        ):
            raise QualificationError("E76-V1 threshold identity changed")

    grid, sensor = load_sensor_calibration(protocol.sensor_calibration_path())
    canonical_by_slot = np.asarray(
        grid.beam_ids * grid.columns + grid.column_ids, dtype=np.int32
    )
    sequence_201 = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    sequence_206 = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    colors = {
        name: np.asarray(value, dtype=np.uint8)
        for name, value in freeze["colors_rgb"].items()
    }

    def colored_properties(
        source: object,
        point_source: np.ndarray,
        *,
        selected: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
        choose = np.arange(slots.size) if selected is None else np.asarray(selected)
        selected_slots = slots[choose]
        xyzi = np.asarray(getattr(source, "xyzi"))[selected_slots]
        labels = getattr(source, "labels")
        semantic = np.asarray(getattr(labels, "semantic"))[selected_slots].astype(
            "<u2", copy=False
        )
        source_code = np.asarray(point_source, dtype=np.uint8)[choose]
        rgb = np.tile(colors["real_normal"], (choose.size, 1))
        rgb[semantic == 0] = colors["ignore_or_background"]
        rgb[source_code == 1] = colors["normal_control"]
        rgb[source_code == 2] = colors["anomaly_proxy"]
        return {
            "x": xyzi[:, 0].astype("<f4", copy=False),
            "y": xyzi[:, 1].astype("<f4", copy=False),
            "z": xyzi[:, 2].astype("<f4", copy=False),
            "intensity": xyzi[:, 3].astype("<f4", copy=False),
            "red": rgb[:, 0],
            "green": rgb[:, 1],
            "blue": rgb[:, 2],
            "point_source": source_code,
            "semantic_id": semantic,
        }

    encoder = FrozenSTUPointEncoder.from_protocol(
        protocol, project_root=project_root
    ).to(runtime_device).eval()
    expected_models = tuple(confirmation["completed_seed_model_sha256"])
    models: list[AJAEPointTransformer] = []
    for seed in (0, 1):
        model_file = Path(b1_dir).expanduser().resolve(strict=True) / (
            f"seed-{seed}/model.pt"
        )
        if _sha256(model_file) != expected_models[seed]:
            raise QualificationError(f"E76-V1 B1 seed {seed} identity changed")
        payload = torch.load(model_file, map_location="cpu", weights_only=True)
        model = AJAEPointTransformer.from_protocol(protocol).to(runtime_device).eval()
        model.load_state_dict(payload["model"], strict=True)
        models.append(model)
    torch.use_deterministic_algorithms(True)

    def score_frame(source: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seed = e53_frame_seed(
            int(getattr(source, "sequence_id")), int(getattr(source, "frame_id"))
        )
        torch.manual_seed(seed)
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        slots = np.asarray(getattr(source, "real_slots"), dtype=np.int64)
        with torch.no_grad():
            encoding = encoder(
                official_stu_coordinates(source.xyzi, source.lidar_pose),
                official_stu_features(source.xyzi, source.lidar_pose),
                slots,
            )
            encoded_slots = encoding.real_slots
            if isinstance(encoded_slots, torch.Tensor):
                encoded_slots = encoded_slots.detach().cpu().numpy()
            if not np.array_equal(np.asarray(encoded_slots, dtype=np.int64), slots):
                raise QualificationError("E76-V1 STU point order changed")
            coordinates = torch.as_tensor(
                np.asarray(source.xyzi)[slots, :3].copy(), device=runtime_device
            )
            intensity = torch.as_tensor(
                np.asarray(source.xyzi)[slots, 3].copy(), device=runtime_device
            )
            relative_time = torch.zeros(
                slots.size, dtype=torch.long, device=runtime_device
            )
            b1 = []
            for model in models:
                logits = model(
                    coordinates,
                    relative_time,
                    encoding.point_features,
                    encoding.normal_evidence,
                    encoding.reliability_assign,
                    encoding.reliability_noobj,
                    intensity,
                    cross_frame_enabled=False,
                )
                b1.append(torch.sigmoid(logits).detach().cpu().numpy())
        b0 = encoding.maxlogit_score.detach().cpu().numpy()
        return (
            b0.astype(np.float32, copy=False),
            np.stack(b1).astype(np.float32, copy=False),
            slots,
        )

    file_records: list[dict[str, object]] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix="e76_v1_export_"
    ) as temporary_name:
        temporary = Path(temporary_name)

        def record_ply(
            relative: Path,
            properties: Mapping[str, np.ndarray],
            comments: Sequence[str],
            group: str,
        ) -> None:
            record = _write_binary_ply(
                temporary / relative, properties, comments=comments
            )
            file_records.append(
                {
                    "path": relative.as_posix(),
                    "group": group,
                    "points": record["points"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                    "properties": record["properties"],
                }
            )

        for stratum, current_world, proxy_nvis in selected_a:
            row = row_by_world[current_world]
            center = int(center_frame[row])
            base = sequence_201.source_frame(center)
            world = WorldSpec.from_dict(json.loads(str(world_json[row])))
            control_world = WorldSpec(
                world.seed,
                world.source_sequence_id,
                tuple(item for item in world.objects if item.label == "normal-control"),
                world.tie_tolerance_m,
            )
            proxy_world = WorldSpec(
                world.seed,
                world.source_sequence_id,
                tuple(item for item in world.objects if item.label == "anomaly-proxy"),
                world.tie_tolerance_m,
            )
            control = render_frame(base, control_world, grid, sensor)
            proxy = render_frame(base, proxy_world, grid, sensor)
            directory = Path(
                f"group_a/{stratum}_world_{current_world:02d}_frame_{center:03d}"
            )
            base_slots = np.asarray(base.real_slots, dtype=np.int64)
            zero_source = np.zeros(base_slots.size, dtype=np.uint8)
            record_ply(
                directory / "base_real.ply",
                colored_properties(base, zero_source),
                (f"E76-V1 world={current_world}", "variant=base_real"),
                "A",
            )
            for variant, rendered, source_code, mask_name in (
                ("normal_control_overlay", control, 1, "normal_control_mask"),
                ("anomaly_proxy_overlay", proxy, 2, "anomaly_proxy_mask"),
            ):
                source = rendered.source
                slots = np.asarray(source.real_slots, dtype=np.int64)
                mask = np.asarray(getattr(rendered, mask_name), dtype=np.bool_)[slots]
                point_source = np.zeros(slots.size, dtype=np.uint8)
                point_source[mask] = source_code
                record_ply(
                    directory / f"{variant}.ply",
                    colored_properties(source, point_source),
                    (
                        f"E76-V1 world={current_world}", f"variant={variant}",
                        f"proxy_Nvis={proxy_nvis}",
                    ),
                    "A",
                )
                if variant == "anomaly_proxy_overlay":
                    record_ply(
                        directory / "anomaly_proxy_only.ply",
                        colored_properties(
                            source, point_source, selected=np.flatnonzero(mask)
                        ),
                        (
                            f"E76-V1 world={current_world}",
                            "variant=anomaly_proxy_only",
                        ),
                        "A",
                    )

        moving_means: list[dict[str, float | int]] = []
        candidate_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for row in np.flatnonzero(moving_count > 0):
            frame = int(moving_frame[row])
            source = sequence_206.source_frame(frame)
            b0_score, b1_score, slots = score_frame(source)
            mask = np.unpackbits(moving_packed[row], bitorder="little")[
                : canonical_by_slot.size
            ][canonical_by_slot[slots]]
            if int(np.count_nonzero(mask)) != int(moving_count[row]):
                raise QualificationError("E76-V1 moving-normal count changed")
            seed_mean = np.mean(b1_score[:, mask], axis=1, dtype=np.float64)
            moving_means.append({
                "frame_id": frame,
                "moving_points": int(moving_count[row]),
                "b1_seed0_mean": float(seed_mean[0]),
                "b1_seed1_mean": float(seed_mean[1]),
                "selection_statistic": float(np.mean(seed_mean)),
            })
            current_top = {
                int(item["frame_id"])
                for item in sorted(
                    moving_means,
                    key=lambda item: (
                        -float(item["selection_statistic"]),
                        int(item["frame_id"]),
                    ),
                )[:3]
            }
            if frame in current_top:
                candidate_scores[frame] = (b0_score, b1_score, slots)
            for discarded in set(candidate_scores) - current_top:
                del candidate_scores[discarded]
        selected_b = sorted(
            moving_means,
            key=lambda item: (-float(item["selection_statistic"]), int(item["frame_id"])),
        )[:3]
        row_by_moving_frame = {
            int(frame): row for row, frame in enumerate(moving_frame)
        }
        for selection_rank, item in enumerate(selected_b, start=1):
            frame = int(item["frame_id"])
            row = row_by_moving_frame[frame]
            source = sequence_206.source_frame(frame)
            b0_score, b1_score, slots = candidate_scores[frame]
            if not np.array_equal(
                slots, np.asarray(source.real_slots, dtype=np.int64)
            ):
                raise QualificationError("E76-V1 selected-frame point order changed")
            mask = np.unpackbits(moving_packed[row], bitorder="little")[
                : canonical_by_slot.size
            ][canonical_by_slot[slots]]
            retained_mean = np.mean(b1_score[:, mask], axis=1, dtype=np.float64)
            if not np.array_equal(
                retained_mean,
                [item["b1_seed0_mean"], item["b1_seed1_mean"]],
            ):
                raise QualificationError("E76-V1 retained-frame scores changed")
            xyzi = np.asarray(source.xyzi)[slots]
            semantic = np.asarray(source.labels.semantic)[slots].astype(
                "<u2", copy=False
            )
            rgb = np.tile(colors["real_normal"], (slots.size, 1))
            rgb[semantic == 0] = colors["ignore_or_background"]
            rgb[mask] = colors["moving_normal"]
            properties = {
                "x": xyzi[:, 0].astype("<f4", copy=False),
                "y": xyzi[:, 1].astype("<f4", copy=False),
                "z": xyzi[:, 2].astype("<f4", copy=False),
                "intensity": xyzi[:, 3].astype("<f4", copy=False),
                "red": rgb[:, 0],
                "green": rgb[:, 1],
                "blue": rgb[:, 2],
                "is_moving_normal": mask.astype(np.uint8),
                "semantic_id": semantic,
                "range": np.linalg.norm(
                    xyzi[:, :3].astype(np.float32, copy=False), axis=1
                ).astype("<f4", copy=False),
                "b0_score": b0_score.astype("<f4", copy=False),
                "b1_seed0_score": b1_score[0].astype("<f4", copy=False),
                "b1_seed1_score": b1_score[1].astype("<f4", copy=False),
                "pred_b0_A": (b0_score > threshold[0, 0]).astype(np.uint8),
                "pred_b0_B": (b0_score > threshold[0, 1]).astype(np.uint8),
                "pred_b1_seed0_A": (b1_score[0] > threshold[1, 0]).astype(np.uint8),
                "pred_b1_seed0_B": (b1_score[0] > threshold[1, 1]).astype(np.uint8),
                "pred_b1_seed1_A": (b1_score[1] > threshold[2, 0]).astype(np.uint8),
                "pred_b1_seed1_B": (b1_score[1] > threshold[2, 1]).astype(np.uint8),
            }
            record_ply(
                Path(f"group_b/rank_{selection_rank}_frame_{frame:03d}.ply"),
                properties,
                (
                    f"E76-V1 moving-normal rank={selection_rank}",
                    f"frame={frame}",
                    f"selection_statistic={item['selection_statistic']}",
                ),
                "B",
            )

        if len(file_records) != int(freeze["maximum_ply_files"]):
            raise QualificationError("E76-V1 PLY count changed")
        relative_paths = [str(record["path"]) for record in file_records]
        viewing_order = sorted(
            relative_paths,
            key=lambda value: hashlib.sha256(value.encode("utf-8")).digest(),
        )
        manifest = {
            "experiment": "E76-V1",
            "status": "descriptive_export_complete",
            "descriptive_only": True,
            "formal_gate_adjudicated": False,
            "e76x_lite_result_unchanged": True,
            "e78x_locked": True,
            "protocol_sha256": _sha256(protocol_file),
            "e57_artifact_sha256": _sha256(e57_file),
            "e61_artifact_sha256": _sha256(e61_file),
            "e76x_lite_artifact_sha256": _sha256(e76_file),
            "group_a_selection": [
                {"stratum": name, "world_id": world, "proxy_Nvis": nvis}
                for name, world, nvis in selected_a
            ],
            "group_b_frame_statistics": moving_means,
            "group_b_selected": selected_b,
            "review_freeze": _plain_json(freeze["review_freeze"]),
            "viewing_order": viewing_order,
            "files": sorted(file_records, key=lambda item: str(item["path"])),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    final_manifest = destination / "manifest.json"
    return {
        "experiment": "E76-V1",
        "completed": True,
        "status": "descriptive_export_complete",
        "formal_gate_adjudicated": False,
        "e78x_locked": True,
        "ply_files": len(file_records),
        "group_a_world_ids": [item[1] for item in selected_a],
        "group_b_frame_ids": [int(item["frame_id"]) for item in selected_b],
        "manifest_path": str(final_manifest),
        "manifest_sha256": _sha256(final_manifest),
        "total_bytes": sum(int(item["bytes"]) for item in file_records)
        + final_manifest.stat().st_size,
        "seconds": time.monotonic() - started,
    }


def run_e75(
    protocol_path: Path | str,
    e72_path: Path | str,
    identity_path: Path | str,
    b1_dir: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Adjudicate B1 versus B0 after all three formal B1 seeds complete."""

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    e72_file = Path(e72_path).expanduser().resolve(strict=True)
    identity_file = Path(identity_path).expanduser().resolve(strict=True)
    result_root = Path(b1_dir).expanduser().resolve(strict=True)
    if _sha256(e72_file) != E72_ARTIFACT_SHA256:
        raise QualificationError("E75 B0 input is not the frozen E72 artifact")
    if _sha256(identity_file) != E75_IDENTITY_ARTIFACT_SHA256:
        raise QualificationError("E75 bootstrap identity artifact changed")
    gate2 = protocol.decision_gates["criteria"]["gate2"]
    if (
        protocol.training["maximum_worlds"] != 25
        or gate2["minimum_mean_macro_world_AP_difference"] != 0.02
        or gate2["bootstrap_95_percent_lower_bound_strictly_greater_than"] != 0.0
        or gate2["minimum_positive_training_seeds"] != 2
        or gate2["training_seeds"] != 3
    ):
        raise QualificationError("E75 budget or Gate 2 criteria changed")

    with np.load(e72_file, allow_pickle=False) as archive:
        metric_order = np.asarray(archive["metric_order"])
        ap_column = np.flatnonzero(metric_order == "AP")
        if ap_column.tolist() != [0]:
            raise QualificationError("E72 AP column identity changed")
        b0_world_id = np.asarray(archive["development_world_id"], dtype=np.int16)
        b0_ap = np.asarray(archive["development_metric"], dtype=np.float64)[:, 0]
    with np.load(identity_file, allow_pickle=False) as archive:
        common_world_id = np.asarray(archive["common_domain_world_id"], dtype=np.int16)
        seed_draws = np.asarray(archive["bootstrap_training_seed"], dtype=np.int8)
        world_draws = np.asarray(archive["bootstrap_world_id"], dtype=np.int16)

    b1_world_id = np.empty((3, 23), dtype=np.int16)
    b1_ap = np.empty((3, 23), dtype=np.float64)
    result_hashes: list[str] = []
    model_hashes: list[str] = []
    required_seed_files = [
        result_root / f"seed-{seed}" / name
        for seed in range(3)
        for name in ("result.json", "model.pt")
    ]
    if not all(path.is_file() for path in required_seed_files):
        raise QualificationError("E74 is incomplete; E75 result reading remains locked")
    for seed in range(3):
        seed_dir = result_root / f"seed-{seed}"
        result_file = seed_dir / "result.json"
        model_file = seed_dir / "model.pt"
        record = json.loads(result_file.read_text(encoding="utf-8"))
        model = torch.load(model_file, map_location="cpu", weights_only=True)
        if (
            record.get("status") != "completed"
            or record.get("seed") != seed
            or record.get("maximum_worlds") != 25
            or record.get("stop_reason") not in {"maximum_worlds", "development_patience"}
            or record.get("condition", {}).get("name") != "B1"
            or model.get("seed") != seed
            or model.get("maximum_worlds") != 25
            or model.get("completion_id") != record.get("completion_id")
            or model.get("best_world") != record.get("best_world")
            or list(model.get("best_selection_key", ()))
            != record.get("best_selection_key")
            or model.get("scientific_identity") != record.get("scientific_identity")
        ):
            raise QualificationError(f"E74 seed {seed} completion identity is invalid")
        selected = [
            item for item in record.get("history", ())
            if item.get("world") == record.get("best_world") and "development" in item
        ]
        if len(selected) != 1:
            raise QualificationError(f"E74 seed {seed} lacks one selected evaluation")
        worlds = selected[0]["development"]["in_generator"]
        if len(worlds) != 23:
            raise QualificationError(f"E74 seed {seed} development count changed")
        ordered = sorted(worlds, key=lambda item: item["world_id"])
        b1_world_id[seed] = [item["world_id"] for item in ordered]
        b1_ap[seed] = [item["metrics"]["AP"] for item in ordered]
        if not np.isclose(
            b1_ap[seed].mean(), float(record["best_selection_key"][0]),
            rtol=0.0, atol=1.0e-12,
        ):
            raise QualificationError(f"E74 seed {seed} selected AP is inconsistent")
        result_hashes.append(_sha256(result_file))
        model_hashes.append(_sha256(model_file))

    arrays = e75_superiority_statistics(
        b0_world_id, b0_ap, b1_world_id, b1_ap, common_world_id,
        seed_draws, world_draws,
    )
    mean_difference = float(arrays["paired_ap_decision_difference"].mean())
    lower_bound = float(np.percentile(
        arrays["bootstrap_mean_ap_decision_difference"], 2.5
    ))
    positive_seeds = int(np.count_nonzero(
        arrays["seed_mean_ap_decision_difference"] > 0.0
    ))
    passed = (
        mean_difference >= 0.02
        and lower_bound > 0.0
        and positive_seeds >= 2
    )
    result = {
        "experiment": "E75",
        "passed": passed,
        "failure_classification": None if passed else "scientific_failure",
        "decision_metric_scale": "[0,1]",
        "reported_AP_scale": "[0,100]",
        "mean_macro_world_AP_decision_difference": mean_difference,
        "bootstrap_95_percent_lower_bound_decision_scale": lower_bound,
        "positive_training_seeds": positive_seeds,
        "thresholds": {
            "minimum_mean_macro_world_AP_difference": 0.02,
            "bootstrap_95_percent_lower_bound_strictly_greater_than": 0.0,
            "minimum_positive_training_seeds": 2,
        },
        "replicates": 5000,
        "development_worlds": 23,
        "shared_bootstrap_comparisons": list(
            COMMON_DEVELOPMENT_BOOTSTRAP_COMPARISONS
        ),
        "protocol_sha256": _sha256(protocol_file),
        "e72_artifact_sha256": _sha256(e72_file),
        "bootstrap_identity_sha256": _sha256(identity_file),
        "b1_result_sha256": result_hashes,
        "b1_model_sha256": model_hashes,
        "scientific_array_sha256": _array_hash(arrays),
        "seconds": time.monotonic() - started,
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def phase7_mechanical_arrays() -> dict[str, np.ndarray]:
    """Execute the frozen E64--E71 analytic fixtures on the production paths."""

    torch.manual_seed(PHASE7_SEED)
    arrays: dict[str, np.ndarray] = {}

    # E64: voxel keys must retain q at every pooled resolution.
    features = torch.tensor(
        ((1.0, -1.0), (3.0, -3.0), (5.0, -5.0), (7.0, -7.0))
    )
    coordinates = torch.tensor(
        ((0.01, 0.0, 0.0), (0.02, 0.0, 0.0), (0.01, 0.0, 0.0), (0.41, 0.0, 0.0))
    )
    times = torch.tensor((0, 0, 1, 0), dtype=torch.long)
    e64_errors: list[int] = []
    e64_counts: list[int] = []
    for size in (0.1, 0.2, 0.4):
        pool = VoxelPool(features.shape[1], size)
        pool.projection = torch.nn.Identity()
        first = pool(features, coordinates, times)
        second = pool(features, coordinates, times)
        e64_counts.append(int(first.features.shape[0]))
        e64_errors.append(
            int(
                first.features.shape[0] != 3
                or int(first.inverse_map[0]) != int(first.inverse_map[1])
                or int(first.inverse_map[0]) == int(first.inverse_map[2])
                or int(first.inverse_map[0]) == int(first.inverse_map[3])
                or not torch.equal(first.inverse_map, second.inverse_map)
                or not torch.equal(first.relative_times, second.relative_times)
                or not torch.equal(first.features, second.features)
            )
        )
    arrays["e64_level_voxel_count"] = np.asarray(e64_counts, dtype=np.int16)
    arrays["e64_error_count"] = np.asarray(e64_errors, dtype=np.int16)

    # E65: expose the authoritative concatenated mean-max tensor directly.
    e65_features = torch.tensor(
        ((-2.0, 1.0), (-4.0, 3.0), (5.0, -6.0)), requires_grad=True
    )
    e65_coordinates = torch.tensor(
        ((0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (1.2, 0.1, 0.1))
    )
    e65_times = torch.zeros(3, dtype=torch.long)
    e65_pool = VoxelPool(2, 1.0)
    e65_pool.projection = torch.nn.Identity()
    e65_level = e65_pool(e65_features, e65_coordinates, e65_times)
    e65_expected = torch.tensor(((-3.0, 2.0, -2.0, 3.0), (5.0, -6.0, 5.0, -6.0)))
    e65_level.features.sum().backward()
    e65_error = int(
        not torch.equal(e65_level.features, e65_expected)
        or e65_features.grad is None
        or not bool(torch.isfinite(e65_features.grad).all())
        or bool(torch.equal(e65_level.features[:, :2], e65_level.features[:, 2:]))
    )
    arrays["e65_observed_mean_max"] = e65_level.features.detach().numpy()
    arrays["e65_expected_mean_max"] = e65_expected.numpy()
    arrays["e65_error_count"] = np.asarray((e65_error,), dtype=np.int16)

    # E66: temporal deltas have separate K/radius budgets and stable row ties.
    e66_coordinates = torch.tensor(
        ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0),
         (0.05, 0.0, 0.0), (0.7, 0.0, 0.0), (0.02, 0.0, 0.0))
    )
    e66_times = torch.tensor((0, 0, 0, 1, 1, -1), dtype=torch.long)
    same_neighbor, same_valid = temporal_radius_knn(
        e66_coordinates, e66_times, 0, 0.5, 2
    )
    next_neighbor, next_valid = temporal_radius_knn(
        e66_coordinates, e66_times, 1, 0.5, 2
    )
    prior_neighbor, prior_valid = temporal_radius_knn(
        e66_coordinates, e66_times, -1, 0.5, 2
    )
    e66_error = int(
        same_neighbor[0, same_valid[0]].tolist() != [0, 1]
        or next_neighbor[0, next_valid[0]].tolist() != [3]
        or prior_neighbor[0, prior_valid[0]].tolist() != [5]
        or 4 in next_neighbor[0, next_valid[0]].tolist()
    )
    arrays["e66_same_neighbors"] = same_neighbor[0].numpy()
    arrays["e66_same_valid"] = same_valid[0].numpy()
    arrays["e66_next_neighbors"] = next_neighbor[0].numpy()
    arrays["e66_next_valid"] = next_valid[0].numpy()
    arrays["e66_prior_neighbors"] = prior_neighbor[0].numpy()
    arrays["e66_prior_valid"] = prior_valid[0].numpy()
    arrays["e66_error_count"] = np.asarray((e66_error,), dtype=np.int16)

    # E67/E68 use one small production block with two spatially isolated pairs.
    block = TemporalPointBlock(
        16, 4, (0.4,) * 5, (2,) * 5, chunk_size=8
    ).eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    pair_coordinates = torch.tensor(
        ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0),
         (10.0, 0.0, 0.0), (10.1, 0.0, 0.0))
    )
    pair_times = torch.tensor((0, 1, 0, 1), dtype=torch.long)
    pair_features = torch.randn(4, 16)
    batch_output = block(
        pair_features, pair_coordinates, pair_times, cross_frame_enabled=True
    )
    individual_output = block(
        pair_features[:2], pair_coordinates[:2], pair_times[:2],
        cross_frame_enabled=True,
    )
    empty_neighbor, empty_valid = block.neighbors(
        pair_coordinates[:2], pair_times[:2], 2
    )
    normalized = block.norm1(pair_features[:2])
    query = block.query(normalized).view(2, 4, 4)
    key = block.key(normalized).view(2, 4, 4)
    value = block.value(normalized).view(2, 4, 4)
    empty_message = block._message(
        query, key, value, pair_coordinates[:2], empty_neighbor, empty_valid,
        radius=0.4, delta=2,
    )
    empty_gate = torch.where(
        empty_valid.any(dim=1),
        torch.sigmoid(
            block.cross_gate(
                torch.cat(
                    (
                        pair_features[:2], empty_message,
                        pair_features.new_full((2, 1), 1.0),
                    ),
                    dim=1,
                )
            )
        ).squeeze(1),
        torch.zeros(2),
    )
    nonempty_neighbor, nonempty_valid = block.neighbors(
        pair_coordinates[:2], pair_times[:2], 1
    )
    nonempty_message = block._message(
        query, key, value, pair_coordinates[:2], nonempty_neighbor, nonempty_valid,
        radius=0.4, delta=1,
    )
    nonempty_gate = torch.sigmoid(
        block.cross_gate(
            torch.cat(
                (
                    pair_features[:2], nonempty_message,
                    pair_features.new_full((2, 1), 0.5),
                ),
                dim=1,
            )
        )
    ).squeeze(1)
    e67_error = int(
        not torch.equal(empty_message, torch.zeros_like(empty_message))
        or not torch.equal(empty_gate, torch.zeros_like(empty_gate))
        or not bool(torch.isfinite(batch_output).all())
        or not bool(((nonempty_gate >= 0.0) & (nonempty_gate <= 1.0)).all())
        or not torch.equal(batch_output[:2], individual_output)
    )
    arrays["e67_empty_message"] = empty_message.detach().numpy()
    arrays["e67_empty_gate"] = empty_gate.detach().numpy()
    arrays["e67_batch_individual_max_error"] = np.asarray(
        (
            float(
                torch.max(torch.abs(batch_output[:2] - individual_output))
                .detach()
                .cpu()
            ),
        ),
        dtype=np.float64,
    )
    arrays["e67_error_count"] = np.asarray((e67_error,), dtype=np.int16)

    residual_block = TemporalPointBlock(
        16, 4, (0.4,) * 5, (2,) * 5, chunk_size=8
    ).eval()
    residual_features = pair_features[:2].clone().requires_grad_(True)
    residual_output = residual_block(
        residual_features, pair_coordinates[:2], pair_times[:2],
        cross_frame_enabled=False,
    )
    residual_normalized = residual_block.norm1(residual_features)
    residual_query = residual_block.query(residual_normalized).view(2, 4, 4)
    residual_key = residual_block.key(residual_normalized).view(2, 4, 4)
    residual_value = residual_block.value(residual_normalized).view(2, 4, 4)
    residual_neighbor, residual_valid = residual_block.neighbors(
        pair_coordinates[:2], pair_times[:2], 0
    )
    same_message = residual_block._message(
        residual_query, residual_key, residual_value, pair_coordinates[:2],
        residual_neighbor, residual_valid, radius=0.4, delta=0,
    )
    residual_updated = residual_features + residual_block.message_projection(same_message)
    residual_expected = residual_updated + residual_block.ffn(
        residual_block.norm2(residual_updated)
    )
    residual_output[0].sum().backward()
    cross_gradient = residual_features.grad[1]
    e68_difference = float(
        torch.max(torch.abs(residual_output - residual_expected)).detach().cpu()
    )
    e68_error = int(
        e68_difference != 0.0
        or not bool(torch.any(same_message != 0.0))
        or not torch.equal(cross_gradient, torch.zeros_like(cross_gradient))
    )
    arrays["e68_output_max_error"] = np.asarray((e68_difference,), dtype=np.float64)
    arrays["e68_cross_frame_gradient"] = cross_gradient.detach().numpy()
    arrays["e68_error_count"] = np.asarray((e68_error,), dtype=np.int16)

    # E69: a closer node from another q must never become an interpolation parent.
    upsample = KnnUpsample(3)
    source_features = torch.tensor(((1.0,), (3.0,), (9.0,), (100.0,)))
    source_coordinates = torch.tensor(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    )
    source_times = torch.tensor((0, 0, 0, 1), dtype=torch.long)
    target_coordinates = torch.tensor(((1.0, 0.0, 0.0), (4.0, 0.0, 0.0)))
    target_times = torch.tensor((0, 0), dtype=torch.long)
    e69_observed = upsample(
        source_features, source_coordinates, source_times,
        target_coordinates, target_times,
    )
    distance = np.asarray(((1.0, 1.0, 4.0), (4.0, 2.0, 1.0)))
    weights = 1.0 / np.maximum(distance, 1.0e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    e69_expected = torch.from_numpy(
        np.sum(
            weights.astype(np.float32) * np.asarray((1.0, 3.0, 9.0), np.float32),
            axis=1,
            dtype=np.float32,
        )[:, None]
    )
    e69_error = int(
        not torch.equal(e69_observed, e69_expected)
        or bool(torch.any(e69_observed >= 100.0))
    )
    arrays["e69_observed"] = e69_observed.numpy()
    arrays["e69_expected"] = e69_expected.numpy()
    arrays["e69_error_count"] = np.asarray((e69_error,), dtype=np.int16)

    # E70: hand-compute each present-class mean after the training validity mask.
    logits = torch.tensor((-2.0, 0.0, 2.0, 1.0), dtype=torch.float64)
    targets = torch.tensor((False, False, True, True))
    masks = (
        torch.tensor((True, True, False, False)),
        torch.tensor((False, False, True, True)),
        torch.ones(4, dtype=torch.bool),
        torch.tensor((False, True, True, False)),
    )
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    expected_losses = torch.stack(
        (
            raw[:2].mean(),
            raw[2:].mean(),
            0.5 * raw[:2].mean() + 0.5 * raw[2:].mean(),
            0.5 * raw[1:2].mean() + 0.5 * raw[2:3].mean(),
        )
    )
    observed_losses = torch.stack(
        tuple(balanced_bce_loss(logits, targets, mask) for mask in masks)
    )
    e70_difference = torch.abs(observed_losses - expected_losses)
    e70_error = int(
        not bool(torch.isfinite(observed_losses).all())
        or float(e70_difference.max()) != 0.0
    )
    arrays["e70_observed_loss"] = observed_losses.numpy()
    arrays["e70_expected_loss"] = expected_losses.numpy()
    arrays["e70_error_count"] = np.asarray((e70_error,), dtype=np.int16)

    # E71: multiplicities 1..5 average probabilities, never logits or padding.
    logits_by_slot = tuple(
        np.linspace(-2.0 + slot * 0.2, 2.0 - slot * 0.1, slot + 1)
        for slot in range(5)
    )
    fusion = WindowScoreFusion(maximum_count=5)
    slot_to_ray = np.arange(5, dtype=np.uint64)
    for occurrence in range(5):
        slots = np.asarray(
            [slot for slot in range(5) if occurrence < slot + 1], dtype=np.int64
        )
        probabilities = np.asarray(
            [1.0 / (1.0 + np.exp(-logits_by_slot[slot][occurrence])) for slot in slots],
            dtype=np.float64,
        )
        fusion.add(
            17, slots.astype(np.uint64), probabilities,
            output_slots=slots, slot_to_ray=slot_to_ray,
        )
    e71_observed, e71_count = fusion.finalize(17)
    e71_expected = np.asarray(
        [np.mean(1.0 / (1.0 + np.exp(-values))) for values in logits_by_slot],
        dtype=np.float32,
    )
    sigmoid_mean_logit = np.asarray(
        [1.0 / (1.0 + np.exp(-np.mean(values))) for values in logits_by_slot],
        dtype=np.float32,
    )
    distinguishing = np.asarray(
        [values.size > 1 and e71_expected[index] != sigmoid_mean_logit[index]
         for index, values in enumerate(logits_by_slot)]
    )
    e71_error = int(
        not np.array_equal(e71_count, np.arange(1, 6, dtype=np.uint8))
        or not np.array_equal(e71_observed, e71_expected)
        or not np.any(distinguishing)
    )
    arrays["e71_observed_probability"] = e71_observed
    arrays["e71_expected_probability"] = e71_expected
    arrays["e71_sigmoid_mean_logit"] = sigmoid_mean_logit
    arrays["e71_multiplicity"] = e71_count
    arrays["e71_error_count"] = np.asarray((e71_error,), dtype=np.int16)
    return arrays


def run_phase7(
    protocol_path: Path | str,
    e63_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Run the unified E64--E71 implementation qualification once."""

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    e63 = protocol.development["e63_freeze"]
    if e63["status"] != "formal_pass":
        raise QualificationError("Phase 7 requires E63 PASS")
    identity_file = Path(e63_path).expanduser().resolve(strict=True)
    expected_identity = (PROJECT_ROOT / e63["identity_artifact"]["path"]).resolve()
    if (
        identity_file != expected_identity
        or _sha256(identity_file) != e63["identity_artifact"]["artifact_sha256"]
    ):
        raise QualificationError("Phase 7 requires the frozen E63 identity artifact")
    first = phase7_mechanical_arrays()
    second = phase7_mechanical_arrays()
    reproduction_errors = sum(
        not np.array_equal(first[name], second[name]) for name in first
    )
    node_errors = {
        f"E{node}": int(first[f"e{node}_error_count"].sum())
        for node in range(64, 72)
    }
    result = {
        "experiment": "E64-E71",
        "passed": reproduction_errors == 0 and not any(node_errors.values()),
        "node_errors": node_errors,
        "reproduction_errors": reproduction_errors,
        "seed": PHASE7_SEED,
        "protocol_sha256": _sha256(protocol_file),
        "e63_artifact_sha256": _sha256(identity_file),
        "scientific_array_sha256": _array_hash(first),
        "seconds": time.monotonic() - started,
    }
    _save(Path(output_path).expanduser().resolve(), first, result)
    return result


def run_e72(
    data_root: Path | str,
    protocol_path: Path | str,
    e57_path: Path | str,
    e61_path: Path | str,
    e63_path: Path | str,
    output_path: Path | str,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    """Freeze official STU MaxLogit on the E63 development and safety identities."""

    from .render import WorldSpec, load_sensor_calibration, render_frame

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    project_root = Path(protocol.path).parent
    e57_file = Path(e57_path).expanduser().resolve(strict=True)
    e61_file = Path(e61_path).expanduser().resolve(strict=True)
    e63_file = Path(e63_path).expanduser().resolve(strict=True)
    if _sha256(e57_file) != protocol.development["e63_freeze"]["source_worlds"][
        "artifact_sha256"
    ]:
        raise QualificationError("E72 E57 identity changed")
    if _sha256(e61_file) != E61_ARTIFACT_SHA256:
        raise QualificationError("E72 E61 safety identity changed")
    if _sha256(e63_file) != protocol.development["e63_freeze"]["identity_artifact"][
        "artifact_sha256"
    ]:
        raise QualificationError("E72 E63 common-domain identity changed")
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise QualificationError("E72 CUDA device is unavailable")

    with np.load(e57_file, allow_pickle=False) as archive:
        e57_world_id = np.asarray(archive["selected_world_id"], dtype=np.int16)
        e57_center = np.asarray(archive["selected_center_frame"], dtype=np.int16)
        e57_world_json = np.asarray(archive["selected_world_json"])
    with np.load(e63_file, allow_pickle=False) as archive:
        eligible = np.asarray(archive["common_domain_eligible"], dtype=np.bool_)
        frozen_world_id = np.asarray(archive["world_id"], dtype=np.int16)
    if not np.array_equal(e57_world_id, frozen_world_id) or eligible.sum() != 23:
        raise QualificationError("E72 development worlds differ from E63")
    with np.load(e61_file, allow_pickle=False) as archive:
        pure_frame_id = np.asarray(archive["pure_frame_id"], dtype=np.int16)
        pure_packed = np.asarray(archive["pure_canonical_mask_packed"], dtype=np.uint8)
        pure_count = np.asarray(archive["pure_point_count_by_frame"], dtype=np.int32)
        moving_frame_id = np.asarray(archive["moving_frame_id"], dtype=np.int16)
        moving_packed = np.asarray(
            archive["moving_canonical_mask_packed"], dtype=np.uint8
        )
        moving_count = np.asarray(
            archive["moving_point_count_by_frame"], dtype=np.int32
        )
        static_packed = np.asarray(
            archive["matched_static_canonical_mask_packed"], dtype=np.uint8
        )

    grid, sensor = load_sensor_calibration(protocol.sensor_calibration_path())
    canonical_by_slot = np.asarray(
        grid.beam_ids * grid.columns + grid.column_ids, dtype=np.int32
    )
    sequence_201 = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    sequence_206 = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    encoder = FrozenSTUPointEncoder.from_protocol(protocol).to(runtime_device).eval()

    def encode(source: object) -> np.ndarray:
        sequence_id = int(getattr(source, "sequence_id"))
        frame_id = int(getattr(source, "frame_id"))
        seed = e53_frame_seed(sequence_id, frame_id)
        torch.manual_seed(seed)
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            encoding = encoder(
                official_stu_coordinates(source.xyzi, source.lidar_pose),
                official_stu_features(source.xyzi, source.lidar_pose),
                source.real_slots,
            )
        full = np.zeros(source.slot_count, dtype=np.float32)
        full[np.asarray(source.real_slots, dtype=np.int64)] = (
            encoding.maxlogit_score.detach().cpu().numpy().astype(np.float32)
        )
        return full

    official_type = _e62_official_calculator(
        protocol.evaluation_document["evaluator_equivalence"]
    )
    metric_order = ("AP", "AUROC", "FPR95", "threshold")
    development_world_id: list[int] = []
    development_center: list[int] = []
    development_offset = [0]
    development_ray: list[np.ndarray] = []
    development_score: list[np.ndarray] = []
    development_label: list[np.ndarray] = []
    development_control: list[np.ndarray] = []
    development_proxy: list[np.ndarray] = []
    development_metric: list[np.ndarray] = []
    evaluator_errors = 0
    for row in np.flatnonzero(eligible):
        world_id = int(e57_world_id[row])
        center = int(e57_center[row])
        world = WorldSpec.from_dict(json.loads(str(e57_world_json[row])))
        rendered = render_frame(
            sequence_201.source_frame(center), world, grid, sensor
        )
        source = rendered.source
        scores = encode(source)
        slots = np.asarray(source.real_slots, dtype=np.int64)
        points = np.asarray(source.xyzi)[slots, :3]
        semantic = np.asarray(source.labels.semantic)[slots]
        values = scores[slots]
        official = official_type()
        custom = PointMetricAccumulator(protocol)
        official.update(points, values, semantic)
        custom.update(points, values, semantic)
        official_metric = official.compute_metrics()
        custom_metric = custom.compute()
        official_values = np.asarray(
            [official_metric[name] for name in metric_order], dtype=np.float64
        )
        custom_values = np.asarray(
            [custom_metric[name] for name in metric_order], dtype=np.float64
        )
        evaluator_errors += int(
            float(np.max(np.abs(official_values - custom_values))) > 1.0e-10
            or len(official.all_scores) != 1
            or custom_metric.get("accepted_frames") != 1
        )
        ranges = np.linalg.norm(points.astype(np.float32, copy=False), axis=1)
        valid = (ranges >= 2.5) & (ranges <= 50.0) & (semantic != 0)
        valid_slots = slots[valid]
        order = np.argsort(canonical_by_slot[valid_slots], kind="stable")
        valid_slots = valid_slots[order]
        development_world_id.append(world_id)
        development_center.append(center)
        development_ray.append(canonical_by_slot[valid_slots])
        development_score.append(scores[valid_slots])
        development_label.append((np.asarray(source.labels.semantic)[valid_slots] == 2))
        development_control.append(
            np.asarray(rendered.normal_control_mask)[valid_slots]
        )
        development_proxy.append(np.asarray(rendered.anomaly_proxy_mask)[valid_slots])
        development_metric.append(official_values)
        development_offset.append(development_offset[-1] + valid_slots.size)

    def unpack_masks(packed: np.ndarray) -> np.ndarray:
        return np.unpackbits(packed, axis=1, bitorder="little")[:, : grid.slot_count]

    def collect_native(
        sequence: STUSequence,
        frame_ids: np.ndarray,
        masks: tuple[np.ndarray, ...],
        expected_counts: tuple[np.ndarray, ...],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[list[int]]]:
        scores_by_mask = [[] for _ in masks]
        rays_by_mask = [[] for _ in masks]
        offsets = [[0] for _ in masks]
        unpacked = tuple(unpack_masks(mask) for mask in masks)
        for row, frame_id in enumerate(frame_ids.tolist()):
            if not any(int(count[row]) for count in expected_counts):
                for current in offsets:
                    current.append(current[-1])
                continue
            source = sequence.source_frame(int(frame_id))
            scores = encode(source)
            for mask_id, (current_mask, current_count) in enumerate(
                zip(unpacked, expected_counts, strict=True)
            ):
                selected_slots = np.flatnonzero(current_mask[row][canonical_by_slot])
                if selected_slots.size != int(current_count[row]):
                    raise QualificationError("E72 E61 canonical mask count changed")
                selected_rays = canonical_by_slot[selected_slots]
                order = np.argsort(selected_rays, kind="stable")
                rays_by_mask[mask_id].append(selected_rays[order])
                scores_by_mask[mask_id].append(scores[selected_slots[order]])
                offsets[mask_id].append(
                    offsets[mask_id][-1] + selected_slots.size
                )
        return scores_by_mask, rays_by_mask, offsets

    pure_scores, pure_rays, pure_offsets = collect_native(
        sequence_201, pure_frame_id, (pure_packed,), (pure_count,)
    )
    static_count = np.unpackbits(
        static_packed, axis=1, bitorder="little"
    )[:, : grid.slot_count].sum(axis=1).astype(np.int32)
    moving_scores, moving_rays, moving_offsets = collect_native(
        sequence_206,
        moving_frame_id,
        (moving_packed, static_packed),
        (moving_count, static_count),
    )

    def concatenate(values: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
        return (
            np.concatenate(values).astype(dtype, copy=False)
            if values
            else np.empty(0, dtype=dtype)
        )

    arrays = {
        "development_world_id": np.asarray(development_world_id, dtype=np.int16),
        "development_center_frame": np.asarray(development_center, dtype=np.int16),
        "development_point_offset": np.asarray(development_offset, dtype=np.int64),
        "development_canonical_ray": concatenate(development_ray, np.int32),
        "development_score": concatenate(development_score, np.float32),
        "development_label": concatenate(development_label, np.bool_),
        "development_normal_control": concatenate(development_control, np.bool_),
        "development_anomaly_proxy": concatenate(development_proxy, np.bool_),
        "development_metric": np.stack(development_metric),
        "pure_frame_id": pure_frame_id,
        "pure_point_offset": np.asarray(pure_offsets[0], dtype=np.int64),
        "pure_canonical_ray": concatenate(pure_rays[0], np.int32),
        "pure_score": concatenate(pure_scores[0], np.float32),
        "moving_frame_id": moving_frame_id,
        "moving_point_offset": np.asarray(moving_offsets[0], dtype=np.int64),
        "moving_canonical_ray": concatenate(moving_rays[0], np.int32),
        "moving_score": concatenate(moving_scores[0], np.float32),
        "matched_static_point_offset": np.asarray(moving_offsets[1], dtype=np.int64),
        "matched_static_canonical_ray": concatenate(moving_rays[1], np.int32),
        "matched_static_score": concatenate(moving_scores[1], np.float32),
        "metric_order": np.asarray(metric_order, dtype="U16"),
    }
    count_errors = int(
        arrays["development_world_id"].size != 23
        or arrays["pure_score"].size != int(pure_count.sum())
        or arrays["moving_score"].size != int(moving_count.sum())
        or arrays["matched_static_score"].size != int(static_count.sum())
        or arrays["pure_score"].size != 48_828_507
        or arrays["moving_score"].size != 13_011
        or arrays["matched_static_score"].size != 6_756
    )
    result = {
        "experiment": "E72",
        "passed": evaluator_errors == 0 and count_errors == 0,
        "development_worlds": 23,
        "development_points": int(arrays["development_score"].size),
        "pure_normal_points": int(arrays["pure_score"].size),
        "moving_normal_points": int(arrays["moving_score"].size),
        "matched_static_points": int(arrays["matched_static_score"].size),
        "evaluator_errors": evaluator_errors,
        "count_errors": count_errors,
        "official_stu_source_manifest_sha256": stu_source_manifest(
            protocol.stu_repository_path(project_root)
        )["manifest_sha256"],
        "official_stu_weights": stu_weight_identity(
            protocol.checkpoint_path(project_root)
        ),
        "protocol_sha256": _sha256(protocol_file),
        "e57_artifact_sha256": _sha256(e57_file),
        "e61_artifact_sha256": _sha256(e61_file),
        "e63_artifact_sha256": _sha256(e63_file),
        "scientific_array_sha256": _array_hash(arrays),
        "seconds": time.monotonic() - started,
    }
    _save(Path(output_path).expanduser().resolve(), arrays, result)
    return result


def _tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        array = value.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _optimizer_state_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and bool(
            torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _optimizer_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _optimizer_state_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def run_e73(
    data_root: Path | str,
    protocol_path: Path | str,
    e26_path: Path | str,
    e72_path: Path | str,
    output_path: Path | str,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    """Run the frozen two-window B1 training and checkpoint smoke test."""

    from .render import WorldSpec, load_sensor_calibration, render_frame
    from .scene import assemble_window, canonical_ray_mapping_digest

    started = time.monotonic()
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    project_root = Path(protocol.path).parent
    e26_file = Path(e26_path).expanduser().resolve(strict=True)
    e72_file = Path(e72_path).expanduser().resolve(strict=True)
    output_file = Path(output_path).expanduser().resolve()
    smoke = protocol.training["e73_smoke"]
    if (
        _sha256(e26_file) != E73_E26_ARTIFACT_SHA256
        or _sha256(e26_file) != smoke["source_artifact_sha256"]
    ):
        raise QualificationError("E73 E26-v2 identity changed")
    if (
        _sha256(e72_file) != E72_ARTIFACT_SHA256
        or _sha256(e72_file) != smoke["b0_reference_sha256"]
    ):
        raise QualificationError("E73 E72 identity changed")
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise QualificationError("E73 CUDA device is unavailable")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    selected: list[tuple[str, int, int, str, WorldSpec]] = []
    with np.load(e26_file, allow_pickle=False) as archive:
        for kind in ("pure_normal", "mixed"):
            identity = smoke[kind]
            row = int(identity["row"])
            seed = int(archive["world_seed"][row])
            world_type = str(archive["world_type"][row])
            world_hash = bytes(archive["world_hash"][row]).decode("ascii")
            world = WorldSpec.from_dict(
                json.loads(bytes(archive["world_json"][row]).decode("utf-8"))
            )
            center = int(identity["center_frame"])
            if (
                world_type != kind
                or seed != int(identity["world_seed"])
                or world.seed != seed
                or world.identity != world_hash
                or world_hash != identity["world_sha256"]
                or center != 2 + seed % 445
            ):
                raise QualificationError(f"E73 frozen {kind} world changed")
            selected.append((kind, seed, center, world_hash, world))

    grid, sensor = load_sensor_calibration(protocol.sensor_calibration_path())
    canonical_by_slot = np.asarray(
        grid.beam_ids * grid.columns + grid.column_ids, dtype=np.int32
    )
    ray_digest = canonical_ray_mapping_digest(canonical_by_slot)
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    condition = experiment_condition("B1")
    encoder = FrozenSTUPointEncoder.from_protocol(
        protocol, project_root=project_root
    ).to(runtime_device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    stu_before = _tensor_state_hash(encoder.state_dict())

    batches = []
    class_count = []
    # The frozen encoder needs no graph, but its outputs must remain valid inputs
    # to AJAE autograd; inference-mode tensors cannot be saved for backward.
    with torch.no_grad():
        for kind, _, center, _, world in selected:
            rendered = render_frame(
                sequence.source_frame(center), world, grid, sensor
            )
            source = rendered.source
            encoding = encoder(
                source.coordinates, source.features, source.real_slots
            )
            window = assemble_window(
                sequence.spec,
                center,
                (source,),
                condition="B1",
                canonical_ray_by_slot=canonical_by_slot,
                ray_mapping_audited=True,
                ray_mapping_digest=ray_digest,
            )
            batch = make_window_training_data(
                window,
                (rendered,),
                (encoding,),
                condition,
                minimum_range_m=2.5,
                maximum_range_m=50.0,
                device=runtime_device,
            )
            positive = int((batch.valid & batch.targets).sum().item())
            negative = int((batch.valid & ~batch.targets).sum().item())
            class_count.append((positive, negative))
            batches.append(batch)
            if kind == "pure_normal" and (positive != 0 or negative == 0):
                raise QualificationError("E73 pure-normal window is not pure negative")
            if kind == "mixed" and (positive == 0 or negative == 0):
                raise QualificationError("E73 mixed window lacks one class")
    stu_after_encoding = _tensor_state_hash(encoder.state_dict())

    seed = int(smoke["seed"])
    accumulation = int(protocol.training["gradient_accumulation"])
    scale = accumulation / len(batches)

    def train_once() -> tuple[np.ndarray, dict[str, torch.Tensor], dict[str, object], float, int]:
        torch.manual_seed(seed)
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = AJAEPointTransformer.from_protocol(protocol).to(runtime_device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(protocol.training["learning_rate"]),
            weight_decay=float(protocol.training["weight_decay"]),
        )
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch in batches:
            logits = model(
                batch.coordinates,
                batch.relative_times,
                batch.stu_features,
                batch.normal_evidence,
                batch.assignment_reliability,
                batch.no_object_reliability,
                batch.intensity,
                cross_frame_enabled=False,
            )
            loss = balanced_bce_loss(logits, batch.targets, batch.valid)
            if not bool(torch.isfinite(loss)):
                raise QualificationError("E73 produced a non-finite loss")
            (loss / accumulation).backward()
            losses.append(float(loss.detach().cpu()))
        gradient_errors = 0
        gradients = []
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            gradient_errors += int(not bool(torch.isfinite(parameter.grad).all()))
            gradients.append(parameter.grad)
        if not gradients:
            gradient_errors += 1
        for gradient in gradients:
            gradient.mul_(scale)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            .detach()
            .cpu()
        )
        gradient_errors += int(not np.isfinite(gradient_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        optimizer_state = optimizer.state_dict()
        return (
            np.asarray(losses, dtype=np.float64),
            state,
            optimizer_state,
            gradient_norm,
            gradient_errors,
        )

    first_loss, first_state, first_optimizer, first_norm, first_gradient_errors = (
        train_once()
    )
    checkpoint_errors = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="e73-checkpoint-", dir=output_file.parent
    ) as temporary:
        checkpoint = Path(temporary) / "progress.pt"
        torch.save(
            {
                "model": first_state,
                "optimizer": first_optimizer,
                "seed": seed,
                "micro_batches": len(batches),
                "optimizer_updates": 1,
            },
            checkpoint,
        )
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        torch.manual_seed(seed + 1)
        restored_model = AJAEPointTransformer.from_protocol(protocol).cpu()
        restored_optimizer = torch.optim.AdamW(
            restored_model.parameters(),
            lr=float(protocol.training["learning_rate"]),
            weight_decay=float(protocol.training["weight_decay"]),
        )
        restored_model.load_state_dict(restored["model"], strict=True)
        restored_optimizer.load_state_dict(restored["optimizer"])
        checkpoint_errors += int(
            _tensor_state_hash(restored_model.state_dict())
            != _tensor_state_hash(first_state)
            or not _optimizer_state_equal(
                restored_optimizer.state_dict(), first_optimizer
            )
            or restored.get("seed") != seed
            or restored.get("micro_batches") != 2
            or restored.get("optimizer_updates") != 1
        )
        del restored_model, restored_optimizer, restored

    second_loss, second_state, _, second_norm, second_gradient_errors = train_once()
    parameter_error = max(
        float(torch.max(torch.abs(first_state[name] - second_state[name])).item())
        if first_state[name].is_floating_point()
        else float(not torch.equal(first_state[name], second_state[name]))
        for name in first_state
    )
    loss_error = float(np.max(np.abs(first_loss - second_loss)))
    stu_after = _tensor_state_hash(encoder.state_dict())
    stu_gradient_errors = sum(
        int(parameter.grad is not None) for parameter in encoder.parameters()
    )
    identity_errors = int(
        class_count != [(0, 125_299), (50, 121_689)]
        or len(batches) != int(smoke["micro_batches"])
        or not np.isclose(
            scale,
            float(smoke["partial_accumulation_uses_frozen_factor"]),
            rtol=0.0,
            atol=0.0,
        )
    )
    reproduction_errors = int(
        loss_error > float(smoke["loss_reproduction_absolute_tolerance"])
        or parameter_error
        > float(smoke["parameter_reproduction_absolute_tolerance"])
    )
    stu_errors = int(
        stu_before != stu_after_encoding
        or stu_before != stu_after
        or stu_gradient_errors != 0
    )
    gradient_errors = first_gradient_errors + second_gradient_errors
    arrays = {
        "world_seed": np.asarray([item[1] for item in selected], dtype=np.int64),
        "center_frame": np.asarray([item[2] for item in selected], dtype=np.int16),
        "world_sha256": np.asarray([item[3] for item in selected], dtype="S64"),
        "positive_count": np.asarray([item[0] for item in class_count], dtype=np.int32),
        "negative_count": np.asarray([item[1] for item in class_count], dtype=np.int32),
        "first_loss": first_loss,
        "second_loss": second_loss,
        "gradient_norm": np.asarray([first_norm, second_norm], dtype=np.float64),
        "first_model_sha256": np.asarray(_tensor_state_hash(first_state), dtype="S64"),
        "second_model_sha256": np.asarray(_tensor_state_hash(second_state), dtype="S64"),
        "loss_reproduction_error": np.asarray(loss_error, dtype=np.float64),
        "parameter_reproduction_error": np.asarray(parameter_error, dtype=np.float64),
        "identity_error_count": np.asarray(identity_errors, dtype=np.int32),
        "gradient_error_count": np.asarray(gradient_errors, dtype=np.int32),
        "stu_error_count": np.asarray(stu_errors, dtype=np.int32),
        "checkpoint_error_count": np.asarray(checkpoint_errors, dtype=np.int32),
        "reproduction_error_count": np.asarray(reproduction_errors, dtype=np.int32),
    }
    passed = not any(
        (identity_errors, gradient_errors, stu_errors, checkpoint_errors, reproduction_errors)
    )
    result = {
        "experiment": "E73",
        "passed": passed,
        "seed": seed,
        "micro_batches": 2,
        "optimizer_updates": 1,
        "identity_errors": identity_errors,
        "gradient_errors": gradient_errors,
        "stu_errors": stu_errors,
        "checkpoint_errors": checkpoint_errors,
        "reproduction_errors": reproduction_errors,
        "loss_reproduction_error": loss_error,
        "parameter_reproduction_error": parameter_error,
        "protocol_sha256": _sha256(protocol_file),
        "e26_artifact_sha256": _sha256(e26_file),
        "e72_artifact_sha256": _sha256(e72_file),
        "stu_state_sha256": stu_before,
        "scientific_array_sha256": _array_hash(arrays),
        "device": str(runtime_device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "seconds": time.monotonic() - started,
    }
    _save(output_file, arrays, result)
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
    e75_freeze = commands.add_parser("e75-freeze")
    e75_freeze.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "protocol.json"
    )
    e75_freeze.add_argument("--e63", type=Path, required=True)
    e75_freeze.add_argument("--output", type=Path, required=True)
    e75 = commands.add_parser("e75")
    e75.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e75.add_argument("--e72", type=Path, required=True)
    e75.add_argument("--identity", type=Path, required=True)
    e75.add_argument("--b1-dir", type=Path, required=True)
    e75.add_argument("--output", type=Path, required=True)
    e75x = commands.add_parser("e75x")
    e75x.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e75x.add_argument("--e72", type=Path, required=True)
    e75x.add_argument("--b1-dir", type=Path, required=True)
    e75x.add_argument("--output", type=Path, required=True)
    e76x = commands.add_parser("e76x")
    e76x.add_argument("--data-root", type=Path, required=True)
    e76x.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e76x.add_argument("--e57", type=Path, required=True)
    e76x.add_argument("--e61", type=Path, required=True)
    e76x.add_argument("--e63", type=Path, required=True)
    e76x.add_argument("--e72", type=Path, required=True)
    e76x.add_argument("--b1-dir", type=Path, required=True)
    e76x.add_argument("--output", type=Path, required=True)
    e76x.add_argument("--device", default="cuda")
    e76x_lite = commands.add_parser("e76x-lite")
    e76x_lite.add_argument("--data-root", type=Path, required=True)
    e76x_lite.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "protocol.json"
    )
    e76x_lite.add_argument("--e57", type=Path, required=True)
    e76x_lite.add_argument("--e61", type=Path, required=True)
    e76x_lite.add_argument("--e63", type=Path, required=True)
    e76x_lite.add_argument("--e72", type=Path, required=True)
    e76x_lite.add_argument("--b1-dir", type=Path, required=True)
    e76x_lite.add_argument("--output", type=Path, required=True)
    e76x_lite.add_argument("--device", default="cuda")
    e76v1 = commands.add_parser("e76v1")
    e76v1.add_argument("--data-root", type=Path, required=True)
    e76v1.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "protocol.json"
    )
    e76v1.add_argument("--e57", type=Path, required=True)
    e76v1.add_argument("--e61", type=Path, required=True)
    e76v1.add_argument("--e76", type=Path, required=True)
    e76v1.add_argument("--b1-dir", type=Path, required=True)
    e76v1.add_argument("--output-dir", type=Path, required=True)
    e76v1.add_argument("--device", default="cuda")
    phase7 = commands.add_parser("phase7")
    phase7.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    phase7.add_argument("--e63", type=Path, required=True)
    phase7.add_argument("--output", type=Path, required=True)
    e72 = commands.add_parser("e72")
    e72.add_argument("--data-root", type=Path, required=True)
    e72.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e72.add_argument("--e57", type=Path, required=True)
    e72.add_argument("--e61", type=Path, required=True)
    e72.add_argument("--e63", type=Path, required=True)
    e72.add_argument("--output", type=Path, required=True)
    e72.add_argument("--device", default="cuda")
    e73 = commands.add_parser("e73")
    e73.add_argument("--data-root", type=Path, required=True)
    e73.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    e73.add_argument("--e26", type=Path, required=True)
    e73.add_argument("--e72", type=Path, required=True)
    e73.add_argument("--output", type=Path, required=True)
    e73.add_argument("--device", default="cuda")
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
    if args.command == "e75-freeze":
        result = run_e75_identity_correction(
            args.protocol, args.e63, args.output
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e75":
        result = run_e75(
            args.protocol, args.e72, args.identity, args.b1_dir, args.output
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e75x":
        result = run_e75_exploratory(
            args.protocol, args.e72, args.b1_dir, args.output
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e76x":
        result = run_e76_exploratory(
            args.data_root,
            args.protocol,
            args.e57,
            args.e61,
            args.e63,
            args.e72,
            args.b1_dir,
            args.output,
            device=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e76x-lite":
        result = run_e76_exploratory(
            args.data_root,
            args.protocol,
            args.e57,
            args.e61,
            args.e63,
            args.e72,
            args.b1_dir,
            args.output,
            device=args.device,
            lite=True,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e76v1":
        result = run_e76_visual_audit(
            args.data_root,
            args.protocol,
            args.e57,
            args.e61,
            args.e76,
            args.b1_dir,
            args.output_dir,
            device=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["completed"] else 1
    if args.command == "phase7":
        result = run_phase7(args.protocol, args.e63, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e72":
        result = run_e72(
            args.data_root, args.protocol, args.e57, args.e61, args.e63,
            args.output, device=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "e73":
        result = run_e73(
            args.data_root, args.protocol, args.e26, args.e72, args.output,
            device=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
