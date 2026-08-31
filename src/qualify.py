#!/usr/bin/env python3
"""Execute the frozen post-Gate-1 mechanical qualifications."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

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
E53_SEED_NAMESPACE = "E53-STU-query-v1"


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
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
