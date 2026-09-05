#!/usr/bin/env python3
"""Rebuild AJAE's frozen runtime inputs from the official STU data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from .protocol import AJAEProtocol, load_protocol
    from .render import (
        SUPPORT_POOL_SEMANTICS,
        calibrated_ray_grid_from_e11,
        calibrate_sensor,
        save_sensor_calibration,
    )
    from .scene import LabelMode, STUSequence
except ImportError:  # Direct execution from src/.
    from protocol import AJAEProtocol, load_protocol
    from render import (  # type: ignore[no-redef]
        SUPPORT_POOL_SEMANTICS,
        calibrated_ray_grid_from_e11,
        calibrate_sensor,
        save_sensor_calibration,
    )
    from scene import LabelMode, STUSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SOURCE_FRAMES = (0, 681)
VALIDATION_ANCHOR_FRAMES = (2, 679)
TRAINING_SOURCE_FRAMES = (0, 448)
TRAINING_ANCHOR_FRAMES = (2, 446)
_SUPPORT_SEQUENCE: STUSequence | None = None


class PreparationError(RuntimeError):
    """Report a runtime input that cannot reproduce the frozen protocol."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _scientific_array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write stable NPZ bytes so the protocol can bind the regenerated file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 9
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    os.replace(temporary, path)


def _pose(frame: object) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(frame.lidar_pose, dtype=np.float64)  # type: ignore[attr-defined]
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise PreparationError("source LiDAR pose must be finite float64[4,4]")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-3, rtol=1.0e-3):
        raise PreparationError("source LiDAR pose rotation is not orthonormal")
    return rotation, pose[:3, 3]


def _splitmix64(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def _trimmed_plane(
    points: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    if points.shape[0] < 32:
        raise PreparationError("insufficient_support")
    center = points.mean(axis=0)
    covariance = (points - center).T @ (points - center) / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or float(eigenvalues[-2]) <= 1.0e-12:
        raise PreparationError("degenerate_covariance")
    residual = np.abs((points - center) @ eigenvectors[:, 0])
    retain = int(math.ceil(0.90 * points.shape[0]))
    retained = points[np.argsort(residual, kind="stable")[:retain]]
    retained_center = retained.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(
        retained - retained_center, full_matrices=False
    )
    if (
        singular_values.size < 2
        or not np.isfinite(singular_values).all()
        or float(np.square(singular_values[-2]) / retained.shape[0]) <= 1.0e-12
    ):
        raise PreparationError("degenerate_covariance")
    normal = vectors[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, retained_center))
    if not np.isfinite(normal).all() or not math.isfinite(offset) or normal[2] <= 0.0:
        raise PreparationError("nonfinite_or_unsolved_plane")
    height = -(normal[0] * anchor[0] + normal[1] * anchor[1] + offset) / normal[2]
    absolute = np.abs(points @ normal + offset)
    if not math.isfinite(float(height)) or not np.isfinite(absolute).all():
        raise PreparationError("nonfinite_or_unsolved_plane")
    return normal, offset, float(height), float(np.quantile(absolute, 0.95))


def _qualified_plane(
    points: np.ndarray, anchor: np.ndarray, radius_m: float
) -> tuple[np.ndarray, float] | None:
    estimates: list[tuple[np.ndarray, float, float, float, float]] = []
    for scale in (0.75, 1.0, 1.25):
        radius = scale * radius_m
        selected = np.linalg.norm(points[:, :2] - anchor[:2], axis=1) <= radius
        try:
            normal, offset, height, q95 = _trimmed_plane(points[selected], anchor)
        except (PreparationError, np.linalg.LinAlgError):
            return None
        median = float(np.median(np.abs(points[selected] @ normal + offset)))
        estimates.append((normal, offset, height, q95, median))
    small, middle, large = estimates
    angle = math.degrees(
        math.acos(float(np.clip(np.dot(small[0], large[0]), -1.0, 1.0)))
    )
    if (
        middle[3] > 0.08
        or middle[4] > 0.03
        or angle > 5.0
        or abs(small[2] - large[2]) > 0.08
    ):
        return None
    return middle[0], middle[1]


def _support_anchor(anchor_frame: int) -> dict[str, np.ndarray]:
    sequence = _SUPPORT_SEQUENCE
    if sequence is None:
        raise PreparationError("support sequence is not initialized")
    frames = tuple(
        sequence.source_frame(frame)
        for frame in range(anchor_frame - 2, anchor_frame + 3)
    )
    anchor = frames[2]
    if anchor.labels is None:
        raise PreparationError("support construction requires train labels")
    point_parts: list[np.ndarray] = []
    semantic_parts: list[np.ndarray] = []
    for frame in frames:
        assert frame.labels is not None
        real = ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
        semantic = np.asarray(frame.labels.semantic)
        ground = real & np.isin(semantic, SUPPORT_POOL_SEMANTICS)
        rotation, translation = _pose(frame)
        point_parts.append(
            np.asarray(frame.xyzi[ground, :3], dtype=np.float64) @ rotation.T
            + translation
        )
        semantic_parts.append(semantic[ground])
    points = np.concatenate(point_parts)
    semantics = np.concatenate(semantic_parts)
    anchor_real = ~np.asarray(anchor.zero_slot_mask, dtype=np.bool_)
    selected = anchor_real & np.isin(anchor.labels.semantic, SUPPORT_POOL_SEMANTICS)
    slots = np.flatnonzero(selected).astype(np.int32)
    rotation, translation = _pose(anchor)
    anchors = np.asarray(anchor.xyzi[slots, :3], dtype=np.float64) @ rotation.T + translation
    ranges = np.linalg.norm(np.asarray(anchor.xyzi[slots, :3], dtype=np.float64), axis=1)
    anchor_semantics = np.asarray(anchor.labels.semantic[slots], dtype=np.uint16)
    cell_x = np.floor(anchors[:, 0] / 0.5).astype(np.int64)
    cell_y = np.floor(anchors[:, 1] / 0.5).astype(np.int64)
    packed = (np.uint64(anchor_frame) << np.uint64(32)) | slots.astype(np.uint64)
    hashes = _splitmix64(
        packed ^ (anchor_semantics.astype(np.uint64) << np.uint64(48))
    )
    order = np.lexsort((slots, hashes, cell_y, cell_x, anchor_semantics))
    groups = np.column_stack((anchor_semantics[order], cell_x[order], cell_y[order]))
    keep = np.ones(order.size, dtype=np.bool_)
    keep[1:] = np.any(groups[1:] != groups[:-1], axis=1)
    retained = order[keep]
    trees = {
        semantic: cKDTree(points[semantics == semantic, :2], compact_nodes=True)
        for semantic in SUPPORT_POOL_SEMANTICS
    }
    semantic_points = {
        semantic: points[semantics == semantic] for semantic in SUPPORT_POOL_SEMANTICS
    }
    output: dict[str, list[object]] = {
        name: []
        for name in (
            "semantic",
            "frame",
            "slot",
            "range_m",
            "selection_hash",
            "anchor_world",
            "normal",
            "offset",
        )
    }
    for index in retained:
        semantic = int(anchor_semantics[index])
        radius = float(np.clip(ranges[index] / 20.0, 1.0, 3.0))
        neighbours = trees[semantic].query_ball_point(anchors[index, :2], 1.25 * radius)
        local = semantic_points[semantic][np.asarray(neighbours, dtype=np.int64)]
        plane = _qualified_plane(local, anchors[index], radius)
        if plane is None:
            continue
        normal, offset = plane
        output["semantic"].append(semantic)
        output["frame"].append(anchor_frame)
        output["slot"].append(int(slots[index]))
        output["range_m"].append(float(ranges[index]))
        output["selection_hash"].append(int(hashes[index]))
        output["anchor_world"].append(anchors[index])
        output["normal"].append(normal)
        output["offset"].append(offset)
    count = len(output["frame"])
    return {
        "semantic": np.asarray(output["semantic"], dtype=np.uint16),
        "frame": np.asarray(output["frame"], dtype=np.int32),
        "slot": np.asarray(output["slot"], dtype=np.int32),
        "range_m": np.asarray(output["range_m"], dtype=np.float64),
        "selection_hash": np.asarray(output["selection_hash"], dtype=np.uint64),
        "anchor_world": np.asarray(output["anchor_world"], dtype=np.float64).reshape(count, 3),
        "normal": np.asarray(output["normal"], dtype=np.float64).reshape(count, 3),
        "offset": np.asarray(output["offset"], dtype=np.float64),
    }


def _build_support_pool(
    sequence: STUSequence,
    output_path: Path,
    *,
    processes: int,
    source_frames: tuple[int, int],
    anchor_frames: tuple[int, int],
    experiment: str,
    split_rule: str,
) -> dict[str, object]:
    if processes < 1:
        raise PreparationError("processes must be positive")
    global _SUPPORT_SEQUENCE
    _SUPPORT_SEQUENCE = sequence
    anchors = tuple(range(anchor_frames[0], anchor_frames[1] + 1))
    with mp.get_context("fork").Pool(processes=processes) as workers:
        records = workers.map(_support_anchor, anchors, chunksize=1)
    arrays = {
        name: np.concatenate([record[name] for record in records])
        for name in records[0]
    }
    count = int(arrays["frame"].size)
    metadata: dict[str, object] = {
        "experiment": experiment,
        "passed": count > 0,
        "source_sequence": f"train/{sequence.spec.sequence_id}",
        "source_frames": list(source_frames),
        "anchor_frames": list(anchor_frames),
        "pool_size": count,
        "semantic_counts": {
            str(semantic): int(np.count_nonzero(arrays["semantic"] == semantic))
            for semantic in SUPPORT_POOL_SEMANTICS
        },
        "covered_anchor_frames": int(np.unique(arrays["frame"]).size),
        "scientific_array_hash": _scientific_array_hash(arrays),
        "estimator": "E21-v4 unchanged three-scale trimmed-SVD",
        "split_rule": split_rule,
    }
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    _save_npz(output_path, payload)
    return metadata


def build_validation_support_pool(
    sequence: STUSequence, output_path: Path, *, processes: int
) -> dict[str, object]:
    """Build patches whose context remains inside complete train/201."""

    return _build_support_pool(
        sequence,
        output_path,
        processes=processes,
        source_frames=VALIDATION_SOURCE_FRAMES,
        anchor_frames=VALIDATION_ANCHOR_FRAMES,
        experiment="schema34-validation-support-pool",
        split_rule="every estimator context frame is inside train/201 frames 0-681",
    )


def build_training_support_pool(
    sequence: STUSequence, output_path: Path, *, processes: int
) -> dict[str, object]:
    """Rebuild the inherited full train/206 support pool."""

    return _build_support_pool(
        sequence,
        output_path,
        processes=processes,
        source_frames=TRAINING_SOURCE_FRAMES,
        anchor_frames=TRAINING_ANCHOR_FRAMES,
        experiment="schema33-training-support-pool",
        split_rule="every estimator context frame is inside train/206 frames 0-448",
    )


def build_calibration(
    data_root: Path, output_path: Path, *, protocol: AJAEProtocol
) -> None:
    record = protocol.artifacts["sensor_calibration"]
    source = (protocol.path.parent / str(record["source_file"])).resolve(strict=True)
    if _sha256(source) != str(record["source_sha256"]):
        raise PreparationError("ray-grid calibration source differs from protocol")
    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    ray_grid = calibrated_ray_grid_from_e11(source)
    sensor = calibrate_sensor(
        (sequence.source_frame(frame) for frame in range(449)),
        ray_grid,
        provenance={
            "authority": "AJAE数据与五帧监督协议v2.md",
            "algorithm_origin_schema": 30,
            "runtime_protocol_schema": 34,
        },
    )
    save_sensor_calibration(output_path, ray_grid, sensor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild AJAE frozen runtime inputs")
    parser.add_argument(
        "target",
        choices=(
            "calibration",
            "support-validation",
            "support-training",
            "support",
            "all",
        ),
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    parser.add_argument("--processes", type=int, default=min(24, os.cpu_count() or 1))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.target in {"calibration", "all"}:
        output = protocol.sensor_calibration_path()
        build_calibration(args.data_root, output, protocol=protocol)
        expected = str(protocol.artifacts["sensor_calibration"]["sha256"])
        if _sha256(output) != expected:
            raise PreparationError("rebuilt sensor calibration differs from protocol")
        print(json.dumps({"calibration": str(output), "sha256": expected}))
    plans = (
        (
            201,
            "support-validation",
            build_validation_support_pool,
        ),
        (
            206,
            "support-training",
            build_training_support_pool,
        ),
    )
    for sequence_id, target, builder in plans:
        if args.target not in {target, "support", "all"}:
            continue
        sequence = STUSequence.open(
            args.data_root,
            protocol=protocol,
            partition="train",
            sequence_id=sequence_id,
            label_mode=LabelMode.REQUIRED,
        )
        output = protocol.support_pool_path(sequence_id)
        metadata = builder(sequence, output, processes=args.processes)
        digest = _sha256(output)
        expected = protocol.artifacts["qualified_support_pools"][
            f"train/{sequence_id}"
        ]["sha256"]
        if expected is not None and digest != expected:
            raise PreparationError(
                f"rebuilt support pool has sha256 {digest}, expected {expected}"
            )
        print(json.dumps({"support": str(output), "sha256": digest, **metadata}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
