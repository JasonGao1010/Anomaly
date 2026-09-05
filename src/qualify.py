#!/usr/bin/env python3
"""Model-independent qualification for the schema-34 frozen data pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from .data import (
        FrozenSyntheticSegment,
        WindowPartition,
        _atomic_json,
        load_pool_manifest,
    )
    from .protocol import AJAEProtocol, SyntheticPoolSpec, load_protocol
    from .render import (
        canonical_ray_slots_for_source,
        collect_observed_obstacle_index,
        load_qualified_support_pool,
        load_sensor_calibration,
        sample_segment_world,
        source_observation_identity,
        world_content_identity,
    )
    from .scene import LabelMode, SceneWindow, STUSequence
except ImportError:  # Direct script execution.
    from data import (  # type: ignore[no-redef]
        FrozenSyntheticSegment,
        WindowPartition,
        _atomic_json,
        load_pool_manifest,
    )
    from protocol import AJAEProtocol, SyntheticPoolSpec, load_protocol
    from render import (  # type: ignore[no-redef]
        canonical_ray_slots_for_source,
        collect_observed_obstacle_index,
        load_qualified_support_pool,
        load_sensor_calibration,
        sample_segment_world,
        source_observation_identity,
        world_content_identity,
    )
    from scene import LabelMode, SceneWindow, STUSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QualificationError(AssertionError):
    """Report a failed model-independent data invariant."""


def _write_window_ply(window: SceneWindow, path: Path) -> int:
    """Export every visible point; colors encode supervision, never features."""

    if window.labels is None:
        raise QualificationError("truth-colored PLY requires point labels")
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    records = np.empty(window.points.count, dtype=dtype)
    for index, name in enumerate(("x", "y", "z")):
        records[name] = window.points.coordinates[:, index]
    # Index 0 is ignore, 1 is normal, and 2 is anomaly.
    palette = np.asarray(((0, 128, 255), (160, 160, 160), (255, 0, 0)), dtype=np.uint8)
    colors = palette[window.labels.anomaly_target + 1]
    for index, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment sequence {window.observation_sequence_id}\n"
        f"comment frames {' '.join(map(str, window.frame_ids))}\n"
        "comment coordinates current_frame_lidar\n"
        "comment truth normal=160,160,160 anomaly=255,0,0 ignore=0,128,255\n"
        f"element vertex {records.size}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    expected_size = len(header) + records.nbytes
    if path.exists():
        expected = hashlib.sha256(header)
        expected.update(records.view(np.uint8))
        if (
            path.stat().st_size != expected_size
            or _sha256(path) != expected.hexdigest()
        ):
            raise QualificationError(f"existing PLY differs from this window: {path}")
        return expected_size
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".ply.tmp")
    with temporary.open("xb") as stream:
        stream.write(header)
        records.tofile(stream)
    temporary.replace(path)
    return expected_size


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _qualify_pool(
    protocol: AJAEProtocol,
    pool: SyntheticPoolSpec,
    manifest: Mapping[str, object],
    source: STUSequence,
    ply_directory: Path | None = None,
) -> dict[str, object]:
    expected_pairs = {
        (sequence, segment)
        for sequence in range(pool.synthetic_sequence_count)
        for segment in range(len(pool.segments))
    }
    observed_pairs: set[tuple[int, int]] = set()
    observed_worlds: set[str] = set()
    observed_world_contents: set[str] = set()
    observed_seeds: set[int] = set()
    output_frames_by_sequence: dict[int, list[int]] = {
        index: [] for index in range(pool.synthetic_sequence_count)
    }
    total_frames = 0
    total_windows = 0
    total_points = 0
    total_supervised = 0
    total_anomalies = 0
    total_objects = 0
    terminal_visible_worlds = 0
    fallback_worlds = 0
    empty_anomaly_windows = 0
    official_eligible_windows = 0
    ply_bytes = 0
    placement_observations: list[dict[str, object]] = []
    for record in manifest["segments"]:
        expected_record_keys = {
            "file",
            "file_sha256",
            "scientific_content_hash",
            "synthetic_sequence_id",
            "synthetic_sequence_index",
            "segment_index",
            "seed",
            "world_identity",
            "world_content_identity",
            "frame_range_inclusive",
            "window_count",
            "changed_slot_count",
        }
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            raise QualificationError("manifest segment record has an invalid schema")
        sequence_index = int(record["synthetic_sequence_index"])
        segment_index = int(record["segment_index"])
        pair = (sequence_index, segment_index)
        if pair in observed_pairs or pair not in expected_pairs:
            raise QualificationError(
                "manifest segment identity is duplicated or invalid"
            )
        observed_pairs.add(pair)
        expected_seed = pool.world_seed(sequence_index, segment_index)
        span = pool.segments[segment_index]
        expected_file = (
            Path(pool.output_directory)
            / f"sequence_{sequence_index:03d}"
            / f"segment_{segment_index:02d}.npz"
        ).as_posix()
        if (
            int(record["seed"]) != expected_seed
            or record["synthetic_sequence_id"]
            != pool.synthetic_sequence_id(sequence_index)
            or record["file"] != expected_file
            or record["frame_range_inclusive"] != [span.start, span.stop - 1]
            or int(record["window_count"]) != len(pool.window_starts(segment_index))
            or int(record["changed_slot_count"]) < 1
        ):
            raise QualificationError(
                "manifest segment differs from the frozen seed, boundary, or identity"
            )
        seed = int(record["seed"])
        world = str(record["world_identity"])
        world_content = str(record["world_content_identity"])
        if (
            seed in observed_seeds
            or world in observed_worlds
            or world_content in observed_world_contents
        ):
            raise QualificationError(
                "formal seeds and physical world contents must be unique"
            )
        observed_seeds.add(seed)
        observed_worlds.add(world)
        observed_world_contents.add(world_content)
        path = (protocol.path.parent / str(record["file"])).resolve(strict=True)
        if _sha256(path) != record["file_sha256"]:
            raise QualificationError("segment file differs from its manifest hash")
        frozen = FrozenSyntheticSegment(path, source, str(record["file_sha256"]))
        metadata = frozen.metadata
        object_count = len(metadata["world"]["objects"])
        if (
            object_count != 1
            or metadata["world_generation_report"]["anomaly_count"] != 1
        ):
            raise QualificationError("a segment must contain exactly one anomaly proxy")
        total_objects += object_count
        if (
            metadata["synthetic_sequence_id"] != record["synthetic_sequence_id"]
            or metadata["synthetic_sequence_index"]
            != record["synthetic_sequence_index"]
            or metadata["segment_index"] != record["segment_index"]
            or metadata["seed"] != record["seed"]
            or metadata["world_identity"] != record["world_identity"]
            or metadata["world_content_identity"] != record["world_content_identity"]
            or metadata["scientific_content_hash"] != record["scientific_content_hash"]
        ):
            raise QualificationError(
                "manifest record and sparse-segment metadata disagree"
            )
        if frozen.frame_ids != tuple(range(span.start, span.stop)):
            raise QualificationError("segment source frames cross a frozen boundary")
        frames = tuple(frozen.frame(frame_id) for frame_id in frozen.frame_ids)
        counts = tuple(map(int, metadata["anomaly_return_counts"]))
        # Match the retained official evaluator's range and per-frame anomaly gate.
        official_counts = []
        for frame in frames:
            anomaly_xyz = frame.xyzi[frame.labels.semantic == 2, :3]
            ranges = np.linalg.norm(anomaly_xyz, axis=1)
            official_counts.append(
                int(np.count_nonzero((ranges >= 2.5) & (ranges <= 50)))
            )
        official_eligible_windows += sum(count >= 5 for count in official_counts[4:])
        mode = metadata["world_generation_report"]["placement_mode"]
        if mode == "terminal_visible" and counts[-1] == 0:
            raise QualificationError("terminal-visible placement lost its final return")
        terminal_visible_worlds += int(counts[-1] > 0)
        fallback_worlds += int(mode == "support_visible_fallback")
        empty_count = sum(sum(counts[i : i + 5]) == 0 for i in range(len(counts) - 4))
        empty_anomaly_windows += empty_count
        center = np.asarray(metadata["world"]["objects"][0]["translation_world_m"])
        distances = np.asarray(
            [np.linalg.norm(center - frame.lidar_pose[:3, 3]) for frame in frames]
        )
        visible = [
            frame.frame_id for frame, count in zip(frames, counts, strict=True) if count
        ]
        placement_observations.append(
            {
                "synthetic_sequence_index": sequence_index,
                "segment_index": segment_index,
                "placement_mode": mode,
                "support_scope": metadata["world_generation_report"]["support_scope"],
                "first_visible_frame": visible[0],
                "last_visible_frame": visible[-1],
                "terminal_anomaly_return_count": counts[-1],
                "terminal_official_anomaly_return_count": official_counts[-1],
                "official_eligible_window_count": sum(
                    count >= 5 for count in official_counts[4:]
                ),
                "first_object_center_distance_m": float(distances[0]),
                "terminal_object_center_distance_m": float(distances[-1]),
                "closest_object_center_frame": frames[
                    int(np.argmin(distances))
                ].frame_id,
                "empty_anomaly_window_count": empty_count,
            }
        )
        total_frames += len(frames)
        first_object_by_frame = {frame.frame_id: frame for frame in frames}
        starts = pool.window_starts(segment_index)
        if tuple(map(int, frozen.metadata["window_starts"])) != starts:
            raise QualificationError("segment windows differ from the frozen plan")
        for start in starts:
            window = frozen.window(start)
            if window.frame_ids != tuple(range(start, start + 5)):
                raise QualificationError("a window is not five consecutive frames")
            if not span.contains(start) or not span.contains(start + 4):
                raise QualificationError("a window crosses an anomaly-world boundary")
            if any(
                item.source is not first_object_by_frame[item.source.frame_id]
                for item in window.frames
            ):
                raise QualificationError(
                    "overlapping windows do not reuse the same reconstructed frame"
                )
            current = window.current_frame.source
            if not np.array_equal(
                window.points.coordinates[window.current_mask],
                current.xyzi[current.real_slots, :3],
            ):
                raise QualificationError("current frame is not a bitwise xyz copy")
            if not np.array_equal(
                window.current_mask,
                window.points.source_frame == window.current_frame_id,
            ):
                raise QualificationError("current_mask is not defined by source frame")
            if window.labels is None or not np.array_equal(
                window.supervision_mask, window.labels.anomaly_target != -1
            ):
                raise QualificationError("three-state labels and loss mask disagree")
            if window.points.count != sum(
                item.source.real_count for item in window.frames
            ):
                raise QualificationError(
                    "a window omitted or duplicated visible returns"
                )
            total_windows += 1
            total_points += window.points.count
            total_supervised += int(window.supervision_mask.sum())
            total_anomalies += int(np.count_nonzero(window.labels.anomaly_target == 1))
            output_frames_by_sequence[sequence_index].append(window.current_frame_id)
            if ply_directory is not None:
                path = (
                    ply_directory
                    / pool.name
                    / f"sequence_{sequence_index:03d}"
                    / f"window_{window.current_frame_id:06d}.ply"
                )
                ply_bytes += _write_window_ply(window, path)
    if observed_pairs != expected_pairs:
        raise QualificationError("formal manifest omits a predeclared segment")
    if total_windows != pool.total_window_count or total_anomalies < 1:
        raise QualificationError(
            "formal pool has the wrong windows or no anomaly labels"
        )
    for sequence_index, outputs in output_frames_by_sequence.items():
        expected = [
            start + 4
            for segment in range(len(pool.segments))
            for start in pool.window_starts(segment)
        ]
        if outputs != expected or len(outputs) != len(set(outputs)):
            raise QualificationError(
                f"synthetic sequence {sequence_index} has invalid online outputs"
            )
    return {
        "generation_identity": manifest["generation_identity"],
        "world_count": len(observed_worlds),
        "anomaly_object_count": total_objects,
        "terminal_visible_world_count": terminal_visible_worlds,
        "support_visible_fallback_world_count": fallback_worlds,
        "empty_anomaly_window_count": empty_anomaly_windows,
        "official_eligible_window_count": official_eligible_windows,
        "ply_bytes": ply_bytes,
        "placement_observations": placement_observations,
        "rendered_frame_count": total_frames,
        "window_count": total_windows,
        "point_observation_count": total_points,
        "supervised_point_observation_count": total_supervised,
        "anomaly_point_observation_count": total_anomalies,
        "distinct_seed_count": len(observed_seeds),
        "distinct_world_count": len(observed_worlds),
        "distinct_physical_world_count": len(observed_world_contents),
    }


def _repeat_first_training_segment(
    protocol: AJAEProtocol,
    manifest: Mapping[str, object],
    sequence: STUSequence,
) -> dict[str, object]:
    pool = protocol.training_pool
    first = manifest["segments"][0]
    if int(first["synthetic_sequence_index"]) != 0 or int(first["segment_index"]) != 0:
        raise QualificationError("training manifest order is not canonical")
    support_record = protocol.artifacts["qualified_support_pools"]["train/206"]
    support = load_qualified_support_pool(
        protocol.verify_support_pool(206),
        source_sequence_id=206,
        expected_sha256=str(support_record["sha256"]),
    )
    ray_grid, sensor = load_sensor_calibration(protocol.verify_sensor_calibration())
    obstacles = collect_observed_obstacle_index(
        (sequence.source_frame(frame_id) for frame_id in range(len(sequence))),
        source_sequence_id=206,
    )
    span = pool.segments[0]
    sources = tuple(
        sequence.source_frame(frame_id) for frame_id in range(span.start, span.stop)
    )
    repeated = sample_segment_world(
        support,
        obstacles,
        sources,
        ray_grid,
        sensor,
        pool.world_seed(0, 0),
        renderer_identity=str(manifest["generation_identity"]),
    )
    path = (protocol.path.parent / str(first["file"])).resolve(strict=True)
    frozen = FrozenSyntheticSegment(path, sequence, str(first["file_sha256"]))
    if (
        repeated.world.identity != first["world_identity"]
        or world_content_identity(repeated.world) != first["world_content_identity"]
        or tuple(repeated.source_observation_identities)
        != tuple(frozen.metadata["rendered_source_identities"])
        or [int(item.inserted_mask.sum()) for item in repeated.rendered_frames]
        != list(frozen.metadata["anomaly_return_counts"])
    ):
        raise QualificationError("same seed did not reproduce the frozen segment")
    return {
        "seed": pool.world_seed(0, 0),
        "world_identity": repeated.world.identity,
        "rendered_frame_identities_exact": True,
        "anomaly_return_counts_exact": True,
    }


def _qualify_duplicate_prefix(
    protocol: AJAEProtocol,
    sequence: STUSequence,
) -> dict[str, object]:
    """Verify and report every retained file slot in train/201 frames 0 through 3."""

    ray_grid, _ = load_sensor_calibration(protocol.verify_sensor_calibration())
    records: dict[str, object] = {}
    for frame_id in range(4):
        source = sequence.source_frame(frame_id)
        mapping = canonical_ray_slots_for_source(source, ray_grid)
        multiplicity = np.bincount(mapping, minlength=ray_grid.slot_count)
        if (
            mapping.shape != (source.slot_count,)
            or np.count_nonzero(multiplicity) != ray_grid.slot_count
        ):
            raise QualificationError(
                "duplicate-prefix mapping does not retain every file slot and ray"
            )
        records[str(frame_id)] = {
            "file_slot_count": source.slot_count,
            "visible_return_count": source.real_count,
            "canonical_ray_count": ray_grid.slot_count,
            "minimum_ray_multiplicity": int(multiplicity.min()),
            "maximum_ray_multiplicity": int(multiplicity.max()),
            "source_observation_identity": source_observation_identity(source),
        }
    return {
        "frame_ids": [0, 1, 2, 3],
        "all_file_slots_retained": True,
        "exact_xyzi_and_label_repetition_verified": True,
        "frames": records,
    }


def qualify_data(
    data_root: Path,
    protocol: AJAEProtocol,
    *,
    output_path: Path | None = None,
    ply_directory: Path | None = None,
) -> dict[str, object]:
    """Run every schema-34 qualification check without loading a model."""

    frozen_target = (
        protocol.path.parent / str(protocol.artifacts["qualification"]["file"])
    ).resolve()
    target = None if output_path is None else output_path.expanduser().resolve()
    if protocol.status["data_pool_frozen"]:
        if target == frozen_target:
            raise QualificationError(
                "rechecks cannot replace frozen qualification evidence"
            )
    elif target is None:
        target = frozen_target
    if target is not None and target.exists():
        raise FileExistsError(target)

    train_manifest_path = protocol.pool_manifest_path("train")
    validation_manifest_path = protocol.pool_manifest_path("validation")
    train_manifest = load_pool_manifest(protocol, protocol.training_pool)
    validation_manifest = load_pool_manifest(protocol, protocol.validation_pool)
    train_sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    validation_sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    if train_sequence.frame_ids != tuple(range(449)):
        raise QualificationError("train/206 is not exactly frames 0 through 448")
    if validation_sequence.frame_ids != tuple(range(682)):
        raise QualificationError("train/201 is not exactly frames 0 through 681")
    duplicate_prefix = _qualify_duplicate_prefix(protocol, validation_sequence)

    # Each pool has its own source cache; point order and reductions stay serial.
    with ThreadPoolExecutor(max_workers=2) as executor:
        train_future = executor.submit(
            _qualify_pool,
            protocol,
            protocol.training_pool,
            train_manifest,
            train_sequence,
            ply_directory,
        )
        validation_future = executor.submit(
            _qualify_pool,
            protocol,
            protocol.validation_pool,
            validation_manifest,
            validation_sequence,
            ply_directory,
        )
        train_result = train_future.result()
        validation_result = validation_future.result()
    normal_outputs: list[int] = []
    normal_points = 0
    for window in WindowPartition(validation_sequence, 4, 681):
        current = window.current_frame.source
        if not np.array_equal(
            window.points.coordinates[window.current_mask],
            current.xyzi[current.real_slots, :3],
        ):
            raise QualificationError("normal 201 current xyz is not a bitwise copy")
        normal_outputs.append(window.current_frame_id)
        normal_points += window.points.count
    if normal_outputs != list(range(4, 682)):
        raise QualificationError("normal 201 does not have 678 unique online outputs")

    determinism = _repeat_first_training_segment(
        protocol, train_manifest, train_sequence
    )
    checks = {name: True for name in protocol.qualification["required_checks"]}
    result: dict[str, object] = {
        "format": "ajae-schema34-data-qualification",
        "schema_version": protocol.schema_version,
        "model_independent": True,
        "passed": all(checks.values()),
        "checks": checks,
        "inputs": {
            "train_manifest": train_manifest_path.relative_to(
                protocol.path.parent
            ).as_posix(),
            "train_manifest_sha256": _sha256(train_manifest_path),
            "validation_manifest": validation_manifest_path.relative_to(
                protocol.path.parent
            ).as_posix(),
            "validation_manifest_sha256": _sha256(validation_manifest_path),
        },
        "train_pool": train_result,
        "synthetic_validation_pool": validation_result,
        "normal_validation": {
            "source_frame_count": len(validation_sequence),
            "window_count": len(normal_outputs),
            "output_frame_range_inclusive": [
                normal_outputs[0],
                normal_outputs[-1],
            ],
            "point_observation_count": normal_points,
            "known_duplicate_prefix_frames_retained": [0, 1, 2, 3],
            "duplicate_prefix": duplicate_prefix,
        },
        "determinism_repeat": determinism,
        "visualizations": None
        if ply_directory is None
        else {
            "directory": ply_directory.resolve()
            .relative_to(protocol.path.parent)
            .as_posix(),
            "format": "binary_little_endian_ply",
            "window_count": train_result["window_count"]
            + validation_result["window_count"],
            "bytes": train_result["ply_bytes"] + validation_result["ply_bytes"],
            "all_visible_points_retained": True,
            "truth_colors": {
                "normal": [160, 160, 160],
                "anomaly": [255, 0, 0],
                "ignore": [0, 128, 255],
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_files_sha256": {
            name: _sha256(PROJECT_ROOT / name)
            for name in (
                "src/data.py",
                "src/protocol.py",
                "src/qualify.py",
                "src/render.py",
                "src/scene.py",
            )
        },
    }
    # Frozen rechecks are read-only unless the caller chooses a new report path.
    if target is not None:
        _atomic_json(target, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify the complete schema-34 data pools"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="new report path; frozen rechecks default to no write",
    )
    parser.add_argument("--ply-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = qualify_data(
        args.data_root,
        load_protocol(args.protocol),
        output_path=args.output,
        ply_directory=args.ply_directory,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "train_windows": result["train_pool"]["window_count"],
                "synthetic_validation_windows": result["synthetic_validation_pool"][
                    "window_count"
                ],
                "normal_validation_windows": result["normal_validation"][
                    "window_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
