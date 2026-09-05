#!/usr/bin/env python3
"""Frozen five-frame partitions and sparse synthetic-segment persistence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

try:
    from .protocol import AJAEProtocol, SyntheticPoolSpec, load_protocol
    from .render import (
        RenderedSegment,
        WorldGenerationReport,
        WorldSpec,
        collect_observed_obstacle_index,
        load_qualified_support_pool,
        load_sensor_calibration,
        rendered_segment_identity,
        rendered_window_identity,
        sample_segment_world,
        source_observation_identity,
        world_content_identity,
    )
    from .scene import (
        LabelMode,
        PointLabels,
        SceneWindow,
        SourceFrame,
        STUSequence,
        assemble_window,
        make_source_frame,
    )
except ImportError:  # Direct script execution.
    from protocol import AJAEProtocol, SyntheticPoolSpec, load_protocol
    from render import (  # type: ignore[no-redef]
        RenderedSegment,
        WorldGenerationReport,
        WorldSpec,
        collect_observed_obstacle_index,
        load_qualified_support_pool,
        load_sensor_calibration,
        rendered_segment_identity,
        rendered_window_identity,
        sample_segment_world,
        source_observation_identity,
        world_content_identity,
    )
    from scene import (  # type: ignore[no-redef]
        LabelMode,
        PointLabels,
        SceneWindow,
        SourceFrame,
        STUSequence,
        assemble_window,
        make_source_frame,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPARSE_SEGMENT_FORMAT = "ajae-sparse-rendered-segment-v1"
POOL_MANIFEST_FORMAT = "ajae-synthetic-pool-manifest-v1"
PREDICTION_FORMAT = "ajae-complete-window-point-prediction-v1"


class DataProtocolError(ValueError):
    """Report data that contradicts the frozen five-frame contract."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_hash(
    metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(b"AJAE-sparse-rendered-content-v1\0")
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _stable_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.ascontiguousarray(arrays[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = 9
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class WindowPartition:
    """Expose only causal windows whose output times lie in one inclusive range."""

    sequence: STUSequence
    output_start: int
    output_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, STUSequence):
            raise TypeError("sequence must be an STUSequence")
        if type(self.output_start) is not int or type(self.output_end) is not int:
            raise TypeError("output bounds must be integers")
        if self.output_start < 4 or self.output_end < self.output_start:
            raise DataProtocolError("output bounds cannot define causal five frames")
        legal = frozenset(self.sequence.window_starts)
        starts = tuple(range(self.output_start - 4, self.output_end - 3))
        if not starts or any(start not in legal for start in starts):
            raise DataProtocolError("partition output bounds leave the sequence role")

    @property
    def window_starts(self) -> tuple[int, ...]:
        return tuple(range(self.output_start - 4, self.output_end - 3))

    def __len__(self) -> int:
        return self.output_end - self.output_start + 1

    def __iter__(self) -> Iterator[SceneWindow]:
        for start in self.window_starts:
            yield self.sequence.window(start)

    def for_output(self, frame_id: int) -> SceneWindow:
        if (
            type(frame_id) is not int
            or not self.output_start <= frame_id <= self.output_end
        ):
            raise IndexError(frame_id)
        return self.sequence.window(frame_id - 4)


@dataclass(frozen=True, slots=True)
class PredictionBatch:
    """All point scores from one window, including historical-frame outputs."""

    observation_sequence_id: str
    window_current_frame: int
    source_frame: np.ndarray
    source_slot: np.ndarray
    anomaly_score: np.ndarray

    def __post_init__(self) -> None:
        count = int(self.source_frame.size)
        if (
            not isinstance(self.observation_sequence_id, str)
            or not self.observation_sequence_id
        ):
            raise TypeError("observation_sequence_id must be a non-empty string")
        if type(self.window_current_frame) is not int or self.window_current_frame < 4:
            raise TypeError("window_current_frame must be an integer >= 4")
        if self.source_frame.dtype != np.int32 or self.source_frame.shape != (count,):
            raise TypeError("source_frame must be int32[M]")
        if self.source_slot.dtype != np.int32 or self.source_slot.shape != (count,):
            raise TypeError("source_slot must be int32[M]")
        if self.anomaly_score.dtype != np.float32 or self.anomaly_score.shape != (
            count,
        ):
            raise TypeError("anomaly_score must be float32[M]")
        if not np.isfinite(self.anomaly_score).all():
            raise DataProtocolError("anomaly scores must be finite")
        if (
            np.any(self.source_frame < self.window_current_frame - 4)
            or np.any(self.source_frame > self.window_current_frame)
            or np.any(self.source_slot < 0)
        ):
            raise DataProtocolError("prediction point identities leave their window")
        for array in (self.source_frame, self.source_slot, self.anomaly_score):
            array.setflags(write=False)

    @classmethod
    def from_window(cls, window: SceneWindow, scores: np.ndarray) -> "PredictionBatch":
        values = np.asarray(scores, dtype=np.float32)
        if values.shape != (window.points.count,):
            raise DataProtocolError("a model must score every point in the window")
        return cls(
            window.observation_sequence_id,
            window.current_frame_id,
            window.points.source_frame.copy(),
            window.points.source_slot.copy(),
            values.copy(),
        )

    @property
    def online_mask(self) -> np.ndarray:
        result = self.source_frame == self.window_current_frame
        result.setflags(write=False)
        return result

    def save(self, path: Path) -> dict[str, object]:
        """Persist every point score; no historical observation may be discarded."""

        arrays = {
            "source_frame": self.source_frame,
            "source_slot": self.source_slot,
            "anomaly_score": self.anomaly_score,
        }
        metadata: dict[str, object] = {
            "format": PREDICTION_FORMAT,
            "synthetic_or_raw_sequence_id": self.observation_sequence_id,
            "window_current_frame": self.window_current_frame,
            "point_count": int(self.anomaly_score.size),
        }
        metadata["content_hash"] = _prediction_content_hash(metadata, arrays)
        payload = dict(arrays)
        payload["metadata_json"] = np.asarray(
            json.dumps(
                metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        _stable_npz(path, payload)
        return {
            "file": path.as_posix(),
            "file_sha256": _sha256(path),
            "content_hash": metadata["content_hash"],
            "point_count": metadata["point_count"],
        }

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> "PredictionBatch":
        """Load one complete-window prediction after content verification."""

        resolved = path.expanduser().resolve(strict=True)
        if expected_sha256 is not None and _sha256(resolved) != expected_sha256:
            raise DataProtocolError("prediction file hash differs")
        with np.load(resolved, allow_pickle=False) as payload:
            if set(payload.files) != {
                "source_frame",
                "source_slot",
                "anomaly_score",
                "metadata_json",
            }:
                raise DataProtocolError("prediction file has unexpected arrays")
            arrays = {
                name: np.asarray(payload[name]).copy()
                for name in ("source_frame", "source_slot", "anomaly_score")
            }
            metadata = json.loads(str(payload["metadata_json"].item()))
        content_hash = metadata.pop("content_hash", None)
        if (
            set(metadata)
            != {
                "format",
                "synthetic_or_raw_sequence_id",
                "window_current_frame",
                "point_count",
            }
            or metadata.get("format") != PREDICTION_FORMAT
            or not isinstance(metadata.get("synthetic_or_raw_sequence_id"), str)
            or not metadata["synthetic_or_raw_sequence_id"]
            or type(metadata.get("window_current_frame")) is not int
            or metadata.get("point_count") != int(arrays["anomaly_score"].size)
            or content_hash != _prediction_content_hash(metadata, arrays)
        ):
            raise DataProtocolError("prediction metadata or content hash differs")
        return cls(
            str(metadata["synthetic_or_raw_sequence_id"]),
            int(metadata["window_current_frame"]),
            arrays["source_frame"],
            arrays["source_slot"],
            arrays["anomaly_score"],
        )


def _prediction_content_hash(
    metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(b"AJAE-complete-window-point-prediction-v1\0")
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in ("source_frame", "source_slot", "anomaly_score"):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_sparse_segment(
    path: Path,
    segment: RenderedSegment,
    raw_sources: Sequence[SourceFrame],
    *,
    pool_name: str,
    synthetic_sequence_id: str,
    synthetic_sequence_index: int,
    segment_index: int,
) -> dict[str, object]:
    """Persist only anomaly-replaced slots; all other bytes remain official data."""

    raw = tuple(sorted(tuple(raw_sources), key=lambda item: item.frame_id))
    if tuple(item.frame_id for item in raw) != segment.frame_ids:
        raise DataProtocolError("raw and rendered segment frames do not align")
    offsets = [0]
    slot_parts: list[np.ndarray] = []
    xyzi_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    object_parts: list[np.ndarray] = []
    raw_identities: list[str] = []
    visible_counts: list[int] = []
    anomaly_counts: list[int] = []
    for source, rendered in zip(raw, segment.rendered_frames, strict=True):
        if source.labels is None:
            raise DataProtocolError("formal synthetic sources require labels")
        changed = np.asarray(rendered.inserted_mask, dtype=np.bool_)
        slots = np.flatnonzero(changed).astype(np.int32)
        if not np.array_equal(rendered.source.xyzi[~changed], source.xyzi[~changed]):
            raise DataProtocolError("the renderer changed a retained source return")
        if not np.array_equal(
            rendered.packed_labels[~changed], source.labels.packed[~changed]
        ):
            raise DataProtocolError("the renderer changed a retained source label")
        if np.any(
            np.all(rendered.source.xyzi[changed] == source.xyzi[changed], axis=1)
        ):
            raise DataProtocolError("an inserted slot has no observation change")
        slot_parts.append(slots)
        xyzi_parts.append(rendered.source.xyzi[slots].copy())
        label_parts.append(rendered.packed_labels[slots].copy())
        object_parts.append(rendered.object_id_internal[slots].copy())
        offsets.append(offsets[-1] + slots.size)
        raw_identities.append(source_observation_identity(source))
        visible_counts.append(rendered.source.real_count)
        anomaly_counts.append(int(slots.size))
    if offsets[-1] == 0:
        raise DataProtocolError("a formal anomaly world produced no visible return")

    arrays = {
        "frame_ids": np.asarray(segment.frame_ids, dtype=np.int32),
        "frame_offsets": np.asarray(offsets, dtype=np.int64),
        "changed_slots": np.concatenate(slot_parts).astype(np.int32, copy=False),
        "changed_xyzi": np.concatenate(xyzi_parts).astype(np.float32, copy=False),
        "changed_packed_labels": np.concatenate(label_parts).astype(
            np.uint32, copy=False
        ),
        "changed_object_ids": np.concatenate(object_parts).astype(np.int32, copy=False),
    }
    metadata: dict[str, object] = {
        "format": SPARSE_SEGMENT_FORMAT,
        "pool_name": pool_name,
        "synthetic_sequence_id": synthetic_sequence_id,
        "synthetic_sequence_index": synthetic_sequence_index,
        "segment_index": segment_index,
        "source_sequence_id": segment.world.source_sequence_id,
        "segment_identity": segment.identity,
        "segment_boundary_inclusive": [segment.frame_ids[0], segment.frame_ids[-1]],
        "seed": segment.world.seed,
        "world_identity": segment.world.identity,
        "world_content_identity": world_content_identity(segment.world),
        "world": segment.world.to_dict(),
        "world_generation_report": segment.report.to_dict(),
        "renderer_identity": segment.renderer_identity,
        "raw_source_identities": raw_identities,
        "rendered_source_identities": list(segment.source_observation_identities),
        "window_identities": [item.identity for item in segment.windows],
        "window_starts": [item.window_start for item in segment.windows],
        "visible_point_counts": visible_counts,
        "anomaly_return_counts": anomaly_counts,
        "changed_slot_count": offsets[-1],
    }
    metadata["scientific_content_hash"] = _canonical_hash(metadata, arrays)
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    _stable_npz(path, payload)
    return {
        "file": path.as_posix(),
        "file_sha256": _sha256(path),
        "scientific_content_hash": metadata["scientific_content_hash"],
        "synthetic_sequence_id": synthetic_sequence_id,
        "synthetic_sequence_index": synthetic_sequence_index,
        "segment_index": segment_index,
        "seed": segment.world.seed,
        "world_identity": segment.world.identity,
        "world_content_identity": world_content_identity(segment.world),
        "frame_range_inclusive": [segment.frame_ids[0], segment.frame_ids[-1]],
        "window_count": len(segment.windows),
        "changed_slot_count": offsets[-1],
    }


@dataclass(slots=True)
class FrozenSyntheticSegment:
    """Reconstruct one frozen synthetic segment from raw STU plus sparse deltas."""

    path: Path
    source_sequence: STUSequence
    expected_sha256: str | None = None
    metadata: Mapping[str, object] = field(init=False)
    arrays: Mapping[str, np.ndarray] = field(init=False, repr=False)
    _frame_cache: dict[int, SourceFrame] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.path = self.path.expanduser().resolve(strict=True)
        if not isinstance(self.source_sequence, STUSequence):
            raise TypeError("source_sequence must be an STUSequence")
        if (
            self.expected_sha256 is not None
            and _sha256(self.path) != self.expected_sha256
        ):
            raise DataProtocolError(
                "sparse segment file hash differs from its manifest"
            )
        with np.load(self.path, allow_pickle=False) as payload:
            required = {
                "frame_ids",
                "frame_offsets",
                "changed_slots",
                "changed_xyzi",
                "changed_packed_labels",
                "changed_object_ids",
                "metadata_json",
            }
            if set(payload.files) != required:
                raise DataProtocolError("sparse segment has unexpected arrays")
            arrays = {
                name: np.asarray(payload[name]).copy()
                for name in required - {"metadata_json"}
            }
            metadata = json.loads(str(payload["metadata_json"].item()))
        required_metadata = {
            "format",
            "pool_name",
            "synthetic_sequence_id",
            "synthetic_sequence_index",
            "segment_index",
            "source_sequence_id",
            "segment_identity",
            "segment_boundary_inclusive",
            "seed",
            "world_identity",
            "world_content_identity",
            "world",
            "world_generation_report",
            "renderer_identity",
            "raw_source_identities",
            "rendered_source_identities",
            "window_identities",
            "window_starts",
            "visible_point_counts",
            "anomaly_return_counts",
            "changed_slot_count",
            "scientific_content_hash",
        }
        if (
            not isinstance(metadata, dict)
            or set(metadata) != required_metadata
            or metadata.get("format") != SPARSE_SEGMENT_FORMAT
        ):
            raise DataProtocolError("sparse segment format is unsupported")
        world = WorldSpec.from_dict(metadata["world"])
        report = WorldGenerationReport.from_dict(metadata["world_generation_report"])
        if (
            world.identity != metadata.get("world_identity")
            or world_content_identity(world) != metadata.get("world_content_identity")
            or world.seed != metadata.get("seed")
            or report.world_seed != world.seed
            or report.source_sequence_id != world.source_sequence_id
        ):
            raise DataProtocolError(
                "stored world parameters or report are inconsistent"
            )
        content_hash = metadata.pop("scientific_content_hash", None)
        if content_hash != _canonical_hash(metadata, arrays):
            raise DataProtocolError("sparse segment scientific content hash differs")
        metadata["scientific_content_hash"] = content_hash
        frame_ids = arrays["frame_ids"]
        offsets = arrays["frame_offsets"]
        count = int(arrays["changed_slots"].size)
        if (
            frame_ids.dtype != np.int32
            or frame_ids.ndim != 1
            or frame_ids.size < 5
            or not np.array_equal(
                frame_ids,
                np.arange(frame_ids[0], frame_ids[0] + frame_ids.size, dtype=np.int32),
            )
            or offsets.dtype != np.int64
            or offsets.shape != (frame_ids.size + 1,)
            or offsets[0] != 0
            or offsets[-1] != count
            or np.any(offsets[1:] < offsets[:-1])
            or arrays["changed_slots"].dtype != np.int32
            or arrays["changed_slots"].shape != (count,)
            or arrays["changed_xyzi"].dtype != np.float32
            or arrays["changed_xyzi"].shape != (count, 4)
            or arrays["changed_packed_labels"].dtype != np.uint32
            or arrays["changed_packed_labels"].shape != (count,)
            or arrays["changed_object_ids"].dtype != np.int32
            or arrays["changed_object_ids"].shape != (count,)
            or not np.isfinite(arrays["changed_xyzi"]).all()
            or np.any(
                (arrays["changed_packed_labels"] & np.uint32(0xFFFF)) != np.uint32(2)
            )
            or np.any(arrays["changed_object_ids"] < 1)
        ):
            raise DataProtocolError("sparse segment arrays are misaligned")
        frame_id_tuple = tuple(map(int, frame_ids))
        starts = tuple(map(int, metadata["window_starts"]))
        raw_identities = tuple(metadata["raw_source_identities"])
        rendered_identities = tuple(metadata["rendered_source_identities"])
        window_identities = tuple(metadata["window_identities"])
        visible_counts = tuple(map(int, metadata["visible_point_counts"]))
        anomaly_counts = tuple(map(int, metadata["anomaly_return_counts"]))
        expected_starts = tuple(range(frame_id_tuple[0], frame_id_tuple[-1] - 3))
        digest_lists = raw_identities + rendered_identities + window_identities
        if (
            int(metadata["source_sequence_id"]) != self.source_sequence.spec.sequence_id
            or metadata["segment_boundary_inclusive"]
            != [frame_id_tuple[0], frame_id_tuple[-1]]
            or len(raw_identities) != frame_ids.size
            or len(rendered_identities) != frame_ids.size
            or len(visible_counts) != frame_ids.size
            or len(anomaly_counts) != frame_ids.size
            or starts != expected_starts
            or len(window_identities) != len(starts)
            or any(
                not isinstance(item, str)
                or len(item) != 64
                or any(character not in "0123456789abcdef" for character in item)
                for item in digest_lists
            )
            or int(metadata["changed_slot_count"]) != count
            or anomaly_counts
            != tuple(
                int(offsets[index + 1] - offsets[index])
                for index in range(frame_ids.size)
            )
            or metadata["segment_identity"]
            != rendered_segment_identity(
                str(metadata["world_identity"]),
                frame_id_tuple[0],
                frame_id_tuple,
                str(metadata["renderer_identity"]),
                rendered_identities,
            )
            or window_identities
            != tuple(
                rendered_window_identity(
                    start,
                    tuple(range(start, start + 5)),
                    rendered_identities[index : index + 5],
                )
                for index, start in enumerate(starts)
            )
        ):
            raise DataProtocolError("sparse segment and raw source sequence differ")
        for index in range(frame_ids.size):
            slots = arrays["changed_slots"][offsets[index] : offsets[index + 1]]
            raw = self.source_sequence.source_frame(int(frame_ids[index]))
            if (
                np.any(slots < 0)
                or np.any(slots >= raw.slot_count)
                or (slots.size > 1 and np.any(slots[1:] <= slots[:-1]))
            ):
                raise DataProtocolError(
                    "changed slots are not sorted unique source slots"
                )
            expected_visible = raw.real_count + int(
                np.count_nonzero(raw.zero_slot_mask[slots])
            )
            if visible_counts[index] != expected_visible:
                raise DataProtocolError(
                    "stored visible count differs from the sparse rendered frame"
                )
        self.metadata = metadata
        self.arrays = arrays

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(map(int, self.arrays["frame_ids"]))

    def frame(self, frame_id: int) -> SourceFrame:
        if frame_id in self._frame_cache:
            return self._frame_cache[frame_id]
        try:
            index = self.frame_ids.index(frame_id)
        except ValueError as error:
            raise IndexError(frame_id) from error
        raw = self.source_sequence.source_frame(frame_id)
        raw_identities = self.metadata["raw_source_identities"]
        if source_observation_identity(raw) != raw_identities[index]:
            raise DataProtocolError(
                "raw source observation differs from the frozen pool"
            )
        offsets = self.arrays["frame_offsets"]
        start, stop = int(offsets[index]), int(offsets[index + 1])
        slots = self.arrays["changed_slots"][start:stop]
        xyzi = raw.xyzi.copy()
        xyzi[slots] = self.arrays["changed_xyzi"][start:stop]
        if raw.labels is None:
            raise DataProtocolError("frozen synthetic data requires source labels")
        packed = raw.labels.packed.copy()
        packed[slots] = self.arrays["changed_packed_labels"][start:stop]
        semantic = (packed & np.uint32(0xFFFF)).astype(np.uint16)
        instance = (packed >> np.uint32(16)).astype(np.uint16)
        semantic_target = None
        if raw.labels.semantic_target is not None:
            semantic_target = raw.labels.semantic_target.copy()
            semantic_target[slots] = np.uint8(255)
        labels = PointLabels(packed, semantic, instance, semantic_target)
        result = make_source_frame(
            frame_id,
            xyzi,
            raw.lidar_pose,
            labels,
            partition=raw.partition,
            sequence_id=raw.sequence_id,
        )
        expected = self.metadata["rendered_source_identities"][index]
        if source_observation_identity(result) != expected:
            raise DataProtocolError("reconstructed frame differs from frozen rendering")
        self._frame_cache[frame_id] = result
        return result

    def window(self, window_start: int) -> SceneWindow:
        if window_start not in tuple(self.metadata["window_starts"]):
            raise IndexError(window_start)
        frame_ids = tuple(range(window_start, window_start + 5))
        return assemble_window(
            self.source_sequence.spec,
            window_start,
            frame_ids,
            tuple(self.frame(frame_id) for frame_id in frame_ids),
            observation_sequence_id=str(self.metadata["synthetic_sequence_id"]),
        )

    def __iter__(self) -> Iterator[SceneWindow]:
        for start in self.metadata["window_starts"]:
            yield self.window(int(start))


def generation_identity(protocol: AJAEProtocol, pool: SyntheticPoolSpec) -> str:
    """Bind a pool to its scientific rules, inputs, and rendering source code."""

    support = protocol.artifacts["qualified_support_pools"][
        f"train/{pool.source_sequence_id}"
    ]
    payload = {
        "schema_version": protocol.schema_version,
        "pool": protocol.synthetic_pools[pool.name],
        "source_role": protocol.data[
            "parameter_update_source"
            if pool.source_sequence_id == 206
            else "model_validation_source"
        ],
        "official_train_archive_sha256": protocol.data["official_archive_sha256"][
            "train.zip"
        ],
        "window": protocol.window,
        "labels": protocol.labels,
        "storage": protocol.storage,
        "calibration_sha256": protocol.artifacts["sensor_calibration"]["sha256"],
        "support_pool_sha256": support["sha256"],
        "source_files_sha256": {
            name: _sha256(PROJECT_ROOT / name)
            for name in ("src/data.py", "src/render.py", "src/scene.py")
        },
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda value: dict(value),
        ).encode("utf-8")
    ).hexdigest()


def _pool_spec(protocol: AJAEProtocol, name: str) -> SyntheticPoolSpec:
    if name == "train_v1":
        return protocol.training_pool
    if name == "validation_v1":
        return protocol.validation_pool
    raise KeyError(name)


def _segment_path(
    output_directory: Path, sequence_index: int, segment_index: int
) -> Path:
    return (
        output_directory
        / f"sequence_{sequence_index:03d}"
        / f"segment_{segment_index:02d}.npz"
    )


_GENERATION_STATE: tuple[object, ...] | None = None


def _generate_one(task: tuple[int, int]) -> dict[str, object]:
    """Generate one independent segment from the fork-shared read-only state."""

    if _GENERATION_STATE is None:
        raise RuntimeError("synthetic generation state is not initialized")
    (
        pool_name,
        pool_spec,
        output,
        sequence,
        support,
        obstacles,
        ray_grid,
        sensor,
        renderer_identity,
        resume,
    ) = _GENERATION_STATE
    sequence_index, segment_index = task
    synthetic_id = pool_spec.synthetic_sequence_id(sequence_index)
    path = _segment_path(output, sequence_index, segment_index)
    seed = pool_spec.world_seed(sequence_index, segment_index)
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        frozen = FrozenSyntheticSegment(path, sequence)
        metadata = frozen.metadata
        if (
            metadata["pool_name"] != pool_name
            or metadata["synthetic_sequence_id"] != synthetic_id
            or int(metadata["segment_index"]) != segment_index
            or int(metadata["seed"]) != seed
            or metadata["renderer_identity"] != renderer_identity
        ):
            raise DataProtocolError(
                "an existing segment does not match the requested generation"
            )
        return {
            "file": path.as_posix(),
            "file_sha256": _sha256(path),
            "scientific_content_hash": metadata["scientific_content_hash"],
            "synthetic_sequence_id": synthetic_id,
            "synthetic_sequence_index": sequence_index,
            "segment_index": segment_index,
            "seed": seed,
            "world_identity": metadata["world_identity"],
            "world_content_identity": metadata["world_content_identity"],
            "frame_range_inclusive": [
                frozen.frame_ids[0],
                frozen.frame_ids[-1],
            ],
            "window_count": len(metadata["window_starts"]),
            "changed_slot_count": int(metadata["changed_slot_count"]),
        }
    span = pool_spec.segments[segment_index]
    raw_sources = tuple(
        sequence.source_frame(frame_id) for frame_id in range(span.start, span.stop)
    )
    rendered = sample_segment_world(
        support,
        obstacles,
        raw_sources,
        ray_grid,
        sensor,
        seed,
        renderer_identity=renderer_identity,
    )
    return save_sparse_segment(
        path,
        rendered,
        raw_sources,
        pool_name=pool_name,
        synthetic_sequence_id=synthetic_id,
        synthetic_sequence_index=sequence_index,
        segment_index=segment_index,
    )


def generate_segments(
    data_root: Path,
    protocol: AJAEProtocol,
    pool_name: str,
    *,
    output_directory: Path | None = None,
    sequence_indices: Sequence[int] | None = None,
    segment_indices: Sequence[int] | None = None,
    resume: bool = False,
    workers: int = 1,
) -> list[dict[str, object]]:
    """Generate selected segments without ever resampling their frozen root seeds."""

    pool_spec = _pool_spec(protocol, pool_name)
    output = (
        (protocol.path.parent / pool_spec.output_directory).resolve()
        if output_directory is None
        else output_directory.expanduser().resolve()
    )
    sequences = (
        tuple(range(pool_spec.synthetic_sequence_count))
        if sequence_indices is None
        else tuple(sequence_indices)
    )
    segments = (
        tuple(range(len(pool_spec.segments)))
        if segment_indices is None
        else tuple(segment_indices)
    )
    if (
        not sequences
        or not segments
        or any(
            type(index) is not int
            or not 0 <= index < pool_spec.synthetic_sequence_count
            for index in sequences
        )
        or any(
            type(index) is not int or not 0 <= index < len(pool_spec.segments)
            for index in segments
        )
        or len(set(sequences)) != len(sequences)
        or len(set(segments)) != len(segments)
    ):
        raise DataProtocolError("selected synthetic sequence or segment is invalid")
    if type(workers) is not int or workers < 1:
        raise DataProtocolError("workers must be a positive integer")

    support_record = protocol.artifacts["qualified_support_pools"][
        f"train/{pool_spec.source_sequence_id}"
    ]
    expected_support = support_record["sha256"]
    if expected_support is None:
        raise DataProtocolError("the selected source support pool is not frozen")
    support_path = protocol.verify_support_pool(pool_spec.source_sequence_id)
    support = load_qualified_support_pool(
        support_path,
        source_sequence_id=pool_spec.source_sequence_id,
        expected_sha256=str(expected_support),
    )
    ray_grid, sensor = load_sensor_calibration(protocol.verify_sensor_calibration())
    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=pool_spec.source_sequence_id,
        label_mode=LabelMode.REQUIRED,
    )
    obstacles = collect_observed_obstacle_index(
        (sequence.source_frame(frame_id) for frame_id in range(len(sequence))),
        source_sequence_id=pool_spec.source_sequence_id,
    )
    renderer_identity = generation_identity(protocol, pool_spec)
    tasks = tuple(
        (sequence_index, segment_index)
        for sequence_index in sequences
        for segment_index in segments
    )
    global _GENERATION_STATE
    _GENERATION_STATE = (
        pool_name,
        pool_spec,
        output,
        sequence,
        support,
        obstacles,
        ray_grid,
        sensor,
        renderer_identity,
        resume,
    )
    try:
        if workers == 1 or len(tasks) == 1:
            iterator = map(_generate_one, tasks)
            records = []
            for record in iterator:
                records.append(record)
                print(
                    json.dumps(
                        {
                            "generated": record["file"],
                            "seed": record["seed"],
                            "changed_slots": record["changed_slot_count"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        else:
            count = min(workers, len(tasks))
            records = []
            with mp.get_context("fork").Pool(processes=count) as processes:
                for record in processes.imap_unordered(
                    _generate_one, tasks, chunksize=1
                ):
                    records.append(record)
                    print(
                        json.dumps(
                            {
                                "generated": record["file"],
                                "seed": record["seed"],
                                "changed_slots": record["changed_slot_count"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        _GENERATION_STATE = None
    return sorted(
        records,
        key=lambda item: (
            int(item["synthetic_sequence_index"]),
            int(item["segment_index"]),
        ),
    )


def build_pool_manifest(
    protocol: AJAEProtocol,
    pool_name: str,
    *,
    output_directory: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """Bind every predeclared segment only after the complete pool exists."""

    pool_spec = _pool_spec(protocol, pool_name)
    output = (
        (protocol.path.parent / pool_spec.output_directory).resolve()
        if output_directory is None
        else output_directory.expanduser().resolve()
    )
    records: list[dict[str, object]] = []
    world_identities: set[str] = set()
    world_content_identities: set[str] = set()
    total_windows = 0
    renderer_identity = generation_identity(protocol, pool_spec)
    for sequence_index in range(pool_spec.synthetic_sequence_count):
        for segment_index, span in enumerate(pool_spec.segments):
            path = _segment_path(output, sequence_index, segment_index)
            if not path.is_file():
                raise DataProtocolError(f"formal pool is missing {path}")
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"].item()))
            expected_seed = pool_spec.world_seed(sequence_index, segment_index)
            expected_id = pool_spec.synthetic_sequence_id(sequence_index)
            if (
                metadata.get("pool_name") != pool_name
                or metadata.get("synthetic_sequence_id") != expected_id
                or metadata.get("synthetic_sequence_index") != sequence_index
                or metadata.get("segment_index") != segment_index
                or metadata.get("seed") != expected_seed
                or metadata.get("renderer_identity") != renderer_identity
                or metadata.get("window_starts")
                != list(pool_spec.window_starts(segment_index))
                or metadata.get("source_sequence_id") != pool_spec.source_sequence_id
                or metadata.get("changed_slot_count", 0) < 1
                or metadata.get("world_identity") in world_identities
                or metadata.get("world_content_identity") in world_content_identities
            ):
                raise DataProtocolError("a formal segment contradicts its pool plan")
            world_identities.add(str(metadata["world_identity"]))
            world_content_identities.add(str(metadata["world_content_identity"]))
            total_windows += len(metadata["window_starts"])
            records.append(
                {
                    "file": path.relative_to(protocol.path.parent).as_posix(),
                    "file_sha256": _sha256(path),
                    "scientific_content_hash": metadata["scientific_content_hash"],
                    "synthetic_sequence_id": expected_id,
                    "synthetic_sequence_index": sequence_index,
                    "segment_index": segment_index,
                    "seed": expected_seed,
                    "world_identity": metadata["world_identity"],
                    "world_content_identity": metadata["world_content_identity"],
                    "frame_range_inclusive": [span.start, span.stop - 1],
                    "window_count": len(metadata["window_starts"]),
                    "changed_slot_count": metadata["changed_slot_count"],
                }
            )
    if (
        len(records) != pool_spec.world_count
        or len(world_identities) != pool_spec.world_count
        or len(world_content_identities) != pool_spec.world_count
        or total_windows != pool_spec.total_window_count
    ):
        raise DataProtocolError("formal pool totals contradict the protocol")
    scientific_hash = hashlib.sha256(
        json.dumps(
            {
                "pool_name": pool_name,
                "generation_identity": renderer_identity,
                "segment_scientific_hashes": [
                    item["scientific_content_hash"] for item in records
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest: dict[str, object] = {
        "format": POOL_MANIFEST_FORMAT,
        "schema_version": protocol.schema_version,
        "pool_name": pool_name,
        "generation_identity": renderer_identity,
        "source_sequence_id": pool_spec.source_sequence_id,
        "synthetic_sequence_count": pool_spec.synthetic_sequence_count,
        "world_count": len(records),
        "window_count": total_windows,
        "scientific_content_hash": scientific_hash,
        "segments": records,
    }
    target = (
        protocol.pool_manifest_path(pool_name)
        if manifest_path is None
        else manifest_path.expanduser().resolve()
    )
    _atomic_json(target, manifest)
    return manifest


def _indices(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    try:
        return tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "indices must be comma-separated integers"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate schema-34 sparse synthetic data segments"
    )
    parser.add_argument("action", choices=("generate", "manifest"))
    parser.add_argument("--pool", required=True, choices=("train_v1", "validation_v1"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--sequence-indices")
    parser.add_argument("--segment-indices")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.action == "generate":
        if args.data_root is None:
            raise DataProtocolError("generation requires --data-root")
        records = generate_segments(
            args.data_root,
            protocol,
            args.pool,
            output_directory=args.output_directory,
            sequence_indices=_indices(args.sequence_indices),
            segment_indices=_indices(args.segment_indices),
            resume=args.resume,
            workers=args.workers,
        )
        print(json.dumps({"generated_segments": len(records)}, sort_keys=True))
    else:
        manifest = build_pool_manifest(
            protocol,
            args.pool,
            output_directory=args.output_directory,
            manifest_path=args.manifest_path,
        )
        print(
            json.dumps(
                {
                    "world_count": manifest["world_count"],
                    "window_count": manifest["window_count"],
                    "scientific_content_hash": manifest["scientific_content_hash"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
