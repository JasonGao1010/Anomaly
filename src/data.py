#!/usr/bin/env python3
"""Frozen five-frame partitions and sparse synthetic-segment persistence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import tempfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

try:
    from .protocol import AJAEProtocol, SyntheticPoolSpec, load_protocol
    from .render import (
        RenderedSegment,
        render_frame,
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
        render_frame,
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
SPARSE_SEGMENT_FORMAT = "ajae-sparse-rendered-segment"
POOL_MANIFEST_FORMAT = "ajae-synthetic-pool-manifest"
PREDICTION_FORMAT = "ajae-complete-window-point-prediction"


class DataProtocolError(ValueError):
    """Report data that contradicts the frozen five-frame contract."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_hash(
    metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(b"AJAE-sparse-rendered-content\0")
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


def _stable_npz(
    path: Path, arrays: Mapping[str, np.ndarray], *, compression_level: int = 9
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp") as temporary:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compression_level,
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
                info._compresslevel = compression_level
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        temporary.flush()
        # Publish atomically without replacing an existing observation or prediction.
        os.link(temporary.name, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp"
    ) as temporary:
        temporary.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary.flush()
        os.link(temporary.name, path)


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
    score_kind: str = "probability"

    def __post_init__(self) -> None:
        if self.score_kind not in ("probability", "logit"):
            raise DataProtocolError("unknown anomaly score kind")
        count = int(self.source_frame.size)
        if (
            not isinstance(self.observation_sequence_id, str)
            or not self.observation_sequence_id
        ):
            raise TypeError("observation_sequence_id must be a non-empty string")
        if type(self.window_current_frame) is not int or self.window_current_frame < 0:
            raise TypeError("window_current_frame must be an integer >= 0")
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
            np.any(self.source_frame < max(0, self.window_current_frame - 4))
            or np.any(self.source_frame > self.window_current_frame)
            or np.any(self.source_slot < 0)
        ):
            raise DataProtocolError("prediction point identities leave their window")
        identities = (self.source_frame.astype(np.int64) << 32) | self.source_slot
        if count == 0 or np.unique(identities).size != count:
            raise DataProtocolError(
                "prediction point identities are empty or duplicated"
            )
        for name in ("source_frame", "source_slot", "anomaly_score"):
            array = getattr(self, name).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def from_window(
        cls, window: SceneWindow, scores: np.ndarray, *, score_kind="probability"
    ) -> "PredictionBatch":
        values = np.asarray(scores, dtype=np.float32)
        if values.shape != (window.points.count,):
            raise DataProtocolError("a model must score every point in the window")
        result = cls(
            window.observation_sequence_id,
            window.current_frame_id,
            window.points.source_frame.copy(),
            window.points.source_slot.copy(),
            values.copy(),
            score_kind,
        )
        result.validate_window(window)
        return result

    def validate_window(self, window: SceneWindow) -> None:
        """Require the complete canonical input rows, not a self-consistent subset."""

        if not isinstance(window, SceneWindow):
            raise TypeError("prediction verification requires the actual SceneWindow")
        if (
            self.observation_sequence_id != window.observation_sequence_id
            or self.window_current_frame != window.current_frame_id
            or self.anomaly_score.shape != (window.points.count,)
            or not np.array_equal(self.source_frame, window.points.source_frame)
            or not np.array_equal(self.source_slot, window.points.source_slot)
            or not np.isfinite(self.anomaly_score).all()
        ):
            raise DataProtocolError(
                "prediction does not match every input window point in row order"
            )

    @property
    def online_mask(self) -> np.ndarray:
        result = self.source_frame == self.window_current_frame
        result.setflags(write=False)
        return result

    def save(self, path: Path, *, window: SceneWindow) -> dict[str, object]:
        """Persist every point score; no historical observation may be discarded."""

        self.validate_window(window)
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
        if self.score_kind == "logit":
            metadata["score_kind"] = self.score_kind
        metadata["content_hash"] = _prediction_content_hash(metadata, arrays)
        payload = dict(arrays)
        # Slot deltas compress monotone identities; content hashes bind decoded rows.
        payload["source_slot_delta"] = np.diff(self.source_slot, prepend=np.int32(0))
        del payload["source_slot"]
        payload["metadata_json"] = np.asarray(
            json.dumps(
                metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        # Scores are incompressible enough that maximum DEFLATE wastes CPU time.
        # Lossless level 1 changes packaging only, not point rows or scientific hashes.
        _stable_npz(path, payload, compression_level=1)
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
        window: SceneWindow,
        expected_sha256: str | None = None,
    ) -> "PredictionBatch":
        """Load one complete-window prediction after content verification."""

        resolved = path.expanduser().resolve(strict=True)
        if expected_sha256 is not None and _sha256(resolved) != expected_sha256:
            raise DataProtocolError("prediction file hash differs")
        with np.load(resolved, allow_pickle=False) as payload:
            slot_key = (
                "source_slot_delta" if "source_slot_delta" in payload else "source_slot"
            )
            if set(payload.files) != {
                "source_frame",
                slot_key,
                "anomaly_score",
                "metadata_json",
            }:
                raise DataProtocolError("prediction file has unexpected arrays")
            arrays = {
                name: np.asarray(payload[name]).copy()
                for name in ("source_frame", "anomaly_score")
            }
            slots = np.asarray(payload[slot_key])
            arrays["source_slot"] = (
                np.cumsum(slots, dtype=np.int32)
                if slot_key == "source_slot_delta"
                else slots.copy()
            )
            metadata = json.loads(str(payload["metadata_json"].item()))
        content_hash = metadata.pop("content_hash", None)
        if (
            set(metadata) - {"score_kind"}
            != {
                "format",
                "synthetic_or_raw_sequence_id",
                "window_current_frame",
                "point_count",
            }
            or metadata.get("format") != PREDICTION_FORMAT
            or metadata.get("score_kind", "probability") not in ("probability", "logit")
            or not isinstance(metadata.get("synthetic_or_raw_sequence_id"), str)
            or not metadata["synthetic_or_raw_sequence_id"]
            or type(metadata.get("window_current_frame")) is not int
            or metadata.get("point_count") != int(arrays["anomaly_score"].size)
            or content_hash != _prediction_content_hash(metadata, arrays)
        ):
            raise DataProtocolError("prediction metadata or content hash differs")
        result = cls(
            str(metadata["synthetic_or_raw_sequence_id"]),
            int(metadata["window_current_frame"]),
            arrays["source_frame"],
            arrays["source_slot"],
            arrays["anomaly_score"],
            metadata.get("score_kind", "probability"),
        )
        result.validate_window(window)
        return result


def _prediction_content_hash(
    metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(b"AJAE-complete-window-point-prediction\0")
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
    segment: RenderedSegment | None,
    raw_sources: Iterable[SourceFrame],
    *,
    pool_name: str,
    synthetic_sequence_id: str,
    synthetic_sequence_index: int,
    segment_index: int,
    world: WorldSpec | None = None,
    report: WorldGenerationReport | None = None,
    renderer_identity: str | None = None,
    ray_grid=None,
    sensor=None,
    statistics: dict | None = None,
) -> dict[str, object]:
    """Stream changed slots and frame references; never expand overlapping windows."""

    if segment is not None:
        world, report, renderer_identity = (
            segment.world,
            segment.report,
            segment.renderer_identity,
        )
        pairs = zip(raw_sources, segment.rendered_frames, strict=True)
    else:
        if world is None or report is None or renderer_identity is None:
            raise DataProtocolError(
                "streamed rendering requires a world and its provenance"
            )
        pairs = (
            (source, render_frame(source, world, ray_grid, sensor))
            for source in raw_sources
        )
    frame_ids: list[int] = []
    rendered_identities: list[str] = []
    offsets = [0]
    slot_parts: list[np.ndarray] = []
    xyzi_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    object_parts: list[np.ndarray] = []
    raw_identities: list[str] = []
    visible_counts: list[int] = []
    anomaly_counts: list[int] = []
    for source, rendered in pairs:
        if source.frame_id != rendered.frame_id or (
            frame_ids and source.frame_id != frame_ids[-1] + 1
        ):
            raise DataProtocolError(
                "raw and rendered frames must align and be consecutive"
            )
        frame_ids.append(source.frame_id)
        rendered_identities.append(source_observation_identity(rendered.source))
        if source.labels is None:
            raise DataProtocolError("formal synthetic sources require labels")
        changed = np.asarray(rendered.changed_mask, dtype=np.bool_)
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
        anomaly_counts.append(int(rendered.inserted_mask.sum()))
        if statistics is not None:
            distance = np.linalg.norm(rendered.xyzi[rendered.inserted_mask, :3], axis=1)
            inside = distance[(distance >= 2.5) & (distance <= 50)]
            statistics.setdefault("frames", []).append(
                dict(
                    frame=source.frame_id,
                    anomaly=int(distance.size),
                    anomaly_in_range=int(inside.size),
                    anomaly_in_range_distance_median=float(np.median(inside))
                    if inside.size
                    else None,
                    cleared_native_slots=int(
                        (
                            rendered.occluded_original_mask & ~rendered.inserted_mask
                        ).sum()
                    ),
                )
            )
    if len(frame_ids) < 5:
        raise DataProtocolError(
            "a stored world must contain a complete five-frame window"
        )
    starts = list(range(frame_ids[0], frame_ids[-1] - 3))
    window_identities = [
        rendered_window_identity(
            start, frame_ids[i : i + 5], rendered_identities[i : i + 5]
        )
        for i, start in enumerate(starts)
    ]

    arrays = {
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
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
        "source_sequence_id": world.source_sequence_id,
        "segment_identity": rendered_segment_identity(
            world.identity,
            frame_ids[0],
            frame_ids,
            renderer_identity,
            rendered_identities,
        ),
        "segment_boundary_inclusive": [frame_ids[0], frame_ids[-1]],
        "seed": world.seed,
        "world_identity": world.identity,
        "world_content_identity": world_content_identity(world),
        "world": world.to_dict(),
        "world_generation_report": report.to_dict(),
        "renderer_identity": renderer_identity,
        "raw_source_identities": raw_identities,
        "rendered_source_identities": rendered_identities,
        "window_identities": window_identities,
        "window_starts": starts,
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
        "seed": world.seed,
        "world_identity": world.identity,
        "world_content_identity": world_content_identity(world),
        "frame_range_inclusive": [frame_ids[0], frame_ids[-1]],
        "window_count": len(starts),
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
    _frame_cache: OrderedDict[int, SourceFrame] = field(
        init=False, default_factory=OrderedDict
    )

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
            or not 1 <= len(world.objects) <= 2
            or report.anomaly_count != len(world.objects)
            or report.placement_mode
            not in {
                "terminal_visible",
                "support_visible_fallback",
                "continuous_observation",
            }
            or report.support_scope
            not in {"nearest_quartile", "all_segment", "full_trajectory"}
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
        ):
            raise DataProtocolError("sparse segment arrays are misaligned")
        returned = np.any(arrays["changed_xyzi"][:, :3] != 0, axis=1)
        packed = arrays["changed_packed_labels"]
        objects = arrays["changed_object_ids"]
        if (
            np.any((packed[returned] & np.uint32(0xFFFF)) != 2)
            or not np.isin(
                objects[returned], [item.object_id for item in world.objects]
            ).all()
            or np.any((packed[returned] >> np.uint32(16)) != 60000 + objects[returned])
            or np.any(packed[~returned] != 0)
            or np.any(objects[~returned] != -1)
            or np.any(arrays["changed_xyzi"][~returned] != 0)
        ):
            raise DataProtocolError(
                "changed slots must be anomaly returns or cleared opaque occlusions"
            )
        frame_id_tuple = tuple(map(int, frame_ids))
        starts = tuple(map(int, metadata["window_starts"]))
        raw_identities = tuple(metadata["raw_source_identities"])
        rendered_identities = tuple(metadata["rendered_source_identities"])
        window_identities = tuple(metadata["window_identities"])
        visible_counts = tuple(map(int, metadata["visible_point_counts"]))
        anomaly_counts = tuple(map(int, metadata["anomaly_return_counts"]))
        if report.placement_mode == "terminal_visible" and (
            not anomaly_counts or anomaly_counts[-1] < 1
        ):
            raise DataProtocolError("terminal-visible placement has no terminal return")
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
                int(returned[offsets[index] : offsets[index + 1]].sum())
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
            expected_visible = (
                raw.real_count
                - int((~raw.zero_slot_mask[slots]).sum())
                + int(returned[offsets[index] : offsets[index + 1]].sum())
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
            result = self._frame_cache.pop(frame_id)
            self._frame_cache[frame_id] = result
            return result
        try:
            index = self.frame_ids.index(frame_id)
        except ValueError as error:
            raise IndexError(frame_id) from error
        raw = self.source_sequence.source_frame(frame_id)
        raw_identities = self.metadata["raw_source_identities"]
        raw_identity = source_observation_identity(raw)
        if raw_identity != raw_identities[index]:
            raise DataProtocolError(
                "raw source observation differs from the frozen pool"
            )
        offsets = self.arrays["frame_offsets"]
        start, stop = int(offsets[index]), int(offsets[index + 1])
        slots = self.arrays["changed_slots"][start:stop]
        if raw.labels is None:
            raise DataProtocolError("frozen synthetic data requires source labels")
        result = raw
        if len(slots):
            xyzi = raw.xyzi.copy()
            xyzi[slots] = self.arrays["changed_xyzi"][start:stop]
            packed = raw.labels.packed.copy()
            packed[slots] = self.arrays["changed_packed_labels"][start:stop]
            semantic = (packed & np.uint32(0xFFFF)).astype(np.uint16)
            instance = (packed >> np.uint32(16)).astype(np.uint16)
            semantic_target = None
            if raw.labels.semantic_target is not None:
                semantic_target = raw.labels.semantic_target.copy()
                semantic_target[slots] = np.uint8(255)
            result = make_source_frame(
                frame_id,
                xyzi,
                raw.lidar_pose,
                PointLabels(packed, semantic, instance, semantic_target),
                partition=raw.partition,
                sequence_id=raw.sequence_id,
            )
        # An unchanged synthetic scan is the identical immutable raw observation.
        expected = self.metadata["rendered_source_identities"][index]
        actual = source_observation_identity(result) if len(slots) else raw_identity
        if actual != expected:
            raise DataProtocolError("reconstructed frame differs from frozen rendering")
        self._frame_cache[frame_id] = result
        # Overlapping causal windows need five immutable frames, not a whole segment.
        while len(self._frame_cache) > 5:
            self._frame_cache.popitem(last=False)
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


def generation_identity(
    protocol: AJAEProtocol,
    pool: SyntheticPoolSpec,
    *,
    source_files_sha256: Mapping[str, str] | None = None,
) -> str:
    """Bind a pool to its scientific rules, inputs, and rendering source code."""

    support = protocol.artifacts["qualified_support_pools"][
        f"train/{pool.source_sequence_id}"
    ]
    source_role = dict(
        protocol.data[
            "parameter_update_source"
            if pool.source_sequence_id == 206
            else "model_validation_source"
        ]
    )
    if pool.source_sequence_id == 201:
        # Preserve the original generation identity; later access policy changes no points.
        source_role["role"] = (
            "only_source_for_model_validation_hyperparameter_tuning_and_model_selection"
        )
    pool_record = dict(protocol.synthetic_pools[pool.name])
    if source_files_sha256 is not None:
        # Frozen v1 evidence records its original location, not today's storage layout.
        pool_record["output_directory"] = f"artifacts/data/{pool.name}"
    payload = {
        "schema_version": protocol.schema_version,
        "pool": pool_record,
        "anomaly_objects_per_segment": protocol.synthetic_pools[
            "anomaly_objects_per_segment"
        ],
        "placement": protocol.synthetic_pools["placement"],
        "source_role": source_role,
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
            if source_files_sha256 is None
            else source_files_sha256[name]
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


def _frozen_qualification(protocol: AJAEProtocol) -> Mapping[str, object]:
    """Read the original evidence by its pinned bytes; never refresh it in place."""

    record = protocol.artifacts["qualification"]
    path = protocol.path.parent / str(record["file"])
    if (
        not path.is_file()
        or record["sha256"] is None
        or _sha256(path) != record["sha256"]
    ):
        raise DataProtocolError("frozen qualification bytes differ from protocol")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("format") != "ajae-schema34-data-qualification"
        or result.get("schema_version") != protocol.schema_version
        or result.get("passed") is not True
        or result.get("model_independent") is not True
        or result.get("checks")
        != {name: True for name in protocol.qualification["required_checks"]}
    ):
        raise DataProtocolError(
            "frozen qualification does not certify the required checks"
        )
    for name in ("train", "validation"):
        record = protocol.artifacts[f"{name}_pool_manifest"]
        if (
            result["inputs"].get(f"{name}_manifest")
            != f"artifacts/data/{name}_manifest.json"
            or result["inputs"].get(f"{name}_manifest_sha256") != record["sha256"]
        ):
            raise DataProtocolError("qualification and frozen manifests disagree")
    return result


def load_pool_manifest(
    protocol: AJAEProtocol, pool: SyntheticPoolSpec
) -> Mapping[str, object]:
    """Verify the authoritative manifest and every declared segment before use."""

    qualification = (
        _frozen_qualification(protocol) if protocol.status["data_pool_frozen"] else None
    )
    path = protocol.pool_manifest_path(pool.name)
    expected_hash = protocol.artifacts[f"{pool.name}_pool_manifest"]["sha256"]
    if not path.is_file() or (
        expected_hash is not None and _sha256(path) != expected_hash
    ):
        raise DataProtocolError(f"{pool.name} manifest bytes differ from protocol")
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Generation provenance belongs to the frozen evidence, not today's consumer code.
    expected_generation = generation_identity(
        protocol,
        pool,
        source_files_sha256=None
        if qualification is None
        else qualification["source_files_sha256"],
    )
    if (
        set(payload)
        != {
            "format",
            "schema_version",
            "pool_name",
            "generation_identity",
            "source_sequence_id",
            "synthetic_sequence_count",
            "world_count",
            "window_count",
            "scientific_content_hash",
            "segments",
        }
        or payload.get("format") != POOL_MANIFEST_FORMAT
        or payload.get("schema_version") != protocol.schema_version
        or payload.get("pool_name") != pool.name
        or payload.get("generation_identity") != expected_generation
        or payload.get("source_sequence_id") != pool.source_sequence_id
        or payload.get("synthetic_sequence_count") != pool.synthetic_sequence_count
        or payload.get("world_count") != pool.world_count
        or payload.get("window_count") != pool.total_window_count
        or not isinstance(payload.get("segments"), list)
        or len(payload["segments"]) != pool.world_count
    ):
        raise DataProtocolError(f"{pool.name} manifest contradicts the protocol")
    scientific_hash = hashlib.sha256(
        json.dumps(
            {
                "pool_name": pool.name,
                "generation_identity": expected_generation,
                "segment_scientific_hashes": [
                    item["scientific_content_hash"] for item in payload["segments"]
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if payload.get("scientific_content_hash") != scientific_hash:
        raise DataProtocolError(f"{pool.name} manifest scientific hash differs")
    worlds: set[str] = set()
    physical_worlds: set[str] = set()
    for index, record in enumerate(payload["segments"]):
        sequence_index, segment_index = divmod(index, len(pool.segments))
        span = pool.segments[segment_index]
        expected = {
            "synthetic_sequence_id": pool.synthetic_sequence_id(sequence_index),
            "synthetic_sequence_index": sequence_index,
            "segment_index": segment_index,
            "seed": pool.world_seed(sequence_index, segment_index),
            "frame_range_inclusive": [span.start, span.stop - 1],
            "window_count": len(pool.window_starts(segment_index)),
            "file": _segment_path(
                Path(f"artifacts/data/{pool.name}"), sequence_index, segment_index
            ).as_posix(),
        }
        if (
            set(record)
            != set(expected)
            | {
                "file_sha256",
                "scientific_content_hash",
                "world_identity",
                "world_content_identity",
                "changed_slot_count",
            }
            or any(record.get(key) != value for key, value in expected.items())
            or type(record["changed_slot_count"]) is not int
            or record["changed_slot_count"] < 1
            or record["world_identity"] in worlds
            or record["world_content_identity"] in physical_worlds
        ):
            raise DataProtocolError(
                "manifest segment order, identity, seed, or boundary differs"
            )
        worlds.add(record["world_identity"])
        physical_worlds.add(record["world_content_identity"])
        # Resolve the current location after validating the unchanged historical manifest.
        current_file = _segment_path(
            Path(pool.output_directory), sequence_index, segment_index
        ).as_posix()
        path = protocol.path.parent / current_file
        if not path.is_file() or _sha256(path) != record["file_sha256"]:
            raise DataProtocolError(
                f"segment file differs from its manifest: {record['file']}"
            )
        record["file"] = current_file
    return payload


class FrozenWindowDataset:
    """Training/validation input whose constructor verifies both frozen pools."""

    def __init__(
        self,
        data_root: Path,
        protocol: AJAEProtocol,
        *,
        pool_name: str,
        segment_cache_bytes: int = 0,
        version: str = "v1",
    ) -> None:
        if (
            not protocol.status["data_pool_frozen"]
            or not protocol.status["training_allowed"]
        ):
            raise DataProtocolError("training data must be frozen and qualified")
        if version == "v2":
            self.pool, self.manifest = observation_pool(pool_name)
        elif version == "v1":
            self.pool = _pool_spec(protocol, pool_name)
            manifests = {
                pool.name: load_pool_manifest(protocol, pool)
                for pool in (protocol.training_pool, protocol.validation_pool)
            }
            self.manifest = manifests[pool_name]
        else:
            raise DataProtocolError("unknown frozen data version")
        self.protocol = protocol
        self.source_sequence = STUSequence.open(
            data_root,
            protocol=protocol,
            partition="train",
            sequence_id=self.pool.source_sequence_id,
            label_mode=LabelMode.REQUIRED,
        )
        self._windows = tuple(
            (index, start)
            for index, record in enumerate(self.manifest["segments"])
            for start in self.pool.window_starts(record["segment_index"])
        )
        self._segment_index: int | None = None
        self._segment: FrozenSyntheticSegment | None = None
        if segment_cache_bytes < 0:
            raise ValueError("segment cache size must be nonnegative")
        self._segment_cache_bytes = segment_cache_bytes
        self._segments: OrderedDict[str, tuple[FrozenSyntheticSegment, int]] = (
            OrderedDict()
        )
        self._cached_bytes = 0

    @property
    def gradient_updates_allowed(self) -> bool:
        return self.pool.name == "train"

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> SceneWindow:
        segment, start = self.segment_for_window(index)
        return segment.window(start)

    def segment_for_window(self, index: int) -> tuple[FrozenSyntheticSegment, int]:
        """Reuse the frozen segment when only the current observation is needed."""
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        segment_index, start = self._windows[index]
        if self._segment_index != segment_index:
            record = self.manifest["segments"][segment_index]
            if self._segment is not None:
                self._segment._frame_cache.clear()
            key = record["file_sha256"]
            cached = self._segments.pop(key, None)
            if cached is None:
                segment = FrozenSyntheticSegment(
                    self.protocol.path.parent / record["file"],
                    self.source_sequence,
                    record["file_sha256"],
                )
            else:
                segment, size = cached
                self._cached_bytes -= size
            if any(
                segment.metadata[key] != record[key]
                for key in (
                    "synthetic_sequence_id",
                    "synthetic_sequence_index",
                    "segment_index",
                    "seed",
                    "world_identity",
                    "world_content_identity",
                    "scientific_content_hash",
                )
            ):
                raise DataProtocolError(
                    "manifest and loaded segment identities disagree"
                )
            # Cache validated sparse deltas, never expanded windows or network features.
            # File identity includes the world and raw-source identities; scope is this dataset.
            size = sum(array.nbytes for array in segment.arrays.values()) + 4 * len(
                json.dumps(segment.metadata)
            )
            if size <= self._segment_cache_bytes:
                self._segments[key] = segment, size
                self._cached_bytes += size
                while self._cached_bytes > self._segment_cache_bytes:
                    _, (_, size) = self._segments.popitem(last=False)
                    self._cached_bytes -= size
            self._segment = segment
            self._segment_index = segment_index
        return self._segment, start


def observation_pool(pool_name):
    """Read completed v2 observations; never substitute pilots or rerender a frame."""
    from .protocol import FrameSpan

    config = json.loads(
        (PROJECT_ROOT / "protocols/observation_match_v2/config.json").read_text()
    )
    pools, manifests, seeds, identities = {}, {}, set(), set()
    for name, source, frames, worlds in (
        ("train", 206, 449, 32),
        ("validation", 201, 682, 8),
    ):
        spec = config["pools"][name]
        directory = PROJECT_ROOT / config["paths"]["data"] / name
        path = directory / "manifest.json"
        fixed = config["completion"]["pools"][name]
        if _sha256(path) != fixed["manifest_sha256"]:
            raise DataProtocolError("completed v2 manifest changed")
        manifest = json.loads(path.read_text())
        if (
            manifest["pilot_round"],
            manifest["source_sequence_id"],
            manifest["world_count"],
            manifest["window_count"],
            len(manifest["segments"]),
        ) != (0, source, worlds, worlds * (frames - 4), worlds):
            raise DataProtocolError("v2 source roles or full-sequence coverage differ")
        pool = SyntheticPoolSpec(
            name,
            source,
            worlds,
            (FrameSpan(0, frames),),
            spec["seed_base"],
            str(directory),
            worlds,
            frames - 4,
            worlds * (frames - 4),
            "v2",
        )
        for index, record in enumerate(manifest["segments"]):
            path = directory / f"world_{index:03d}.npz"
            if any(
                (
                    record["file"] != path.name,
                    record["synthetic_sequence_id"]
                    != pool.synthetic_sequence_id(index),
                    record["synthetic_sequence_index"] != index,
                    record["segment_index"] != 0,
                    record["seed"] != pool.world_seed(index, 0),
                    record["frame_range_inclusive"] != [0, frames - 1],
                    record["window_count"] != frames - 4,
                    record["seed"] in seeds,
                    record["world_content_identity"] in identities,
                    _sha256(path) != record["file_sha256"],
                )
            ):
                raise DataProtocolError(
                    "v2 identity, independence or full window range differs"
                )
            seeds.add(record["seed"])
            identities.add(record["world_content_identity"])
            record["file"] = str(path)
        pools[name], manifests[name] = pool, manifest
    return pools[pool_name], manifests[pool_name]


def normal_group_targets(labels):
    """Only binary-normal returns receive a semantic group; ignore remains ignore."""
    if labels.semantic_target is None:
        raise DataProtocolError(
            "fine semantic supervision is unavailable on real anomaly val"
        )
    groups = np.full(labels.semantic.shape, -1, np.int64)
    normal = labels.anomaly_target == 0
    groups[normal] = np.where(
        labels.semantic_target[normal] == 255, 19, labels.semantic_target[normal]
    )
    return groups


def training_label_statistics(data_root):
    """Count actual v2 supervision by reusing each source scan across sparse worlds."""
    pool, manifest = observation_pool("train")
    protocol = load_protocol()
    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    counts = np.zeros((32, 449, 20), np.int64)
    anomalies = np.zeros((32, 449), np.int64)
    arrays = []
    for record in manifest["segments"]:
        with np.load(record["file"], allow_pickle=False) as payload:
            arrays.append(
                {
                    k: payload[k].copy()
                    for k in (
                        "changed_slots",
                        "changed_xyzi",
                        "changed_packed_labels",
                        "frame_offsets",
                    )
                }
            )
    for frame in range(449):
        raw = sequence.source_frame(frame)
        if np.any(raw.labels.anomaly_target[raw.real_slots] == 1):
            raise DataProtocolError("normal 206 source contains anomaly returns")
        groups = normal_group_targets(raw.labels)
        groups[raw.zero_slot_mask] = -1
        base = np.bincount(groups[groups >= 0], minlength=20)
        for world, a in enumerate(arrays):
            start, stop = a["frame_offsets"][frame : frame + 2]
            slots = a["changed_slots"][start:stop]
            removed = groups[slots]
            counts[world, frame] = base - np.bincount(
                removed[removed >= 0], minlength=20
            )
            returned = np.any(a["changed_xyzi"][start:stop, :3] != 0, axis=1)
            if np.any((a["changed_packed_labels"][start:stop][returned] & 0xFFFF) != 2):
                raise DataProtocolError("new v2 returns must be anomalies")
            anomalies[world, frame] = int(returned.sum())
    # Count repeated history exactly as it appears in the fixed training windows.
    exposure = np.convolve(np.ones(445, np.int64), np.ones(5, np.int64))
    total = (counts * exposure[None, :, None]).sum(axis=(0, 1))
    present_n = present_a = 0
    for world in range(32):
        present_n += int(
            (
                np.convolve(counts[world].sum(1), np.ones(5, np.int64), mode="valid")
                > 0
            ).sum()
        )
        present_a += int(
            (
                np.convolve(anomalies[world], np.ones(5, np.int64), mode="valid") > 0
            ).sum()
        )
    active = np.flatnonzero(total)
    weights = np.clip(np.sqrt(total[active].mean() / total[active]), 0.25, 4)
    return dict(
        source="frozen v2 train/206 only; actual full-window point exposures",
        manifest_sha256=_sha256(PROJECT_ROOT / "artifacts/data/v2/train/manifest.json"),
        window_count=pool.total_window_count,
        normal_present_windows=present_n,
        anomaly_present_windows=present_a,
        pi_normal=present_n / pool.total_window_count,
        pi_anomaly=present_a / pool.total_window_count,
        normal_group_point_counts=total.tolist(),
        active_groups=active.tolist(),
        class_weights=weights.tolist(),
        weight_rule="sqrt(mean active-group count / group count), clipped to [0.25,4] before weighted CE normalization",
        other_normal_group=19,
    )


def _pool_spec(protocol: AJAEProtocol, name: str) -> SyntheticPoolSpec:
    if name == "train":
        return protocol.training_pool
    if name == "validation":
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
    target = (
        protocol.pool_manifest_path(pool_name)
        if manifest_path is None
        else manifest_path.expanduser().resolve()
    )
    if protocol.status["data_pool_frozen"] and target == protocol.pool_manifest_path(
        pool_name
    ):
        raise DataProtocolError(
            "frozen manifests are read-only; load and verify them instead"
        )
    if target.exists():
        raise FileExistsError(target)
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
    _atomic_json(target, manifest)
    return manifest


_OBSERVATION_STATE = None


def observation_distribution(frames):
    """Use raw visibility for five-bit states and official current points for the joint grid."""
    patterns = np.zeros(32, dtype=np.int64)
    joint = np.zeros((4, 4), dtype=np.int64)
    joint_points = np.zeros((4, 4), dtype=np.int64)
    current_absent_history_present = 0
    for t in range(4, len(frames)):
        bits = [int(row["anomaly"] > 0) for row in frames[t - 4 : t + 1]]
        pattern = sum(bit << (4 - i) for i, bit in enumerate(bits))
        patterns[pattern] += 1
        current_absent_history_present += int(not bits[-1] and any(bits[:-1]))
        row = frames[t]
        if row["anomaly_in_range"] >= 5:
            c = int(
                np.searchsorted([20, 100, 500], row["anomaly_in_range"], side="right")
            )
            d = int(
                np.searchsorted(
                    [10, 20, 35], row["anomaly_in_range_distance_median"], side="right"
                )
            )
            joint[c, d] += 1
            joint_points[c, d] += row["anomaly_in_range"]
    return dict(
        pattern_counts=patterns.tolist(),
        qualified_joint_counts=joint.tolist(),
        qualified_joint_anomaly_points=joint_points.tolist(),
        current_absent_history_present=current_absent_history_present,
        cleared_native_slots=sum(r["cleared_native_slots"] for r in frames),
        frame_count=len(frames),
        window_count=len(frames) - 4,
    )


def _observation_candidate(task):
    """Evaluate one bounded, independent candidate; no anomaly model is loaded."""
    import time
    from dataclasses import replace
    from .render import (
        MaterialSpec,
        PlacementError,
        WorldGenerationReport,
        WorldSpec,
        _grounding_qualified_shape,
        place_object,
    )

    state = _OBSERVATION_STATE
    sequence, support, obstacles, grid, sensor = state["inputs"]
    rules = state["config"]["generation"]
    world_index, attempt = task
    world_seed = state["seed_base"] + 1000 * world_index
    streams = (
        np.random.SeedSequence([world_seed, attempt]).generate_state(5).astype(np.int64)
    )
    shape_seed, material_seed, yaw_seed, placement_seed, factor_seed = map(int, streams)
    rng = np.random.default_rng(factor_seed)
    role = str(
        rng.choice(rules["candidate_roles"], p=rules["candidate_role_probabilities"])
    )
    family = str(
        rng.choice(
            list(rules["desired_shape_family_probabilities"]),
            p=list(rules["desired_shape_family_probabilities"].values()),
        )
    )
    broad = bool(rng.random() < rules["broad_world_probability"])
    size = tuple(rules["broad_size_range_m"] if broad else rules["size_range_m"])
    identity = f"synthetic/{state['namespace']}/{state['pool']}/{world_index:03d}"
    path = state["output"] / f"candidate_{world_index:03d}_{attempt:02d}.npz"
    started = time.monotonic()
    result = dict(
        world_index=world_index,
        attempt=attempt,
        seed=world_seed,
        streams=streams.tolist(),
        role=role,
        desired_family=family,
        broad_shape=broad,
        size_range_m=list(size),
    )
    try:
        shape, shape_report, grounding, proposed, rejected = _grounding_qualified_shape(
            shape_seed,
            stride=1,
            maximum_proposals=rules["maximum_shape_proposals"],
            size_m_range=size,
            desired_family=family,
        )
        rows = state["support_rows"][role]
        if not len(rows):
            raise PlacementError(
                "no source support satisfies the fixed trajectory range stratum"
            )
        item, placement = place_object(
            shape,
            MaterialSpec.sample(material_seed),
            support,
            obstacles,
            object_id=1,
            label="anomaly-proxy",
            proposal_namespace=f"{world_seed}:{attempt}",
            proposal_stream=placement_seed,
            yaw_rad=float(np.random.default_rng(yaw_seed).uniform(-np.pi, np.pi)),
            material_seed=material_seed,
            yaw_seed=yaw_seed,
            shape_seed=proposed[-1],
            shape_generation_report=shape_report,
            proposal_rows=rows,
            maximum_candidates=rules["maximum_support_proposals"],
            grounding_eligibility=grounding,
        )
        placement = replace(
            placement,
            shape_proposal_seeds=proposed,
            grounding_rejection_seeds=rejected,
            accepted_shape_proposal=len(proposed) - 1,
        )
        world = WorldSpec(world_seed, sequence.spec.sequence_id, (item,))
        report = WorldGenerationReport(
            world_seed,
            sequence.spec.sequence_id,
            "anomaly_only",
            attempt,
            1,
            placement_seed,
            (placement,),
            "continuous_observation",
            "full_trajectory",
        )
        statistics = {}
        record = save_sparse_segment(
            path,
            None,
            (sequence.source_frame(t) for t in sequence.frame_ids),
            world=world,
            report=report,
            renderer_identity=state["identity"],
            ray_grid=grid,
            sensor=sensor,
            statistics=statistics,
            pool_name=state["pool"],
            synthetic_sequence_id=identity,
            synthetic_sequence_index=world_index,
            segment_index=0,
        )
        result.update(
            status="rendered",
            record=record,
            world=world.to_dict(),
            placement=report.to_dict(),
            frames=statistics["frames"],
            distribution=observation_distribution(statistics["frames"]),
        )
    except PlacementError as error:
        result.update(status="rejected", rejection=str(error))
        for key in (
            "proposal_pool_indices",
            "rejection_reasons",
            "minimum_obstacle_sdf_m",
        ):
            if hasattr(error, key):
                result[key] = list(getattr(error, key))
    sequence._frames.clear()
    result["wall_seconds"] = time.monotonic() - started
    return result


def select_observation_candidates(candidates, world_count, config):
    """Keep a bounded set of pool combinations instead of irrevocable greedy choices."""
    rules = config["generation"]
    targets = config["targets"]
    real = targets["real_visibility_counts"]
    reference = (
        np.array([real[k] for k in ("00000", "11111", "partial")]) / real["denominator"]
    )
    limits = np.array([targets["visibility"][k] for k in ("00000", "11111", "partial")])
    target_joint = np.array(targets["qualified_joint_probability"])
    beam = [((), np.zeros(32, np.int64), np.zeros((4, 4), np.int64), (3, 1.0, 1.0))]
    for i in range(world_count):
        options = [
            candidates[(i, j)]
            for j in range(rules["maximum_world_candidates"])
            if candidates[(i, j)]["status"] == "rendered"
        ]
        if not options:
            raise DataProtocolError(
                f"world {i} exhausted all fixed geometry/support candidates"
            )
        expanded = []
        for chosen, patterns, joint, _ in beam:
            for row in options:
                p = patterns + np.array(row["distribution"]["pattern_counts"], np.int64)
                q = joint + np.array(
                    row["distribution"]["qualified_joint_counts"], np.int64
                )
                fractions = np.array([p[0], p[31], p.sum() - p[0] - p[31]]) / p.sum()
                # Core coverage and predeclared visibility ranges precede softer joint matching.
                cost = (
                    int(sum(q[a, b] == 0 for a, b in targets["required_far_cells"])),
                    float(
                        np.maximum(limits[:, 0] - fractions, 0).sum()
                        + np.maximum(fractions - limits[:, 1], 0).sum()
                    ),
                    float(
                        0.5 * np.abs(fractions - reference).sum()
                        + 0.5 * np.abs(q / max(1, q.sum()) - target_joint).sum()
                    ),
                )
                expanded.append((chosen + (row["attempt"],), p, q, cost))
        beam = sorted(expanded, key=lambda item: (item[3], item[0]))[
            : rules["pool_selection_beam_width"]
        ]
    return beam[0]


def generate_observation_match(
    data_root, pool_name, *, config_path, pilot_round, workers
):
    """Generate full-sequence candidates, then select worlds only by fixed data targets."""
    import time
    from scipy.spatial import cKDTree
    from .train import host_disk

    global _OBSERVATION_STATE
    protocol = load_protocol()
    config = json.loads(Path(config_path).read_text())
    if pilot_round not in (0, 1, 2) or workers < 1:
        raise DataProtocolError("use at most two pilot rounds and positive workers")
    if not pilot_round:
        # Formal object streams must not repeat either debugging pool or the other split.
        seen = {
            base + r * config["pilot"]["second_round_seed_offset"] + 1000 * i
            for name, base in config["pilot"]["seed_bases"].items()
            for r in range(config["pilot"]["maximum_rounds"])
            for i in range(config["pilot"]["worlds"][name])
        }
        for specification in config["pools"].values():
            seeds = {
                specification["seed_base"] + 1000 * i
                for i in range(specification["world_count"])
            }
            if seen.intersection(seeds):
                raise DataProtocolError("formal and pilot world random streams overlap")
            seen.update(seeds)
    if not pilot_round and config["status"] != "generation_rules_fixed":
        raise DataProtocolError(
            "formal generation requires the completed pilot decision"
        )
    pool = config["pools"][pool_name]
    source_id = pool["source_sequence"]
    count = config["pilot"]["worlds"][pool_name] if pilot_round else pool["world_count"]
    seed_base = (
        config["pilot"]["seed_bases"][pool_name]
        + (pilot_round - 1) * config["pilot"]["second_round_seed_offset"]
        if pilot_round
        else pool["seed_base"]
    )
    output = (
        Path(config["pilot"]["output"]) / f"pilot_{pilot_round}" / pool_name
        if pilot_round
        else Path(config["paths"]["data"]) / pool_name
    )
    output.mkdir(parents=True, exist_ok=True)
    scientific = {k: config[k] for k in ("targets", "generation", "pools", "seed_rule")}
    identity = hashlib.sha256(
        json.dumps(scientific, sort_keys=True).encode()
        + Path(__file__).read_bytes()
        + Path(__file__).with_name("render.py").read_bytes()
    ).hexdigest()
    disk = host_disk()
    # Candidates are sparse; two GiB includes scratch, selected worlds and atomic writes.
    if disk["SizeRemaining"] - disk["reserve_bytes"] < 2 * 2**30:
        raise OSError("insufficient host capacity for bounded generation")
    spec = dict(
        experiment=config["experiment"],
        pilot_round=pilot_round,
        pool=pool_name,
        seed_base=seed_base,
        world_count=count,
        candidate_limit=config["generation"]["maximum_world_candidates"],
        scientific=scientific,
        renderer_identity=identity,
        workers=workers,
        host_disk=disk,
    )
    spec_path = output / "spec.json"
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        if any(old[k] != spec[k] for k in spec if k not in ("workers", "host_disk")):
            raise DataProtocolError(
                "saved candidate inputs differ; do not reuse another pilot's state"
            )
    else:
        _atomic_json(spec_path, spec)
    if (output / "manifest.json").exists():
        return json.loads((output / "manifest.json").read_text())
    started = time.monotonic()
    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=source_id,
        label_mode=LabelMode.REQUIRED,
    )
    sequence._cache_frames = 5
    support = load_qualified_support_pool(
        protocol.verify_support_pool(source_id),
        source_sequence_id=source_id,
        expected_sha256=protocol.artifacts["qualified_support_pools"][
            f"train/{source_id}"
        ]["sha256"],
    )
    nearest, _ = cKDTree(sequence._lidar_poses[:, :3, 3]).query(
        support.anchors_world_m, workers=workers
    )
    support_rows = {}
    for role, (lower, upper) in config["generation"][
        "candidate_minimum_trajectory_range_m"
    ].items():
        keep = (nearest >= lower) & (nearest < upper)
        if role == "complete_pass":
            a, b = config["generation"]["complete_pass_support_time_fraction"]
            keep &= (support.frames >= a * (len(sequence) - 1)) & (
                support.frames <= b * (len(sequence) - 1)
            )
        support_rows[role] = np.flatnonzero(keep)
    print(
        json.dumps(
            dict(
                event="support_strata",
                pool=pool_name,
                counts={k: len(v) for k, v in support_rows.items()},
            )
        ),
        flush=True,
    )
    obstacles = collect_observed_obstacle_index(
        (sequence.source_frame(t) for t in sequence.frame_ids),
        source_sequence_id=source_id,
    )
    sequence._frames.clear()
    grid, sensor = load_sensor_calibration(protocol.verify_sensor_calibration())
    _OBSERVATION_STATE = dict(
        inputs=(sequence, support, obstacles, grid, sensor),
        config=config,
        support_rows=support_rows,
        identity=identity,
        output=output,
        pool=pool_name,
        namespace=f"pilot_v2_r{pilot_round}" if pilot_round else "v2",
        seed_base=seed_base,
    )
    candidates = {}
    trace = output / "candidates.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            candidates[(row["world_index"], row["attempt"])] = row
    tasks = [
        (i, j)
        for i in range(count)
        for j in range(spec["candidate_limit"])
        if (i, j) not in candidates
    ]
    try:
        with (
            trace.open("a") as stream,
            mp.get_context("fork").Pool(processes=workers) as processes,
        ):
            for row in processes.imap_unordered(
                _observation_candidate, tasks, chunksize=1
            ):
                candidates[(row["world_index"], row["attempt"])] = row
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        dict(
                            event="candidate",
                            pool=pool_name,
                            world=row["world_index"],
                            attempt=row["attempt"],
                            status=row["status"],
                            wall_seconds=row["wall_seconds"],
                        )
                    ),
                    flush=True,
                )
                volume = host_disk()
                if volume["SizeRemaining"] - volume["reserve_bytes"] < 512 * 2**20:
                    raise OSError("candidate writes approached the host reserve")
    finally:
        _OBSERVATION_STATE = None
    choices, patterns, joint, cost = select_observation_candidates(
        candidates, count, config
    )
    records, selections = [], []
    for i, j in enumerate(choices):
        chosen = candidates[(i, j)]
        record = dict(chosen["record"])
        target = output / f"world_{i:03d}.npz"
        Path(record["file"]).rename(target)
        record["file"] = target.name
        records.append(record)
        selections.append(
            dict(
                world_index=i,
                selected_attempt=j,
                pool_objective=list(cost),
                distribution=chosen["distribution"],
                frames=chosen["frames"],
                rejected_rendered_attempts=[
                    a
                    for a in range(spec["candidate_limit"])
                    if a != j and candidates[(i, a)]["status"] == "rendered"
                ],
                rejection_reason="not retained by the fixed bounded pool combination search",
            )
        )
    manifest = dict(
        format="ajae-observation-match-pool",
        experiment=config["experiment"],
        pilot_round=pilot_round,
        pool_name=pool_name,
        source_sequence_id=source_id,
        generation_identity=identity,
        world_count=count,
        window_count=int(patterns.sum()),
        frame_count=count * pool["frame_count"],
        segments=records,
        pattern_counts=patterns.tolist(),
        qualified_joint_counts=joint.tolist(),
        selections=selections,
        wall_seconds=time.monotonic() - started,
    )
    _atomic_json(output / "manifest.json", manifest)
    # Candidate deltas are temporary; their seeds, physical parameters and statistics remain in the trace.
    for row in candidates.values():
        if row["status"] == "rendered":
            Path(row["record"]["file"]).unlink(missing_ok=True)
    print(
        json.dumps(
            dict(
                event="pool_generated",
                pool=pool_name,
                world_count=count,
                patterns=patterns.tolist(),
                joint=joint.tolist(),
                wall_seconds=manifest["wall_seconds"],
            )
        ),
        flush=True,
    )
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
        description="Generate or verify schema-34 frozen window data"
    )
    parser.add_argument(
        "action", choices=("generate", "manifest", "check", "observe", "labels")
    )
    parser.add_argument("--pool", required=True, choices=("train", "validation"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "protocol.json")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--sequence-indices")
    parser.add_argument("--segment-indices")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pilot-round", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "protocols/observation_match_v2/config.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.action == "labels":
        if (
            args.data_root is None
            or args.pool != "train"
            or args.output_directory is None
        ):
            raise DataProtocolError(
                "normal-group statistics require train, data-root and an output directory"
            )
        statistics = training_label_statistics(args.data_root)
        args.output_directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.output_directory / "labels.json", statistics)
        print(json.dumps(statistics, indent=2))
    elif args.action == "observe":
        if args.data_root is None:
            raise DataProtocolError("observation matching requires --data-root")
        generate_observation_match(
            args.data_root,
            args.pool,
            config_path=args.config,
            pilot_round=args.pilot_round,
            workers=args.workers,
        )
    elif args.action == "generate":
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
    elif args.action == "manifest":
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
    else:
        if args.data_root is None:
            raise DataProtocolError("training-input verification requires --data-root")
        dataset = FrozenWindowDataset(args.data_root, protocol, pool_name=args.pool)
        print(
            json.dumps(
                {
                    "pool": dataset.pool.name,
                    "window_count": len(dataset),
                    "gradient_updates_allowed": dataset.gradient_updates_allowed,
                    "both_frozen_pools_verified": True,
                    "first_window_current_frame": dataset[0].current_frame_id,
                    "last_window_current_frame": dataset[
                        len(dataset) - 1
                    ].current_frame_id,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
