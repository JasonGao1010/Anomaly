"""Single-scan prediction and frozen synthetic sample storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .scene import PointLabels, SourceFrame, STUSequence, make_source_frame
from .protocol import load_protocol


class DataProtocolError(ValueError):
    """Report scores that cannot be assigned to the declared source returns."""


def source_identity(source):
    """Bind a lossless delta to the exact scan, labels, pose and file identity."""
    if source.labels is None:
        raise DataProtocolError("frozen training sources require original labels")
    digest = hashlib.sha256()
    digest.update(f"{source.partition}/{source.sequence_id}/{source.frame_id}".encode())
    for array in (source.xyzi, source.labels.packed, source.lidar_pose):
        digest.update(array.tobytes())
    if source.labels.semantic_target is not None:
        digest.update(source.labels.semantic_target.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenFrame:
    """One independent rendered scan; masks always address original file slots."""

    source: SourceFrame
    world_identity: str
    inserted_mask: np.ndarray
    occluded_original_mask: np.ndarray

    def __post_init__(self):
        source = self.source
        if source.labels is None or len(self.world_identity) != 64:
            raise DataProtocolError("frozen sample needs labels and a world identity")
        for name in ("inserted_mask", "occluded_original_mask"):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.bool_ or value.shape != (source.slot_count,):
                raise DataProtocolError(f"{name} must be bool[original file slot]")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        inserted, occluded = self.inserted_mask, self.occluded_original_mask
        if np.any(source.zero_slot_mask & inserted):
            raise DataProtocolError("inserted mask includes a missing return")
        if np.any(source.labels.semantic[inserted] != 2) or np.any(
            source.labels.instance[inserted] != 60001
        ):
            raise DataProtocolError("V1 has exactly one inserted object with ID 1")
        missing = occluded & ~inserted
        if np.any(source.xyzi[missing] != 0) or np.any(source.labels.packed[missing]):
            raise DataProtocolError(
                "opaque occlusion without a return must clear the slot"
            )

    @property
    def anomaly_target(self):
        """Original ignored semantics stay ignored; only inserted returns are positive."""
        labels = self.source.labels
        if labels.semantic_target is None:
            raise DataProtocolError("normal supervision requires the train class map")
        target = np.full(self.source.slot_count, -1, np.int8)
        target[(labels.semantic_target != 255) & ~self.source.zero_slot_mask] = 0
        target[self.inserted_mask] = 1
        return target

    def save(self, path, original):
        source = self.source
        if (
            (source.partition, source.sequence_id, source.frame_id)
            != (original.partition, original.sequence_id, original.frame_id)
            or source.slot_count != original.slot_count
            or not np.array_equal(source.lidar_pose, original.lidar_pose)
        ):
            raise DataProtocolError(
                "frozen delta and original identify different scans"
            )
        changed = self.inserted_mask | self.occluded_original_mask
        if original.labels.semantic_target is None:
            raise DataProtocolError("normal supervision requires the train class map")
        expected_target = original.labels.semantic_target.copy()
        expected_target[changed] = 255
        if not np.array_equal(source.labels.semantic_target, expected_target):
            raise DataProtocolError(
                "rendered normal class targets disagree with insertion masks"
            )
        if not np.array_equal(
            source.xyzi[~changed], original.xyzi[~changed]
        ) or not np.array_equal(
            source.labels.packed[~changed], original.labels.packed[~changed]
        ):
            raise DataProtocolError("unrecorded changes outside the rendered slots")
        if np.any(self.occluded_original_mask & original.zero_slot_mask):
            raise DataProtocolError("an originally empty slot cannot be occluded")
        slots = np.flatnonzero(changed).astype(np.int32)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(
                    stream,
                    format=np.asarray("stu-frozen-frame"),
                    source_identity=np.asarray(source_identity(original)),
                    world_identity=np.asarray(self.world_identity),
                    source_slot=slots,
                    xyzi=source.xyzi[slots],
                    packed_labels=source.labels.packed[slots],
                    inserted_slot=np.flatnonzero(self.inserted_mask).astype(np.int32),
                    occluded_slot=np.flatnonzero(self.occluded_original_mask).astype(
                        np.int32
                    ),
                )
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, original, world_identity):
        with np.load(path, allow_pickle=False) as saved:
            expected = {
                "format",
                "source_identity",
                "world_identity",
                "source_slot",
                "xyzi",
                "packed_labels",
                "inserted_slot",
                "occluded_slot",
            }
            if (
                set(saved.files) != expected
                or saved["format"].item() != "stu-frozen-frame"
            ):
                raise DataProtocolError("not a frozen single-frame delta")
            if saved["source_identity"].item() != source_identity(original):
                raise DataProtocolError(
                    "original scan, labels or pose changed after freezing"
                )
            if saved["world_identity"].item() != world_identity:
                raise DataProtocolError("frame belongs to another fixed world")
            arrays = {
                key: saved[key]
                for key in ("source_slot", "inserted_slot", "occluded_slot")
            }
            for name, slots in arrays.items():
                if (
                    slots.dtype != np.int32
                    or slots.ndim != 1
                    or np.any(slots < 0)
                    or np.any(slots >= original.slot_count)
                    or np.any(np.diff(slots) <= 0)
                ):
                    raise DataProtocolError(
                        f"{name} is not a sorted unique file-slot set"
                    )
            slots = arrays["source_slot"]
            if not np.array_equal(
                slots, np.union1d(arrays["inserted_slot"], arrays["occluded_slot"])
            ):
                raise DataProtocolError("delta slots disagree with rendered masks")
            if saved["xyzi"].dtype != np.float32 or saved["xyzi"].shape != (
                len(slots),
                4,
            ):
                raise DataProtocolError("delta xyzi must be float32[changed slot, 4]")
            if saved["packed_labels"].dtype != np.uint32 or saved[
                "packed_labels"
            ].shape != (len(slots),):
                raise DataProtocolError("delta labels must be uint32[changed slot]")
            xyzi, packed = original.xyzi.copy(), original.labels.packed.copy()
            xyzi[slots], packed[slots] = saved["xyzi"], saved["packed_labels"]
        target = original.labels.semantic_target.copy()
        target[slots] = 255
        labels = PointLabels(
            packed,
            (packed & 65535).astype(np.uint16),
            (packed >> 16).astype(np.uint16),
            target,
        )
        source = make_source_frame(
            original.frame_id,
            xyzi,
            original.lidar_pose,
            labels,
            partition=original.partition,
            sequence_id=original.sequence_id,
        )
        inserted = np.zeros(original.slot_count, bool)
        occluded = inserted.copy()
        inserted[arrays["inserted_slot"]] = True
        occluded[arrays["occluded_slot"]] = True
        if np.any(occluded & original.zero_slot_mask):
            raise DataProtocolError("an originally empty slot cannot be occluded")
        return cls(source, world_identity, inserted, occluded)


class FrozenDataset:
    """Read complete frozen worlds as independent scans without importing a renderer."""

    def __init__(self, directory, data_root, split):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if (
            manifest.get("format") != "stu-frozen-dataset"
            or manifest.get("status") != "frozen"
        ):
            raise DataProtocolError("dataset has not completed full-sequence freezing")
        if split not in {"train", "validation"}:
            raise DataProtocolError("synthetic split must be train or validation")
        sequence = manifest["splits"][split]["source_sequence"]
        if (
            manifest["splits"]["train"]["source_sequence"] != 206
            or manifest["splits"]["validation"]["source_sequence"] != 201
        ):
            raise DataProtocolError(
                "V1 training and validation use disjoint normal sources"
            )
        self.sequence = STUSequence.open(
            data_root,
            protocol=load_protocol(),
            partition="train",
            sequence_id=sequence,
            label_mode="required",
        )
        self.samples = []
        identities = set()
        for entry in manifest["splits"][split]["worlds"]:
            world_dir = self.directory / entry["path"]
            world = json.loads((world_dir / "manifest.json").read_text())
            identity = entry["world_identity"]
            definition = json.loads((world_dir / "world.json").read_text())["world"]
            actual_identity = hashlib.sha256(
                json.dumps(
                    definition,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if (
                actual_identity != identity
                or len(definition["objects"]) != 1
                or definition["source_sequence_id"] != sequence
            ):
                raise DataProtocolError(
                    "fixed world definition changed after rendering"
                )
            if identity in identities or world["world_identity"] != identity:
                raise DataProtocolError("world identity is duplicated or mismatched")
            identities.add(identity)
            frames = world["frames"]
            if world["source_sequence"] != sequence or [
                r["frame"] for r in frames
            ] != list(self.sequence.frame_ids):
                raise DataProtocolError(
                    "world must contain every source frame in order"
                )
            self.samples.extend(
                (world_dir / "frames" / f"{r['frame']:06d}.npz", identity, r["frame"])
                for r in frames
            )
        if len(self.samples) != manifest["splits"][split]["samples"]:
            raise DataProtocolError("manifest sample count disagrees with full worlds")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, identity, frame = self.samples[index]
        return FrozenFrame.load(path, self.sequence[frame], identity)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def host_disk():
    """Query the physical E: volume backing this WSL installation."""
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-Volume -DriveLetter E | Select-Object Size,SizeRemaining | ConvertTo-Json -Compress",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    volume = json.loads(result.stdout)
    reserve = 10_000_000_000
    if volume["SizeRemaining"] <= reserve:
        raise OSError("host E: has reached the required 10 GB reserve")
    return {**volume, "reserve_bytes": reserve}


@dataclass(frozen=True, slots=True)
class FramePrediction:
    partition: str
    sequence_id: int
    frame_id: int
    source_slot: np.ndarray
    anomaly_score: np.ndarray

    def __post_init__(self):
        if self.partition not in {"train", "val", "test", "fixture"}:
            raise DataProtocolError("invalid prediction partition")
        for name in ("sequence_id", "frame_id"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise DataProtocolError(f"{name} must be a nonnegative integer")
        slots, scores = np.asarray(self.source_slot), np.asarray(self.anomaly_score)
        if slots.ndim != 1 or not np.issubdtype(slots.dtype, np.integer):
            raise DataProtocolError(
                "source_slot must be a one-dimensional integer array"
            )
        if np.any(slots < 0) or np.any(slots > np.iinfo(np.int32).max):
            raise DataProtocolError("source slots are outside the supported file range")
        if len(np.unique(slots)) != len(slots):
            raise DataProtocolError("duplicate source slots")
        # Require an explicit float32 conversion at the producer, never round on save.
        if (
            scores.dtype != np.float32
            or scores.shape != slots.shape
            or not np.isfinite(scores).all()
        ):
            raise DataProtocolError(
                "anomaly_score must be finite float32 with one value per slot"
            )
        slots, scores = slots.astype(np.int32, copy=True), scores.copy()
        slots.setflags(write=False)
        scores.setflags(write=False)
        object.__setattr__(self, "source_slot", slots)
        object.__setattr__(self, "anomaly_score", scores)

    def validate(self, source: SourceFrame):
        if (self.partition, self.sequence_id, self.frame_id) != (
            source.partition,
            source.sequence_id,
            source.frame_id,
        ):
            raise DataProtocolError("prediction and source identify different scans")
        if not np.array_equal(np.sort(self.source_slot), source.real_slots):
            raise DataProtocolError(
                "source_slot must cover exactly SourceFrame.real_slots"
            )

    def restore(self, source: SourceFrame):
        """Assign by file slot, so producer row order cannot change point identity."""
        self.validate(source)
        scores = np.zeros(source.slot_count, np.float32)
        scores[self.source_slot] = self.anomaly_score
        return scores

    def save(self, path, source: SourceFrame):
        self.validate(source)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(
                    stream,
                    format=np.asarray("stu-frame-prediction"),
                    partition=np.asarray(self.partition),
                    sequence_id=np.asarray(self.sequence_id),
                    frame_id=np.asarray(self.frame_id),
                    source_slot=self.source_slot,
                    anomaly_score=self.anomaly_score,
                )
            # A complete file appears at once; an existing prediction is never overwritten.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, source: SourceFrame):
        with np.load(path, allow_pickle=False) as saved:
            expected = {
                "format",
                "partition",
                "sequence_id",
                "frame_id",
                "source_slot",
                "anomaly_score",
            }
            if (
                set(saved.files) != expected
                or saved["format"].item() != "stu-frame-prediction"
            ):
                raise DataProtocolError("not a single-scan prediction")
            result = cls(
                saved["partition"].item(),
                saved["sequence_id"].item(),
                saved["frame_id"].item(),
                saved["source_slot"],
                saved["anomaly_score"],
            )
        result.validate(source)
        return result
