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
from scipy.spatial import cKDTree

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
            raise DataProtocolError("frozen samples have exactly one inserted object with ID 1")
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

    def __init__(self, directory, data_root, split, *, allow_candidates=False):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if (
            manifest.get("format") != "stu-frozen-dataset"
            or manifest.get("status") not in ({"frozen", "candidates_complete"} if allow_candidates else {"frozen"})
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
                "frozen training and validation data use disjoint normal sources"
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

    def sampling_probabilities(self, path):
        """Align declared scan probabilities by fixed world and source-frame identity."""
        with np.load(path, allow_pickle=False) as values:
            keys = list(zip(values["world_identity"].tolist(), values["frame"].tolist()))
            weights = np.asarray(values["probability"], np.float64)
        expected = [(identity, frame) for _, identity, frame in self.samples]
        if len(set(keys)) != len(keys) or set(keys) != set(expected) or len(weights) != len(keys):
            raise DataProtocolError("sampling probabilities do not describe this complete dataset split")
        if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., atol=1e-12):
            raise DataProtocolError("every frozen scan needs positive normalized sampling probability")
        lookup = dict(zip(keys, weights))
        return np.array([lookup[key] for key in expected])


def low_support_slots(source, radius_m, minimum_neighbors, *, workers=1):
    """Count distinct positions in the complete original scan, before label filtering."""
    xyz, inverse = np.unique(source.xyzi[source.real_slots, :3].astype(np.float64),
                             axis=0, return_inverse=True)
    if not len(xyz):
        return np.empty(0, np.int32)
    # Self occupies rank one; rank k+1 decides whether at least k neighbors exist.
    distance = cKDTree(xyz).query(xyz, k=[minimum_neighbors + 1], workers=workers)[0][:, 0]
    return source.real_slots[(distance > radius_m)[inverse]].astype(np.int32)


def retained_normal_slots(frozen, original):
    retained = np.flatnonzero(~frozen.inserted_mask & ~frozen.occluded_original_mask
        & ~original.zero_slot_mask & (original.labels.semantic_target != 255))
    if not (np.array_equal(original.xyzi[retained], frozen.source.xyzi[retained])
            and np.array_equal(original.labels.packed[retained], frozen.source.labels.packed[retained])
            and np.all(frozen.anomaly_target[retained] == 0)):
        raise DataProtocolError("retained-normal pairing changed the original physical return or label")
    return retained


def normal_conditions(frozen, original, sparse_slots, radius_m):
    """S is native low support; C is proximity to any changed return, including missing returns."""
    kept = retained_normal_slots(frozen, original)
    sparse = np.intersect1d(kept, sparse_slots)
    changed = np.concatenate((frozen.source.xyzi[frozen.inserted_mask, :3],
                              original.xyzi[frozen.occluded_original_mask, :3]))
    near = np.empty(0, dtype=kept.dtype)
    if len(changed) and len(kept):
        near = kept[cKDTree(changed).query(original.xyzi[kept, :3], workers=1)[0] <= radius_m]
    return kept, sparse, near


class ConditionIndex:
    """Read native slot sets bound to complete source contents and one world-frame pool."""

    def __init__(self, path, dataset, parameters):
        with np.load(path, allow_pickle=False) as saved:
            if (saved["format"].item() != "ajae-v2-conditions"
                    or json.loads(saved["parameters"].item()) != parameters):
                raise DataProtocolError("condition index uses different scientific parameters")
            keys = list(zip(saved["world_identity"].tolist(), saved["frame"].tolist()))
            expected = [(world, frame) for _, world, frame in dataset.samples]
            if keys != expected:
                raise DataProtocolError("condition index belongs to another frozen world-frame pool")
            frames, identities = saved["source_frame"], saved["source_identity"]
            offsets, slots = saved["sparse_offsets"], saved["sparse_slot"]
            if (frames.tolist() != list(dataset.sequence.frame_ids) or len(identities) != len(frames)
                    or len(offsets) != len(frames) + 1 or offsets[0] != 0
                    or offsets[-1] != len(slots) or np.any(np.diff(offsets) < 0)):
                raise DataProtocolError("condition index does not cover the complete training source")
            self.sources = {int(frame): (str(identity), slots[offsets[i]:offsets[i + 1]].copy())
                for i, (frame, identity) in enumerate(zip(frames, identities, strict=True))}
        for _, slots in self.sources.values():
            if slots.dtype != np.int32 or np.any(slots < 0) or np.any(np.diff(slots) <= 0):
                raise DataProtocolError("low-support slots must be sorted and unique")

    def slots(self, original):
        identity, slots = self.sources[original.frame_id]
        if source_identity(original) != identity or np.any(slots >= original.slot_count):
            raise DataProtocolError("low-support index source contents changed")
        return slots

    def state_dict(self):
        return {str(frame): [identity, slots.tolist()] for frame, (identity, slots) in self.sources.items()}


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


def runtime_resources():
    """Record physical pressure without changing computation or power settings."""
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()}
    gpu = subprocess.run(["nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
    return dict(cpu_load_average=list(os.getloadavg()),
        cpu_ticks=[int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]],
        memory_available_bytes=memory["MemAvailable"], swap_used_bytes=memory["SwapTotal"]-memory["SwapFree"],
        gpu_fields="utilization_percent,memory_MiB,power_W,limit_W,temperature_C", gpu=gpu.stdout.strip())


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
