"""Restore synthetic scans from saved deltas and the original STU data.

The reader returns raw XYZI, source labels, and insertion/occlusion masks.
It does not choose a model, training target, sampler, or evaluation rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .scene import PointLabels, SourceFrame, STUSequence, make_source_frame


DEFAULT_SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class DataProtocolError(ValueError):
    """Report a saved sample that disagrees with its source scan or world."""


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
            source.validate_duplicate_values(value, name)
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        inserted, occluded = self.inserted_mask, self.occluded_original_mask
        if np.any(source.zero_slot_mask & inserted):
            raise DataProtocolError("inserted mask includes a missing return")
        if np.any(source.labels.semantic[inserted] != 2) or np.any(
            source.labels.instance[inserted] != 60001
        ):
            raise DataProtocolError(
                "frozen samples have exactly one inserted object with ID 1"
            )
        missing = occluded & ~inserted
        if np.any(source.xyzi[missing] != 0) or np.any(source.labels.packed[missing]):
            raise DataProtocolError(
                "opaque occlusion without a return must clear the slot"
            )

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
        expected_worlds = 240 if split == "train" else 120
        if len(manifest["splits"][split]["worlds"]) != expected_worlds:
            raise DataProtocolError("V3 uses the complete fixed 240/120-world pool")
        sequence = manifest["splits"][split]["source_sequence"]
        if (
            manifest["splits"]["train"]["source_sequence"] != 206
            or manifest["splits"]["validation"]["source_sequence"] != 201
        ):
            raise DataProtocolError(
                "frozen training and validation data use disjoint normal sources"
            )
        self.sequence = STUSequence(data_root, sequence)
        self.split = split
        self.samples = []
        self.worlds = []
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
            if definition["objects"][0]["object_id"] != 1:
                raise DataProtocolError(
                    "stored ID 60001 must denote generated object 1"
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
            self.worlds.append(
                dict(
                    identity=identity,
                    path=world_dir,
                    height_m=world.get("geometry", {}).get("height_m"),
                    height_source="generator_object_local_bounds",
                    frames=frames,
                )
            )
        if len(self.samples) != manifest["splits"][split]["samples"]:
            raise DataProtocolError("manifest sample count disagrees with full worlds")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, identity, frame = self.samples[index]
        return FrozenFrame.load(path, self.sequence[frame], identity)

    def pair(self, index):
        """Original and inserted scans share slots, never inserted-return identities."""
        path, identity, frame = self.samples[index]
        original = self.sequence[frame]
        return original, FrozenFrame.load(path, original, identity)


def detection_targets(source, *, inserted=None, real_anomalies=False, records=False):
    """Range limits supervision only; callers still pass every observed return."""
    if source.labels is None:
        raise DataProtocolError("detection supervision requires raw labels")
    slots = source.real_slots if records else source.observation_slots
    raw = source.labels.semantic[slots]
    radius = np.linalg.norm(source.xyzi[slots, :3], axis=1)
    valid = (radius >= 2.5) & (radius <= 50.0)
    target = np.full(len(slots), -1, np.int8)
    target[valid & (raw != 0) & (raw != 2)] = 0
    anomaly = (raw == 2) if real_anomalies else np.zeros(len(slots), bool)
    if inserted is not None:
        anomaly = inserted[slots]
    target[valid & anomaly] = 1
    return target


def read_real_frame(data_root, sequence, frame):
    """Read val19 without fitting statistics or guessing object geometry."""
    directory = Path(data_root) / "val" / str(sequence)
    xyzi = np.fromfile(directory / "velodyne" / f"{frame:06d}.bin", dtype="<f4")
    if xyzi.size % 4:
        raise DataProtocolError("real scan is not XYZI")
    packed = np.fromfile(directory / "labels" / f"{frame:06d}.label", dtype="<u4")
    labels = PointLabels(
        packed,
        (packed & 65535).astype(np.uint16),
        (packed >> 16).astype(np.uint16),
        None,
    )
    return make_source_frame(
        frame,
        xyzi.reshape(-1, 4),
        np.eye(4, dtype=np.float64),
        labels,
        partition="val",
        sequence_id=int(sequence),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/jasongao/Data/STU")
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    dataset = FrozenDataset(args.samples, args.data_root, args.split)
    sample = dataset[args.index]
    print(
        json.dumps(
            {
                "split": args.split,
                "samples": len(dataset),
                "index": args.index,
                "source_sequence": sample.source.sequence_id,
                "source_frame": sample.source.frame_id,
                "file_slots": sample.source.slot_count,
                "actual_returns": sample.source.real_count,
                "inserted_slots": int(sample.inserted_mask.sum()),
                "occluded_original_slots": int(sample.occluded_original_mask.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
