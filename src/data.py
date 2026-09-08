"""Single-scan prediction identity and storage."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .scene import SourceFrame


class DataProtocolError(ValueError):
    """Report scores that cannot be assigned to the declared source returns."""


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
