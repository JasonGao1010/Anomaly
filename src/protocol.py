#!/usr/bin/env python3
"""Read the active single-scan STU data and evaluation contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "protocol.json"
SCHEMA_VERSION = 1
PUBLIC_ANOMALY_IDS = (
    125,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    169,
)


class ProtocolError(ValueError):
    """Report a contradiction in the active STU contract."""


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    partition: str
    sequence_id: int
    role: str
    labels_available: bool
    frame_count: int | None = None

    def __post_init__(self):
        if self.partition not in {"train", "val", "test"}:
            raise ProtocolError("invalid STU partition")
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise ProtocolError("sequence_id must be a nonnegative integer")
        if type(self.labels_available) is not bool:
            raise ProtocolError("labels_available must be boolean")
        if self.frame_count is not None and (
            type(self.frame_count) is not int or self.frame_count < 1
        ):
            raise ProtocolError("frame_count must be a positive integer")

    def with_observed_frame_count(self, count):
        if type(count) is not int or count < 1:
            raise ProtocolError("observed frame count must be positive")
        if self.frame_count is not None and self.frame_count != count:
            raise ProtocolError("observed frame count conflicts with source metadata")
        return replace(self, frame_count=count)


@dataclass(frozen=True)
class STUProtocol:
    path: Path
    document: Mapping

    @property
    def public_sequence_ids(self):
        return tuple(self.document["data"]["development"]["sequence_ids"])

    @property
    def semantic_class_map(self):
        return {
            int(raw): int(target)
            for raw, target in self.document["labels"][
                "normal_semantic_class_map"
            ].items()
        }

    def sequence(self, partition, sequence_id):
        if type(sequence_id) is not int:
            raise ProtocolError("sequence_id must be an integer")
        if partition == "val" and sequence_id in self.public_sequence_ids:
            return SequenceSpec("val", sequence_id, "public_development", True)
        if partition == "train":
            for source in self.document["data"]["normal_sources"]:
                if source["sequence_id"] == sequence_id:
                    return SequenceSpec(
                        "train",
                        sequence_id,
                        "normal_source",
                        True,
                        source["frame_count"],
                    )
        # The development contract does not open hidden test data.
        raise ProtocolError(
            f"{partition}/{sequence_id} is outside the active data scope"
        )


def load_protocol(path=DEFAULT_PROTOCOL_PATH):
    path = Path(path).expanduser().resolve(strict=True)
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("format") != "stu-single-frame"
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise ProtocolError("expected the single-scan STU protocol")
    data = document["data"]
    if (
        data["development"]["partition"] != "val"
        or tuple(data["development"]["sequence_ids"]) != PUBLIC_ANOMALY_IDS
    ):
        raise ProtocolError("development must contain the 19 public anomaly sequences")
    if data["hidden_test"]["in_active_scope"]:
        raise ProtocolError("hidden test is outside development")
    if document["input"]["unit"] != "one_current_scan" or document["input"][
        "channels"
    ] != ["x", "y", "z", "intensity"]:
        raise ProtocolError("model input must be one current scan")
    evaluation = document["evaluation"]
    if (
        evaluation["distance_m"],
        evaluation["ignore_semantic"],
        evaluation["anomaly_semantic"],
        evaluation["min_anomaly_points_per_frame"],
    ) != ([2.5, 50], 0, 2, 5):
        raise ProtocolError("official STU point and frame filters cannot change")
    if evaluation["fpr_limit"] != 0.01:
        raise ProtocolError("the global low-FPR operating point is fixed at 1%")
    return STUProtocol(path, document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    print(json.dumps(protocol.document, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
