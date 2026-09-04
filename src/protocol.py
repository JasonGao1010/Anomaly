#!/usr/bin/env python3
"""Load the compact AJAE schema-33 feasibility contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "protocol.json"
SCHEMA_VERSION = 33
WINDOW_FRAMES = 5
WINDOW_MEMBER_OFFSETS = (0, 1, 2, 3, 4)

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
HIDDEN_TEST_IDS = (
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
)


class ProtocolError(ValueError):
    """Report a schema or data-role contradiction."""


class InputMode(str, Enum):
    """The two frozen-STU inputs compared before any AJAE training."""

    SINGLE_STU = "single_stu"
    DENSE_STU = "dense_stu"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ProtocolError(f"{name} must be a JSON object")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} must be a non-empty string")
    return value


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be an array")
    return tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(value))


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _sha256(value: object, name: str) -> str:
    digest = _string(value, name)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True, slots=True)
class FrameSpan:
    """A half-open contiguous source-frame interval."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        _integer(self.start, "frame span start")
        _integer(self.stop, "frame span stop", minimum=1)
        if self.stop <= self.start:
            raise ProtocolError("frame span stop must exceed start")

    def __len__(self) -> int:
        return self.stop - self.start

    def contains(self, frame_id: int) -> bool:
        return self.start <= frame_id < self.stop


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    """One physical STU sequence or one contiguous role inside it."""

    partition: str
    sequence_id: int
    role: str
    labels_available: bool
    span: FrameSpan | None
    excluded_source_frames: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.partition not in {"train", "val", "test"}:
            raise ProtocolError("partition must be train, val, or test")
        _integer(self.sequence_id, "sequence id")
        _string(self.role, "sequence role")
        if type(self.labels_available) is not bool:
            raise ProtocolError("labels_available must be boolean")
        if (
            tuple(sorted(set(self.excluded_source_frames)))
            != self.excluded_source_frames
        ):
            raise ProtocolError("excluded frames must be sorted and unique")
        if self.span is not None and any(
            not self.span.contains(frame) for frame in self.excluded_source_frames
        ):
            raise ProtocolError("an excluded frame lies outside its span")

    @property
    def supports_counterfactuals(self) -> bool:
        return self.partition == "train" and self.sequence_id in {201, 206}

    def with_observed_frame_count(self, frame_count: int) -> "SequenceSpec":
        count = _integer(frame_count, "frame count", minimum=1)
        if self.span is not None:
            if self.span.start != 0 or self.span.stop != count:
                raise ProtocolError("observed frame count conflicts with the protocol")
            return self
        return SequenceSpec(
            self.partition,
            self.sequence_id,
            self.role,
            self.labels_available,
            FrameSpan(0, count),
            self.excluded_source_frames,
        )

    def legal_window_starts(self) -> tuple[int, ...]:
        if self.span is None:
            raise ProtocolError("frame count must be observed before making windows")
        excluded = frozenset(self.excluded_source_frames)
        return tuple(
            start
            for start in range(self.span.start, self.span.stop - WINDOW_FRAMES + 1)
            if not excluded.intersection(range(start, start + WINDOW_FRAMES))
        )

    def window_frame_ids(self, window_start: int) -> tuple[int, ...]:
        start = _integer(window_start, "window start")
        if start not in frozenset(self.legal_window_starts()):
            raise ProtocolError(f"window start {start} is outside {self.role}")
        return tuple(start + offset for offset in WINDOW_MEMBER_OFFSETS)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    minimum_range_m: float
    maximum_range_m: float
    minimum_anomaly_points: int

    def range_mask(self, ranges: object) -> object:
        import numpy as np

        values = np.asarray(ranges, dtype=np.float32)
        return (values >= self.minimum_range_m) & (values <= self.maximum_range_m)


class AJAEProtocol:
    """Validated immutable view of the schema-33 feasibility route."""

    def __init__(self, document: Mapping[str, object], *, path: Path) -> None:
        self._validate(document)
        self.path = path.expanduser().resolve(strict=True)
        self.schema_version = SCHEMA_VERSION
        self._document = _freeze(document)
        for name in (
            "status",
            "authority",
            "window",
            "methods",
            "data",
            "labels",
            "render",
            "stu",
            "feasibility",
            "evaluation_document",
            "claims",
        ):
            source_name = "evaluation" if name == "evaluation_document" else name
            setattr(self, name, self._document[source_name])

        data = _mapping(document["data"], "data")
        self.normal_training = self._sequence(data, "future_training")
        self.normal_development = self._sequence(data, "normal_development")
        self.normal_confirmation = self._sequence(data, "normal_confirmation")
        # The loader opens train/201 once; experiment code applies the disjoint role spans.
        self.development_sequence = SequenceSpec(
            "train", 201, "normal_201_split", True, FrameSpan(0, 682), (0, 1, 2, 3)
        )
        public = _mapping(data["public_anomaly_validation"], "public anomaly data")
        hidden = _mapping(data["hidden_test"], "hidden test data")
        self.public_validation = tuple(
            SequenceSpec("val", item, str(public["role"]), True, None)
            for item in PUBLIC_ANOMALY_IDS
        )
        self.hidden_test = tuple(
            SequenceSpec("test", item, str(hidden["role"]), False, None)
            for item in HIDDEN_TEST_IDS
        )
        self._sequences = {
            (item.partition, item.sequence_id): item
            for item in (
                self.normal_training,
                self.development_sequence,
                *self.public_validation,
                *self.hidden_test,
            )
        }
        class_map = _mapping(
            _mapping(document["labels"], "labels")["normal_semantic_class_map"],
            "class map",
        )
        self.normal_training_class_map = MappingProxyType(
            {
                int(raw): _integer(target, f"class map {raw}")
                for raw, target in class_map.items()
            }
        )
        evaluation = _mapping(document["evaluation"], "evaluation")
        self.evaluation = EvaluationSpec(
            _number(evaluation["minimum_range_m_inclusive"], "minimum range"),
            _number(evaluation["maximum_range_m_inclusive"], "maximum range"),
            _integer(
                evaluation["minimum_anomaly_points_per_evaluated_frame"],
                "minimum anomaly points",
                minimum=1,
            ),
        )
        self.evaluation_spec = self.evaluation

    @staticmethod
    def _sequence(data: Mapping[str, object], key: str) -> SequenceSpec:
        record = _mapping(data[key], f"data.{key}")
        bounds = _int_tuple(record["frame_range_inclusive"], f"{key} frame range")
        if len(bounds) != 2 or bounds[1] < bounds[0]:
            raise ProtocolError(f"{key} frame range must be [first,last]")
        return SequenceSpec(
            str(record["partition"]),
            _integer(record["sequence_id"], f"{key} sequence"),
            str(record["role"]),
            bool(record["labels_available"]),
            FrameSpan(bounds[0], bounds[1] + 1),
            _int_tuple(record["excluded_source_frames"], f"{key} exclusions"),
        )

    @property
    def document(self) -> Mapping[str, object]:
        return self._document  # type: ignore[return-value]

    @property
    def scientific_identity(self) -> str:
        payload = json.dumps(
            _plain(self._document),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def window_frames(self) -> int:
        return WINDOW_FRAMES

    @property
    def window_member_offsets(self) -> tuple[int, ...]:
        return WINDOW_MEMBER_OFFSETS

    @property
    def public_sequence_ids(self) -> tuple[int, ...]:
        return PUBLIC_ANOMALY_IDS

    @property
    def hidden_sequence_ids(self) -> tuple[int, ...]:
        return HIDDEN_TEST_IDS

    def sequence(self, partition: str, sequence_id: int) -> SequenceSpec:
        try:
            return self._sequences[(partition, sequence_id)]
        except KeyError as error:
            raise ProtocolError(
                f"sequence {partition}/{sequence_id} is outside schema 33"
            ) from error

    def checkpoint_path(self, project_root: Path | str | None = None) -> Path:
        root = (
            self.path.parent
            if project_root is None
            else Path(project_root).expanduser().resolve()
        )
        return (root / str(self.stu["checkpoint"])).resolve()

    def stu_repository_path(self, project_root: Path | str | None = None) -> Path:
        root = (
            self.path.parent
            if project_root is None
            else Path(project_root).expanduser().resolve()
        )
        return (root / str(self.stu["repository"])).resolve()

    def sensor_calibration_path(self, project_root: Path | str | None = None) -> Path:
        root = (
            self.path.parent
            if project_root is None
            else Path(project_root).expanduser().resolve()
        )
        return (root / str(self.render["calibration_file"])).resolve()

    def support_pool_path(
        self, sequence_id: int, project_root: Path | str | None = None
    ) -> Path:
        root = (
            self.path.parent
            if project_root is None
            else Path(project_root).expanduser().resolve()
        )
        pools = _mapping(self.render["qualified_support_pools"], "support pools")
        record = _mapping(
            pools[f"train/{_integer(sequence_id, 'support sequence')}"], "support pool"
        )
        return (root / str(record["file"])).resolve()

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "current_stage": self.status["current_stage"],
            "route": self.status["route"],
            "methods": [item.value for item in InputMode],
            "development_windows": len(self.normal_development.legal_window_starts()),
            "confirmation_windows": len(self.normal_confirmation.legal_window_starts()),
            "training_allowed": self.status["training_allowed"],
            "performance_claims_available": self.status["performance_claims_available"],
        }

    @classmethod
    def _validate(cls, source: Mapping[str, object]) -> None:
        expected_root = {
            "schema_version",
            "status",
            "authority",
            "research_question",
            "window",
            "methods",
            "data",
            "labels",
            "render",
            "stu",
            "feasibility",
            "evaluation",
            "stages",
            "claims",
        }
        if (
            source.get("schema_version") != SCHEMA_VERSION
            or set(source) != expected_root
        ):
            raise ProtocolError("protocol must be the sole schema-33 contract")
        status = _mapping(source["status"], "status")
        if (
            status.get("current_stage") != "F1"
            or status.get("route") != "frozen_stu_dense_input_feasibility"
        ):
            raise ProtocolError("schema 33 must wait at the F1 geometry stage")
        if (
            status.get("experiments_started") is not False
            or status.get("training_allowed") is not False
        ):
            raise ProtocolError(
                "schema 33 cannot claim experiments or training before F1 runs"
            )
        claims = _mapping(source["claims"], "claims")
        if (
            claims.get("implementation_ready") is not True
            or claims.get("F1_completed") is not False
        ):
            raise ProtocolError("F0 must be complete while F1 remains unexecuted")
        window = _mapping(source["window"], "window")
        if window.get("frames") != WINDOW_FRAMES:
            raise ProtocolError("AJAE requires five scans")
        if tuple(window.get("member_offsets_from_start", ())) != WINDOW_MEMBER_OFFSETS:
            raise ProtocolError("window offsets must be 0 through 4")
        if (
            window.get("coordinate_frame") != "latest_scan_t"
            or window.get("output_points") != "latest_scan_t_only"
        ):
            raise ProtocolError(
                "alignment and output must both target the current scan"
            )
        if window.get("overlap_fusion") != "none":
            raise ProtocolError("online schema 33 has no overlapping-score fusion")
        methods = _mapping(source["methods"], "methods")
        if set(methods) != {item.value for item in InputMode}:
            raise ProtocolError("only single_stu and dense_stu may be active")
        data = _mapping(source["data"], "data")
        development = cls._sequence(data, "normal_development")
        confirmation = cls._sequence(data, "normal_confirmation")
        if development.partition != "train" or development.sequence_id != 201:
            raise ProtocolError("normal development must use train/201")
        if confirmation.partition != "train" or confirmation.sequence_id != 201:
            raise ProtocolError("normal confirmation must use train/201")
        if (
            development.span is None
            or confirmation.span is None
            or development.span.stop != confirmation.span.start
        ):
            raise ProtocolError(
                "development and confirmation must be disjoint contiguous spans"
            )
        if (
            len(development.legal_window_starts()) != 546
            or len(confirmation.legal_window_starts()) != 124
        ):
            raise ProtocolError("train/201 split has unexpected window counts")
        public = _mapping(data["public_anomaly_validation"], "public anomalies")
        hidden = _mapping(data["hidden_test"], "hidden test")
        if _int_tuple(public["sequence_ids"], "public ids") != PUBLIC_ANOMALY_IDS:
            raise ProtocolError("public anomaly sequence set changed")
        if _int_tuple(hidden["sequence_ids"], "hidden ids") != HIDDEN_TEST_IDS:
            raise ProtocolError("hidden test sequence set changed")
        stu = _mapping(source["stu"], "stu")
        if (
            stu.get("source") != "STU_official_Mask4Former3D"
            or stu.get("score") != "official_STU_MaxLogit"
            or stu.get("frozen") is not True
        ):
            raise ProtocolError(
                "both feasibility methods must use frozen official STU MaxLogit"
            )
        _sha256(stu["checkpoint_sha256"], "STU checkpoint")
        _sha256(stu["model_state_tensor_sha256"], "STU tensor state")
        f2 = _mapping(
            _mapping(source["feasibility"], "feasibility")["F2_normal_stability"], "F2"
        )
        current_frames = _int_tuple(f2["current_frames"], "F2 current frames")
        if len(current_frames) != 24 or any(
            frame - 4 not in development.legal_window_starts()
            for frame in current_frames
        ):
            raise ProtocolError("F2 must use 24 legal development endpoints")
        f3 = _mapping(
            _mapping(source["feasibility"], "feasibility")["F3_proxy_signal"], "F3"
        )
        starts = _int_tuple(f3["source_starts"], "F3 starts")
        length = _integer(f3["frames_per_sequence"], "F3 sequence length", minimum=5)
        candidates = _integer(
            f3["candidate_current_frames_per_sequence"],
            "F3 candidate current frames",
            minimum=1,
        )
        if candidates != length - 4:
            raise ProtocolError(
                "F3 candidate current-frame count must equal sequence length minus four"
            )
        if len(starts) != _integer(
            f3["sequence_count"], "F3 sequence count", minimum=1
        ):
            raise ProtocolError("F3 start count differs from its world count")
        occupied: set[int] = set()
        for start in starts:
            frames = set(range(start, start + length))
            if (
                development.span is None
                or not all(development.span.contains(frame) for frame in frames)
                or not occupied.isdisjoint(frames)
            ):
                raise ProtocolError(
                    "F3 source segments must be disjoint inside development"
                )
            occupied.update(frames)


def load_protocol(path: Path | str = DEFAULT_PROTOCOL_PATH) -> AJAEProtocol:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError("protocol is unreadable") from error
    return AJAEProtocol(_mapping(document, "protocol"), path=resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the AJAE schema-33 contract."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    protocol = load_protocol(_parser().parse_args(argv).protocol)
    print(json.dumps(protocol.summary(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
