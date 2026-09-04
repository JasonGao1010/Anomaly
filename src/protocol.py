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
STAGES = ("F0", "F1", "F2", "F3", "F4", "C1", "V1", "T1")
ACTIVE_STAGES = STAGES[1:]

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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        calibration = _mapping(self.render["calibration"], "calibration")
        return (root / str(calibration["file"])).resolve()

    def verify_sensor_calibration(self, project_root: Path | str | None = None) -> Path:
        path = self.sensor_calibration_path(project_root)
        calibration = _mapping(self.render["calibration"], "calibration")
        expected = str(calibration["sha256"])
        if not path.is_file() or _file_sha256(path) != expected:
            raise ProtocolError("sensor calibration bytes differ from protocol")
        return path

    def verify_official_point_evaluator(
        self, project_root: Path | str | None = None
    ) -> Path:
        root = (
            self.path.parent
            if project_root is None
            else Path(project_root).expanduser().resolve()
        )
        evaluator = _mapping(self.stu["official_point_evaluator"], "point evaluator")
        path = (root / str(evaluator["file"])).resolve()
        if not path.is_file() or _file_sha256(path) != str(evaluator["sha256"]):
            raise ProtocolError("official point evaluator bytes differ from protocol")
        return path

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

    def verify_support_pool(
        self, sequence_id: int, project_root: Path | str | None = None
    ) -> Path:
        path = self.support_pool_path(sequence_id, project_root)
        pools = _mapping(self.render["qualified_support_pools"], "support pools")
        record = _mapping(pools[f"train/{sequence_id}"], "support pool")
        if not path.is_file() or _file_sha256(path) != str(record["sha256"]):
            raise ProtocolError("support-pool bytes differ from protocol")
        return path

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
        stage = status.get("current_stage")
        if (
            stage not in ACTIVE_STAGES
            or status.get("route") != "frozen_stu_dense_input_feasibility"
        ):
            raise ProtocolError("schema 33 has an invalid active stage or route")
        for key in (
            "experiments_started",
            "training_allowed",
            "performance_claims_available",
        ):
            if type(status.get(key)) is not bool:
                raise ProtocolError(f"status.{key} must be boolean")
        if status["experiments_started"] is not (stage != "F1"):
            raise ProtocolError("experiments_started contradicts the active stage")
        if status["training_allowed"] is not (stage == "F4"):
            raise ProtocolError("training is allowed only while F4 is active")
        if status["performance_claims_available"] is not (stage == "T1"):
            raise ProtocolError(
                "performance claims become available only after public validation"
            )
        claims = _mapping(source["claims"], "claims")
        completion_keys = (
            "F1_completed",
            "F2_completed",
            "F3_completed",
            "C1_completed",
            "training_performed",
            "real_anomaly_validation_performed",
        )
        if claims.get("implementation_ready") is not True or any(
            type(claims.get(key)) is not bool for key in completion_keys
        ):
            raise ProtocolError("claims must record boolean stage completion")
        required_completed = {
            "F1": (),
            "F2": ("F1_completed",),
            "F3": ("F1_completed", "F2_completed"),
            "F4": ("F1_completed", "F2_completed", "F3_completed"),
            "C1": ("F1_completed", "F2_completed", "F3_completed"),
            "V1": (
                "F1_completed",
                "F2_completed",
                "F3_completed",
                "C1_completed",
            ),
            "T1": (
                "F1_completed",
                "F2_completed",
                "F3_completed",
                "C1_completed",
                "real_anomaly_validation_performed",
            ),
        }[str(stage)]
        if any(claims[key] is not True for key in required_completed):
            raise ProtocolError("a prerequisite stage is not marked complete")
        future_completion = {
            "F1": (
                "F1_completed",
                "F2_completed",
                "F3_completed",
                "C1_completed",
                "real_anomaly_validation_performed",
            ),
            "F2": (
                "F2_completed",
                "F3_completed",
                "C1_completed",
                "real_anomaly_validation_performed",
            ),
            "F3": (
                "F3_completed",
                "C1_completed",
                "real_anomaly_validation_performed",
            ),
            "F4": ("C1_completed", "real_anomaly_validation_performed"),
            "C1": ("C1_completed", "real_anomaly_validation_performed"),
            "V1": ("real_anomaly_validation_performed",),
            "T1": (),
        }[str(stage)]
        if any(claims[key] is not False for key in future_completion):
            raise ProtocolError("a future stage is prematurely marked complete")
        if stage in {"F1", "F2", "F3", "F4"} and claims["training_performed"]:
            raise ProtocolError("training cannot be claimed before leaving F4")
        window = _mapping(source["window"], "window")
        if window.get("frames") != WINDOW_FRAMES:
            raise ProtocolError("AJAE requires five scans")
        if tuple(window.get("member_offsets_from_start", ())) != WINDOW_MEMBER_OFFSETS:
            raise ProtocolError("window offsets must be 0 through 4")
        if (
            window.get("coordinate_frame") != "latest_scan_t"
            or window.get("stu_output_points")
            != "all_input_points_from_X_(t-4)_through_X_t"
            or window.get("primary_comparison_points")
            != "original_visible_points_of_current_scan_X_t"
        ):
            raise ProtocolError(
                "alignment, full STU output, or current-point comparison changed"
            )
        if window.get("overlap_fusion") != "none":
            raise ProtocolError("online schema 33 has no overlapping-score fusion")
        methods = _mapping(source["methods"], "methods")
        if set(methods) != {item.value for item in InputMode}:
            raise ProtocolError("only single_stu and dense_stu may be active")
        dense_method = _mapping(methods[InputMode.DENSE_STU.value], "dense STU")
        if (
            dense_method.get("actual_output") != "all_input_points_from_five_scans"
            or dense_method.get("F2_F3_comparison_view") != "rows_originating_from_X_t"
        ):
            raise ProtocolError("dense STU output and comparison view changed")
        data = _mapping(source["data"], "data")
        archives = _mapping(data["official_archive_sha256"], "STU archives")
        if set(archives) != {"train.zip", "val.zip", "test.zip"}:
            raise ProtocolError("STU archive identities are incomplete")
        for name, digest in archives.items():
            _sha256(digest, f"STU archive {name}")
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
        confirmation_record = _mapping(
            data["normal_confirmation"], "normal confirmation"
        )
        if (
            _int_tuple(
                confirmation_record["output_frame_range_inclusive"],
                "confirmation outputs",
            )
            != (558, 681)
            or confirmation_record.get("output_window_count") != 124
        ):
            raise ProtocolError("normal confirmation outputs must be 558 through 681")
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
            or stu.get("official_semantic_prediction")
            != "query_class_of_argmax(mask_probability*query_class_confidence)"
        ):
            raise ProtocolError(
                "both feasibility methods must use frozen official STU MaxLogit"
            )
        _sha256(stu["checkpoint_sha256"], "STU checkpoint")
        _sha256(stu["model_state_tensor_sha256"], "STU tensor state")
        evaluator = _mapping(stu["official_point_evaluator"], "point evaluator")
        _string(evaluator["file"], "point evaluator file")
        _sha256(evaluator["sha256"], "point evaluator")
        calibration = _mapping(
            _mapping(source["render"], "render")["calibration"], "calibration"
        )
        _string(calibration["file"], "calibration file")
        _sha256(calibration["sha256"], "calibration")
        _string(calibration["source_file"], "calibration source file")
        _sha256(calibration["source_sha256"], "calibration source")
        pools = _mapping(
            _mapping(source["render"], "render")["qualified_support_pools"],
            "support pools",
        )
        if set(pools) != {"train/201", "train/206"}:
            raise ProtocolError("support pools must cover train/201 and train/206")
        for name, record_value in pools.items():
            record = _mapping(record_value, f"support pool {name}")
            _string(record["file"], f"support pool {name} file")
            _sha256(record["sha256"], f"support pool {name}")
        f2 = _mapping(
            _mapping(source["feasibility"], "feasibility")["F2_normal_stability"], "F2"
        )
        current_frames = _int_tuple(f2["current_frames"], "F2 current frames")
        if len(current_frames) != 24 or any(
            frame - 4 not in development.legal_window_starts()
            for frame in current_frames
        ):
            raise ProtocolError("F2 must use 24 legal development endpoints")
        precheck = _int_tuple(
            f2["official_normal_class_precheck_frames"], "F2 semantic precheck"
        )
        if not 3 <= len(precheck) <= 5 or not set(precheck) <= set(current_frames):
            raise ProtocolError("F2 semantic precheck must use 3 to 5 F2 frames")
        masks = _mapping(f2["masks"], "F2 masks")
        if set(masks) != {"normal_anomaly_mask", "semantic_class_mask"} or (
            masks.get("normal_anomaly_mask")
            != "raw_semantic_not_0_and_not_2_and_range_2.5_to_50m_inclusive"
            or masks.get("semantic_class_mask")
            != "semantic_target_not_255_and_range_2.5_to_50m_inclusive"
        ):
            raise ProtocolError("F2 must declare separate anomaly and class masks")
        f3 = _mapping(
            _mapping(source["feasibility"], "feasibility")["F3_proxy_signal"], "F3"
        )
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
        plans = (
            ("screen", 8, True),
            ("extension_if_screen_is_inconclusive", 16, False),
        )
        all_pairs: set[tuple[int, int]] = set()
        all_seeds: set[int] = set()
        for name, expected_count, disjoint in plans:
            plan = _mapping(f3[name], f"F3 {name}")
            count = _integer(plan["world_count"], f"F3 {name} count", minimum=1)
            starts = _int_tuple(plan["source_starts"], f"F3 {name} starts")
            seeds = _int_tuple(plan["world_root_seeds"], f"F3 {name} seeds")
            if count != expected_count or len(starts) != count or len(seeds) != count:
                raise ProtocolError(f"F3 {name} plan has the wrong size")
            occupied: set[int] = set()
            for start, seed in zip(starts, seeds, strict=True):
                frames = set(range(start, start + length))
                pair = (start, seed)
                if (
                    development.span is None
                    or not all(development.span.contains(frame) for frame in frames)
                    or pair in all_pairs
                    or seed in all_seeds
                    or (disjoint and not occupied.isdisjoint(frames))
                ):
                    raise ProtocolError(f"F3 {name} contains an invalid fixed world")
                occupied.update(frames)
                all_pairs.add(pair)
                all_seeds.add(seed)
        retry = _mapping(
            _mapping(source["render"], "render")["F3_world_retry"], "F3 retry"
        )
        if (
            retry.get("maximum_placement_attempts_per_root_seed") != 48
            or retry.get("allowed_retry_cause")
            != "PlacementError_during_physical_world_construction_before_rendering"
            or retry.get("placement_attempt_seed_formula")
            != "root_seed+1000003*attempt_index"
            or retry.get("world_root_seed_substitution") != "forbidden"
            or retry.get("retry_after_rendering_begins") != "forbidden"
            or retry.get("retry_for_an_invisible_window") != "forbidden"
        ):
            raise ProtocolError("F3 retry identity or visibility rule changed")
        if (
            tuple(f3.get("metrics", ())) != ("AP", "AUROC", "FPR95")
            or f3.get("unevaluable_world_rule")
            != "record_without_seed_substitution_and_exclude_from_metric_bootstrap"
            or f3.get("screen_decision")
            != "only_if_all_8_worlds_are_evaluable: support_if_delta_AP_lower_95_bound_above_zero; reject_if_mean_and_median_delta_AP_are_both_not_positive; otherwise_run_the_preplanned_16_world_extension"
        ):
            raise ProtocolError("F3 official metrics or two-phase decision changed")
        if tuple(source["stages"]) != STAGES:
            raise ProtocolError("schema 33 stage order changed")


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
