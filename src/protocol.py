#!/usr/bin/env python3
"""Load the sole AJAE schema-30 protocol and fixed development worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "protocol.json"
SCHEMA_VERSION = 30
WINDOW_FRAMES = 5
RELATIVE_TIMES = (-2, -1, 0, 1, 2)
CAUSAL_OFFSETS = (-4, -3, -2, -1, 0)
PUBLIC_ANOMALY_IDS = (
    125, 137, 138, 139, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 149, 150, 151, 152, 153, 169,
)
HIDDEN_TEST_IDS = (
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
)
NORMAL_CONTROL_SEMANTICS = (10, 11, 15, 18, 20, 30, 31, 32)
MOVING_NORMAL_SEMANTICS = (252, 253, 254, 255, 256, 257, 258, 259)
GATE1_EVIDENCE = (
    "ray_slot_audit",
    "range_image_round_trip",
    "render_source_leakage",
    "beam_range_intensity",
)
ROUND_TRIP_POINT_TOLERANCE_M = 1.0e-9
ROUND_TRIP_DIRECTION_TOLERANCE_RAD = 1.0e-6


class ProtocolError(ValueError):
    """Report a protocol or development-world semantic violation."""


class ExperimentCondition(str, Enum):
    """Pre-registered conditions in the minimum comparison matrix."""

    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"

    @property
    def trainable(self) -> bool:
        return self in {self.B1, self.B2, self.B3, self.B5}

    @property
    def frame_offsets(self) -> tuple[int, ...]:
        if self in {self.B0, self.B1}:
            return (0,)
        if self is self.B5:
            return CAUSAL_OFFSETS
        return RELATIVE_TIMES

    @property
    def output_local_indices(self) -> tuple[int, ...]:
        if self in {self.B0, self.B1}:
            return (0,)
        if self is self.B4:
            return tuple(range(WINDOW_FRAMES))
        if self is self.B5:
            return (WINDOW_FRAMES - 1,)
        return (2,)

    @property
    def cross_frame_enabled(self) -> bool:
        return self not in {self.B0, self.B1, self.B2}

    @property
    def trained_checkpoint_condition(self) -> ExperimentCondition:
        return self.B3 if self is self.B4 else self


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ProtocolError(f"{name} must be a JSON object with string keys")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be a JSON array")
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


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ProtocolError(f"{name} keys differ; missing={missing}, extra={extra}")


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    return tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(_list(value, name)))


def _signed_int_tuple(value: object, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for index, item in enumerate(_list(value, name)):
        if type(item) is not int:
            raise ProtocolError(f"{name}[{index}] must be an integer")
        result.append(item)
    return tuple(result)


def _float_matrix(value: object, name: str, rows: int, columns: int) -> tuple[tuple[float, ...], ...]:
    outer = _list(value, name)
    if len(outer) != rows:
        raise ProtocolError(f"{name} must contain {rows} rows")
    result: list[tuple[float, ...]] = []
    for row_index, row in enumerate(outer):
        values = tuple(
            _number(item, f"{name}[{row_index}][{column}]")
            for column, item in enumerate(_list(row, f"{name}[{row_index}]"))
        )
        if len(values) != columns:
            raise ProtocolError(f"{name}[{row_index}] must contain {columns} values")
        result.append(values)
    return tuple(result)


def _int_matrix(value: object, name: str, rows: int, columns: int) -> tuple[tuple[int, ...], ...]:
    matrix = _float_matrix(value, name, rows, columns)
    if any(number != int(number) or number < 1 for row in matrix for number in row):
        raise ProtocolError(f"{name} must contain positive integers")
    return tuple(tuple(int(number) for number in row) for row in matrix)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FrameSpan:
    """Half-open source-frame span."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        _integer(self.start, "FrameSpan.start")
        _integer(self.stop, "FrameSpan.stop", minimum=1)
        if self.stop <= self.start:
            raise ProtocolError("FrameSpan.stop must exceed start")

    def __len__(self) -> int:
        return self.stop - self.start

    def contains(self, frame_id: int) -> bool:
        return self.start <= frame_id < self.stop


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    """One immutable sequence role and its legal source span."""

    partition: str
    sequence_id: int
    role: str
    labels_available: bool
    span: FrameSpan | None
    excluded_source_frames: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.partition not in {"train", "val", "test"}:
            raise ProtocolError("sequence partition must be train, val, or test")
        _integer(self.sequence_id, "sequence_id")
        if not self.role:
            raise ProtocolError("sequence role must be non-empty")
        if type(self.labels_available) is not bool:
            raise ProtocolError("labels_available must be boolean")
        if tuple(sorted(set(self.excluded_source_frames))) != self.excluded_source_frames:
            raise ProtocolError("excluded source frames must be sorted and unique")
        if self.span is not None and any(not self.span.contains(item) for item in self.excluded_source_frames):
            raise ProtocolError("excluded frame lies outside its sequence span")

    @property
    def frames(self) -> int | None:
        return None if self.span is None else len(self.span)

    @property
    def uses_gradients(self) -> bool:
        return self.partition == "train" and self.sequence_id == 206

    @property
    def supports_counterfactuals(self) -> bool:
        return self.partition == "train" and self.sequence_id in {201, 206}

    def legal_anchors(self, offsets: Sequence[int]) -> tuple[int, ...]:
        if self.span is None:
            raise ProtocolError("hidden sequences need their observed frame count before windowing")
        frozen_offsets = tuple(int(item) for item in offsets)
        excluded = frozenset(self.excluded_source_frames)
        return tuple(
            anchor
            for anchor in range(self.span.start, self.span.stop)
            if all(self.span.contains(anchor + offset) and anchor + offset not in excluded for offset in frozen_offsets)
        )

    def center_frames(self) -> tuple[int, ...]:
        return self.legal_anchors(RELATIVE_TIMES)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    minimum_range_m: float
    maximum_range_m: float
    minimum_anomaly_points: int
    normal_point_alarm_rate: float
    dbscan_eps_m: float
    dbscan_min_samples: int

    def range_mask(self, ranges: object) -> object:
        import numpy as np

        values = np.asarray(ranges, dtype=np.float32)
        return (values >= np.float32(self.minimum_range_m)) & (
            values <= np.float32(self.maximum_range_m)
        )


@dataclass(frozen=True, slots=True)
class DevelopmentWorld:
    world_id: int
    seed: int
    center_frame: int
    world: Mapping[str, object]
    difficulty: tuple[Mapping[str, object], ...]
    mechanism: str


@dataclass(frozen=True, slots=True)
class DevelopmentWorlds:
    format: str
    protocol_schema: int
    sequence_id: int
    status: str
    validation: Mapping[str, bool]
    gate1: Mapping[str, object]
    gate1_evidence_valid: bool
    difficulty_coverage_valid: bool
    in_generator: tuple[DevelopmentWorld, ...]
    generator_held_out: tuple[DevelopmentWorld, ...]

    @property
    def validated(self) -> bool:
        return (
            self.status == "validated_frozen"
            and bool(self.validation)
            and all(self.validation.values())
            and self.gate1.get("status") == "passed_with_real_evidence"
            and self.gate1_evidence_valid
            and self.difficulty_coverage_valid
        )


class AJAEProtocol:
    """Validated immutable view of the schema-30 research route."""

    def __init__(self, document: Mapping[str, object], *, path: Path) -> None:
        self._validate(document)
        self.path = path.expanduser().resolve(strict=True)
        self.schema_version = SCHEMA_VERSION
        self._document = _freeze(document)
        self.authority = self._document["authority"]
        self.task = self._document["task"]
        self.data = self._document["data"]
        self.labels = self._document["labels"]
        self.stu = self._document["stu"]
        self.render = self._document["render"]
        self.window = self._document["window"]
        self.model = self._document["model"]
        self.training = self._document["training"]
        self.development = self._document["development"]
        self.experiments = self._document["experiments"]
        self.evaluation_document = self._document["evaluation"]
        self.visual_reviews = self._document["visual_reviews"]
        self.decision_gates = self._document["decision_gates"]
        self.status = self._document["status"]

        source_data = _mapping(document["data"], "data")
        self.normal_training = self._sequence_from_record(
            _mapping(source_data["normal_training"], "data.normal_training")
        )
        self.development_sequence = self._sequence_from_record(
            _mapping(source_data["development"], "data.development")
        )
        public = _mapping(source_data["public_anomaly_validation"], "public validation")
        public_counts = _mapping(public["sequence_frame_counts"], "public frame counts")
        self.public_validation = tuple(
            SequenceSpec(
                "val", identifier, str(public["role"]), True,
                FrameSpan(0, _integer(public_counts[str(identifier)], f"public frame count {identifier}")),
            )
            for identifier in PUBLIC_ANOMALY_IDS
        )
        hidden = _mapping(source_data["hidden_test"], "hidden test")
        self.hidden_test = tuple(
            SequenceSpec("test", identifier, str(hidden["role"]), False, None)
            for identifier in HIDDEN_TEST_IDS
        )
        all_sequences = (
            self.normal_training,
            self.development_sequence,
            *self.public_validation,
            *self.hidden_test,
        )
        self._sequences = {(item.partition, item.sequence_id): item for item in all_sequences}
        class_map = _mapping(_mapping(document["labels"], "labels")["normal_semantic_class_map"], "normal class map")
        self.normal_training_class_map = MappingProxyType(
            {int(raw): _integer(target, f"normal class {raw}") for raw, target in class_map.items()}
        )
        evaluation = _mapping(document["evaluation"], "evaluation")
        self.evaluation_spec = EvaluationSpec(
            _number(evaluation["minimum_range_m"], "evaluation.minimum_range_m"),
            _number(evaluation["maximum_range_m"], "evaluation.maximum_range_m"),
            _integer(evaluation["minimum_anomaly_points"], "evaluation.minimum_anomaly_points", minimum=1),
            _number(evaluation["normal_point_alarm_rate"], "evaluation.normal_point_alarm_rate"),
            _number(evaluation["dbscan_eps_m"], "evaluation.dbscan_eps_m"),
            _integer(evaluation["dbscan_min_samples"], "evaluation.dbscan_min_samples", minimum=1),
        )
        self.evaluation = self.evaluation_spec

    @staticmethod
    def _sequence_from_record(record: Mapping[str, object]) -> SequenceSpec:
        inclusive = _int_tuple(record["frame_range"], "frame_range")
        if len(inclusive) != 2 or inclusive[1] < inclusive[0]:
            raise ProtocolError("frame_range must be [first,last] inclusive")
        return SequenceSpec(
            str(record["partition"]),
            _integer(record["sequence_id"], "sequence_id"),
            str(record["role"]),
            bool(record["labels_available"]),
            FrameSpan(inclusive[0], inclusive[1] + 1),
            _int_tuple(record["excluded_source_frames"], "excluded_source_frames"),
        )

    @property
    def document(self) -> Mapping[str, object]:
        return self._document  # type: ignore[return-value]

    def plain_document(self) -> dict[str, object]:
        plain = _plain(self._document)
        if not isinstance(plain, dict):
            raise AssertionError("protocol root is not a dictionary")
        return plain

    @property
    def window_frames(self) -> int:
        return WINDOW_FRAMES

    @property
    def relative_times(self) -> tuple[int, ...]:
        return RELATIVE_TIMES

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
            raise ProtocolError(f"sequence {partition}/{sequence_id} is outside this protocol") from error

    def anchors(
        self,
        partition: str,
        sequence_id: int,
        condition: ExperimentCondition | str = ExperimentCondition.B3,
    ) -> tuple[int, ...]:
        selected = ExperimentCondition(condition)
        return self.sequence(partition, sequence_id).legal_anchors(selected.frame_offsets)

    def center_frames(self, partition: str, sequence_id: int) -> tuple[int, ...]:
        return self.anchors(partition, sequence_id, ExperimentCondition.B3)

    def window_frame_ids(
        self,
        partition: str,
        sequence_id: int,
        anchor: int,
        condition: ExperimentCondition | str = ExperimentCondition.B3,
    ) -> tuple[int, ...]:
        selected = ExperimentCondition(condition)
        legal = self.anchors(partition, sequence_id, selected)
        if anchor not in frozenset(legal):
            raise ProtocolError(f"frame {anchor} is not a legal {selected.value} anchor")
        return tuple(anchor + offset for offset in selected.frame_offsets)

    def checkpoint_path(self, project_root: Path | str | None = None) -> Path:
        root = self.path.parent if project_root is None else Path(project_root).expanduser().resolve()
        return (root / str(self.stu["checkpoint"])).resolve()

    def stu_repository_path(self, project_root: Path | str | None = None) -> Path:
        root = self.path.parent if project_root is None else Path(project_root).expanduser().resolve()
        return (root / str(self.stu["repository"])).resolve()

    def development_worlds_path(self, project_root: Path | str | None = None) -> Path:
        root = self.path.parent if project_root is None else Path(project_root).expanduser().resolve()
        return (root / str(self.development["worlds_file"])).resolve()

    def sensor_calibration_path(self, project_root: Path | str | None = None) -> Path:
        root = self.path.parent if project_root is None else Path(project_root).expanduser().resolve()
        return (root / str(self.render["calibration_file"])).resolve()

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "authority": self.authority["document"],
            "train_windows": len(self.normal_training.center_frames()),
            "development_windows": len(self.development_sequence.center_frames()),
            "public_sequences": len(self.public_validation),
            "hidden_sequences": len(self.hidden_test),
            "training_seeds": list(self.training["seeds"]),
            "model_levels": self.model["levels"],
        }

    @classmethod
    def _validate(cls, source: Mapping[str, object]) -> None:
        expected = {
            "schema_version", "authority", "task", "data", "labels", "stu",
            "render", "window", "model", "training", "development",
            "experiments", "evaluation", "visual_reviews", "decision_gates",
            "status",
        }
        _exact_keys(source, expected, "protocol")
        if _integer(source["schema_version"], "schema_version") != SCHEMA_VERSION:
            raise ProtocolError(f"schema_version must be {SCHEMA_VERSION}")
        cls._validate_data(_mapping(source["data"], "data"))
        cls._validate_labels(_mapping(source["labels"], "labels"))
        cls._validate_stu(_mapping(source["stu"], "stu"))
        cls._validate_render(_mapping(source["render"], "render"))
        cls._validate_window(_mapping(source["window"], "window"))
        cls._validate_model(_mapping(source["model"], "model"))
        cls._validate_training(_mapping(source["training"], "training"))
        cls._validate_development(_mapping(source["development"], "development"))
        cls._validate_evaluation(_mapping(source["evaluation"], "evaluation"))
        visual_reviews = _mapping(source["visual_reviews"], "visual_reviews")
        _exact_keys(
            visual_reviews,
            {"global_discipline", "E26-V1", "E45-V1", "E88-V1"},
            "visual_reviews",
        )
        for name in visual_reviews:
            _mapping(visual_reviews[name], f"visual_reviews.{name}")
        experiments = _mapping(source["experiments"], "experiments")
        if set(experiments) != {item.value for item in ExperimentCondition}:
            raise ProtocolError("experiments must define exactly B0 through B5")
        gates = _mapping(source["decision_gates"], "decision_gates")
        _exact_keys(
            gates,
            {"gate1", "gate2", "gate3", "gate4", "criteria", "verdict_rule"},
            "decision_gates",
        )
        if tuple(_list(gates["gate1"], "decision_gates.gate1")) != GATE1_EVIDENCE:
            raise ProtocolError("gate1 evidence identities changed")
        criteria = _mapping(gates["criteria"], "decision_gates.criteria")
        _exact_keys(
            criteria,
            {
                "status",
                "gate1",
                "gate2",
                "gate3",
                "gate4",
                "development_difficulty_coverage",
            },
            "decision_gates.criteria",
        )
        if criteria["status"] not in {
            "unresolved_requires_owner_decision",
            "frozen_before_training",
        }:
            raise ProtocolError("decision-gate criteria status is invalid")
        if criteria["status"] == "frozen_before_training" and any(
            not isinstance(criteria[name], Mapping)
            for name in (
                "gate1",
                "gate2",
                "gate3",
                "gate4",
                "development_difficulty_coverage",
            )
        ):
            raise ProtocolError("all scientific gate criteria must be frozen mappings")

    @staticmethod
    def _validate_data(data: Mapping[str, object]) -> None:
        _exact_keys(data, {"normal_training", "development", "public_anomaly_validation", "hidden_test"}, "data")
        train = _mapping(data["normal_training"], "data.normal_training")
        development = _mapping(data["development"], "data.development")
        if (train["partition"], train["sequence_id"], train["role"]) != (
            "train", 206, "AJAE_parameter_updates_and_renderer_calibration"
        ):
            raise ProtocolError("train/206 must be the only AJAE update and renderer-calibration source")
        if (development["partition"], development["sequence_id"], development["role"]) != (
            "train", 201, "development_only_no_gradients"
        ):
            raise ProtocolError("train/201 must remain development-only")
        public = _mapping(data["public_anomaly_validation"], "public validation")
        hidden = _mapping(data["hidden_test"], "hidden test")
        if _int_tuple(public["sequence_ids"], "public ids") != PUBLIC_ANOMALY_IDS:
            raise ProtocolError("public validation must contain the fixed 19 sequences")
        if _int_tuple(hidden["sequence_ids"], "hidden ids") != HIDDEN_TEST_IDS:
            raise ProtocolError("hidden test must contain the fixed 51 sequences")
        if public.get("method_freeze_required") is not True or hidden.get("method_freeze_required") is not True:
            raise ProtocolError("public and hidden roles require method freeze")

    @staticmethod
    def _validate_labels(labels: Mapping[str, object]) -> None:
        binary = _mapping(labels["binary_anomaly"], "labels.binary_anomaly")
        if set(binary) != {"raw_semantic_0", "raw_semantic_2", "other_nonzero_semantics", "normal_control_return", "anomaly_proxy_return"}:
            raise ProtocolError("binary label sources are incomplete")
        if binary["normal_control_return"] != "normal" or binary["anomaly_proxy_return"] != "anomaly":
            raise ProtocolError("normal controls and anomaly proxies have wrong targets")
        controls = _mapping(labels["normal_control_classes"], "normal control classes")
        if tuple(sorted(map(int, controls))) != NORMAL_CONTROL_SEMANTICS:
            raise ProtocolError("normal-control semantic classes changed")
        if _int_tuple(labels["moving_normal_semantic_ids"], "moving semantics") != MOVING_NORMAL_SEMANTICS:
            raise ProtocolError("moving-normal diagnostic classes changed")

    @staticmethod
    def _validate_stu(stu: Mapping[str, object]) -> None:
        required = {
            "source", "checkpoint", "repository", "voxel_size_m", "input_channels",
            "point_feature_dim", "normal_evidence_dim", "assignment_reliability_dim",
            "no_object_reliability_dim", "query_count", "point_feature_source",
            "normal_evidence_rule", "assignment_reliability_rule",
            "no_object_reliability_rule", "b0_score", "frozen", "full_forward_is_eval",
        }
        _exact_keys(stu, required, "stu")
        dimensions = (
            stu["point_feature_dim"], stu["normal_evidence_dim"],
            stu["assignment_reliability_dim"], stu["no_object_reliability_dim"],
            stu["query_count"], stu["frozen"], stu["full_forward_is_eval"],
        )
        if dimensions != (128, 19, 1, 1, 100, True, True):
            raise ProtocolError("frozen STU interface dimensions changed")
        if not math.isclose(_number(stu["voxel_size_m"], "stu.voxel_size_m"), 0.05):
            raise ProtocolError("STU voxel size must be 0.05 m")
        if "argmax_q" not in str(stu["normal_evidence_rule"]) or stu["b0_score"] != "official_STU_MaxLogit":
            raise ProtocolError("STU must use unique-query evidence and official MaxLogit B0")

    @staticmethod
    def _validate_render(render: Mapping[str, object]) -> None:
        if render.get("source_sequence_id") != 206:
            raise ProtocolError("renderer calibration and templates must come from 206")
        calibration_file = render.get("calibration_file")
        if calibration_file != "runs/ajae/calibration.pt":
            raise ProtocolError("schema 30 has one authoritative sensor calibration path")
        ray = _mapping(render["ray_grid"], "render.ray_grid")
        if (ray.get("beam_count"), ray.get("column_count"), tuple(ray.get("canonical_identity", ()))) != (
            128, 1024, ("beam_id", "azimuth_column")
        ):
            raise ProtocolError("canonical OS1-128 ray identity changed")
        controls = _mapping(render["normal_controls"], "render.normal_controls")
        if _int_tuple(controls["semantic_ids"], "normal control semantics") != NORMAL_CONTROL_SEMANTICS:
            raise ProtocolError("normal-control source classes changed")
        sensor = _mapping(render["sensor_model"], "render.sensor_model")
        if tuple(sensor.get("conditioning", ())) != ("beam", "range", "incidence", "material"):
            raise ProtocolError("return and intensity models must share beam/range/incidence/material conditioning")
        probabilities = _mapping(render["world_types"], "render.world_types")
        values = tuple(_number(value, f"render.world_types.{key}") for key, value in probabilities.items())
        if set(probabilities) != {"pure_normal", "control_only", "mixed", "anomaly_only"} or any(value <= 0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ProtocolError("all four world types need positive probabilities summing to one")

    @staticmethod
    def _validate_window(window: Mapping[str, object]) -> None:
        offline = _mapping(window["offline_main"], "window.offline_main")
        causal = _mapping(window["causal_ablation"], "window.causal_ablation")
        if _signed_int_tuple(offline["frame_offsets"], "offline offsets") != RELATIVE_TIMES:
            raise ProtocolError("offline main window must be center-symmetric five frames")
        # Negative values need direct validation because _int_tuple is non-negative.
        causal_offsets = tuple(causal["frame_offsets"]) if isinstance(causal.get("frame_offsets"), list) else ()
        if causal_offsets != CAUSAL_OFFSETS:
            raise ProtocolError("causal ablation must use [t-4,t]")
        if tuple(window.get("point_identity", ())) != ("frame_id", "beam_id", "azimuth_column") or window.get("file_slot_is_not_identity") is not True:
            raise ProtocolError("point identity must be canonical frame-ray, not file slot")
        if window.get("padding") is not False or window.get("frame_repetition") is not False:
            raise ProtocolError("sequence boundaries cannot be padded or repeated")

    @staticmethod
    def _validate_model(model: Mapping[str, object]) -> None:
        if (
            model.get("input_dim"), model.get("levels"), model.get("pooling"),
            model.get("upsample"), model.get("upsample_neighbors"),
        ) != (150, 4, "per_time_mean_max", "same_time_3NN_with_high_resolution_skip", 3):
            raise ProtocolError("AJAE fixed four-level model identity changed")
        if _signed_int_tuple(model["attention_deltas"], "attention deltas") != RELATIVE_TIMES:
            raise ProtocolError("attention deltas must be -2 through +2")
        voxels = tuple(_number(item, "voxel size") for item in _list(model["voxel_sizes_m"], "voxel sizes"))
        if len(voxels) != 3 or not all(right > left > 0 for left, right in zip(voxels, voxels[1:])):
            raise ProtocolError("L1-L3 require three increasing voxel sizes")
        radii = _float_matrix(model["attention_radii_m"], "attention radii", 4, 5)
        neighbors = _int_matrix(model["neighbors"], "neighbors", 4, 5)
        if any(radius <= 0 for row in radii for radius in row) or any(
            not radii[level + 1][delta] > radii[level][delta]
            for level in range(3) for delta in range(5)
        ):
            raise ProtocolError("every time-stratified radius must grow from L0 to L3")
        if not neighbors:
            raise ProtocolError("time-stratified neighbor budgets are empty")

    @staticmethod
    def _validate_training(training: Mapping[str, object]) -> None:
        banned = {"lambda_cf", "memory_beta", "memory_warmup_worlds", "memory_key", "point_window_weight"}
        if set(training).intersection(banned):
            raise ProtocolError("schema 30 forbids counterfactual memory and extra loss terms")
        if (training.get("source_partition"), training.get("source_sequence_id"), training.get("micro_batch")) != ("train", 206, 1):
            raise ProtocolError("training must use train/206 with one complete window per micro-batch")
        if (training.get("maximum_worlds"), training.get("patience")) != (40, 4):
            raise ProtocolError("formal training must stop by 40 worlds with patience 4")
        seeds = _int_tuple(training["seeds"], "training.seeds")
        if len(seeds) < 3 or len(set(seeds)) != len(seeds):
            raise ProtocolError("formal development requires at least three unique training seeds")
        if "only" not in str(training.get("loss", "")):
            raise ProtocolError("training loss must explicitly contain only balanced BCE")
        probabilities = _mapping(training["world_type_probabilities"], "training world probabilities")
        if set(probabilities) != {"pure_normal", "control_only", "mixed", "anomaly_only"}:
            raise ProtocolError("training must sample all four world types")
        values = tuple(_number(value, f"training world probability {key}") for key, value in probabilities.items())
        if any(value <= 0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ProtocolError("training world probabilities must be positive and sum to one")

    @staticmethod
    def _validate_development(development: Mapping[str, object]) -> None:
        _exact_keys(
            development,
            {
                "phase6_version", "worlds_file", "sequence_id",
                "in_generator_worlds", "generator_held_out_worlds",
                "held_out_affects_selection", "pure_normal_is_separate",
                "qualification", "fixed_world_evaluation",
                "checkpoint_selection", "difficulty_statistics",
                "boundary_leakage_radius_m", "position_score_scale",
            },
            "development",
        )
        if (
            development.get("phase6_version"),
            development.get("worlds_file"), development.get("sequence_id"),
            development.get("in_generator_worlds"),
            development.get("generator_held_out_worlds"),
            development.get("held_out_affects_selection"),
        ) != ("E57-v2", "dev.json", 201, 24, 6, False):
            raise ProtocolError("development must preserve E57-v2 and the fixed 24+6 split on 201")
        qualification = _mapping(
            development["qualification"], "development.qualification"
        )
        _exact_keys(
            qualification,
            {
                "status", "source_candidate_bank", "selection",
                "hard_requirements", "descriptive_characterization",
            },
            "development.qualification",
        )
        source_bank = _mapping(
            qualification["source_candidate_bank"],
            "development.qualification.source_candidate_bank",
        )
        if (
            qualification.get("status") != "frozen_before_e57"
            or source_bank.get("path") != "runs/ajae/e45b-v2_bank_1024.npz"
            or source_bank.get("sha256")
            != "d3088e29e4c6179999ccb34088dae558fa402bf6b1455394acdc99cac4118463"
            or source_bank.get("scientific_array_sha256")
            != "f4fb2081b346c686e2d6930a03e3f17bb6c6d3eee4fcfc16984c1a9c1d8de4f5"
            or source_bank.get("candidate_worlds") != 1024
            or source_bank.get("e45b_matching_or_e48_scores_forbidden") is not True
        ):
            raise ProtocolError("E57-v2 source-bank identity or score isolation changed")
        selection = _mapping(
            qualification["selection"], "development.qualification.selection"
        )
        if (
            selection.get("selected_worlds") != 24
            or selection.get("rule")
            != "rank_normalized_generator_descriptors_center_then_greedy_maximin_hash_tie"
            or tuple(selection.get("descriptors", ()))
            != (
                "control_Nvis", "control_O", "control_d", "control_V",
                "proxy_Nvis", "proxy_O", "proxy_d", "proxy_V",
            )
            or selection.get("model_outputs_forbidden") is not True
            or selection.get("exact_bin_quotas_forbidden") is not True
        ):
            raise ProtocolError("E57-v2 model-independent selection rule changed")
        hard = _mapping(
            qualification["hard_requirements"],
            "development.qualification.hard_requirements",
        )
        if (
            hard.get("legal_mixed_worlds"),
            hard.get("minimum_center_anomaly_points_per_world"),
            hard.get("minimum_center_normal_points_per_world"),
            hard.get("minimum_multiframe_worlds_per_label"),
            hard.get("development_sequence"),
            hard.get("gradients_forbidden"),
        ) != (24, 5, 1, 12, "train/201", True):
            raise ProtocolError("E57-v2 hard non-degeneracy requirements changed")
        characterization = _mapping(
            qualification["descriptive_characterization"],
            "development.qualification.descriptive_characterization",
        )
        if (
            characterization.get("status") != "nonblocking"
            or tuple(characterization.get("statistics", ())) != ("d", "Nvis", "O", "V")
            or tuple(characterization.get("distance_bin_edges_m", ())) != (10.0, 20.0, 30.0)
            or tuple(characterization.get("Nvis_bin_edges", ())) != (8, 32, 128)
            or tuple(characterization.get("O_bin_edges", ())) != (0.25, 0.5, 0.75)
            or tuple(characterization.get("V_values", ())) != (1, 2, 3, 4, 5)
        ):
            raise ProtocolError("E59/E60 characterization must remain complete and nonblocking")
        world_evaluation = _mapping(
            development["fixed_world_evaluation"],
            "development.fixed_world_evaluation",
        )
        _exact_keys(
            world_evaluation,
            {"status", "scope"},
            "development.fixed_world_evaluation",
        )
        if world_evaluation.get("status") not in {
            "unresolved_requires_owner_decision",
            "frozen_before_training",
        }:
            raise ProtocolError("fixed-world evaluation must declare its freeze status")
        if world_evaluation.get("status") == "frozen_before_training" and not isinstance(
            world_evaluation.get("scope"), Mapping
        ):
            raise ProtocolError("frozen fixed-world evaluation requires an explicit scope")
        selection = _mapping(development["checkpoint_selection"], "checkpoint selection")
        if selection.get("status") not in {
            "proposed_requires_owner_confirmation",
            "frozen_before_training",
        } or selection.get("held_out_input_forbidden") is not True:
            raise ProtocolError("checkpoint selection must exclude held-out worlds and declare its freeze status")
        if _number(selection["tie_tolerance"], "checkpoint tie tolerance") <= 0:
            raise ProtocolError("checkpoint tie tolerance must be positive")
        difficulty = _mapping(development["difficulty_statistics"], "difficulty statistics")
        if set(difficulty) != {"Nvis", "O", "d", "V"}:
            raise ProtocolError("development difficulty must define Nvis, O, d, and V")

    @staticmethod
    def _validate_evaluation(evaluation: Mapping[str, object]) -> None:
        if (
            evaluation.get("minimum_range_m"), evaluation.get("maximum_range_m"),
            evaluation.get("minimum_range_inclusive"), evaluation.get("maximum_range_inclusive"),
            evaluation.get("minimum_anomaly_points"), evaluation.get("score_fusion"),
        ) != (2.5, 50.0, True, True, 5, "equal_mean_of_probabilities_by_frame_and_canonical_ray"):
            raise ProtocolError("official point range, frame gate, or probability fusion changed")
        if tuple(evaluation.get("point_metrics", ())) != ("AP", "AUROC", "FPR95"):
            raise ProtocolError("official point metrics changed")
        frame_domain = _mapping(
            evaluation["comparison_frame_domain"], "comparison frame domain"
        )
        if (
            frame_domain.get("status")
            not in {
                "proposed_requires_owner_confirmation",
                "frozen_before_evaluation",
            }
            or
            frame_domain.get("rule")
            != "intersection_of_complete_centered_q0_and_complete_causal_current_frames"
            or _signed_int_tuple(
                frame_domain.get("required_source_offsets"),
                "comparison frame offsets",
            )
            != (-4, -3, -2, -1, 0, 1, 2)
            or tuple(frame_domain.get("applies_to", ()))
            != tuple(f"B{index}" for index in range(6))
            or frame_domain.get("padding_or_zero_fill_forbidden") is not True
            or frame_domain.get("coverage_manifest_required") is not True
        ):
            raise ProtocolError(
                "all B0--B5 comparisons require the same complete-window frame domain"
            )
        if evaluation.get("threshold_comparison") != "score_strictly_greater_than_threshold":
            raise ProtocolError("object threshold comparison must remain strict")


def _finite_statistics(value: Mapping[str, object], name: str) -> None:
    found = False
    stack: list[tuple[str, object]] = [(name, value)]
    while stack:
        path, item = stack.pop()
        if isinstance(item, Mapping):
            stack.extend((f"{path}.{key}", nested) for key, nested in item.items())
        elif isinstance(item, (list, tuple)):
            stack.extend((f"{path}[{index}]", nested) for index, nested in enumerate(item))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            _number(item, path)
            found = True
    if not found:
        raise ProtocolError(f"{name} must contain at least one finite numeric statistic")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate1_verdict_is_explicit(
    value: object, expected_criterion: object
) -> bool:
    if not isinstance(value, Mapping) or not isinstance(expected_criterion, Mapping):
        return False
    criterion_id = expected_criterion.get("criterion_id")
    return (
        value.get("passed") is True
        and value.get("decided_before_training") is True
        and isinstance(criterion_id, str)
        and bool(criterion_id.strip())
        and value.get("criterion") == criterion_id
        and isinstance(value.get("judgment"), str)
        and bool(str(value.get("judgment")).strip())
    )


def _validate_gate1_evidence(
    evidence: Mapping[str, object],
    *,
    protocol: AJAEProtocol,
) -> bool:
    """Validate evidence identity and return whether all four verdicts explicitly pass."""

    calibration_digest = _sha256_file(protocol.sensor_calibration_path())
    criteria_document = protocol.decision_gates["criteria"]
    gate1_criteria = (
        criteria_document.get("gate1")
        if criteria_document.get("status") == "frozen_before_training"
        else None
    )
    all_verdicts_pass = isinstance(gate1_criteria, Mapping)
    for name in GATE1_EVIDENCE:
        raw_item = evidence[name]
        if raw_item is None:
            all_verdicts_pass = False
            continue
        item = _mapping(raw_item, f"dev.gate1.evidence.{name}")
        _finite_statistics(item, name)
        identity = _mapping(item.get("input_identity"), f"{name}.input_identity")
        required_identity = {
            "protocol_schema": SCHEMA_VERSION,
            "sequence_id": 206,
            "partition": "train",
            "first_frame": 0,
            "last_frame": 448,
            "frame_count": 449,
            "calibration_sha256": calibration_digest,
        }
        if any(identity.get(key) != value for key, value in required_identity.items()):
            # A replaced authoritative calibration invalidates old evidence.  It
            # must keep the development gate closed without making the pending
            # development-world document unreadable.
            all_verdicts_pass = False
            continue
        audited_returns = _integer(
            identity.get("audited_real_returns_all_frames"),
            f"{name}.audited_real_returns_all_frames",
            minimum=1,
        )
        provenance = _mapping(item.get("provenance"), f"{name}.provenance")
        if not provenance:
            raise ProtocolError(f"{name} provenance cannot be empty")

        if name == "ray_slot_audit":
            audit = _mapping(item.get("audit"), "ray_slot_audit.audit")
            layout = _mapping(audit.get("slot_layout"), "ray_slot_audit.slot_layout")
            round_trip = _mapping(
                audit.get("round_trip"), "ray_slot_audit.round_trip"
            )
            if (
                _integer(audit.get("frame_count"), "ray_slot_audit.frame_count", minimum=1)
                != 17
                or _integer(
                    layout.get("forward_reverse_mismatches"),
                    "ray_slot_audit.forward_reverse_mismatches",
                    minimum=0,
                )
                != 0
                or _number(
                    round_trip.get("maximum_point_error_m"),
                    "ray_slot_audit.maximum_point_error_m",
                )
                > ROUND_TRIP_POINT_TOLERANCE_M
                or _number(
                    round_trip.get("maximum_direction_error_rad"),
                    "ray_slot_audit.maximum_direction_error_rad",
                )
                > ROUND_TRIP_DIRECTION_TOLERANCE_RAD
            ):
                raise ProtocolError("ray-slot evidence failed its exact identity checks")
        elif name == "range_image_round_trip":
            aggregate = _mapping(item.get("aggregate"), "range_image_round_trip.aggregate")
            if (
                _integer(
                    aggregate.get("return_count_mismatch_frames"),
                    "range_image_round_trip.return_count_mismatch_frames",
                    minimum=0,
                )
                != 0
                or _integer(
                    aggregate.get("total_real_returns"),
                    "range_image_round_trip.total_real_returns",
                    minimum=1,
                )
                != audited_returns
                or _number(
                    aggregate.get("maximum_point_error_m"),
                    "range_image_round_trip.maximum_point_error_m",
                )
                > ROUND_TRIP_POINT_TOLERANCE_M
                or _number(
                    aggregate.get("maximum_range_error_m"),
                    "range_image_round_trip.maximum_range_error_m",
                )
                > ROUND_TRIP_POINT_TOLERANCE_M
                or _number(
                    aggregate.get("maximum_direction_error_rad"),
                    "range_image_round_trip.maximum_direction_error_rad",
                )
                > ROUND_TRIP_DIRECTION_TOLERANCE_RAD
            ):
                raise ProtocolError("range-image evidence does not preserve all returns")
        elif name == "render_source_leakage":
            audit = _mapping(item.get("audit"), "render_source_leakage.audit")
            train_groups = {
                _integer(value, "render_source_leakage.train_group", minimum=0)
                for value in _list(audit.get("train_groups"), "render_source_leakage.train_groups")
            }
            test_groups = {
                _integer(value, "render_source_leakage.test_group", minimum=0)
                for value in _list(audit.get("test_groups"), "render_source_leakage.test_groups")
            }
            spatial_match = _mapping(
                item.get("spatial_match_distance_m"),
                "render_source_leakage.spatial_match_distance_m",
            )
            matched = _integer(
                item.get("matched_samples_per_class"),
                "render_source_leakage.matched_samples_per_class",
                minimum=1,
            )
            if (
                audit.get("split_unit") != "frame_or_world_group"
                or not train_groups
                or not test_groups
                or not train_groups.isdisjoint(test_groups)
                or _integer(audit.get("train_samples"), "source train samples", minimum=1) < 1
                or _integer(audit.get("test_samples"), "source test samples", minimum=1) < 1
                or not 0.0 <= _number(
                    audit.get("balanced_accuracy"), "source balanced accuracy"
                ) <= 1.0
                or not 0.0 <= _number(audit.get("auroc"), "source AUROC") <= 1.0
                or _integer(spatial_match.get("count"), "source match count", minimum=1)
                != matched
            ):
                raise ProtocolError("source-leakage evidence is not group-disjoint and matched")
        else:
            statistics = _mapping(item.get("statistics"), "beam_range_intensity.statistics")
            comparison = _mapping(
                item.get("comparison_summary"),
                "beam_range_intensity.comparison_summary",
            )
            if (
                _integer(statistics.get("frames"), "beam_range_intensity.frames", minimum=1)
                != 449
                or _integer(
                    statistics.get("normal_control_returns"),
                    "beam_range_intensity.normal_control_returns",
                    minimum=1,
                )
                < 1
                or not comparison
            ):
                raise ProtocolError("beam-range-intensity evidence lacks full sensor coverage")

        conclusion = item.get("threshold_conclusion")
        if name == "range_image_round_trip" and conclusion is None:
            conclusion = _mapping(
                item.get("aggregate"), "range_image_round_trip.aggregate"
            ).get("threshold_conclusion")
        expected_criterion = (
            gate1_criteria.get(name)
            if isinstance(gate1_criteria, Mapping)
            else None
        )
        all_verdicts_pass &= _gate1_verdict_is_explicit(
            conclusion, expected_criterion
        )
    return all_verdicts_pass


def _development_difficulty_coverage_is_valid(
    worlds: Sequence[DevelopmentWorld],
    criteria: object,
) -> bool:
    """Check only E57-v2 cross-frame non-degeneracy, never bin quotas."""

    if not isinstance(criteria, Mapping):
        return False
    minimum_worlds = criteria.get("minimum_multiframe_worlds_per_label")
    if type(minimum_worlds) is not int or minimum_worlds < 1:
        return False
    multiframe_worlds: dict[str, set[int]] = {
        "normal-control": set(),
        "anomaly-proxy": set(),
    }
    for world in worlds:
        objects = _list(world.world.get("objects"), "development world objects")
        labels = {
            _integer(_mapping(item, "development object").get("object_id"), "object_id"):
            str(_mapping(item, "development object").get("label"))
            for item in objects
        }
        for entry in world.difficulty:
            label = labels[int(entry["object_id"])]
            if int(entry["V"]) >= 2:
                multiframe_worlds[label].add(world.world_id)
    return all(len(items) >= minimum_worlds for items in multiframe_worlds.values())


def load_development_worlds(
    path: Path | str,
    *,
    protocol: AJAEProtocol,
) -> DevelopmentWorlds:
    """Load fixed 201 worlds; booleans alone can never validate gate 1."""

    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        source = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read development worlds: {resolved}") from error
    root = _mapping(source, "dev.json")
    expected = {
        "format", "protocol_schema", "sequence_id", "status", "validation",
        "gate1", "in_generator", "generator_held_out",
    }
    _exact_keys(root, expected, "dev.json")
    if (root["format"], root["protocol_schema"], root["sequence_id"]) != (
        "ajae-development-worlds-v2", SCHEMA_VERSION, 201
    ):
        raise ProtocolError("dev.json is not the schema-30 fixed-world format")
    validation_source = _mapping(root["validation"], "dev.validation")
    required_validation = {
        "physical_placement", "sequence_visibility", "difficulty_coverage",
        "normal_control_and_proxy_composition", "held_out_mechanism_isolation",
    }
    _exact_keys(validation_source, required_validation, "dev.validation")
    validation: dict[str, bool] = {}
    for key, value in validation_source.items():
        if type(value) is not bool:
            raise ProtocolError(f"dev.validation.{key} must be boolean")
        validation[key] = value
    gate1_source = _mapping(root["gate1"], "dev.gate1")
    _exact_keys(gate1_source, {"status", "evidence"}, "dev.gate1")
    evidence = _mapping(gate1_source["evidence"], "dev.gate1.evidence")
    _exact_keys(evidence, set(GATE1_EVIDENCE), "dev.gate1.evidence")
    evidence_valid = _validate_gate1_evidence(evidence, protocol=protocol)
    try:
        from .render import (
            HeldOutTorusShape,
            NormalTemplateShape,
            ShapeSpec,
            WorldSpec,
        )
    except ImportError:  # pragma: no cover - direct script execution
        from render import (  # type: ignore[no-redef]
            HeldOutTorusShape,
            NormalTemplateShape,
            ShapeSpec,
            WorldSpec,
        )

    def parse_group(value: object, name: str, mechanism: str) -> tuple[DevelopmentWorld, ...]:
        records = _list(value, name)
        parsed: list[DevelopmentWorld] = []
        for index, record in enumerate(records):
            item = _mapping(record, f"{name}[{index}]")
            _exact_keys(
                item,
                {
                    "world_id",
                    "seed",
                    "center_frame",
                    "world",
                    "difficulty",
                    "mechanism",
                },
                f"{name}[{index}]",
            )
            center_frame = _integer(item["center_frame"], "center_frame")
            if center_frame not in frozenset(protocol.development_sequence.center_frames()):
                raise ProtocolError(
                    "development center_frame is not a legal unexcluded five-frame center"
                )
            world = _mapping(item["world"], f"{name}[{index}].world")
            try:
                parsed_world = WorldSpec.from_dict(world)
            except (TypeError, ValueError) as error:
                raise ProtocolError(
                    f"{name}[{index}] is not a valid authoritative WorldSpec"
                ) from error
            if (
                parsed_world.seed != _integer(item["seed"], "seed")
                or parsed_world.source_sequence_id != 201
                or parsed_world.world_type != "mixed"
            ):
                raise ProtocolError("every fixed development world must be mixed train/201")
            difficulty_values = tuple(
                _mapping(entry, f"{name}[{index}].difficulty")
                for entry in _list(item["difficulty"], f"{name}[{index}].difficulty")
            )
            for entry in difficulty_values:
                if set(entry) != {"object_id", "Nvis", "O", "d", "V"}:
                    raise ProtocolError("every entity difficulty record must define object_id,Nvis,O,d,V")
                if _number(entry["Nvis"], "difficulty.Nvis") <= 0:
                    raise ProtocolError("difficulty Nvis must be positive")
                if not 0 <= _number(entry["O"], "difficulty.O") <= 1:
                    raise ProtocolError("difficulty O must lie in [0,1]")
                if _number(entry["d"], "difficulty.d") <= 0:
                    raise ProtocolError("difficulty d must be positive")
                if not 1 <= _integer(entry["V"], "difficulty.V", minimum=1) <= 5:
                    raise ProtocolError("difficulty V must lie in [1,5]")
            difficulty_ids = [int(entry["object_id"]) for entry in difficulty_values]
            if (
                len(difficulty_ids) != len(parsed_world.objects)
                or len(set(difficulty_ids)) != len(difficulty_ids)
                or set(difficulty_ids)
                != {obj.object_id for obj in parsed_world.objects}
            ):
                raise ProtocolError(
                    "difficulty records must identify every world object exactly once"
                )
            actual_mechanism = str(item["mechanism"])
            if mechanism == "in_generator" and actual_mechanism != "in_generator":
                raise ProtocolError("in-generator world uses a held-out mechanism")
            if mechanism == "held_out" and actual_mechanism != "torus_SDF":
                raise ProtocolError("held-out worlds must use the unseen torus_SDF mechanism")
            objects = _list(world.get("objects"), f"{name}[{index}].world.objects")
            object_records = tuple(_mapping(obj, "world object") for obj in objects)
            labels = {str(obj.get("label")) for obj in object_records}
            if labels != {"normal-control", "anomaly-proxy"}:
                raise ProtocolError(
                    "every fixed development world must contain controls and proxies"
                )
            expected_shape_kind = {
                "normal-control": "normal-template-convex-hull",
                "anomaly-proxy": (
                    "procedural-csg" if mechanism == "in_generator" else "held-out-torus-sdf"
                ),
            }
            for obj in object_records:
                label = str(obj.get("label"))
                shape = _mapping(obj.get("shape"), "world object shape")
                if label not in expected_shape_kind or shape.get("kind") != expected_shape_kind[label]:
                    raise ProtocolError(
                        "development object label and generator mechanism are inconsistent"
                    )
            for obj in parsed_world.objects:
                if obj.label == "normal-control" and not isinstance(
                    obj.shape, NormalTemplateShape
                ):
                    raise ProtocolError("normal controls must use 206 normal templates")
                if obj.label == "anomaly-proxy":
                    expected_type = (
                        ShapeSpec if mechanism == "in_generator" else HeldOutTorusShape
                    )
                    if not isinstance(obj.shape, expected_type):
                        raise ProtocolError(
                            "parsed anomaly shape violates generator-mechanism isolation"
                        )
            parsed.append(
                DevelopmentWorld(
                    _integer(item["world_id"], "world_id"),
                    _integer(item["seed"], "seed"),
                    center_frame,
                    _freeze(world),  # type: ignore[arg-type]
                    tuple(_freeze(entry) for entry in difficulty_values),  # type: ignore[arg-type]
                    actual_mechanism,
                )
            )
        return tuple(parsed)

    in_generator = parse_group(root["in_generator"], "in_generator", "in_generator")
    held_out = parse_group(root["generator_held_out"], "generator_held_out", "held_out")
    if len(in_generator) != int(protocol.development["in_generator_worlds"]) or len(held_out) != int(protocol.development["generator_held_out_worlds"]):
        raise ProtocolError("dev.json does not contain the fixed 24+6 worlds")
    identifiers = tuple(item.world_id for item in (*in_generator, *held_out))
    if identifiers != tuple(range(30)):
        raise ProtocolError("development world IDs must be exactly 0 through 29")
    qualification = protocol.development["qualification"]
    difficulty_coverage_valid = _development_difficulty_coverage_is_valid(
        in_generator,
        qualification["hard_requirements"],
    )
    return DevelopmentWorlds(
        str(root["format"]), SCHEMA_VERSION, 201, str(root["status"]),
        MappingProxyType(validation),
        _freeze(gate1_source),  # type: ignore[arg-type]
        evidence_valid,
        difficulty_coverage_valid,
        in_generator, held_out,
    )


def load_protocol(path: Path | str = DEFAULT_PROTOCOL_PATH) -> AJAEProtocol:
    """Load and validate the single active AJAE protocol."""

    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        source = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read protocol: {resolved}") from error
    return AJAEProtocol(_mapping(source, "protocol"), path=resolved)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the AJAE schema-30 route.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    output: dict[str, object] = protocol.summary()
    if args.development:
        worlds = load_development_worlds(protocol.development_worlds_path(), protocol=protocol)
        output["development"] = {
            "status": worlds.status,
            "validated": worlds.validated,
            "in_generator": len(worlds.in_generator),
            "held_out": len(worlds.generator_held_out),
            "gate1": worlds.gate1.get("status"),
        }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
