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
                "decision_metric_scale",
                "development_difficulty_coverage",
                "B4_contribution",
            },
            "decision_gates.criteria",
        )
        if criteria["status"] != "frozen_before_training":
            raise ProtocolError("decision-gate criteria status is invalid")
        if any(
            not isinstance(criteria[name], Mapping)
            for name in (
                "gate1",
                "gate2",
                "gate3",
                "gate4",
                "decision_metric_scale",
                "development_difficulty_coverage",
                "B4_contribution",
            )
        ):
            raise ProtocolError("all scientific gate criteria must be frozen mappings")
        metric_scale = _mapping(
            criteria["decision_metric_scale"], "decision metric scale"
        )
        if (
            tuple(metric_scale.get("scale", ())) != (0.0, 1.0)
            or tuple(metric_scale.get("reported_percent_metrics", ()))
            != ("AP", "AUROC", "FPR95")
            or metric_scale.get("reported_to_decision_conversion") != "divide by 100"
            or metric_scale.get("normal_set_FPR_already_on_decision_scale") is not True
            or metric_scale.get("checkpoint_tie_tolerance_remains_on_reported_percent_scale")
            != 0.001
        ):
            raise ProtocolError("decision metric scale is not the result-blind freeze")
        gate2 = _mapping(criteria["gate2"], "Gate 2 criteria")
        e76 = _mapping(gate2.get("E76"), "E76 criteria")
        if (
            gate2.get("maximum_mean_safety_worsening") != 0.03
            or gate2.get("safety_worsening_is_signed_not_absolute") is not True
            or e76.get("prerequisite") != "E75 PASS"
            or tuple(e76.get("models", ()))
            != ("B0", "B1 seed 0", "B1 seed 1", "B1 seed 2")
            or tuple(e76.get("fold_A_world_ids", ()))
            != (2, 3, 6, 8, 9, 11, 13, 18, 20, 21, 22)
            or tuple(e76.get("fold_B_world_ids", ()))
            != (0, 1, 4, 7, 10, 12, 14, 15, 16, 17, 19, 23)
            or e76.get("per_seed_values_reported_but_not_independently_gated") is not True
        ):
            raise ProtocolError("E76 result-blind safety definition changed")

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
        if (training.get("maximum_worlds"), training.get("patience")) != (25, 4):
            raise ProtocolError("formal training must stop by 25 worlds with patience 4")
        revision = _mapping(
            training.get("result_blind_budget_revision"),
            "training result-blind budget revision",
        )
        _exact_keys(
            revision,
            {
                "version", "status", "previous_maximum_worlds", "maximum_worlds",
                "scope_conditions", "development_metric_values_read",
                "checkpoint_prefix_reuse", "prefix_reuse_rule",
            },
            "training.result_blind_budget_revision",
        )
        prefix = _mapping(
            revision.get("checkpoint_prefix_reuse"),
            "training result-blind prefix reuse",
        )
        _exact_keys(
            prefix,
            {
                "condition", "seed", "progress_sha256",
                "scientific_identity_sha256", "phase", "cursor",
                "history_worlds", "completed_development_evaluations",
            },
            "training.result_blind_budget_revision.checkpoint_prefix_reuse",
        )
        cursor = _mapping(prefix.get("cursor"), "training result-blind cursor")
        if (
            revision.get("version") != "E74-result-blind-budget-reduction-v1"
            or revision.get("status") != "frozen_before_result_exposure"
            or revision.get("previous_maximum_worlds") != 40
            or revision.get("maximum_worlds") != 25
            or tuple(revision.get("scope_conditions", ())) != ("B1", "B2", "B3")
            or revision.get("development_metric_values_read") is not False
            or (prefix.get("condition"), prefix.get("seed"), prefix.get("phase"))
            != ("B1", 0, "windows")
            or prefix.get("progress_sha256")
            != "f2df8555226e2ca7b9b8ba70066e130659dc1f89818165f3f31bd806730b20df"
            or prefix.get("scientific_identity_sha256")
            != "e0da006e987252a85be37a80f7e78a908c7ffe56245d8ed36fb918067cb56d42"
            or dict(cursor)
            != {
                "world_index": 22,
                "block_index": 15,
                "window_index": 0,
                "windows_completed": 225,
            }
            or prefix.get("history_worlds") != 22
            or prefix.get("completed_development_evaluations") != 4
            or not isinstance(revision.get("prefix_reuse_rule"), str)
            or not revision.get("prefix_reuse_rule")
        ):
            raise ProtocolError("result-blind 40-to-25 world revision identity changed")
        if training.get("deterministic_algorithms") is not True:
            raise ProtocolError("formal training must use deterministic algorithms")
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
        smoke = _mapping(training["e73_smoke"], "training.e73_smoke")
        pure = _mapping(smoke["pure_normal"], "training.e73_smoke.pure_normal")
        mixed = _mapping(smoke["mixed"], "training.e73_smoke.mixed")
        if (
            smoke.get("status") not in {
                "frozen_before_model_execution", "formal_pass"
            }
            or smoke.get("seed") != 73002026
            or smoke.get("source_artifact")
            != "runs/ajae/e26_v2_world_builder.npz"
            or smoke.get("source_artifact_sha256")
            != "2653f705d2e890d99cda732a7a00387b5621cd05abb9c4681c7a9f284c34363c"
            or smoke.get("b0_reference") != "runs/ajae/e72_b0_reference.npz"
            or smoke.get("b0_reference_sha256")
            != "208487d5c91b131856e908988cf6d955305fa09364450d509e32f617295b5863"
            or dict(pure)
            != {
                "row": 0,
                "world_seed": 2600000,
                "center_frame": 312,
                "world_sha256": "27a1654c7241bb616964a3b47502c60b5376cfef189392f9eb2e4c76154246ea",
            }
            or dict(mixed)
            != {
                "row": 128,
                "world_seed": 2600128,
                "center_frame": 440,
                "world_sha256": "c83062ae310e2d468eaec74471235dabfa41b1405292f4229d8d0ce718b17a7a",
            }
            or smoke.get("optimizer_updates") != 1
            or smoke.get("micro_batches") != 2
            or _number(
                smoke["partial_accumulation_uses_frozen_factor"],
                "E73 partial accumulation factor",
            )
            != 4.0
            or _number(
                smoke["loss_reproduction_absolute_tolerance"],
                "E73 loss reproduction tolerance",
            )
            != 1.0e-7
            or _number(
                smoke["parameter_reproduction_absolute_tolerance"],
                "E73 parameter reproduction tolerance",
            )
            != 1.0e-7
            or smoke.get("model_quality_use_forbidden") is not True
        ):
            raise ProtocolError("E73 smoke identity changed")
        if smoke.get("status") == "formal_pass":
            result = _mapping(smoke["result"], "training.e73_smoke.result")
            if (
                result.get("path") != "runs/ajae/e73_b1_smoke.npz"
                or result.get("artifact_sha256")
                != "7d4eed7af2207cfffe10501cbbcf582f6a16c3fd1f258351a299f32dc540cff3"
                or result.get("scientific_array_sha256")
                != "2cf7449e0b101c12ba47d7049cc777a3332f5b66d5334faf373b6e5fab2d218e"
                or result.get("protocol_sha256_at_execution")
                != "54565db9e6887ffa62fed4ddb8bb9951b4626bebcf3cb25ae0f2bdf1d2299ebc"
                or any(
                    result.get(name) != 0
                    for name in (
                        "identity_errors", "gradient_errors", "stu_errors",
                        "checkpoint_errors", "reproduction_errors",
                        "loss_reproduction_error", "parameter_reproduction_error",
                    )
                )
                or result.get("independent_read_only_validation") is not True
            ):
                raise ProtocolError("E73 formal result identity changed")

    @staticmethod
    def _validate_development(development: Mapping[str, object]) -> None:
        _exact_keys(
            development,
            {
                "phase6_version", "worlds_file", "sequence_id",
                "in_generator_worlds", "generator_held_out_worlds",
                "held_out_affects_selection", "pure_normal_is_separate",
                "safety_sets", "qualification", "fixed_world_evaluation",
                "checkpoint_selection", "e63_freeze", "difficulty_statistics",
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
        safety = _mapping(development["safety_sets"], "development.safety_sets")
        _exact_keys(
            safety,
            {"status", "pure_normal", "moving_normal", "static_match",
             "pass_conditions", "claim_limit"},
            "development.safety_sets",
        )
        pure = _mapping(safety["pure_normal"], "E61 pure-normal safety")
        moving = _mapping(safety["moving_normal"], "E61 moving-normal safety")
        match = _mapping(safety["static_match"], "E61 static matching")
        if (
            safety.get("status") != "frozen_before_e61"
            or (pure.get("partition"), pure.get("sequence_id"),
                tuple(pure.get("frame_range", ())), tuple(pure.get("range_m", ())),
                pure.get("expected_points"), pure.get("labels_are_evaluation_only"))
            != ("train", 201, (4, 681), (2.5, 50.0), 48828507, True)
            or (moving.get("partition"), moving.get("sequence_id"),
                tuple(moving.get("frame_range", ())), tuple(moving.get("range_m", ())),
                tuple(moving.get("semantic_ids", ())), moving.get("expected_points"),
                moving.get("retain_all_eligible_points"),
                moving.get("held_out_or_unseen_generalization_claim_forbidden"),
                moving.get("labels_are_evaluation_only"))
            != ("train", 206, (0, 448), (2.5, 50.0),
                MOVING_NORMAL_SEMANTICS, 13011, True, True, True)
        ):
            raise ProtocolError("E61 pure/moving safety identities changed")
        expected_mapping = {
            "252": 10, "253": 31, "254": 30, "255": 32,
            "256": 16, "257": 13, "258": 18, "259": 20,
        }
        if (
            dict(_mapping(match.get("moving_to_static_semantic"),
                          "E61 semantic matching")) != expected_mapping
            or tuple(match.get("range_bin_edges_m", ()))
            != (2.5, 10.0, 20.0, 30.0, 50.0)
            or match.get("identity_hash_namespace") != "E61-static-match-v1"
            or tuple(match.get("identity_hash_fields", ()))
            != ("sequence_id", "frame_id", "canonical_ray_id")
            or match.get("replacement") is not False
            or match.get("point_reuse") is not False
            or match.get("same_frame_matching") is not False
            or match.get("insufficient_cell_coverage_is_nonblocking") is not True
            or match.get("unmatched_moving_points_remain_in_moving_safety") is not True
            or set(match.get("forbidden_matching_inputs", ()))
            != {"intensity", "occlusion", "point density", "STU feature",
                "voxel density", "AJAE score", "STU anomaly score"}
        ):
            raise ProtocolError("E61 static matching rule changed")
        qualification = _mapping(
            development["qualification"], "development.qualification"
        )
        _exact_keys(
            qualification,
            {
                "status", "source_candidate_bank", "selection",
                "hard_requirements", "descriptive_characterization",
                "held_out_diagnostics",
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
        held_out = _mapping(
            qualification["held_out_diagnostics"],
            "development.qualification.held_out_diagnostics",
        )
        held_out_source = _mapping(
            held_out["source_e57_artifact"],
            "development.qualification.held_out_diagnostics.source_e57_artifact",
        )
        if (
            held_out.get("status") != "frozen_before_e58"
            or held_out_source.get("path")
            != "runs/ajae/e57_development_worlds.npz"
            or held_out_source.get("sha256")
            != "b14efc1aad86ac67b5bf7c8631f02b2e68664e071b747b7b210d5f7a30f5d123"
            or held_out_source.get("scientific_array_sha256")
            != "590c467da2dec0a161688f2587dc1c37cea2b0f42f326b9918fd6dc9df81f6ec"
            or held_out.get("worlds") != 6
            or held_out.get("shape_mechanism") != "held-out-torus-sdf"
            or held_out.get("seed_namespace") != "E58-held-out-torus-v1"
            or held_out.get("selection_rule")
            != "lowest namespace hash among legal visible center-evaluable replacements of the 24 E57 worlds"
            or held_out.get("model_outputs_forbidden") is not True
            or held_out.get(
                "training_checkpoint_threshold_and_pass_use_forbidden"
            ) is not True
        ):
            raise ProtocolError("E58 held-out identity or isolation rule changed")
        world_evaluation = _mapping(
            development["fixed_world_evaluation"],
            "development.fixed_world_evaluation",
        )
        _exact_keys(
            world_evaluation,
            {"status", "scope", "b0_reference"},
            "development.fixed_world_evaluation",
        )
        if world_evaluation.get("status") != "frozen_before_training":
            raise ProtocolError("fixed-world evaluation must declare its freeze status")
        if not isinstance(world_evaluation.get("scope"), Mapping):
            raise ProtocolError("frozen fixed-world evaluation requires an explicit scope")
        b0 = _mapping(
            world_evaluation["b0_reference"],
            "development.fixed_world_evaluation.b0_reference",
        )
        if (
            b0.get("status") != "formal_pass"
            or b0.get("path") != "runs/ajae/e72_b0_reference.npz"
            or b0.get("artifact_sha256")
            != "208487d5c91b131856e908988cf6d955305fa09364450d509e32f617295b5863"
            or b0.get("scientific_array_sha256")
            != "49fd285bb7dba95f33a9606309418987e93799470ce96016323cd29b0968c95a"
            or b0.get("development_worlds") != 23
            or b0.get("development_points") != 2_110_885
            or b0.get("pure_normal_points") != 48_828_507
            or b0.get("moving_normal_points") != 13_011
            or b0.get("matched_static_points") != 6_756
            or b0.get("evaluator_errors") != 0
            or b0.get("count_errors") != 0
            or b0.get("independent_read_only_validation") is not True
        ):
            raise ProtocolError("E72 B0 reference identity changed")
        selection = _mapping(development["checkpoint_selection"], "checkpoint selection")
        _exact_keys(
            selection,
            {
                "status", "primary", "tie_tolerance", "first_tie_break",
                "second_tie_break", "third_tie_break", "eligible_world_ids",
                "held_out_input_forbidden",
            },
            "development.checkpoint_selection",
        )
        if (
            selection.get("status") != "frozen_before_training"
            or selection.get("primary")
            != "maximum macro mean of per-world AP over the E63 common-domain eligible in-generator worlds"
            or _number(selection["tie_tolerance"], "checkpoint tie tolerance")
            != 0.001
            or selection.get("first_tie_break")
            != "lower development macro mean FPR95"
            or selection.get("second_tie_break")
            != "lower pure-normal cross-fit FPR"
            or selection.get("third_tie_break")
            != "earlier completed world index"
            or selection.get("held_out_input_forbidden") is not True
        ):
            raise ProtocolError("checkpoint selection is not the frozen E63-v2 rule")
        eligible_world_ids = _int_tuple(
            selection["eligible_world_ids"], "checkpoint eligible world IDs"
        )
        if len(set(eligible_world_ids)) != len(eligible_world_ids) or any(
            world_id >= 24 for world_id in eligible_world_ids
        ):
            raise ProtocolError("checkpoint eligible world IDs must be a unique E57 subset")
        AJAEProtocol._validate_e63(
            _mapping(development["e63_freeze"], "development.e63_freeze")
        )
        if (
            _mapping(development["e63_freeze"], "development.e63_freeze").get(
                "status"
            )
            == "formal_pass"
            and eligible_world_ids
            != tuple(world_id for world_id in range(24) if world_id != 5)
        ):
            raise ProtocolError("checkpoint world IDs differ from the E63 common domain")
        difficulty = _mapping(development["difficulty_statistics"], "difficulty statistics")
        if set(difficulty) != {"Nvis", "O", "d", "V"}:
            raise ProtocolError("development difficulty must define Nvis, O, d, and V")

    @staticmethod
    def _validate_e63(specification: Mapping[str, object]) -> None:
        """Keep the approved E63-v2 preregistration machine-readable."""

        _exact_keys(
            specification,
            {
                "version", "status", "source_worlds", "common_domain",
                "safety_crossfit", "hierarchical_paired_bootstrap",
                "common_domain_paired_bootstrap",
                "shared_training", "sealed_data", "identity_artifact",
            },
            "development.e63_freeze",
        )
        if specification.get("version") != "E63-v2" or specification.get(
            "status"
        ) not in {"frozen_before_identity_generation", "formal_pass"}:
            raise ProtocolError("E63-v2 status is invalid")
        source = _mapping(specification["source_worlds"], "E63 source worlds")
        if (
            source.get("artifact") != "runs/ajae/e57_development_worlds.npz"
            or source.get("artifact_sha256")
            != "b14efc1aad86ac67b5bf7c8631f02b2e68664e071b747b7b210d5f7a30f5d123"
            or source.get("scientific_array_sha256")
            != "590c467da2dec0a161688f2587dc1c37cea2b0f42f326b9918fd6dc9df81f6ec"
            or source.get("worlds") != 24
            or source.get("world_id_array") != "selected_world_id"
            or source.get("world_identity_array") != "selected_candidate_sha256"
            or source.get("center_frame_array") != "selected_center_frame"
        ):
            raise ProtocolError("E63 source-world identity changed")
        domain = _mapping(specification["common_domain"], "E63 common domain")
        if (
            (domain.get("partition"), domain.get("sequence_id")) != ("train", 201)
            or tuple(domain.get("available_frame_range", ())) != (4, 681)
            or _signed_int_tuple(
                domain.get("required_source_offsets"), "E63 source offsets"
            ) != (-4, -3, -2, -1, 0, 1, 2)
            or domain.get("target") != "center q=0"
            or tuple(domain.get("applies_to", ()))
            != tuple(f"B{index}" for index in range(6))
            or domain.get("identity_only_before_training") is not True
            or domain.get("padding_or_zero_fill_forbidden") is not True
        ):
            raise ProtocolError("E63 common comparison domain changed")
        crossfit = _mapping(specification["safety_crossfit"], "E63 safety crossfit")
        if (
            crossfit.get("namespace") != "E63-safety-crossfit-v1"
            or crossfit.get("hash_payload")
            != "UTF-8 namespace, then colon byte, then lowercase hexadecimal world identity"
            or crossfit.get("assignment")
            != "ascending SHA-256 rank; first 12 Fold A and last 12 Fold B"
            or crossfit.get("source_worlds") != 24
            or tuple(crossfit.get("fold_sizes", ())) != (12, 12)
            or crossfit.get("threshold_comparison")
            != "score strictly greater than threshold"
            or crossfit.get("seed_search_forbidden") is not True
        ):
            raise ProtocolError("E63 safety cross-fit identity changed")
        bootstrap = _mapping(
            specification["hierarchical_paired_bootstrap"], "E63 bootstrap"
        )
        if (
            bootstrap.get("namespace")
            != "E63-hierarchical-paired-bootstrap-v1"
            or bootstrap.get("generator") != "NumPy PCG64"
            or bootstrap.get("seed") != 63002026
            or bootstrap.get("replicates") != 5000
            or tuple(bootstrap.get("training_seed_population", ())) != (0, 1, 2)
            or bootstrap.get("training_seed_draws_per_replicate") != 3
            or bootstrap.get("development_world_population") != 24
            or bootstrap.get("development_world_draws_per_replicate") != 24
            or bootstrap.get("replacement") is not True
            or bootstrap.get("paired_models_share_realized_indices") is not True
            or bootstrap.get("gate_lower_bound_percentile") != 2.5
        ):
            raise ProtocolError("E63 hierarchical paired bootstrap changed")
        common_bootstrap = _mapping(
            specification["common_domain_paired_bootstrap"],
            "common-domain paired bootstrap",
        )
        if (
            common_bootstrap.get("status") != "frozen_before_any_gate2_result"
            or common_bootstrap.get("source_artifact_path")
            != "runs/ajae/e75_bootstrap_identity.npz"
            or common_bootstrap.get("source_artifact_sha256")
            != "1bae1dbe4b5ded34cf9cebd818b4877368973114c0e7046840c0ff342fb73b9d"
            or common_bootstrap.get("namespace")
            != "E75-common-domain-bootstrap-correction-v1"
            or common_bootstrap.get("generator") != "NumPy PCG64"
            or common_bootstrap.get("seed") != 63002026
            or common_bootstrap.get("replicates") != 5000
            or tuple(common_bootstrap.get("training_seed_population", ()))
            != (0, 1, 2)
            or common_bootstrap.get("training_seed_draws_per_replicate") != 3
            or tuple(common_bootstrap.get("development_world_ids", ()))
            != tuple(world_id for world_id in range(24) if world_id != 5)
            or common_bootstrap.get("development_world_draws_per_replicate") != 23
            or common_bootstrap.get("replacement") is not True
            or common_bootstrap.get("paired_models_share_realized_indices") is not True
            or tuple(common_bootstrap.get("applies_to", ()))
            != ("E75", "E81", "E82", "E88")
            or common_bootstrap.get("new_random_arrays_for_these_comparisons_forbidden")
            is not True
        ):
            raise ProtocolError("common-domain paired bootstrap changed")
        shared = _mapping(specification["shared_training"], "E63 shared training")
        if (
            tuple(shared.get("conditions", ())) != ("B1", "B2", "B3")
            or tuple(shared.get("seeds", ())) != (0, 1, 2)
            or shared.get("optimizer") != "AdamW"
            or shared.get("learning_rate") != 1.0e-4
            or shared.get("weight_decay") != 1.0e-4
            or shared.get("micro_batch") != 1
            or shared.get("gradient_accumulation") != 8
            or shared.get("maximum_complete_worlds_per_seed") != 25
            or shared.get("evaluate_every_complete_worlds") != 5
            or shared.get("patience_evaluations") != 4
            or dict(_mapping(shared.get("world_type_probabilities"), "E63 world types"))
            != {"pure_normal": 0.2, "control_only": 0.2, "mixed": 0.4, "anomaly_only": 0.2}
            or shared.get("same_budget_and_checkpoint_rule_required") is not True
        ):
            raise ProtocolError("E63 shared training rule changed")
        sealed = _mapping(specification["sealed_data"], "E63 sealed data")
        if (
            sealed.get("held_out_torus_worlds") != 6
            or sealed.get("public_real_ood_sequences") != 19
            or sealed.get("hidden_test_sequences") != 51
            or sealed.get("all_forbidden_for_e63_identity_generation") is not True
        ):
            raise ProtocolError("E63 sealed-data boundary changed")
        artifact = _mapping(specification["identity_artifact"], "E63 identity artifact")
        if artifact.get("path") != "runs/ajae/e63_training_freeze.npz":
            raise ProtocolError("E63 identity-artifact path changed")
        if specification.get("status") == "formal_pass" and (
            artifact.get("artifact_sha256")
            != "5dbf99eaa59a05a83774e42beb6b8d7a95cf9309ebd42ab7870604a20d410dd9"
            or artifact.get("scientific_array_sha256")
            != "e0df86313f27524fba9ed1d2bc563d94def568d36925c184f13e41a72540d207"
            or artifact.get("eligible_worlds") != 23
            or artifact.get("excluded_worlds") != 1
            or artifact.get("fold_a_worlds") != 12
            or artifact.get("fold_b_worlds") != 12
            or tuple(artifact.get("bootstrap_shape", ())) != (5000, 3, 5000, 24)
            or artifact.get("independent_read_only_validation") is not True
        ):
            raise ProtocolError("E63 formal identity artifact is incomplete")

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
        equivalence = _mapping(
            evaluation["evaluator_equivalence"],
            "evaluation.evaluator_equivalence",
        )
        status = equivalence.get("status")
        equivalence_keys = {
            "version", "status", "scientific_role", "official", "fixtures",
            "comparison", "forbidden_data", "pass_definition", "failure_route",
        }
        if status == "formal_pass":
            equivalence_keys.add("result")
        _exact_keys(
            equivalence,
            equivalence_keys,
            "evaluation.evaluator_equivalence",
        )
        official = _mapping(equivalence["official"], "E62 official evaluator")
        fixtures = _mapping(equivalence["fixtures"], "E62 fixtures")
        numerical = _mapping(fixtures["numerical_fixture"], "E62 numerical fixture")
        comparison = _mapping(equivalence["comparison"], "E62 comparison")
        _exact_keys(
            official,
            {"repository", "commit", "source_file", "source_sha256"},
            "E62 official evaluator",
        )
        _exact_keys(
            fixtures,
            {
                "artifact", "artifact_sha256", "scientific_array_sha256",
                "analytic_cases", "declared_range_boundaries_m",
                "range_norm_semantics", "numerical_fixture",
            },
            "E62 fixtures",
        )
        _exact_keys(
            numerical,
            {"kind", "namespace", "pcg64_seed", "frames", "points_per_frame"},
            "E62 numerical fixture",
        )
        _exact_keys(
            comparison,
            {
                "discrete_exact", "metrics", "threshold_if_exposed",
                "maximum_absolute_difference", "fpr95_tpr_rule",
            },
            "E62 comparison",
        )
        if (
            equivalence.get("version") != "E62-v2"
            or status not in {
                "protocol_completed_before_fixture_freeze",
                "fixtures_frozen_before_formal_comparison",
                "formal_pass",
            }
            or (
                official.get("repository"), official.get("commit"),
                official.get("source_file"), official.get("source_sha256"),
            )
            != (
                "/home/jasongao/Study/DynaCAN-deps/stu_dataset",
                "8f0f09c2ca4bf7b665e0ae5919b4092ddae140a2",
                "compute_point_level_ood.py",
                "ed0330f80fbd3cd4cefafed33d6c747c51f2de521ef191e2868eb24f84b9ce61",
            )
            or fixtures.get("artifact") != "runs/ajae/e62_evaluator_fixtures.npz"
            or tuple(fixtures.get("analytic_cases", ()))
            != (
                "range_ignore_and_post_filter_frame_gate",
                "all_scores_tied",
                "strict_tpr_above_0.95",
                "mixed_repeated_scores",
            )
            or tuple(fixtures.get("declared_range_boundaries_m", ()))
            != (2.499999, 2.5, 50.0, 50.000001)
            or fixtures.get("range_norm_semantics")
            != "numpy float32 norm exactly as the official evaluator"
            or (
                numerical.get("kind"), numerical.get("namespace"),
                numerical.get("pcg64_seed"), numerical.get("frames"),
                numerical.get("points_per_frame"),
            )
            != (
                "frozen non-symbolic constructed numerical predictions",
                "E62-numerical-fixture-v1",
                62002026,
                10,
                96,
            )
            or tuple(comparison.get("metrics", ())) != ("AP", "AUROC", "FPR95")
            or comparison.get("threshold_if_exposed") is not True
            or _number(
                comparison.get("maximum_absolute_difference"),
                "E62 metric tolerance",
            )
            != 1.0e-10
            or comparison.get("fpr95_tpr_rule")
            != "first official ROC threshold with TPR strictly greater than 0.95"
            or set(equivalence.get("forbidden_data", ()))
            != {
                "public real-OOD 19 sequences",
                "hidden-test 51 sequences",
                "any real anomaly sequence",
            }
            or equivalence.get("failure_route")
            != "implementation mismatch only; repair the evaluator or harness and rerun the unchanged frozen fixtures"
        ):
            raise ProtocolError("E62 evaluator-equivalence protocol changed")
        expected_discrete = {
            "accepted frame identities", "skipped frame identities",
            "selected point identities", "valid point count",
            "positive point count", "negative point count", "pooled labels",
            "pooled scores",
        }
        if set(comparison.get("discrete_exact", ())) != expected_discrete:
            raise ProtocolError("E62 discrete equivalence checks changed")
        hashes = (
            fixtures.get("artifact_sha256"),
            fixtures.get("scientific_array_sha256"),
        )
        if status == "protocol_completed_before_fixture_freeze":
            if hashes != (None, None):
                raise ProtocolError("unfrozen E62 fixtures cannot declare hashes")
        elif any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ProtocolError("frozen E62 fixture hashes are invalid")
        if status == "formal_pass":
            result = _mapping(equivalence["result"], "E62 result")
            _exact_keys(
                result,
                {
                    "passed", "artifact", "artifact_sha256",
                    "scientific_array_sha256", "protocol_sha256_at_execution",
                    "cases", "accepted_frames", "skipped_frames", "valid_points",
                    "positive_points", "negative_points", "discrete_errors",
                    "maximum_metric_absolute_difference", "metric_tolerance",
                    "independent_read_only_validation",
                },
                "E62 result",
            )
            if (
                result.get("passed") is not True
                or result.get("artifact")
                != "runs/ajae/e62_evaluator_equivalence.npz"
                or result.get("artifact_sha256")
                != "a561c2da0922a99bfe000e29a5f9cfedee432fdf17e3433e2c01d4b56c305226"
                or result.get("scientific_array_sha256")
                != "54d82af072df6fb3adb9d36d77c7dd8d0407b27cd1e53679fa489423d9121101"
                or result.get("protocol_sha256_at_execution")
                != "157b311ffc87ab076e9a4c006b2e6bc8be159feca0937d0bb4115e1e5ea866e7"
                or (
                    result.get("cases"), result.get("accepted_frames"),
                    result.get("skipped_frames"), result.get("valid_points"),
                    result.get("positive_points"), result.get("negative_points"),
                    result.get("discrete_errors"),
                    result.get("maximum_metric_absolute_difference"),
                    result.get("metric_tolerance"),
                    result.get("independent_read_only_validation"),
                )
                != (5, 13, 2, 864, 119, 745, 0, 0.0, 1.0e-10, True)
            ):
                raise ProtocolError("E62 formal result changed")
        frame_domain = _mapping(
            evaluation["comparison_frame_domain"], "comparison frame domain"
        )
        if (
            frame_domain.get("status") != "frozen_before_evaluation"
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
    gate1_policy = (
        criteria_document.get("gate1")
        if criteria_document.get("status") == "frozen_before_training"
        else None
    )
    gate1_criteria = (
        gate1_policy.get("legacy_evidence_compatibility")
        if isinstance(gate1_policy, Mapping)
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
