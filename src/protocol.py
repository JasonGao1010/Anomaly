#!/usr/bin/env python3
"""Load the sole active AJAE schema-31 scientific contract."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "protocol.json"
SCHEMA_VERSION = 31
WINDOW_FRAMES = 5
WINDOW_MEMBER_OFFSETS = (0, 1, 2, 3, 4)
DEVELOPMENT_FORMAT = "ajae-development-window-worlds-v3"

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

_STATE_MACHINE = (
    "R00", "R01", "R02", "R03", "R04", "R05",
    "G2", "G3", "S01", "M01", "V01", "T01",
)
_ROOT_KEYS = {
    "schema_version", "status", "authority", "scientific_contract",
    "claims", "claim_exclusions", "data", "window", "labels", "render",
    "stu", "model", "experiments", "training", "evaluation",
    "state_machine", "decision_gates", "historical_evidence",
}


class ProtocolError(ValueError):
    """Report a protocol or development-data semantic violation."""


class GroupingMode(str, Enum):
    """Whether spatial operations isolate or join scan groups."""

    SINGLE = "single"
    PER_SCAN = "per_scan"
    JOINT = "joint"


class ExperimentCondition(str, Enum):
    """The only registered schema-31 comparison conditions."""

    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"

    @property
    def trainable(self) -> bool:
        return self is not self.B0

    @property
    def grouping_mode(self) -> GroupingMode:
        return {
            self.B0: GroupingMode.SINGLE,
            self.B1: GroupingMode.SINGLE,
            self.B2: GroupingMode.PER_SCAN,
            self.B3: GroupingMode.JOINT,
        }[self]

    @property
    def frame_offsets(self) -> tuple[int, ...]:
        """Physical offsets only; they never enter learned features."""

        return (0,) if self is self.B0 else WINDOW_MEMBER_OFFSETS

    @property
    def input_member_indices(self) -> tuple[int, ...]:
        return self.frame_offsets

    @property
    def output_local_indices(self) -> tuple[int, ...]:
        return self.frame_offsets


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


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be boolean")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ProtocolError(f"{name} keys differ; missing={missing}, extra={extra}")


def _expect(value: object, expected: object, name: str) -> None:
    """Compare JSON values without accepting bool/int aliases."""

    if type(value) is not type(expected) or value != expected:
        raise ProtocolError(f"{name} must equal {expected!r}")


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{name}[{index}]")
        for index, item in enumerate(_list(value, name))
    )


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    return tuple(
        _nonempty_string(item, f"{name}[{index}]")
        for index, item in enumerate(_list(value, name))
    )


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
    """A half-open observed source-frame span."""

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
    """One immutable data role and its currently observed frame span."""

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
        _nonempty_string(self.role, "sequence role")
        _boolean(self.labels_available, "labels_available")
        if tuple(sorted(set(self.excluded_source_frames))) != self.excluded_source_frames:
            raise ProtocolError("excluded source frames must be sorted and unique")
        if self.span is not None and any(
            not self.span.contains(item) for item in self.excluded_source_frames
        ):
            raise ProtocolError("excluded source frame lies outside the observed span")

    @property
    def frames(self) -> int | None:
        return None if self.span is None else len(self.span)

    @property
    def uses_gradients(self) -> bool:
        return self.partition == "train" and self.sequence_id == 206

    @property
    def supports_counterfactuals(self) -> bool:
        return self.partition == "train" and self.sequence_id in {201, 206}

    def with_observed_frame_count(self, frame_count: int) -> SequenceSpec:
        count = _integer(frame_count, "frame_count", minimum=1)
        if self.span is not None:
            if len(self.span) != count:
                raise ProtocolError(
                    f"observed frame count {count} conflicts with frozen span {len(self.span)}"
                )
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
            raise ProtocolError(
                "sequence frame count must be observed before computing legal windows"
            )
        excluded = frozenset(self.excluded_source_frames)
        return tuple(
            start
            for start in range(self.span.start, self.span.stop - WINDOW_FRAMES + 1)
            if all(start + offset not in excluded for offset in WINDOW_MEMBER_OFFSETS)
        )

    def window_frame_ids(self, window_start: int) -> tuple[int, ...]:
        start = _integer(window_start, "window_start")
        if start not in frozenset(self.legal_window_starts()):
            raise ProtocolError(
                f"{self.partition}/{self.sequence_id} window start {start} is not legal"
            )
        return tuple(start + offset for offset in WINDOW_MEMBER_OFFSETS)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """The frozen point-evaluation domain."""

    minimum_range_m: float
    maximum_range_m: float
    minimum_anomaly_points: int

    def range_mask(self, ranges: object) -> object:
        import numpy as np

        values = np.asarray(ranges, dtype=np.float32)
        return (values >= np.float32(self.minimum_range_m)) & (
            values <= np.float32(self.maximum_range_m)
        )


@dataclass(frozen=True, slots=True)
class DevelopmentWorld:
    """One five-scan window belonging to a shared synthetic clip world."""

    world_identity: str
    seed: int
    window_start: int
    frame_ids: tuple[int, ...]
    world: Mapping[str, object]
    descriptors: Mapping[str, object]
    mechanism: str


@dataclass(frozen=True, slots=True)
class DevelopmentWorlds:
    format: str
    protocol_schema: int
    sequence_id: int
    status: str
    validation: Mapping[str, bool]
    in_generator: tuple[DevelopmentWorld, ...]
    generator_held_out: tuple[DevelopmentWorld, ...]

    @property
    def validated(self) -> bool:
        return (
            self.status == "validated_frozen"
            and bool(self.validation)
            and all(self.validation.values())
        )


class AJAEProtocol:
    """Validated immutable view of the schema-31 route."""

    def __init__(self, document: Mapping[str, object], *, path: Path) -> None:
        self._validate(document)
        self.path = path.expanduser().resolve(strict=True)
        self.schema_version = SCHEMA_VERSION
        self._document = _freeze(document)
        for name in (
            "status", "authority", "scientific_contract", "claims",
            "claim_exclusions", "data", "window", "labels", "render", "stu",
            "model", "experiments", "training", "state_machine",
            "decision_gates", "historical_evidence",
        ):
            setattr(self, name, self._document[name])
        self.evaluation_document = self._document["evaluation"]

        raw_data = _mapping(document["data"], "data")
        self.normal_training = self._sequence_from_record(
            _mapping(raw_data["normal_training"], "data.normal_training")
        )
        self.development_sequence = self._sequence_from_record(
            _mapping(raw_data["development"], "data.development")
        )
        public = _mapping(raw_data["public_anomaly_validation"], "public data")
        hidden = _mapping(raw_data["hidden_test"], "hidden data")
        self.public_validation = tuple(
            SequenceSpec("val", item, str(public["role"]), True, None)
            for item in PUBLIC_ANOMALY_IDS
        )
        self.hidden_test = tuple(
            SequenceSpec("test", item, str(hidden["role"]), False, None)
            for item in HIDDEN_TEST_IDS
        )
        all_sequences = (
            self.normal_training, self.development_sequence,
            *self.public_validation, *self.hidden_test,
        )
        self._sequences = {
            (item.partition, item.sequence_id): item for item in all_sequences
        }

        class_map = _mapping(
            _mapping(document["labels"], "labels")["normal_semantic_class_map"],
            "labels.normal_semantic_class_map",
        )
        self.normal_training_class_map = MappingProxyType(
            {
                int(raw): _integer(target, f"normal semantic class {raw}")
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
    def _sequence_from_record(record: Mapping[str, object]) -> SequenceSpec:
        bounds = _int_tuple(record["frame_range_inclusive"], "frame range")
        if len(bounds) != 2 or bounds[1] < bounds[0]:
            raise ProtocolError("frame_range_inclusive must be [first, last]")
        return SequenceSpec(
            str(record["partition"]),
            _integer(record["sequence_id"], "sequence_id"),
            str(record["role"]),
            _boolean(record["labels_available"], "labels_available"),
            FrameSpan(bounds[0], bounds[1] + 1),
            _int_tuple(record["excluded_source_frames"], "excluded source frames"),
        )

    @property
    def document(self) -> Mapping[str, object]:
        return self._document  # type: ignore[return-value]

    def plain_document(self) -> dict[str, object]:
        value = _plain(self._document)
        if not isinstance(value, dict):
            raise AssertionError("protocol root is not a dictionary")
        return value

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
                f"sequence {partition}/{sequence_id} is outside schema 31"
            ) from error

    def legal_window_starts(self, partition: str, sequence_id: int) -> tuple[int, ...]:
        return self.sequence(partition, sequence_id).legal_window_starts()

    def window_frame_ids(
        self, partition: str, sequence_id: int, window_start: int
    ) -> tuple[int, ...]:
        return self.sequence(partition, sequence_id).window_frame_ids(window_start)

    def checkpoint_path(self, project_root: Path | str | None = None) -> Path:
        root = self.path.parent if project_root is None else Path(project_root).expanduser().resolve()
        return (root / str(self.stu["checkpoint"])).resolve()

    def stu_repository_path(self, project_root: Path | str | None = None) -> Path:
        root = self.path.parent if project_root is None else Path(project_root).expanduser().resolve()
        return (root / str(self.stu["repository"])).resolve()

    def sensor_calibration_path(self, project_root: Path | str | None = None) -> Path:
        root = self.path.parent if project_root is None else Path(project_root).expanduser().resolve()
        return (root / str(self.render["calibration_file"])).resolve()

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "current_node": self.status["current_node"],
            "scientific_document": self.authority["scientific_document"],
            "conditions": [item.value for item in ExperimentCondition],
            "train_windows": len(self.normal_training.legal_window_starts()),
            "development_windows": len(self.development_sequence.legal_window_starts()),
            "public_sequences": len(self.public_validation),
            "hidden_sequences": len(self.hidden_test),
            "formal_training_allowed": self.status["formal_training_allowed"],
        }

    @classmethod
    def _validate(cls, source: Mapping[str, object]) -> None:
        # Reject old science before interpreting any legacy-shaped subtree.
        if source.get("schema_version") != SCHEMA_VERSION:
            raise ProtocolError(
                f"schema_version must be {SCHEMA_VERSION}; schema 30 is retired"
            )
        _exact_keys(source, _ROOT_KEYS, "protocol")
        cls._validate_authority(source)
        cls._validate_data(_mapping(source["data"], "data"))
        cls._validate_window(_mapping(source["window"], "window"))
        cls._validate_labels(_mapping(source["labels"], "labels"))
        cls._validate_render(_mapping(source["render"], "render"))
        cls._validate_stu(_mapping(source["stu"], "stu"))
        cls._validate_model(_mapping(source["model"], "model"))
        cls._validate_experiments(_mapping(source["experiments"], "experiments"))
        cls._validate_training(_mapping(source["training"], "training"))
        cls._validate_evaluation(_mapping(source["evaluation"], "evaluation"))
        cls._validate_status(source)

    @staticmethod
    def _validate_authority(source: Mapping[str, object]) -> None:
        authority = _mapping(source["authority"], "authority")
        _exact_keys(
            authority,
            {"scientific_document", "state_machine", "history_baseline_commit", "history_tag", "supersedes"},
            "authority",
        )
        _expect(authority["scientific_document"], "AJAE新主线方案.md", "authority.scientific_document")
        _expect(authority["state_machine"], "AJAE实验执行状态机.md", "authority.state_machine")
        digest = _nonempty_string(authority["history_baseline_commit"], "history baseline")
        if not re.fullmatch(r"[0-9a-f]{40}", digest):
            raise ProtocolError("history_baseline_commit must be a lowercase SHA-1")
        _expect(authority["history_tag"], "schema30-history-baseline", "authority.history_tag")
        _expect(authority["supersedes"], "schema30_center_target_temporal_message_passing_route", "authority.supersedes")

        contract = _mapping(source["scientific_contract"], "scientific contract")
        expected = {
            "observation_unit": "one_complete_five_scan_window",
            "privileged_frame": None,
            "all_window_members_equally_supervised": True,
            "all_visible_returns_receive_logits": True,
            "learned_time_or_position_input": False,
            "full_model_spatial_operations": [
                "joint_voxelization", "joint_radius_neighborhood", "joint_knn_decoding"
            ],
            "sequence_score": "equal_mean_of_window_probabilities_by_original_frame_ray_identity",
        }
        _exact_keys(contract, set(expected), "scientific_contract")
        for name, value in expected.items():
            _expect(contract[name], value, f"scientific_contract.{name}")

        claims = _mapping(source["claims"], "claims")
        _exact_keys(claims, {"proxy_supervision", "joint_densification", "real_ood_transfer"}, "claims")
        for name, value in claims.items():
            _nonempty_string(value, f"claims.{name}")
        _expect(
            source["claim_exclusions"],
            ["motion_unknown_detection", "explicit_motion_understanding", "object_tracking", "future_frame_assistance", "privileged_frame_completion"],
            "claim_exclusions",
        )
        _expect(source["state_machine"], list(_STATE_MACHINE), "state_machine")

        history = _mapping(source["historical_evidence"], "historical evidence")
        history_keys = {
            "evidence_source_schema", "inheritance_rule", "continues_with_original_scope",
            "old_distribution_only", "reusable_only_after_schema31_requalification",
            "excluded_from_schema31_statistics",
        }
        _exact_keys(history, history_keys, "historical_evidence")
        _expect(history["evidence_source_schema"], 30, "historical evidence schema")
        _nonempty_string(history["inheritance_rule"], "historical inheritance rule")
        for name in history_keys - {"evidence_source_schema", "inheritance_rule"}:
            if not _string_tuple(history[name], f"historical_evidence.{name}"):
                raise ProtocolError(f"historical_evidence.{name} cannot be empty")

    @staticmethod
    def _validate_data(data: Mapping[str, object]) -> None:
        _exact_keys(data, {"normal_training", "development", "public_anomaly_validation", "hidden_test"}, "data")
        normal = _mapping(data["normal_training"], "normal training")
        normal_expected = {
            "partition": "train", "sequence_id": 206,
            "role": "renderer_calibration_window_bank_and_all_parameter_updates",
            "labels_available": True, "frame_range_inclusive": [0, 448],
            "excluded_source_frames": [],
        }
        _exact_keys(normal, set(normal_expected), "data.normal_training")
        for name, value in normal_expected.items():
            _expect(normal[name], value, f"data.normal_training.{name}")

        development = _mapping(data["development"], "development sequence")
        development_expected = {
            "partition": "train", "sequence_id": 201,
            "role": "development_only_no_formal_gradients", "labels_available": True,
            "frame_range_inclusive": [0, 681], "excluded_source_frames": [0, 1, 2, 3],
            "exclusion_reason": "verified exact internal duplication in scans and labels",
        }
        _exact_keys(development, set(development_expected), "data.development")
        for name, value in development_expected.items():
            _expect(development[name], value, f"data.development.{name}")

        common_keys = {
            "partition", "role", "labels_available", "method_freeze_required",
            "sequence_ids",
        }
        public = _mapping(data["public_anomaly_validation"], "public validation")
        hidden = _mapping(data["hidden_test"], "hidden test")
        for record, name, partition, role, labels, ids in (
            (public, "public_anomaly_validation", "val", "one_time_real_anomaly_confirmation_after_method_freeze", True, PUBLIC_ANOMALY_IDS),
            (hidden, "hidden_test", "test", "final_hidden_test_after_supported_public_confirmation", False, HIDDEN_TEST_IDS),
        ):
            _exact_keys(record, common_keys, f"data.{name}")
            _expect(record["partition"], partition, f"data.{name}.partition")
            _expect(record["role"], role, f"data.{name}.role")
            _expect(record["labels_available"], labels, f"data.{name}.labels_available")
            _expect(record["method_freeze_required"], True, f"data.{name}.method_freeze_required")
            if _int_tuple(record["sequence_ids"], f"data.{name}.sequence_ids") != ids:
                raise ProtocolError(f"data.{name}.sequence_ids changed")

    @staticmethod
    def _validate_window(window: Mapping[str, object]) -> None:
        expected = {
            "frames": WINDOW_FRAMES,
            "member_offsets_from_start": list(WINDOW_MEMBER_OFFSETS),
            "identity": ["partition", "sequence_id", "window_start"],
            "frame_ids": "window_start_plus_member_offsets",
            "padding": False,
            "frame_repetition": False,
            "member_order_is_model_input": False,
            "point_identity": ["source_frame", "source_ray"],
            "restoration_fields": ["source_frame", "source_slot", "source_ray"],
            "scan_group_use": "B2_grouped_operations_only_never_a_learned_feature",
        }
        _exact_keys(window, {*expected, "reference_pose"}, "window")
        for name, value in expected.items():
            _expect(window[name], value, f"window.{name}")
        pose = _mapping(window["reference_pose"], "window.reference_pose")
        expected_pose = {
            "translation": "arithmetic_mean_of_five_sensor_translations",
            "rotation": "SO3_chordal_mean_of_five_sensor_rotations",
            "proper_rotation_projection": "U_diag_1_1_det_UVt_Vt",
            "permutation_invariant": True,
            "global_rigid_transform_invariant_coordinates": True,
        }
        _exact_keys(pose, set(expected_pose), "window.reference_pose")
        for name, value in expected_pose.items():
            _expect(pose[name], value, f"window.reference_pose.{name}")

    @staticmethod
    def _validate_labels(labels: Mapping[str, object]) -> None:
        _exact_keys(labels, {"packed_label", "binary_anomaly", "normal_control_semantic_ids", "moving_normal_semantic_ids", "normal_semantic_class_map"}, "labels")
        _expect(labels["packed_label"], {"semantic_bits_inclusive": [0, 15], "instance_bits_inclusive": [16, 31]}, "labels.packed_label")
        _expect(labels["binary_anomaly"], {
            "raw_semantic_0": "ignore_unless_replaced_by_a_valid_inserted_return",
            "raw_semantic_2": "anomaly", "other_nonzero_semantics": "normal",
            "normal_control_return": "normal", "anomaly_proxy_return": "anomaly",
        }, "labels.binary_anomaly")
        if _int_tuple(labels["normal_control_semantic_ids"], "normal controls") != NORMAL_CONTROL_SEMANTICS:
            raise ProtocolError("normal-control semantic IDs changed")
        if _int_tuple(labels["moving_normal_semantic_ids"], "moving normals") != MOVING_NORMAL_SEMANTICS:
            raise ProtocolError("moving-normal semantic IDs changed")
        class_map = _mapping(labels["normal_semantic_class_map"], "normal class map")
        if not class_map or any(not raw.isdigit() for raw in class_map):
            raise ProtocolError("normal semantic class map keys must be decimal IDs")
        for raw, target in class_map.items():
            _integer(target, f"normal semantic class {raw}")
        if any(str(item) not in class_map for item in (*NORMAL_CONTROL_SEMANTICS, *MOVING_NORMAL_SEMANTICS)):
            raise ProtocolError("normal class map omits a renderer/STU semantic")

    @staticmethod
    def _validate_render(render: Mapping[str, object]) -> None:
        keys = {
            "geometry_schema", "source_sequence_id", "sensor", "calibration_file",
            "ray_grid", "normal_controls", "anomaly_proxies",
            "world_type_probabilities", "common_entity_rules", "sensor_model",
            "physical_scope_excludes", "world_unit", "freeze_before_render",
            "shared_renderer_for_normal_control_and_proxy", "window_descriptors",
            "proxy_control_matching", "forbidden_densification",
        }
        _exact_keys(render, keys, "render")
        _expect(render["geometry_schema"], 7, "render.geometry_schema")
        _expect(render["source_sequence_id"], 206, "render.source_sequence_id")
        _expect(render["sensor"], "OS1-128_canonical_ray_first_return_approximation", "render.sensor")
        _nonempty_string(render["calibration_file"], "render.calibration_file")
        _expect(render["world_unit"], "WindowWorld", "render.world_unit")
        _expect(render["shared_renderer_for_normal_control_and_proxy"], True, "render.shared_renderer")
        _expect(render["freeze_before_render"], ["five_source_frames", "normal_controls", "anomaly_proxies", "world_positions", "orientations", "scales", "materials", "all_random_seeds"], "render.freeze_before_render")
        _expect(render["forbidden_densification"], ["synthetic_point_completion", "bottom_return_insertion", "scan_duplication", "single_scan_copying"], "render.forbidden_densification")

        ray = _mapping(render["ray_grid"], "render.ray_grid")
        _expect(ray, {"beam_count": 128, "column_count": 1024, "canonical_identity": ["beam_id", "azimuth_column"], "file_slot_role": "input_output_mapping_only"}, "render.ray_grid")
        controls = _mapping(render["normal_controls"], "render.normal_controls")
        if _int_tuple(controls.get("semantic_ids"), "normal-control semantics") != NORMAL_CONTROL_SEMANTICS:
            raise ProtocolError("renderer normal-control semantics differ from labels")
        _expect(controls.get("source_sequence_id"), 206, "normal-control source")
        proxies = _mapping(render["anomaly_proxies"], "render.anomaly_proxies")
        _expect(proxies.get("training_mechanisms"), ["superquadric", "constructive_overlap_union", "bend", "twist", "taper", "low_frequency_surface"], "proxy mechanisms")
        _expect(proxies.get("held_out_mechanism"), "torus_SDF", "held-out mechanism")

        probabilities = _mapping(render["world_type_probabilities"], "world probabilities")
        _exact_keys(probabilities, {"pure_normal", "control_only", "mixed", "anomaly_only"}, "world probabilities")
        total = sum(_number(item, "world probability") for item in probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ProtocolError("world type probabilities must sum to one")
        common = _mapping(render["common_entity_rules"], "common entity rules")
        for name in ("static_world_pose", "support_plane_required", "observed_surface_collision_rejection", "inserted_entity_collision_rejection"):
            _expect(common.get(name), True, f"common_entity_rules.{name}")
        sensor = _mapping(render["sensor_model"], "sensor model")
        for name in ("return_probability", "intensity_quantiles", "nearest_accepted_return", "allow_new_return_on_empty_ray", "bidirectional_occlusion"):
            _expect(sensor.get(name), True, f"sensor_model.{name}")

        descriptors = _mapping(render["window_descriptors"], "window descriptors")
        _exact_keys(descriptors, {"density_voxel_size_m", "density_coordinate_system", "density_voxel_quantization", "definitions", "required"}, "window descriptors")
        if _number(descriptors["density_voxel_size_m"], "density voxel size") <= 0:
            raise ProtocolError("density voxel size must be positive")
        _expect(descriptors["density_coordinate_system"], "symmetric_window_coordinates", "density coordinate system")
        required = _string_tuple(descriptors["required"], "window descriptor names")
        expected_required = (
            "joint_visible_return_count", "joint_spatial_voxel_count",
            "maximum_single_scan_spatial_voxel_count", "densification_gain",
            "duplicate_fraction", "distance", "occlusion", "support_surface",
            "visible_scan_count", "minimum_visible_return_height_above_support",
            "intensity_distribution", "beam_distribution",
        )
        if required != expected_required:
            raise ProtocolError("window descriptor identities changed")
        definitions = _mapping(descriptors["definitions"], "descriptor definitions")
        if set(definitions) != {
            *expected_required[:5],
            "empty_entity_rule",
        }:
            raise ProtocolError("density descriptor definitions are incomplete")
        matching = _mapping(render["proxy_control_matching"], "proxy/control matching")
        _expect(matching.get("unit"), "complete_five_scan_window", "matching unit")
        _expect(matching.get("thresholds"), "freeze_result_blind_in_R02", "matching status")
        covariates = set(_string_tuple(matching.get("required_covariates"), "matching covariates"))
        if len(covariates) != 7 or not covariates.issubset(set(required)):
            raise ProtocolError("proxy/control matching covariates are invalid")

    @staticmethod
    def _validate_stu(stu: Mapping[str, object]) -> None:
        expected = {
            "source": "STU_official_Mask4Former3D", "checkpoint_bytes": 476261075,
            "voxel_size_m": 0.05, "point_feature_dim": 128,
            "normal_evidence_dim": 19, "assignment_reliability_dim": 1,
            "no_object_reliability_dim": 1, "b0_score": "official_STU_MaxLogit",
            "frozen": True,
        }
        _exact_keys(stu, {*expected, "checkpoint", "checkpoint_sha256", "repository"}, "stu")
        for name, value in expected.items():
            _expect(stu[name], value, f"stu.{name}")
        _nonempty_string(stu["checkpoint"], "stu.checkpoint")
        _nonempty_string(stu["repository"], "stu.repository")
        digest = _nonempty_string(stu["checkpoint_sha256"], "stu.checkpoint_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProtocolError("stu.checkpoint_sha256 must be lowercase SHA-256")

    @staticmethod
    def _validate_model(model: Mapping[str, object]) -> None:
        keys = {
            "name", "input_features", "spatial_coordinates", "bookkeeping_inputs",
            "input_dim", "forbidden_features", "hidden_dim", "heads", "levels",
            "voxel_sizes_m", "radius_neighbors", "voxel_feature",
            "neighborhood_feature", "upsample_neighbors", "grouping_modes",
            "B2_B3_shared_class_and_parameterization", "output",
            "scan_permutation_equivariant",
        }
        _exact_keys(model, keys, "model")
        expected = {
            "name": "JointWindowPointTransformer",
            "input_features": ["stu_point_feature_128d", "normal_evidence_19d", "assignment_reliability", "no_object_reliability", "intensity"],
            "spatial_coordinates": "symmetric_window_coordinates_may_enter_order_invariant_position_encoding_relative_displacement_voxel_neighborhood_and_decode_geometry",
            "bookkeeping_inputs": "point_identity_for_restoration_and_scan_group_for_B2_isolation_only",
            "input_dim": 150,
            "forbidden_features": ["source_frame", "window_member_index", "relative_time", "absolute_time", "time_embedding", "reversible_time_encoding"],
            "hidden_dim": 128, "heads": 4, "levels": 4,
            "voxel_sizes_m": [0.1, 0.2, 0.4],
            "radius_neighbors": {"radii_m": [0.25, 0.5, 1.0, 2.0], "maximum_neighbors": [12, 16, 24, 32]},
            "voxel_feature": ["mean", "max", "log1p_population"],
            "neighborhood_feature": "log1p_uncapped_count_of_all_points_strictly_inside_radius_before_top_K_selection",
            "upsample_neighbors": 3,
            "grouping_modes": {"B1": "single", "B2": "per_scan", "B3": "joint"},
            "B2_B3_shared_class_and_parameterization": True,
            "output": "one_anomaly_logit_for_every_visible_input_return",
            "scan_permutation_equivariant": True,
        }
        for name, value in expected.items():
            _expect(model[name], value, f"model.{name}")

    @staticmethod
    def _validate_experiments(experiments: Mapping[str, object]) -> None:
        _exact_keys(experiments, {item.value for item in ExperimentCondition}, "experiments")
        definitions = {
            "B0": "frozen_STU_official_single_scan_MaxLogit",
            "B1": "five_independent_single_scan_forwards_per_WindowWorld_with_one_all_point_loss",
            "B2": "all_five_scans_input_output_and_supervised_with_voxel_neighborhood_and_decode_isolated_by_scan_group",
            "B3": "all_five_scans_jointly_voxelized_neighbored_and_decoded_in_symmetric_window_coordinates",
        }
        for condition in ExperimentCondition:
            record = _mapping(experiments[condition.value], f"experiments.{condition.value}")
            keys = {"trainable", "definition"}
            if condition.trainable:
                keys.add("grouping_mode")
            _exact_keys(record, keys, f"experiments.{condition.value}")
            _expect(record["trainable"], condition.trainable, f"{condition.value}.trainable")
            _expect(record["definition"], definitions[condition.value], f"{condition.value}.definition")
            if condition.trainable:
                _expect(record["grouping_mode"], condition.grouping_mode.value, f"{condition.value}.grouping_mode")
                if condition.output_local_indices != WINDOW_MEMBER_OFFSETS:
                    raise ProtocolError(f"{condition.value} must output all five members")

    @staticmethod
    def _validate_training(training: Mapping[str, object]) -> None:
        keys = {
            "source_partition", "source_sequence_id", "bank", "micro_batch",
            "effective_batch", "epoch", "loss",
            "forced_partial_step_at_world_boundary", "modes", "tiny_overfit",
            "pilot", "formal", "checkpoint_selection", "deterministic_algorithms",
        }
        _exact_keys(training, keys, "training")
        expected = {
            "source_partition": "train", "source_sequence_id": 206,
            "micro_batch": "one_complete_WindowWorld",
            "effective_batch": "all_complete_WindowWorld_micro_batches_accumulated_into_one_optimizer_step",
            "epoch": "one_complete_pass_over_the_frozen_window_train_bank",
            "forced_partial_step_at_world_boundary": False,
            "modes": ["tiny_overfit", "pilot", "formal"],
            "deterministic_algorithms": True,
        }
        for name, value in expected.items():
            _expect(training[name], value, f"training.{name}")
        bank = _mapping(training["bank"], "training.bank")
        _expect(bank, {
            "name": "window_train_bank", "unit": "WindowWorld",
            "shared_by": ["B1", "B2", "B3"],
            "required_identity": ["five_source_frames", "WorldSpec", "renderer_identity", "STU_identity", "point_identity", "labels"],
            "condition_invariants": ["same_five_aligned_points", "same_point_labels", "same_entry_order", "same_model_parameter_shapes_and_initialization", "same_loss", "same_optimizer"],
        }, "training.bank")
        _expect(training["loss"], {
            "name": "effective_batch_empty_class_safe_balanced_binary_cross_entropy",
            "aggregation": "first_sum_positive_and_negative_unreduced_losses_and_counts_over_the_entire_effective_batch_then_average_each_present_class",
            "two_class_formula": "0.5*positive_loss_sum/positive_count+0.5*negative_loss_sum/negative_count",
            "one_class_rule": "mean_loss_of_the_single_present_class",
        }, "training.loss")
        _expect(training["tiny_overfit"], {
            "windows": 8, "maximum_updates": 500,
            "pass_any": {"training_AP_minimum_percent": 99.0, "loss_strictly_below": 0.02},
        }, "training.tiny_overfit")
        _expect(training["pilot"], {
            "windows": 128, "seeds": [1001, 1002],
            "learning_rates": [0.00003, 0.0001, 0.0003],
            "gradient_accumulation": [1, 2], "screen_updates": [50, 200, 600],
            "schedulers_after_learning_rate_selection": ["constant", "five_percent_warmup_cosine"],
            "weight_decay_after_scheduler_selection": [0.0, 0.00001, 0.0001],
        }, "training.pilot")
        formal = _mapping(training["formal"], "training.formal")
        _exact_keys(formal, {"seeds", "recipe_status", "allowed_only_after"}, "training.formal")
        _expect(formal["seeds"], [0, 1, 2], "training.formal.seeds")
        if formal["recipe_status"] not in {
            "pending_result_blind_R05_freeze", "frozen_result_blind_in_R05"
        }:
            raise ProtocolError("training.formal.recipe_status is not recognized")
        _expect(formal["allowed_only_after"], "R05", "training.formal.allowed_only_after")
        selection = _mapping(training["checkpoint_selection"], "checkpoint selection")
        _exact_keys(selection, {"metric", "fusion_scope", "status"}, "checkpoint selection")
        _expect(selection["metric"], "macro_mean_of_per_DevelopmentClipWorld_all_occurrence_fused_point_AP", "checkpoint metric")
        _expect(selection["fusion_scope"], "within_one_frozen_world_identity_only", "checkpoint fusion scope")
        _nonempty_string(selection["status"], "checkpoint selection status")

    @staticmethod
    def _validate_evaluation(evaluation: Mapping[str, object]) -> None:
        expected = {
            "legal_window_starts": "every_start_with_five_real_consecutive_nonexcluded_frames",
            "model_output_members": list(WINDOW_MEMBER_OFFSETS),
            "real_sequence_fusion_key": ["partition", "sequence_id", "source_frame", "source_ray"],
            "synthetic_fusion_key": ["world_identity", "source_frame", "source_ray"],
            "fusion_value": "sigmoid_probability",
            "fusion_reduction": "equal_arithmetic_mean_over_every_legal_window_occurrence",
            "domain": "every_visible_return_covered_by_at_least_one_legal_five_scan_window",
            "occurrence_count_strata": [1, 2, 3, 4, 5],
            "minimum_range_m_inclusive": 2.5,
            "maximum_range_m_inclusive": 50.0,
            "minimum_anomaly_points_per_evaluated_frame": 5,
            "point_metrics": ["AP", "AUROC", "FPR95"],
            "official_evaluator_equivalence_required": True,
            "sealed_validation_and_test_access": True,
        }
        _exact_keys(evaluation, {*expected, "synthetic_development"}, "evaluation")
        for name, value in expected.items():
            _expect(evaluation[name], value, f"evaluation.{name}")
        _expect(evaluation["synthetic_development"], {
            "unit": "DevelopmentClipWorld",
            "world_rule": "one_WorldSpec_and_random_identity_are_frozen_before_rendering_every_scan_used_by_all_overlapping_windows_in_the_clip",
            "minimum_frames_to_expose_all_occurrence_strata": 9,
            "exact_clip_length_and_count": "freeze_result_blind_in_R02_before_generation",
            "cross_world_fusion_forbidden": True,
        }, "evaluation.synthetic_development")

    @staticmethod
    def _validate_status(source: Mapping[str, object]) -> None:
        status = _mapping(source["status"], "status")
        _exact_keys(status, {"implementation_target", "current_node", "scientific_route", "formal_training_allowed", "performance_claims_available"}, "status")
        _expect(status["implementation_target"], "schema31", "status.implementation_target")
        _expect(status["scientific_route"], "window_proxy_and_joint_five_scan_dense_point_cloud", "status.scientific_route")
        node = _nonempty_string(status["current_node"], "status.current_node")
        if node not in _STATE_MACHINE:
            raise ProtocolError("status.current_node is outside the schema-31 state machine")
        formal_allowed = _boolean(status["formal_training_allowed"], "formal training status")
        claims_available = _boolean(status["performance_claims_available"], "performance claim status")
        formal = _mapping(_mapping(source["training"], "training")["formal"], "training.formal")
        frozen_recipe = formal["recipe_status"] == "frozen_result_blind_in_R05"
        if formal_allowed and (
            not frozen_recipe
            or _STATE_MACHINE.index(node) <= _STATE_MACHINE.index("R05")
        ):
            raise ProtocolError("formal training cannot be enabled before frozen R05")
        if claims_available and _STATE_MACHINE.index(node) < _STATE_MACHINE.index("V01"):
            raise ProtocolError("performance claims cannot exist before V01")

        gates = _mapping(source["decision_gates"], "decision gates")
        _exact_keys(gates, {"G2", "G3", "V01"}, "decision_gates")
        for name, comparisons in (
            ("G2", {"B1_vs_B0"}),
            ("G3", {"B3_vs_B1", "B3_vs_B2"}),
        ):
            gate = _mapping(gates[name], f"decision_gates.{name}")
            _exact_keys(gate, {"comparisons", "formal_seeds", "criteria_status"}, f"decision_gates.{name}")
            if set(_string_tuple(gate["comparisons"], f"{name}.comparisons")) != comparisons:
                raise ProtocolError(f"decision_gates.{name}.comparisons changed")
            _expect(gate["formal_seeds"], [0, 1, 2], f"{name}.formal_seeds")
            _nonempty_string(gate["criteria_status"], f"{name}.criteria_status")
        v01 = _mapping(gates["V01"], "decision_gates.V01")
        _exact_keys(v01, {"public_sequences", "one_time_only", "criteria_status"}, "decision_gates.V01")
        _expect(v01["public_sequences"], len(PUBLIC_ANOMALY_IDS), "V01.public_sequences")
        _expect(v01["one_time_only"], True, "V01.one_time_only")
        _nonempty_string(v01["criteria_status"], "V01.criteria_status")


def _validate_development_descriptors(
    descriptors: Mapping[str, object], required: tuple[str, ...], name: str
) -> None:
    _exact_keys(descriptors, set(required), name)
    returns = _integer(descriptors["joint_visible_return_count"], f"{name}.joint_visible_return_count", minimum=1)
    joint = _integer(descriptors["joint_spatial_voxel_count"], f"{name}.joint_spatial_voxel_count", minimum=1)
    single = _integer(descriptors["maximum_single_scan_spatial_voxel_count"], f"{name}.maximum_single_scan_spatial_voxel_count", minimum=1)
    if not single <= joint <= returns:
        raise ProtocolError(f"{name} must satisfy single voxels <= joint voxels <= returns")
    gain = _number(descriptors["densification_gain"], f"{name}.densification_gain")
    duplicate = _number(descriptors["duplicate_fraction"], f"{name}.duplicate_fraction")
    if not math.isclose(gain, joint / single, rel_tol=1e-9, abs_tol=1e-12):
        raise ProtocolError(f"{name}.densification_gain disagrees with counts")
    if not math.isclose(duplicate, 1.0 - joint / returns, rel_tol=1e-9, abs_tol=1e-12):
        raise ProtocolError(f"{name}.duplicate_fraction disagrees with counts")
    if _number(descriptors["distance"], f"{name}.distance") < 0:
        raise ProtocolError(f"{name}.distance cannot be negative")
    occlusion = _number(descriptors["occlusion"], f"{name}.occlusion")
    if not 0.0 <= occlusion <= 1.0:
        raise ProtocolError(f"{name}.occlusion must lie in [0,1]")
    visible = _integer(descriptors["visible_scan_count"], f"{name}.visible_scan_count", minimum=1)
    if visible > WINDOW_FRAMES:
        raise ProtocolError(f"{name}.visible_scan_count cannot exceed five")
    _number(
        descriptors["minimum_visible_return_height_above_support"],
        f"{name}.minimum_visible_return_height_above_support",
    )


def load_development_worlds(
    path: Path | str, *, protocol: AJAEProtocol
) -> DevelopmentWorlds:
    """Load schema-31 windows grouped by shared clip-world identity."""

    if protocol.schema_version != SCHEMA_VERSION:
        raise ProtocolError("development data require an active schema-31 protocol")
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        source = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read development worlds: {resolved}") from error
    root = _mapping(source, "development data")
    if root.get("format") == "ajae-development-worlds-v2" or root.get("protocol_schema") == 30:
        raise ProtocolError(
            "schema-30 centered dev.json is retired; regenerate schema-31 window/clip data"
        )
    _exact_keys(root, {"format", "protocol_schema", "sequence_id", "status", "validation", "in_generator", "generator_held_out"}, "development data")
    _expect(root["format"], DEVELOPMENT_FORMAT, "development format")
    _expect(root["protocol_schema"], SCHEMA_VERSION, "development protocol_schema")
    _expect(root["sequence_id"], 201, "development sequence_id")
    status = _nonempty_string(root["status"], "development status")
    validation_source = _mapping(root["validation"], "development.validation")
    if not validation_source:
        raise ProtocolError("development.validation cannot be empty")
    validation = {
        name: _boolean(value, f"development.validation.{name}")
        for name, value in validation_source.items()
    }
    required = _string_tuple(
        protocol.render["window_descriptors"]["required"],
        "render.window_descriptors.required",
    )

    def parse_group(value: object, name: str, held_out: bool) -> tuple[DevelopmentWorld, ...]:
        parsed: list[DevelopmentWorld] = []
        clip_identity: dict[str, tuple[int, str, str]] = {}
        starts_by_clip: dict[str, list[int]] = {}
        for index, raw in enumerate(_list(value, name)):
            record_name = f"{name}[{index}]"
            item = _mapping(raw, record_name)
            _exact_keys(item, {"world_identity", "seed", "window_start", "frame_ids", "world", "descriptors", "mechanism"}, record_name)
            identity = _nonempty_string(item["world_identity"], f"{record_name}.world_identity")
            seed = _integer(item["seed"], f"{record_name}.seed")
            start = _integer(item["window_start"], f"{record_name}.window_start")
            frame_ids = _int_tuple(item["frame_ids"], f"{record_name}.frame_ids")
            if frame_ids != tuple(start + offset for offset in WINDOW_MEMBER_OFFSETS):
                raise ProtocolError(f"{record_name}.frame_ids must be five consecutive members")
            if start not in frozenset(protocol.development_sequence.legal_window_starts()):
                raise ProtocolError(f"{record_name}.window_start is illegal for train/201")
            world = _mapping(item["world"], f"{record_name}.world")
            if world.get("source_sequence_id") not in {None, 201}:
                raise ProtocolError(f"{record_name}.world must belong to train/201")
            descriptors = _mapping(item["descriptors"], f"{record_name}.descriptors")
            _validate_development_descriptors(descriptors, required, f"{record_name}.descriptors")
            mechanism = _nonempty_string(item["mechanism"], f"{record_name}.mechanism")
            if held_out != (mechanism == "torus_SDF"):
                raise ProtocolError(f"{record_name} violates held-out torus isolation")
            world_token = json.dumps(world, sort_keys=True, separators=(",", ":"))
            frozen = (seed, world_token, mechanism)
            if identity in clip_identity and clip_identity[identity] != frozen:
                raise ProtocolError(f"clip {identity!r} changes WorldSpec, seed, or mechanism")
            clip_identity[identity] = frozen
            starts_by_clip.setdefault(identity, []).append(start)
            parsed.append(
                DevelopmentWorld(
                    identity, seed, start, frame_ids,
                    _freeze(world),  # type: ignore[arg-type]
                    _freeze(descriptors),  # type: ignore[arg-type]
                    mechanism,
                )
            )
        identities = [(item.world_identity, item.window_start) for item in parsed]
        if len(identities) != len(set(identities)):
            raise ProtocolError(f"{name} repeats a clip/window identity")
        # Five overlapping windows span nine frames and expose occurrence strata 1..5.
        for identity, starts in starts_by_clip.items():
            ordered = sorted(starts)
            if len(ordered) < 5 or any(
                right != left + 1 for left, right in zip(ordered, ordered[1:])
            ):
                raise ProtocolError(
                    f"clip {identity!r} needs at least five consecutive window starts"
                )
        return tuple(parsed)

    in_generator = parse_group(root["in_generator"], "in_generator", False)
    held_out = parse_group(root["generator_held_out"], "generator_held_out", True)
    if {item.world_identity for item in in_generator} & {
        item.world_identity for item in held_out
    }:
        raise ProtocolError("in-generator and held-out clips share world identities")
    return DevelopmentWorlds(
        DEVELOPMENT_FORMAT, SCHEMA_VERSION, 201, status,
        MappingProxyType(validation), in_generator, held_out,
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
    parser = argparse.ArgumentParser(description="Inspect the AJAE schema-31 route.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    args = parser.parse_args()
    print(json.dumps(load_protocol(args.protocol).summary(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
