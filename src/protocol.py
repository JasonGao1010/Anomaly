#!/usr/bin/env python3
"""Read and validate AJAE's current causal-window scientific protocol.

``split.json`` is the sole source for data roles, STU initialization, the
current-anchored temporal mechanism, training boundaries, and official
evaluation. This module exposes only the small stable interface needed by
data loading, modeling, inference, and evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


DEFAULT_SPLIT_PATH = Path(__file__).resolve().parents[1] / "split.json"
SCHEMA_VERSION = 28
WINDOW_FRAMES = 5
GENERAL_COMPLETE_WINDOW_MINIMUM_CURRENT_FRAME = WINDOW_FRAMES - 1
NORMAL_201_COMPLETE_WINDOW_MINIMUM_CURRENT_FRAME = 8
HISTORY_LENGTHS = (0, 1, 2, 4)
TEMPORAL_SCALES = ("p16", "p8", "p4")
MODEL_NAME = "current-anchored factorized causal window encoder"
CLEAN_SELECT_RULE = (
    "At each scale and frame age, construct exactly one truth-selected h_mix real "
    "candidate from the current synthetic fraction and valid static/object history, "
    "plus one null candidate. Missing mass stays zero, and h_mix is never duplicated."
)
PROPOSAL_ORACLE_CANDIDATES_RULE = (
    "At every sparse query, present static-aligned and truth-motion-aligned evidence "
    "as exchangeable real hypotheses plus null, with shared candidate parameters "
    "inside each independent Proposal arm and no candidate-type embedding."
)
ORACLE_TRUTH_RULE = (
    "Generator truth may construct Clean Select, provide the truth-motion proposal, "
    "and define same-object or null supervision; it never enters q, k, v, temporal "
    "context, the anomaly classifier, a learned gate, or candidate-slot identity."
)
NULL_RULE = (
    "Append one learned-score, zero-value null candidate independently at every "
    "temporal scale and every available history age."
)
TEMPORAL_OUTPUT_RULE = (
    "For current point i, use z_win_i=z_cur_i+delta_i with "
    "delta_i=4*tanh(h_temporal(f_cur_i,h_hist_i)); zero real-history support gives "
    "delta_i=0 by construction."
)
TEMPORAL_STATE_ISOLATION_RULE = (
    "All three arms start from the same frozen stage-A current head and identical "
    "temporal initialization, then use independent temporal parameters, optimizers, "
    "and random states. Proposal arms receive no Clean Select gradient."
)

NORMAL_TRAINING_ID = ("train", 206)
NORMAL_VALIDATION_ID = ("train", 201)
PUBLIC_VALIDATION_IDS = (
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
HIDDEN_TEST_IDS = tuple(range(100, 125)) + tuple(range(126, 137)) + tuple(
    range(154, 169)
)
NORMAL_TRAINING_CLASS_MAP = {
    0: 255,
    1: 255,
    2: 255,
    10: 0,
    11: 1,
    13: 4,
    15: 2,
    16: 4,
    18: 3,
    20: 4,
    30: 5,
    31: 6,
    32: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    52: 255,
    60: 8,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
    99: 255,
    252: 0,
    253: 6,
    254: 5,
    255: 7,
    256: 4,
    257: 4,
    258: 3,
    259: 4,
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "dataset",
    "purpose",
    "task",
    "data",
    "label_semantics",
    "pretrained_model",
    "counterfactual_anomalies",
    "model",
    "training",
    "inference",
    "evaluation",
}
class ProtocolError(ValueError):
    """Report a malformed or scientifically inconsistent configuration."""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ProtocolError(f"{name} must be an object with string keys")
    return value


def _field(parent: Mapping[str, object], key: str, name: str) -> object:
    if key not in parent:
        raise ProtocolError(f"{name} is missing {key!r}")
    return parent[key]


def _object(parent: Mapping[str, object], key: str, name: str) -> Mapping[str, object]:
    return _mapping(_field(parent, key, name), f"{name}.{key}")


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProtocolError(
            f"{name} has missing keys {sorted(expected - actual)} and "
            f"unexpected keys {sorted(actual - expected)}"
        )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be a finite number")
    return result


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be an array")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    items = _list(value, name)
    result = tuple(_string(item, f"{name}[{i}]") for i, item in enumerate(items))
    if len(result) != len(set(result)):
        raise ProtocolError(f"{name} contains duplicate values")
    return result


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    items = _list(value, name)
    result = tuple(_integer(item, f"{name}[{i}]") for i, item in enumerate(items))
    if len(result) != len(set(result)):
        raise ProtocolError(f"{name} contains duplicate values")
    return result


def _frame_span(value: object, name: str, frames: int) -> FrameSpan:
    items = _list(value, name)
    if len(items) != 2:
        raise ProtocolError(f"{name} must contain inclusive start and end frames")
    span = FrameSpan(
        start=_integer(items[0], f"{name}[0]"),
        stop=_integer(items[1], f"{name}[1]") + 1,
    )
    if len(span) != frames:
        raise ProtocolError(f"{name} contains {len(span)} frames, expected {frames}")
    return span


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
    """A half-open contiguous range of source frames."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        _integer(self.start, "FrameSpan.start")
        _integer(self.stop, "FrameSpan.stop", minimum=1)
        if self.stop <= self.start:
            raise ProtocolError("FrameSpan must satisfy start < stop")

    def __len__(self) -> int:
        return self.stop - self.start

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.stop


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    """One complete STU sequence and its sole role in the study."""

    partition: str
    sequence_id: int
    role: str
    labels_available: bool
    span: FrameSpan | None

    def __post_init__(self) -> None:
        if self.partition not in {"train", "val", "test"}:
            raise ProtocolError("sequence partition must be train, val, or test")
        _integer(self.sequence_id, "SequenceSpec.sequence_id")
        _string(self.role, "SequenceSpec.role")
        if type(self.labels_available) is not bool:
            raise ProtocolError("SequenceSpec.labels_available must be boolean")
        if self.span is not None and not isinstance(self.span, FrameSpan):
            raise ProtocolError("SequenceSpec.span must be FrameSpan or None")

    @property
    def frames(self) -> int | None:
        return None if self.span is None else len(self.span)

    @property
    def uses_gradients(self) -> bool:
        return self.role == "normal_training"

    @property
    def supports_counterfactuals(self) -> bool:
        return self.role in {"normal_training", "normal_validation"}


@dataclass(frozen=True, slots=True)
class PretrainedModelSpec:
    """The official STU model state used to initialize AJAE."""

    source: str
    checkpoint: str
    initialized_components: tuple[str, ...]
    input_channels: tuple[str, ...]
    voxel_size_m: float
    query_count: int


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """The fixed STU point and anomaly-instance evaluation parameters."""

    minimum_range_m: float
    maximum_range_m: float
    minimum_anomaly_points_per_frame: int
    point_metrics: tuple[str, ...]
    normal_point_alarm_rate: float
    dbscan_epsilon_m: float
    dbscan_minimum_samples: int


class AJAEProtocol:
    """Validated, immutable view of one schema-28 ``split.json`` document."""

    def __init__(self, document: Mapping[str, object], *, path: Path) -> None:
        source = _mapping(document, "split.json")
        self.path = path.expanduser().resolve(strict=True)
        self._validate(source)
        self._document = _freeze(source)

        data = _object(source, "data", "split.json")
        normal_training = _object(data, "normal_training", "data")
        normal_validation = _object(data, "normal_validation", "data")
        public = _object(data, "public_anomaly_validation", "data")
        training_frames = _integer(
            _field(normal_training, "frames", "data.normal_training"),
            "data.normal_training.frames",
            minimum=1,
        )
        validation_frames = _integer(
            _field(normal_validation, "frames", "data.normal_validation"),
            "data.normal_validation.frames",
            minimum=1,
        )
        self.normal_training = SequenceSpec(
            partition="train",
            sequence_id=206,
            role="normal_training",
            labels_available=True,
            span=_frame_span(
                _field(normal_training, "frame_range", "data.normal_training"),
                "data.normal_training.frame_range",
                training_frames,
            ),
        )
        self.normal_validation = SequenceSpec(
            partition="train",
            sequence_id=201,
            role="normal_validation",
            labels_available=True,
            span=_frame_span(
                _field(normal_validation, "frame_range", "data.normal_validation"),
                "data.normal_validation.frame_range",
                validation_frames,
            ),
        )
        counts = _mapping(
            _field(public, "sequence_frame_counts", "data.public_anomaly_validation"),
            "data.public_anomaly_validation.sequence_frame_counts",
        )
        self.public_validation = tuple(
            SequenceSpec(
                partition="val",
                sequence_id=sequence_id,
                role="public_anomaly_validation",
                labels_available=True,
                span=FrameSpan(
                    0,
                    _integer(
                        _field(counts, str(sequence_id), "public.sequence_frame_counts"),
                        f"public.sequence_frame_counts.{sequence_id}",
                        minimum=1,
                    ),
                ),
            )
            for sequence_id in PUBLIC_VALIDATION_IDS
        )
        self.hidden_test = tuple(
            SequenceSpec(
                partition="test",
                sequence_id=sequence_id,
                role="hidden_test",
                labels_available=False,
                span=None,
            )
            for sequence_id in HIDDEN_TEST_IDS
        )

        pretrained = _object(source, "pretrained_model", "split.json")
        official_input = _object(pretrained, "official_input", "pretrained_model")
        self.pretrained_model = PretrainedModelSpec(
            source=_string(_field(pretrained, "source", "pretrained_model"), "pretrained_model.source"),
            checkpoint=_string(
                _field(pretrained, "checkpoint", "pretrained_model"),
                "pretrained_model.checkpoint",
            ),
            initialized_components=_string_tuple(
                _field(pretrained, "initialized_components", "pretrained_model"),
                "pretrained_model.initialized_components",
            ),
            input_channels=_string_tuple(
                _field(official_input, "channels", "pretrained_model.official_input"),
                "pretrained_model.official_input.channels",
            ),
            voxel_size_m=_number(
                _field(official_input, "voxel_size_m", "pretrained_model.official_input"),
                "pretrained_model.official_input.voxel_size_m",
            ),
            query_count=_integer(
                _field(pretrained, "query_count", "pretrained_model"),
                "pretrained_model.query_count",
                minimum=1,
            ),
        )

        evaluation = _object(source, "evaluation", "split.json")
        point_filter = _object(evaluation, "point_filter", "evaluation")
        point_range = _list(
            _field(point_filter, "source_frame_range_m", "evaluation.point_filter"),
            "evaluation.point_filter.source_frame_range_m",
        )
        threshold = _object(evaluation, "normal_alarm_threshold", "evaluation")
        instances = _object(evaluation, "anomaly_instances", "evaluation")
        spatial = _object(instances, "spatial_split", "evaluation.anomaly_instances")
        self.evaluation = EvaluationSpec(
            minimum_range_m=_number(point_range[0], "evaluation.point_filter.range[0]"),
            maximum_range_m=_number(point_range[1], "evaluation.point_filter.range[1]"),
            minimum_anomaly_points_per_frame=_integer(
                _field(point_filter, "minimum_anomaly_points_per_frame", "evaluation.point_filter"),
                "evaluation.point_filter.minimum_anomaly_points_per_frame",
                minimum=1,
            ),
            point_metrics=_string_tuple(
                _field(evaluation, "point_metrics", "evaluation"),
                "evaluation.point_metrics",
            ),
            normal_point_alarm_rate=_number(
                _field(threshold, "normal_point_alarm_rate", "evaluation.normal_alarm_threshold"),
                "evaluation.normal_alarm_threshold.normal_point_alarm_rate",
            ),
            dbscan_epsilon_m=_number(
                _field(spatial, "epsilon_m", "evaluation.anomaly_instances.spatial_split"),
                "evaluation.anomaly_instances.spatial_split.epsilon_m",
            ),
            dbscan_minimum_samples=_integer(
                _field(spatial, "minimum_samples", "evaluation.anomaly_instances.spatial_split"),
                "evaluation.anomaly_instances.spatial_split.minimum_samples",
                minimum=1,
            ),
        )

        # These immutable mappings preserve the existing scene/model/train API.
        self.counterfactual_anomalies = self._document["counterfactual_anomalies"]
        self.model = self._document["model"]
        self.training = self._document["training"]
        self.inference = self._document["inference"]
        self.normal_training_class_map = MappingProxyType(dict(NORMAL_TRAINING_CLASS_MAP))
        all_specs = (
            self.normal_training,
            self.normal_validation,
            *self.public_validation,
            *self.hidden_test,
        )
        self._sequences = {(spec.partition, spec.sequence_id): spec for spec in all_specs}

    @property
    def document(self) -> Mapping[str, object]:
        return self._document

    def plain_document(self) -> dict[str, object]:
        """Return the complete validated configuration using plain JSON values."""

        value = _plain(self._document)
        if not isinstance(value, dict):
            raise AssertionError("validated split document is not an object")
        return value

    @property
    def window_frames(self) -> int:
        return WINDOW_FRAMES

    @property
    def public_sequence_ids(self) -> tuple[int, ...]:
        return tuple(spec.sequence_id for spec in self.public_validation)

    @property
    def hidden_sequence_ids(self) -> tuple[int, ...]:
        return tuple(spec.sequence_id for spec in self.hidden_test)

    def sequence(self, partition: str, sequence_id: int) -> SequenceSpec:
        key = (_string(partition, "partition"), _integer(sequence_id, "sequence_id"))
        try:
            return self._sequences[key]
        except KeyError as error:
            raise ProtocolError(f"sequence {partition}/{sequence_id} has no AJAE role") from error

    def window_frame_ids(
        self, partition: str, sequence_id: int, current_frame: int
    ) -> tuple[int, ...]:
        """Return available causal scans without padding or frame repetition."""

        spec = self.sequence(partition, sequence_id)
        frame = _integer(current_frame, "current_frame")
        start = 0
        if spec.span is not None:
            if not spec.span.contains(frame):
                raise ProtocolError(f"frame {frame} lies outside {partition}/{sequence_id}")
            start = spec.span.start
        return tuple(range(max(start, frame - WINDOW_FRAMES + 1), frame + 1))

    def complete_window_frame_ids(
        self, partition: str, sequence_id: int, current_frame: int
    ) -> tuple[int, ...]:
        """Return one schema-28 five-scan window from an audited eligible anchor."""

        identity = (_string(partition, "partition"), _integer(sequence_id, "sequence_id"))
        frame = _integer(current_frame, "current_frame")
        minimum = (
            NORMAL_201_COMPLETE_WINDOW_MINIMUM_CURRENT_FRAME
            if identity == NORMAL_VALIDATION_ID
            else GENERAL_COMPLETE_WINDOW_MINIMUM_CURRENT_FRAME
        )
        if frame < minimum:
            raise ProtocolError(
                f"a complete schema-28 window for {identity[0]}/{identity[1]} "
                f"requires current_frame >= {minimum}"
            )
        result = self.window_frame_ids(identity[0], identity[1], frame)
        if len(result) != WINDOW_FRAMES:
            raise ProtocolError("a complete schema-28 window must contain five scans")
        return result

    def checkpoint_path(self, project_root: Path | str | None = None) -> Path:
        """Resolve the configured STU checkpoint without opening it."""

        root = self.path.parent if project_root is None else Path(project_root).expanduser().resolve()
        return (root / self.pretrained_model.checkpoint).resolve()

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": "STU",
            "model": MODEL_NAME,
            "window_frames": WINDOW_FRAMES,
            "general_complete_window_minimum_current_frame": (
                GENERAL_COMPLETE_WINDOW_MINIMUM_CURRENT_FRAME
            ),
            "normal_201_complete_window_minimum_current_frame": (
                NORMAL_201_COMPLETE_WINDOW_MINIMUM_CURRENT_FRAME
            ),
            "history_lengths": list(HISTORY_LENGTHS),
            "normal_training": {
                "partition": self.normal_training.partition,
                "sequence": self.normal_training.sequence_id,
                "frames": self.normal_training.frames,
            },
            "normal_validation": {
                "partition": self.normal_validation.partition,
                "sequence": self.normal_validation.sequence_id,
                "frames": self.normal_validation.frames,
            },
            "public_validation_sequences": list(self.public_sequence_ids),
            "public_evaluation_after_method_freeze": True,
            "hidden_test_sequences": list(self.hidden_sequence_ids),
            "checkpoint": self.pretrained_model.checkpoint,
            "frozen_spatial_representation": True,
            "point_metrics": list(self.evaluation.point_metrics),
        }

    @staticmethod
    def _validate(source: Mapping[str, object]) -> None:
        _exact_keys(source, _TOP_LEVEL_KEYS, "split.json")
        if _integer(_field(source, "schema_version", "split.json"), "schema_version") != SCHEMA_VERSION:
            raise ProtocolError(f"schema_version must be {SCHEMA_VERSION}")
        if _string(_field(source, "dataset", "split.json"), "dataset") != "STU":
            raise ProtocolError("dataset must be STU")
        _string(_field(source, "purpose", "split.json"), "purpose")
        AJAEProtocol._validate_task(source)
        AJAEProtocol._validate_data(source)
        AJAEProtocol._validate_labels_and_weights(source)
        AJAEProtocol._validate_counterfactuals(source)
        AJAEProtocol._validate_model_and_training(source)
        AJAEProtocol._validate_inference_and_evaluation(source)

    @staticmethod
    def _validate_task(source: Mapping[str, object]) -> None:
        task = _object(source, "task", "split.json")
        _exact_keys(
            task,
            {
                "name",
                "input",
                "output",
                "causal_window_frames",
                "history_lengths",
                "current_frame_only_output",
                "window_startup_rule",
            },
            "task",
        )
        if _string(task["name"], "task.name") != "AJAE":
            raise ProtocolError("task.name must be AJAE")
        if _integer(task["causal_window_frames"], "task.causal_window_frames") != WINDOW_FRAMES:
            raise ProtocolError("AJAE requires one current and four causal history scans")
        if _integer_tuple(task["history_lengths"], "task.history_lengths") != HISTORY_LENGTHS:
            raise ProtocolError("history ablations must be K=0,1,2,4")
        if task["current_frame_only_output"] is not True:
            raise ProtocolError("AJAE must output only the current frame")
        for key in ("input", "output"):
            _string(task[key], f"task.{key}")
        startup = _object(task, "window_startup_rule", "task")
        _exact_keys(
            startup,
            {
                "available_history_only",
                "history_padding_forbidden",
                "frame_repetition_forbidden",
                "general_complete_window_minimum_current_frame",
                "normal_201_complete_window_minimum_current_frame",
                "complete_window_reason",
            },
            "task.window_startup_rule",
        )
        _string(startup["available_history_only"], "task.window_startup_rule.available_history_only")
        _string(startup["complete_window_reason"], "task.window_startup_rule.complete_window_reason")
        if (
            startup["history_padding_forbidden"] is not True
            or startup["frame_repetition_forbidden"] is not True
            or _integer(
                startup["general_complete_window_minimum_current_frame"],
                "task.window_startup_rule.general_complete_window_minimum_current_frame",
            )
            != GENERAL_COMPLETE_WINDOW_MINIMUM_CURRENT_FRAME
            or _integer(
                startup["normal_201_complete_window_minimum_current_frame"],
                "task.window_startup_rule.normal_201_complete_window_minimum_current_frame",
            )
            != NORMAL_201_COMPLETE_WINDOW_MINIMUM_CURRENT_FRAME
        ):
            raise ProtocolError(
                "schema-28 complete windows start at frame 4 except normal 201, "
                "which starts at frame 8 after duplicate-source exclusion"
            )

    @staticmethod
    def _validate_data(source: Mapping[str, object]) -> None:
        data = _object(source, "data", "split.json")
        _exact_keys(
            data,
            {"normal_training", "normal_validation", "public_anomaly_validation", "hidden_test"},
            "data",
        )
        for key, identity, expected_frames in (
            ("normal_training", NORMAL_TRAINING_ID, 449),
            ("normal_validation", NORMAL_VALIDATION_ID, 682),
        ):
            value = _object(data, key, "data")
            found = (
                _string(_field(value, "partition", f"data.{key}"), f"data.{key}.partition"),
                _integer(_field(value, "sequence_id", f"data.{key}"), f"data.{key}.sequence_id"),
            )
            if found != identity:
                raise ProtocolError(f"data.{key} must identify {identity[0]}/{identity[1]}")
            frames = _integer(_field(value, "frames", f"data.{key}"), f"data.{key}.frames", minimum=1)
            if frames != expected_frames:
                raise ProtocolError(f"data.{key} must contain {expected_frames} frames")
            _frame_span(_field(value, "frame_range", f"data.{key}"), f"data.{key}.frame_range", frames)

        normal_training = _object(data, "normal_training", "data")
        if (
            _integer_tuple(
                _field(
                    normal_training,
                    "complete_causal_window_current_range",
                    "data.normal_training",
                ),
                "data.normal_training.complete_causal_window_current_range",
            )
            != (4, 448)
            or _integer(
                _field(
                    normal_training,
                    "complete_causal_windows",
                    "data.normal_training",
                ),
                "data.normal_training.complete_causal_windows",
            )
            != 445
        ):
            raise ProtocolError("normal 206 must use complete-window anchors 4 through 448")

        normal_validation = _object(data, "normal_validation", "data")
        if (
            _integer_tuple(
                _field(
                    normal_validation,
                    "complete_causal_window_current_range",
                    "data.normal_validation",
                ),
                "data.normal_validation.complete_causal_window_current_range",
            )
            != (8, 681)
            or _integer(
                _field(
                    normal_validation,
                    "complete_causal_windows",
                    "data.normal_validation",
                ),
                "data.normal_validation.complete_causal_windows",
            )
            != 674
            or _integer_tuple(
                _field(
                    normal_validation,
                    "future_window_threshold_excluded_current_frames",
                    "data.normal_validation",
                ),
                "data.normal_validation.future_window_threshold_excluded_current_frames",
            )
            != tuple(range(8))
        ):
            raise ProtocolError("normal 201 must use complete-window anchors 8 through 681")
        duplicate = _object(
            normal_validation, "source_duplicate_audit", "data.normal_validation"
        )
        if (
            duplicate.get("status")
            != "verified_exact_internal_scan_and_label_duplication"
            or _integer_tuple(
                _field(duplicate, "affected_source_frames", "source_duplicate_audit"),
                "source_duplicate_audit.affected_source_frames",
            )
            != (0, 1, 2, 3)
            or _integer(
                _field(duplicate, "extra_duplicate_slots", "source_duplicate_audit"),
                "source_duplicate_audit.extra_duplicate_slots",
            )
            != 786432
            or _integer(
                _field(
                    duplicate,
                    "extra_duplicate_real_returns",
                    "source_duplicate_audit",
                ),
                "source_duplicate_audit.extra_duplicate_real_returns",
            )
            != 627219
            or _integer(
                _field(
                    duplicate,
                    "extra_duplicate_valid_normal_points_under_binary_rule",
                    "source_duplicate_audit",
                ),
                "source_duplicate_audit.extra_duplicate_valid_normal_points_under_binary_rule",
            )
            != 572792
        ):
            raise ProtocolError("normal 201 duplicate-source audit changed")
        for key in ("frame_0", "frame_1", "frame_2", "frame_3", "scope"):
            _string(_field(duplicate, key, "source_duplicate_audit"), f"source_duplicate_audit.{key}")

        public = _object(data, "public_anomaly_validation", "data")
        if _string(_field(public, "partition", "public"), "public.partition") != "val":
            raise ProtocolError("public anomaly validation must use val")
        public_ids = _integer_tuple(_field(public, "sequence_ids", "public"), "public.sequence_ids")
        if public_ids != PUBLIC_VALIDATION_IDS:
            raise ProtocolError("public validation must use all 19 official sequences")
        if _integer(_field(public, "sequences", "public"), "public.sequences") != 19:
            raise ProtocolError("public validation sequence count must be 19")
        counts = _mapping(_field(public, "sequence_frame_counts", "public"), "public.sequence_frame_counts")
        if set(counts) != {str(item) for item in PUBLIC_VALIDATION_IDS}:
            raise ProtocolError("public frame counts do not match sequence ids")
        total = sum(
            _integer(counts[str(item)], f"public.sequence_frame_counts.{item}", minimum=1)
            for item in PUBLIC_VALIDATION_IDS
        )
        if total != _integer(_field(public, "frames", "public"), "public.frames", minimum=1):
            raise ProtocolError("public validation frame total is inconsistent")
        if public.get("use") != "official_public_evaluation_after_method_freeze" or public.get("method_freeze_required") is not True:
            raise ProtocolError("public anomaly labels may be accessed only after method freeze")

        hidden = _object(data, "hidden_test", "data")
        if _string(_field(hidden, "partition", "hidden"), "hidden.partition") != "test":
            raise ProtocolError("hidden sequences must use test")
        hidden_ids = _integer_tuple(_field(hidden, "sequence_ids", "hidden"), "hidden.sequence_ids")
        if hidden_ids != HIDDEN_TEST_IDS:
            raise ProtocolError("hidden test must use the 51 official identities")
        if _integer(_field(hidden, "sequences", "hidden"), "hidden.sequences") != 51:
            raise ProtocolError("hidden test sequence count must be 51")
        if hidden.get("labels_available") is not False:
            raise ProtocolError("hidden test labels must be unavailable")
        if hidden.get("use") != "official_hidden_test_submission_after_method_freeze":
            raise ProtocolError("hidden-test submission is allowed only after method freeze")

    @staticmethod
    def _validate_labels_and_weights(source: Mapping[str, object]) -> None:
        labels = _object(source, "label_semantics", "split.json")
        class_map = _mapping(_field(labels, "normal_training_class_map", "labels"), "labels.normal_training_class_map")
        expected = {str(raw): target for raw, target in NORMAL_TRAINING_CLASS_MAP.items()}
        if dict(class_map) != expected or any(type(value) is not int for value in class_map.values()):
            raise ProtocolError("normal training targets must match the official STU class map")
        binary = _object(labels, "binary_anomaly_training", "label_semantics")
        _exact_keys(
            binary,
            {
                "raw_semantic_0",
                "raw_semantic_2",
                "other_nonzero_semantics",
                "synthetic_members",
                "strict_zero_coordinate_slots",
                "normal_206_native_raw_2_points",
                "normal_201_native_raw_2_points",
                "audited_nonzero_semantic_target_255_normal_negatives",
                "basis",
            },
            "label_semantics.binary_anomaly_training",
        )
        if (
            binary["raw_semantic_0"] != "ignore"
            or binary["raw_semantic_2"] != "anomaly"
            or binary["other_nonzero_semantics"] != "normal"
            or _integer(
                binary["normal_206_native_raw_2_points"],
                "binary_anomaly_training.normal_206_native_raw_2_points",
            )
            != 0
            or _integer(
                binary["normal_201_native_raw_2_points"],
                "binary_anomaly_training.normal_201_native_raw_2_points",
            )
            != 0
        ):
            raise ProtocolError("binary anomaly labels must be raw 0 ignore and raw 2 positive")
        audited = _mapping(
            _field(
                binary,
                "audited_nonzero_semantic_target_255_normal_negatives",
                "binary_anomaly_training",
            ),
            "binary_anomaly_training.audited_nonzero_semantic_target_255_normal_negatives",
        )
        if dict(audited) != {
            "normal_206_raw_1_52_99": 1589676,
            "normal_201_raw_1_52_99": 76735,
        }:
            raise ProtocolError("audited binary normal-negative counts changed")
        for key in ("synthetic_members", "strict_zero_coordinate_slots", "basis"):
            _string(binary[key], f"binary_anomaly_training.{key}")
        public_labels = _object(labels, "public_anomaly_evaluation", "label_semantics")
        if dict(public_labels) != {
            "raw_semantic_0": "ignore",
            "raw_semantic_2": "anomaly",
            "other_nonzero_semantics": "normal",
        }:
            raise ProtocolError("public anomaly label meanings differ from STU")

        pretrained = _object(source, "pretrained_model", "split.json")
        if _string(_field(pretrained, "checkpoint", "pretrained"), "pretrained.checkpoint") != "weights/59p6pq_ens1.ckpt":
            raise ProtocolError("AJAE must use the STU 59p6 ensemble member")
        official_input = _object(pretrained, "official_input", "pretrained_model")
        if _string_tuple(_field(official_input, "channels", "official_input"), "official_input.channels") != ("intensity", "official_STU_distance"):
            raise ProtocolError("STU input must be intensity and official distance")
        if not math.isclose(_number(_field(official_input, "voxel_size_m", "official_input"), "official_input.voxel_size_m"), 0.05):
            raise ProtocolError("STU voxel size must be 0.05 m")
        if _integer(_field(pretrained, "query_count", "pretrained"), "pretrained.query_count") != 100:
            raise ProtocolError("STU Mask4Former-3D must use 100 queries")
        _string_tuple(_field(pretrained, "initialized_components", "pretrained"), "pretrained.initialized_components")
        _string(_field(pretrained, "initialization_role", "pretrained"), "pretrained.initialization_role")

    @staticmethod
    def _validate_counterfactuals(source: Mapping[str, object]) -> None:
        synthetic = _object(source, "counterfactual_anomalies", "split.json")
        _exact_keys(
            synthetic,
            {
                "source_data",
                "views",
                "object_construction",
                "one_physical_trajectory_per_window",
                "history_lengths",
                "history_ablation",
                "ray_rendering",
                "intensity_sampling",
                "current_visibility_rule",
                "ineligible_ground_rule",
                "oracle_truth_restriction",
                "validation_generation",
            },
            "counterfactual_anomalies",
        )
        if synthetic["source_data"] != "normal_training":
            raise ProtocolError("counterfactuals must come from normal 206")
        if _string_tuple(synthetic["views"], "counterfactual_anomalies.views") != (
            "original_normal_window",
            "counterfactual_window_with_anomaly",
        ):
            raise ProtocolError("counterfactual training requires original and inserted-object views")
        if synthetic["one_physical_trajectory_per_window"] is not True:
            raise ProtocolError("each counterfactual window must contain one physical trajectory")
        if _integer_tuple(synthetic["history_lengths"], "counterfactual_anomalies.history_lengths") != HISTORY_LENGTHS:
            raise ProtocolError("counterfactual history ablations must be K=0,1,2,4")
        for key in (
            "object_construction",
            "history_ablation",
            "current_visibility_rule",
            "ineligible_ground_rule",
            "oracle_truth_restriction",
            "validation_generation",
        ):
            _string(synthetic[key], f"counterfactual_anomalies.{key}")
        ray = _object(synthetic, "ray_rendering", "counterfactual_anomalies")
        for key in ("unit", "nearest_return_rule", "synthetic_front", "original_front", "no_intersection"):
            _string(_field(ray, key, "ray_rendering"), f"ray_rendering.{key}")
        if ray["synthetic_front"] != (
            "Emit one anomaly return and occlude the original return only when the "
            "synthetic surface is strictly more than 0.05 m nearer, i.e. "
            "d_synthetic+0.05<d_original."
        ):
            raise ProtocolError("synthetic occlusion must use the strict 0.05 m rule")
        _string_tuple(_field(ray, "preserved_properties", "ray_rendering"), "ray_rendering.preserved_properties")
        intensity = _object(
            synthetic, "intensity_sampling", "counterfactual_anomalies"
        )
        _exact_keys(
            intensity,
            {
                "reference",
                "strata",
                "rule",
                "verified_global_evidence",
                "provenance_limit",
            },
            "intensity_sampling",
        )
        _string(intensity["reference"], "intensity_sampling.reference")
        if _string_tuple(intensity["strata"], "intensity_sampling.strata") != (
            "range",
            "laser_beam",
            "surface_incidence_angle",
        ):
            raise ProtocolError("synthetic intensity strata changed")
        _string(intensity["rule"], "intensity_sampling.rule")
        _string(
            intensity["verified_global_evidence"],
            "intensity_sampling.verified_global_evidence",
        )
        _string(intensity["provenance_limit"], "intensity_sampling.provenance_limit")

    @staticmethod
    def _validate_model_and_training(source: Mapping[str, object]) -> None:
        model = _object(source, "model", "split.json")
        _exact_keys(
            model,
            {
                "name",
                "spatial_initialization",
                "current_anchor",
                "temporal_scales",
                "history_correspondence",
                "explicit_null",
                "factorized_temporal_update",
                "temporal_output",
                "prediction",
            },
            "model",
        )
        if model["name"] != MODEL_NAME:
            raise ProtocolError("schema 28 requires the current-anchored factorized window model")
        if _string_tuple(model["temporal_scales"], "model.temporal_scales") != TEMPORAL_SCALES:
            raise ProtocolError("temporal scales must be p16, p8, and p4")
        correspondence = _object(model, "history_correspondence", "model")
        _exact_keys(
            correspondence,
            {"clean_select", "proposal_oracle_candidates", "truth_use"},
            "model.history_correspondence",
        )
        expected_correspondence = {
            "clean_select": CLEAN_SELECT_RULE,
            "proposal_oracle_candidates": PROPOSAL_ORACLE_CANDIDATES_RULE,
            "truth_use": ORACLE_TRUTH_RULE,
        }
        if dict(correspondence) != expected_correspondence:
            raise ProtocolError("Oracle correspondence semantics changed")
        null = _object(model, "explicit_null", "model")
        _exact_keys(null, {"rule", "scales", "ages", "feature_value", "counts_as_real_history_support"}, "model.explicit_null")
        if null["rule"] != NULL_RULE:
            raise ProtocolError("explicit null must use one learned score per scale and age")
        if _string_tuple(null["scales"], "model.explicit_null.scales") != TEMPORAL_SCALES:
            raise ProtocolError("explicit null must exist at p16, p8, and p4")
        if _integer_tuple(null["ages"], "model.explicit_null.ages") != (1, 2, 3, 4):
            raise ProtocolError("explicit null must exist independently at each history age")
        if _number(null["feature_value"], "model.explicit_null.feature_value") != 0.0 or null["counts_as_real_history_support"] is not False:
            raise ProtocolError("null must be zero-valued and excluded from real support")
        for key in ("spatial_initialization", "current_anchor", "factorized_temporal_update", "temporal_output", "prediction"):
            _string(model[key], f"model.{key}")
        if model["temporal_output"] != TEMPORAL_OUTPUT_RULE:
            raise ProtocolError("the bounded current-anchored temporal output changed")

        training = _object(source, "training", "split.json")
        _exact_keys(
            training,
            {"gradient_data", "parameter_groups", "stage_a", "stage_b", "gradient_audit", "mechanism_experiment", "future_matcher"},
            "training",
        )
        if _string_tuple(training["gradient_data"], "training.gradient_data") != (
            "normal_206_original_windows",
            "normal_206_counterfactual_windows",
        ):
            raise ProtocolError("gradients may use only normal 206 and its counterfactuals")
        groups = _object(training, "parameter_groups", "training")
        _exact_keys(groups, {"always_frozen", "stage_a_trainable", "stage_b_trainable"}, "training.parameter_groups")
        frozen = _string_tuple(groups["always_frozen"], "training.parameter_groups.always_frozen")
        stage_a = _string_tuple(groups["stage_a_trainable"], "training.parameter_groups.stage_a_trainable")
        stage_b = _string_tuple(groups["stage_b_trainable"], "training.parameter_groups.stage_b_trainable")
        if frozen != (
            "STU_sparse_backbone",
            "STU_object_queries",
            "STU_normal_semantic_branch",
            "STU_instance_mask_branch",
        ) or stage_a != ("current_point_anomaly_head",) or stage_b != (
            "frame_age_embeddings",
            "temporal_p16",
            "temporal_p8",
            "temporal_p4",
            "temporal_point_delta",
        ):
            raise ProtocolError("schema-28 parameter groups changed")
        if set(frozen) & (set(stage_a) | set(stage_b)) or set(stage_a) & set(stage_b):
            raise ProtocolError("training parameter groups overlap")

        stage_a_spec = _object(training, "stage_a", "training")
        _exact_keys(stage_a_spec, {"purpose", "objective", "state_rule"}, "training.stage_a")
        if stage_a_spec["objective"] != "L_cur=balanced_BCE(z_cur,y)":
            raise ProtocolError("stage A must train only balanced current-frame BCE")
        stage_b_spec = _object(training, "stage_b", "training")
        _exact_keys(
            stage_b_spec,
            {
                "purpose",
                "history_lengths",
                "independent_arms",
                "state_isolation",
                "classification_objective",
                "window_loss",
                "normal_safety",
                "magnitude_control",
                "direct_match",
                "direct_null",
                "direct_weights",
                "state_rule",
            },
            "training.stage_b",
        )
        if _integer_tuple(stage_b_spec["history_lengths"], "training.stage_b.history_lengths") != (1, 2, 4):
            raise ProtocolError("stage B must expose K=1,2,4 histories")
        arms = _mapping(
            _field(stage_b_spec, "independent_arms", "training.stage_b"),
            "training.stage_b.independent_arms",
        )
        if dict(arms) != {
            "clean_select": "classification objective only; one truth-selected real candidate plus null per age",
            "proposal_direct": "Proposal-only classification plus direct same-object probability-mass and null supervision",
            "proposal_classification": "Proposal-only classification objective only",
        }:
            raise ProtocolError("stage B must use the three independent selector arms")
        if stage_b_spec["state_isolation"] != TEMPORAL_STATE_ISOLATION_RULE:
            raise ProtocolError("selector arms must not share temporal updates")
        expected_objectives = {
            "classification_objective": "L_class=L_win+L_safe+0.1*L_mag",
            "window_loss": "L_win=balanced_BCE(z_win,y)",
            "normal_safety": "L_safe=mean_{i in normal} ReLU(z_win_i-stop_gradient(z_cur_i))",
            "magnitude_control": "L_mag is class-balanced SmoothL1(delta_i,0) with beta=1.0.",
            "direct_match": "L_match=-sum_(i,k) r_i*log(sum_{j in P_(i,k)} a_direct_(i,j,k))/sum_(i,k) r_i over generated-object query-age pairs with at least one same-object candidate; P_(i,k) may contain both static and truth-motion candidates.",
            "direct_null": "L_null=-sum_(i,k) r_i*log(a_direct_(i,null,k))/sum_(i,k) r_i over generated-object query-age pairs with no same-object candidate but at least one valid competing real candidate. When every real candidate is invalid, null is structurally certain and the zero-gradient pair is reported but excluded from the loss denominator.",
            "direct_weights": "proposal_direct uses L_class+1.0*(L_match+L_null), with p16, p8, and p4 weighted equally; the other arms do not use direct correspondence loss.",
        }
        for key, expected in expected_objectives.items():
            if stage_b_spec[key] != expected:
                raise ProtocolError(f"training.stage_b.{key} changes the frozen objective")
        for value, name in ((stage_a_spec, "stage_a"), (stage_b_spec, "stage_b")):
            for key, item in value.items():
                if key not in {"history_lengths", "independent_arms"}:
                    _string(item, f"training.{name}.{key}")

        audit = _object(training, "gradient_audit", "training")
        if _integer(_field(audit, "windows", "gradient_audit"), "gradient_audit.windows") != 8:
            raise ProtocolError("the gradient audit must use eight windows")
        if _string_tuple(_field(audit, "terms", "gradient_audit"), "gradient_audit.terms") != (
            "L_win",
            "L_safe",
            "0.1*L_mag",
            "L_match",
            "L_null",
        ):
            raise ProtocolError("the gradient audit must separate classification and direct terms")
        if _string_tuple(
            _field(audit, "branches", "gradient_audit"),
            "gradient_audit.branches",
        ) != stage_b[1:]:
            raise ProtocolError("the gradient audit must cover every temporal branch")
        _string(_field(audit, "rule", "gradient_audit"), "gradient_audit.rule")

        mechanism = _object(training, "mechanism_experiment", "training")
        if mechanism.get("status") != "exploratory_only":
            raise ProtocolError("the Oracle mechanism work is exploratory only")
        completed = _object(
            mechanism, "completed_schema27_oracle_experiment", "mechanism_experiment"
        )
        screen = _object(
            mechanism, "next_deconfounded_screen", "mechanism_experiment"
        )
        follow_up = _object(
            mechanism, "conditional_confirmation", "mechanism_experiment"
        )
        if (
            completed.get("status") != "completed_historical_evidence_only"
            or completed.get("training_windows") != 96
            or completed.get("validation_windows") != 64
            or screen.get("status") != "not_run"
            or screen.get("training_windows") != 24
            or screen.get("validation_windows") != 16
            or tuple(screen.get("arms", ()))
            != ("clean_select", "proposal_direct", "proposal_classification")
            or follow_up.get("status") != "not_authorized_until_screen_passes"
            or follow_up.get("training_windows") != 96
            or follow_up.get("validation_windows") != 64
            or follow_up.get("requires_new_frozen_manifest") is not True
        ):
            raise ProtocolError("schema-28 experiments must proceed deconfounded 24/16 then new 96/64")
        _string(_field(completed, "boundary", "completed_schema27_oracle_experiment"), "completed_schema27_oracle_experiment.boundary")
        _string(_field(mechanism, "order", "mechanism_experiment"), "mechanism_experiment.order")
        public_rule = _string(_field(mechanism, "public_anomaly_access", "mechanism_experiment"), "mechanism_experiment.public_anomaly_access")
        if "Do not access" not in public_rule or "method is frozen" not in public_rule:
            raise ProtocolError("public anomaly evaluation must remain unavailable before freeze")
        matcher = _object(training, "future_matcher", "training")
        _exact_keys(
            matcher,
            {
                "implemented",
                "activation_rule",
                "direct_supervision",
                "truth_motion_after_activation",
            },
            "training.future_matcher",
        )
        if matcher["implemented"] is not False:
            raise ProtocolError("the learned candidate generator is not implemented")
        _string(matcher["activation_rule"], "training.future_matcher.activation_rule")
        _string(matcher["direct_supervision"], "training.future_matcher.direct_supervision")
        _string(
            matcher["truth_motion_after_activation"],
            "training.future_matcher.truth_motion_after_activation",
        )

    @staticmethod
    def _validate_inference_and_evaluation(source: Mapping[str, object]) -> None:
        inference = _object(source, "inference", "split.json")
        if _string(_field(inference, "mode", "inference"), "inference.mode") != "causal_current_frame":
            raise ProtocolError("inference must produce only the causal current frame")
        for key in ("history", "online_cache", "cache_reset", "output_restoration"):
            _string(_field(inference, key, "inference"), f"inference.{key}")
        _string_tuple(_field(inference, "reported_resources", "inference"), "inference.reported_resources")

        evaluation = _object(source, "evaluation", "split.json")
        point_filter = _object(evaluation, "point_filter", "evaluation")
        point_range = _list(_field(point_filter, "source_frame_range_m", "point_filter"), "point_filter.source_frame_range_m")
        if len(point_range) != 2 or not math.isclose(_number(point_range[0], "point_filter.range[0]"), 2.5) or not math.isclose(_number(point_range[1], "point_filter.range[1]"), 50.0):
            raise ProtocolError("official point range must be 2.5 through 50 m")
        if _integer(_field(point_filter, "minimum_anomaly_points_per_frame", "point_filter"), "point_filter.minimum_anomaly_points_per_frame") != 5:
            raise ProtocolError("official evaluation requires 5 anomaly points")
        if _string_tuple(_field(evaluation, "point_metrics", "evaluation"), "evaluation.point_metrics") != ("AP", "AUROC", "FPR95"):
            raise ProtocolError("point metrics must be AP, AUROC, and FPR95")
        threshold = _object(evaluation, "normal_alarm_threshold", "evaluation")
        if (
            threshold.get("source")
            != "normal_201_complete_causal_window_original_view"
            or _integer_tuple(
                _field(threshold, "eligible_current_frame_range", "threshold"),
                "threshold.eligible_current_frame_range",
            )
            != (8, 681)
            or _integer(
                _field(threshold, "eligible_windows", "threshold"),
                "threshold.eligible_windows",
            )
            != 674
            or _integer_tuple(
                _field(threshold, "excluded_current_frames", "threshold"),
                "threshold.excluded_current_frames",
            )
            != tuple(range(8))
        ):
            raise ProtocolError(
                "normal window threshold must exclude current frames 0 through 7"
            )
        if not math.isclose(_number(_field(threshold, "normal_point_alarm_rate", "threshold"), "threshold.normal_point_alarm_rate"), 0.001):
            raise ProtocolError("normal point alarm rate must be 0.001")
        instances = _object(evaluation, "anomaly_instances", "evaluation")
        spatial = _object(instances, "spatial_split", "anomaly_instances")
        if spatial.get("method") != "DBSCAN" or not math.isclose(_number(_field(spatial, "epsilon_m", "spatial"), "spatial.epsilon_m"), 1.0) or _integer(_field(spatial, "minimum_samples", "spatial"), "spatial.minimum_samples") != 1:
            raise ProtocolError("DBSCAN must use epsilon 1.0 m and one sample")
        baseline = _object(evaluation, "baseline", "evaluation")
        if baseline.get("method") != "NDP" or baseline.get("used_inside_AJAE") is not False:
            raise ProtocolError("NDP must remain an independent baseline")
        if "After method freeze" not in _string(_field(evaluation, "public_evaluation", "evaluation"), "evaluation.public_evaluation"):
            raise ProtocolError("official public evaluation must occur after method freeze")


def load_protocol(path: Path | str = DEFAULT_SPLIT_PATH) -> AJAEProtocol:
    """Load one JSON document and validate every scientific role."""

    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load protocol: {resolved}") from error
    return AJAEProtocol(_mapping(document, "split.json"), path=resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect AJAE schema-28 data roles and causal windows."
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--partition", choices=("train", "val", "test"))
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--frame", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_protocol(args.split)
    supplied = (args.partition, args.sequence, args.frame)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            raise SystemExit("--partition, --sequence, and --frame must be used together")
        output: object = {
            "partition": args.partition,
            "sequence": args.sequence,
            "current_frame": args.frame,
            "frame_ids": list(protocol.window_frame_ids(args.partition, args.sequence, args.frame)),
        }
    else:
        output = protocol.summary()
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
