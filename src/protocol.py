#!/usr/bin/env python3
"""Load the sole active AJAE schema-31 scientific contract."""

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
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "protocol.json"
SCHEMA_VERSION = 31
WINDOW_FRAMES = 5
WINDOW_MEMBER_OFFSETS = (0, 1, 2, 3, 4)
DEVELOPMENT_FORMAT = "ajae-development-window-worlds-v3"
R02_THRESHOLD_FORMAT = "ajae-r02-result-blind-thresholds-v1"
R02_VERDICT_FORMAT = "ajae-r02-scientific-verdict-v1"
R02_VALIDATION_KEYS = (
    "visual_review_passed",
    "descriptor_integrity_passed",
    "proxy_control_matching_passed",
    "densification_passed",
    "shortcut_audit_passed",
)
R02_MATCHING_FEATURES = (
    "log1p_joint_visible_return_count",
    "log1p_joint_spatial_voxel_count",
    "log_densification_gain",
    "median_distance_m",
    "occlusion_rate",
    "minimum_visible_return_height_m",
    "visible_scan_count",
)
R02_SHORTCUT_SEED = 3102

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
NORMAL_CONTROL_SEMANTICS = (10, 11, 15, 18, 20, 30, 31, 32)
MOVING_NORMAL_SEMANTICS = (252, 253, 254, 255, 256, 257, 258, 259)

_STATE_MACHINE = (
    "R00",
    "R01",
    "R02",
    "R03",
    "R04",
    "R05",
    "G2",
    "G3",
    "S01",
    "M01",
    "V01",
    "T01",
)
_ROOT_KEYS = {
    "schema_version",
    "status",
    "authority",
    "scientific_contract",
    "claims",
    "claim_exclusions",
    "data",
    "window",
    "labels",
    "render",
    "stu",
    "model",
    "experiments",
    "training",
    "evaluation",
    "state_machine",
    "decision_gates",
    "historical_evidence",
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


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: object, name: str) -> str:
    digest = _nonempty_string(value, name)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _plan_digest(namespace: str, *parts: object) -> bytes:
    value = ":".join((namespace, *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("ascii")).digest()


def _plan_seed(namespace: str, position: int, start: int) -> int:
    return int.from_bytes(
        _plan_digest(namespace, "world", position, start)[:8], "big"
    ) & ((1 << 63) - 1)


def r02_audit_algorithm_identity() -> str:
    """Identify the deterministic matching and shortcut algorithms used at R02."""

    return _sha256_json(
        {
            "format": "ajae-schema31-r02-audit-algorithms-v2",
            "matching": "support_stratified_linear_sum_assignment_standardized_euclidean",
            "features": R02_MATCHING_FEATURES,
            "shortcut_model": "standardized_logistic_regression",
            "shortcut_seed": R02_SHORTCUT_SEED,
            "shortcut_split": "world_identity_grouped_80_20",
            "shortcut_iterations": 300,
            "shortcut_l2": 0.001,
            "loader_rule": "recompute_all_matching_and_shortcut_statistics_from_frozen_descriptors",
            "not_computable_rule": "persist_irreversible_R02_failure",
        }
    )


def _r02_thresholds(value: object) -> Mapping[str, object] | None:
    """Validate a pending marker or the sole structured R02 threshold record."""

    if value == "freeze_result_blind_in_R02":
        return None
    record = _mapping(value, "render.proxy_control_matching.thresholds")
    keys = {
        "format",
        "frozen_result_blind",
        "minimum_visual_reviewed_clips",
        "minimum_matched_pairs",
        "maximum_absolute_standardized_mean_difference",
        "minimum_median_proxy_joint_visible_return_count",
        "minimum_median_proxy_densification_gain",
        "minimum_proxy_fraction_densification_gain_above_one",
        "maximum_shortcut_balanced_accuracy",
        "maximum_shortcut_absolute_auroc_deviation_from_half",
    }
    _exact_keys(record, keys, "R02 thresholds")
    _expect(record["format"], R02_THRESHOLD_FORMAT, "R02 threshold format")
    _expect(record["frozen_result_blind"], True, "R02 result-blind freeze")
    if (
        _integer(
            record["minimum_visual_reviewed_clips"],
            "R02 minimum visual clips",
            minimum=1,
        )
        != 24
    ):
        raise ProtocolError("R02 visual review must cover all 24 development clips")
    _integer(record["minimum_matched_pairs"], "R02 minimum matched pairs", minimum=10)
    if (
        _number(
            record["maximum_absolute_standardized_mean_difference"],
            "R02 maximum matching imbalance",
        )
        < 0.0
    ):
        raise ProtocolError("R02 maximum matching imbalance cannot be negative")
    if (
        _number(
            record["minimum_median_proxy_joint_visible_return_count"],
            "R02 minimum median proxy returns",
        )
        <= 0.0
    ):
        raise ProtocolError("R02 minimum median proxy returns must be positive")
    if (
        _number(
            record["minimum_median_proxy_densification_gain"],
            "R02 minimum median densification gain",
        )
        <= 1.0
    ):
        raise ProtocolError("R02 median densification threshold must exceed one")
    fraction = _number(
        record["minimum_proxy_fraction_densification_gain_above_one"],
        "R02 minimum densified proxy fraction",
    )
    if not 0.0 <= fraction <= 1.0:
        raise ProtocolError("R02 minimum densified proxy fraction must be in [0,1]")
    balanced = _number(
        record["maximum_shortcut_balanced_accuracy"],
        "R02 maximum shortcut balanced accuracy",
    )
    if not 0.5 <= balanced <= 1.0:
        raise ProtocolError("R02 shortcut balanced-accuracy limit must be in [0.5,1]")
    deviation = _number(
        record["maximum_shortcut_absolute_auroc_deviation_from_half"],
        "R02 maximum shortcut AUROC deviation",
    )
    if not 0.0 <= deviation <= 0.5:
        raise ProtocolError("R02 shortcut AUROC deviation must be in [0,0.5]")
    return record


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
        if (
            tuple(sorted(set(self.excluded_source_frames)))
            != self.excluded_source_frames
        ):
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
class DevelopmentWindow:
    """One five-scan view within a frozen synthetic clip world."""

    identity: str
    window_start: int
    frame_ids: tuple[int, ...]
    source_observation_identities: tuple[str, ...]
    descriptors: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class DevelopmentClip:
    """One WorldSpec shared without change by all overlapping windows."""

    identity: str
    world_identity: str
    clip_start: int
    frame_ids: tuple[int, ...]
    renderer_identity: str
    mechanism: str
    source_observation_identities: tuple[str, ...]
    world: Mapping[str, object]
    report: Mapping[str, object]
    windows: tuple[DevelopmentWindow, ...]


@dataclass(frozen=True, slots=True)
class DevelopmentWorlds:
    format: str
    protocol_schema: int
    protocol_identity: str
    plan_identity: str
    population_identity: str | None
    sequence_id: int
    status: str
    validation: Mapping[str, bool]
    scientific_verdict: Mapping[str, object] | None
    clips: tuple[DevelopmentClip, ...]

    @property
    def validated(self) -> bool:
        return (
            self.status == "validated_frozen"
            and set(self.validation) == set(R02_VALIDATION_KEYS)
            and all(self.validation.values())
            and self.scientific_verdict is not None
        )

    @property
    def adjudicated(self) -> bool:
        return (
            self.status in {"validated_frozen", "adjudicated_failed_R02"}
            and self.scientific_verdict is not None
        )

    @property
    def windows(self) -> tuple[DevelopmentWindow, ...]:
        return tuple(window for clip in self.clips for window in clip.windows)

    @property
    def in_generator(self) -> tuple[DevelopmentClip, ...]:
        return tuple(item for item in self.clips if item.mechanism == "in_generator")


class AJAEProtocol:
    """Validated immutable view of the schema-31 route."""

    def __init__(self, document: Mapping[str, object], *, path: Path) -> None:
        self._validate(document)
        self.path = path.expanduser().resolve(strict=True)
        self.schema_version = SCHEMA_VERSION
        self._document = _freeze(document)
        for name in (
            "status",
            "authority",
            "scientific_contract",
            "claims",
            "claim_exclusions",
            "data",
            "window",
            "labels",
            "render",
            "stu",
            "model",
            "experiments",
            "training",
            "state_machine",
            "decision_gates",
            "historical_evidence",
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
            self.normal_training,
            self.development_sequence,
            *self.public_validation,
            *self.hidden_test,
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
    def scientific_identity(self) -> str:
        """Hash only rules that determine data, models, training, and evaluation."""

        names = (
            "scientific_contract",
            "data",
            "window",
            "labels",
            "render",
            "stu",
            "model",
            "experiments",
            "training",
            "evaluation",
            "decision_gates",
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            **{name: _plain(self._document[name]) for name in names},
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def development_population_identity(self) -> str:
        """Bind development worlds only to rules that determine their content."""

        data = _mapping(self._document["data"], "data")
        evaluation = _mapping(self._document["evaluation"], "evaluation")
        payload = {
            "format": "ajae-schema31-development-population-protocol-v1",
            "schema_version": SCHEMA_VERSION,
            "scientific_contract": _plain(self._document["scientific_contract"]),
            "development_data": _plain(data["development"]),
            "window": _plain(self._document["window"]),
            "labels": _plain(self._document["labels"]),
            "render": _plain(self._document["render"]),
            "synthetic_development": _plain(evaluation["synthetic_development"]),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def renderer_identity(self) -> str:
        """Bind the shared renderer to its rules and frozen input artifacts."""

        return _sha256_json(
            {
                "format": "ajae-schema31-renderer-identity-v1",
                "render": self._document["render"],
                "labels": self._document["labels"],
            }
        )

    @property
    def r02_thresholds(self) -> Mapping[str, object] | None:
        return _r02_thresholds(self.render["proxy_control_matching"]["thresholds"])

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
        identifier = _integer(sequence_id, "support pool sequence")
        key = f"train/{identifier}"
        pools = _mapping(self.render["qualified_support_pools"], "support pools")
        if key not in pools:
            raise ProtocolError(f"no frozen support pool exists for {key}")
        record = _mapping(pools[key], f"support pool {key}")
        root = (
            self.path.parent
            if project_root is None
            else Path(project_root).expanduser().resolve()
        )
        return (root / str(record["file"])).resolve()

    def training_bank_plan(self) -> tuple[Mapping[str, object], ...]:
        """Return the sole ordered 445-entry train/206 generation plan."""

        plan = _mapping(
            self.training["bank"]["generation_plan"], "bank generation plan"
        )
        namespace = str(plan["namespace"])
        starts = sorted(
            self.normal_training.legal_window_starts(),
            key=lambda start: (_plan_digest(namespace, "window", start), start),
        )
        count = int(plan["formal_entry_count"])
        if len(starts) != count:
            raise ProtocolError(
                f"training plan requires {count} legal starts, observed {len(starts)}"
            )
        cycle = tuple(str(value) for value in plan["world_type_cycle"])
        stride = int(plan["observation_attempt_stride"])
        attempts = int(plan["maximum_observation_attempts"])
        return tuple(
            MappingProxyType(
                {
                    "position": position,
                    "source_frames": tuple(
                        start + offset for offset in WINDOW_MEMBER_OFFSETS
                    ),
                    "world_type": cycle[position % len(cycle)],
                    "root_seed": _plan_seed(namespace, position, start),
                    "observation_attempt_selection": plan[
                        "observation_attempt_selection"
                    ],
                    "observation_attempt_stride": stride,
                    "maximum_observation_attempts": attempts,
                }
            )
            for position, start in enumerate(starts)
        )

    @property
    def training_bank_plan_identity(self) -> str:
        return _sha256_json(
            {
                "format": "ajae-schema31-window-train-bank-plan-v1",
                "entries": self.training_bank_plan(),
            }
        )

    def _all_synthetic_clip_plan(self) -> tuple[Mapping[str, object], ...]:
        """Freeze 24 development and six unopened S01 designs without rendering."""

        synthetic = _mapping(
            self.evaluation_document["synthetic_development"],
            "synthetic development",
        )
        generation = _mapping(synthetic["generation_plan"], "development plan")
        namespace = str(generation["namespace"])
        span = self.development_sequence.span
        if span is None:
            raise ProtocolError("development sequence span is unavailable")
        excluded = frozenset(self.development_sequence.excluded_source_frames)
        candidates = tuple(
            start
            for start in range(span.start, span.stop - 9 + 1)
            if not excluded.intersection(range(start, start + 9))
        )
        ranked = sorted(
            candidates,
            key=lambda start: (_plan_digest(namespace, "clip", start), start),
        )
        selected: list[int] = []
        occupied: set[int] = set()
        for start in ranked:
            frames = set(range(start, start + 9))
            if occupied.isdisjoint(frames):
                selected.append(start)
                occupied.update(frames)
                if len(selected) == 30:
                    break
        if len(selected) != 30:
            raise ProtocolError(
                "development plan cannot select 30 non-overlapping clips"
            )
        stride = int(generation["observation_attempt_stride"])
        attempts = int(generation["maximum_observation_attempts"])
        return tuple(
            MappingProxyType(
                {
                    "position": position,
                    "clip_start": start,
                    "source_frames": tuple(range(start, start + 9)),
                    "mechanism": "in_generator" if position < 24 else "torus_SDF",
                    "root_seed": _plan_seed(namespace, position, start),
                    "observation_attempt_selection": generation[
                        "observation_attempt_selection"
                    ],
                    "observation_attempt_stride": stride,
                    "maximum_observation_attempts": attempts,
                }
            )
            for position, start in enumerate(selected)
        )

    def development_clip_plan(self) -> tuple[Mapping[str, object], ...]:
        """Return only the 24 in-generator clips allowed before S01."""

        return self._all_synthetic_clip_plan()[:24]

    @property
    def development_clip_plan_identity(self) -> str:
        return _sha256_json(
            {
                "format": "ajae-schema31-development-clip-plan-v1",
                "entries": self.development_clip_plan(),
            }
        )

    def held_out_synthetic_shift_plan(self) -> tuple[Mapping[str, object], ...]:
        """Return the six frozen designs whose observations stay sealed until S01."""

        return self._all_synthetic_clip_plan()[24:]

    @property
    def held_out_synthetic_shift_plan_identity(self) -> str:
        return _sha256_json(
            {
                "format": "ajae-schema31-held-out-synthetic-shift-plan-v1",
                "renderer_identity": self.renderer_identity,
                "recipe": self.render["anomaly_proxies"]["held_out_recipe"],
                "entries": self.held_out_synthetic_shift_plan(),
            }
        )

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
            {
                "scientific_document",
                "state_machine",
                "history_baseline_commit",
                "history_tag",
                "supersedes",
            },
            "authority",
        )
        _expect(
            authority["scientific_document"],
            "AJAE新主线方案.md",
            "authority.scientific_document",
        )
        _expect(
            authority["state_machine"],
            "AJAE实验执行状态机.md",
            "authority.state_machine",
        )
        digest = _nonempty_string(
            authority["history_baseline_commit"], "history baseline"
        )
        if not re.fullmatch(r"[0-9a-f]{40}", digest):
            raise ProtocolError("history_baseline_commit must be a lowercase SHA-1")
        _expect(
            authority["history_tag"],
            "schema30-history-baseline",
            "authority.history_tag",
        )
        _expect(
            authority["supersedes"],
            "schema30_center_target_temporal_message_passing_route",
            "authority.supersedes",
        )

        contract = _mapping(source["scientific_contract"], "scientific contract")
        expected = {
            "observation_unit": "one_complete_five_scan_window",
            "privileged_frame": None,
            "all_window_members_equally_supervised": True,
            "all_visible_returns_receive_logits": True,
            "learned_scan_time_or_member_position_input": False,
            "full_model_spatial_operations": [
                "joint_voxelization",
                "joint_radius_neighborhood",
                "joint_knn_decoding",
            ],
            "sequence_score": "equal_mean_of_window_probabilities_by_original_frame_ray_identity",
        }
        _exact_keys(contract, set(expected), "scientific_contract")
        for name, value in expected.items():
            _expect(contract[name], value, f"scientific_contract.{name}")

        claims = _mapping(source["claims"], "claims")
        _exact_keys(
            claims,
            {"proxy_supervision", "joint_densification", "real_ood_transfer"},
            "claims",
        )
        for name, value in claims.items():
            _nonempty_string(value, f"claims.{name}")
        _expect(
            source["claim_exclusions"],
            [
                "motion_unknown_detection",
                "explicit_motion_understanding",
                "object_tracking",
                "future_frame_assistance",
                "privileged_frame_completion",
            ],
            "claim_exclusions",
        )
        _expect(source["state_machine"], list(_STATE_MACHINE), "state_machine")

        history = _mapping(source["historical_evidence"], "historical evidence")
        history_keys = {
            "evidence_source_schema",
            "inheritance_rule",
            "continues_with_original_scope",
            "old_distribution_only",
            "reusable_only_after_schema31_requalification",
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
        _exact_keys(
            data,
            {
                "normal_training",
                "development",
                "public_anomaly_validation",
                "hidden_test",
            },
            "data",
        )
        normal = _mapping(data["normal_training"], "normal training")
        normal_expected = {
            "partition": "train",
            "sequence_id": 206,
            "role": "renderer_calibration_window_bank_and_all_parameter_updates",
            "labels_available": True,
            "frame_range_inclusive": [0, 448],
            "excluded_source_frames": [],
        }
        _exact_keys(normal, set(normal_expected), "data.normal_training")
        for name, value in normal_expected.items():
            _expect(normal[name], value, f"data.normal_training.{name}")

        development = _mapping(data["development"], "development sequence")
        development_expected = {
            "partition": "train",
            "sequence_id": 201,
            "role": "development_only_no_formal_gradients",
            "labels_available": True,
            "frame_range_inclusive": [0, 681],
            "excluded_source_frames": [0, 1, 2, 3],
            "exclusion_reason": "verified exact internal duplication in scans and labels",
        }
        _exact_keys(development, set(development_expected), "data.development")
        for name, value in development_expected.items():
            _expect(development[name], value, f"data.development.{name}")

        common_keys = {
            "partition",
            "role",
            "labels_available",
            "method_freeze_required",
            "sequence_ids",
        }
        public = _mapping(data["public_anomaly_validation"], "public validation")
        hidden = _mapping(data["hidden_test"], "hidden test")
        for record, name, partition, role, labels, ids in (
            (
                public,
                "public_anomaly_validation",
                "val",
                "one_time_real_anomaly_confirmation_after_method_freeze",
                True,
                PUBLIC_ANOMALY_IDS,
            ),
            (
                hidden,
                "hidden_test",
                "test",
                "final_hidden_test_after_supported_public_confirmation",
                False,
                HIDDEN_TEST_IDS,
            ),
        ):
            _exact_keys(record, common_keys, f"data.{name}")
            _expect(record["partition"], partition, f"data.{name}.partition")
            _expect(record["role"], role, f"data.{name}.role")
            _expect(record["labels_available"], labels, f"data.{name}.labels_available")
            _expect(
                record["method_freeze_required"],
                True,
                f"data.{name}.method_freeze_required",
            )
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
        _exact_keys(
            labels,
            {
                "packed_label",
                "binary_anomaly",
                "normal_control_semantic_ids",
                "moving_normal_semantic_ids",
                "normal_semantic_class_map",
            },
            "labels",
        )
        _expect(
            labels["packed_label"],
            {"semantic_bits_inclusive": [0, 15], "instance_bits_inclusive": [16, 31]},
            "labels.packed_label",
        )
        _expect(
            labels["binary_anomaly"],
            {
                "raw_semantic_0": "ignore_unless_replaced_by_a_valid_inserted_return",
                "raw_semantic_2": "anomaly",
                "other_nonzero_semantics": "normal",
                "normal_control_return": "normal",
                "anomaly_proxy_return": "anomaly",
            },
            "labels.binary_anomaly",
        )
        if (
            _int_tuple(labels["normal_control_semantic_ids"], "normal controls")
            != NORMAL_CONTROL_SEMANTICS
        ):
            raise ProtocolError("normal-control semantic IDs changed")
        if (
            _int_tuple(labels["moving_normal_semantic_ids"], "moving normals")
            != MOVING_NORMAL_SEMANTICS
        ):
            raise ProtocolError("moving-normal semantic IDs changed")
        class_map = _mapping(labels["normal_semantic_class_map"], "normal class map")
        if not class_map or any(not raw.isdigit() for raw in class_map):
            raise ProtocolError("normal semantic class map keys must be decimal IDs")
        for raw, target in class_map.items():
            _integer(target, f"normal semantic class {raw}")
        if any(
            str(item) not in class_map
            for item in (*NORMAL_CONTROL_SEMANTICS, *MOVING_NORMAL_SEMANTICS)
        ):
            raise ProtocolError("normal class map omits a renderer/STU semantic")

    @staticmethod
    def _validate_render(render: Mapping[str, object]) -> None:
        keys = {
            "geometry_schema",
            "source_sequence_id",
            "sensor",
            "calibration_file",
            "calibration_sha256",
            "qualified_support_pools",
            "ray_grid",
            "normal_controls",
            "anomaly_proxies",
            "world_type_probabilities",
            "common_entity_rules",
            "sensor_model",
            "physical_scope_excludes",
            "world_unit",
            "freeze_before_render",
            "shared_renderer_for_normal_control_and_proxy",
            "window_descriptors",
            "proxy_control_matching",
            "forbidden_densification",
            "rendered_observation_identity_fields",
        }
        _exact_keys(render, keys, "render")
        _expect(render["geometry_schema"], 7, "render.geometry_schema")
        _expect(render["source_sequence_id"], 206, "render.source_sequence_id")
        _expect(
            render["sensor"],
            "OS1-128_canonical_ray_first_return_approximation",
            "render.sensor",
        )
        _nonempty_string(render["calibration_file"], "render.calibration_file")
        _sha256(render["calibration_sha256"], "render.calibration_sha256")
        support_pools = _mapping(
            render["qualified_support_pools"], "render.qualified_support_pools"
        )
        _exact_keys(
            support_pools, {"train/206", "train/201"}, "qualified support pools"
        )
        expected_pool_digests = {
            "train/206": "0e6e7299157f5e9ced0716f6dd14881c66ba1bca0cc9c372550e56f426ea844d",
            "train/201": "fc3646fbc145cdc29d2cf203835a3e0018bacbc6eaf714e091d21f7b93bfaf50",
        }
        for source, expected_digest in expected_pool_digests.items():
            record = _mapping(support_pools[source], f"support pool {source}")
            _exact_keys(record, {"file", "sha256"}, f"support pool {source}")
            _nonempty_string(record["file"], f"support pool {source} file")
            _expect(record["sha256"], expected_digest, f"support pool {source} digest")
        _expect(render["world_unit"], "WindowWorld", "render.world_unit")
        _expect(
            render["shared_renderer_for_normal_control_and_proxy"],
            True,
            "render.shared_renderer",
        )
        _expect(
            render["freeze_before_render"],
            [
                "five_source_frames",
                "normal_controls",
                "anomaly_proxies",
                "world_positions",
                "orientations",
                "scales",
                "materials",
                "all_random_seeds",
            ],
            "render.freeze_before_render",
        )
        _expect(
            render["forbidden_densification"],
            [
                "synthetic_point_completion",
                "bottom_return_insertion",
                "scan_duplication",
                "single_scan_copying",
            ],
            "render.forbidden_densification",
        )
        _expect(
            render["rendered_observation_identity_fields"],
            [
                "partition",
                "sequence_id",
                "frame_id",
                "xyzi",
                "lidar_pose",
                "real_slots",
                "packed_labels",
            ],
            "render.rendered_observation_identity_fields",
        )

        ray = _mapping(render["ray_grid"], "render.ray_grid")
        _expect(
            ray,
            {
                "beam_count": 128,
                "column_count": 1024,
                "canonical_identity": ["beam_id", "azimuth_column"],
                "canonical_sha256": "b2ca37bf8e288aa3f7d0ca571fd3e459b054b5445e2c9da3b49e3af02c1dc627",
                "file_slot_role": "input_output_mapping_only",
            },
            "render.ray_grid",
        )
        controls = _mapping(render["normal_controls"], "render.normal_controls")
        if (
            _int_tuple(controls.get("semantic_ids"), "normal-control semantics")
            != NORMAL_CONTROL_SEMANTICS
        ):
            raise ProtocolError("renderer normal-control semantics differ from labels")
        _expect(controls.get("source_sequence_id"), 206, "normal-control source")
        proxies = _mapping(render["anomaly_proxies"], "render.anomaly_proxies")
        _expect(
            proxies.get("training_mechanisms"),
            [
                "superquadric",
                "constructive_overlap_union",
                "bend",
                "twist",
                "taper",
                "low_frequency_surface",
            ],
            "proxy mechanisms",
        )
        _expect(proxies.get("held_out_mechanism"), "torus_SDF", "held-out mechanism")
        _expect(
            proxies.get("held_out_recipe"),
            {
                "anomaly_proxy_count": 1,
                "normal_control_count": 0,
                "outer_diameter_m_range_inclusive": [0.4, 3.0],
                "tube_to_outer_radius_range_inclusive": [0.18, 0.35],
                "minimum_tube_radius_m": 0.04,
                "minimum_major_radius_m": 0.15,
                "intersection_steps": 160,
                "support_semantic_ids": [40, 48, 49],
                "material_sampler": "shared_MaterialSpec_seeded_quantile_roughness_return_bias",
                "placement_pipeline": "shared_support_pool_grounding_collision_and_visibility",
                "placement_namespace": "schema31-held-out-torus-v1",
            },
            "held-out recipe",
        )

        probabilities = _mapping(
            render["world_type_probabilities"], "world probabilities"
        )
        _exact_keys(
            probabilities,
            {"pure_normal", "control_only", "mixed", "anomaly_only"},
            "world probabilities",
        )
        total = sum(
            _number(item, "world probability") for item in probabilities.values()
        )
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ProtocolError("world type probabilities must sum to one")
        common = _mapping(render["common_entity_rules"], "common entity rules")
        for name in (
            "static_world_pose",
            "support_plane_required",
            "observed_surface_collision_rejection",
            "inserted_entity_collision_rejection",
        ):
            _expect(common.get(name), True, f"common_entity_rules.{name}")
        sensor = _mapping(render["sensor_model"], "sensor model")
        for name in (
            "return_probability",
            "intensity_quantiles",
            "nearest_accepted_return",
            "allow_new_return_on_empty_ray",
            "bidirectional_occlusion",
        ):
            _expect(sensor.get(name), True, f"sensor_model.{name}")

        descriptors = _mapping(render["window_descriptors"], "window descriptors")
        _exact_keys(
            descriptors,
            {
                "density_voxel_size_m",
                "density_coordinate_system",
                "density_voxel_quantization",
                "definitions",
                "required",
            },
            "window descriptors",
        )
        if _number(descriptors["density_voxel_size_m"], "density voxel size") <= 0:
            raise ProtocolError("density voxel size must be positive")
        _expect(
            descriptors["density_coordinate_system"],
            "symmetric_window_coordinates",
            "density coordinate system",
        )
        required = _string_tuple(descriptors["required"], "window descriptor names")
        expected_required = (
            "object_id",
            "label",
            "visible_returns_by_scan",
            "spatial_voxels_by_scan",
            "joint_visible_return_count",
            "joint_spatial_voxel_count",
            "maximum_single_scan_spatial_voxel_count",
            "densification_gain",
            "duplicate_fraction",
            "median_distance_m",
            "occlusion_rate",
            "support_semantic_id",
            "visible_scan_count",
            "minimum_visible_return_height_m",
            "intensity_q05_median_q95",
            "beam_histogram",
        )
        if required != expected_required:
            raise ProtocolError("window descriptor identities changed")
        definitions = _mapping(descriptors["definitions"], "descriptor definitions")
        if set(definitions) != {
            "joint_visible_return_count",
            "joint_spatial_voxel_count",
            "maximum_single_scan_spatial_voxel_count",
            "densification_gain",
            "duplicate_fraction",
            "empty_entity_rule",
        }:
            raise ProtocolError("density descriptor definitions are incomplete")
        matching = _mapping(render["proxy_control_matching"], "proxy/control matching")
        _exact_keys(
            matching,
            {"unit", "required_covariates", "thresholds"},
            "proxy/control matching",
        )
        _expect(matching.get("unit"), "complete_five_scan_window", "matching unit")
        _r02_thresholds(matching.get("thresholds"))
        covariates = set(
            _string_tuple(matching.get("required_covariates"), "matching covariates")
        )
        expected_covariates = {
            "joint_visible_return_count",
            "joint_spatial_voxel_count",
            "densification_gain",
            "median_distance_m",
            "occlusion_rate",
            "support_semantic_id",
            "visible_scan_count",
            "minimum_visible_return_height_m",
        }
        if covariates != expected_covariates:
            raise ProtocolError("proxy/control matching covariates are invalid")

    @staticmethod
    def _validate_stu(stu: Mapping[str, object]) -> None:
        expected = {
            "source": "STU_official_Mask4Former3D",
            "checkpoint_bytes": 476261075,
            "voxel_size_m": 0.05,
            "point_feature_dim": 128,
            "normal_evidence_dim": 19,
            "assignment_reliability_dim": 1,
            "no_object_reliability_dim": 1,
            "b0_score": "official_STU_MaxLogit",
            "frozen": True,
        }
        _exact_keys(
            stu,
            {
                *expected,
                "checkpoint",
                "checkpoint_sha256",
                "model_state_tensor_sha256",
                "source_manifest_sha256",
                "repository",
            },
            "stu",
        )
        for name, value in expected.items():
            _expect(stu[name], value, f"stu.{name}")
        _nonempty_string(stu["checkpoint"], "stu.checkpoint")
        _nonempty_string(stu["repository"], "stu.repository")
        digest = _nonempty_string(stu["checkpoint_sha256"], "stu.checkpoint_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProtocolError("stu.checkpoint_sha256 must be lowercase SHA-256")
        _expect(
            stu["model_state_tensor_sha256"],
            "0be4805592a3d064b21655c6c6eeeb7227322c9670873345be52747b0a24d1fb",
            "stu.model_state_tensor_sha256",
        )
        _expect(
            stu["source_manifest_sha256"],
            "f0cead4f5e721262f9f1c26231d116406bb4fb0a43139f22e3706be89b914891",
            "stu.source_manifest_sha256",
        )

    @staticmethod
    def _validate_model(model: Mapping[str, object]) -> None:
        keys = {
            "name",
            "input_features",
            "spatial_coordinates",
            "bookkeeping_inputs",
            "input_dim",
            "forbidden_features",
            "hidden_dim",
            "heads",
            "levels",
            "voxel_sizes_m",
            "radius_neighbors",
            "voxel_feature",
            "neighborhood_feature",
            "upsample_neighbors",
            "grouping_modes",
            "B2_B3_shared_class_and_parameterization",
            "output",
            "scan_permutation_equivariant",
        }
        _exact_keys(model, keys, "model")
        expected = {
            "name": "JointWindowPointTransformer",
            "input_features": [
                "stu_point_feature_128d",
                "normal_evidence_19d",
                "assignment_reliability",
                "no_object_reliability",
                "intensity",
            ],
            "spatial_coordinates": "symmetric_window_coordinates_may_enter_order_invariant_position_encoding_relative_displacement_voxel_neighborhood_and_decode_geometry",
            "bookkeeping_inputs": "point_identity_for_restoration_and_scan_group_for_B2_isolation_only",
            "input_dim": 150,
            "forbidden_features": [
                "source_frame",
                "window_member_index",
                "relative_time",
                "absolute_time",
                "time_embedding",
                "reversible_time_encoding",
            ],
            "hidden_dim": 128,
            "heads": 4,
            "levels": 4,
            "voxel_sizes_m": [0.1, 0.2, 0.4],
            "radius_neighbors": {
                "radii_m": [0.25, 0.5, 1.0, 2.0],
                "maximum_neighbors": [12, 16, 24, 32],
            },
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
        _exact_keys(
            experiments, {item.value for item in ExperimentCondition}, "experiments"
        )
        definitions = {
            "B0": "frozen_STU_official_single_scan_MaxLogit",
            "B1": "five_independent_single_scan_forwards_per_WindowWorld_with_one_all_point_loss",
            "B2": "all_five_scans_input_output_and_supervised_with_voxel_neighborhood_and_decode_isolated_by_scan_group",
            "B3": "all_five_scans_jointly_voxelized_neighbored_and_decoded_in_symmetric_window_coordinates",
        }
        for condition in ExperimentCondition:
            record = _mapping(
                experiments[condition.value], f"experiments.{condition.value}"
            )
            keys = {"trainable", "definition"}
            if condition.trainable:
                keys.add("grouping_mode")
            _exact_keys(record, keys, f"experiments.{condition.value}")
            _expect(
                record["trainable"], condition.trainable, f"{condition.value}.trainable"
            )
            _expect(
                record["definition"],
                definitions[condition.value],
                f"{condition.value}.definition",
            )
            if condition.trainable:
                _expect(
                    record["grouping_mode"],
                    condition.grouping_mode.value,
                    f"{condition.value}.grouping_mode",
                )
                if condition.output_local_indices != WINDOW_MEMBER_OFFSETS:
                    raise ProtocolError(
                        f"{condition.value} must output all five members"
                    )

    @staticmethod
    def _validate_training(training: Mapping[str, object]) -> None:
        keys = {
            "source_partition",
            "source_sequence_id",
            "bank",
            "micro_batch",
            "effective_batch",
            "epoch",
            "loss",
            "forced_partial_step_at_world_boundary",
            "modes",
            "tiny_overfit",
            "pilot",
            "formal",
            "checkpoint_selection",
            "deterministic_algorithms",
        }
        _exact_keys(training, keys, "training")
        expected = {
            "source_partition": "train",
            "source_sequence_id": 206,
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
        _exact_keys(
            bank,
            {
                "name",
                "unit",
                "shared_by",
                "required_identity",
                "condition_invariants",
                "generation_plan",
            },
            "training.bank",
        )
        expected_bank = {
            "name": "window_train_bank",
            "unit": "WindowWorld",
            "shared_by": ["B1", "B2", "B3"],
            "required_identity": [
                "five_source_frames",
                "WorldSpec",
                "five_rendered_point_arrays",
                "renderer_identity",
                "STU_identity",
                "point_identity",
                "labels",
                "window_level_density_descriptors",
            ],
            "condition_invariants": [
                "same_five_aligned_points",
                "same_point_labels",
                "same_entry_order",
                "same_model_parameter_shapes_and_initialization",
                "same_loss",
                "same_optimizer",
            ],
        }
        for name, value in expected_bank.items():
            _expect(bank[name], value, f"training.bank.{name}")
        _expect(
            bank["generation_plan"],
            {
                "namespace": "ajae-schema31-window-train-bank-plan-v1",
                "formal_entry_count": 445,
                "allowed_prefix_counts": [8, 128, 445],
                "window_start_order": "ascending_sha256_namespace_and_window_start",
                "world_type_cycle": [
                    "pure_normal",
                    "control_only",
                    "mixed",
                    "mixed",
                    "anomaly_only",
                ],
                "world_seed_rule": "first_63_bits_of_sha256_namespace_position_and_window_start",
                "observation_attempt_selection": "first_success_in_ascending_retry_index_no_manual_selection",
                "observation_attempt_stride": 10_000_019,
                "maximum_observation_attempts": 48,
            },
            "training.bank.generation_plan",
        )
        _expect(
            training["loss"],
            {
                "name": "effective_batch_empty_class_safe_balanced_binary_cross_entropy",
                "aggregation": "first_sum_positive_and_negative_unreduced_losses_and_counts_over_the_entire_effective_batch_then_average_each_present_class",
                "two_class_formula": "0.5*positive_loss_sum/positive_count+0.5*negative_loss_sum/negative_count",
                "one_class_rule": "mean_loss_of_the_single_present_class",
            },
            "training.loss",
        )
        _expect(
            training["tiny_overfit"],
            {
                "windows": 8,
                "seed": 2001,
                "maximum_updates": 500,
                "config": {
                    "learning_rate": 0.0001,
                    "gradient_accumulation": 1,
                    "weight_decay": 0.0,
                    "scheduler": "constant",
                    "epochs": 1,
                    "maximum_updates": 500,
                    "evaluation_interval_updates": 500,
                    "gradient_clip_norm": None,
                },
                "pass_any": {
                    "training_AP_minimum_percent": 99.0,
                    "loss_strictly_below": 0.02,
                },
            },
            "training.tiny_overfit",
        )
        pilot = _mapping(training["pilot"], "training.pilot")
        fixed_pilot = {
            "windows": 128,
            "condition": "B3",
            "seeds": [1001, 1002],
            "stage_order": ["learning_rate_and_batch", "scheduler", "weight_decay"],
            "learning_rates": [0.00003, 0.0001, 0.0003],
            "gradient_accumulation": [1, 2],
            "screen_updates": [50, 200, 600],
            "schedulers_after_learning_rate_selection": [
                "constant",
                "five_percent_warmup_cosine",
            ],
            "weight_decay_after_scheduler_selection": [0.0, 0.00001, 0.0001],
            "fixed_run_parameters": {
                "epochs": 1,
                "evaluation_interval_updates": 50,
                "gradient_clip_norm": None,
            },
            "selection": {
                "metric": "best_development_macro_fused_point_ap",
                "score_comparison": "strictly_greater",
                "exact_tie_break": "protocol_grid_order",
                "probability_saturation_epsilon": 0.000001,
                "maximum_complete_saturation_fraction_exclusive": 1.0,
                "screen_50_rule": "all_six_candidates_must_produce_finite_auditable_runs_then_complete_probability_saturation_is_excluded_before_screen_200",
                "screen_50_minimum_survivors": 2,
                "screen_200_finalist_count": 2,
                "screen_600_seed": 1002,
            },
        }
        _exact_keys(pilot, {*fixed_pilot, "frozen_stage_winners"}, "training.pilot")
        for name, value in fixed_pilot.items():
            _expect(pilot[name], value, f"training.pilot.{name}")
        winners = _mapping(
            pilot["frozen_stage_winners"], "training.pilot.frozen_stage_winners"
        )
        _exact_keys(
            winners,
            {"learning_rate", "gradient_accumulation", "scheduler", "weight_decay"},
            "training.pilot.frozen_stage_winners",
        )
        learning_rate = winners["learning_rate"]
        accumulation = winners["gradient_accumulation"]
        scheduler = winners["scheduler"]
        weight_decay = winners["weight_decay"]
        if learning_rate is None:
            if any(
                value is not None for value in (accumulation, scheduler, weight_decay)
            ):
                raise ProtocolError(
                    "pilot stage winners must be frozen in their declared order"
                )
        else:
            if _number(learning_rate, "pilot selected learning rate") not in (
                0.00003,
                0.0001,
                0.0003,
            ) or _integer(
                accumulation, "pilot selected accumulation", minimum=1
            ) not in (1, 2):
                raise ProtocolError(
                    "pilot learning-rate/batch winner is outside the grid"
                )
            if scheduler is not None and scheduler not in {
                "constant",
                "five_percent_warmup_cosine",
            }:
                raise ProtocolError("pilot scheduler winner is outside the grid")
            if weight_decay is not None and (
                scheduler is None
                or _number(weight_decay, "pilot selected weight decay")
                not in (0.0, 0.00001, 0.0001)
            ):
                raise ProtocolError(
                    "pilot weight-decay winner requires a frozen scheduler"
                )
        formal = _mapping(training["formal"], "training.formal")
        _exact_keys(
            formal,
            {
                "seeds",
                "deployment_seed",
                "deployment_condition",
                "recipe_status",
                "recipe",
                "bank_identity",
                "development_population_identity",
                "allowed_only_after",
            },
            "training.formal",
        )
        _expect(formal["seeds"], [0, 1, 2], "training.formal.seeds")
        _expect(formal["deployment_seed"], 0, "training.formal.deployment_seed")
        _expect(
            formal["deployment_condition"],
            "B3",
            "training.formal.deployment_condition",
        )
        if formal["recipe_status"] not in {
            "pending_result_blind_R05_freeze",
            "frozen_result_blind_in_R05",
        }:
            raise ProtocolError("training.formal.recipe_status is not recognized")
        if formal["recipe_status"] == "pending_result_blind_R05_freeze":
            if any(
                formal[name] is not None
                for name in (
                    "recipe",
                    "bank_identity",
                    "development_population_identity",
                )
            ):
                raise ProtocolError(
                    "pending R05 cannot carry a formal recipe or data identity"
                )
        else:
            recipe = _mapping(formal["recipe"], "training.formal.recipe")
            _exact_keys(
                recipe,
                {
                    "learning_rate",
                    "gradient_accumulation",
                    "weight_decay",
                    "scheduler",
                    "epochs",
                    "maximum_updates",
                    "evaluation_interval_updates",
                    "gradient_clip_norm",
                },
                "training.formal.recipe",
            )
            if _number(recipe["learning_rate"], "formal learning rate") <= 0.0:
                raise ProtocolError("formal learning rate must be positive")
            _integer(recipe["gradient_accumulation"], "formal accumulation", minimum=1)
            if _number(recipe["weight_decay"], "formal weight decay") < 0.0:
                raise ProtocolError("formal weight decay cannot be negative")
            if recipe["scheduler"] not in {"constant", "five_percent_warmup_cosine"}:
                raise ProtocolError("formal scheduler is unsupported")
            _integer(recipe["epochs"], "formal epochs", minimum=1)
            if recipe["maximum_updates"] is not None:
                _integer(recipe["maximum_updates"], "formal maximum updates", minimum=1)
            _integer(
                recipe["evaluation_interval_updates"],
                "formal evaluation interval",
                minimum=1,
            )
            if (
                recipe["gradient_clip_norm"] is not None
                and _number(recipe["gradient_clip_norm"], "formal gradient clip") <= 0.0
            ):
                raise ProtocolError("formal gradient clip must be positive")
            if any(value is None for value in winners.values()) or any(
                recipe[name] != winners[name]
                for name in (
                    "learning_rate",
                    "gradient_accumulation",
                    "scheduler",
                    "weight_decay",
                )
            ):
                raise ProtocolError(
                    "formal recipe must preserve every frozen pilot-stage winner"
                )
            _sha256(formal["bank_identity"], "formal bank identity")
            _sha256(
                formal["development_population_identity"],
                "formal development population identity",
            )
        _expect(
            formal["allowed_only_after"], "R05", "training.formal.allowed_only_after"
        )
        selection = _mapping(training["checkpoint_selection"], "checkpoint selection")
        _exact_keys(
            selection,
            {
                "metric",
                "eligible_mechanism",
                "eligible_clip_count",
                "held_out_torus_use",
                "fusion_scope",
                "score_comparison",
                "exact_tie_break",
                "status",
            },
            "checkpoint selection",
        )
        _expect(
            selection["metric"],
            "macro_mean_of_per_in_generator_DevelopmentClipWorld_all_occurrence_fused_point_AP",
            "checkpoint metric",
        )
        _expect(
            selection["eligible_mechanism"],
            "in_generator",
            "checkpoint eligible mechanism",
        )
        _expect(selection["eligible_clip_count"], 24, "checkpoint eligible clip count")
        _expect(
            selection["held_out_torus_use"],
            "unopened_until_S01_after_all_training_before_M01_no_method_changes_after_opening",
            "held-out torus use",
        )
        _expect(
            selection["fusion_scope"],
            "within_one_frozen_world_identity_only",
            "checkpoint fusion scope",
        )
        _expect(
            selection["score_comparison"],
            "strictly_greater",
            "checkpoint score comparison",
        )
        _expect(
            selection["exact_tie_break"],
            "earliest_evaluated_update",
            "checkpoint exact tie break",
        )
        _expect(
            selection["status"],
            "metric_population_and_tie_break_frozen_result_blind",
            "checkpoint selection status",
        )

    @staticmethod
    def _validate_evaluation(evaluation: Mapping[str, object]) -> None:
        expected = {
            "legal_window_starts": "every_start_with_five_real_consecutive_nonexcluded_frames",
            "model_output_members": list(WINDOW_MEMBER_OFFSETS),
            "real_sequence_fusion_key": [
                "partition",
                "sequence_id",
                "source_frame",
                "source_ray",
            ],
            "synthetic_fusion_key": ["world_identity", "source_frame", "source_ray"],
            "fusion_values": {
                "B0": {
                    "input": "raw_finite_frozen_STU_official_MaxLogit_score",
                    "per_occurrence_transform": "none",
                    "repeated_window_mean": "identity_preserving",
                },
                "B1_B2_B3": {
                    "input": "anomaly_logit",
                    "per_occurrence_transform": "sigmoid",
                },
            },
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
        _exact_keys(
            evaluation,
            {*expected, "synthetic_development", "held_out_synthetic_shift"},
            "evaluation",
        )
        for name, value in expected.items():
            _expect(evaluation[name], value, f"evaluation.{name}")
        _expect(
            evaluation["synthetic_development"],
            {
                "unit": "DevelopmentClipWorld",
                "world_rule": "one_WorldSpec_and_random_identity_are_frozen_before_rendering_every_scan_used_by_all_overlapping_windows_in_the_clip",
                "minimum_frames_to_expose_all_occurrence_strata": 9,
                "exact_clip_length_and_count": {
                    "frames_per_clip": 9,
                    "overlapping_windows_per_clip": 5,
                    "in_generator_clips": 24,
                    "total_clips": 24,
                    "freeze_rule": "fixed_before_any_schema31_development_world_is_generated_or_scored",
                },
                "generation_plan": {
                    "namespace": "ajae-schema31-development-clip-plan-v1",
                    "clip_start_selection": "sha256_ranked_greedy_nonoverlapping_nine_frame_clips",
                    "plan_positions_inclusive": [0, 23],
                    "mechanism": "in_generator",
                    "world_seed_rule": "first_63_bits_of_sha256_namespace_position_and_clip_start",
                    "observation_attempt_selection": "first_success_in_ascending_retry_index_no_manual_selection",
                    "observation_attempt_stride": 10_000_019,
                    "maximum_observation_attempts": 48,
                },
                "cross_world_fusion_forbidden": True,
            },
            "evaluation.synthetic_development",
        )
        _expect(
            evaluation["held_out_synthetic_shift"],
            {
                "unit": "DevelopmentClipWorld",
                "mechanism": "torus_SDF",
                "clip_count": 6,
                "frames_per_clip": 9,
                "overlapping_windows_per_clip": 5,
                "generation_plan_namespace": "ajae-schema31-development-clip-plan-v1",
                "plan_positions_inclusive": [24, 29],
                "allowed_only_at_or_after": "S01",
                "excluded_before_S01_from": [
                    "rendering",
                    "visual_review",
                    "proxy_qualification",
                    "hyperparameter_selection",
                    "checkpoint_selection",
                    "method_changes",
                ],
            },
            "evaluation.held_out_synthetic_shift",
        )

    @staticmethod
    def _validate_status(source: Mapping[str, object]) -> None:
        status = _mapping(source["status"], "status")
        _exact_keys(
            status,
            {
                "implementation_target",
                "current_node",
                "scientific_route",
                "r02_population_identity",
                "r02_verdict_identity",
                "r03_qualification_identity",
                "r04_training_bank_identity",
                "r04_training_qualification_identity",
                "r05_freeze_identity",
                "g2_verdict_identity",
                "g3_verdict_identity",
                "s01_shift_audit_identity",
                "m01_method_freeze_identity",
                "v01_verdict_identity",
                "formal_training_allowed",
                "performance_claims_available",
            },
            "status",
        )
        _expect(
            status["implementation_target"], "schema31", "status.implementation_target"
        )
        _expect(
            status["scientific_route"],
            "window_proxy_and_joint_five_scan_dense_point_cloud",
            "status.scientific_route",
        )
        node = _nonempty_string(status["current_node"], "status.current_node")
        if node not in _STATE_MACHINE:
            raise ProtocolError(
                "status.current_node is outside the schema-31 state machine"
            )
        thresholds = _r02_thresholds(
            _mapping(source["render"], "render")["proxy_control_matching"]["thresholds"]
        )
        if thresholds is None and _STATE_MACHINE.index(node) > _STATE_MACHINE.index(
            "R02"
        ):
            raise ProtocolError(
                "R02 thresholds must be frozen before advancing past R02"
            )
        r02_population = status["r02_population_identity"]
        r02_verdict = status["r02_verdict_identity"]
        if _STATE_MACHINE.index(node) <= _STATE_MACHINE.index("R02"):
            if r02_population is not None or r02_verdict is not None:
                raise ProtocolError("R02 evidence cannot be claimed before leaving R02")
        else:
            _sha256(r02_population, "status.r02_population_identity")
            _sha256(r02_verdict, "status.r02_verdict_identity")
        bank_identity = status["r04_training_bank_identity"]
        if _STATE_MACHINE.index(node) < _STATE_MACHINE.index("R04"):
            if bank_identity is not None:
                raise ProtocolError(
                    "R04 training bank cannot be frozen before entering R04"
                )
        else:
            _sha256(bank_identity, "status.r04_training_bank_identity")
        evidence_milestones = (
            ("r03_qualification_identity", "R03"),
            ("r04_training_qualification_identity", "R04"),
            ("r05_freeze_identity", "R05"),
            ("g2_verdict_identity", "G2"),
            ("g3_verdict_identity", "G3"),
            ("s01_shift_audit_identity", "S01"),
            ("m01_method_freeze_identity", "M01"),
            ("v01_verdict_identity", "V01"),
        )
        for field, milestone in evidence_milestones:
            value = status[field]
            if _STATE_MACHINE.index(node) <= _STATE_MACHINE.index(milestone):
                if value is not None:
                    raise ProtocolError(
                        f"{field} cannot be claimed before leaving {milestone}"
                    )
            else:
                _sha256(value, f"status.{field}")
        pilot = _mapping(
            _mapping(source["training"], "training")["pilot"], "training.pilot"
        )
        winners = _mapping(
            pilot["frozen_stage_winners"], "training.pilot.frozen_stage_winners"
        )
        winner_values = tuple(winners.values())
        if _STATE_MACHINE.index(node) < _STATE_MACHINE.index("R04") and any(
            value is not None for value in winner_values
        ):
            raise ProtocolError("pilot winners cannot be frozen before R04")
        if _STATE_MACHINE.index(node) >= _STATE_MACHINE.index("R05") and any(
            value is None for value in winner_values
        ):
            raise ProtocolError("all pilot-stage winners must be frozen before R05")
        formal_allowed = _boolean(
            status["formal_training_allowed"], "formal training status"
        )
        claims_available = _boolean(
            status["performance_claims_available"], "performance claim status"
        )
        formal = _mapping(
            _mapping(source["training"], "training")["formal"], "training.formal"
        )
        frozen_recipe = formal["recipe_status"] == "frozen_result_blind_in_R05"
        if frozen_recipe and formal["bank_identity"] != bank_identity:
            raise ProtocolError(
                "formal recipe must preserve the R04 training bank identity"
            )
        if (
            frozen_recipe
            and formal["development_population_identity"] != r02_population
        ):
            raise ProtocolError(
                "formal recipe must preserve the qualified R02 population identity"
            )
        expected_formal_allowed = node in {"G2", "G3"}
        if formal_allowed != expected_formal_allowed:
            raise ProtocolError("formal training is enabled only during G2 and G3")
        if _STATE_MACHINE.index(node) > _STATE_MACHINE.index("R05") and (
            not frozen_recipe
        ):
            raise ProtocolError(
                "nodes after R05 require a preserved frozen formal recipe"
            )
        if claims_available and _STATE_MACHINE.index(node) <= _STATE_MACHINE.index(
            "V01"
        ):
            raise ProtocolError("performance claims cannot exist before V01")
        if (
            _STATE_MACHINE.index(node) > _STATE_MACHINE.index("V01")
            and not claims_available
        ):
            raise ProtocolError(
                "nodes after V01 require a supported public-validation claim"
            )

        gates = _mapping(source["decision_gates"], "decision gates")
        _exact_keys(gates, {"G2", "G3", "V01"}, "decision_gates")
        for name, comparisons in (
            ("G2", {"B1_vs_B0"}),
            ("G3", {"B3_vs_B1", "B3_vs_B2"}),
        ):
            gate = _mapping(gates[name], f"decision_gates.{name}")
            _exact_keys(
                gate,
                {
                    "comparisons",
                    "formal_seeds",
                    "criteria_status",
                    "criteria",
                    "criteria_identity",
                },
                f"decision_gates.{name}",
            )
            if (
                set(_string_tuple(gate["comparisons"], f"{name}.comparisons"))
                != comparisons
            ):
                raise ProtocolError(f"decision_gates.{name}.comparisons changed")
            _expect(gate["formal_seeds"], [0, 1, 2], f"{name}.formal_seeds")
            criteria_status = _nonempty_string(
                gate["criteria_status"], f"{name}.criteria_status"
            )
            if criteria_status == "pending_result_blind_R05_freeze":
                if (
                    gate["criteria"] is not None
                    or gate["criteria_identity"] is not None
                ):
                    raise ProtocolError(
                        f"pending decision_gates.{name} cannot carry criteria"
                    )
            elif criteria_status == "frozen_result_blind_in_R05":
                criteria = _mapping(gate["criteria"], f"decision_gates.{name}.criteria")
                _exact_keys(
                    criteria,
                    {
                        "format",
                        "decision_metric_scale",
                        "primary_metric",
                        "difference_direction",
                        "statistical_unit",
                        "confidence_interval",
                        "comparison_thresholds",
                        "normal_safety",
                        "decision_rule",
                    },
                    f"decision_gates.{name}.criteria",
                )
                _expect(
                    criteria["format"],
                    "ajae-schema31-formal-gate-criteria-v1",
                    f"{name} criteria format",
                )
                _expect(
                    criteria["decision_metric_scale"],
                    "unit_interval",
                    f"{name} decision metric scale",
                )
                _expect(
                    criteria["primary_metric"],
                    "macro_fused_point_ap",
                    f"{name} primary metric",
                )
                _expect(
                    criteria["difference_direction"],
                    "left_condition_minus_right_condition",
                    f"{name} difference direction",
                )
                _expect(
                    criteria["statistical_unit"],
                    "paired_formal_seed",
                    f"{name} statistical unit",
                )
                interval = _mapping(
                    criteria["confidence_interval"], f"{name} confidence interval"
                )
                _exact_keys(
                    interval,
                    {"method", "confidence_level"},
                    f"{name} confidence interval",
                )
                _expect(
                    interval["method"],
                    "paired_two_sided_student_t_v1",
                    f"{name} interval method",
                )
                confidence = _number(
                    interval["confidence_level"], f"{name} confidence level"
                )
                if not 0.0 < confidence < 1.0:
                    raise ProtocolError(f"{name} confidence level must lie in (0,1)")
                thresholds = _mapping(
                    criteria["comparison_thresholds"],
                    f"{name} comparison thresholds",
                )
                _exact_keys(thresholds, comparisons, f"{name} comparison thresholds")
                for comparison in comparisons:
                    threshold = _mapping(
                        thresholds[comparison], f"{name}.{comparison} threshold"
                    )
                    _exact_keys(
                        threshold,
                        {
                            "minimum_mean_difference",
                            "minimum_confidence_interval_lower_bound",
                            "minimum_positive_seed_count",
                        },
                        f"{name}.{comparison} threshold",
                    )
                    mean_difference = _number(
                        threshold["minimum_mean_difference"],
                        f"{name}.{comparison} minimum mean difference",
                    )
                    lower_bound = _number(
                        threshold["minimum_confidence_interval_lower_bound"],
                        f"{name}.{comparison} minimum confidence lower bound",
                    )
                    if (
                        not -1.0 <= mean_difference <= 1.0
                        or not -1.0 <= lower_bound <= 1.0
                    ):
                        raise ProtocolError(
                            f"{name}.{comparison} thresholds must lie in [-1,1]"
                        )
                    positive_seeds = _integer(
                        threshold["minimum_positive_seed_count"],
                        f"{name}.{comparison} minimum positive seeds",
                        minimum=1,
                    )
                    if positive_seeds > 3:
                        raise ProtocolError(
                            f"{name}.{comparison} positive seed count exceeds three"
                        )
                safety = _mapping(criteria["normal_safety"], f"{name} normal safety")
                _exact_keys(
                    safety,
                    {
                        "evaluation_set_identity",
                        "threshold_rule_identity",
                        "statistics",
                        "maximum_allowed_signed_worsening",
                    },
                    f"{name} normal safety",
                )
                _sha256(
                    safety["evaluation_set_identity"],
                    f"{name} normal-safety set identity",
                )
                _sha256(
                    safety["threshold_rule_identity"],
                    f"{name} normal-safety threshold-rule identity",
                )
                statistics = _string_tuple(
                    safety["statistics"], f"{name} normal-safety statistics"
                )
                if statistics != (
                    "normal_false_positive_rate_at_frozen_threshold_unit_interval",
                ):
                    raise ProtocolError(
                        f"{name} normal-safety statistic or scale changed"
                    )
                worsening = _number(
                    safety["maximum_allowed_signed_worsening"],
                    f"{name} maximum normal-safety worsening",
                )
                if not 0.0 <= worsening <= 1.0:
                    raise ProtocolError(
                        f"{name} normal-safety worsening must lie in [0,1]"
                    )
                _expect(
                    criteria["decision_rule"],
                    "all_comparisons_and_normal_safety_must_pass",
                    f"{name} decision rule",
                )
                _expect(
                    gate["criteria_identity"],
                    _sha256_json(criteria),
                    f"{name} criteria identity",
                )
            else:
                raise ProtocolError(f"decision_gates.{name}.criteria_status is invalid")
            if _STATE_MACHINE.index(node) > _STATE_MACHINE.index("R05") and (
                criteria_status != "frozen_result_blind_in_R05"
            ):
                raise ProtocolError(
                    f"decision_gates.{name} criteria must be frozen before formal runs"
                )
        v01 = _mapping(gates["V01"], "decision_gates.V01")
        _exact_keys(
            v01,
            {
                "public_sequences",
                "one_time_only",
                "comparisons",
                "criteria_status",
                "criteria",
                "criteria_identity",
            },
            "decision_gates.V01",
        )
        _expect(
            v01["public_sequences"], len(PUBLIC_ANOMALY_IDS), "V01.public_sequences"
        )
        _expect(v01["one_time_only"], True, "V01.one_time_only")
        v01_comparisons = {
            "frozen_final_vs_B0",
            "frozen_final_vs_B1",
        }
        if set(_string_tuple(v01["comparisons"], "V01.comparisons")) != v01_comparisons:
            raise ProtocolError("decision_gates.V01.comparisons changed")
        v01_status = _nonempty_string(v01["criteria_status"], "V01.criteria_status")
        if v01_status == "pending_result_blind_R05_freeze":
            if v01["criteria"] is not None or v01["criteria_identity"] is not None:
                raise ProtocolError("pending V01 cannot carry criteria")
        elif v01_status == "frozen_result_blind_in_R05":
            criteria = _mapping(v01["criteria"], "decision_gates.V01.criteria")
            _exact_keys(
                criteria,
                {
                    "format",
                    "decision_metric_scale",
                    "primary_metric",
                    "difference_direction",
                    "statistical_unit",
                    "confidence_interval",
                    "comparison_thresholds",
                    "normal_safety",
                    "failure_action",
                },
                "decision_gates.V01.criteria",
            )
            _expect(
                criteria["format"],
                "ajae-schema31-v01-transfer-criteria-v1",
                "V01 criteria format",
            )
            _expect(
                criteria["decision_metric_scale"],
                "unit_interval",
                "V01 decision metric scale",
            )
            _expect(
                criteria["primary_metric"],
                "per_sequence_AP_unit_interval",
                "V01 primary metric",
            )
            _expect(
                criteria["difference_direction"],
                "left_method_minus_right_method",
                "V01 difference direction",
            )
            _expect(
                criteria["statistical_unit"],
                "real_anomaly_sequence",
                "V01 statistical unit",
            )
            interval = _mapping(
                criteria["confidence_interval"], "V01 confidence interval"
            )
            _exact_keys(
                interval,
                {"method", "confidence_level"},
                "V01 confidence interval",
            )
            _expect(
                interval["method"],
                "paired_two_sided_student_t_v1",
                "V01 interval method",
            )
            confidence = _number(interval["confidence_level"], "V01 confidence level")
            if not 0.0 < confidence < 1.0:
                raise ProtocolError("V01 confidence level must lie in (0,1)")
            thresholds = _mapping(
                criteria["comparison_thresholds"], "V01 comparison thresholds"
            )
            _exact_keys(thresholds, v01_comparisons, "V01 comparison thresholds")
            for comparison in v01_comparisons:
                threshold = _mapping(
                    thresholds[comparison], f"V01.{comparison} threshold"
                )
                _exact_keys(
                    threshold,
                    {
                        "minimum_mean_AP_difference",
                        "minimum_confidence_interval_lower_bound",
                        "minimum_positive_sequence_count",
                    },
                    f"V01.{comparison} threshold",
                )
                mean_difference = _number(
                    threshold["minimum_mean_AP_difference"],
                    f"V01.{comparison} minimum mean AP difference",
                )
                lower_bound = _number(
                    threshold["minimum_confidence_interval_lower_bound"],
                    f"V01.{comparison} minimum confidence lower bound",
                )
                if not -1.0 <= mean_difference <= 1.0 or not -1.0 <= lower_bound <= 1.0:
                    raise ProtocolError(
                        f"V01.{comparison} thresholds must lie in [-1,1]"
                    )
                positive_sequences = _integer(
                    threshold["minimum_positive_sequence_count"],
                    f"V01.{comparison} minimum positive sequences",
                    minimum=1,
                )
                if positive_sequences > len(PUBLIC_ANOMALY_IDS):
                    raise ProtocolError(
                        f"V01.{comparison} positive sequence count exceeds the population"
                    )
            safety = _mapping(criteria["normal_safety"], "V01 normal safety")
            _exact_keys(
                safety,
                {
                    "evaluation_set_identity",
                    "threshold_rule_identity",
                    "statistics",
                    "maximum_allowed_signed_worsening",
                },
                "V01 normal safety",
            )
            _sha256(
                safety["evaluation_set_identity"],
                "V01 normal-safety set identity",
            )
            _sha256(
                safety["threshold_rule_identity"],
                "V01 normal-safety threshold-rule identity",
            )
            statistics = _string_tuple(
                safety["statistics"], "V01 normal-safety statistics"
            )
            if statistics != (
                "normal_false_positive_rate_at_frozen_threshold_unit_interval",
            ):
                raise ProtocolError("V01 normal-safety statistic or scale changed")
            worsening = _number(
                safety["maximum_allowed_signed_worsening"],
                "V01 maximum normal-safety worsening",
            )
            if not 0.0 <= worsening <= 1.0:
                raise ProtocolError("V01 normal-safety worsening must lie in [0,1]")
            _expect(
                criteria["failure_action"],
                "stop_current_research_cycle_without_tuning_on_opened_sequences",
                "V01 failure action",
            )
            _expect(
                v01["criteria_identity"],
                _sha256_json(criteria),
                "V01 criteria identity",
            )
        else:
            raise ProtocolError("decision_gates.V01.criteria_status is invalid")
        if _STATE_MACHINE.index(node) > _STATE_MACHINE.index("R05") and (
            v01_status != "frozen_result_blind_in_R05"
        ):
            raise ProtocolError("V01 criteria must be frozen in R05 before formal runs")

        if _STATE_MACHINE.index(node) > _STATE_MACHINE.index("R05"):
            freeze_payload = {
                "format": "ajae-schema31-r05-freeze-v1",
                "r04_training_qualification_identity": status[
                    "r04_training_qualification_identity"
                ],
                "formal": _plain(formal),
                "checkpoint_selection": _plain(
                    _mapping(source["training"], "training")["checkpoint_selection"]
                ),
                "G2_criteria_identity": gates["G2"]["criteria_identity"],
                "G3_criteria_identity": gates["G3"]["criteria_identity"],
                "V01_criteria_identity": gates["V01"]["criteria_identity"],
            }
            _expect(
                status["r05_freeze_identity"],
                _sha256_json(freeze_payload),
                "status.r05_freeze_identity",
            )


def _validate_development_descriptors(
    descriptors: Mapping[str, object], required: tuple[str, ...], name: str
) -> None:
    _exact_keys(descriptors, set(required), name)
    _integer(descriptors["object_id"], f"{name}.object_id", minimum=1)
    label = _nonempty_string(descriptors["label"], f"{name}.label")
    if label not in {"normal-control", "anomaly-proxy"}:
        raise ProtocolError(f"{name}.label is invalid")
    per_scan_returns = _int_tuple(
        descriptors["visible_returns_by_scan"], f"{name}.visible_returns_by_scan"
    )
    per_scan_voxels = _int_tuple(
        descriptors["spatial_voxels_by_scan"], f"{name}.spatial_voxels_by_scan"
    )
    if len(per_scan_returns) != WINDOW_FRAMES or len(per_scan_voxels) != WINDOW_FRAMES:
        raise ProtocolError(f"{name} requires five per-scan counts")
    returns = _integer(
        descriptors["joint_visible_return_count"],
        f"{name}.joint_visible_return_count",
        minimum=1,
    )
    joint = _integer(
        descriptors["joint_spatial_voxel_count"],
        f"{name}.joint_spatial_voxel_count",
        minimum=1,
    )
    single = _integer(
        descriptors["maximum_single_scan_spatial_voxel_count"],
        f"{name}.maximum_single_scan_spatial_voxel_count",
        minimum=1,
    )
    if (
        sum(per_scan_returns) != returns
        or max(per_scan_voxels) != single
        or not single <= joint <= returns
    ):
        raise ProtocolError(
            f"{name} must satisfy single voxels <= joint voxels <= returns"
        )
    gain = _number(descriptors["densification_gain"], f"{name}.densification_gain")
    duplicate = _number(descriptors["duplicate_fraction"], f"{name}.duplicate_fraction")
    if not math.isclose(gain, joint / single, rel_tol=1e-9, abs_tol=1e-12):
        raise ProtocolError(f"{name}.densification_gain disagrees with counts")
    if not math.isclose(duplicate, 1.0 - joint / returns, rel_tol=1e-9, abs_tol=1e-12):
        raise ProtocolError(f"{name}.duplicate_fraction disagrees with counts")
    if _number(descriptors["median_distance_m"], f"{name}.median_distance_m") <= 0:
        raise ProtocolError(f"{name}.median_distance_m must be positive")
    occlusion = _number(descriptors["occlusion_rate"], f"{name}.occlusion_rate")
    if not 0.0 <= occlusion <= 1.0:
        raise ProtocolError(f"{name}.occlusion_rate must lie in [0,1]")
    _integer(
        descriptors["support_semantic_id"], f"{name}.support_semantic_id", minimum=1
    )
    visible = _integer(
        descriptors["visible_scan_count"], f"{name}.visible_scan_count", minimum=1
    )
    if visible != sum(value > 0 for value in per_scan_returns):
        raise ProtocolError(
            f"{name}.visible_scan_count disagrees with per-scan returns"
        )
    if visible > WINDOW_FRAMES:
        raise ProtocolError(f"{name}.visible_scan_count cannot exceed five")
    _number(
        descriptors["minimum_visible_return_height_m"],
        f"{name}.minimum_visible_return_height_m",
    )
    intensity = tuple(
        _number(value, f"{name}.intensity_q05_median_q95")
        for value in _list(
            descriptors["intensity_q05_median_q95"],
            f"{name}.intensity_q05_median_q95",
        )
    )
    if len(intensity) != 3 or intensity != tuple(sorted(intensity)):
        raise ProtocolError(f"{name}.intensity_q05_median_q95 is invalid")
    beam = _int_tuple(descriptors["beam_histogram"], f"{name}.beam_histogram")
    if len(beam) != 128 or sum(beam) != returns:
        raise ProtocolError(f"{name}.beam_histogram is invalid")


def _recompute_r02_audits(
    clips: tuple[DevelopmentClip, ...],
) -> tuple[
    tuple[object, ...] | None,
    Mapping[str, object] | None,
    Mapping[str, object] | None,
]:
    """Re-run matching and shortcut statistics from frozen manifest descriptors."""

    try:
        from .render import (
            RenderError,
            WindowEntityDescriptor,
            match_window_descriptor_records,
            window_matching_balance,
            window_shortcut_audit,
        )
    except ImportError:  # pragma: no cover - direct module execution
        from render import (
            RenderError,
            WindowEntityDescriptor,
            match_window_descriptor_records,
            window_matching_balance,
            window_shortcut_audit,
        )

    records: list[tuple[str, str, object]] = []
    try:
        for clip in clips:
            for window in clip.windows:
                for value in window.descriptors:
                    descriptor = WindowEntityDescriptor(
                        object_id=int(value["object_id"]),
                        label=str(value["label"]),
                        visible_returns_by_scan=tuple(
                            int(item) for item in value["visible_returns_by_scan"]
                        ),
                        spatial_voxels_by_scan=tuple(
                            int(item) for item in value["spatial_voxels_by_scan"]
                        ),
                        joint_visible_return_count=int(
                            value["joint_visible_return_count"]
                        ),
                        joint_spatial_voxel_count=int(
                            value["joint_spatial_voxel_count"]
                        ),
                        maximum_single_scan_spatial_voxel_count=int(
                            value["maximum_single_scan_spatial_voxel_count"]
                        ),
                        densification_gain=float(value["densification_gain"]),
                        duplicate_fraction=float(value["duplicate_fraction"]),
                        median_distance_m=float(value["median_distance_m"]),
                        occlusion_rate=float(value["occlusion_rate"]),
                        support_semantic_id=int(value["support_semantic_id"]),
                        visible_scan_count=int(value["visible_scan_count"]),
                        minimum_visible_return_height_m=float(
                            value["minimum_visible_return_height_m"]
                        ),
                        intensity_q05_median_q95=tuple(
                            float(item) for item in value["intensity_q05_median_q95"]
                        ),
                        beam_histogram=tuple(
                            int(item) for item in value["beam_histogram"]
                        ),
                    )
                    records.append((clip.world_identity, window.identity, descriptor))
    except (RenderError, TypeError, ValueError, KeyError) as error:
        raise ProtocolError(
            "R02 descriptors cannot be reconstructed from the manifest"
        ) from error
    try:
        pairs = match_window_descriptor_records(records)
        balance = window_matching_balance(pairs)
    except RenderError:
        return None, None, None
    try:
        shortcut = window_shortcut_audit(pairs, seed=R02_SHORTCUT_SEED)
    except RenderError:
        shortcut = None
    return pairs, balance, shortcut


def _validate_r02_scientific_verdict(
    value: object,
    *,
    thresholds: Mapping[str, object],
    protocol_identity: str,
    population_identity: str,
    plan_identity: str,
    validation: Mapping[str, bool],
    clips: tuple[DevelopmentClip, ...],
) -> Mapping[str, object]:
    """Validate one content-bound pass or stop decision from the R02 adjudicator."""

    verdict = _mapping(value, "development.scientific_verdict")
    _exact_keys(
        verdict,
        {
            "format",
            "development_protocol_identity",
            "formal_population_identity",
            "development_plan_identity",
            "thresholds_identity",
            "audit_algorithm_identity",
            "visual_review",
            "descriptor_integrity",
            "matching",
            "densification",
            "shortcut_audit",
            "component_decisions",
            "decision",
            "record_sha256",
        },
        "R02 scientific verdict",
    )
    _expect(verdict["format"], R02_VERDICT_FORMAT, "R02 verdict format")
    _expect(
        verdict["development_protocol_identity"],
        protocol_identity,
        "R02 verdict protocol",
    )
    _expect(
        verdict["formal_population_identity"],
        population_identity,
        "R02 formal population identity",
    )
    _expect(
        verdict["development_plan_identity"],
        plan_identity,
        "R02 development plan identity",
    )
    _expect(
        verdict["thresholds_identity"],
        _sha256_json(thresholds),
        "R02 verdict threshold identity",
    )
    _expect(
        verdict["audit_algorithm_identity"],
        r02_audit_algorithm_identity(),
        "R02 audit algorithm identity",
    )
    if set(validation) != set(R02_VALIDATION_KEYS):
        raise ProtocolError("R02 validation keys differ from the scientific contract")

    def artifact(record: Mapping[str, object], name: str) -> None:
        payload = dict(record)
        declared = _sha256(payload.pop("artifact_sha256"), f"{name}.artifact_sha256")
        if _sha256_json(payload) != declared:
            raise ProtocolError(f"{name} content hash changed")

    visual = _mapping(verdict["visual_review"], "R02 visual review")
    _exact_keys(
        visual,
        {
            "artifact_sha256",
            "reviewed_clip_count",
            "reviewed_clip_identities",
            "reviewer",
            "reviewed_at_utc",
            "checklist",
            "findings",
            "passed",
        },
        "R02 visual review",
    )
    artifact(visual, "R02 visual review")
    reviewed = _integer(visual["reviewed_clip_count"], "reviewed clip count")
    reviewed_identities = tuple(
        _sha256(value, f"reviewed clip identity {index}")
        for index, value in enumerate(
            _list(visual["reviewed_clip_identities"], "reviewed clip identities")
        )
    )
    if reviewed_identities != tuple(item.identity for item in clips):
        raise ProtocolError(
            "R02 visual review does not bind the ordered formal population"
        )
    if reviewed != len(clips) or reviewed < int(
        thresholds["minimum_visual_reviewed_clips"]
    ):
        raise ProtocolError("R02 visual review omits a formal development clip")
    _nonempty_string(visual["reviewer"], "R02 visual reviewer")
    _nonempty_string(visual["reviewed_at_utc"], "R02 visual review time")
    _nonempty_string(visual["findings"], "R02 visual findings")
    checklist = _mapping(visual["checklist"], "R02 visual checklist")
    checklist_keys = {
        "world_fixed_before_all_scans",
        "no_synthetic_point_completion",
        "no_bottom_return_insertion",
        "no_scan_duplication_or_copying",
        "placements_visually_plausible",
        "returns_and_occlusion_visually_plausible",
    }
    _exact_keys(checklist, checklist_keys, "R02 visual checklist")
    visual_pass = all(
        _boolean(checklist[name], f"R02 visual checklist {name}")
        for name in sorted(checklist_keys)
    )
    if _boolean(visual["passed"], "R02 visual review pass") is not visual_pass:
        raise ProtocolError("R02 visual decision disagrees with its checklist")

    integrity = _mapping(verdict["descriptor_integrity"], "R02 descriptor integrity")
    _exact_keys(
        integrity,
        {"checked_window_count", "checked_descriptor_count", "passed"},
        "R02 descriptor integrity",
    )
    expected_windows = sum(len(item.windows) for item in clips)
    expected_descriptors = sum(
        len(window.descriptors) for item in clips for window in item.windows
    )
    if (
        _integer(integrity["checked_window_count"], "checked window count")
        != expected_windows
        or _integer(integrity["checked_descriptor_count"], "checked descriptor count")
        != expected_descriptors
    ):
        raise ProtocolError("R02 descriptor audit does not cover the formal population")
    integrity_pass = _boolean(integrity["passed"], "R02 descriptor integrity pass")
    if not integrity_pass:
        raise ProtocolError(
            "structurally invalid descriptors cannot be loaded as R02 evidence"
        )
    reproduced_pairs, reproduced_balance, reproduced_shortcut = _recompute_r02_audits(
        clips
    )

    matching = _mapping(verdict["matching"], "R02 matching")
    matching_common = {
        "artifact_sha256",
        "status",
        "eligible_mechanism",
        "algorithm",
        "feature_names",
        "exact_matching_stratum",
        "passed",
    }
    matching_status = matching.get("status")
    if matching_status == "computed":
        _exact_keys(
            matching,
            matching_common
            | {
                "matched_pairs_sha256",
                "pair_count",
                "standardized_mean_difference",
                "support_semantic_counts",
                "maximum_absolute_standardized_mean_difference",
            },
            "R02 matching",
        )
    elif matching_status == "not_computable":
        _exact_keys(
            matching,
            matching_common | {"failure_code"},
            "R02 matching",
        )
    else:
        raise ProtocolError("R02 matching status is unsupported")
    artifact(matching, "R02 matching")
    _expect(matching["eligible_mechanism"], "in_generator", "R02 matching scope")
    _expect(
        matching["algorithm"],
        "support_stratified_linear_sum_assignment_standardized_euclidean",
        "R02 matching algorithm",
    )
    _expect(
        matching["feature_names"], list(R02_MATCHING_FEATURES), "R02 matching features"
    )
    _expect(
        matching["exact_matching_stratum"],
        "support_semantic_id",
        "R02 matching stratum",
    )
    if matching_status == "not_computable":
        _expect(
            matching["failure_code"],
            "matching_not_computable",
            "R02 matching failure code",
        )
        if reproduced_pairs is not None or reproduced_balance is not None:
            raise ProtocolError("R02 matching was reproducible but recorded as failed")
        matching_pass = False
    else:
        if reproduced_pairs is None or reproduced_balance is None:
            raise ProtocolError(
                "R02 matching claims statistics that cannot be reproduced"
            )
        pair_count = _integer(matching["pair_count"], "R02 matched pairs")
        pair_identity = _sha256(
            matching["matched_pairs_sha256"], "R02 matched-pair identity"
        )
        smd = tuple(
            _number(value, f"R02 matching SMD {index}")
            for index, value in enumerate(
                _list(
                    matching["standardized_mean_difference"],
                    "R02 standardized mean differences",
                )
            )
        )
        if len(smd) != len(R02_MATCHING_FEATURES):
            raise ProtocolError("R02 matching SMD has the wrong dimension")
        support_counts = _mapping(
            matching["support_semantic_counts"], "R02 support counts"
        )
        if (
            not support_counts
            or sum(
                _integer(value, f"R02 support count {key}", minimum=1)
                for key, value in support_counts.items()
            )
            != pair_count
        ):
            raise ProtocolError("R02 support counts disagree with matched pairs")
        imbalance = _number(
            matching["maximum_absolute_standardized_mean_difference"],
            "R02 matching imbalance",
        )
        if not math.isclose(
            imbalance, max(map(abs, smd)), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ProtocolError(
                "R02 maximum matching imbalance disagrees with its vector"
            )
        reproduced_smd = tuple(
            float(value) for value in reproduced_balance["standardized_mean_difference"]
        )
        if (
            pair_count != len(reproduced_pairs)
            or pair_identity
            != _sha256_json([item.to_dict() for item in reproduced_pairs])
            or dict(support_counts) != reproduced_balance["support_semantic_counts"]
            or len(smd) != len(reproduced_smd)
            or any(
                not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
                for left, right in zip(smd, reproduced_smd, strict=True)
            )
            or not math.isclose(
                imbalance,
                float(
                    reproduced_balance["maximum_absolute_standardized_mean_difference"]
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ProtocolError(
                "R02 matching evidence differs from deterministic recomputation"
            )
        matching_pass = pair_count >= int(
            thresholds["minimum_matched_pairs"]
        ) and imbalance <= float(
            thresholds["maximum_absolute_standardized_mean_difference"]
        )
    if _boolean(matching["passed"], "R02 matching pass") is not matching_pass:
        raise ProtocolError("R02 matching decision disagrees with its thresholds")

    density = _mapping(verdict["densification"], "R02 densification")
    _exact_keys(
        density,
        {
            "artifact_sha256",
            "eligible_mechanism",
            "descriptor_population_sha256",
            "proxy_window_entity_count",
            "median_proxy_joint_visible_return_count",
            "median_proxy_densification_gain",
            "proxy_fraction_densification_gain_above_one",
            "passed",
        },
        "R02 densification",
    )
    artifact(density, "R02 densification")
    _expect(density["eligible_mechanism"], "in_generator", "R02 density scope")
    proxy_descriptors = [
        descriptor
        for clip in clips
        for window in clip.windows
        for descriptor in window.descriptors
        if descriptor["label"] == "anomaly-proxy"
    ]
    _expect(
        density["descriptor_population_sha256"],
        _sha256_json(proxy_descriptors),
        "R02 proxy descriptor population identity",
    )
    proxy_count = _integer(
        density["proxy_window_entity_count"], "R02 proxy window entities", minimum=1
    )
    if proxy_count != len(proxy_descriptors):
        raise ProtocolError(
            "R02 densification count differs from development descriptors"
        )
    returns = sorted(
        float(item["joint_visible_return_count"]) for item in proxy_descriptors
    )
    gains = sorted(float(item["densification_gain"]) for item in proxy_descriptors)

    def median(values: list[float]) -> float:
        middle = len(values) // 2
        return (
            values[middle]
            if len(values) % 2
            else 0.5 * (values[middle - 1] + values[middle])
        )

    observed_returns = _number(
        density["median_proxy_joint_visible_return_count"],
        "R02 median proxy returns",
    )
    observed_gain = _number(
        density["median_proxy_densification_gain"],
        "R02 median proxy densification gain",
    )
    observed_fraction = _number(
        density["proxy_fraction_densification_gain_above_one"],
        "R02 densified proxy fraction",
    )
    expected_fraction = sum(value > 1.0 for value in gains) / len(gains)
    if not (
        math.isclose(observed_returns, median(returns), rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(observed_gain, median(gains), rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(
            observed_fraction, expected_fraction, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise ProtocolError("R02 densification summary differs from frozen descriptors")
    density_pass = (
        observed_returns
        >= float(thresholds["minimum_median_proxy_joint_visible_return_count"])
        and observed_gain
        >= float(thresholds["minimum_median_proxy_densification_gain"])
        and observed_fraction
        >= float(thresholds["minimum_proxy_fraction_densification_gain_above_one"])
    )
    if _boolean(density["passed"], "R02 densification pass") is not density_pass:
        raise ProtocolError("R02 densification decision disagrees with its thresholds")

    shortcut = _mapping(verdict["shortcut_audit"], "R02 shortcut audit")
    shortcut_common = {
        "artifact_sha256",
        "status",
        "eligible_mechanism",
        "algorithm",
        "feature_names",
        "seed",
        "split_unit",
        "passed",
    }
    shortcut_status = shortcut.get("status")
    if shortcut_status == "computed":
        _exact_keys(
            shortcut,
            shortcut_common
            | {
                "train_world_identities",
                "test_world_identities",
                "train_samples",
                "test_samples",
                "standardized_coefficients",
                "intercept",
                "balanced_accuracy",
                "auroc",
            },
            "R02 shortcut audit",
        )
    elif shortcut_status == "not_computable":
        _exact_keys(
            shortcut,
            shortcut_common | {"failure_code"},
            "R02 shortcut audit",
        )
    else:
        raise ProtocolError("R02 shortcut status is unsupported")
    artifact(shortcut, "R02 shortcut audit")
    _expect(shortcut["eligible_mechanism"], "in_generator", "R02 shortcut scope")
    _expect(
        shortcut["algorithm"],
        "standardized_logistic_regression",
        "R02 shortcut algorithm",
    )
    _expect(shortcut["feature_names"], list(R02_MATCHING_FEATURES), "shortcut features")
    _expect(shortcut["seed"], R02_SHORTCUT_SEED, "R02 shortcut seed")
    _expect(shortcut["split_unit"], "world_identity", "R02 shortcut split unit")
    if shortcut_status == "not_computable":
        expected_failure = (
            "matching_not_computable"
            if reproduced_pairs is None
            else "shortcut_not_computable"
        )
        _expect(
            shortcut["failure_code"],
            expected_failure,
            "R02 shortcut failure code",
        )
        if reproduced_shortcut is not None:
            raise ProtocolError("R02 shortcut was reproducible but recorded as failed")
        shortcut_pass = False
    else:
        if reproduced_pairs is None or reproduced_shortcut is None:
            raise ProtocolError(
                "R02 shortcut claims statistics that cannot be reproduced"
            )
        train_worlds = tuple(
            _sha256(value, f"shortcut train world {index}")
            for index, value in enumerate(
                _list(shortcut["train_world_identities"], "shortcut train worlds")
            )
        )
        test_worlds = tuple(
            _sha256(value, f"shortcut test world {index}")
            for index, value in enumerate(
                _list(shortcut["test_world_identities"], "shortcut test worlds")
            )
        )
        formal_worlds = {item.world_identity for item in clips}
        if (
            not train_worlds
            or not test_worlds
            or len(set(train_worlds)) != len(train_worlds)
            or len(set(test_worlds)) != len(test_worlds)
            or not set(train_worlds).isdisjoint(test_worlds)
            or not set(train_worlds).union(test_worlds).issubset(formal_worlds)
        ):
            raise ProtocolError("R02 shortcut world split is invalid")
        train_samples = _integer(
            shortcut["train_samples"], "R02 shortcut train samples", minimum=2
        )
        test_samples = _integer(
            shortcut["test_samples"], "R02 shortcut test samples", minimum=2
        )
        coefficients = tuple(
            _number(value, f"R02 shortcut coefficient {index}")
            for index, value in enumerate(
                _list(shortcut["standardized_coefficients"], "shortcut coefficients")
            )
        )
        if len(coefficients) != len(R02_MATCHING_FEATURES):
            raise ProtocolError("R02 shortcut coefficients have the wrong dimension")
        intercept = _number(shortcut["intercept"], "R02 shortcut intercept")
        balanced_accuracy = _number(
            shortcut["balanced_accuracy"], "shortcut balanced accuracy"
        )
        auroc = _number(shortcut["auroc"], "shortcut AUROC")
        if not 0.0 <= balanced_accuracy <= 1.0 or not 0.0 <= auroc <= 1.0:
            raise ProtocolError("R02 shortcut metrics must be in [0,1]")
        reproduced_worlds = sorted(
            {item.control_world_identity for item in reproduced_pairs}
            | {item.proxy_world_identity for item in reproduced_pairs}
        )
        reproduced_train_worlds = tuple(
            reproduced_worlds[int(index)]
            for index in reproduced_shortcut["train_groups"]
        )
        reproduced_test_worlds = tuple(
            reproduced_worlds[int(index)]
            for index in reproduced_shortcut["test_groups"]
        )
        reproduced_coefficients = tuple(
            float(value) for value in reproduced_shortcut["standardized_coefficients"]
        )
        if (
            train_worlds != reproduced_train_worlds
            or test_worlds != reproduced_test_worlds
            or train_samples != int(reproduced_shortcut["train_samples"])
            or test_samples != int(reproduced_shortcut["test_samples"])
            or len(coefficients) != len(reproduced_coefficients)
            or any(
                not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
                for left, right in zip(
                    coefficients, reproduced_coefficients, strict=True
                )
            )
            or not math.isclose(
                intercept,
                float(reproduced_shortcut["intercept"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                balanced_accuracy,
                float(reproduced_shortcut["balanced_accuracy"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                auroc,
                float(reproduced_shortcut["auroc"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ProtocolError(
                "R02 shortcut evidence differs from deterministic recomputation"
            )
        shortcut_pass = balanced_accuracy <= float(
            thresholds["maximum_shortcut_balanced_accuracy"]
        ) and abs(auroc - 0.5) <= float(
            thresholds["maximum_shortcut_absolute_auroc_deviation_from_half"]
        )
    if _boolean(shortcut["passed"], "R02 shortcut pass") is not shortcut_pass:
        raise ProtocolError("R02 shortcut decision disagrees with its thresholds")

    decisions = {
        "visual_review_passed": visual_pass,
        "descriptor_integrity_passed": integrity_pass,
        "proxy_control_matching_passed": matching_pass,
        "densification_passed": density_pass,
        "shortcut_audit_passed": shortcut_pass,
    }
    supplied_decisions = _mapping(verdict["component_decisions"], "R02 decisions")
    _exact_keys(supplied_decisions, set(R02_VALIDATION_KEYS), "R02 decisions")
    for name, expected in decisions.items():
        _expect(supplied_decisions[name], expected, f"R02 decision {name}")
        _expect(validation[name], expected, f"R02 validation {name}")
    expected_decision = "pass" if all(decisions.values()) else "fail"
    _expect(verdict["decision"], expected_decision, "R02 scientific decision")
    identity_payload = dict(verdict)
    record_sha256 = _sha256(identity_payload.pop("record_sha256"), "R02 record hash")
    if _sha256_json(identity_payload) != record_sha256:
        raise ProtocolError("R02 scientific verdict content hash changed")
    return _freeze(verdict)  # type: ignore[return-value]


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
    if (
        root.get("format") == "ajae-development-worlds-v2"
        or root.get("protocol_schema") == 30
    ):
        raise ProtocolError(
            "schema-30 centered dev.json is retired; regenerate schema-31 window/clip data"
        )
    root_keys = {
        "format",
        "protocol_schema",
        "protocol_identity",
        "plan_identity",
        "population_identity",
        "sequence_id",
        "status",
        "validation",
        "clip_count",
        "window_count",
        "clips",
        "scientific_verdict",
    }
    _exact_keys(root, root_keys, "development data")
    _expect(root["format"], DEVELOPMENT_FORMAT, "development format")
    _expect(root["protocol_schema"], SCHEMA_VERSION, "development protocol_schema")
    _expect(
        root["protocol_identity"],
        protocol.development_population_identity,
        "development protocol_identity",
    )
    _expect(
        root["plan_identity"],
        protocol.development_clip_plan_identity,
        "development plan_identity",
    )
    _expect(root["sequence_id"], 201, "development sequence_id")
    status = _nonempty_string(root["status"], "development status")
    if status not in {
        "not_generated_R02",
        "definitions_only_unvalidated",
        "validated_frozen",
        "adjudicated_failed_R02",
    }:
        raise ProtocolError("development status is unsupported")
    validation_source = _mapping(root["validation"], "development.validation")
    validation = {
        name: _boolean(value, f"development.validation.{name}")
        for name, value in validation_source.items()
    }
    required = _string_tuple(
        _plain(protocol.render["window_descriptors"]["required"]),
        "render.window_descriptors.required",
    )

    def digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def sha(value: object, name: str) -> str:
        result = _nonempty_string(value, name)
        if not re.fullmatch(r"[0-9a-f]{64}", result):
            raise ProtocolError(f"{name} must be a lowercase SHA-256 digest")
        return result

    legal_starts = frozenset(protocol.development_sequence.legal_window_starts())
    clips: list[DevelopmentClip] = []
    plan = protocol.development_clip_plan()
    for clip_index, raw_clip in enumerate(_list(root["clips"], "development.clips")):
        if clip_index >= len(plan):
            raise ProtocolError("development data exceed the frozen 24-clip plan")
        planned = plan[clip_index]
        clip_name = f"development.clips[{clip_index}]"
        clip = _mapping(raw_clip, clip_name)
        clip_keys = {
            "format",
            "identity",
            "world_identity",
            "clip_start",
            "frame_ids",
            "renderer_identity",
            "mechanism",
            "source_observation_identities",
            "world",
            "report",
            "windows",
        }
        _exact_keys(clip, clip_keys, clip_name)
        _expect(clip["format"], "ajae-development-clip-world-v1", f"{clip_name}.format")
        identity = sha(clip["identity"], f"{clip_name}.identity")
        world_identity = sha(clip["world_identity"], f"{clip_name}.world_identity")
        renderer_identity = sha(
            clip["renderer_identity"], f"{clip_name}.renderer_identity"
        )
        if renderer_identity != protocol.renderer_identity:
            raise ProtocolError(f"{clip_name} uses a different renderer identity")
        start = _integer(clip["clip_start"], f"{clip_name}.clip_start")
        frame_ids = _int_tuple(clip["frame_ids"], f"{clip_name}.frame_ids")
        if start != planned["clip_start"] or frame_ids != planned["source_frames"]:
            raise ProtocolError(f"{clip_name} differs from the frozen generation plan")
        if len(frame_ids) < 9 or frame_ids != tuple(
            range(start, start + len(frame_ids))
        ):
            raise ProtocolError(
                f"{clip_name} requires at least nine consecutive frames"
            )
        if any(
            window_start not in legal_starts
            for window_start in range(start, frame_ids[-1] - 3)
        ):
            raise ProtocolError(f"{clip_name} contains an illegal train/201 window")
        mechanism = _nonempty_string(clip["mechanism"], f"{clip_name}.mechanism")
        if mechanism != "in_generator" or mechanism != planned["mechanism"]:
            raise ProtocolError(f"{clip_name} opens a held-out mechanism before S01")
        source_observation_identities = tuple(
            sha(value, f"{clip_name}.source_observation_identities[{index}]")
            for index, value in enumerate(
                _list(
                    clip["source_observation_identities"],
                    f"{clip_name}.source_observation_identities",
                )
            )
        )
        if len(source_observation_identities) != len(frame_ids):
            raise ProtocolError(
                f"{clip_name} must bind one observation identity per source frame"
            )
        world = _mapping(clip["world"], f"{clip_name}.world")
        report = _mapping(clip["report"], f"{clip_name}.report")
        if (
            world.get("source_sequence_id") != 201
            or report.get("source_sequence_id") != 201
        ):
            raise ProtocolError(f"{clip_name} must use train/201")
        root_seed = int(planned["root_seed"])
        stride = int(planned["observation_attempt_stride"])
        attempts = int(planned["maximum_observation_attempts"])
        world_seed = _integer(world.get("seed"), f"{clip_name}.world.seed")
        if (
            world_seed < root_seed
            or (world_seed - root_seed) % stride
            or (world_seed - root_seed) // stride >= attempts
        ):
            raise ProtocolError(f"{clip_name}.world.seed differs from its frozen plan")
        if digest(world) != world_identity:
            raise ProtocolError(f"{clip_name}.world_identity does not hash WorldSpec")
        objects = _list(world.get("objects"), f"{clip_name}.world.objects")
        object_labels: dict[int, str] = {}
        proxy_kinds: list[str] = []
        for object_index, raw_object in enumerate(objects):
            object_name = f"{clip_name}.world.objects[{object_index}]"
            object_ = _mapping(raw_object, object_name)
            object_id = _integer(
                object_.get("object_id"), f"{object_name}.object_id", minimum=1
            )
            label = _nonempty_string(object_.get("label"), f"{object_name}.label")
            if object_id in object_labels or label not in {
                "normal-control",
                "anomaly-proxy",
            }:
                raise ProtocolError(f"{object_name} has an invalid identity or label")
            object_labels[object_id] = label
            if label == "anomaly-proxy":
                shape = _mapping(object_.get("shape"), f"{object_name}.shape")
                proxy_kinds.append(str(shape.get("kind")))
        if not {"normal-control", "anomaly-proxy"}.issubset(
            set(object_labels.values())
        ):
            raise ProtocolError(f"{clip_name} must contain both controls and proxies")
        if any(kind == "held-out-torus-sdf" for kind in proxy_kinds):
            raise ProtocolError(f"{clip_name} opens a held-out proxy before S01")

        parsed_windows: list[DevelopmentWindow] = []
        raw_windows = _list(clip["windows"], f"{clip_name}.windows")
        expected_starts = tuple(range(start, frame_ids[-1] - 3))
        if len(raw_windows) != len(expected_starts):
            raise ProtocolError(f"{clip_name} omits an overlapping window")
        for window_index, (raw_window, expected_start) in enumerate(
            zip(raw_windows, expected_starts, strict=True)
        ):
            window_name = f"{clip_name}.windows[{window_index}]"
            window = _mapping(raw_window, window_name)
            _exact_keys(
                window,
                {
                    "identity",
                    "window_start",
                    "frame_ids",
                    "source_observation_identities",
                    "descriptors",
                },
                window_name,
            )
            window_identity = sha(window["identity"], f"{window_name}.identity")
            window_start = _integer(
                window["window_start"], f"{window_name}.window_start"
            )
            window_frames = _int_tuple(window["frame_ids"], f"{window_name}.frame_ids")
            if window_start != expected_start or window_frames != tuple(
                window_start + offset for offset in WINDOW_MEMBER_OFFSETS
            ):
                raise ProtocolError(f"{window_name} has the wrong five-scan identity")
            window_observation_identities = tuple(
                sha(
                    value,
                    f"{window_name}.source_observation_identities[{index}]",
                )
                for index, value in enumerate(
                    _list(
                        window["source_observation_identities"],
                        f"{window_name}.source_observation_identities",
                    )
                )
            )
            clip_offset = window_start - start
            if (
                len(window_observation_identities) != WINDOW_FRAMES
                or window_observation_identities
                != source_observation_identities[
                    clip_offset : clip_offset + WINDOW_FRAMES
                ]
            ):
                raise ProtocolError(
                    f"{window_name} observation identities differ from its clip"
                )
            expected_window_identity = digest(
                {
                    "format": "ajae-window-world-v1",
                    "world_identity": world_identity,
                    "partition": "train",
                    "sequence_id": 201,
                    "window_start": window_start,
                    "frame_ids": window_frames,
                    "renderer_identity": renderer_identity,
                    "source_observation_identities": window_observation_identities,
                }
            )
            if window_identity != expected_window_identity:
                raise ProtocolError(f"{window_name}.identity does not match its inputs")
            descriptor_items: list[Mapping[str, object]] = []
            descriptor_ids: list[int] = []
            for descriptor_index, raw_descriptor in enumerate(
                _list(window["descriptors"], f"{window_name}.descriptors")
            ):
                descriptor_name = f"{window_name}.descriptors[{descriptor_index}]"
                descriptor = _mapping(raw_descriptor, descriptor_name)
                _validate_development_descriptors(descriptor, required, descriptor_name)
                object_id = int(descriptor["object_id"])
                if object_labels.get(object_id) != descriptor["label"]:
                    raise ProtocolError(
                        f"{descriptor_name} does not identify a world object"
                    )
                descriptor_ids.append(object_id)
                descriptor_items.append(_freeze(descriptor))  # type: ignore[arg-type]
            if tuple(descriptor_ids) != tuple(sorted(object_labels)):
                raise ProtocolError(
                    f"{window_name} descriptors do not cover every object"
                )
            parsed_windows.append(
                DevelopmentWindow(
                    window_identity,
                    window_start,
                    window_frames,
                    window_observation_identities,
                    tuple(descriptor_items),
                )
            )
        expected_clip_identity = digest(
            {
                "format": "ajae-development-clip-world-v1",
                "world_identity": world_identity,
                "clip_start": start,
                "frame_ids": frame_ids,
                "renderer_identity": renderer_identity,
                "mechanism": mechanism,
                "source_observation_identities": source_observation_identities,
            }
        )
        if identity != expected_clip_identity:
            raise ProtocolError(f"{clip_name}.identity does not match its inputs")
        clips.append(
            DevelopmentClip(
                identity,
                world_identity,
                start,
                frame_ids,
                renderer_identity,
                mechanism,
                source_observation_identities,
                _freeze(world),  # type: ignore[arg-type]
                _freeze(report),  # type: ignore[arg-type]
                tuple(parsed_windows),
            )
        )
    if len({item.identity for item in clips}) != len(clips) or len(
        {item.world_identity for item in clips}
    ) != len(clips):
        raise ProtocolError("development data repeat a clip or world identity")
    if clips and (
        len(clips) != len(plan)
        or any(item.mechanism != "in_generator" for item in clips)
        or any(len(item.frame_ids) != 9 or len(item.windows) != 5 for item in clips)
    ):
        raise ProtocolError("formal development data violate the frozen 24-clip design")
    clip_count = _integer(root["clip_count"], "development.clip_count")
    window_count = _integer(root["window_count"], "development.window_count")
    if clip_count != len(clips) or window_count != sum(
        len(item.windows) for item in clips
    ):
        raise ProtocolError("development clip/window counts are inconsistent")
    expected_population_identity = (
        None
        if not clips
        else _sha256_json(
            {
                "format": "ajae-schema31-formal-development-population-v1",
                "protocol_identity": protocol.development_population_identity,
                "plan_identity": protocol.development_clip_plan_identity,
                "clips": root["clips"],
            }
        )
    )
    _expect(
        root["population_identity"],
        expected_population_identity,
        "development population_identity",
    )
    scientific_verdict: Mapping[str, object] | None = None
    raw_scientific_verdict = root["scientific_verdict"]
    if status == "not_generated_R02":
        if clips or validation or raw_scientific_verdict is not None:
            raise ProtocolError("not-generated development data cannot carry evidence")
    elif not clips:
        raise ProtocolError("generated development data cannot be empty")
    if status == "definitions_only_unvalidated" and (
        validation or raw_scientific_verdict is not None
    ):
        raise ProtocolError(
            "unvalidated development definitions cannot carry a verdict"
        )
    if status in {"validated_frozen", "adjudicated_failed_R02"}:
        thresholds = _r02_thresholds(
            protocol.render["proxy_control_matching"]["thresholds"]
        )
        if thresholds is None:
            raise ProtocolError(
                "R02 thresholds must be frozen before development validation"
            )
        scientific_verdict = _validate_r02_scientific_verdict(
            raw_scientific_verdict,
            thresholds=thresholds,
            protocol_identity=protocol.development_population_identity,
            population_identity=expected_population_identity,  # type: ignore[arg-type]
            plan_identity=protocol.development_clip_plan_identity,
            validation=validation,
            clips=tuple(clips),
        )
        expected_decision = "pass" if status == "validated_frozen" else "fail"
        if scientific_verdict["decision"] != expected_decision:
            raise ProtocolError("development status disagrees with the R02 decision")
    node = str(protocol.status["current_node"])
    if _STATE_MACHINE.index(node) > _STATE_MACHINE.index("R02"):
        if status != "validated_frozen" or scientific_verdict is None:
            raise ProtocolError("nodes after R02 require a passed development verdict")
        _expect(
            protocol.status["r02_population_identity"],
            expected_population_identity,
            "status R02 population identity",
        )
        _expect(
            protocol.status["r02_verdict_identity"],
            scientific_verdict["record_sha256"],
            "status R02 verdict identity",
        )
    return DevelopmentWorlds(
        DEVELOPMENT_FORMAT,
        SCHEMA_VERSION,
        protocol.development_population_identity,
        protocol.development_clip_plan_identity,
        expected_population_identity,
        201,
        status,
        MappingProxyType(validation),
        scientific_verdict,
        tuple(clips),
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
    print(
        json.dumps(
            load_protocol(args.protocol).summary(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
