#!/usr/bin/env python3
"""Load and validate the AJAE schema-34 data and supervision contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "protocol.json"
SCHEMA_VERSION = 34
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
    """Report a contradiction in the active data contract."""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ProtocolError(f"{name} must be a JSON object")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} must be a non-empty string")
    return value


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"{name} must be an array")
    return tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(value))


def _sha256(value: object, name: str, *, pending_allowed: bool = False) -> str | None:
    if value is None and pending_allowed:
        return None
    digest = _string(value, name)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FrameSpan:
    """A half-open contiguous frame interval."""

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
    """One physical STU sequence and its complete active role."""

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
class SyntheticPoolSpec:
    """A predeclared set of synthetic sequences and segment-local windows."""

    name: str
    source_sequence_id: int
    synthetic_sequence_count: int
    segments: tuple[FrameSpan, ...]
    seed_base: int
    output_directory: str
    declared_world_count: int
    declared_windows_per_sequence: int
    declared_total_window_count: int

    def __post_init__(self) -> None:
        _string(self.name, "pool name")
        _integer(self.source_sequence_id, "pool source sequence")
        _integer(
            self.synthetic_sequence_count,
            "synthetic sequence count",
            minimum=1,
        )
        _integer(self.seed_base, "seed base")
        _string(self.output_directory, "output directory")
        if not self.segments:
            raise ProtocolError("a synthetic pool must contain segments")
        cursor = self.segments[0].start
        for segment in self.segments:
            if segment.start != cursor or len(segment) < WINDOW_FRAMES:
                raise ProtocolError(
                    "pool segments must be contiguous and at least five frames"
                )
            cursor = segment.stop
        worlds = self.synthetic_sequence_count * len(self.segments)
        windows = sum(len(item) - WINDOW_FRAMES + 1 for item in self.segments)
        if (
            self.declared_world_count != worlds
            or self.declared_windows_per_sequence != windows
            or self.declared_total_window_count
            != self.synthetic_sequence_count * windows
        ):
            raise ProtocolError(
                f"{self.name} declares inconsistent world or window counts"
            )

    @property
    def source_span(self) -> FrameSpan:
        return FrameSpan(self.segments[0].start, self.segments[-1].stop)

    @property
    def world_count(self) -> int:
        return self.synthetic_sequence_count * len(self.segments)

    @property
    def windows_per_sequence(self) -> int:
        return sum(len(item) - WINDOW_FRAMES + 1 for item in self.segments)

    @property
    def total_window_count(self) -> int:
        return self.synthetic_sequence_count * self.windows_per_sequence

    def synthetic_sequence_id(self, sequence_index: int) -> str:
        index = _integer(sequence_index, "synthetic sequence index")
        if index >= self.synthetic_sequence_count:
            raise IndexError(index)
        return f"{self.name}/{index:03d}"

    def world_seed(self, sequence_index: int, segment_index: int) -> int:
        sequence = _integer(sequence_index, "synthetic sequence index")
        segment = _integer(segment_index, "segment index")
        if sequence >= self.synthetic_sequence_count or segment >= len(self.segments):
            raise IndexError((sequence, segment))
        return self.seed_base + 1000 * sequence + segment

    def window_starts(self, segment_index: int) -> tuple[int, ...]:
        segment = self.segments[_integer(segment_index, "segment index")]
        return tuple(range(segment.start, segment.stop - WINDOW_FRAMES + 1))


class AJAEProtocol:
    """Validated immutable view of the active schema-34 data contract."""

    def __init__(self, document: Mapping[str, object], *, path: Path) -> None:
        self._validate(document)
        self.path = path.expanduser().resolve(strict=True)
        self.schema_version = SCHEMA_VERSION
        self._document = _freeze(document)
        for name in (
            "status",
            "authority",
            "data",
            "window",
            "labels",
            "synthetic_pools",
            "storage",
            "predictions",
            "artifacts",
            "qualification",
        ):
            setattr(self, name, self._document[name])

        data = _mapping(document["data"], "data")
        self.training_sequence = self._sequence(data, "parameter_update_source")
        self.validation_sequence = self._sequence(data, "model_validation_source")
        public = _mapping(data["real_anomaly_final_test"], "real anomaly test")
        hidden = _mapping(data["hidden_test"], "hidden test")
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
                self.training_sequence,
                self.validation_sequence,
                *self.public_validation,
                *self.hidden_test,
            )
        }
        class_map = _mapping(
            _mapping(document["labels"], "labels")["normal_semantic_class_map"],
            "normal semantic class map",
        )
        self.semantic_class_map = MappingProxyType(
            {
                int(raw): _integer(target, f"class map {raw}")
                for raw, target in class_map.items()
            }
        )
        pools = _mapping(document["synthetic_pools"], "synthetic pools")
        self.training_pool = self._pool("train_v1", pools["train_v1"])
        self.validation_pool = self._pool("validation_v1", pools["validation_v1"])

    @staticmethod
    def _sequence(data: Mapping[str, object], key: str) -> SequenceSpec:
        record = _mapping(data[key], f"data.{key}")
        bounds = _int_tuple(record["frame_range_inclusive"], f"{key} range")
        if len(bounds) != 2 or bounds[1] < bounds[0]:
            raise ProtocolError(f"{key} frame range must be [first,last]")
        labels_available = record["labels_available"]
        if type(labels_available) is not bool:
            raise ProtocolError(f"{key}.labels_available must be boolean")
        return SequenceSpec(
            _string(record["partition"], f"{key} partition"),
            _integer(record["sequence_id"], f"{key} sequence"),
            _string(record["role"], f"{key} role"),
            labels_available,
            FrameSpan(bounds[0], bounds[1] + 1),
        )

    @staticmethod
    def _pool(name: str, value: object) -> SyntheticPoolSpec:
        record = _mapping(value, f"synthetic_pools.{name}")
        raw_segments = record["segment_boundaries_inclusive"]
        if not isinstance(raw_segments, (list, tuple)):
            raise ProtocolError(f"{name} segment boundaries must be an array")
        segments: list[FrameSpan] = []
        for index, raw in enumerate(raw_segments):
            bounds = _int_tuple(raw, f"{name} segment {index}")
            if len(bounds) != 2 or bounds[1] < bounds[0]:
                raise ProtocolError(f"{name} segment {index} must be [first,last]")
            segments.append(FrameSpan(bounds[0], bounds[1] + 1))
        return SyntheticPoolSpec(
            name=name,
            source_sequence_id=_integer(record["source_sequence_id"], f"{name} source"),
            synthetic_sequence_count=_integer(
                record["synthetic_sequence_count"],
                f"{name} sequence count",
                minimum=1,
            ),
            segments=tuple(segments),
            seed_base=_integer(record["seed_base"], f"{name} seed base"),
            output_directory=_string(
                record["output_directory"], f"{name} output directory"
            ),
            declared_world_count=_integer(
                record["world_count"], f"{name} world count", minimum=1
            ),
            declared_windows_per_sequence=_integer(
                record["windows_per_synthetic_sequence"],
                f"{name} windows per sequence",
                minimum=1,
            ),
            declared_total_window_count=_integer(
                record["total_window_count"],
                f"{name} total window count",
                minimum=1,
            ),
        )

    @property
    def document(self) -> Mapping[str, object]:
        return self._document  # type: ignore[return-value]

    @property
    def contract_identity(self) -> str:
        source = _plain(self._document)
        contract = {
            key: source[key]
            for key in (
                "schema_version",
                "authority",
                "research_question",
                "data",
                "window",
                "labels",
                "synthetic_pools",
                "storage",
                "predictions",
                "artifacts",
            )
        }
        payload = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def execution_identity(self) -> str:
        return _file_sha256(self.path)

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
                f"sequence {partition}/{sequence_id} is outside schema 34"
            ) from error

    def _artifact_path(self, record: Mapping[str, object]) -> Path:
        return (self.path.parent / str(record["file"])).resolve()

    def sensor_calibration_path(self) -> Path:
        record = _mapping(self.artifacts["sensor_calibration"], "sensor calibration")
        return self._artifact_path(record)

    def verify_sensor_calibration(self) -> Path:
        record = _mapping(self.artifacts["sensor_calibration"], "sensor calibration")
        path = self._artifact_path(record)
        if not path.is_file() or _file_sha256(path) != str(record["sha256"]):
            raise ProtocolError("sensor calibration bytes differ from protocol")
        return path

    def support_pool_path(self, sequence_id: int) -> Path:
        pools = _mapping(self.artifacts["qualified_support_pools"], "support pools")
        key = f"train/{_integer(sequence_id, 'support sequence')}"
        return self._artifact_path(_mapping(pools[key], "support pool"))

    def verify_support_pool(self, sequence_id: int) -> Path:
        pools = _mapping(self.artifacts["qualified_support_pools"], "support pools")
        key = f"train/{_integer(sequence_id, 'support sequence')}"
        record = _mapping(pools[key], "support pool")
        expected = record["sha256"]
        if expected is None:
            raise ProtocolError("support pool is not frozen yet")
        path = self._artifact_path(record)
        if not path.is_file() or _file_sha256(path) != expected:
            raise ProtocolError("support-pool bytes differ from protocol")
        return path

    def pool_manifest_path(self, pool_name: str) -> Path:
        key = {
            "train_v1": "train_pool_manifest",
            "validation_v1": "validation_pool_manifest",
        }.get(pool_name)
        if key is None:
            raise KeyError(pool_name)
        return self._artifact_path(_mapping(self.artifacts[key], key))

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "route": self.status["route"],
            "state": self.status["state"],
            "training_source": "train/206 frames 0-448",
            "validation_source": "train/201 frames 0-681",
            "normal_validation_windows": len(
                self.validation_sequence.legal_window_starts()
            ),
            "synthetic_training_windows": self.training_pool.total_window_count,
            "synthetic_validation_windows": self.validation_pool.total_window_count,
            "training_allowed": self.status["training_allowed"],
            "real_anomaly_access_allowed": self.status["real_anomaly_access_allowed"],
        }

    @classmethod
    def _validate(cls, source: Mapping[str, object]) -> None:
        expected_root = {
            "schema_version",
            "status",
            "authority",
            "research_question",
            "data",
            "window",
            "labels",
            "synthetic_pools",
            "storage",
            "predictions",
            "artifacts",
            "qualification",
        }
        if (
            source.get("schema_version") != SCHEMA_VERSION
            or set(source) != expected_root
        ):
            raise ProtocolError("protocol must be the sole schema-34 contract")

        status = _mapping(source["status"], "status")
        if status.get("route") != "ajae_data_and_five_frame_supervision_v2":
            raise ProtocolError("schema 34 route is invalid")
        state = status.get("state")
        if state not in {"qualification_pending", "frozen"}:
            raise ProtocolError("schema 34 state is invalid")
        for key in (
            "data_pool_frozen",
            "training_allowed",
            "validation_tuning_allowed",
            "real_anomaly_access_allowed",
            "old_F2_F3_retired",
        ):
            if type(status.get(key)) is not bool:
                raise ProtocolError(f"status.{key} must be boolean")
        frozen = state == "frozen"
        if (
            status["data_pool_frozen"] is not frozen
            or status["training_allowed"] is not frozen
            or status["validation_tuning_allowed"] is not frozen
            or status["real_anomaly_access_allowed"] is not False
            or status["old_F2_F3_retired"] is not True
        ):
            raise ProtocolError("schema 34 execution permissions contradict its state")

        authority = _mapping(source["authority"], "authority")
        history = _mapping(authority["history"], "history")
        if (
            authority.get("scientific_document") != "AJAE数据与五帧监督协议v2.md"
            or authority.get("supersedes")
            != "schema33_frozen_stu_dense_input_feasibility"
            or history.get("schema33_protocol") != "history/schema33/protocol.json"
            or history.get("F0_artifact") != "artifacts/f0_qualification.json"
            or history.get("F1_artifact") != "artifacts/f1_geometry.json"
        ):
            raise ProtocolError("schema-34 authority or historical boundary changed")
        for key in ("schema33_protocol_sha256", "F0_sha256", "F1_sha256"):
            _sha256(history[key], f"history.{key}")
        if history.get("interpretation") != "historical_mechanism_evidence_only":
            raise ProtocolError("schema 33 evidence cannot be active evidence")

        data = _mapping(source["data"], "data")
        archives = _mapping(data["official_archive_sha256"], "official archives")
        if set(archives) != {"train.zip", "val.zip", "test.zip"}:
            raise ProtocolError("official archive identities are incomplete")
        for name, digest in archives.items():
            _sha256(digest, f"archive {name}")
        training = cls._sequence(data, "parameter_update_source")
        validation = cls._sequence(data, "model_validation_source")
        training_record = _mapping(data["parameter_update_source"], "training")
        if (
            training.partition != "train"
            or training.sequence_id != 206
            or training.span != FrameSpan(0, 449)
            or training.role != "only_source_allowed_to_influence_parameter_updates"
            or training.labels_available is not True
            or training_record.get("gradient_updates_allowed") is not True
        ):
            raise ProtocolError("train/206 frames 0-448 must be the sole update source")
        validation_record = _mapping(data["model_validation_source"], "validation")
        expected_duplicate_runs = {
            "0": ((0, 131072, 0), (131072, 131072, 0), (262144, 131072, 0)),
            "1": ((0, 131072, 0), (131072, 131072, 0), (262144, 131072, 0)),
            "2": ((0, 29184, 0), (29184, 131072, 0), (160256, 131072, 0)),
            "3": ((0, 131072, 0), (131072, 131072, 0)),
        }
        duplicate_runs = _mapping(
            validation_record["duplicate_prefix_ray_runs"],
            "duplicate prefix ray runs",
        )
        if (
            validation.partition != "train"
            or validation.sequence_id != 201
            or validation.span != FrameSpan(0, 682)
            or validation.role
            != "only_source_for_model_validation_hyperparameter_tuning_and_model_selection"
            or validation.labels_available is not True
            or validation_record.get("gradient_updates_allowed") is not False
            or validation_record.get("normal_window_count") != 678
            or _int_tuple(
                validation_record["normal_output_frame_range_inclusive"],
                "normal outputs",
            )
            != (4, 681)
            or _int_tuple(
                validation_record["known_duplicate_prefix_frames"],
                "duplicate prefix",
            )
            != (0, 1, 2, 3)
            or set(duplicate_runs) != set(expected_duplicate_runs)
            or any(
                tuple(
                    _int_tuple(run, f"duplicate frame {frame} run")
                    for run in duplicate_runs[frame]
                )
                != expected
                for frame, expected in expected_duplicate_runs.items()
            )
            or validation_record.get("duplicate_policy") != "retain_and_report"
        ):
            raise ProtocolError(
                "train/201 must be one complete no-gradient validation sequence"
            )
        public = _mapping(data["real_anomaly_final_test"], "real anomaly test")
        hidden = _mapping(data["hidden_test"], "hidden test")
        if (
            public.get("partition") != "val"
            or public.get("role")
            != "sealed_until_model_structure_training_recipe_hyperparameters_and_selection_rule_are_fixed"
            or public.get("labels_available") is not True
            or _int_tuple(public["sequence_ids"], "public ids") != PUBLIC_ANOMALY_IDS
        ):
            raise ProtocolError("the 19 real anomaly sequences changed")
        if (
            hidden.get("partition") != "test"
            or hidden.get("role") != "sealed_final_hidden_test"
            or hidden.get("labels_available") is not False
            or _int_tuple(hidden["sequence_ids"], "hidden ids") != HIDDEN_TEST_IDS
        ):
            raise ProtocolError("hidden test sequence identities changed")

        window = _mapping(source["window"], "window")
        if (
            window.get("frames") != WINDOW_FRAMES
            or tuple(window.get("causal_offsets_from_current", ()))
            != (-4, -3, -2, -1, 0)
            or window.get("prediction_scope") != "all_visible_points_in_the_five_frames"
            or window.get("supervision_scope")
            != "all_visible_points_with_valid_binary_truth_in_the_five_frames"
            or window.get("online_output") != "current_frame_point_anomaly_scores_only"
            or window.get("current_mask_role")
            != "online_output_extraction_only_never_training_loss"
            or window.get("coordinate_reference") != "current_frame_lidar"
            or window.get("current_transform") != "direct_bitwise_copy_of_raw_xyz"
            or _mapping(
                window.get("historical_coordinate_tolerance"),
                "historical coordinate tolerance",
            )
            != {"absolute_m": 1e-6, "relative": 1e-5}
            or _mapping(window.get("rigid_pose_tolerance"), "rigid pose tolerance")
            != {"absolute": 0.001, "relative": 0.001}
            or tuple(window.get("point_identity", ()))
            != (
                "synthetic_or_raw_sequence_id",
                "source_frame",
                "source_slot",
            )
            or window.get("overlap_fusion_for_online_result") != "forbidden"
            or tuple(window.get("discrete_fields_exact", ()))
            != (
                "source_frame",
                "source_slot",
                "label_state",
                "current_mask",
                "point_count",
            )
        ):
            raise ProtocolError(
                "the five-frame observation or supervision meaning changed"
            )

        labels = _mapping(source["labels"], "labels")
        states = _mapping(labels["states"], "label states")
        raw_rule = _mapping(labels["raw_rule"], "raw label rule")
        if (
            labels.get("field") != "anomaly_target"
            or states != {"-1": "ignore", "0": "normal", "1": "anomaly"}
            or raw_rule
            != {
                "semantic_0": "ignore",
                "semantic_2": "anomaly",
                "other_nonzero_semantics": "normal",
            }
            or labels.get("loss_rule")
            != "binary_loss_over_anomaly_target_in_{0,1}_only"
            or labels.get("feature_leakage")
            != "labels_masks_and_world_parameters_are_forbidden_model_features"
        ):
            raise ProtocolError("pointwise supervision semantics changed")

        storage = _mapping(source["storage"], "storage")
        if (
            storage.get("format") != "ajae-sparse-rendered-segment-v1"
            or storage.get("source_frames")
            != "official_raw_STU_files_bound_by_per_frame_content_hashes"
            or storage.get("synthetic_delta")
            != "only_slots_whose_visible_return_is_replaced_by_an_anomaly_proxy"
            or storage.get("one_file_per_segment") is not True
            or storage.get("window_storage")
            != "identities_and_frame_references_only_no_duplicate_point_arrays"
            or tuple(storage.get("segment_required_arrays", ()))
            != (
                "frame_ids",
                "frame_offsets",
                "changed_slots",
                "changed_xyzi",
                "changed_packed_labels",
                "changed_object_ids",
            )
            or tuple(storage.get("segment_required_metadata", ()))
            != (
                "synthetic_sequence_id",
                "segment_index",
                "segment_boundary_inclusive",
                "seed",
                "world_identity",
                "world_content_identity",
                "world",
                "world_generation_report",
                "renderer_identity",
                "raw_source_identities",
                "rendered_source_identities",
                "window_identities",
                "scientific_content_hash",
            )
        ):
            raise ProtocolError("frozen sparse-segment storage semantics changed")

        predictions = _mapping(source["predictions"], "predictions")
        if (
            predictions.get("format") != "ajae-complete-window-point-prediction-v1"
            or tuple(predictions.get("required_point_record_fields", ()))
            != (
                "synthetic_or_raw_sequence_id",
                "window_current_frame",
                "source_frame",
                "source_slot",
                "anomaly_score",
            )
            or predictions.get("online_selection")
            != "source_frame_equals_window_current_frame"
            or predictions.get("historical_context_scores")
            != "persist_unchanged_but_forbidden_from_online_primary_metric"
            or predictions.get("future_offline_fusion") != "outside_active_protocol"
        ):
            raise ProtocolError("point-score persistence or online selection changed")

        pools = _mapping(source["synthetic_pools"], "synthetic pools")
        if (
            pools.get("generation_order")
            != "sample_one_world_per_segment_then_render_each_segment_frame_once_then_cut_windows"
            or pools.get("cross_segment_windows") != "forbidden"
            or pools.get("rerender_same_frame_for_overlapping_windows") != "forbidden"
            or pools.get("root_seed_substitution_after_protocol_freeze") != "forbidden"
        ):
            raise ProtocolError("synthetic world-before-window generation changed")
        train_record = _mapping(pools["train_v1"], "train_v1")
        validation_record = _mapping(pools["validation_v1"], "validation_v1")
        train_pool = cls._pool("train_v1", train_record)
        validation_pool = cls._pool("validation_v1", validation_record)
        if (
            train_pool.source_sequence_id != 206
            or train_pool.synthetic_sequence_count != 8
            or train_pool.seed_base != 34100000
            or train_pool.output_directory != "artifacts/data_v2/train"
            or tuple(map(len, train_pool.segments)) != (28,) * 15 + (29,)
            or _int_tuple(train_record["segment_lengths"], "train segment lengths")
            != (28,) * 15 + (29,)
            or train_record.get("seed_formula")
            != "seed_base+1000*synthetic_sequence_index+segment_index"
            or train_pool.source_span != FrameSpan(0, 449)
            or train_pool.world_count != 128
            or train_pool.windows_per_sequence != 385
            or train_pool.total_window_count != 3080
        ):
            raise ProtocolError("the frozen train/206 pool plan changed")
        if (
            validation_pool.source_sequence_id != 201
            or validation_pool.synthetic_sequence_count != 4
            or validation_pool.seed_base != 34200000
            or validation_pool.output_directory != "artifacts/data_v2/validation"
            or tuple(map(len, validation_pool.segments)) != (28,) * 22 + (66,)
            or _int_tuple(
                validation_record["segment_lengths"],
                "validation segment lengths",
            )
            != (28,) * 22 + (66,)
            or validation_record.get("seed_formula")
            != "seed_base+1000*synthetic_sequence_index+segment_index"
            or validation_pool.source_span != FrameSpan(0, 682)
            or validation_pool.world_count != 92
            or validation_pool.windows_per_sequence != 590
            or validation_pool.total_window_count != 2360
        ):
            raise ProtocolError(
                "the frozen synthetic train/201 validation plan changed"
            )
        train_seeds = {
            train_pool.world_seed(sequence, segment)
            for sequence in range(train_pool.synthetic_sequence_count)
            for segment in range(len(train_pool.segments))
        }
        validation_seeds = {
            validation_pool.world_seed(sequence, segment)
            for sequence in range(validation_pool.synthetic_sequence_count)
            for segment in range(len(validation_pool.segments))
        }
        if (
            len(train_seeds) != train_pool.world_count
            or len(validation_seeds) != validation_pool.world_count
            or not train_seeds.isdisjoint(validation_seeds)
        ):
            raise ProtocolError("formal world seeds must be globally unique")

        artifacts = _mapping(source["artifacts"], "artifacts")
        calibration = _mapping(artifacts["sensor_calibration"], "sensor calibration")
        if (
            calibration.get("file") != "artifacts/calibration.pt"
            or calibration.get("source_file") != "artifacts/e11_d4b_calibration.npz"
        ):
            raise ProtocolError("sensor calibration paths changed")
        _sha256(calibration["sha256"], "sensor calibration")
        _sha256(calibration["source_sha256"], "sensor calibration source")
        support = _mapping(artifacts["qualified_support_pools"], "support pools")
        expected_support = {
            206: {
                "file": "artifacts/training_206_support_pool.npz",
                "frame_range_inclusive": (0, 448),
                "anchor_range_inclusive": (2, 446),
                "qualified_anchor_count": 445,
                "pool_size": 772602,
                "scientific_array_hash": "0de96f149b1ae0154c2befbdf69e0fcd912bcda0fad18f72a6bf9e93f2608910",
            },
            201: {
                "file": "artifacts/validation_201_support_pool.npz",
                "frame_range_inclusive": (0, 681),
                "anchor_range_inclusive": (2, 679),
                "qualified_anchor_count": 640,
                "pool_size": 1210186,
                "scientific_array_hash": "8865d0b65cc6814650213adcaff429a9ad75309871f99bee56936458b5249a7c",
            },
        }
        if set(support) != {"train/206", "train/201"}:
            raise ProtocolError("qualified support-pool roles changed")
        for sequence_id, expected_record in expected_support.items():
            record = _mapping(
                support[f"train/{sequence_id}"],
                f"support train/{sequence_id}",
            )
            if (
                record.get("file") != expected_record["file"]
                or tuple(record.get("frame_range_inclusive", ()))
                != expected_record["frame_range_inclusive"]
                or tuple(record.get("anchor_range_inclusive", ()))
                != expected_record["anchor_range_inclusive"]
                or record.get("qualified_anchor_count")
                != expected_record["qualified_anchor_count"]
                or record.get("pool_size") != expected_record["pool_size"]
                or record.get("scientific_array_hash")
                != expected_record["scientific_array_hash"]
            ):
                raise ProtocolError(
                    f"qualified support-pool metadata changed for train/{sequence_id}"
                )
            _sha256(
                record["sha256"],
                f"support train/{sequence_id}",
                pending_allowed=sequence_id == 201,
            )
        expected_artifact_paths = {
            "train_pool_manifest": "artifacts/data_v2/train_manifest.json",
            "validation_pool_manifest": "artifacts/data_v2/validation_manifest.json",
            "qualification": "artifacts/data_v2/qualification.json",
        }
        for key, expected_path in expected_artifact_paths.items():
            record = _mapping(artifacts[key], key)
            if record.get("file") != expected_path:
                raise ProtocolError(f"{key} path changed")
            digest = _sha256(record["sha256"], key, pending_allowed=True)
            if frozen and digest is None:
                raise ProtocolError(f"frozen schema 34 requires {key}")
        if (
            frozen
            and _mapping(support["train/201"], "support train/201")["sha256"] is None
        ):
            raise ProtocolError(
                "frozen schema 34 requires the full train/201 support pool"
            )

        qualification = _mapping(source["qualification"], "qualification")
        expected = "passed" if frozen else "pending_formal_generation_and_qualification"
        checks = qualification.get("required_checks")
        expected_checks = [
            "train_206_has_exactly_frames_0_through_448",
            "train_segments_are_disjoint_and_cover_0_through_448",
            "one_fixed_world_per_segment",
            "same_seed_repeats_bitwise",
            "different_formal_seeds_produce_different_physical_world_contents",
            "each_segment_frame_is_rendered_once",
            "every_window_is_inside_one_segment",
            "every_visible_return_appears_once_per_window",
            "coordinates_identity_and_labels_are_row_aligned",
            "current_frame_xyz_is_a_bitwise_copy",
            "historical_registration_meets_the_frozen_tolerance",
            "one_rendered_frame_is_bitwise_identical_across_overlapping_windows",
            "train_201_has_exactly_frames_0_through_681",
            "known_duplicate_prefix_is_retained_with_frozen_ray_runs",
            "normal_201_has_exactly_678_windows_and_unique_outputs_4_through_681",
            "all_world_parameters_seeds_boundaries_raw_identities_and_scientific_hashes_are_saved",
        ]
        if (
            qualification.get("model_independent") is not True
            or qualification.get("status") != expected
            or checks != expected_checks
        ):
            raise ProtocolError("qualification state contradicts the data-pool state")


def load_protocol(path: Path | str = DEFAULT_PROTOCOL_PATH) -> AJAEProtocol:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError("protocol is unreadable") from error
    return AJAEProtocol(_mapping(document, "protocol"), path=resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the AJAE schema-34 data contract"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    protocol = load_protocol(_parser().parse_args(argv).protocol)
    print(json.dumps(protocol.summary(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
