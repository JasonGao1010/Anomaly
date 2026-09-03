#!/usr/bin/env python3
"""Focused scientific-semantic tests for the sole AJAE schema-32 route.

Every fixture is small and synthetic. The suite never opens the real STU
sequences, renders a bank, loads the released network, or starts training.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import src.model as model_module
from src.evaluate import (
    AJAE_FUSION_VALUE,
    B0_FUSION_VALUE,
    FUSION_SEMANTICS,
    METHOD_FREEZE_FORMAT,
    METHOD_FREEZE_STATUS,
    AJAEInference,
    DevelopmentClipResult,
    DevelopmentFusedAP,
    EvaluationError,
    EvaluationIdentity,
    MethodFreezeRecord,
    PointMetricAccumulator,
    WindowScoreFusion,
    open_sealed_sequence,
)
from src.model import (
    GroupedKnnUpsample,
    GroupedRadiusKNN,
    GroupedVoxelPool,
    JointWindowPointTransformer,
    STUPointEncoding,
    assigned_stu_evidence,
    stu_input_identity,
)
from src.protocol import (
    R02_THRESHOLD_FORMAT,
    R02_VALIDATION_KEYS,
    SCHEMA_VERSION,
    WINDOW_MEMBER_OFFSETS,
    AJAEProtocol,
    ExperimentCondition,
    FrameSpan,
    ProtocolError,
    SequenceSpec,
    load_development_worlds,
    load_protocol,
    validate_confirmation_seal,
)
from src.qualify import joint_gradient_audit, scan_permutation_audit
from src.render import (
    DevelopmentClipWorld,
    HeldOutTorusShape,
    MaterialSpec,
    NormalTemplateShape,
    ObjectSpec,
    RayGrid,
    RenderError,
    SensorCalibration,
    ShapeSpec,
    WindowEntityDescriptor,
    WindowWorld,
    WorldGenerationReport,
    WorldSpec,
    match_window_entities,
    render_frame,
    save_development_worlds,
    source_observation_identity,
    window_matching_balance,
)
from src.scene import (
    LabelMode,
    PointLabels,
    SceneDataError,
    STUSequence,
    WindowReferencePose,
    assemble_window,
    canonical_ray_mapping_digest,
    make_source_frame,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocol.json"
DEVELOPMENT_PATH = ROOT / "dev.json"


def _labels(semantic: np.ndarray) -> PointLabels:
    values = np.asarray(semantic, dtype=np.uint16)
    target = np.full(values.shape, 255, dtype=np.uint8)
    target[values == 10] = 0
    return PointLabels(
        packed=values.astype(np.uint32),
        semantic=values,
        instance=np.zeros(values.shape, dtype=np.uint16),
        semantic_target=target,
    )


def _yaw_pose(angle: float, translation: tuple[float, float, float]) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = (
        (cosine, -sine, 0.0),
        (sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )
    pose[:3, 3] = translation
    return pose


def _window_sources() -> tuple[object, ...]:
    output = []
    for offset, frame_id in enumerate(range(10, 15)):
        xyzi = np.asarray(
            (
                (1.0 + 0.1 * offset, 0.2, 0.1, 0.20),
                (0.0, 0.0, 0.0, 0.00),
                (2.0, -0.3 + 0.02 * offset, 0.4, 0.35),
                (3.0, 0.4, -0.2 + 0.01 * offset, 0.50),
            ),
            dtype=np.float32,
        )
        pose = _yaw_pose(
            -0.10 + 0.05 * offset,
            (0.4 * offset, -0.15 * offset, 0.03 * offset),
        )
        output.append(
            make_source_frame(
                frame_id,
                xyzi,
                pose,
                _labels(np.asarray((10, 0, 10, 10), dtype=np.uint16)),
                partition="train",
                sequence_id=206,
            )
        )
    return tuple(output)


def _one_return_source(frame_id: int, distance: float, intensity: float) -> object:
    return make_source_frame(
        frame_id,
        np.asarray(
            ((distance, 0.0, 0.0, intensity), (0.0, 0.0, 0.0, 0.0)),
            dtype=np.float32,
        ),
        np.eye(4, dtype=np.float64),
        _labels(np.asarray((10, 0), dtype=np.uint16)),
        partition="train",
        sequence_id=206,
    )


def _stu_encoding_for(source: object) -> STUPointEncoding:
    return STUPointEncoding(
        point_features=torch.zeros((1, 128)),
        assigned_query=torch.zeros(1, dtype=torch.long),
        normal_evidence=torch.zeros((1, 19)),
        reliability_assign=torch.zeros(1),
        reliability_noobj=torch.zeros(1),
        maxlogit_score=torch.zeros(1),
        inverse_map=torch.zeros(1, dtype=torch.long),
        real_slots=torch.from_numpy(source.real_slots.astype(np.int64, copy=True)),
        input_identity=stu_input_identity(
            source.coordinates, source.features, source.real_slots
        ),
    )


def _descriptor(
    object_id: int,
    label: str,
    support: int,
    *,
    joint_voxels: int = 6,
    distance: float = 12.0,
) -> WindowEntityDescriptor:
    returns = (2, 2, 2, 2, 2)
    per_scan_voxels = (2, 2, 2, 2, 2)
    beam = [0] * 128
    beam[object_id % 128] = sum(returns)
    return WindowEntityDescriptor(
        object_id=object_id,
        label=label,  # type: ignore[arg-type]
        visible_returns_by_scan=returns,
        spatial_voxels_by_scan=per_scan_voxels,
        joint_visible_return_count=sum(returns),
        joint_spatial_voxel_count=joint_voxels,
        maximum_single_scan_spatial_voxel_count=max(per_scan_voxels),
        densification_gain=joint_voxels / max(per_scan_voxels),
        duplicate_fraction=1.0 - joint_voxels / sum(returns),
        median_distance_m=distance,
        occlusion_rate=0.2,
        support_semantic_id=support,
        visible_scan_count=5,
        minimum_visible_return_height_m=0.03,
        intensity_q05_median_q95=(0.1, 0.3, 0.7),
        beam_histogram=tuple(beam),
    )


def _stub_window(
    seed: str, descriptors: tuple[WindowEntityDescriptor, ...]
) -> WindowWorld:
    """Build only the validated interface needed to isolate matching logic."""

    item = object.__new__(WindowWorld)
    digest = hashlib.sha256(seed.encode("ascii")).hexdigest()
    sources = tuple(
        make_source_frame(
            frame_id,
            np.asarray(((4.0, 0.0, 0.0, 0.2),), dtype=np.float32),
            np.eye(4, dtype=np.float64),
            _labels(np.asarray((10,), dtype=np.uint16)),
            partition="train",
            sequence_id=201,
        )
        for frame_id in range(4, 9)
    )
    object.__setattr__(item, "window_start", 4)
    object.__setattr__(item, "frame_ids", (4, 5, 6, 7, 8))
    object.__setattr__(
        item,
        "world",
        SimpleNamespace(identity=digest, source_sequence_id=201),
    )
    object.__setattr__(item, "report", None)
    object.__setattr__(item, "renderer_identity", "a" * 64)
    object.__setattr__(item, "reference_pose", None)
    object.__setattr__(
        item,
        "rendered_frames",
        tuple(SimpleNamespace(source=source) for source in sources),
    )
    object.__setattr__(item, "descriptors", descriptors)
    return item


def _training_window(renderer_identity: str) -> object:
    from src.train import WindowTrainingData

    world = WorldSpec(3101, 206)
    report = WorldGenerationReport(
        world_seed=world.seed,
        source_sequence_id=world.source_sequence_id,
        world_type=world.world_type,
        world_attempt=0,
        normal_count=0,
        anomaly_count=0,
        count_seed=3102,
        label_order_seed=3103,
    )
    frame_ids = (0, 1, 2, 3, 4)
    sources = tuple(
        _one_return_source(frame_id, 4.0 + frame_id, 0.2) for frame_id in frame_ids
    )
    observation_identities = tuple(
        source_observation_identity(source) for source in sources
    )
    stu_identities = tuple(
        stu_input_identity(source.coordinates, source.features, source.real_slots)
        for source in sources
    )
    identity_payload = {
        "format": "ajae-window-world-v1",
        "world_identity": world.identity,
        "partition": "train",
        "sequence_id": 206,
        "window_start": 0,
        "frame_ids": frame_ids,
        "renderer_identity": renderer_identity,
        "source_observation_identities": observation_identities,
    }
    window_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        **identity_payload,
        "identity": window_identity,
        "world": world.to_dict(),
        "report": report.to_dict(),
        "descriptors": [],
    }
    groups = torch.arange(5, dtype=torch.long)
    generator = torch.Generator().manual_seed(3104)
    return WindowTrainingData(
        coordinates=torch.column_stack(
            (groups.float(), torch.zeros(5), torch.zeros(5))
        ),
        native_coordinates=torch.column_stack(
            (groups.float() + 10.0, torch.zeros(5), torch.zeros(5))
        ),
        scan_group=groups,
        stu_features=torch.randn((5, 128), generator=generator),
        normal_evidence=torch.randn((5, 19), generator=generator),
        reliability_assign=torch.rand(5, generator=generator),
        reliability_noobj=torch.rand(5, generator=generator),
        intensity=torch.rand(5, generator=generator),
        target=torch.zeros(5, dtype=torch.bool),
        valid=torch.ones(5, dtype=torch.bool),
        source_frame=groups,
        source_slot=torch.zeros(5, dtype=torch.long),
        source_ray=torch.zeros(5, dtype=torch.long),
        world_identity=world.identity,
        source_observation_identities=observation_identities,
        stu_input_identities=stu_identities,
        window_manifest=manifest,
    )


def _minimal_development_clips(
    protocol: AJAEProtocol, *, proxy_support: int = 10
) -> tuple[DevelopmentClipWorld, ...]:
    """Build the protocol-planned 24-clip population without rendering."""

    normal_shape = NormalTemplateShape(
        np.asarray(
            ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.3))
        ),
        np.empty((0, 3), dtype=np.int32),
        206,
        0,
        10,
        1,
        (0.0, 0.0, 0.0),
    )
    procedural_shape = ShapeSpec(
        ((0.2, 0.2, 0.2),),
        ((0.0, 0.0, 0.0),),
        ((1.0, 1.0),),
        (0.0,),
        ("union",),
    )
    material = MaterialSpec(0.5, 0.1)
    rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    renderer_identity = protocol.renderer_identity
    clips = []
    for plan in protocol.development_clip_plan():
        objects = (
            ObjectSpec(
                1,
                "normal-control",
                normal_shape,
                material,
                (8.0, 0.0, 0.0),
                rotation,
            ),
            ObjectSpec(
                2,
                "anomaly-proxy",
                procedural_shape,
                material,
                (10.0, 0.0, 0.0),
                rotation,
            ),
        )
        world = WorldSpec(int(plan["root_seed"]), 201, objects)
        report = WorldGenerationReport(
            world_seed=world.seed,
            source_sequence_id=201,
            world_type="mixed",
            world_attempt=0,
            normal_count=1,
            anomaly_count=1,
            count_seed=world.seed + 1,
            label_order_seed=world.seed + 2,
        )
        start = int(plan["clip_start"])
        frame_ids = tuple(int(value) for value in plan["source_frames"])
        sources = tuple(
            make_source_frame(
                frame_id,
                np.asarray(((4.0, 0.0, 0.0, 0.2),), dtype=np.float32),
                np.eye(4, dtype=np.float64),
                _labels(np.asarray((10,), dtype=np.uint16)),
                partition="train",
                sequence_id=201,
            )
            for frame_id in frame_ids
        )
        windows = []
        for offset in range(5):
            window = object.__new__(WindowWorld)
            members = sources[offset : offset + 5]
            object.__setattr__(window, "window_start", start + offset)
            object.__setattr__(
                window, "frame_ids", tuple(item.frame_id for item in members)
            )
            object.__setattr__(window, "world", world)
            object.__setattr__(window, "report", report)
            object.__setattr__(window, "renderer_identity", renderer_identity)
            object.__setattr__(window, "reference_pose", None)
            object.__setattr__(
                window,
                "rendered_frames",
                tuple(
                    SimpleNamespace(frame_id=item.frame_id, source=item)
                    for item in members
                ),
            )
            object.__setattr__(
                window,
                "descriptors",
                (
                    _descriptor(1, "normal-control", 10),
                    _descriptor(2, "anomaly-proxy", proxy_support),
                ),
            )
            windows.append(window)
        clips.append(
            DevelopmentClipWorld(
                start,
                tuple(item.frame_id for item in sources),
                world,
                report,
                renderer_identity,
                tuple(windows),
                "in_generator",
            )
        )
    return tuple(clips)


def _protocol_with_frozen_r02_thresholds(
    tmp_path: Path,
    *,
    name: str = "protocol.json",
    node: str = "R02",
    pilot_winners: dict[str, object] | None = None,
) -> AJAEProtocol:
    """Freeze harmless test thresholds so forged verdict content is inspected."""

    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    document["render"]["proxy_control_matching"]["thresholds"] = {
        "format": R02_THRESHOLD_FORMAT,
        "frozen_result_blind": True,
        "minimum_visual_reviewed_clips": 24,
        "minimum_matched_pairs": 10,
        "maximum_absolute_standardized_mean_difference": 0.25,
        "minimum_median_proxy_joint_visible_return_count": 2.0,
        "minimum_median_proxy_densification_gain": 1.1,
        "minimum_proxy_fraction_densification_gain_above_one": 0.5,
        "maximum_shortcut_balanced_accuracy_exclusive": 0.95,
        "maximum_shortcut_auroc_exclusive": 0.95,
    }
    thresholds = document["render"]["proxy_control_matching"]["thresholds"]
    document["status"]["r02_thresholds_frozen_result_blind"] = True
    document["status"]["r02_threshold_identity"] = hashlib.sha256(
        json.dumps(
            thresholds,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    document["status"]["current_node"] = node
    if node != "R02":
        document["status"]["r02_population_identity"] = "a" * 64
        document["status"]["r02_verdict_identity"] = "b" * 64
    state = (
        "R00",
        "R01",
        "R02",
        "R03",
        "R04",
        "R05",
        "F01",
        "G2",
        "G3",
        "S01",
        "M01",
        "V01",
        "T01",
    )
    milestones = (
        ("r03_qualification_identity", "R03", "c"),
        ("r04_training_qualification_identity", "R04", "d"),
        ("r05_freeze_identity", "R05", "e"),
        ("f01_formal_training_identity", "F01", "0"),
        ("g2_verdict_identity", "G2", "f"),
        ("g3_verdict_identity", "G3", "1"),
        ("s01_shift_audit_identity", "S01", "2"),
        ("m01_method_freeze_identity", "M01", "3"),
        ("v01_verdict_identity", "V01", "4"),
    )
    for field, milestone, digit in milestones:
        if state.index(node) > state.index(milestone):
            document["status"][field] = digit * 64
    if state.index(node) >= state.index("R04"):
        document["status"]["r04_training_bank_identity"] = "9" * 64
    if pilot_winners is not None:
        document["training"]["pilot"]["frozen_stage_winners"] = pilot_winners
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_protocol(path)


def test_protocol_is_the_only_schema32_b0_to_b3_contract() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol.schema_version == SCHEMA_VERSION == 32
    assert [item.value for item in ExperimentCondition] == ["B0", "B1", "B2", "B3"]
    assert set(protocol.experiments) == {"B0", "B1", "B2", "B3"}
    assert protocol.window_member_offsets == WINDOW_MEMBER_OFFSETS == (0, 1, 2, 3, 4)
    assert ExperimentCondition.B0.output_local_indices == (0,)
    for condition in (
        ExperimentCondition.B1,
        ExperimentCondition.B2,
        ExperimentCondition.B3,
    ):
        assert condition.input_member_indices == WINDOW_MEMBER_OFFSETS
        assert condition.output_local_indices == WINDOW_MEMBER_OFFSETS
    assert protocol.status["formal_training_allowed"] is False
    assert protocol.model["input_dim"] == 150
    assert {"relative_time", "absolute_time", "time_embedding"}.issubset(
        protocol.model["forbidden_features"]
    )


def test_schema30_protocol_and_development_payload_are_rejected_early(
    tmp_path: Path,
) -> None:
    old_protocol = tmp_path / "protocol30.json"
    old_protocol.write_text(json.dumps({"schema_version": 30}), encoding="utf-8")
    with pytest.raises(ProtocolError, match="schema 30 is retired"):
        load_protocol(old_protocol)

    protocol = load_protocol(PROTOCOL_PATH)
    old_development = tmp_path / "dev30.json"
    old_development.write_text(
        json.dumps({"format": "ajae-development-worlds-v2", "protocol_schema": 30}),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="schema-30 centered"):
        load_development_worlds(old_development, protocol=protocol)


def test_training_entry_rejects_schema30_before_full_loader_or_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.train as train_module

    old_protocol = tmp_path / "protocol30.json"
    old_protocol.write_text(json.dumps({"schema_version": 30}), encoding="utf-8")
    calls = 0

    def forbidden_loader(_path: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("the full protocol loader must not see schema 30")

    monkeypatch.setattr(train_module, "load_protocol", forbidden_loader)
    bank = tmp_path / "bank-must-not-be-opened"
    with pytest.raises(train_module.TrainingError, match="requires schema 32"):
        train_module.run_training(
            protocol_path=old_protocol,
            bank_path=bank,
            output_directory=tmp_path / "output",
            mode="formal",
            condition="B3",
            seed=3101,
            device="cpu",
            config=None,  # type: ignore[arg-type]
        )
    assert calls == 0
    assert not bank.exists()


@pytest.mark.parametrize(
    ("mode", "seed", "message"),
    (
        ("tiny_overfit", 1001, "permitted only during R04"),
        ("pilot", 1001, "permitted only during R04"),
        ("formal", 0, "completed R05 freeze"),
    ),
)
def test_schema32_training_is_blocked_before_bank_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    seed: int,
    message: str,
) -> None:
    import src.train as train_module

    calls = 0

    def forbidden_bank(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("a disabled formal route must not read the bank")

    monkeypatch.setattr(train_module, "load_window_train_bank", forbidden_bank)
    with pytest.raises(train_module.TrainingError, match=message):
        train_module.run_training(
            protocol_path=PROTOCOL_PATH,
            bank_path=tmp_path / "bank-must-not-be-opened",
            output_directory=tmp_path / "output",
            mode=mode,
            condition="B3",
            seed=seed,
            device="cpu",
            config=None,  # type: ignore[arg-type]
        )
    assert calls == 0


def test_pilot_stages_bind_prior_winners_and_fixed_run_parameters(
    tmp_path: Path,
) -> None:
    from src.train import TrainingConfig, TrainingError, validate_training_request

    empty_winners = {
        "learning_rate": None,
        "gradient_accumulation": None,
        "scheduler": None,
        "weight_decay": None,
    }
    first_protocol = _protocol_with_frozen_r02_thresholds(
        tmp_path,
        name="pilot-first.json",
        node="R04",
        pilot_winners=empty_winners,
    )
    first = TrainingConfig(
        learning_rate=0.0001,
        gradient_accumulation=1,
        scheduler="constant",
        weight_decay=0.0,
        maximum_updates=50,
        evaluation_interval_updates=50,
        gradient_clip_norm=None,
    )
    request = validate_training_request(
        first_protocol,
        "pilot",
        "B3",
        1001,
        first,
        bank_size=128,
        pilot_stage="learning_rate_and_batch",
    )
    assert request.pilot_stage == "learning_rate_and_batch"
    with pytest.raises(TrainingError, match="evaluation_interval_updates differs"):
        validate_training_request(
            first_protocol,
            "pilot",
            "B3",
            1001,
            replace(first, evaluation_interval_updates=25),
            bank_size=128,
            pilot_stage="learning_rate_and_batch",
        )
    with pytest.raises(TrainingError, match="requires the frozen learning-rate"):
        validate_training_request(
            first_protocol,
            "pilot",
            "B3",
            1002,
            replace(
                first,
                maximum_updates=600,
                scheduler="five_percent_warmup_cosine",
            ),
            bank_size=128,
            pilot_stage="scheduler",
        )

    selected_batch = {
        **empty_winners,
        "learning_rate": 0.0001,
        "gradient_accumulation": 2,
    }
    scheduler_protocol = _protocol_with_frozen_r02_thresholds(
        tmp_path,
        name="pilot-scheduler.json",
        node="R04",
        pilot_winners=selected_batch,
    )
    scheduler = replace(
        first,
        gradient_accumulation=2,
        scheduler="five_percent_warmup_cosine",
        maximum_updates=600,
    )
    validate_training_request(
        scheduler_protocol,
        "pilot",
        "B3",
        1002,
        scheduler,
        bank_size=128,
        pilot_stage="scheduler",
    )
    with pytest.raises(TrainingError, match="frozen learning-rate/batch winner"):
        validate_training_request(
            scheduler_protocol,
            "pilot",
            "B3",
            1002,
            replace(scheduler, learning_rate=0.0003),
            bank_size=128,
            pilot_stage="scheduler",
        )

    selected_scheduler = {
        **selected_batch,
        "scheduler": "five_percent_warmup_cosine",
    }
    weight_decay_protocol = _protocol_with_frozen_r02_thresholds(
        tmp_path,
        name="pilot-weight-decay.json",
        node="R04",
        pilot_winners=selected_scheduler,
    )
    weight_decay = replace(scheduler, weight_decay=0.00001)
    validate_training_request(
        weight_decay_protocol,
        "pilot",
        "B3",
        1002,
        weight_decay,
        bank_size=128,
        pilot_stage="weight_decay",
    )
    with pytest.raises(TrainingError, match="frozen scheduler-stage winners"):
        validate_training_request(
            weight_decay_protocol,
            "pilot",
            "B3",
            1002,
            replace(weight_decay, scheduler="constant"),
            bank_size=128,
            pilot_stage="weight_decay",
        )


def test_training_budget_uses_exactly_one_unit() -> None:
    from src.train import TrainingConfig, TrainingError

    with pytest.raises(TrainingError, match="maximum_updates only"):
        TrainingConfig(
            learning_rate=1.0e-4,
            gradient_accumulation=1,
            epochs=1,
            maximum_updates=10,
        )
    with pytest.raises(TrainingError, match="epochs only"):
        TrainingConfig(
            learning_rate=1.0e-4,
            gradient_accumulation=1,
            budget_unit="epochs",
            epochs=1,
            maximum_updates=10,
        )


def test_all_formal_results_are_authorized_only_during_f01() -> None:
    from src.train import (
        TrainMode,
        TrainingConfig,
        TrainingError,
        _validate_result_request,
    )

    base = load_protocol(PROTOCOL_PATH)
    training = base.plain_document()["training"]
    config = TrainingConfig(
        learning_rate=0.0001,
        gradient_accumulation=1,
        weight_decay=0.0,
        scheduler="constant",
        budget_unit="epochs",
        epochs=1,
        maximum_updates=None,
        evaluation_interval_updates=50,
        gradient_clip_norm=None,
    )
    training["formal"]["recipe"] = {  # type: ignore[index]
        "learning_rate": 0.0001,
        "gradient_accumulation": 1,
        "weight_decay": 0.0,
        "scheduler": "constant",
        "budget_unit": "epochs",
        "epochs": 1,
        "evaluation_interval_updates": 50,
        "gradient_clip_norm": None,
    }
    protocol = SimpleNamespace(
        training=training,
        status={
            "r04_training_qualification_identity": "d" * 64,
            "r05_freeze_identity": "e" * 64,
            "f01_formal_training_identity": "0" * 64,
            "g2_verdict_identity": "f" * 64,
        },
    )
    common = {
        "protocol": protocol,
        "mode": TrainMode.FORMAL,
        "seed": 0,
        "config": config,
        "pilot_stage": None,
        "window_count": 445,
        "maximum_updates": 445,
        "r04_identity": "d" * 64,
        "r05_identity": "e" * 64,
    }
    with pytest.raises(TrainingError, match="R05-authorized request"):
        _validate_result_request(
            **common,
            condition=ExperimentCondition.B2,
            authorization_node="G3",
            g2_identity=None,
        )
    _validate_result_request(
        **common,
        condition=ExperimentCondition.B2,
        authorization_node="F01",
        g2_identity=None,
    )
    _validate_result_request(
        **common,
        condition=ExperimentCondition.B1,
        authorization_node="F01",
        g2_identity=None,
    )


def test_formal_gate_cannot_open_with_pending_decision_criteria(
    tmp_path: Path,
) -> None:
    base = _protocol_with_frozen_r02_thresholds(tmp_path)
    document = base.plain_document()
    document["status"].update(
        {
            "current_node": "G2",
            "r02_population_identity": "a" * 64,
            "r02_verdict_identity": "b" * 64,
            "r03_qualification_identity": "c" * 64,
            "r04_training_bank_identity": "9" * 64,
            "r04_training_qualification_identity": "d" * 64,
            "r05_freeze_identity": "e" * 64,
            "f01_formal_training_identity": "0" * 64,
            "formal_training_allowed": False,
        }
    )
    winners = {
        "learning_rate": 0.0001,
        "gradient_accumulation": 2,
        "scheduler": "five_percent_warmup_cosine",
        "weight_decay": 0.00001,
    }
    document["training"]["pilot"]["frozen_stage_winners"] = winners
    document["training"]["formal"].update(
        {
            "recipe_status": "frozen_result_blind_in_R05",
            "recipe": {
                **winners,
                "budget_unit": "epochs",
                "epochs": 1,
                "evaluation_interval_updates": 50,
                "gradient_clip_norm": None,
            },
            "bank_identity": "9" * 64,
            "development_population_identity": "a" * 64,
        }
    )
    path = tmp_path / "g2-with-pending-criteria.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        ProtocolError, match="criteria must be frozen before formal runs"
    ):
        load_protocol(path)


def test_frozen_sequence_spans_produce_exact_legal_window_counts() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    training = protocol.normal_training.legal_window_starts()
    development = protocol.development_sequence.legal_window_starts()
    assert len(training) == 445 and training == tuple(range(0, 445))
    assert len(development) == 674 and development == tuple(range(4, 678))
    assert protocol.window_frame_ids("train", 206, 444) == (444, 445, 446, 447, 448)
    assert protocol.window_frame_ids("train", 201, 677) == (677, 678, 679, 680, 681)


def test_schema32_generation_plans_are_unique_and_have_frozen_composition() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    bank = protocol.training_bank_plan()
    assert len(bank) == 445
    assert {int(item["source_frames"][0]) for item in bank} == set(range(445))
    assert len({int(item["root_seed"]) for item in bank}) == 445
    assert {
        world_type: sum(item["world_type"] == world_type for item in bank)
        for world_type in ("pure_normal", "control_only", "mixed", "anomaly_only")
    } == {
        "pure_normal": 89,
        "control_only": 89,
        "mixed": 178,
        "anomaly_only": 89,
    }

    development = protocol.development_clip_plan()
    confirmation = protocol.confirmation_clip_plan()
    held_out = protocol.held_out_synthetic_shift_plan()
    assert tuple(item["position"] for item in development) == tuple(range(24))
    assert tuple(item["position"] for item in confirmation) == tuple(range(24, 48))
    assert tuple(item["position"] for item in held_out) == tuple(range(48, 54))
    development_frames = {
        frame for item in development for frame in item["source_frames"]
    }
    confirmation_frames = {
        frame for item in confirmation for frame in item["source_frames"]
    }
    assert development_frames.isdisjoint(confirmation_frames)
    assert {item["mechanism"] for item in development} == {"in_generator"}
    assert {item["mechanism"] for item in confirmation} == {"in_generator"}
    assert {item["mechanism"] for item in held_out} == {"torus_SDF"}
    all_frames = [
        int(frame)
        for item in (*development, *confirmation, *held_out)
        for frame in item["source_frames"]
    ]
    assert len(all_frames) == len(set(all_frames)) == 54 * 9
    assert (
        len(
            {
                int(item["root_seed"])
                for item in (*development, *confirmation, *held_out)
            }
        )
        == 54
    )


def test_confirmation_population_is_sealed_before_formal_gate() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    seal = validate_confirmation_seal(ROOT / "confirm.json", protocol=protocol)
    confirmation = load_development_worlds(
        ROOT / "confirm.json", protocol=protocol, population_role="confirmation"
    )
    assert seal["status"] == "sealed_not_generated_before_G2"
    assert seal["clip_count"] == 0
    assert confirmation.status == "sealed_not_generated_before_G2"
    assert confirmation.clips == ()


def test_formal_development_generation_rejects_placeholder_thresholds(
    tmp_path: Path,
) -> None:
    from src.train import TrainingError, generate_development_worlds

    with pytest.raises(TrainingError, match="freeze result-blind R02 thresholds"):
        generate_development_worlds(
            protocol_path=PROTOCOL_PATH,
            data_root=tmp_path / "must-not-be-opened",
            destination=tmp_path / "must-not-be-written.json",
        )
    assert not (tmp_path / "must-not-be-written.json").exists()

    with pytest.raises(TrainingError, match="confirmation population.*G2"):
        generate_development_worlds(
            protocol_path=PROTOCOL_PATH,
            data_root=tmp_path / "must-not-be-opened",
            destination=tmp_path / "confirm-must-not-be-written.json",
            population_role="confirmation",
        )


def test_empty_schema32_development_manifest_is_explicitly_not_evidence() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    development = load_development_worlds(DEVELOPMENT_PATH, protocol=protocol)
    assert development.format == "ajae-development-window-worlds-v3"
    assert development.status == "not_generated_R02"
    assert development.clips == development.windows == ()
    assert not development.validated


def test_old_empty_placeholder_can_be_replaced_once_but_definitions_cannot(
    tmp_path: Path,
) -> None:
    protocol = _protocol_with_frozen_r02_thresholds(tmp_path)
    clips = _minimal_development_clips(protocol)
    path = tmp_path / "development.json"
    path.write_text(
        json.dumps(
            {
                "format": "ajae-development-window-worlds-v3",
                "protocol_schema": 32,
                "protocol_identity": "0" * 64,
                "plan_identity": "1" * 64,
                "population_identity": None,
                "sequence_id": 201,
                "status": "not_generated_R02",
                "validation": {},
                "clip_count": 0,
                "window_count": 0,
                "clips": [],
                "scientific_verdict": None,
            }
        ),
        encoding="utf-8",
    )

    save_development_worlds(
        path,
        clips,
        protocol_identity=protocol.development_population_identity,
        plan_identity=protocol.development_clip_plan_identity,
    )
    replaced = load_development_worlds(path, protocol=protocol)
    assert replaced.status == "definitions_only_unvalidated"
    assert replaced.protocol_identity == protocol.development_population_identity
    assert replaced.plan_identity == protocol.development_clip_plan_identity
    assert len(replaced.clips) == 24

    with pytest.raises(RenderError, match="refusing to replace"):
        save_development_worlds(
            path,
            clips,
            protocol_identity=protocol.development_population_identity,
            plan_identity=protocol.development_clip_plan_identity,
        )


def test_boolean_checks_cannot_forge_a_validated_development_freeze(
    tmp_path: Path,
) -> None:
    protocol = _protocol_with_frozen_r02_thresholds(tmp_path)
    path = tmp_path / "development.json"
    save_development_worlds(
        path,
        _minimal_development_clips(protocol),
        protocol_identity=protocol.development_population_identity,
        plan_identity=protocol.development_clip_plan_identity,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "definitions_only_unvalidated"
    assert payload["validation"] == {}
    assert payload["scientific_verdict"] is None
    assert not load_development_worlds(path, protocol=protocol).validated

    payload["status"] = "validated_frozen"
    payload["validation"] = dict.fromkeys(R02_VALIDATION_KEYS, True)
    payload["scientific_verdict"] = dict.fromkeys(R02_VALIDATION_KEYS, True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="R02 scientific verdict keys differ"):
        load_development_worlds(path, protocol=protocol)


def test_uncomputable_r02_matching_is_persisted_as_an_irreversible_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.train as train_module

    protocol = _protocol_with_frozen_r02_thresholds(tmp_path)
    clips = _minimal_development_clips(protocol, proxy_support=11)
    path = tmp_path / "development.json"
    save_development_worlds(
        path,
        clips,
        protocol_identity=protocol.development_population_identity,
        plan_identity=protocol.development_clip_plan_identity,
    )
    definitions = load_development_worlds(path, protocol=protocol)
    review_path = tmp_path / "visual-review.json"
    review_path.write_text(
        json.dumps(
            {
                "format": "ajae-r02-visual-review-v1",
                "population_identity": definitions.population_identity,
                "reviewed_clip_identities": [
                    item.identity for item in definitions.clips
                ],
                "reviewer": "test reviewer",
                "reviewed_at_utc": "2026-09-03T00:00:00Z",
                "checklist": {
                    "world_fixed_before_all_scans": True,
                    "no_synthetic_point_completion": True,
                    "no_bottom_return_insertion": True,
                    "no_scan_duplication_or_copying": True,
                    "placements_visually_plausible": True,
                    "returns_and_occlusion_visually_plausible": True,
                },
                "findings": "Synthetic fixture review.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_module,
        "_render_development_definitions",
        lambda *_args, **_kwargs: clips,
    )

    adjudicated = train_module.finalize_r02_development_worlds(
        protocol_path=protocol.path,
        development_worlds_path=path,
        data_root=tmp_path / "unused",
        visual_review_path=review_path,
    )
    assert adjudicated.status == "adjudicated_failed_R02"
    assert adjudicated.adjudicated and not adjudicated.validated
    assert adjudicated.scientific_verdict is not None
    assert adjudicated.scientific_verdict["decision"] == "fail"
    matching = adjudicated.scientific_verdict["matching"]
    assert matching["status"] == "not_computable"
    assert matching["failure_code"] == "matching_not_computable"
    assert matching["passed"] is False
    assert adjudicated.scientific_verdict["shortcut_audit"]["failure_code"] == (
        "matching_not_computable"
    )

    with pytest.raises(train_module.TrainingError, match="unadjudicated"):
        train_module.finalize_r02_development_worlds(
            protocol_path=protocol.path,
            development_worlds_path=path,
            data_root=tmp_path / "unused",
            visual_review_path=review_path,
        )


def test_symmetric_reference_pose_is_permutation_deterministic_and_rigid() -> None:
    poses = tuple(
        _yaw_pose(0.04 * index - 0.08, (index * 0.5, -index * 0.2, index * 0.03))
        for index in range(5)
    )
    expected = WindowReferencePose.from_sensor_poses(poses)
    observed = WindowReferencePose.from_sensor_poses(
        tuple(poses[index] for index in (3, 0, 4, 1, 2))
    )
    np.testing.assert_array_equal(observed.rotation, expected.rotation)
    np.testing.assert_array_equal(observed.translation, expected.translation)
    np.testing.assert_allclose(
        expected.rotation.T @ expected.rotation, np.eye(3), atol=1e-12
    )
    assert np.linalg.det(expected.rotation) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(
        expected.world_from_window @ expected.window_from_world,
        np.eye(4),
        atol=1e-12,
    )


def test_symmetric_reference_pose_is_equivariant_to_a_global_rigid_transform() -> None:
    poses = tuple(
        _yaw_pose(0.03 * index, (0.7 * index, 0.1 * index, -0.02 * index))
        for index in range(5)
    )
    global_from_world = _yaw_pose(0.37, (4.0, -3.0, 1.5))
    reference = WindowReferencePose.from_sensor_poses(poses)
    transformed = WindowReferencePose.from_sensor_poses(
        tuple(global_from_world @ pose for pose in poses)
    )
    np.testing.assert_allclose(
        transformed.world_from_window,
        global_from_world @ reference.world_from_window,
        atol=2e-12,
        rtol=2e-12,
    )


def test_assemble_window_recovers_every_point_under_scan_permutation() -> None:
    spec = SequenceSpec("train", 206, "fixture", True, FrameSpan(0, 449))
    sources = _window_sources()
    ordered = assemble_window(spec, 10, (10, 11, 12, 13, 14), sources)
    shuffled = assemble_window(
        spec,
        10,
        (10, 11, 12, 13, 14),
        tuple(sources[index] for index in (3, 0, 4, 1, 2)),
    )
    assert ordered.points.count == shuffled.points.count == 15
    assert not hasattr(ordered.points, "frame_offsets")
    np.testing.assert_array_equal(
        ordered.reference_pose.world_from_window,
        shuffled.reference_pose.world_from_window,
    )

    def rows(window: object) -> dict[tuple[int, int], tuple[np.ndarray, int, int]]:
        points = window.points
        return {
            (int(frame), int(slot)): (coordinate, int(group), int(ray))
            for coordinate, group, frame, slot, ray in zip(
                points.coordinates,
                points.scan_group,
                points.source_frame,
                points.source_slot,
                points.source_ray,
                strict=True,
            )
        }

    first, second = rows(ordered), rows(shuffled)
    assert first.keys() == second.keys()
    for identity in first:
        np.testing.assert_allclose(first[identity][0], second[identity][0], atol=1e-7)
        assert first[identity][1:] == second[identity][1:]

    restored = ordered.restore_source_frame(12, ordered.points.source_frame)
    np.testing.assert_array_equal(restored, np.asarray((12, 0, 12, 12), dtype=np.int32))
    point = ordered.points.point_id(0)
    assert (point.frame_id, point.ray.beam_id, point.ray.azimuth_column) == (10, 0, 0)


def test_canonical_ray_mapping_and_round_trip_preserve_physical_identity() -> None:
    mapping = np.arange(128 * 1024, dtype=np.int32)
    digest = canonical_ray_mapping_digest(mapping)
    assert len(digest) == 64
    changed = mapping.copy()
    changed[0], changed[1] = changed[1], changed[0]
    assert canonical_ray_mapping_digest(changed) != digest

    directions = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    origins = np.asarray(((0.1, 0.0, 0.05), (0.0, 0.1, 0.05)))
    grid = RayGrid(
        directions,
        np.zeros(1),
        np.asarray((0.0, -np.pi / 2.0)),
        beam_count=1,
        origins_sensor=origins,
    )
    distances = np.asarray((5.0, 7.0))
    xyzi = np.zeros((2, 4), dtype=np.float32)
    xyzi[:, :3] = (origins + distances[:, None] * directions).astype(np.float32)
    frame = make_source_frame(
        0,
        xyzi,
        np.eye(4, dtype=np.float64),
        _labels(np.full(2, 10, dtype=np.uint16)),
        partition="fixture",
        sequence_id=99,
    )
    np.testing.assert_allclose(grid.ranges(frame), distances, atol=1e-7)
    np.testing.assert_allclose(grid.points_from_ranges(distances, frame), xyzi[:, :3])
    assert grid.round_trip(frame)["maximum_point_error_m"] < 1e-7


def test_common_renderer_is_deterministic_and_uses_nearest_first_return() -> None:
    xyzi = np.asarray(((5.0, 0.0, 0.0, 0.2),), dtype=np.float32)
    frame = make_source_frame(
        0,
        xyzi,
        np.eye(4, dtype=np.float64),
        _labels(np.asarray((10,), dtype=np.uint16)),
        partition="fixture",
        sequence_id=99,
    )
    grid = RayGrid(
        np.asarray(((1.0, 0.0, 0.0),)),
        np.asarray((0.0,)),
        np.asarray((0.0,)),
        beam_count=1,
    )
    shape = ShapeSpec(
        ((0.5, 0.5, 0.5),),
        ((0.0, 0.0, 0.0),),
        ((1.0, 1.0),),
        (0.0,),
        ("union",),
    )
    material = MaterialSpec(0.5, 0.1)
    rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    farther = ObjectSpec(1, "anomaly-proxy", shape, material, (4.0, 0.0, 0.0), rotation)
    nearer = ObjectSpec(2, "anomaly-proxy", shape, material, (3.0, 0.0, 0.0), rotation)
    world = WorldSpec(31, 99, (farther, nearer))
    sensor = SensorCalibration.constant(0.4)
    first = render_frame(frame, world, grid, sensor)
    second = render_frame(frame, world, grid, sensor)
    np.testing.assert_array_equal(first.source.xyzi, second.source.xyzi)
    np.testing.assert_array_equal(first.packed_labels, second.packed_labels)
    assert first.object_id_internal.tolist() == [2]
    assert first.anomaly_proxy_mask.tolist() == [True]
    assert first.occluded_original_mask.tolist() == [True]
    assert float(first.source.xyzi[0, 0]) == pytest.approx(2.5, abs=2e-4)


def test_window_descriptor_enforces_observed_density_formulas() -> None:
    descriptor = _descriptor(1, "normal-control", 10)
    assert descriptor.densification_gain == pytest.approx(3.0)
    assert descriptor.duplicate_fraction == pytest.approx(0.4)
    payload = descriptor.to_dict()
    payload["densification_gain"] = 2.9
    with pytest.raises(RenderError, match="densification gain"):
        WindowEntityDescriptor(**payload)  # type: ignore[arg-type]
    payload = descriptor.to_dict()
    payload["duplicate_fraction"] = 0.3
    with pytest.raises(RenderError, match="duplicate fraction"):
        WindowEntityDescriptor(**payload)  # type: ignore[arg-type]


def test_window_matching_is_exact_within_support_semantic_strata() -> None:
    controls = _stub_window(
        "controls",
        (
            _descriptor(1, "normal-control", 10, joint_voxels=5, distance=20.0),
            _descriptor(2, "normal-control", 11, joint_voxels=7, distance=8.0),
        ),
    )
    proxies = _stub_window(
        "proxies",
        (
            _descriptor(3, "anomaly-proxy", 10, joint_voxels=7, distance=8.1),
            _descriptor(4, "anomaly-proxy", 11, joint_voxels=5, distance=19.9),
        ),
    )
    pairs = match_window_entities((controls, proxies))
    assert len(pairs) == 2
    assert {item.support_semantic_id for item in pairs} == {10, 11}
    assert all(
        next(
            descriptor.support_semantic_id
            for descriptor in controls.descriptors
            if descriptor.object_id == item.control_object_id
        )
        == next(
            descriptor.support_semantic_id
            for descriptor in proxies.descriptors
            if descriptor.object_id == item.proxy_object_id
        )
        == item.support_semantic_id
        for item in pairs
    )
    balance = window_matching_balance(pairs)
    assert balance["exact_matching_stratum"] == "support_semantic_id"
    assert balance["pair_count"] == 2


def test_assigned_stu_evidence_uses_one_minimum_index_query() -> None:
    logits = torch.zeros(100, 20)
    masks = torch.full((2, 100), -20.0)
    logits[0, 0] = logits[1, 1] = 5.0
    masks[:, :2] = 2.0
    evidence = assigned_stu_evidence(logits, masks)
    torch.testing.assert_close(
        evidence.assigned_query, torch.zeros(2, dtype=torch.long)
    )
    probability = logits.softmax(dim=1)
    mask_probability = torch.sigmoid(torch.tensor(2.0))
    assignment = mask_probability * probability[0, :19].max()
    torch.testing.assert_close(evidence.reliability_assign, assignment.expand(2))
    torch.testing.assert_close(
        evidence.normal_evidence,
        (mask_probability * probability[0, :19]).expand(2, 19),
    )
    torch.testing.assert_close(evidence.reliability_noobj, probability[0, 19].expand(2))


def test_reused_stu_encoding_is_rejected_when_content_changes_but_slots_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.train as train_module

    sources = tuple(
        _one_return_source(frame_id, 4.0 + frame_id, 0.1 + 0.05 * frame_id)
        for frame_id in range(5)
    )
    assert all(
        np.array_equal(source.real_slots, sources[0].real_slots) for source in sources
    )
    encoding = _stu_encoding_for(sources[0])
    assert encoding.input_identity != stu_input_identity(
        sources[1].coordinates, sources[1].features, sources[1].real_slots
    )

    reference = WindowReferencePose.from_sensor_poses(
        tuple(source.lidar_pose for source in sources)
    )
    window = object.__new__(WindowWorld)
    object.__setattr__(window, "window_start", 0)
    object.__setattr__(window, "frame_ids", (0, 1, 2, 3, 4))
    object.__setattr__(
        window,
        "rendered_frames",
        tuple(SimpleNamespace(source=source) for source in sources),
    )
    object.__setattr__(window, "reference_pose", reference)
    monkeypatch.setattr(
        train_module,
        "assemble_window",
        lambda *_args, **_kwargs: SimpleNamespace(
            labels=object(), reference_pose=reference
        ),
    )
    with pytest.raises(train_module.TrainingError, match="rendered frame 1"):
        train_module.window_training_data(
            window,
            {frame_id: encoding for frame_id in range(5)},
            canonical_ray_by_slot=np.arange(2, dtype=np.int32),
            ray_mapping_digest="0" * 64,
            protocol=load_protocol(PROTOCOL_PATH),
        )

    class ReusedEncoding(nn.Module):
        def forward(self, *_args: object) -> STUPointEncoding:
            return encoding

    inference = AJAEInference._for_test(
        None,
        ReusedEncoding(),
        condition="B0",
        slot_to_ray=lambda source: np.arange(source.slot_count, dtype=np.int32),
    )
    assert inference._encode(sources[0]) is encoding
    with pytest.raises(EvaluationError, match="different source-frame content"):
        inference._encode(sources[1])


def test_model_has_one_time_free_window_transformer_class() -> None:
    transformer_classes = [
        name
        for name, value in vars(model_module).items()
        if inspect.isclass(value)
        and issubclass(value, nn.Module)
        and "Transformer" in name
    ]
    assert transformer_classes == ["JointWindowPointTransformer"]
    parameters = tuple(
        inspect.signature(JointWindowPointTransformer.forward).parameters
    )
    assert parameters == (
        "self",
        "coordinates",
        "stu_features",
        "normal_evidence",
        "reliability_assign",
        "reliability_noobj",
        "intensity",
        "scan_group",
        "grouping_mode",
    )
    source = inspect.getsource(JointWindowPointTransformer.forward)
    assert not any(name in source for name in ("time", "frame_id", "member_index"))
    model = JointWindowPointTransformer.from_protocol(load_protocol(PROTOCOL_PATH))
    assert not any("time" in name.lower() for name in model.state_dict())


def test_b2_b3_voxel_keys_and_population_are_semantically_distinct() -> None:
    torch.manual_seed(31)
    pool = GroupedVoxelPool(hidden_dim=4, voxel_size=1.0)
    coordinates = torch.asarray(((0.10, 0.0, 0.0), (0.20, 0.0, 0.0)))
    features = torch.asarray(((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)))
    groups = torch.asarray((0, 1), dtype=torch.long)
    b2 = pool(features, coordinates, groups, grouping_mode="per_scan")
    b3 = pool(features, coordinates, groups, grouping_mode="joint")
    assert b2.coordinates.shape[0] == 2
    assert b2.population.tolist() == [1, 1]
    assert b3.coordinates.shape[0] == 1
    assert b3.population.tolist() == [2]
    assert b3.inverse_map.tolist() == [0, 0]


def test_b3_radius_uses_cross_scan_points_and_reports_uncapped_count() -> None:
    coordinates = torch.asarray(
        ((0.00, 0.0, 0.0), (0.08, 0.0, 0.0), (0.12, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    groups = torch.asarray((0, 1, 1, 0), dtype=torch.long)
    neighborhood = GroupedRadiusKNN(radius=0.20, k=2, workers=1)
    joint_neighbor, joint_valid, joint_count = neighborhood(
        coordinates, groups, grouping_mode="joint"
    )
    isolated_neighbor, isolated_valid, isolated_count = neighborhood(
        coordinates, groups, grouping_mode="per_scan"
    )
    assert int(joint_count[0]) == 3 > neighborhood.k
    assert int(joint_valid[0].sum()) == neighborhood.k
    assert {1, 2} & set(joint_neighbor[0, joint_valid[0]].tolist())
    assert int(isolated_count[0]) == 1
    assert isolated_neighbor[0, isolated_valid[0]].tolist() == [0]


def test_b3_upsampling_can_select_the_nearest_cross_scan_source() -> None:
    upsample = GroupedKnnUpsample(k=1, workers=1)
    source_features = torch.asarray(((2.0,), (9.0,)))
    source_coordinates = torch.asarray(((3.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    source_groups = torch.asarray((0, 1), dtype=torch.long)
    target_coordinates = torch.asarray(((0.01, 0.0, 0.0),))
    target_groups = torch.asarray((0,), dtype=torch.long)
    joint = upsample(
        source_features,
        source_coordinates,
        source_groups,
        target_coordinates,
        target_groups,
        grouping_mode="joint",
    )
    isolated = upsample(
        source_features,
        source_coordinates,
        source_groups,
        target_coordinates,
        target_groups,
        grouping_mode="per_scan",
    )
    torch.testing.assert_close(joint, torch.asarray(((9.0,),)))
    torch.testing.assert_close(isolated, torch.asarray(((2.0,),)))


def test_b2_and_b3_models_are_scan_block_permutation_equivariant() -> None:
    from src.qualify import _model_inputs, _tiny_model

    result = scan_permutation_audit()
    assert result["passed"] is True
    assert result["points"] == 20
    assert result["maximum_logit_error"] <= 2e-6

    model = _tiny_model().eval()
    inputs = _model_inputs()
    order = (3, 0, 4, 1, 2)
    permutation = torch.cat(
        tuple(torch.nonzero(inputs[-1] == group).flatten() for group in order)
    )
    relabel = torch.empty(5, dtype=torch.long)
    for new_group, old_group in enumerate(order):
        relabel[old_group] = new_group
    with torch.no_grad():
        expected = model(*inputs, grouping_mode="per_scan")
        observed = model(
            *(value[permutation] for value in inputs[:-1]),
            relabel[inputs[-1][permutation]],
            grouping_mode="per_scan",
        )
    torch.testing.assert_close(observed, expected[permutation], atol=2e-6, rtol=0.0)


def test_joint_model_scores_all_points_and_backpropagates_all_levels() -> None:
    result = joint_gradient_audit()
    assert result["passed"] is True
    assert result["scored_points"] == 20
    assert result["minimum_point_gradient_l1"] > 0.0
    assert len(result["hierarchy_gradient_l1"]) == 4
    assert all(value > 0.0 for value in result["hierarchy_gradient_l1"])


def test_effective_batch_bce_balances_after_all_windows_and_handles_empty_class() -> (
    None
):
    from src.train import TrainingError, effective_batch_balanced_bce

    positive_logits = torch.asarray((0.0, 2.0))
    negative_logits = torch.asarray((-1.0, 1.0, 4.0))
    positive_target = torch.ones(2, dtype=torch.bool)
    negative_target = torch.zeros(3, dtype=torch.bool)
    positive_valid = torch.ones(2, dtype=torch.bool)
    negative_valid = torch.asarray((True, True, False))
    raw_positive = torch.nn.functional.binary_cross_entropy_with_logits(
        positive_logits, positive_target.float(), reduction="none"
    )
    raw_negative = torch.nn.functional.binary_cross_entropy_with_logits(
        negative_logits, negative_target.float(), reduction="none"
    )
    observed = effective_batch_balanced_bce(
        (
            (positive_logits, positive_target, positive_valid),
            (negative_logits, negative_target, negative_valid),
        )
    )
    torch.testing.assert_close(
        observed,
        0.5 * raw_positive.mean() + 0.5 * raw_negative[:2].mean(),
    )
    torch.testing.assert_close(
        effective_batch_balanced_bce(
            ((negative_logits, negative_target, negative_valid),)
        ),
        raw_negative[:2].mean(),
    )
    with pytest.raises(TrainingError, match="no valid targets"):
        effective_batch_balanced_bce(
            ((negative_logits, negative_target, torch.zeros(3, dtype=torch.bool)),)
        )


def test_one_frozen_bank_round_trips_and_is_shared_by_b1_b2_b3(tmp_path: Path) -> None:
    from src.train import (
        _predict_window_for_test,
        _write_window_train_bank_for_test,
        load_window_train_bank,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    renderer_identity = "b" * 64
    source = _training_window(renderer_identity)
    bank_path = tmp_path / "bank"
    bank = _write_window_train_bank_for_test(
        bank_path,
        (source,),
        protocol=protocol,
        renderer_identity=renderer_identity,
    )
    document = json.loads((bank_path / "manifest.json").read_text(encoding="utf-8"))
    assert document["test_fixture"] is True
    assert document["shared_by"] == ["B1", "B2", "B3"]
    assert document["entry_count"] == len(bank) == 1
    stored_world = WorldSpec.from_dict(
        document["entries"][0]["window_manifest"]["world"]
    )
    assert stored_world.identity == source.world_identity
    assert document["entries"][0]["valid_count"] == 5
    assert document["entries"][0]["anomaly_count"] == 0
    assert document["entries"][0]["normal_count"] == 5

    loaded = load_window_train_bank(bank_path, protocol=protocol)[0]
    assert loaded.window_identity == source.window_identity
    assert loaded.world_identity == source.world_identity
    assert loaded.source_observation_identities == source.source_observation_identities
    assert loaded.stu_input_identities == source.stu_input_identities
    assert loaded.five_source_frames == (0, 1, 2, 3, 4)
    for name in (
        "coordinates",
        "scan_group",
        "stu_features",
        "normal_evidence",
        "reliability_assign",
        "reliability_noobj",
        "intensity",
        "target",
        "valid",
        "source_frame",
        "source_slot",
        "source_ray",
    ):
        torch.testing.assert_close(getattr(loaded, name), getattr(source, name))

    class RecordingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(()))
            self.calls: list[tuple[str, tuple[int, ...]]] = []

        def forward(
            self,
            coordinates: torch.Tensor,
            _stu: torch.Tensor,
            _normal: torch.Tensor,
            _assign: torch.Tensor,
            _no_object: torch.Tensor,
            _intensity: torch.Tensor,
            scan_group: torch.Tensor,
            *,
            grouping_mode: object,
        ) -> torch.Tensor:
            mode = getattr(grouping_mode, "value", str(grouping_mode))
            self.calls.append((str(mode), tuple(scan_group.tolist())))
            return coordinates[:, 0] + self.bias

    expected_calls = {"B1": 5, "B2": 1, "B3": 1}
    expected_mode = {"B1": "single", "B2": "per_scan", "B3": "joint"}
    for condition in ("B1", "B2", "B3"):
        model = RecordingModel()
        logits = _predict_window_for_test(model, loaded, condition)
        expected = (
            loaded.native_coordinates[:, 0]
            if condition == "B1"
            else loaded.coordinates[:, 0]
        )
        torch.testing.assert_close(logits, expected)
        assert len(model.calls) == expected_calls[condition]
        assert {mode for mode, _ in model.calls} == {expected_mode[condition]}
        assert loaded.window_identity == source.window_identity


def test_b1_target_scan_is_independent_of_the_other_four_frames() -> None:
    from src.train import _predict_window_for_test

    original = _training_window("b" * 64)
    target = original.scan_group == 2
    changed = ~target
    coordinates = original.coordinates.clone()
    coordinates[target] += 1000.0  # A changed window pose must not reach native B1.
    coordinates[changed] -= 500.0
    stu_features = original.stu_features.clone()
    stu_features[changed] += 77.0
    modified = replace(
        original,
        coordinates=coordinates,
        stu_features=stu_features,
    )

    class CoordinateFeatureModel(nn.Module):
        def forward(
            self,
            coordinates: torch.Tensor,
            stu: torch.Tensor,
            _normal: torch.Tensor,
            _assign: torch.Tensor,
            _no_object: torch.Tensor,
            _intensity: torch.Tensor,
            _scan_group: torch.Tensor,
            *,
            grouping_mode: object,
        ) -> torch.Tensor:
            assert getattr(grouping_mode, "value", str(grouping_mode)) == "single"
            return coordinates[:, 0] + stu[:, 0]

    model = CoordinateFeatureModel()
    before = _predict_window_for_test(model, original, "B1")
    after = _predict_window_for_test(model, modified, "B1")
    torch.testing.assert_close(before[target], after[target], rtol=0.0, atol=0.0)


def test_b1_reuses_one_exact_native_score_per_overlapping_frame_ray() -> None:
    sources = tuple(
        _one_return_source(frame_id, 4.0 + frame_id, 0.2) for frame_id in range(6)
    )

    class Encoder(nn.Module):
        def forward(
            self,
            coordinates: torch.Tensor,
            features: torch.Tensor,
            real_slots: np.ndarray,
        ) -> STUPointEncoding:
            count = len(real_slots)
            return STUPointEncoding(
                point_features=torch.zeros((count, 128)),
                assigned_query=torch.zeros(count, dtype=torch.long),
                normal_evidence=torch.zeros((count, 19)),
                reliability_assign=torch.zeros(count),
                reliability_noobj=torch.zeros(count),
                maxlogit_score=torch.zeros(count),
                inverse_map=torch.arange(count),
                real_slots=torch.from_numpy(real_slots.astype(np.int64, copy=True)),
                input_identity=stu_input_identity(coordinates, features, real_slots),
            )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(
            self,
            coordinates: torch.Tensor,
            *_args: torch.Tensor,
            grouping_mode: object,
        ) -> torch.Tensor:
            assert getattr(grouping_mode, "value", str(grouping_mode)) == "single"
            self.calls += 1
            return coordinates[:, 0]

    def window(selected: tuple[object, ...], shift: float) -> object:
        return SimpleNamespace(
            frames=tuple(SimpleNamespace(source=source) for source in selected),
            points=SimpleNamespace(
                count=5,
                coordinates=np.column_stack(
                    (np.arange(5, dtype=np.float32) + shift, np.zeros((5, 2)))
                ),
                scan_group=np.arange(5, dtype=np.int64),
            ),
        )

    model = Model()
    inference = AJAEInference._for_test(
        model,
        Encoder(),
        condition="B1",
        slot_to_ray=lambda source: np.arange(source.slot_count, dtype=np.int32),
    )
    inference._cache_namespace = ("fixture-world",)
    first = inference._model_logits(window(sources[:5], 0.0))
    second = inference._model_logits(window(sources[1:], 1000.0))
    assert model.calls == 6
    np.testing.assert_array_equal(first[1:], second[:4])


def test_frozen_bank_rejects_a_tampered_shard_before_use(tmp_path: Path) -> None:
    from src.train import (
        TrainingError,
        _write_window_train_bank_for_test,
        load_window_train_bank,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    bank_path = tmp_path / "bank"
    bank = _write_window_train_bank_for_test(
        bank_path,
        (_training_window("c" * 64),),
        protocol=protocol,
        renderer_identity="c" * 64,
    )
    shard = bank.entries[0].shard
    shard.write_bytes(shard.read_bytes() + b"tampered")
    reloaded = load_window_train_bank(bank_path, protocol=protocol)
    with pytest.raises(TrainingError, match="shard hash changed"):
        _ = reloaded[0]


def test_frozen_bank_rejects_an_entry_without_content_identities(
    tmp_path: Path,
) -> None:
    from src.train import (
        TrainingError,
        _write_window_train_bank_for_test,
        load_window_train_bank,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    bank_path = tmp_path / "bank"
    _write_window_train_bank_for_test(
        bank_path,
        (_training_window("e" * 64),),
        protocol=protocol,
        renderer_identity="e" * 64,
    )
    manifest_path = bank_path / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["entries"][0].pop("source_observation_identities")
    document["entries"][0].pop("stu_input_identities")
    identity_payload = dict(document)
    identity_payload.pop("bank_identity")
    document["bank_identity"] = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TrainingError, match=r"entries\[0\] keys differ"):
        load_window_train_bank(bank_path, protocol=protocol)


def test_train_206_bank_rejects_held_out_torus_geometry() -> None:
    from src.train import TrainingError, _bank_window_world

    rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    world = WorldSpec(
        3101,
        206,
        (
            ObjectSpec(
                1,
                "anomaly-proxy",
                HeldOutTorusShape(0.4, 0.1),
                MaterialSpec(0.5, 0.1),
                (10.0, 0.0, 0.0),
                rotation,
            ),
        ),
    )
    with pytest.raises(TrainingError, match="held-out torus geometry"):
        _bank_window_world({"world": world.to_dict()})


def test_pre_s01_development_render_rejects_torus_before_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.render as render_module

    calls = 0

    def forbidden_render(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("held-out observations must not be rendered")

    monkeypatch.setattr(render_module, "render_frames", forbidden_render)
    sources = tuple(
        make_source_frame(
            frame_id,
            np.asarray(((4.0, 0.0, 0.0, 0.2),), dtype=np.float32),
            np.eye(4, dtype=np.float64),
            _labels(np.asarray((10,), dtype=np.uint16)),
            partition="train",
            sequence_id=201,
        )
        for frame_id in range(9)
    )
    rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    world = WorldSpec(
        3101,
        201,
        (
            ObjectSpec(
                1,
                "anomaly-proxy",
                HeldOutTorusShape(0.4, 0.1),
                MaterialSpec(0.5, 0.1),
                (10.0, 0.0, 0.0),
                rotation,
            ),
        ),
    )
    report = WorldGenerationReport(
        world_seed=world.seed,
        source_sequence_id=201,
        world_type="anomaly_only",
        world_attempt=0,
        normal_count=0,
        anomaly_count=1,
        count_seed=3102,
        label_order_seed=3103,
    )
    with pytest.raises(RenderError, match="remain unopened until S01"):
        render_module.render_development_clip_world(
            world,
            report,
            sources,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            renderer_identity="a" * 64,
        )
    assert calls == 0


@pytest.mark.parametrize("changed_field", ("point", "label"))
def test_window_identity_rejects_changed_rendered_observation(
    changed_field: str,
) -> None:
    from src.train import TrainingError

    source = _training_window("d" * 64)
    original = _one_return_source(0, 4.0, 0.2)
    if changed_field == "point":
        xyzi = original.xyzi.copy()
        xyzi[0, 0] += np.float32(0.01)
        changed = make_source_frame(
            original.frame_id,
            xyzi,
            original.lidar_pose,
            original.labels,
            partition=original.partition,
            sequence_id=original.sequence_id,
        )
    else:
        changed = make_source_frame(
            original.frame_id,
            original.xyzi,
            original.lidar_pose,
            _labels(np.asarray((11, 0), dtype=np.uint16)),
            partition=original.partition,
            sequence_id=original.sequence_id,
        )
    changed_identity = source_observation_identity(changed)
    assert changed.frame_id == original.frame_id
    assert changed.sequence_id == original.sequence_id
    assert changed_identity != source.source_observation_identities[0]

    observations = (changed_identity, *source.source_observation_identities[1:])
    manifest = {
        **source.window_manifest,
        "source_observation_identities": list(observations),
    }
    assert manifest["frame_ids"] == source.window_manifest["frame_ids"]
    assert manifest["world_identity"] == source.world_identity
    with pytest.raises(TrainingError, match="identity does not match its inputs"):
        replace(
            source,
            source_observation_identities=observations,
            window_manifest=manifest,
            window_identity="",
        )


def test_nine_frames_expose_every_window_and_occurrence_stratum() -> None:
    spec = SequenceSpec("train", 201, "fixture", True, FrameSpan(0, 9))
    starts = spec.legal_window_starts()
    assert starts == (0, 1, 2, 3, 4)
    assert tuple(spec.window_frame_ids(start) for start in starts) == (
        (0, 1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (3, 4, 5, 6, 7),
        (4, 5, 6, 7, 8),
    )

    world = "world-A"
    fusion = WindowScoreFusion()
    expected: dict[int, list[float]] = {frame: [] for frame in range(9)}
    for start in starts:
        frames = np.arange(start, start + 5, dtype=np.int32)
        probabilities = 0.08 + 0.07 * start + 0.01 * frames
        logits = np.log(probabilities / (1.0 - probabilities))
        fusion.add_synthetic(
            world_identity=world,
            window_start=start,
            source_frame=frames,
            source_ray=1000 + frames,
            source_slot=np.zeros(5, dtype=np.int32),
            logits=logits,
        )
        for frame, probability in zip(frames, probabilities, strict=True):
            expected[int(frame)].append(float(probability))

    # Identical physical keys remain independent in another world and real sequence.
    fusion.add_synthetic(
        world_identity="world-B",
        window_start=0,
        source_frame=np.asarray((0,), dtype=np.int32),
        source_ray=np.asarray((1000,), dtype=np.int32),
        source_slot=np.asarray((0,), dtype=np.int32),
        logits=np.asarray((math.log(0.99 / 0.01),)),
    )
    fusion.add_real(
        partition="train",
        sequence_id=201,
        window_start=0,
        source_frame=np.asarray((0,), dtype=np.int32),
        source_ray=np.asarray((1000,), dtype=np.int32),
        source_slot=np.asarray((0,), dtype=np.int32),
        logits=np.asarray((math.log(0.25 / 0.75),)),
    )
    fusion.add_real(
        partition="train",
        sequence_id=206,
        window_start=0,
        source_frame=np.asarray((0,), dtype=np.int32),
        source_ray=np.asarray((1000,), dtype=np.int32),
        source_slot=np.asarray((0,), dtype=np.int32),
        logits=np.asarray((0.0,)),
    )
    fusion.add_real(
        partition="val",
        sequence_id=125,
        window_start=0,
        source_frame=np.asarray((0,), dtype=np.int32),
        source_ray=np.asarray((1000,), dtype=np.int32),
        source_slot=np.asarray((0,), dtype=np.int32),
        logits=np.asarray((math.log(0.75 / 0.25),)),
    )

    result = fusion.finalize_synthetic(world)
    assert result.source_frame.tolist() == list(range(9))
    assert result.occurrence_count.tolist() == [1, 2, 3, 4, 5, 4, 3, 2, 1]
    assert result.occurrence_histogram == {"1": 2, "2": 2, "3": 2, "4": 2, "5": 1}
    independent = np.asarray(
        [
            np.sum(expected[frame], dtype=np.float64) / len(expected[frame])
            for frame in range(9)
        ]
    )
    np.testing.assert_allclose(result.probability, independent, atol=2e-16, rtol=2e-15)
    assert fusion.finalize_synthetic("world-B").probability[0] == pytest.approx(0.99)
    assert fusion.finalize_real("train", 201).probability[0] == pytest.approx(0.25)
    assert fusion.finalize_real("train", 206).probability[0] == pytest.approx(0.5)
    assert fusion.finalize_real("val", 125).probability[0] == pytest.approx(0.75)


def test_b0_fusion_preserves_the_frozen_stu_score_domain() -> None:
    fusion = WindowScoreFusion(fusion_value=B0_FUSION_VALUE)
    for window_start in (0, 1):
        fusion.add_real_scores(
            partition="train",
            sequence_id=206,
            window_start=window_start,
            source_frame=np.asarray((1,), dtype=np.int32),
            source_ray=np.asarray((7,), dtype=np.int32),
            source_slot=np.asarray((3,), dtype=np.int32),
            scores=np.asarray((-4.25,), dtype=np.float64),
        )
    result = fusion.finalize_real("train", 206)
    assert result.score.tolist() == [-4.25]
    assert result.occurrence_count.tolist() == [2]
    with pytest.raises(EvaluationError, match="not probabilities"):
        _ = result.probability


def test_development_scoring_is_exactly_24_in_generator_clips() -> None:
    import src.train as train_module

    identity = EvaluationIdentity(
        protocol_schema=32,
        protocol_identity="1" * 64,
        condition="B3",
        fusion_value=AJAE_FUSION_VALUE,
        model_class="JointWindowPointTransformer",
        model_state_sha256="2" * 64,
        stu_class="FrozenSTUPointEncoder",
        stu_checkpoint_sha256="3" * 64,
        stu_model_state_sha256="4" * 64,
        stu_source_manifest_sha256="5" * 64,
        calibration_sha256="6" * 64,
        ray_mapping_sha256="7" * 64,
    )

    def result(index: int, mechanism: str = "in_generator") -> DevelopmentClipResult:
        clip_identity = hashlib.sha256(f"clip-{index}".encode()).hexdigest()
        world_identity = hashlib.sha256(f"world-{index}".encode()).hexdigest()
        observations = tuple(
            hashlib.sha256(f"observation-{index}-{frame}".encode()).hexdigest()
            for frame in range(9)
        )
        return DevelopmentClipResult(
            clip_identity=clip_identity,
            world_identity=world_identity,
            source_observation_identities=observations,
            mechanism=mechanism,
            fused_point_ap=0.2,
            unique_point_count=1,
            occurrence_count=1,
            occurrence_histogram={
                "1": 1,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 0,
            },
            frame_count=9,
            window_count=5,
        )

    with pytest.raises(EvaluationError, match="restricted to in-generator clips"):
        result(24, "torus_SDF")

    clips = tuple(result(index) for index in range(24))
    definitions = []
    for clip in clips:
        definitions.append(
            SimpleNamespace(
                identity=clip.clip_identity,
                world_identity=clip.world_identity,
                mechanism=clip.mechanism,
                frame_ids=tuple(range(9)),
                windows=tuple(range(5)),
                source_observation_identities=clip.source_observation_identities,
            )
        )
    with pytest.raises(EvaluationError, match="frozen 24 in-generator clips"):
        DevelopmentFusedAP("B3", clips[:-1], identity, FUSION_SEMANTICS)

    evidence = DevelopmentFusedAP("B3", clips, identity, FUSION_SEMANTICS)
    assert evidence.macro_fused_point_ap == pytest.approx(0.2)
    serialized = evidence.to_dict()
    assert serialized["mechanism"] == "in_generator"
    assert serialized["clip_count"] == 24
    assert "held_out_torus_macro_fused_point_ap" not in serialized

    development = SimpleNamespace(
        definitions=SimpleNamespace(
            protocol_identity=identity.protocol_identity,
            clips=tuple(definitions),
        ),
        identity="8" * 64,
    )
    record = train_module._development_record(
        evidence, ExperimentCondition.B3, development
    )
    assert record["selection_metric"] == "macro_fused_point_ap"
    assert record["macro_fused_point_ap"] == pytest.approx(0.2)
    assert record["scope"] == (
        "24_in_generator_clips_only_held_out_torus_unopened_until_S01"
    )
    assert "held_out_torus_macro_fused_point_ap" not in record


def test_development_raw_samples_must_reproduce_every_clip_ap() -> None:
    identity = EvaluationIdentity(
        protocol_schema=32,
        protocol_identity="1" * 64,
        condition="B3",
        fusion_value=AJAE_FUSION_VALUE,
        model_class="JointWindowPointTransformer",
        model_state_sha256="2" * 64,
        stu_class="FrozenSTUPointEncoder",
        stu_checkpoint_sha256="3" * 64,
        stu_model_state_sha256="4" * 64,
        stu_source_manifest_sha256="5" * 64,
        calibration_sha256="6" * 64,
        ray_mapping_sha256="7" * 64,
    )
    clips = tuple(
        DevelopmentClipResult(
            clip_identity=hashlib.sha256(f"raw-clip-{index}".encode()).hexdigest(),
            world_identity=hashlib.sha256(f"raw-world-{index}".encode()).hexdigest(),
            source_observation_identities=tuple(
                hashlib.sha256(f"raw-source-{index}-{frame}".encode()).hexdigest()
                for frame in range(9)
            ),
            mechanism="in_generator",
            fused_point_ap=1.0,
            unique_point_count=2,
            occurrence_count=2,
            occurrence_histogram={
                "1": 2,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 0,
            },
            frame_count=9,
            window_count=5,
        )
        for index in range(24)
    )
    samples = {
        clip.clip_identity: (
            np.asarray((False, True), dtype=np.bool_),
            np.asarray((0.0, 1.0), dtype=np.float64),
        )
        for clip in clips
    }
    DevelopmentFusedAP("B3", clips, identity, FUSION_SEMANTICS, samples)
    changed = dict(samples)
    changed[clips[0].clip_identity] = (
        np.asarray((False, True), dtype=np.bool_),
        np.asarray((1.0, 0.0), dtype=np.float64),
    )
    with pytest.raises(EvaluationError, match="do not reproduce AP"):
        DevelopmentFusedAP("B3", clips, identity, FUSION_SEMANTICS, changed)


def test_raw_metric_artifacts_are_recomputed_and_content_addressed(
    tmp_path: Path,
) -> None:
    from src.evaluate import (
        FormalMetricEvidence,
        NormalSafetyEvidence,
        PublicMetricEvidence,
        _save_public_metric_arrays,
        save_formal_metric_evidence,
        save_normal_safety_evidence,
    )

    protocol = _protocol_with_frozen_r02_thresholds(tmp_path)
    identity = EvaluationIdentity(
        protocol_schema=32,
        protocol_identity=protocol.scientific_identity,
        condition="B3",
        fusion_value=AJAE_FUSION_VALUE,
        model_class="JointWindowPointTransformer",
        model_state_sha256="2" * 64,
        stu_class="FrozenSTUPointEncoder",
        stu_checkpoint_sha256=protocol.stu["checkpoint_sha256"],
        stu_model_state_sha256=protocol.stu["model_state_tensor_sha256"],
        stu_source_manifest_sha256=protocol.stu["source_manifest_sha256"],
        calibration_sha256=protocol.render["calibration_sha256"],
        ray_mapping_sha256=protocol.render["ray_grid"]["canonical_sha256"],
    )
    clip_ids = tuple(
        hashlib.sha256(f"metric-clip-{index}".encode()).hexdigest()
        for index in range(24)
    )
    samples = {
        clip_id: (
            np.asarray((False, True), dtype=np.bool_),
            np.asarray((0.0, 1.0), dtype=np.float64),
        )
        for clip_id in clip_ids
    }
    clips = tuple(
        DevelopmentClipResult(
            clip_identity=clip_id,
            world_identity=hashlib.sha256(f"metric-world-{index}".encode()).hexdigest(),
            source_observation_identities=tuple(
                hashlib.sha256(f"metric-source-{index}-{frame}".encode()).hexdigest()
                for frame in range(9)
            ),
            mechanism="in_generator",
            fused_point_ap=1.0,
            unique_point_count=2,
            occurrence_count=2,
            occurrence_histogram={"1": 2, "2": 0, "3": 0, "4": 0, "5": 0},
            frame_count=9,
            window_count=5,
        )
        for index, clip_id in enumerate(clip_ids)
    )
    development_evidence = DevelopmentFusedAP(
        "B3", clips, identity, FUSION_SEMANTICS, samples
    )
    formal_reference = save_formal_metric_evidence(
        "formal-evidence.npz",
        protocol=protocol,
        development_evidence=development_evidence,
        population_identity="a" * 64,
    )
    formal = FormalMetricEvidence.load(
        formal_reference,
        protocol=protocol,
        expected_identity=identity,
        expected_population_identity="a" * 64,
        expected_clip_identities=clip_ids,
    )
    assert set(formal.clip_ap.values()) == {1.0}

    safety_reference = save_normal_safety_evidence(
        "normal-safety.npz",
        protocol=protocol,
        evaluation_identity=identity,
        evaluation_set_identity="b" * 64,
        threshold_rule_identity="c" * 64,
        decision_threshold=0.5,
        labels=np.zeros(4, dtype=np.bool_),
        scores=np.asarray((0.1, 0.4, 0.6, 0.9), dtype=np.float64),
    )
    safety = NormalSafetyEvidence.load(
        safety_reference, protocol=protocol, expected_identity=identity
    )
    assert safety.false_positive_rate == 0.5

    # Public evidence only needs the current-node guard; use a trusted protocol clone
    # to isolate raw metric recomputation from the later M01/G3 record graph.
    public_protocol = object.__new__(AJAEProtocol)
    public_protocol.__dict__.update(protocol.__dict__)
    public_protocol.status = {**dict(protocol.status), "current_node": "V01"}
    points = np.column_stack((np.arange(3.0, 15.0), np.zeros((12, 2)))).astype(
        np.float32
    )
    semantic = np.asarray((10,) * 6 + (2,) * 6, dtype=np.int64)
    scores = np.linspace(0.0, 1.0, 12, dtype=np.float64)
    public_reference = _save_public_metric_arrays(
        "public-evidence.npz",
        protocol=public_protocol,
        evaluation_identity=identity,
        sequence_id=125,
        source_frame=np.zeros(12, dtype=np.int64),
        source_ray=np.arange(12, dtype=np.int64),
        source_slot=np.arange(12, dtype=np.int64),
        occurrence_count=np.ones(12, dtype=np.int64),
        coordinates=points[:, :3],
        raw_semantic=semantic,
        scores=scores,
    )
    public = PublicMetricEvidence.load(
        public_reference,
        protocol=public_protocol,
        expected_identity=identity,
        expected_sequence_id=125,
    )
    assert public.metrics_unit_interval["AP"] == pytest.approx(1.0)
    artifact = tmp_path / "public-evidence.npz"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(EvaluationError, match="file identity changed"):
        PublicMetricEvidence.load(
            public_reference,
            protocol=public_protocol,
            expected_identity=identity,
            expected_sequence_id=125,
        )


def test_s01_generator_is_blocked_before_any_held_out_access(tmp_path: Path) -> None:
    from src.evaluate import generate_s01_clip_evidence

    with pytest.raises(EvaluationError, match="authorized only at S01"):
        generate_s01_clip_evidence(
            protocol_path=PROTOCOL_PATH,
            g3_verdict_path=tmp_path / "must-not-be-opened.json",
            data_root=tmp_path / "must-not-be-opened",
            plan_position=24,
            destination=tmp_path / "must-not-exist.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()


@pytest.mark.parametrize(
    ("partition", "sequence_id", "label_mode"),
    (("val", 125, LabelMode.REQUIRED), ("test", 100, LabelMode.FORBIDDEN)),
)
def test_sealed_sequences_are_rejected_before_path_resolution(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    partition: str,
    sequence_id: int,
    label_mode: LabelMode,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    missing = tmp_path / "must-not-be-opened"
    with pytest.raises(EvaluationError, match="sealed .* unavailable"):
        open_sealed_sequence(
            missing,
            protocol=protocol,
            partition=partition,
            sequence_id=sequence_id,
            condition="B3",
            label_mode=label_mode,
        )
    with pytest.raises(SceneDataError, match="sealed until"):
        STUSequence.open(
            missing,
            protocol=protocol,
            partition=partition,
            sequence_id=sequence_id,
            label_mode=label_mode,
        )
    assert "Refused sealed sequence access" in caplog.text
    assert not missing.exists()


def test_method_freeze_record_is_content_addressed_and_population_complete(
    tmp_path: Path,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    identity = EvaluationIdentity(
        protocol_schema=32,
        protocol_identity=protocol.scientific_identity,
        condition="B3",
        fusion_value=AJAE_FUSION_VALUE,
        model_class="JointWindowPointTransformer",
        model_state_sha256="2" * 64,
        stu_class="FrozenSTUPointEncoder",
        stu_checkpoint_sha256="3" * 64,
        stu_model_state_sha256="4" * 64,
        stu_source_manifest_sha256="5" * 64,
        calibration_sha256="6" * 64,
        ray_mapping_sha256="7" * 64,
    )
    unsigned = {
        "format": METHOD_FREEZE_FORMAT,
        "status": METHOD_FREEZE_STATUS,
        "evaluation_identity": identity.to_dict(),
        "sealed_sequences": {
            "val": list(protocol.public_sequence_ids),
            "test": list(protocol.hidden_sequence_ids),
        },
    }
    payload = {
        **unsigned,
        "record_sha256": hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    path = tmp_path / "method-freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = MethodFreezeRecord.load(
        path, expected_identity=identity, protocol=protocol
    )
    assert loaded.sealed_sequences["val"] == protocol.public_sequence_ids
    assert loaded.sealed_sequences["test"] == protocol.hidden_sequence_ids

    payload["sealed_sequences"]["val"] = payload["sealed_sequences"]["val"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationError, match="content hash"):
        MethodFreezeRecord.load(path, expected_identity=identity, protocol=protocol)


def test_point_metrics_match_the_released_stu_calculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    official_root = protocol.stu_repository_path().parent
    official_script = official_root / "compute_point_level_ood.py"
    monkeypatch.syspath_prepend(str(official_root))
    specification = importlib.util.spec_from_file_location(
        "ajae_test_official_point_schema32", official_script
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    official = module.PointOODMetricsCalculator()
    ours = PointMetricAccumulator(protocol)

    points = np.column_stack((np.arange(3.0, 15.0), np.zeros((12, 2)))).astype(
        np.float32
    )
    scores = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.35, 0.45, 0.55, 0.7, 0.8, 0.9))
    semantic = np.asarray((10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 2, 2), dtype=np.uint16)
    official.update(points, scores, semantic)
    assert ours.update(points, scores, semantic)
    official_result = official.compute_metrics()
    our_result = ours.compute()
    for metric in ("AP", "AUROC", "FPR95", "threshold"):
        assert our_result[metric] == pytest.approx(official_result[metric])
