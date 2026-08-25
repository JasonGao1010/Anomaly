#!/usr/bin/env python3
"""Focused scientific-semantic tests for the sole AJAE schema-30 route."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import src.render as render_module
from src.evaluate import (
    AJAEInference,
    EvaluationError,
    MovingNormalDiagnostic,
    ObjectScaleDiagnostic,
    PointMetricAccumulator,
    WindowScoreFusion,
    _protocol_slot_to_ray,
    _validate_public_result,
    load_prediction_coverage,
)
from src.model import AJAEPointTransformer, assigned_stu_evidence, temporal_radius_knn
from src.protocol import (
    CAUSAL_OFFSETS,
    RELATIVE_TIMES,
    ExperimentCondition,
    FrameSpan,
    ProtocolError,
    SequenceSpec,
    load_development_worlds,
    load_protocol,
)
from src.render import (
    HeldOutTorusShape,
    NormalTemplateShape,
    RayGrid,
    SensorCalibration,
    ShapeSpec,
    SupportPoints,
    WorldSpec,
    load_sensor_calibration,
    render_frame,
    sample_held_out_anomaly_shape,
    sample_training_world,
    sample_training_anomaly_shape,
)
from src.scene import (
    LabelMode,
    PointLabels,
    SceneDataError,
    STUSequence,
    assemble_window,
    canonical_ray_mapping_digest,
    make_source_frame,
)
from src.train import (
    AJAETrainer,
    DevelopmentEvidence,
    DevelopmentWorldMetrics,
    TrainingError,
    balanced_bce_loss,
    checkpoint_selection_key,
    experiment_condition,
    validate_formal_preflight,
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


def test_public_sequences_are_sealed_before_path_resolution(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    with pytest.raises(SceneDataError, match="sealed until"):
        STUSequence.open(
            tmp_path / "does-not-exist",
            protocol=protocol,
            partition="val",
            sequence_id=protocol.public_sequence_ids[0],
            label_mode=LabelMode.REQUIRED,
        )
    assert "Refused sealed sequence access" in caplog.text
    assert not (tmp_path / "does-not-exist").exists()


def _organized_frame(frame_id: int, *, real_slot: int = 0) -> object:
    xyzi = np.zeros((128 * 1024, 4), dtype=np.float32)
    xyzi[real_slot] = (5.0 + frame_id, 0.1, 0.2, 0.4)
    semantic = np.zeros(xyzi.shape[0], dtype=np.uint16)
    semantic[real_slot] = 10
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 0.1 * frame_id
    return make_source_frame(
        frame_id,
        xyzi,
        pose,
        _labels(semantic),
        partition="train",
        sequence_id=206,
    )


def _small_ray_fixture() -> tuple[object, RayGrid]:
    columns = 8
    azimuth = np.arange(columns, dtype=np.float64) * (-2.0 * np.pi / columns)
    directions = np.stack((np.cos(azimuth), np.sin(azimuth), np.zeros(columns)), axis=1)
    grid = RayGrid(directions, np.zeros(1), azimuth, beam_count=1)
    ranges = np.linspace(4.0, 8.0, columns, dtype=np.float32)
    xyzi = np.zeros((columns, 4), dtype=np.float32)
    xyzi[:, :3] = (directions * ranges[:, None]).astype(np.float32)
    xyzi[:, 3] = np.linspace(0.1, 0.8, columns, dtype=np.float32)
    frame = make_source_frame(
        0,
        xyzi,
        np.eye(4, dtype=np.float64),
        _labels(np.full(columns, 10, dtype=np.uint16)),
        partition="fixture",
        sequence_id=99,
    )
    return frame, grid


def _development_evidence(ap: float, normal_q: float) -> DevelopmentEvidence:
    return DevelopmentEvidence(
        tuple(DevelopmentWorldMetrics(world_id, {"AP": ap}) for world_id in range(24)),
        {"q99.9": normal_q},
    )


def _selection_rule() -> dict[str, object]:
    return {
        "status": "frozen_before_training",
        "primary": "maximum macro mean of per-world AP over the 24 in-generator worlds",
        "tie_tolerance": 1.0e-6,
        "first_tie_break": "lower pure-normal score q99.9",
        "second_tie_break": "earlier completed world index",
        "held_out_input_forbidden": True,
    }


def test_protocol_is_only_schema30_route() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol.schema_version == 30
    assert protocol.authority["document"] == "AJAE新主线方案.md"
    assert protocol.normal_training.sequence_id == 206
    assert protocol.development_sequence.sequence_id == 201
    assert protocol.normal_training.uses_gradients
    assert not protocol.development_sequence.uses_gradients
    assert protocol.stu["frozen"] is True
    assert protocol.model["input_dim"] == 150
    assert protocol.model["levels"] == 4
    assert protocol.model["pooling"] == "per_time_mean_max"
    assert tuple(protocol.training["seeds"]) == (0, 1, 2)
    assert protocol.training["loss"].endswith("binary cross entropy only")
    assert protocol.evaluation.minimum_range_m == 2.5
    assert protocol.evaluation.maximum_range_m == 50.0
    assert protocol.development["checkpoint_selection"]["status"] == (
        "proposed_requires_owner_confirmation"
    )
    assert protocol.development["fixed_world_evaluation"]["status"] == (
        "unresolved_requires_owner_decision"
    )
    assert protocol.evaluation_document["comparison_frame_domain"]["status"] == (
        "proposed_requires_owner_confirmation"
    )
    assert protocol.decision_gates["criteria"]["status"] == (
        "unresolved_requires_owner_decision"
    )


def test_protocol_contains_no_retired_training_route() -> None:
    text = json.dumps(json.loads(PROTOCOL_PATH.read_text()), sort_keys=True)
    for retired in (
        "lambda_cf",
        "memory_beta",
        "memory_warmup_worlds",
        "point_window_weight",
        "Hungarian",
        "object_adapter",
    ):
        assert retired not in text


def test_real_development_worlds_are_fixed_but_unvalidated() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    worlds = load_development_worlds(DEVELOPMENT_PATH, protocol=protocol)
    all_worlds = (*worlds.in_generator, *worlds.generator_held_out)
    assert [item.world_id for item in all_worlds] == list(range(30))
    assert len(worlds.in_generator) == 24 and len(worlds.generator_held_out) == 6
    assert not worlds.validated
    assert not worlds.difficulty_coverage_valid
    assert worlds.status == "definitions_only_unvalidated"
    assert worlds.gate1["status"] == "pending_scientific_verdict"
    legal = frozenset(protocol.development_sequence.center_frames())
    assert all(item.center_frame in legal for item in all_worlds)
    for item in worlds.in_generator:
        world = WorldSpec.from_dict(item.world)
        assert world.world_type == "mixed"
        assert all(
            isinstance(obj.shape, ShapeSpec)
            for obj in world.objects
            if obj.label == "anomaly-proxy"
        )
    for item in worlds.generator_held_out:
        world = WorldSpec.from_dict(item.world)
        assert all(
            isinstance(obj.shape, HeldOutTorusShape)
            for obj in world.objects
            if obj.label == "anomaly-proxy"
        )


def test_gate1_cannot_be_unlocked_by_editing_status_or_arbitrary_numbers(
    tmp_path: Path,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    document = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    document["status"] = "validated_frozen"
    document["validation"] = {
        name: True for name in document["validation"]
    }
    document["gate1"]["status"] = "passed_with_real_evidence"
    edited = tmp_path / "edited-dev.json"
    edited.write_text(json.dumps(document), encoding="utf-8")
    worlds = load_development_worlds(edited, protocol=protocol)
    assert not worlds.gate1_evidence_valid
    assert not worlds.validated

    document["gate1"]["evidence"] = {
        name: {"x": 1}
        for name in (
            "ray_slot_audit",
            "range_image_round_trip",
            "render_source_leakage",
            "beam_range_intensity",
        )
    }
    edited.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ProtocolError, match="input_identity"):
        load_development_worlds(edited, protocol=protocol)


def test_causal_window_separates_physical_and_model_time() -> None:
    frames = tuple(_organized_frame(frame_id) for frame_id in range(5))
    spec = SequenceSpec("train", 206, "fixture", True, FrameSpan(0, 5))
    mapping = np.arange(128 * 1024, dtype=np.int32)
    digest = canonical_ray_mapping_digest(mapping)
    causal = assemble_window(
        spec,
        4,
        frames,
        condition=ExperimentCondition.B5,
        canonical_ray_by_slot=mapping,
        ray_mapping_audited=True,
        ray_mapping_digest=digest,
    )
    assert tuple(item.source_offset for item in causal.frames) == CAUSAL_OFFSETS
    assert tuple(item.model_time_position for item in causal.frames) == RELATIVE_TIMES
    assert causal.reference.frame_id == 4
    assert tuple(causal.points.relative_time.tolist()) == RELATIVE_TIMES
    assert causal.points.ray_mapping_digest == digest


def test_causal_training_supervises_all_five_frames_but_outputs_current() -> None:
    condition = experiment_condition("B5")
    assert condition.frame_offsets == CAUSAL_OFFSETS
    assert condition.model_times == RELATIVE_TIMES
    assert condition.supervised_times == RELATIVE_TIMES
    assert condition.prediction_rule == "causal_current_frame_at_model_position_plus2"


def test_audited_mapping_requires_exact_calibration_digest() -> None:
    frame = _organized_frame(0)
    spec = SequenceSpec("train", 206, "fixture", True, FrameSpan(0, 1))
    mapping = np.arange(128 * 1024, dtype=np.int32)
    digest = canonical_ray_mapping_digest(mapping)
    with pytest.raises(SceneDataError, match="requires an explicit"):
        assemble_window(
            spec,
            0,
            (frame,),
            condition="B1",
            ray_mapping_audited=True,
            ray_mapping_digest=digest,
        )
    changed = mapping.copy()
    changed[[0, 1]] = changed[[1, 0]]
    with pytest.raises(SceneDataError, match="does not match"):
        assemble_window(
            spec,
            0,
            (frame,),
            condition="B1",
            canonical_ray_by_slot=changed,
            ray_mapping_audited=True,
            ray_mapping_digest=digest,
        )


def test_authoritative_calibration_is_complete_train206() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    grid, sensor = load_sensor_calibration(protocol.sensor_calibration_path())
    assert grid.beam_count == 128 and grid.columns == 1024
    assert grid.calibration_frame_ids == tuple(range(449))
    assert sensor.source_sequence_id == 206
    provenance = dict(sensor.provenance)
    assert provenance["protocol_schema"] == "30"
    assert provenance["partition"] == "train"
    assert provenance["frames"] == "449"
    mapper, ray_digest, calibration_digest = _protocol_slot_to_ray(protocol)
    assert canonical_ray_mapping_digest(mapper(_organized_frame(0))) == ray_digest
    assert len(calibration_digest) == 64


def test_assigned_stu_evidence_uses_one_minimum_index_query() -> None:
    logits = torch.zeros(100, 20)
    masks = torch.full((2, 100), -20.0)
    logits[0, 0] = logits[1, 1] = 5.0
    masks[:, :2] = 2.0
    evidence = assigned_stu_evidence(logits, masks)
    probability = logits.softmax(dim=1)
    mask_probability = torch.sigmoid(torch.tensor(2.0))
    assignment = mask_probability * probability[0, :19].max()
    torch.testing.assert_close(evidence.reliability_assign, assignment.expand(2))
    torch.testing.assert_close(
        evidence.normal_evidence,
        (mask_probability * probability[0, :19]).expand(2, 19),
    )
    torch.testing.assert_close(evidence.reliability_noobj, probability[0, 19].expand(2))


def test_temporal_neighbors_are_partitioned_by_exact_delta() -> None:
    coordinates = torch.zeros(5, 3)
    times = torch.tensor(RELATIVE_TIMES, dtype=torch.long)
    neighbor, valid = temporal_radius_knn(coordinates, times, 1, 0.5, 2)
    assert not bool(valid[-1].any())
    assert valid[:-1, 0].all()
    assert torch.equal(neighbor[:-1, 0], torch.arange(1, 5))


def test_four_level_model_forward_backward_and_b2_isolation() -> None:
    torch.manual_seed(7)
    model = AJAEPointTransformer(
        hidden_dim=16,
        voxel_sizes=(0.1, 0.2, 0.4),
        neighbor_radii=((0.5,) * 5, (1.0,) * 5, (2.0,) * 5, (4.0,) * 5),
        neighbor_k=((2,) * 5,) * 4,
        heads=4,
        attention_chunk_size=32,
    )
    count = 10
    coordinates = torch.randn(count, 3) * 0.05
    times = torch.tensor(RELATIVE_TIMES, dtype=torch.long).repeat_interleave(2)
    features = torch.randn(count, 128)
    evidence = torch.rand(count, 19)
    assign, noobj, intensity = torch.rand(count), torch.rand(count), torch.rand(count)
    logits = model(
        coordinates,
        times,
        features,
        evidence,
        assign,
        noobj,
        intensity,
        cross_frame_enabled=True,
    )
    assert logits.shape == (count,)
    logits.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        disabled = model(
            coordinates,
            times,
            features,
            evidence,
            assign,
            noobj,
            intensity,
            cross_frame_enabled=False,
        )
        changed = features.clone()
        changed[times != 0] += 100.0
        disabled_changed = model(
            coordinates,
            times,
            changed,
            evidence,
            assign,
            noobj,
            intensity,
            cross_frame_enabled=False,
        )
    torch.testing.assert_close(disabled[times == 0], disabled_changed[times == 0])


def test_training_and_heldout_geometry_are_disjoint_and_bounded() -> None:
    for seed in range(40):
        training_shape = sample_training_anomaly_shape(seed)
        heldout_shape = sample_held_out_anomaly_shape(seed)
        assert isinstance(training_shape, ShapeSpec)
        assert isinstance(heldout_shape, HeldOutTorusShape)
        lower, upper = training_shape.local_bounds()
        assert 0.2 <= float(np.max(upper - lower)) <= 3.0
        report = training_shape.geometry_report()
        assert report["bounded"] and report["closed"] and report["components"] == 1


def test_mixed_training_world_pairs_support_distance_size_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elevation = np.linspace(-0.2, 0.2, 128)
    directions = np.column_stack((np.cos(elevation), np.zeros(128), np.sin(elevation)))
    grid = RayGrid(directions, elevation, np.asarray((0.0,)), beam_count=128)
    frames = []
    for frame_id in range(5):
        xyzi = np.zeros((128, 4), dtype=np.float32)
        xyzi[:, :3] = directions * 5.0
        xyzi[:, 3] = 0.4
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 0.1 * frame_id
        frames.append(
            make_source_frame(
                frame_id,
                xyzi,
                pose,
                _labels(np.full(128, 10, dtype=np.uint16)),
                partition="train",
                sequence_id=206,
            )
        )
    x, y = np.meshgrid(
        np.arange(5.0, 45.1, 0.5),
        np.arange(-10.0, 10.1, 0.5),
        indexing="ij",
    )
    ground = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    context = SupportPoints(
        ground,
        np.full(ground.shape[0], 40, dtype=np.uint16),
        np.empty((0, 3)),
    )
    vertices = np.asarray(
        [
            (x_value, y_value, z_value)
            for x_value in (-1.0, 1.0)
            for y_value in (-0.5, 0.5)
            for z_value in (-0.5, 0.5)
        ]
    )
    template = NormalTemplateShape(
        vertices,
        np.empty((0, 3), dtype=np.int32),
        206,
        0,
        10,
        1,
        (0.0, 0.0, 0.0),
    )
    sensor = SensorCalibration.constant(0.4)
    counts = [(1, 1)]
    placement_calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    original_place = render_module.place_object

    def record_place(*args: object, **kwargs: object) -> object:
        semantic = np.asarray(kwargs["ground_semantic_ids"], dtype=np.uint16)
        allowed = tuple(int(value) for value in kwargs["allowed_support_semantics"])
        placement_calls.append(
            (str(kwargs["label"]), tuple(map(int, np.unique(semantic))), allowed)
        )
        return original_place(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(render_module, "collect_support_context", lambda _: context)
    monkeypatch.setattr(render_module, "_training_entity_counts", lambda *_: counts[0])
    monkeypatch.setattr(
        render_module,
        "validate_world_visibility",
        lambda *args, **kwargs: {1: 10, 2: 10, 3: 10},
    )
    monkeypatch.setattr(render_module, "place_object", record_place)

    first = sample_training_world(frames, (template,), grid, sensor, "mixed", 7)
    first_calls = tuple(placement_calls)
    placement_calls.clear()
    repeated = sample_training_world(frames, (template,), grid, sensor, "mixed", 7)
    assert first.to_dict() == repeated.to_dict()
    assert first_calls == tuple(placement_calls)
    assert len(first_calls) == 2
    assert {call[0] for call in first_calls} == {
        "normal-control",
        "anomaly-proxy",
    }
    assert {call[1] for call in first_calls} == {(40,)}
    assert {call[2] for call in first_calls} == {(40,)}

    normal = next(item for item in first.objects if item.label == "normal-control")
    proxy = next(item for item in first.objects if item.label == "anomaly-proxy")
    normal_position = np.asarray(normal.translation_world_m)
    proxy_position = np.asarray(proxy.translation_world_m)
    assert np.linalg.norm(normal_position[:2] - proxy_position[:2]) <= 6.25
    origin = frames[2].lidar_pose[:3, 3]
    normal_tier = np.searchsorted(
        sensor.range_edges_m, np.linalg.norm(normal_position - origin), side="right"
    )
    proxy_tier = np.searchsorted(
        sensor.range_edges_m, np.linalg.norm(proxy_position - origin), side="right"
    )
    assert normal_tier == proxy_tier
    normal_extent = float(
        np.max(normal.shape.local_bounds()[1] - normal.shape.local_bounds()[0])
    )
    proxy_extent = float(
        np.max(proxy.shape.local_bounds()[1] - proxy.shape.local_bounds()[0])
    )
    target_extent = float(np.clip(normal_extent, 0.2, 3.0))
    assert 0.85 * target_extent <= proxy_extent <= 1.15 * target_extent
    assert normal.material != proxy.material

    alternate = sample_training_world(frames, (template,), grid, sensor, "mixed", 8)
    assert alternate.objects[0].label != first.objects[0].label
    counts[0] = (2, 1)
    remainder = sample_training_world(frames, (template,), grid, sensor, "mixed", 11)
    assert {item.label for item in remainder.objects[:2]} == {
        "normal-control",
        "anomaly-proxy",
    }
    assert remainder.objects[2].label == "normal-control"


def test_common_renderer_is_deterministic_for_pure_normal_world() -> None:
    frame, grid = _small_ray_fixture()
    world = WorldSpec(9, 99)
    sensor = SensorCalibration.constant(0.4)
    first = render_frame(frame, world, grid, sensor)
    second = render_frame(frame, world, grid, sensor)
    np.testing.assert_array_equal(first.source.xyzi, second.source.xyzi)
    np.testing.assert_array_equal(first.packed_labels, second.packed_labels)
    assert not bool(first.inserted_mask.any())
    assert not bool(first.anomaly_proxy_mask.any())
    assert np.all(first.unchanged_normal_mask)


def test_balanced_bce_is_empty_class_safe() -> None:
    logits = torch.tensor((0.0, 1.0, -1.0))
    targets = torch.tensor((True, False, False))
    valid = torch.ones(3, dtype=torch.bool)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    torch.testing.assert_close(
        balanced_bce_loss(logits, targets, valid),
        0.5 * raw[:1].mean() + 0.5 * raw[1:].mean(),
    )
    torch.testing.assert_close(
        balanced_bce_loss(logits, targets, torch.tensor((False, True, True))),
        raw[1:].mean(),
    )
    with pytest.raises(TrainingError, match="no valid"):
        balanced_bce_loss(logits, targets, torch.zeros(3, dtype=torch.bool))


def test_checkpoint_selection_global_band_does_not_drift() -> None:
    trainer = object.__new__(AJAETrainer)
    trainer.model = nn.Linear(1, 1)
    trainer.seed = 0
    trainer.condition = experiment_condition("B1")
    trainer.selection_rule = _selection_rule()
    trainer.best_key, trainer.best_world, trainer.best_state = None, -1, None
    trainer.maximum_primary, trainer.selection_candidates = None, []
    trainer.stale_evaluations = 0
    evidence = iter(
        (
            _development_evidence(0.5, 0.3),
            _development_evidence(0.4999994, 0.2),
            _development_evidence(0.4999988, 0.1),
        )
    )
    trainer.development_evaluator = lambda *_: next(evidence)
    for world_id in range(3):
        trainer._development_update(world_id, 1.0)
    assert trainer.maximum_primary == pytest.approx(0.5)
    assert trainer.best_world == 1
    assert trainer.best_key[0] >= trainer.maximum_primary - 1.0e-6


def test_checkpoint_key_structurally_requires_all_24_worlds() -> None:
    evidence = _development_evidence(0.6, 0.2)
    assert checkpoint_selection_key(_selection_rule(), evidence) == pytest.approx(
        (0.6, -0.2)
    )
    with pytest.raises(ValueError, match="exactly 24"):
        DevelopmentEvidence(evidence.in_generator[:-1], evidence.pure_normal)


def test_world_budget_exhaustion_cannot_publish_a_completed_model(
    tmp_path: Path,
) -> None:
    trainer = object.__new__(AJAETrainer)
    trainer.best_state = {}
    trainer.best_key = (0.5, -0.2)
    trainer.stop_reason = "maximum_worlds"
    with pytest.raises(TrainingError, match="development-patience"):
        trainer._finalize()

    trainer.phase = "between_worlds"
    trainer.commit_id = 3
    trainer.run_dir = tmp_path
    trainer.condition = experiment_condition("B1")
    trainer.seed = 0
    trainer.maximum_worlds = 20
    trainer.resume_world = 20
    trainer.best_world = 14
    trainer.scientific_identity = {"protocol_schema": 30}
    trainer.history = []
    trainer.save_progress = lambda: None
    result = trainer._record_budget_exhaustion()
    assert result["status"] == "budget_exhausted_unfinished"
    assert not (tmp_path / "model.pt").exists()


def test_formal_preflight_refuses_pending_evidence() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    worlds = load_development_worlds(DEVELOPMENT_PATH, protocol=protocol)
    document = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    with pytest.raises(TrainingError, match="not frozen and validated"):
        validate_formal_preflight(protocol, worlds, document)


def test_probability_fusion_uses_canonical_ray_probabilities() -> None:
    mapping = np.asarray((2, 0, 3, 1), dtype=np.int32)
    fusion = WindowScoreFusion(maximum_count=2)
    for probabilities in ((0.2, 0.8), (0.4, 0.6)):
        fusion.add(
            5,
            np.asarray((2, 3), dtype=np.int32),
            np.asarray(probabilities, dtype=np.float32),
            output_slots=np.asarray((0, 2), dtype=np.int32),
            slot_to_ray=mapping,
        )
    scores, counts = fusion.finalize(5)
    np.testing.assert_allclose(scores[[0, 2]], (0.3, 0.7))
    np.testing.assert_array_equal(counts[[0, 2]], (2, 2))


def test_common_frame_domain_and_coverage_forbid_zero_fill(tmp_path: Path) -> None:
    sequence = SimpleNamespace(
        frame_ids=tuple(range(10)),
        spec=SimpleNamespace(excluded_source_frames=()),
    )
    assert AJAEInference._comparison_frame_ids(sequence) == (4, 5, 6, 7)
    directory = tmp_path / "B3" / "125"
    directory.mkdir(parents=True)
    payload = {
        "format": "ajae-prediction-coverage-v1",
        "condition": "B3",
        "frame_domain": (
            "intersection_of_complete_centered_q0_and_complete_causal_current_frames"
        ),
        "frame_ids": [4, 5, 6, 7],
        "padding_or_zero_fill_used": False,
    }
    (directory / "coverage.json").write_text(json.dumps(payload))
    assert load_prediction_coverage(
        directory, condition="B3", expected_frame_ids=(4, 5, 6, 7)
    ) == (4, 5, 6, 7)
    payload["padding_or_zero_fill_used"] = True
    (directory / "coverage.json").write_text(json.dumps(payload))
    with pytest.raises(EvaluationError, match="frame domain"):
        load_prediction_coverage(
            directory, condition="B3", expected_frame_ids=(4, 5, 6, 7)
        )


def test_official_point_gate_and_moving_normal_safety() -> None:
    accumulator = PointMetricAccumulator()
    points = np.column_stack((np.full(8, 5.0), np.zeros((8, 2)))).astype(np.float32)
    scores = np.linspace(0.1, 0.8, 8, dtype=np.float32)
    semantic = np.asarray((10, 10, 10, 2, 2, 2, 2, 2), dtype=np.uint16)
    assert accumulator.update(points, scores, semantic)
    assert set(("AP", "AUROC", "FPR95")).issubset(accumulator.compute())
    moving = MovingNormalDiagnostic(0.5)
    moving.update(
        points[:4],
        np.asarray((0.1, 0.7, 0.2, 0.8), dtype=np.float32),
        np.asarray((252, 252, 10, 10), dtype=np.uint16),
    )
    safety = moving.compute()
    assert safety["moving_points"] == 2
    assert safety["moving_false_positive_rate"] == pytest.approx(0.5)


def test_point_metrics_match_released_stu_calculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    official_root = protocol.stu_repository_path().parent
    official_script = official_root / "compute_point_level_ood.py"
    monkeypatch.syspath_prepend(str(official_root))
    specification = importlib.util.spec_from_file_location(
        "ajae_test_official_point", official_script
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    official = module.PointOODMetricsCalculator()
    ours = PointMetricAccumulator(protocol)

    points = np.column_stack(
        (
            np.asarray((2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 50.0, 51.0)),
            np.zeros((10, 2)),
        )
    ).astype(np.float32)
    scores = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    semantic = np.asarray((10, 10, 10, 10, 2, 2, 2, 2, 2, 2), dtype=np.uint16)
    official.update(points, scores, semantic)
    assert ours.update(points, scores, semantic)

    skipped_semantic = np.asarray((10, 10, 10, 10, 10, 10, 2, 2, 2, 2))
    official.update(points, scores, skipped_semantic)
    assert not ours.update(points, scores, skipped_semantic)
    official_result = official.compute_metrics()
    our_result = ours.compute()
    for metric in ("AP", "AUROC", "FPR95", "threshold"):
        assert our_result[metric] == pytest.approx(official_result[metric])


def test_public_result_requires_complete_finite_19_sequence_evidence(
    tmp_path: Path,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    public_sequences: dict[str, object] = {}
    total_frames = 0
    for specification in protocol.public_validation:
        frame_count = len(
            set(specification.legal_anchors(RELATIVE_TIMES))
            & set(specification.legal_anchors(CAUSAL_OFFSETS))
        )
        total_frames += frame_count
        public_sequences[str(specification.sequence_id)] = {
            "comparison_frame_count": frame_count,
            "point": {
                "AP": 50.0,
                "AUROC": 60.0,
                "FPR95": 20.0,
                "threshold": 0.5,
                "accepted_frames": frame_count,
                "skipped_frames": 0,
            },
            "moving_normal": {
                "strict_threshold": 0.5,
                "moving_points": 1,
                "moving_mean": 0.2,
                "moving_false_positive_rate": 0.0,
                "static_points": 1,
                "static_mean": 0.1,
                "static_false_positive_rate": 0.0,
                "moving_minus_static_mean": 0.1,
            },
        }
    payload = {
        "format": "ajae-public-validation-result-v1",
        "protocol_schema": 30,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "condition": "B3",
        "seeds": {},
        "development_worlds": {},
        "public_sequences": public_sequences,
        "pooled": {
            "point": {
                "AP": 50.0,
                "AUROC": 60.0,
                "FPR95": 20.0,
                "threshold": 0.5,
                "accepted_frames": total_frames,
                "skipped_frames": 0,
            },
            "moving_normal": {
                "strict_threshold": 0.5,
                "moving_points": 19,
                "moving_mean": 0.2,
                "moving_false_positive_rate": 0.0,
                "static_points": 19,
                "static_mean": 0.1,
                "static_false_positive_rate": 0.0,
                "moving_minus_static_mean": 0.1,
            },
        },
        "cost": None,
        "method_freeze": {"object_score_threshold": 0.5},
    }
    result_path = tmp_path / "public.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_public_result(
        result_path, protocol=protocol, condition="B3"
    )["condition"] == "B3"
    payload["public_sequences"][str(protocol.public_sequence_ids[0])][
        "moving_normal"
    ] = {}
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationError, match="strict_threshold"):
        _validate_public_result(result_path, protocol=protocol, condition="B3")


def test_object_scale_diagnostic_excludes_normal_control_ids() -> None:
    diagnostic = ObjectScaleDiagnostic()
    points = np.asarray(
        ((5.0, 0.0, 0.0), (5.1, 0.0, 0.0), (5.2, 0.0, 0.0), (5.25, 0.0, 0.0)),
        dtype=np.float32,
    )
    diagnostic.update_window(
        world_id=0,
        window_id=0,
        points=points,
        scores=np.asarray((0.9, 0.8, 0.1, 0.2), dtype=np.float32),
        object_ids=np.asarray((2, 2, 1, -1), dtype=np.int32),
        relative_times=np.asarray((-1, 0, 0, 0), dtype=np.int8),
        raw_semantic=np.asarray((2, 2, 10, 10), dtype=np.uint16),
    )
    result = diagnostic.compute()
    assert len(result["objects"]) == 1
    assert result["objects"][0]["object_id"] == 2
    assert result["objects"][0]["visibility"] == 2
