#!/usr/bin/env python3
"""Focused scientific-semantic tests for the sole AJAE schema-30 route."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
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
from src.qualify import PHASE5_FRAMES, independent_sparse_quantize, phase5_frame_ids
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
    PROCEDURAL_GENERATOR_SCHEMA,
    HeldOutTorusShape,
    NormalTemplateShape,
    RayGrid,
    SensorCalibration,
    ShapeSpec,
    QualifiedSupportPool,
    ObservedObstacleIndex,
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
    FrameCache,
    FrameCacheKey,
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


def test_hidden_sequences_are_sealed_before_path_resolution(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    with pytest.raises(SceneDataError, match="sealed until"):
        STUSequence.open(
            tmp_path / "does-not-exist",
            protocol=protocol,
            partition="test",
            sequence_id=protocol.hidden_sequence_ids[0],
            label_mode=LabelMode.FORBIDDEN,
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


def test_ray_grid_round_trip_uses_calibrated_beam_origin() -> None:
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
        np.eye(4),
        _labels(np.full(2, 10, dtype=np.uint16)),
        partition="fixture",
        sequence_id=99,
    )
    np.testing.assert_allclose(grid.ranges(frame), distances, atol=1.0e-7)
    np.testing.assert_allclose(grid.points_from_ranges(distances, frame), xyzi[:, :3])
    assert grid.round_trip(frame)["maximum_point_error_m"] < 1.0e-7


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


def test_e45_overlap_weights_balance_common_population() -> None:
    rng = np.random.default_rng(4504)
    records: dict[str, list[np.ndarray]] = {
        name: [] for name in render_module._E45_UNIT_FIELDS
    }
    for source_id, shift in ((0, -0.15), (1, 0.15)):
        for index in range(320):
            values = rng.normal(size=5)
            values[0] = 15.0 + 2.0 * values[0] + shift
            values[1] = 60.0 + 3.0 * values[1] + shift
            nvis = max(1, int(round(np.exp(3.0 + 0.2 * values[2] + shift) - 1.0)))
            occlusion = float(np.clip(0.45 + 0.08 * values[3] + shift / 4.0, 0.26, 0.74))
            density = max(0.01, float(np.exp(0.5 + 0.2 * values[4] + shift) - 1.0))
            row = {
                "bank_seed": np.asarray(index + 1000 * source_id, dtype=np.int64),
                "source": np.asarray(source_id, dtype=np.uint8),
                "center_frame": np.asarray(index % 160, dtype=np.int16),
                "frame_id": np.asarray(index % 160, dtype=np.int16),
                "support_semantic": np.asarray(40, dtype=np.uint16),
                "range_bin": np.asarray(1, dtype=np.int8),
                "azimuth_sector": np.asarray(index % 4, dtype=np.int8),
                "median_distance_m": np.asarray(values[0]),
                "median_beam": np.asarray(values[1]),
                "Nvis": np.asarray(nvis, dtype=np.int32),
                "O_hat": np.asarray(occlusion),
                "local_density": np.asarray(density),
                "geometry_hits": np.asarray(nvis + 2, dtype=np.int32),
                "point_count": np.asarray(min(nvis, 64), dtype=np.int16),
                "point_features": np.zeros((64, 7), dtype=np.float64),
                "unit_hash": np.asarray(index + 10_000 * source_id, dtype=np.uint64),
            }
            for name, value in row.items():
                records[name].append(value)
    units = {}
    for name, values in records.items():
        stacked = np.stack(values)
        units[name] = stacked.reshape(128, 5, *stacked.shape[1:])
    first = render_module._e45_weighted_balance(units)
    second = render_module._e45_weighted_balance(units)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first["common_cell_key"].shape == (4, 4)
    assert np.all(first["effective_sample_size"] > 50)
    assert float(np.max(first["weighted_smd"])) < 1.0e-6
    assert float(np.max(first["weighted_ks"])) <= 0.06
    assert float(first["maximum_cell_mass_difference"]) < 1.0e-6


def test_e48_joint_fold_plan_never_splits_pairs_or_center_frames() -> None:
    centers = np.asarray(
        ((10, 10), (10, 11), (12, 13), (14, 12), (15, 16), (17, 17)),
        dtype=np.int16,
    )
    plan = render_module._e48_fold_plan(centers)
    pair_fold = plan["pair_center_frame_fold"]
    for fold in range(5):
        test = plan["fold_test_pair"][fold]
        train = plan["fold_train_pair"][fold]
        excluded = plan["fold_excluded_pair"][fold]
        np.testing.assert_array_equal(test | train | excluded, np.ones(6, dtype=np.bool_))
        assert not np.any(test & train)
        assert np.all(pair_fold[test] == fold)
        assert np.all(pair_fold[train] != fold)
        assert np.intersect1d(centers[test], centers[train]).size == 0


def test_e48_matched_pair_bootstrap_is_deterministic() -> None:
    labels = np.tile(np.asarray((0, 0, 1, 1), dtype=np.uint8), 4)
    scores = np.tile(np.asarray((0.1, 0.2, 0.8, 0.9)), 4)
    predictions = (scores >= 0.5).astype(np.uint8)
    point_pair = np.repeat(np.arange(4, dtype=np.int64), 4)
    weights = np.full(16, 0.5, dtype=np.float64)
    first = render_module._e48_pair_bootstrap(
        labels, scores, predictions, point_pair, weights, 4
    )
    second = render_module._e48_pair_bootstrap(
        labels, scores, predictions, point_pair, weights, 4
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.ones((2000, 4)))


def test_phase5_frame_identity_is_frozen_before_stu_outputs() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert phase5_frame_ids(protocol, 206) == PHASE5_FRAMES[206]
    assert phase5_frame_ids(protocol, 201) == PHASE5_FRAMES[201]


def test_independent_sparse_quantize_preserves_first_occurrence_rows() -> None:
    points = np.asarray(
        [[0.11, 0.0, 0.0], [0.01, 0.0, 0.0], [0.12, 0.0, 0.0],
         [-0.01, 0.0, 0.0], [-0.06, 0.0, 0.0]],
        dtype=np.float64,
    )
    rows, unique, inverse = independent_sparse_quantize(points, 0.05)
    np.testing.assert_array_equal(rows[:, 0], [2, 0, -1, -2])
    np.testing.assert_array_equal(unique, [0, 1, 3, 4])
    np.testing.assert_array_equal(inverse, [0, 1, 0, 2, 3])


def test_shared_stu_features_retain_one_final_logit_per_raw_point() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    torch.manual_seed(5200)
    model = AJAEPointTransformer.from_protocol(protocol).eval()
    coordinates = torch.tensor(
        [[0.001, 0.0, 0.0], [0.049, 0.0, 0.0],
         [0.101, 0.0, 0.0], [0.149, 0.0, 0.0]]
    )
    times = torch.zeros(4, dtype=torch.long)
    shared_features = torch.ones(4, 128)
    evidence = torch.zeros(4, 19)
    reliability = torch.zeros(4)
    intensity = torch.tensor([0.1, 0.9, 0.2, 0.8])
    order = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        logits = model(
            coordinates, times, shared_features, evidence, reliability,
            reliability, intensity, cross_frame_enabled=False,
        )
        permuted = model(
            coordinates[order], times[order], shared_features[order],
            evidence[order], reliability[order], reliability[order],
            intensity[order], cross_frame_enabled=False,
        )
    assert logits.shape == (4,)
    torch.testing.assert_close(permuted, logits[order], rtol=0.0, atol=0.0)


def test_schema4_development_worlds_are_rejected_after_world_v3_freeze() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    with pytest.raises(ProtocolError, match="authoritative WorldSpec"):
        load_development_worlds(DEVELOPMENT_PATH, protocol=protocol)


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
    with pytest.raises(ProtocolError, match="authoritative WorldSpec"):
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
        lower, upper = (
            training_shape.continuous_bounds(
                maximum_iterations=80, population_size=10
            )
            if training_shape.primitive_count == 1
            else training_shape.tight_continuous_outer_bounds()
        )
        assert 0.2 <= float(np.max(upper - lower)) <= 3.0
        report = training_shape.geometry_report()
        assert report["bounded"] and report["closed"] and report["components"] == 1


def test_generator_schema7_preserves_single_continuous_acceptance() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    historical = protocol.render["anomaly_proxies"][
        "continuous_size_acceptance_generator"
    ]
    current = protocol.render["anomaly_proxies"]["integrated_generator_schema7"]
    assert historical["generator_schema"] == 2
    assert PROCEDURAL_GENERATOR_SCHEMA == current["generator_schema"] == 7

    for seed in range(8):
        shape, report = ShapeSpec.sample_with_report(seed, primitive_count=1)
        repeated_shape, repeated_report = ShapeSpec.sample_with_report(
            seed, primitive_count=1
        )
        assert shape.to_dict() == repeated_shape.to_dict()
        assert report == repeated_report
        assert report.generator_schema == 7
        assert report.shape_family in {"general", "blocky", "flat", "elongated"}
        assert report.child_parent_indices == ()
        assert report.shared_witnesses_undeformed_m == ()
        assert report.witness_parent_margins_m == ()
        assert report.witness_child_margins_m == ()
        assert report.size_definition == "continuous-deformed-surface-aabb"
        assert 0.2 <= report.accepted_size_lower_m
        assert report.accepted_size_upper_m <= 3.0
        lower, upper = shape.continuous_bounds(
            maximum_iterations=80, population_size=10
        )
        np.testing.assert_array_equal(report.outer_lower_m, lower)
        np.testing.assert_array_equal(report.outer_upper_m, upper)
        expected = float(np.max(upper - lower))
        assert report.accepted_size_lower_m == expected
        assert report.accepted_size_upper_m == expected


def test_generator_schema7_base_families_keep_the_qualified_supports() -> None:
    observed: Counter[str] = Counter()
    for seed in range(4096):
        scale, family = ShapeSpec._schema7_base_scale(seed, 1.0)
        ordered = np.sort(np.asarray(scale))[::-1]
        r21, r31 = ordered[1] / ordered[0], ordered[2] / ordered[0]
        observed[family] += 1
        if family == "blocky":
            assert 0.75 <= r31 <= r21 <= 1.0
        elif family == "flat":
            assert 0.75 <= r21 <= 1.0 and 0.20 <= r31 <= 0.40
        elif family == "elongated":
            assert 0.30 <= r21 <= 0.50 and 0.15 <= r31 <= min(0.40, r21)
        else:
            assert 0.0 < r31 <= r21 <= 1.0
    assert observed == {
        "general": 1647,
        "blocky": 846,
        "flat": 809,
        "elongated": 794,
    }


def test_generator_schema7_keeps_all_nonreplaced_schema6_draws() -> None:
    seed, count = 0, 5
    shape, report = ShapeSpec.sample_with_report(seed, primitive_count=count)
    assert report.proposal_count == 1
    rng = np.random.default_rng(seed)
    half = float(rng.uniform(0.1, 1.5))
    rng.uniform(0.65, 1.25, size=3)
    base, _ = ShapeSpec._schema7_base_scale(seed, half)
    expected_scales = [base]
    expected_exponents = [tuple(rng.uniform(0.55, 1.65, 2))]
    expected_yaws = [float(rng.uniform(-np.pi, np.pi))]
    expected_parents = []
    for child_index in range(1, count):
        expected_parents.append(int(rng.integers(0, child_index)))
        rng.integers(0, 3)
        rng.integers(0, 2)
        rng.uniform(0.10, 0.50)
        expected_scales.append(tuple(np.asarray(base) * rng.uniform(0.32, 0.78, 3)))
        expected_exponents.append(tuple(rng.uniform(0.5, 1.8, 2)))
        expected_yaws.append(float(rng.uniform(-np.pi, np.pi)))
    expected_amplitude = float(rng.uniform(0.0, 0.08 * min(base)))
    expected_twist = float(rng.uniform(-0.65, 0.65))
    expected_bend = tuple(rng.uniform(-0.12, 0.12, 2))
    expected_taper = tuple(rng.uniform(-0.18, 0.18, 2))
    expected_frequency = tuple(rng.uniform(0.6, 2.2, 3))
    expected_phase = tuple(rng.uniform(-np.pi, np.pi, 3))
    np.testing.assert_array_equal(shape.primitive_scales_m, expected_scales)
    np.testing.assert_array_equal(shape.primitive_exponents, expected_exponents)
    np.testing.assert_array_equal(shape.primitive_yaws_rad, expected_yaws)
    assert report.child_parent_indices == tuple(expected_parents)
    assert shape.surface_amplitude_m == expected_amplitude
    assert shape.twist_rad_per_m == expected_twist
    assert shape.bend_per_m == expected_bend
    assert shape.taper_per_m == expected_taper
    assert shape.surface_frequency_per_m == expected_frequency
    assert shape.surface_phase_rad == expected_phase


def test_generator_schema7_uses_tight_union_certificate_for_multi_primitive() -> None:
    shape, report = ShapeSpec.sample_with_report(0, primitive_count=2)
    certificate = shape.continuous_size_certificate(
        sobol_probes=4096, maximum_interior_lines=64
    )
    tight_lower, tight_upper = shape.tight_continuous_outer_bounds()
    assert report.generator_schema == 7
    assert report.size_definition == "continuous-union-tight-certified-interval"
    assert report.accepted_size_lower_m == certificate.lower_size_m
    assert report.accepted_size_upper_m == float(np.max(tight_upper - tight_lower))
    np.testing.assert_array_equal(report.outer_lower_m, tight_lower)
    np.testing.assert_array_equal(report.outer_upper_m, tight_upper)
    assert 0.2 <= report.accepted_size_lower_m
    assert report.accepted_size_upper_m <= 3.0


def test_generator_schema7_records_the_authoritative_overlap_tree() -> None:
    for primitive_count in range(2, 6):
        for seed in range(8):
            shape, report = ShapeSpec.sample_with_report(
                seed, primitive_count=primitive_count
            )
            assert report.generator_schema == 7
            assert shape.operations == ("union",) * primitive_count
            assert shape.connectivity_certificate.source == "connected_union_graph"
            assert len(report.child_parent_indices) == primitive_count - 1
            assert len(report.shared_witnesses_undeformed_m) == primitive_count - 1
            assert all(
                shape._primitive_star_certificate(index)
                for index in range(primitive_count)
            )
            for index in range(1, primitive_count):
                parent = report.child_parent_indices[index - 1]
                assert 0 <= parent < index
                witness = np.asarray(report.shared_witnesses_undeformed_m[index - 1])[None]
                parent_value = float(shape._primitive_perturbed_value(parent, witness)[0])
                child_value = float(shape._primitive_perturbed_value(index, witness)[0])
                assert parent_value < 0.0 and child_value < 0.0
                assert report.witness_parent_margins_m[index - 1] == -parent_value
                assert report.witness_child_margins_m[index - 1] == -child_value


@pytest.mark.parametrize(
    ("seed", "origin", "direction", "reference_root"),
    (
        (
            99,
            (0.7682490608623493, 0.5460731616297538, 4.583701428336049),
            (-0.14755561851059673, -0.21011651960391545, -0.9664773083914038),
            4.3844959571405315,
        ),
        (
            15,
            (-0.8973395825762842, -0.02533799707999456, -4.3863570066163655),
            (0.26606760813298547, -0.067353110887987, 0.9615984538028869),
            3.7124355859608693,
        ),
    ),
)
def test_intersection_refines_a_narrow_segment_before_a_later_coarse_hit(
    seed: int,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    reference_root: float,
) -> None:
    shape, _ = ShapeSpec.sample_with_report(seed)
    distance, normal, hit = shape.intersect(
        np.asarray(origin), np.asarray(direction)[None]
    )
    assert hit[0]
    assert abs(distance[0] - reference_root) <= 1.0e-4
    assert np.isfinite(normal[0]).all()
    assert abs(np.linalg.norm(normal[0]) - 1.0) <= 1.0e-12


def test_tight_continuous_outer_bound_is_conservative_and_no_looser() -> None:
    shape = ShapeSpec(
        ((1.0, 0.6, 0.8), (0.5, 0.4, 0.6)),
        ((0.0, 0.0, 0.0), (0.3, 0.0, 0.1)),
        ((1.0, 0.8), (1.2, 1.1)),
        (0.35, -0.4),
        ("union", "union"),
        twist_rad_per_m=0.7,
        bend_per_m=(0.04, -0.03),
        taper_per_m=(0.08, -0.06),
        surface_amplitude_m=0.01,
    )
    old_lower, old_upper = shape._continuous_outer_bounds()
    new_lower, new_upper = shape.tight_continuous_outer_bounds()
    assert np.all(new_lower >= old_lower)
    assert np.all(new_upper <= old_upper)
    rng = np.random.default_rng(20260826)
    points = rng.uniform(old_lower, old_upper, size=(131_072, 3))
    inside = points[shape.signed_distance(points) <= 0.0]
    assert len(inside) > 0
    assert np.all(inside >= new_lower)
    assert np.all(inside <= new_upper)


def test_continuous_primitive_bounds_match_an_analytic_rotated_ellipsoid() -> None:
    scales = (0.3, 0.7, 1.1)
    yaw = 0.4
    shape = ShapeSpec(
        (scales,),
        ((0.0, 0.0, 0.0),),
        ((1.0, 1.0),),
        (yaw,),
        ("union",),
    )
    lower, upper = shape.continuous_bounds(
        maximum_iterations=80, population_size=10
    )
    expected = np.asarray(
        (
            np.hypot(scales[0] * np.cos(yaw), scales[1] * np.sin(yaw)),
            np.hypot(scales[0] * np.sin(yaw), scales[1] * np.cos(yaw)),
            scales[2],
        )
    )
    np.testing.assert_allclose(lower, -expected, atol=5.0e-6, rtol=0.0)
    np.testing.assert_allclose(upper, expected, atol=5.0e-6, rtol=0.0)


def test_csg_continuous_size_certificate_encloses_analytic_lens() -> None:
    shape = ShapeSpec(
        ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
        ((1.0, 1.0), (1.0, 1.0)),
        (0.0, 0.0),
        ("union", "intersection"),
    )
    standard = shape.continuous_size_certificate(
        sobol_probes=4096, maximum_interior_lines=64
    )
    strict = shape.continuous_size_certificate(
        sobol_probes=32768, maximum_interior_lines=256
    )
    exact_size = np.sqrt(3.0)
    assert standard.lower_size_m <= exact_size <= standard.upper_size_m
    assert strict.lower_size_m >= standard.lower_size_m
    assert strict.outer_lower_m == standard.outer_lower_m
    assert strict.outer_upper_m == standard.outer_upper_m
    assert standard.maximum_surface_residual_m < 1.0e-8


def test_training_world_uses_only_the_qualified_placement_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    anchors = np.column_stack((4.0 * x.ravel(), 4.0 * y.ravel(), np.zeros(x.size)))
    count = anchors.shape[0]
    pool = QualifiedSupportPool(
        np.arange(count), np.full(count, 40, np.uint16), np.arange(count) % 449,
        np.arange(count), np.linalg.norm(anchors, axis=1), np.arange(count, dtype=np.uint64),
        anchors, np.tile((0.0, 0.0, 1.0), (count, 1)), np.zeros(count),
    )
    obstacles = ObservedObstacleIndex(
        np.asarray(((1000.0, 1000.0, 1000.0),)), np.asarray((1,), np.uint64)
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
    counts = [(1, 1)]
    monkeypatch.setattr(render_module, "_training_entity_counts", lambda *_: counts[0])
    fixture_frame, grid = _small_ray_fixture()
    frames = tuple(make_source_frame(
        frame_id,
        fixture_frame.xyzi,
        fixture_frame.lidar_pose,
        fixture_frame.labels,
        partition="train",
        sequence_id=206,
    ) for frame_id in range(449))
    context = render_module.build_coverage_control_context(
        frames, pool, grid, SensorCalibration.constant(1.0)
    )
    observed_worlds: list[tuple[int, tuple[int, ...]]] = []

    def passing_observation(
        _: object, item: object, patch: object, world_seed: int,
        assigned_range_bin: int, world_objects: object,
    ) -> object:
        observed_worlds.append((
            world_seed, tuple(value.object_id for value in world_objects)
        ))
        return render_module._E25NewObservation(
            patch.frame_id, 1, 1, 1, 1, 5.0, 0.0,
            assigned_range_bin, 0, 0.0,
        )

    monkeypatch.setattr(
        render_module, "_coverage_control_observation", passing_observation
    )
    first, first_report = sample_training_world(
        (template,), pool, obstacles, "mixed", 7,
        control_context=context,
    )
    repeated, repeated_report = sample_training_world(
        (template,), pool, obstacles, "mixed", 7,
        control_context=context,
    )
    assert first.to_dict() == repeated.to_dict()
    assert first_report.to_dict() == repeated_report.to_dict()
    assert render_module.WorldGenerationReport.from_dict(
        json.loads(first_report.to_json())
    ).to_json() == first_report.to_json()
    assert first_report.normal_count == 1
    assert first_report.anomaly_count == 1
    assert first_report.count_seed == 7
    assert first.world_type == "mixed"
    assert len(first_report.placements) == 2
    assert all(record.support_semantic == 40 for record in first_report.placements)
    assert all(record.accepted_proposal < 128 for record in first_report.placements)
    assert all(not record.rejection_reasons or set(record.rejection_reasons) <= {
        "observed_normal_deep_penetration", "obvious_pair_penetration",
    } for record in first_report.placements)
    assert observed_worlds
    assert all(world_seed == 7 for world_seed, _ in observed_worlds)
    assert any(len(object_ids) == 2 for _, object_ids in observed_worlds)


def test_shape_stream_rejects_e22_invalid_shape_before_support_sampling() -> None:
    _, _, grounding, proposed, rejected = render_module._grounding_qualified_shape(
        3_000_471, stride=3072, maximum_proposals=64
    )
    assert proposed == (3_000_471, 3_003_543)
    assert rejected == (3_000_471,)
    assert grounding.passed


def test_e25_new_contract_has_one_fixed_template_range_assignment() -> None:
    assigned = np.asarray([
        render_module._e25_new_assigned_range_bin(index)
        for index in range(256)
    ])
    np.testing.assert_array_equal(
        np.bincount(assigned, minlength=5), np.asarray((52, 51, 51, 51, 51))
    )
    arguments = render_module._render_parser().parse_args([
        "qualify-e25-new-normal-control",
        "--data-root", "/data",
        "--support-pool", "support.npz",
        "--calibration", "calibration.pt",
        "--output", "e25-new.npz",
    ])
    assert arguments.processes == 24
    e26 = render_module._render_parser().parse_args([
        "qualify-e26-v2",
        "--data-root", "/data",
        "--support-pool", "support.npz",
        "--calibration", "calibration.pt",
        "--output", "e26-v2.npz",
    ])
    assert e26.processes == 24
    e38 = render_module._render_parser().parse_args([
        "qualify-e38-v2",
        "--data-root", "/data",
        "--e25-new-artifact", "e25-new.npz",
        "--support-pool", "support-201.npz",
        "--calibration", "calibration.pt",
        "--candidate-bank-output", "gate1-v2.npz",
        "--output", "e38-v2.npz",
    ])
    assert e38.processes == 24
    assert e38.e25_new_artifact == Path("e25-new.npz")
    assert e38.support_pool == Path("support-201.npz")


def test_gate1_v2_template_draw_and_five_frame_filter_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for attempt_seed in (3_800_000, 4_800_003, 51_800_141):
        expected = int(
            np.random.default_rng(attempt_seed + 1).integers(0, 256)
        )
        assert render_module._gate1_control_template_assignment(
            attempt_seed
        ) == (expected, expected % 5)

    frames = np.asarray((4, 8, 9, 10, 12, 13), dtype=np.int16)
    context = SimpleNamespace(
        support_pool=SimpleNamespace(frames=frames)
    )
    global_stream = np.asarray((5, 2, 4, 1, 3, 0), dtype=np.int64)
    monkeypatch.setattr(
        render_module,
        "_coverage_control_support_stream",
        lambda *_: global_stream,
    )
    selected = render_module._gate1_control_rows(context, 7, 10, 2, 10)
    np.testing.assert_array_equal(selected, np.asarray((2, 4, 1, 3)))


def test_streaming_control_context_loads_only_requested_frames() -> None:
    fixture, grid = _small_ray_fixture()
    frames = {
        frame_id: make_source_frame(
            frame_id,
            fixture.xyzi,
            fixture.lidar_pose,
            fixture.labels,
            partition="train",
            sequence_id=201,
        )
        for frame_id in (0, 1)
    }
    pool = QualifiedSupportPool(
        np.asarray((0, 1)), np.asarray((40, 40), dtype=np.uint16),
        np.asarray((0, 1)), np.asarray((0, 1)), np.asarray((4.0, 4.0)),
        np.asarray((0, 1), dtype=np.uint64),
        np.asarray(((4.0, 0.0, 0.0), (4.0, 0.0, 0.0))),
        np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))), np.zeros(2),
    )
    requested: list[int] = []

    def load(frame_id: int) -> object:
        requested.append(frame_id)
        return frames[frame_id]

    context = render_module.build_coverage_control_context(
        (), pool, grid, SensorCalibration.constant(1.0),
        frame_loader=load, frame_ids=(0, 1), source_sequence_id=201,
        trajectory_yaws={0: 0.0, 1: 0.0},
    )
    assert context.frames_by_id == {}
    first = render_module._e25_new_frame_context(context, 0)
    repeated = render_module._e25_new_frame_context(context, 0)
    assert first is repeated
    assert requested == [0]
    assert context.frames_by_id == {}
    wrong = render_module.build_coverage_control_context(
        (), pool, grid, SensorCalibration.constant(1.0),
        frame_loader=lambda _: frames[1], frame_ids=(0, 1),
        source_sequence_id=201, trajectory_yaws={0: 0.0, 1: 0.0},
    )
    with pytest.raises(render_module.RenderError, match="identity changed"):
        render_module._e25_new_frame_context(wrong, 0)


def test_gate1_v2_loader_rejects_historical_bank_schema(tmp_path: Path) -> None:
    artifact = tmp_path / "historical-gate1.npz"
    np.savez_compressed(
        artifact,
        metadata_json=np.asarray(json.dumps({
            "experiment": "Gate1-candidate-bank-v1",
            "passed": True,
            "capacity": 256,
        })),
    )
    with pytest.raises(render_module.RenderError, match="not qualified"):
        render_module._load_gate1_bank(artifact)


def test_gate1_trace_uses_the_official_range_offset() -> None:
    columns = 8
    azimuth = np.arange(columns, dtype=np.float64) * (-2.0 * np.pi / columns)
    directions = np.stack(
        (np.cos(azimuth), np.sin(azimuth), np.zeros(columns)), axis=1
    )
    grid = RayGrid(
        directions, np.zeros(1), azimuth, beam_count=1,
        official_range_offset_m=0.2,
    )
    xyzi = np.zeros((columns, 4), dtype=np.float32)
    frame = make_source_frame(
        8, xyzi, np.eye(4), _labels(np.zeros(columns, dtype=np.uint16)),
        partition="train", sequence_id=201,
    )
    vertices = np.asarray([
        (x_value, y_value, z_value)
        for x_value in (-0.1, 0.1)
        for y_value in (-0.2, 0.2)
        for z_value in (-0.2, 0.2)
    ])
    shape = NormalTemplateShape(
        vertices, np.empty((0, 3), dtype=np.int32),
        206, 0, 10, 1, (0.0, 0.0, 0.0),
    )
    item = render_module.ObjectSpec(
        1, "normal-control", shape, render_module.MaterialSpec(0.5, 0.2),
        (9.93, 0.0, 0.0),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    world = WorldSpec(3_800_000, 201, (item,))
    geometry, accepted, distance, rendered = (
        render_module._gate1_single_object_trace(
            frame, world, grid, SensorCalibration.constant(1.0)
        )
    )
    returned = np.asarray(rendered.normal_control_mask)
    assert geometry[0] and accepted[0] and returned[0]
    assert distance[0] == pytest.approx(10.03, abs=1.0e-10)
    assert render_module._gate1_range_bin(distance)[0] == 1
    rendered_range = np.asarray(grid.official_ranges(rendered.source))[returned]
    assert rendered_range[0] == pytest.approx(10.03, abs=1.0e-6)


def test_e38_shared_trace_contract_cross_checks_all_count_views() -> None:
    _, grid = _small_ray_fixture()
    units = tuple(
        SimpleNamespace(bank_seed=3_800_000 + index, center_frame=10)
        for index in range(256)
    )
    seeds = np.broadcast_to(
        np.arange(3_800_000, 3_800_256, dtype=np.int64)[:, None, None],
        (256, 3, 5),
    ).copy()
    sources = np.broadcast_to(
        np.arange(3, dtype=np.uint8)[None, :, None], (256, 3, 5)
    ).copy()
    frames = np.broadcast_to(
        np.arange(8, 13, dtype=np.int16)[None, None, :], (256, 3, 5)
    ).copy()
    trace = {
        "bank_seed": seeds, "source": sources, "frame_id": frames,
        "support_semantic": np.full((256, 3, 5), 40, dtype=np.uint16),
        "opportunity": np.zeros((256, 3, 5, 1), dtype=np.int32),
        "return_count": np.zeros((256, 3, 5, 1), dtype=np.int32),
        "median_distance_m": np.zeros((256, 3, 5)),
        "median_beam": np.zeros((256, 3, 5)),
        "range_opportunity": np.zeros((256, 3, 5, 5), dtype=np.int32),
        "range_return_count": np.zeros((256, 3, 5, 5), dtype=np.int32),
        "geometry_hits": np.zeros((256, 3, 5), dtype=np.int32),
        "accepted_hits": np.zeros((256, 3, 5), dtype=np.int32),
        "visible_returns": np.zeros((256, 3, 5), dtype=np.int32),
        "visible_distance_m": np.zeros((256, 3, 5)),
        "empty_slots": np.zeros((256, 2, 5, 1), dtype=np.int32),
        "empty_geometry": np.zeros((256, 2, 5, 1, 5), dtype=np.int32),
        "empty_accepted": np.zeros((256, 2, 5, 1, 5), dtype=np.int32),
        "empty_final_new": np.zeros((256, 2, 5, 1, 5), dtype=np.int32),
        "intensity_source": np.asarray((1,), dtype=np.uint8),
        "intensity_bank_seed": np.asarray((3_800_000,), dtype=np.int64),
        "intensity_frame": np.asarray((8,), dtype=np.int16),
        "intensity_slot": np.asarray((0,), dtype=np.int32),
        "intensity_beam": np.asarray((0,), dtype=np.int16),
        "intensity_official_range_m": np.asarray((5.0,)),
        "intensity_range_bin": np.asarray((0,), dtype=np.int8),
        "intensity_value": np.asarray((0.5,), dtype=np.float32),
    }
    trace["opportunity"][0, 1, 0, 0] = 1
    trace["return_count"][0, 1, 0, 0] = 1
    trace["range_opportunity"][0, 1, 0, 0] = 1
    trace["range_return_count"][0, 1, 0, 0] = 1
    trace["geometry_hits"][0, 1, 0] = 1
    trace["accepted_hits"][0, 1, 0] = 1
    trace["visible_returns"][0, 1, 0] = 1
    assert render_module._e38_trace_contract_errors(trace, grid, units) == 0
    trace["range_return_count"][0, 1, 0, 0] = 0
    assert render_module._e38_trace_contract_errors(trace, grid, units) > 0


def test_e26_manifest_audit_does_not_reclassify_finite_sampling_exhaustion() -> None:
    records = ({"hard_error": 0, "placement_exhaustion": 1},)
    assert render_module._e26_single_manifest_errors(records) == 0


def test_placement_postcheck_rejects_only_the_current_support_proposal() -> None:
    vertices = np.asarray([
        (x_value, y_value, z_value)
        for x_value in (-1.0, 1.0)
        for y_value in (-0.5, 0.5)
        for z_value in (-0.5, 0.5)
    ])
    template = NormalTemplateShape(
        vertices, np.empty((0, 3), dtype=np.int32),
        206, 0, 10, 1, (0.0, 0.0, 0.0),
    )
    pool = QualifiedSupportPool(
        np.asarray((0, 1)), np.asarray((40, 40), dtype=np.uint16),
        np.asarray((0, 1)), np.asarray((0, 1)), np.asarray((5.0, 6.0)),
        np.asarray((0, 1), dtype=np.uint64),
        np.asarray(((5.0, 0.0, 0.0), (6.0, 0.0, 0.0))),
        np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))),
        np.zeros(2),
    )
    obstacles = ObservedObstacleIndex(
        np.asarray(((1000.0, 1000.0, 1000.0),)), np.asarray((1,), np.uint64)
    )
    checked: list[int] = []

    def postcheck(_: object, patch: object) -> str | None:
        checked.append(patch.pool_index)
        return "fixture_sensor_rejection" if len(checked) == 1 else None

    _, record = render_module.place_object(
        template,
        render_module.MaterialSpec(0.5, 0.2),
        pool,
        obstacles,
        object_id=1,
        label="normal-control",
        proposal_namespace="fixture",
        proposal_stream=0,
        yaw_rad=0.0,
        material_seed=1,
        yaw_seed=2,
        template_identity="fixture",
        proposal_rows=(0, 1),
        post_placement_rejection=postcheck,
    )
    assert checked == [0, 1]
    assert record.accepted_proposal == 1
    assert record.rejection_reasons == ("fixture_sensor_rejection",)


def test_e25_new_sparse_trace_is_rechecked_by_the_complete_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_frame, grid = _small_ray_fixture()
    frame = make_source_frame(
        0,
        fixture_frame.xyzi,
        fixture_frame.lidar_pose,
        fixture_frame.labels,
        partition="train",
        sequence_id=206,
    )
    vertices = np.asarray([
        (x_value, y_value, z_value)
        for x_value in (-1.0, 1.0)
        for y_value in (-0.5, 0.5)
        for z_value in (-0.5, 0.5)
    ])
    template = NormalTemplateShape(
        vertices, np.empty((0, 3), dtype=np.int32),
        206, 0, 10, 1, (0.0, 0.0, 0.0),
    )
    item = render_module.ObjectSpec(
        1,
        "normal-control",
        template,
        render_module.MaterialSpec(0.5, 0.2),
        (4.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    sensor = SensorCalibration.constant(0.5)
    pool = QualifiedSupportPool(
        np.asarray((0,)), np.asarray((40,), dtype=np.uint16),
        np.asarray((0,)), np.asarray((0,)), np.asarray((4.0,)),
        np.asarray((0,), dtype=np.uint64),
        np.asarray(((4.0, 0.0, 0.0),)),
        np.asarray(((0.0, 0.0, 1.0),)), np.zeros(1),
    )
    context = render_module.build_coverage_control_context(
        (
            frame,
            make_source_frame(
                1,
                fixture_frame.xyzi,
                fixture_frame.lidar_pose,
                fixture_frame.labels,
                partition="train",
                sequence_id=206,
            ),
        ),
        pool,
        grid,
        sensor,
    )
    monkeypatch.setattr(render_module, "_E25_NEW_CONTROL_CONTEXT", context)
    patch = render_module.SupportPatch(
        0, 40, 0, 0, 4.0, 0,
        (4.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0,
    )
    observation = render_module._e25_new_observation(item, patch, 2_500_000, 0)
    assert observation.visible_returns >= 1
    assert observation.range_bin == 0


def test_normal_template_pca_axis_is_aligned_before_support_pose() -> None:
    angle = math.radians(37.0)
    vertices = np.asarray([
        (x_value, y_value, z_value)
        for x_value in (-2.0, 2.0)
        for y_value in (-0.5, 0.5)
        for z_value in (-0.5, 0.5)
    ])
    rotation = np.asarray([
        (math.cos(angle), -math.sin(angle)),
        (math.sin(angle), math.cos(angle)),
    ])
    vertices[:, :2] = vertices[:, :2] @ rotation.T
    source = NormalTemplateShape(
        vertices, np.empty((0, 3), dtype=np.int32),
        206, 0, 10, 1, (0.0, 0.0, 0.0),
    )
    aligned = render_module._aligned_scaled_template(source, (1.0, 1.0, 1.0))
    covariance = np.cov(aligned.vertices_m[:, :2], rowvar=False, bias=True)
    assert covariance[0, 0] > covariance[1, 1]
    assert abs(covariance[0, 1]) < 1.0e-10


def test_normal_template_surface_sampling_uses_a_convex_hull_interior() -> None:
    vertices = np.asarray([
        (x_value, y_value, z_value)
        for x_value in (2.0, 4.0)
        for y_value in (-0.5, 0.5)
        for z_value in (-0.5, 0.5)
    ])
    template = NormalTemplateShape(
        vertices, np.empty((0, 3), dtype=np.int32),
        206, 0, 10, 1, (0.0, 0.0, 0.0),
    )
    points = render_module._fibonacci_surface_points(template, 256)
    assert points.shape == (256, 3)
    assert np.max(np.abs(template.signed_distance(points))) < 1.0e-10


def test_placement_authority_audit_ignores_its_own_string_literals() -> None:
    source = Path(render_module.__file__).read_text(encoding="utf-8")
    assert render_module._placement_authority_errors(source) == 0


def test_xy_hull_distance_uses_closed_polygon_euclidean_distance() -> None:
    polygon = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    equations = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, -1.0),
         (0.0, 1.0, -1.0), (-1.0, 0.0, 0.0))
    )
    points = np.asarray(((0.5, 0.5), (1.3, 0.4), (-0.3, -0.4)))
    assert np.allclose(
        render_module._xy_hull_distance(points, polygon, equations),
        np.asarray((0.0, 0.3, 0.5)),
    )


@pytest.mark.parametrize("slope_deg", [0.0, 5.0, 10.0])
def test_e21_support_plane_fixtures(slope_deg: float) -> None:
    coordinate = np.linspace(-1.25, 1.25, 51)
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    slope = math.radians(slope_deg)
    z = np.tan(slope) * x + 0.002 * np.sin(7.0 * x + 3.0 * y)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    result = render_module.qualify_support_plane(
        points, np.zeros(3, dtype=np.float64)
    )
    assert result.qualified
    expected = np.asarray((-math.sin(slope), 0.0, math.cos(slope)))
    angle = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(result.estimates[1].normal, expected), -1.0, 1.0
                )
            )
        )
    )
    assert angle <= 0.5
    assert abs(result.estimates[1].anchor_height_m) <= 0.01


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


def test_frame_cache_isolates_same_frame_across_worlds() -> None:
    cache = FrameCache(4)
    digest = lambda value: hashlib.sha256(value.encode("ascii")).hexdigest()
    trainer = object.__new__(AJAETrainer)
    trainer.training_source_identity = digest("train/206/content")
    trainer.renderer_generator_identity = digest("renderer-generator-v1")
    trainer.stu_identity = digest("stu-v1")
    first_key = trainer._cache_key(WorldSpec(9, 206), 7)
    second_key = trainer._cache_key(WorldSpec(10, 206), 7)
    assert isinstance(first_key, FrameCacheKey)
    assert first_key.frame_identity == second_key.frame_identity
    assert first_key.world_identity != second_key.world_identity
    first_uncached = np.asarray((1.0, 7.0), dtype=np.float32)
    second_uncached = np.asarray((2.0, 7.0), dtype=np.float32)
    first_cached = cache.rendered_frame(first_key, lambda: first_uncached.copy())
    second_cached = cache.rendered_frame(second_key, lambda: second_uncached.copy())
    np.testing.assert_array_equal(first_cached, first_uncached)
    np.testing.assert_array_equal(second_cached, second_uncached)
    assert not np.array_equal(first_cached, second_cached)
    np.testing.assert_array_equal(
        cache.rendered_frame(first_key, lambda: np.asarray((-1.0, -1.0))),
        first_uncached,
    )
    first_encoded = cache.encoded_frame(first_key, lambda: first_uncached.copy())
    second_encoded = cache.encoded_frame(second_key, lambda: second_uncached.copy())
    np.testing.assert_array_equal(first_encoded, first_uncached)
    np.testing.assert_array_equal(second_encoded, second_uncached)


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
    with pytest.raises(ProtocolError, match="authoritative WorldSpec"):
        load_development_worlds(DEVELOPMENT_PATH, protocol=protocol)


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
