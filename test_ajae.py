from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluate import (
    EvaluationError,
    f3_final_action,
    f3_screen_action,
    f2_point_masks,
    geometry_record,
    official_point_metrics,
    require_experiment_stage,
    score_window,
    window_stu_inputs,
)
from src.model import (
    MASK_DIM,
    NUM_NORMAL_CLASSES,
    NUM_QUERIES,
    STUPointEncoding,
    assigned_stu_evidence,
    official_stu_semantic_class,
    official_stu_sparse_quantize,
    stu_input_identity,
    stu_source_manifest,
)
from src.protocol import (
    InputMode,
    ProtocolError,
    SCHEMA_VERSION,
    FrameSpan,
    SequenceSpec,
    load_protocol,
)
from src.qualify import run_schema33_qualification
from src.render import (
    QualifiedSupportPool,
    MaterialSpec,
    ObjectSpec,
    PlacementError,
    RayGrid,
    SensorCalibration,
    ShapeSpec,
    WorldGenerationReport,
    WorldSpec,
    _identity_order,
    render_development_clip_world,
    render_frame,
)
from src.scene import (
    PointLabels,
    assemble_window,
    canonical_ray_mapping_digest,
    make_source_frame,
)


ROOT = Path(__file__).resolve().parent


def _source(frame_id: int, pose_x: float, *, sequence_id: int = 201) -> object:
    xyzi = np.asarray(
        [[10.0, 0.0, 0.0, 0.25], [11.0, 1.0, 0.0, 0.75]], dtype=np.float32
    )
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = pose_x
    packed = np.asarray([40, 40], dtype=np.uint32)
    labels = PointLabels(
        packed=packed,
        semantic=packed.astype(np.uint16),
        instance=np.zeros(2, dtype=np.uint16),
        semantic_target=np.asarray([8, 8], dtype=np.uint8),
    )
    return make_source_frame(
        frame_id,
        xyzi,
        pose,
        labels,
        partition="train",
        sequence_id=sequence_id,
    )


def _window(order: tuple[int, ...] = (0, 1, 2, 3, 4)) -> object:
    spec = SequenceSpec("train", 201, "fixture", True, FrameSpan(0, 5))
    by_id = {index: _source(index, float(index)) for index in range(5)}
    return assemble_window(
        spec, 0, tuple(range(5)), tuple(by_id[index] for index in order)
    )


def _protocol_at_stage(tmp_path: Path, stage: str) -> object:
    payload = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    order = ("F1", "F2", "F3", "F4", "C1", "V1", "T1")
    index = order.index(stage)
    payload["status"].update(
        current_stage=stage,
        experiments_started=stage != "F1",
        training_allowed=stage == "F4",
        performance_claims_available=stage == "T1",
        selected_method=("direct_dense_stu" if stage in {"C1", "V1", "T1"} else None),
        f4_required=stage == "F4",
        f4_completed=False,
    )
    claims = payload["claims"]
    claims["F1_completed"] = index >= order.index("F2")
    claims["F2_completed"] = index >= order.index("F3")
    claims["F3_completed"] = index >= order.index("F4")
    claims["C1_completed"] = index >= order.index("V1")
    claims["real_anomaly_validation_performed"] = stage == "T1"
    claims["training_performed"] = False
    path = tmp_path / f"{stage}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_protocol(path)


def test_protocol_is_the_only_schema33_pretraining_contract() -> None:
    protocol = load_protocol()
    assert protocol.schema_version == SCHEMA_VERSION == 33
    assert [item.value for item in InputMode] == ["single_stu", "dense_stu"]
    assert set(protocol.methods) == {"single_stu", "dense_stu"}
    assert protocol.status["current_stage"] == "F1"
    assert protocol.status["training_allowed"] is False
    assert protocol.status["performance_claims_available"] is False


def test_old_schema_is_rejected_before_interpretation(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 32
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="schema-33"):
        load_protocol(path)


def test_protocol_accepts_every_valid_forward_stage(tmp_path: Path) -> None:
    for stage in ("F1", "F2", "F3", "F4", "C1", "V1", "T1"):
        assert _protocol_at_stage(tmp_path, stage).status["current_stage"] == stage


def test_evaluator_only_allows_the_active_feasibility_stage(tmp_path: Path) -> None:
    for stage in ("F1", "F2", "F3"):
        protocol = _protocol_at_stage(tmp_path, stage)
        require_experiment_stage(protocol, stage)
        with pytest.raises(EvaluationError, match="does not authorize"):
            require_experiment_stage(protocol, "F1" if stage != "F1" else "F2")


def test_train_201_development_and_confirmation_are_disjoint() -> None:
    protocol = load_protocol()
    development = protocol.normal_development
    confirmation = protocol.normal_confirmation
    assert development.span == FrameSpan(4, 554)
    assert confirmation.span == FrameSpan(554, 682)
    assert len(development.legal_window_starts()) == 546
    assert len(confirmation.legal_window_starts()) == 124
    assert development.legal_window_starts()[-1] + 4 == 553
    assert confirmation.legal_window_starts()[0] == 554
    assert confirmation.legal_window_starts()[0] + 4 == 558


def test_f2_endpoints_are_unique_legal_development_outputs() -> None:
    protocol = load_protocol()
    endpoints = tuple(protocol.feasibility["F2_normal_stability"]["current_frames"])
    assert len(endpoints) == len(set(endpoints)) == 24
    legal = {start + 4 for start in protocol.normal_development.legal_window_starts()}
    assert set(endpoints) <= legal


def test_f3_screen_and_extension_sources_are_globally_disjoint() -> None:
    protocol = load_protocol()
    settings = protocol.feasibility["F3_proxy_signal"]
    length = int(settings["frames_per_sequence"])
    starts = tuple(settings["screen"]["source_starts"]) + tuple(
        settings["extension_if_screen_is_inconclusive"]["source_starts"]
    )
    groups = [
        set(range(int(start), int(start) + length))
        for start in starts
    ]
    assert all(
        left.isdisjoint(right)
        for i, left in enumerate(groups)
        for right in groups[i + 1 :]
    )
    seeds = tuple(settings["screen"]["world_root_seeds"]) + tuple(
        settings["extension_if_screen_is_inconclusive"]["world_root_seeds"]
    )
    assert len(groups) == len(seeds) == len(set(seeds)) == 16


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (
            {
                "planned_worlds": 8,
                "evaluable_worlds": 8,
                "mean_delta_AP": 1.0,
                "median_delta_AP": 1.0,
                "paired_world_bootstrap_95_interval": [0.1, 1.9],
            },
            "extend",
        ),
        (
            {
                "planned_worlds": 8,
                "evaluable_worlds": 8,
                "mean_delta_AP": 0.0,
                "median_delta_AP": -0.1,
                "paired_world_bootstrap_95_interval": [-0.5, 0.5],
            },
            "reject",
        ),
        (
            {
                "planned_worlds": 8,
                "evaluable_worlds": 7,
                "mean_delta_AP": 2.0,
                "median_delta_AP": 2.0,
                "paired_world_bootstrap_95_interval": [1.0, 3.0],
            },
            "extend",
        ),
    ],
)
def test_f3_screen_action_requires_all_eight_worlds(
    summary: dict[str, object], expected: str
) -> None:
    assert f3_screen_action(summary) == expected


def test_f3_final_support_requires_twelve_evaluable_worlds() -> None:
    positive = {
        "evaluable_worlds": 11,
        "paired_world_bootstrap_95_interval": [0.1, 1.9],
    }
    assert (
        f3_final_action(positive, minimum_evaluable_worlds=12)
        == "insufficient_evidence"
    )
    positive["evaluable_worlds"] = 12
    assert f3_final_action(positive, minimum_evaluable_worlds=12) == "support"


def test_failed_direct_branch_cannot_enter_c1(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    payload["status"].update(
        current_stage="C1",
        experiments_started=True,
        training_allowed=False,
        performance_claims_available=False,
        selected_method=None,
        f4_required=True,
        f4_completed=False,
    )
    payload["claims"].update(
        F1_completed=True,
        F2_completed=True,
        F3_completed=True,
    )
    path = tmp_path / "invalid_C1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="completed method branch"):
        load_protocol(path)


def test_support_candidate_order_is_seeded_and_reproducible() -> None:
    count = 32
    pool = QualifiedSupportPool(
        pool_indices=np.arange(count),
        semantics=np.full(count, 40, dtype=np.uint16),
        frames=np.arange(count, dtype=np.int32) + 6,
        slots=np.arange(count, dtype=np.int32) * 17,
        ranges_m=np.full(count, 10.0),
        selection_hashes=np.arange(count, dtype=np.uint64),
        anchors_world_m=np.column_stack((np.arange(count), np.zeros((count, 2)))),
        normals_world=np.tile(np.asarray((0.0, 0.0, 1.0)), (count, 1)),
        offsets=np.zeros(count),
        source_sequence_id=201,
    )
    eligible = np.arange(4, 28, dtype=np.int64)
    first = _identity_order(pool, (40,), "test", 33000, eligible)
    repeated = _identity_order(pool, (40,), "test", 33000, eligible)
    different = _identity_order(pool, (40,), "test", 33001, eligible)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first[:8], different[:8])


def test_protocol_binds_the_sensor_calibration_bytes() -> None:
    protocol = load_protocol()
    assert protocol.verify_sensor_calibration().name == "calibration.pt"
    assert (
        protocol.verify_official_point_evaluator().name == "compute_point_level_ood.py"
    )


def test_runtime_sources_and_inputs_are_workspace_local() -> None:
    protocol = load_protocol()
    repository = protocol.stu_repository_path()
    paths = (
        repository,
        protocol.verify_official_point_evaluator(),
        protocol.verify_sensor_calibration(),
        protocol.verify_support_pool(201),
        protocol.verify_support_pool(206),
    )
    assert all(path.is_relative_to(ROOT) for path in paths)
    assert (
        stu_source_manifest(repository)["manifest_sha256"]
        == protocol.stu["source_manifest_sha256"]
    )


def test_window_uses_latest_scan_as_its_exact_coordinate_frame() -> None:
    window = _window()
    latest = window.current_frame.source
    np.testing.assert_array_equal(
        window.points.coordinates[window.current_mask],
        latest.xyzi[latest.real_slots, :3],
    )
    np.testing.assert_array_equal(
        window.current_pose.world_from_current, latest.lidar_pose
    )


def test_past_scans_are_transformed_to_the_current_frame() -> None:
    window = _window()
    # The first scan point is at world x=10 and the current sensor is at world x=4.
    assert window.points.coordinates[0, 0] == 6.0
    np.testing.assert_allclose(
        window.frames[0].source_to_current,
        np.linalg.inv(window.current_frame.source.lidar_pose)
        @ window.frames[0].source.lidar_pose,
    )


def test_alignment_is_independent_of_source_argument_order() -> None:
    ordered = _window()
    shuffled = _window((4, 2, 0, 3, 1))

    def rows(window: object) -> dict[tuple[int, int], np.ndarray]:
        return {
            (int(frame), int(slot)): point
            for frame, slot, point in zip(
                window.points.source_frame,
                window.points.source_slot,
                window.points.coordinates,
                strict=True,
            )
        }

    expected = rows(ordered)
    observed = rows(shuffled)
    assert set(expected) == set(observed)
    for identity in expected:
        np.testing.assert_array_equal(expected[identity], observed[identity])


def test_f2_uses_distinct_anomaly_and_semantic_masks() -> None:
    semantics = np.asarray((0, 2, 40, 52), dtype=np.uint32)
    targets = np.asarray((255, 255, 8, 255), dtype=np.uint8)
    xyzi = np.asarray(
        (
            (10.0, 0.0, 0.0, 0.2),
            (11.0, 0.0, 0.0, 0.3),
            (12.0, 0.0, 0.0, 0.4),
            (13.0, 0.0, 0.0, 0.5),
        ),
        dtype=np.float32,
    )
    labels = PointLabels(
        packed=semantics,
        semantic=semantics.astype(np.uint16),
        instance=np.zeros(4, dtype=np.uint16),
        semantic_target=targets,
    )
    sources = tuple(
        make_source_frame(
            frame,
            xyzi,
            np.eye(4, dtype=np.float64),
            labels,
            partition="train",
            sequence_id=201,
        )
        for frame in range(5)
    )
    spec = SequenceSpec("train", 201, "fixture", True, FrameSpan(0, 5))
    masks = f2_point_masks(
        assemble_window(spec, 0, tuple(range(5)), sources), load_protocol()
    )
    assert masks.normal_anomaly.tolist() == [False, False, True, True]
    assert masks.semantic_class.tolist() == [False, False, True, False]


def test_canonical_ray_mapping_identity_changes_with_the_mapping() -> None:
    mapping = np.arange(128 * 1024, dtype=np.int32)
    digest = canonical_ray_mapping_digest(mapping)
    changed = mapping.copy()
    changed[0], changed[1] = changed[1], changed[0]
    assert len(digest) == 64
    assert canonical_ray_mapping_digest(changed) != digest


def test_renderer_remains_deterministic_and_uses_the_nearest_return() -> None:
    xyzi = np.asarray(((5.0, 0.0, 0.0, 0.2),), dtype=np.float32)
    packed = np.asarray((10,), dtype=np.uint32)
    labels = PointLabels(
        packed=packed,
        semantic=packed.astype(np.uint16),
        instance=np.zeros(1, dtype=np.uint16),
    )
    frame = make_source_frame(
        0,
        xyzi,
        np.eye(4, dtype=np.float64),
        labels,
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
    assert float(first.source.xyzi[0, 0]) == pytest.approx(2.5, abs=2.0e-4)


def test_proxy_feasibility_samples_one_anomaly_only_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.render as render_module

    world = object()
    report = object()
    clip = object()
    observed: dict[str, object] = {}
    calls = 0

    def fake_sample(
        support_pool: object,
        obstacles: object,
        seed: int,
        **kwargs: object,
    ) -> tuple[object, object]:
        nonlocal calls
        calls += 1
        observed.update(
            seed=seed,
            source_sequence_id=kwargs["source_sequence_id"],
            support_frame_ids=kwargs["support_frame_ids"],
        )
        return world, report

    def fake_render(*args: object, **kwargs: object) -> object:
        assert args[0:2] == (world, report)
        return clip

    monkeypatch.setattr(render_module, "sample_anomaly_world", fake_sample)
    monkeypatch.setattr(render_module, "render_development_clip_world", fake_render)
    sources = tuple(_source(frame, float(frame)) for frame in range(4, 32))
    result = render_module.sample_development_clip_world(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        sources,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        33000,
        renderer_identity="a" * 64,
    )
    assert result is clip
    assert calls == 1
    assert observed == {
        "seed": 33000,
        "source_sequence_id": 201,
        "support_frame_ids": tuple(range(6, 30)),
    }


def test_f3_does_not_substitute_a_root_seed_after_placement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.render as render_module

    seeds: list[int] = []

    def fail_sample(
        support_pool: object,
        obstacles: object,
        seed: int,
        **kwargs: object,
    ) -> tuple[object, object]:
        del support_pool, obstacles, kwargs
        seeds.append(seed)
        raise PlacementError("physical placement exhausted")

    monkeypatch.setattr(render_module, "sample_anomaly_world", fail_sample)
    sources = tuple(_source(frame, float(frame)) for frame in range(4, 32))
    with pytest.raises(PlacementError, match="physical placement exhausted"):
        render_module.sample_development_clip_world(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            sources,
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            33000,
            renderer_identity="a" * 64,
        )
    assert seeds == [33000]


def test_invisible_windows_do_not_reject_a_fixed_f3_world() -> None:
    packed = np.asarray((40,), dtype=np.uint32)
    labels = PointLabels(
        packed=packed,
        semantic=packed.astype(np.uint16),
        instance=np.zeros(1, dtype=np.uint16),
        semantic_target=np.asarray((8,), dtype=np.uint8),
    )
    sources = tuple(
        make_source_frame(
            frame,
            np.asarray(((5.0, 0.0, 0.0, 0.2),), dtype=np.float32),
            np.eye(4, dtype=np.float64),
            labels,
            partition="train",
            sequence_id=201,
        )
        for frame in range(9)
    )
    shape = ShapeSpec(
        ((0.5, 0.5, 0.5),),
        ((0.0, 0.0, 0.0),),
        ((1.0, 1.0),),
        (0.0,),
        ("union",),
    )
    rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    world = WorldSpec(
        33000,
        201,
        (
            ObjectSpec(
                1,
                "anomaly-proxy",
                shape,
                MaterialSpec(0.5, 0.1),
                (5.0, 100.0, 0.0),
                rotation,
            ),
        ),
    )
    report = WorldGenerationReport(33000, 201, "anomaly_only", 0, 1, 33000, 33000)
    grid = RayGrid(
        np.asarray(((1.0, 0.0, 0.0),)),
        np.asarray((0.0,)),
        np.asarray((0.0,)),
        beam_count=1,
    )
    clip = render_development_clip_world(
        world,
        report,
        sources,
        grid,
        SensorCalibration.constant(0.4),
        renderer_identity="a" * 64,
    )
    assert len(clip.windows) == 5
    assert all(
        not np.any(frame.anomaly_proxy_mask)
        for window in clip.windows
        for frame in window.rendered_frames
    )


def test_dense_stu_input_contains_all_scans_and_identifies_current_rows() -> None:
    inputs = window_stu_inputs(_window())
    assert inputs.single_real_slots.size == 2
    assert inputs.dense_coordinates.shape == (10, 3)
    np.testing.assert_array_equal(inputs.dense_current_rows, np.asarray([8, 9]))
    assert inputs.dense_features.shape == (10, 2)


def test_geometry_record_reports_real_point_and_voxel_gain() -> None:
    record = geometry_record(_window(), (0.05,))
    assert record["single_visible_returns"] == 2
    assert record["dense_visible_returns"] == 10
    assert record["visible_return_ratio"] == 5.0
    voxel = record["voxels"]["0.05"]
    assert voxel["dense_unique_voxels"] > voxel["single_unique_voxels"]
    assert voxel["new_voxels_not_present_in_current_scan"] > 0


class _DummyEncoder:
    def __call__(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        real_slots: np.ndarray | None = None,
    ) -> STUPointEncoding:
        rows = (
            np.arange(coordinates.shape[0], dtype=np.int64)
            if real_slots is None
            else np.asarray(real_slots)
        )
        count = rows.size
        score = torch.as_tensor(rows, dtype=torch.float32) / 10.0
        evidence = torch.zeros((count, NUM_NORMAL_CLASSES), dtype=torch.float32)
        evidence[:, 8] = 1.0
        return STUPointEncoding(
            point_features=torch.zeros((count, MASK_DIM)),
            assigned_query=torch.zeros(count, dtype=torch.long),
            normal_evidence=evidence,
            reliability_assign=torch.ones(count),
            reliability_noobj=torch.zeros(count),
            maxlogit_score=score,
            normal_class=torch.full((count,), 8, dtype=torch.long),
            inverse_map=torch.arange(count, dtype=torch.long),
            real_slots=torch.as_tensor(rows, dtype=torch.long),
            input_identity=stu_input_identity(coordinates, features, rows),
        )


def test_score_window_retains_all_dense_outputs_and_current_view() -> None:
    scores = score_window(_DummyEncoder(), _window())  # type: ignore[arg-type]
    np.testing.assert_allclose(scores.single_score, (0.0, 0.1))
    np.testing.assert_allclose(scores.dense_all_score, np.arange(10) / 10.0)
    np.testing.assert_allclose(scores.dense_current_score, (0.8, 0.9))
    np.testing.assert_array_equal(scores.single_class, scores.dense_current_class)


def test_dense_input_degenerates_exactly_when_history_has_no_returns() -> None:
    spec = SequenceSpec("train", 201, "fixture", True, FrameSpan(0, 5))
    sources = []
    for frame in range(5):
        xyzi = (
            np.zeros((1, 4), dtype=np.float32)
            if frame < 4
            else np.asarray(((10.0, 0.0, 0.0, 0.5),), dtype=np.float32)
        )
        packed = np.asarray((40,), dtype=np.uint32)
        sources.append(
            make_source_frame(
                frame,
                xyzi,
                np.eye(4, dtype=np.float64),
                PointLabels(
                    packed=packed,
                    semantic=packed.astype(np.uint16),
                    instance=np.zeros(1, dtype=np.uint16),
                    semantic_target=np.asarray((8,), dtype=np.uint8),
                ),
                partition="train",
                sequence_id=201,
            )
        )
    window = assemble_window(spec, 0, tuple(range(5)), tuple(sources))
    inputs = window_stu_inputs(window)
    np.testing.assert_array_equal(inputs.single_coordinates, inputs.dense_coordinates)
    np.testing.assert_array_equal(inputs.single_features, inputs.dense_features)
    scores = score_window(_DummyEncoder(), window)  # type: ignore[arg-type]
    np.testing.assert_array_equal(scores.single_score, scores.dense_current_score)


def test_official_voxel_inverse_recovers_current_point_after_collision() -> None:
    spec = SequenceSpec("train", 201, "fixture", True, FrameSpan(0, 5))
    sources = tuple(_source(frame, 0.0) for frame in range(5))
    window = assemble_window(spec, 0, tuple(range(5)), sources)
    inputs = window_stu_inputs(window)
    _, _, _, inverse = official_stu_sparse_quantize(
        inputs.dense_coordinates, inputs.dense_features
    )
    inverse_map = np.asarray(inverse, dtype=np.int64)
    current = int(inputs.dense_current_rows[0])
    assert inverse_map[0] == inverse_map[current]
    scores_for_current = inverse_map[inputs.dense_current_rows]
    assert scores_for_current.shape == inputs.dense_current_rows.shape


def test_normal_class_matches_the_official_assigned_query_prediction() -> None:
    logits = torch.full((NUM_QUERIES, NUM_NORMAL_CLASSES + 1), -20.0)
    logits[:, -1] = 20.0
    logits[0] = -20.0
    logits[0, 0] = 5.0
    logits[0, -1] = 0.0
    for query in (1, 2):
        logits[query] = -20.0
        logits[query, 1] = 4.0
        logits[query, -1] = 0.0
    masks = torch.full((1, NUM_QUERIES), -20.0)
    masks[0, 0] = 3.0
    masks[0, 1:3] = 1.0
    evidence = assigned_stu_evidence(logits, masks)
    reference = official_stu_semantic_class(logits, masks)
    assert int(evidence.assigned_query[0]) == 0
    assert torch.equal(evidence.normal_class, reference)
    assert int(reference[0]) == 0
    assert int(evidence.normal_evidence.argmax(dim=1)[0]) == 0


def test_official_point_metrics_are_invariant_to_order_with_tied_scores() -> None:
    labels = np.asarray([True, False, True, False])
    scores = np.asarray([0.5, 0.5, 0.9, 0.1])
    first = official_point_metrics(labels, scores)
    order = np.asarray([1, 0, 2, 3])
    second = official_point_metrics(labels[order], scores[order])
    assert first == second
    assert set(first) == {"AP", "AUROC", "FPR95", "threshold"}
    with pytest.raises(Exception, match="at least one anomaly"):
        official_point_metrics(np.zeros(4, dtype=np.bool_), np.arange(4.0))
    with pytest.raises(Exception, match="at least one normal"):
        official_point_metrics(np.ones(4, dtype=np.bool_), np.arange(4.0))


def test_schema33_qualification_covers_the_five_core_invariants() -> None:
    result = run_schema33_qualification()
    assert result["mechanical"]["passed"] is True
    assert result["mechanical"]["check_count"] == 8
    assert result["scientific_status"] == "pending_real_F1_F2_F3_execution"
    assert result["performance_claim_available"] is False
