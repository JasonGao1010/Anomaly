from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluate import (
    average_precision,
    geometry_record,
    score_window,
    window_stu_inputs,
)
from src.model import (
    MASK_DIM,
    NUM_NORMAL_CLASSES,
    NUM_QUERIES,
    STUPointEncoding,
    assigned_stu_evidence,
    stu_input_identity,
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
    MaterialSpec,
    ObjectSpec,
    RayGrid,
    SensorCalibration,
    ShapeSpec,
    WorldSpec,
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


def test_f2_endpoints_are_unique_legal_development_outputs() -> None:
    protocol = load_protocol()
    endpoints = tuple(protocol.feasibility["F2_normal_stability"]["current_frames"])
    assert len(endpoints) == len(set(endpoints)) == 24
    legal = {start + 4 for start in protocol.normal_development.legal_window_starts()}
    assert set(endpoints) <= legal


def test_f3_virtual_sequence_sources_are_disjoint() -> None:
    protocol = load_protocol()
    settings = protocol.feasibility["F3_proxy_signal"]
    length = int(settings["frames_per_sequence"])
    groups = [
        set(range(int(start), int(start) + length))
        for start in settings["source_starts"]
    ]
    assert all(
        left.isdisjoint(right)
        for i, left in enumerate(groups)
        for right in groups[i + 1 :]
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

    def fake_sample(
        templates: object,
        support_pool: object,
        obstacles: object,
        world_type: str,
        seed: int,
        **kwargs: object,
    ) -> tuple[object, object]:
        observed.update(
            templates=templates,
            world_type=world_type,
            seed=seed,
            source_sequence_id=kwargs["source_sequence_id"],
        )
        return world, report

    def fake_render(*args: object, **kwargs: object) -> object:
        assert args[0:2] == (world, report)
        return clip

    monkeypatch.setattr(render_module, "sample_world_spec", fake_sample)
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
    assert observed == {
        "templates": (),
        "world_type": "anomaly_only",
        "seed": 33000,
        "source_sequence_id": 201,
    }


def test_dense_stu_input_contains_all_scans_but_scores_current_rows_only() -> None:
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


def test_score_window_extracts_only_current_dense_rows() -> None:
    scores = score_window(_DummyEncoder(), _window())  # type: ignore[arg-type]
    np.testing.assert_allclose(scores.single_score, (0.0, 0.1))
    np.testing.assert_allclose(scores.dense_score, (0.8, 0.9))
    np.testing.assert_array_equal(scores.single_class, scores.dense_class)


def test_normal_class_uses_the_official_all_query_aggregation() -> None:
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
    assert int(evidence.assigned_query[0]) == 0
    assert int(evidence.normal_class[0]) == 1


def test_average_precision_uses_point_ranking() -> None:
    labels = np.asarray([False, True, False, True])
    assert average_precision(labels, np.asarray([0.1, 0.9, 0.2, 0.8])) == 1.0
    with pytest.raises(Exception, match="at least one anomaly"):
        average_precision(np.zeros(4, dtype=np.bool_), np.arange(4.0))
    with pytest.raises(Exception, match="at least one normal"):
        average_precision(np.ones(4, dtype=np.bool_), np.arange(4.0))


def test_schema33_qualification_covers_the_five_core_invariants() -> None:
    result = run_schema33_qualification()
    assert result["mechanical"]["passed"] is True
    assert result["mechanical"]["check_count"] == 5
    assert result["scientific_status"] == "pending_real_F1_F2_F3_execution"
    assert result["performance_claim_available"] is False
