from dataclasses import replace

import numpy as np
import pytest

from src.data import DataProtocolError, FramePrediction, FrozenFrame
from src.protocol import ProtocolError, load_protocol
from src.scene import (
    LabelMode,
    PointLabels,
    SceneDataError,
    STUSequence,
    make_source_frame,
)


def test_source_scan_preserves_slots_labels_and_official_arrays(tmp_path):
    directory = tmp_path / "val" / "125"
    (directory / "velodyne").mkdir(parents=True)
    (directory / "labels").mkdir()
    calibration = np.eye(4)
    calibration[:3, 3] = [1, 2, 3]
    (directory / "calib.txt").write_text(
        "Tr: " + " ".join(map(str, calibration[:3].ravel())) + "\n"
    )
    poses = np.stack((np.eye(4), np.eye(4)))
    poses[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    poses[1, :3, 3] = [5, 6, 7]
    np.savetxt(directory / "poses.txt", poses[:, :3].reshape(2, 12))
    xyzi = np.array([[2, 0, 0, 0.4], [0, 0, 0, 8.2], [10, 1, 2, 1.8]], np.float32)
    packed = np.array([2 + (3 << 16), 0, 40 + (7 << 16)], np.uint32)
    # A malformed preceding scan must not affect reading the selected current scan.
    (directory / "velodyne" / "000000.bin").write_bytes(b"bad")
    xyzi.tofile(directory / "velodyne" / "000001.bin")
    packed.tofile(directory / "labels" / "000000.label")
    packed.tofile(directory / "labels" / "000001.label")
    sequence = STUSequence.open(
        tmp_path,
        protocol=load_protocol(),
        partition="val",
        sequence_id=125,
        label_mode=LabelMode.REQUIRED,
    )
    source = sequence.source_frame(1)
    np.testing.assert_array_equal(source.xyzi, xyzi)
    np.testing.assert_array_equal(source.real_slots, [0, 2])
    np.testing.assert_array_equal(source.labels.packed, packed)
    np.testing.assert_array_equal(source.labels.semantic, packed & 65535)
    np.testing.assert_array_equal(source.labels.instance, packed >> 16)
    pose = np.linalg.solve(calibration, poses[1] @ calibration)
    coordinates = (pose @ np.column_stack((xyzi[:, :3], np.ones(3))).T).T[:, :3]
    np.testing.assert_allclose(source.coordinates, coordinates, rtol=0, atol=1e-12)
    features = np.column_stack(
        (
            xyzi[:, 3],
            np.linalg.norm(coordinates - coordinates.mean(axis=0), axis=1),
        )
    ).astype(np.float32)
    np.testing.assert_array_equal(source.features, features)
    with pytest.raises(SceneDataError, match="byte length"):
        sequence.source_frame(0)
    unread = STUSequence.open(
        tmp_path,
        protocol=load_protocol(),
        partition="val",
        sequence_id=125,
        label_mode=LabelMode.FORBIDDEN,
    ).source_frame(1)
    assert unread.labels is None
    np.testing.assert_array_equal(unread.xyzi, source.xyzi)


def test_frame_predictions_assign_scores_by_complete_source_slot_identity(tmp_path):
    source = make_source_frame(
        7,
        np.array([[0, 0, 0, 0], [10, 0, 0, 0.2], [11, 0, 0, 0.3]], np.float32),
        np.eye(4),
        partition="fixture",
        sequence_id=1,
    )
    prediction = FramePrediction(
        "fixture",
        1,
        7,
        np.array([2, 1]),
        np.array([81, -80], np.float32),
    )
    path = tmp_path / "prediction.npz"
    prediction.save(path, source)
    restored = FramePrediction.load(path, source)
    np.testing.assert_array_equal(restored.restore(source), [0, -80, 81])
    with pytest.raises(FileExistsError):
        prediction.save(path, source)
    with pytest.raises(DataProtocolError, match="different scans"):
        restored.validate(replace(source, frame_id=8))
    for slots in ([0, 1], [1], [1, 3]):
        wrong = FramePrediction(
            "fixture", 1, 7, np.array(slots), np.zeros(len(slots), np.float32)
        )
        with pytest.raises(DataProtocolError, match="exactly"):
            wrong.save(tmp_path / "invalid.npz", source)
    with pytest.raises(DataProtocolError, match="duplicate"):
        FramePrediction("fixture", 1, 7, np.array([1, 1]), np.zeros(2, np.float32))
    for scores in (np.array([np.nan], np.float32), np.array([1.0], np.float64)):
        with pytest.raises(DataProtocolError, match="finite float32"):
            FramePrediction("fixture", 1, 7, np.array([1]), scores)
    # Loading must recheck the source, even when the archive itself is well formed.
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["source_slot"] = np.array([0, 1], np.int32)
    np.savez(tmp_path / "wrong.npz", **arrays)
    with pytest.raises(DataProtocolError, match="exactly"):
        FramePrediction.load(tmp_path / "wrong.npz", source)


def test_active_protocol_keeps_test_outside_development():
    protocol = load_protocol()
    assert len(protocol.public_sequence_ids) == 19
    assert protocol.sequence("train", 201).frame_count == 682
    assert protocol.sequence("train", 206).frame_count == 449
    with pytest.raises(ProtocolError, match="outside"):
        protocol.sequence("test", 100)


def test_frozen_frame_restores_insertions_missing_returns_and_ignored_labels(tmp_path):
    xyzi = np.array(
        [
            [5, 0, 0, 0.2],
            [0, 0, 0, 8.2],
            [6, 1, 0, 0.4],
            [5, -1, 0, 0.3],
            [8, 1, 1, 0.8],
            [7, 1, 0, 0.5],
        ],
        np.float32,
    )
    packed = np.array([40, 0, 40, 2, 40, 1], np.uint32)
    target = np.array([8, 255, 8, 255, 8, 255], np.uint8)
    labels = PointLabels(
        packed, packed.astype(np.uint16), np.zeros(6, np.uint16), target
    )
    original = make_source_frame(
        7, xyzi, np.eye(4), labels, partition="train", sequence_id=206
    )
    rendered_xyzi = xyzi.copy()
    rendered_xyzi[0] = 0  # Occlusion without a returned pulse.
    rendered_xyzi[1:3] = [[3, 0, 0, 0.1], [4, 1, 0, 0.2]]
    rendered_packed = packed.copy()
    rendered_packed[0] = 0
    rendered_packed[1:3] = 2 | (60001 << 16)
    rendered_target = target.copy()
    rendered_target[:3] = 255
    labels = PointLabels(
        rendered_packed,
        (rendered_packed & 65535).astype(np.uint16),
        (rendered_packed >> 16).astype(np.uint16),
        rendered_target,
    )
    rendered = make_source_frame(
        7, rendered_xyzi, np.eye(4), labels, partition="train", sequence_id=206
    )
    inserted = np.array([False, True, True, False, False, False])
    occluded = np.array([True, False, True, False, False, False])
    sample = FrozenFrame(rendered, "a" * 64, inserted, occluded)
    path = tmp_path / "frame.npz"
    sample.save(path, original)
    restored = FrozenFrame.load(path, original, "a" * 64)
    np.testing.assert_array_equal(restored.source.xyzi, rendered_xyzi)
    np.testing.assert_array_equal(restored.source.labels.packed, rendered_packed)
    np.testing.assert_array_equal(restored.anomaly_target, [-1, 1, 1, -1, 0, -1])
    with pytest.raises(DataProtocolError, match="another fixed world"):
        FrozenFrame.load(path, original, "b" * 64)
    moved = np.eye(4)
    moved[0, 3] = 1
    with pytest.raises(DataProtocolError, match="changed after freezing"):
        FrozenFrame.load(path, replace(original, lidar_pose=moved), "a" * 64)
    changed_classes = original.labels.semantic_target.copy()
    changed_classes[5] = 0
    relabeled = replace(
        original, labels=replace(original.labels, semantic_target=changed_classes)
    )
    with pytest.raises(DataProtocolError, match="changed after freezing"):
        FrozenFrame.load(path, relabeled, "a" * 64)
    with pytest.raises(FileExistsError):
        sample.save(path, original)
    # A world with no visible anomaly still contributes the complete normal scan.
    normal = FrozenFrame(original, "a" * 64, np.zeros(6, bool), np.zeros(6, bool))
    normal.save(tmp_path / "normal.npz", original)
    normal = FrozenFrame.load(tmp_path / "normal.npz", original, "a" * 64)
    np.testing.assert_array_equal(normal.source.xyzi, original.xyzi)
    np.testing.assert_array_equal(normal.anomaly_target, [0, -1, 0, -1, 0, -1])


def test_single_scan_renderer_keeps_physical_occlusion(monkeypatch):
    from src.render import (
        MaterialSpec,
        ObjectSpec,
        RayGrid,
        SensorCalibration,
        ShapeSpec,
        WorldSpec,
        render_frame,
    )

    semantic = np.array([40], np.uint16)
    source = make_source_frame(
        0,
        np.array([[5, 0, 0, 0.2]], np.float32),
        np.eye(4),
        PointLabels(semantic.astype(np.uint32), semantic, np.zeros(1, np.uint16)),
        partition="train",
        sequence_id=206,
    )
    shape = ShapeSpec(
        ((0.5, 0.5, 0.5),),
        ((0.0, 0.0, 0.0),),
        ((1.0, 1.0),),
        (0.0,),
        ("union",),
    )
    front = ObjectSpec(
        1,
        "anomaly-proxy",
        shape,
        MaterialSpec(0.5, 0.1),
        (3.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    rear = replace(front, object_id=2, translation_world_m=(4.0, 0.0, 0.0))
    world = WorldSpec(13, 206, (front, rear))
    grid = RayGrid(
        np.array([[1.0, 0.0, 0.0]]), np.array([0.0]), np.array([0.0]), beam_count=1
    )
    sensor = SensorCalibration.constant(1.234567)
    # An opaque foreground surface still blocks the rear when it returns no signal.
    monkeypatch.setattr(
        SensorCalibration,
        "return_chance",
        lambda self, beam, distance, incidence, bias: (distance > 3).astype(float),
    )
    missing = render_frame(source, world, grid, sensor)
    assert not missing.inserted_mask.any()
    assert missing.occluded_original_mask.all() and missing.changed_mask.all()
    assert not missing.xyzi.any() and not missing.packed_labels.any()
    monkeypatch.setattr(
        SensorCalibration,
        "return_chance",
        lambda self, beam, distance, incidence, bias: np.ones_like(distance),
    )
    observed = render_frame(source, world, grid, sensor)
    assert observed.inserted_mask.all() and (observed.object_id_internal == 1).all()
    assert observed.xyzi[0, 3] == np.float32(round(3500 * 1.234567) / 3500)


def test_composed_shape_bounds_grounding_and_distinct_contacts():
    from src.generate import make_shape
    from src.render import qualify_grounding
    from src.coverage import conditions

    profile = dict(length_m=[1.2, 1.2], width_m=[.8, .8], height_m=[.3, .3])
    config = dict(exponents=[1., 1.], composed_exponents=[1., 1.],
                  part_scale_multiplier=[1., 1.], part_offset_jitter=0., part_yaw_rad=[0., 0.])
    for family, count in (("single", 1), ("step", 2), ("elbow", 3), ("bridge", 3)):
        shape, geometry = make_shape(np.random.default_rng(13), dict(profile, shape=family), config)
        assert len(shape.primitive_scales_m) == count
        assert shape.continuous_connectivity_certificate().state == "connected"
        np.testing.assert_allclose([geometry[k] for k in ("length_m", "width_m", "height_m")], [1.2, .8, .3], atol=3e-6)
        assert not conditions(geometry, dict(count=5, in_range=5, range=10))["low_eligible"]
        if family == "step":
            assert 2 * shape.primitive_scales_m[0][2] < .2
        grounding = qualify_grounding(shape)
        assert grounding.passed and abs(grounding.strict_lower_support_m + .15) < 1e-6
        if family == "bridge":
            # The center under the crosspiece is empty; both leg centers touch the support plane.
            offsets = np.array(shape.primitive_offsets_m)
            points = np.vstack(([0, 0, -.14], np.column_stack((offsets[:2, :2], np.full(2, -.14)))))
            signed = shape.signed_distance(points)
            assert signed[0] > 0 and (signed[1:] < 0).all()


def test_proposal_counts_true_surface_gaps_occlusion_and_the_same_signal(monkeypatch):
    from src.generate import make_shape, ray_observation
    from src.render import MaterialSpec, ObjectSpec, RayGrid, SensorCalibration, WorldSpec, render_frame

    shape, geometry = make_shape(np.random.default_rng(13),
                                dict(shape="bridge", length_m=[1.2, 1.2], width_m=[.8, .8], height_m=[.3, .3]),
                                dict(exponents=[1., 1.], composed_exponents=[1., 1.],
                                     part_scale_multiplier=[1., 1.], part_offset_jitter=0., part_yaw_rad=[0., 0.]))
    x = np.asarray(shape.primitive_offsets_m)[:2, 0]
    origins = np.array([[0, 0, -.14], [0, x[0], -.14], [0, x[1], -.14]])
    directions = np.tile([1., 0, 0], (3, 1))
    grid = RayGrid(directions, np.array([0.]), np.zeros(3), beam_count=1, origins_sensor=origins)
    semantic = np.full(3, 40, np.uint16)
    xyz = origins + 6 * directions
    source = make_source_frame(0, np.column_stack((xyz, [.2] * 3)).astype(np.float32), np.eye(4),
                               PointLabels(semantic.astype(np.uint32), semantic, np.zeros(3, np.uint16)),
                               partition="train", sequence_id=206)
    item = ObjectSpec(1, "anomaly-proxy", shape, MaterialSpec(.5, .1), (3., 0., 0.),
                      ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
    world = WorldSpec(13, 206, (item,))
    sensor = SensorCalibration.constant(1.)
    monkeypatch.setattr(SensorCalibration, "return_chance", lambda self, beam, distance, incidence, bias: np.ones_like(distance))
    result, foreground = ray_observation(source, world, grid, sensor, geometry)
    assert (result["available_box_rays"], result["foreground_surface_rays"], result["final_anomaly_slots"]) == (3, 2, 2)
    rendered = render_frame(source, world, grid, sensor)
    np.testing.assert_array_equal(foreground, rendered.inserted_mask)
    blocked_xyz = source.xyzi.copy()
    blocked_xyz[1, :3] = origins[1] + directions[1]
    blocked = make_source_frame(0, blocked_xyz, np.eye(4), source.labels, partition="train", sequence_id=206)
    blocked_result, _ = ray_observation(blocked, world, grid, sensor, geometry)
    assert blocked_result["foreground_surface_rays"] == 1
    monkeypatch.setattr(SensorCalibration, "return_chance", lambda self, beam, distance, incidence, bias: np.zeros_like(distance))
    missing, surface = ray_observation(source, world, grid, sensor, geometry)
    assert missing["foreground_surface_rays"] == 2 and missing["final_anomaly_slots"] == 0
    np.testing.assert_array_equal(surface, render_frame(source, world, grid, sensor).occluded_original_mask)
