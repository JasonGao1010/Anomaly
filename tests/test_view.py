from dataclasses import replace
import json

import numpy as np

from src.data import FrozenFrame
from src.protocol import load_protocol
from src.scene import PointLabels, make_source_frame
from src.view import Camera, labelled_points, load_frame


def test_equidistant_projection_angles_axes_field_of_view_and_fixed_offset():
    camera = Camera(1200, 800, 120, (-0.3, 0.35, -0.6), 1)
    angles = np.deg2rad([0, 30, -30, 59.9, -59.9, 60.1, -60.1, 180])
    xyz = np.column_stack((10 * np.cos(angles), 10 * np.sin(angles), np.zeros(8)))
    points = xyz + camera.offset_lidar_m
    indices, uv, ranges = camera.project(points)
    np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(uv[:3], [[600, 400], [300, 400], [900, 400]])
    np.testing.assert_allclose(ranges, 10)
    # The 30-degree point lies halfway from the centre to the 60-degree edge.
    pinhole_displacement = 600 / np.tan(np.deg2rad(60)) * np.tan(np.deg2rad(30))
    assert not np.isclose(600 - uv[1, 0], pinhole_displacement)
    above = np.array([[10 * np.cos(np.pi / 6), 0, 10 * np.sin(np.pi / 6)]])
    np.testing.assert_allclose(camera.project(above + camera.offset_lidar_m)[1], [[600, 100]])
    for shift in ((0, 0, 0), (20, -5, 3)):
        moved = replace(camera, offset_lidar_m=tuple(np.asarray(camera.offset_lidar_m) + shift))
        np.testing.assert_allclose(moved.project(points + shift)[1], uv)


def test_nearest_point_wins_without_label_priority_and_empty_front_view_is_valid():
    camera = Camera(120, 80, 120, (0, 0, 0), 1)
    # A far anomaly, a nearer normal and a coincident ignored point share the ray.
    xyz = np.array([[20, 0, 0], [5, 0, 0], [5, 0, 0], [-5, 0, 0]])
    labels = np.array([1, 0, -1, 1])
    owner, projected = camera.rasterize(xyz)
    np.testing.assert_array_equal(projected, [0, 1, 2])
    assert np.count_nonzero(owner >= 0) == 5
    np.testing.assert_array_equal(np.unique(owner[owner >= 0]), [1])
    assert np.all(labels[owner[owner >= 0]] == 0)
    # Different pixel centres have overlapping footprints; depth still wins there.
    angle = 1.2 / camera.focal_px
    points = np.array([[5, 0, 0], [20 * np.cos(angle), -20 * np.sin(angle), 0]])
    owners, _ = camera.rasterize(points)
    assert owners[40, 61] == 0
    assert owners[40, 62] == 1
    for points in (xyz[3:], np.empty((0, 3))):
        empty, ids = camera.rasterize(points)
        assert np.all(empty == -1) and ids.size == 0


def test_single_frame_reader_uses_frozen_truth_and_original_sensor_coordinates(tmp_path, monkeypatch):
    xyzi = np.array([[8, 0, 0, .2], [9, 1, 0, .3], [8, -1, 0, .1], [0, 0, 0, 9]], np.float32)
    packed = np.array([40, 2, 1, 0], np.uint32)
    semantic_target = np.array([8, 255, 255, 255], np.uint8)
    labels = PointLabels(packed, packed.astype(np.uint16), np.zeros(4, np.uint16), semantic_target)
    pose = np.eye(4)
    pose[:3, 3] = [100, 200, 300]
    original = make_source_frame(0, xyzi, pose, labels, partition="train", sequence_id=206)
    points, target = labelled_points(original)
    np.testing.assert_array_equal(points, xyzi[:3, :3])
    np.testing.assert_array_equal(target, [0, -1, -1])
    public = replace(original, partition="val", sequence_id=125)
    np.testing.assert_array_equal(labelled_points(public)[1], [0, 1, 0])

    changed = xyzi.copy()
    changed[1] = [4, 0, 0, .4]
    packed = packed.copy()
    packed[1] = 2 | (60001 << 16)
    inserted_labels = PointLabels(packed, (packed & 65535).astype(np.uint16),
                                  (packed >> 16).astype(np.uint16), semantic_target.copy())
    source = make_source_frame(0, changed, pose, inserted_labels, partition="train", sequence_id=206)
    mask = np.array([False, True, False, False])
    sample = FrozenFrame(source, "a" * 64, mask, mask)
    world = tmp_path / "world_1"
    path = world / "frames/000000.npz"
    sample.save(path, original)
    (world / "manifest.json").write_text(json.dumps(dict(
        source_sequence=206, world_identity="a" * 64, frames=[dict(frame=0)])))
    calls = []

    def open_sequence(root, **kwargs):
        calls.append(kwargs)
        return {0: original}

    monkeypatch.setattr("src.view.STUSequence.open", open_sequence)
    restored = load_frame(path, tmp_path, load_protocol())
    points, target = labelled_points(restored)
    np.testing.assert_array_equal(points, changed[:3, :3])
    np.testing.assert_array_equal(target, [0, 1, -1])
    assert calls[0]["partition"] == "train" and calls[0]["sequence_id"] == 206
