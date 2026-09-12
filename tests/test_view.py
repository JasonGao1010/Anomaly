from dataclasses import replace
import json

import numpy as np

from src.data import FrozenFrame
from src.protocol import load_protocol
from src.scene import PointLabels, make_source_frame
from src.view import Camera, intensity_colors, labelled_points, load_frame


def test_rectilinear_projection_preserves_lines_field_of_view_and_fixed_offset():
    camera = Camera(1200, 800, 120, (-0.3, 0, -0.6), 1)
    angles = np.deg2rad([0, 30, -30, 59.9, -59.9, 60.1, -60.1, 180])
    xyz = np.column_stack((10 * np.cos(angles), 10 * np.sin(angles), np.zeros(8)))
    points = xyz + camera.offset_lidar_m
    indices, uv = camera.project(points)
    np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(uv[:3], [[600, 400], [400, 400], [800, 400]])
    above = np.array([[10 * np.cos(np.pi / 6), 0, 10 * np.sin(np.pi / 6)]])
    np.testing.assert_allclose(camera.project(above + camera.offset_lidar_m)[1], [[600, 200]])
    for shift in ((0, 0, 0), (20, -5, 3)):
        moved = replace(camera, offset_lidar_m=tuple(np.asarray(camera.offset_lidar_m) + shift))
        np.testing.assert_allclose(moved.project(points + shift)[1], uv)
    # Test off-axis horizontal/vertical lines and an oblique line with varying depth.
    t = np.linspace(-3, 3, 31)
    lines = [np.column_stack((np.full_like(t, 10), t, np.full_like(t, 2))),
             np.column_stack((np.full_like(t, 10), np.full_like(t, 2), t)),
             np.column_stack((10 + 2 * t, t, 2 + .3 * t))]
    for line in lines:
        indices, image_line = camera.project(line + camera.offset_lidar_m)
        np.testing.assert_array_equal(indices, np.arange(len(t)))
        relative = image_line - image_line[0]
        cross = relative[:, 0] * relative[-1, 1] - relative[:, 1] * relative[-1, 0]
        np.testing.assert_allclose(cross, 0, atol=1e-10)
    with np.errstate(all="raise"):
        excluded = np.array([[0, 0, 0], [0, 2, 3], [-1, 0, 0], [1e-12, 10, 10]])
        assert camera.project(excluded + camera.offset_lidar_m)[0].size == 0
    widescreen = replace(camera, width=1920, height=1080)
    assert 88.5 < widescreen.vertical_fov_degrees < 88.6


def test_all_overlapping_returns_contribute_without_depth_or_order_rejection():
    camera = Camera(120, 80, 120, (0, 0, 0), 1)
    # All three returns contribute, including a far point and coincident source slots.
    xyz = np.array([[20, 0, 0], [5, 0, 0], [5, 0, 0], [-5, 0, 0]])
    colors = np.array([[215, 50, 30], [20, 100, 180], [110, 115, 125], [215, 50, 30]])
    background = (12, 16, 23)
    rgb, projected, count = camera.rasterize(xyz, colors, background)
    np.testing.assert_array_equal(projected, [0, 1, 2])
    assert np.count_nonzero(count) == 5
    assert count.sum() == 15 and count[40, 60] == 3
    np.testing.assert_array_equal(rgb[40, 60], np.rint(colors[:3].mean(axis=0)))
    reversed_rgb, _, reversed_count = camera.rasterize(xyz[::-1], colors[::-1], background)
    np.testing.assert_array_equal(reversed_rgb, rgb)
    np.testing.assert_array_equal(reversed_count, count)
    # Different centres also accumulate in the intersection of their footprints.
    angle = np.arctan(1.2 / camera.focal_px)
    points = np.array([[5, 0, 0], [20 * np.cos(angle), -20 * np.sin(angle), 0]])
    rgb, _, count = camera.rasterize(points, colors[:2], background)
    assert count[40, 61] == 2 and count[40, 62] == 1
    np.testing.assert_array_equal(rgb[40, 61], np.rint(colors[:2].mean(axis=0)))
    np.testing.assert_array_equal(rgb[40, 62], colors[1])
    for points in (xyz[3:], np.empty((0, 3))):
        empty, ids, count = camera.rasterize(points, colors[:len(points)], background)
        np.testing.assert_array_equal(empty, np.broadcast_to(background, empty.shape))
        assert ids.size == 0 and count.sum() == 0


def test_intensity_shading_is_fixed_monotonic_and_keeps_weak_returns_visible():
    colors = {-1: [80, 210, 100], 0: [72, 167, 235], 1: [255, 80, 45]}
    background = np.array([12, 16, 23])
    settings = dict(half_saturation=.25, minimum_contrast=.4)
    intensity = np.array([-1, 0, .25, 1, 2])
    for label in (-1, 0, 1):
        targets = np.full(len(intensity), label)
        rgb = intensity_colors(intensity, targets, colors, background, settings)
        np.testing.assert_allclose(rgb[0], .6 * background + .4 * np.array(colors[label]))
        np.testing.assert_allclose(rgb[1], rgb[0])
        assert np.all(np.diff(rgb[1:], axis=0) > 0)
        assert np.all(rgb > background)  # Even zero intensity retains a visible mark.
        for i in range(len(intensity)):
            individual = intensity_colors(intensity[i:i+1], targets[i:i+1], colors, background, settings)
            np.testing.assert_array_equal(individual[0], rgb[i])


def test_single_frame_reader_uses_frozen_truth_and_original_sensor_coordinates(tmp_path, monkeypatch):
    xyzi = np.array([[8, 0, 0, .2], [9, 1, 0, .3], [8, -1, 0, .1], [0, 0, 0, 9]], np.float32)
    packed = np.array([40, 2, 1, 0], np.uint32)
    semantic_target = np.array([8, 255, 255, 255], np.uint8)
    labels = PointLabels(packed, packed.astype(np.uint16), np.zeros(4, np.uint16), semantic_target)
    pose = np.eye(4)
    pose[:3, 3] = [100, 200, 300]
    original = make_source_frame(0, xyzi, pose, labels, partition="train", sequence_id=206)
    points, target = labelled_points(original)
    np.testing.assert_array_equal(points, xyzi[:3])
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
    np.testing.assert_array_equal(points, changed[:3])
    np.testing.assert_array_equal(target, [0, 1, -1])
    assert calls[0]["partition"] == "train" and calls[0]["sequence_id"] == 206
