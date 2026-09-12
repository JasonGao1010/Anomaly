from dataclasses import replace
import json

import numpy as np
import pytest

from src.data import FrozenFrame
from src.protocol import load_protocol
from src.scene import PointLabels, make_source_frame
from src.view import Camera, intensity_colors, labelled_points, load_frame, map_object, map_pixels, map_returns, select_preview


def test_world_map_uses_equal_xy_scale_and_rotated_physical_bounds():
    bounds = np.array([[0., 0.], [10., 20.]])
    xy = np.array([[0., 0.], [10., 20.], [5., 10.]])
    np.testing.assert_allclose(map_pixels(xy, bounds, (100, 200)), [[0, 200], [100, 0], [50, 100]])
    pixels = map_pixels(np.array([[5, 10], [6, 10], [5, 11]]), bounds, (300, 200))
    np.testing.assert_allclose(pixels - pixels[0], [[0, 0], [10, 0], [0, -10]])
    record = dict(world=dict(objects=[dict(translation_world_m=[10., 20., 30.],
        rotation_world_from_local=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])]),
        generation=dict(geometry=dict(lower_local_m=[-2., -1., -3.], upper_local_m=[2., 1., 3.])))
    corners, centre = map_object(record)
    np.testing.assert_allclose(centre, [10, 20, 30])
    np.testing.assert_allclose(corners.min(0), [9, 18, 27])
    np.testing.assert_allclose(corners.max(0), [11, 22, 33])


def test_map_background_transforms_once_and_keeps_overlapping_returns(monkeypatch):
    import src.view as view
    xyz = np.array([[1., 0., 0., .25], [1., 0., 2., .5], [0., 0., 0., 1.]], np.float32)
    packed = np.array([40, 40, 0], np.uint32)
    labels = PointLabels(packed, packed.astype(np.uint16), np.zeros(3, np.uint16), np.array([8, 8, 255], np.uint8))
    pose = np.array([[0., -1., 0., 2.05], [1., 0., 0., 3.05], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    sample = make_source_frame(0, xyz, pose, labels, partition="train", sequence_id=206)
    settings = dict(label_colors={"-1":[80,210,100], "0":[72,167,235], "1":[255,80,45]},
                    background_rgb=[12,16,23], intensity=dict(half_saturation=.25, minimum_contrast=.4))
    monkeypatch.setattr(view, "_view_sources", {"train":{0:sample, 1:sample}}, raising=False)
    monkeypatch.setattr(view, "_view_settings", settings, raising=False)
    _, count, total, actual, included = view._map_background(("train", [0, 1], np.array([[0,0],[10,10]]), np.array([100,100])))
    assert actual == included == count.sum() == 4
    assert np.count_nonzero(count) == 1 and count[40*100+20] == 4
    colors = intensity_colors(xyz[:2,3], np.zeros(2, np.int8),
        {int(k):v for k,v in settings["label_colors"].items()}, settings["background_rgb"], settings["intensity"])
    np.testing.assert_allclose(total[40*100+20] / 4, colors.mean(0))


def test_map_returns_use_actual_inserted_slots_and_path_visibility_colors(tmp_path, monkeypatch):
    import src.view as view
    from types import SimpleNamespace
    (tmp_path / "frames").mkdir()
    xyzi = np.array([[50, 40, 30, .7], [2, 0, 1, .2], [2, 0, 1, .4]], np.float32)
    np.savez(tmp_path / "frames/000001.npz", world_identity="a" * 64,
             source_slot=[3, 8, 9], inserted_slot=[8, 9], xyzi=xyzi)
    pose = np.array([[0., -1., 0., 10.], [1., 0., 0., 20.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    source = SimpleNamespace(lidar_pose=lambda frame: pose)
    manifest = dict(world_identity="a" * 64, frames=[dict(frame=0, count=0), dict(frame=1, count=2)])
    points = map_returns(tmp_path, manifest, source)
    np.testing.assert_allclose(points[:, :3], [[10, 22, 1], [10, 22, 1]])
    np.testing.assert_array_equal(points[:, 3], xyzi[1:, 3])
    manifest["frames"][1]["count"] = 3
    with pytest.raises(ValueError, match="visibility"):
        map_returns(tmp_path, manifest, source)
    settings = dict(label_colors={"-1":[80,210,100], "0":[72,167,235], "1":[255,80,45]},
                    background_rgb=[12,16,23], intensity=dict(half_saturation=.25, minimum_contrast=.4))
    monkeypatch.setattr(view, "_view_settings", settings, raising=False)
    monkeypatch.setattr(view, "_text_writer", lambda draw, size: (lambda *args: None, []))
    monkeypatch.setattr(view, "_map_arrow", lambda *args: None)
    bounds = np.array([[-5., -5.], [5., 5.]])
    path = np.array([[-3., -1.], [-1., -1.], [1., -1.], [3., -1.]])
    base = view.Image.new("RGB", (100, 100), (12,16,23))
    panel, _ = view._map_panel(base, bounds, bounds, (400, 400), path, np.empty((0, 4)),
                               np.array([0,4,0]), 0, np.array([False,True,False,True]))
    pixels = map_pixels(path, bounds, (400, 400)).astype(int)
    for (x,y), expected in zip(pixels, [(194,116,255),(255,218,45),(194,116,255),(255,218,45)]):
        assert panel.getpixel((x,y)) == expected


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


def test_preview_selection_requires_front_visible_anomaly_and_preserves_distance_ranking(tmp_path):
    camera = Camera(1200, 800, 120, (-.3, 0, -.6), 1)
    frames = tmp_path / "frames"
    frames.mkdir()
    rows = []
    # The closest scan is behind the camera; the other two tie in distance.
    for frame, distance, xyz in ((0, 42.5, [-42.5, 0, 0]),
                                  (1, 41., [41., 0, 0]), (2, 44., [44., 0, 0])):
        np.savez(frames / f"{frame:06d}.npz", world_identity="a" * 64,
                 source_identity=f"source-{frame}", source_slot=np.array([7, 9]),
                 inserted_slot=np.array([9]), xyzi=np.array([[5, 0, 0, .1], [*xyz, .2]]))
        rows.append(dict(frame=frame, range=distance, source_identity=f"source-{frame}"))
    chosen, reason = select_preview(rows[::-1], tmp_path, "a" * 64, camera, [35, 50])
    assert chosen["frame"] == 1 and reason is None
    assert select_preview(rows[:1], tmp_path, "a" * 64, camera, [35, 50]) == (None, "outside_front_view")
    assert select_preview([], tmp_path, "a" * 64, camera, [35, 50]) == (None, "no_range_observation")
    with pytest.raises(ValueError, match="different frozen scan"):
        select_preview(rows[:1], tmp_path, "b" * 64, camera, [35, 50])
