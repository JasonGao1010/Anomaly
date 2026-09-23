"""Independent geometry checks for source-only visible-surface insertion."""

import numpy as np

from src.nuscenes import _basis, _intersections, _triangles, transplant


def test_open_surfaces_use_nearest_forward_hit_and_barycentric_intensity():
    face = np.array([[5., -1., -1.], [5., 1., -1.], [5., 1., 1.], [5., -1., 1.]])
    vertices = np.concatenate((face, face * [2., 1., 1.]))
    triangles = np.array([[4, 5, 6], [4, 6, 7], [0, 1, 2], [0, 2, 3]])
    intensity = .5 + .1 * vertices[:, 1] + .05 * vertices[:, 2]
    directions = np.array([[10., 0., 0.], [10., .2, .3], [10., 3., 0.], [-1., 0., 0.]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distance, value = _intersections(directions, vertices, triangles, intensity)
    expected_points = np.array([[5., 0., 0.], [5., .1, .15]])
    np.testing.assert_allclose(distance[:2, None] * directions[:2], expected_points, atol=1e-12)
    np.testing.assert_allclose(value[:2], [.5, .5175], atol=1e-12)
    assert np.isinf(distance[2:]).all()
    # Face order cannot change visibility, and no enclosing surface is invented.
    reordered, _ = _intersections(directions, vertices, triangles[::-1], intensity)
    np.testing.assert_array_equal(reordered, distance)


def test_road_frames_preserve_shape_and_height_above_each_support_plane():
    source_plane, target_plane = np.array([.12, -.08, 2.]), np.array([-.15, .03, -1.])
    source, target = _basis([1., 2.], source_plane), _basis([-2., 1.], target_plane)
    local = np.array([[0., 0., 0.], [.2, .5, 0.], [-.2, .3, .4], [.1, -.2, 1.]])
    for frame in (source, target):
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(frame), 1., atol=1e-12)
    original = local @ source.T + [0., 0., source_plane[2]]
    placed = local @ target.T + [0., 0., target_plane[2]]
    np.testing.assert_allclose(np.linalg.norm(original[:, None] - original, axis=-1),
                               np.linalg.norm(placed[:, None] - placed, axis=-1), atol=1e-12)
    for points, plane in ((original, source_plane), (placed, target_plane)):
        height = (points[:, 2] - points[:, :2] @ plane[:2] - plane[2]) / np.sqrt(1 + plane[:2] @ plane[:2])
        np.testing.assert_allclose(height, local[:, 2], atol=1e-12)


def test_surface_builder_does_not_bridge_missing_beams_or_large_depth_jumps():
    azimuth, elevation = np.meshgrid(np.array([-.005, .005]), np.array([-.015, .015]))
    rays = np.column_stack((np.cos(elevation.ravel()) * np.cos(azimuth.ravel()),
                            np.cos(elevation.ravel()) * np.sin(azimuth.ravel()), np.sin(elevation.ravel())))
    xyz, rings, rank = 10 * rays, np.array([0, 0, 1, 1]), np.arange(32)
    assert len(_triangles(xyz, rings, rank, .01)) == 2
    assert not len(_triangles(xyz, rings * 2, rank, .01))
    assert not len(_triangles(xyz * np.array([1., 1., 2., 2.])[:, None], rings, rank, .01))


def test_out_of_supervision_range_foreground_still_occludes_background():
    x, y = np.meshgrid([2.8, 3., 3.2], [-.2, 0., .2])
    road = np.column_stack((x.ravel(), y.ravel(), np.full(9, -1.5)))
    raw = np.column_stack((np.vstack((road, [8., 0., -2.])), np.full(10, 90.), np.zeros(10))).astype(np.float32)
    donor = dict(xyz=np.array([[-1., -.4, .3], [-1., .4, .3], [-1., .4, 1.2], [-1., -.4, 1.2]]),
                 triangles=np.array([[0, 1, 2], [0, 2, 3]]), intensity=np.full(4, .4), range=3.)
    result = transplant(raw, np.r_[np.ones(9), 0], dict(pose=np.eye(4)), donor, 1, np.random.default_rng(0))
    assert result is not None
    slots, xyzi, _ = result
    distance = np.linalg.norm(xyzi[:, :3], axis=1)
    original_distance = np.linalg.norm(raw[slots, :3], axis=1)
    assert np.any(distance < 2.5) and np.all(distance < original_distance)
    np.testing.assert_allclose(xyzi[:, :3] / distance[:, None],
                               raw[slots, :3] / original_distance[:, None], atol=1e-7)
