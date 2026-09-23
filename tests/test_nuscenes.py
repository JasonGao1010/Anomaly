"""Independent geometry checks for source-only visible-surface insertion."""

from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from src.nuscenes import _basis, _instance_slots, _intersections, _triangles, transplant


def test_background_split_keeps_ignored_context_and_never_enters_object_generation(tmp_path, monkeypatch):
    import json
    import pytest
    from src import nuscenes
    from src.data import Scans, load_manifest

    meta = tmp_path / "v1.0-trainval"
    meta.mkdir()
    (meta / "log.json").write_text(json.dumps([dict(token=s, location="test") for s in ("train", "val")]))
    mapping = [dict(raw=i, name=str(i), target=int(i == 24)) for i in range(32)]
    raw = np.array([[2.5, 0, 0, 255, 0], [50, 0, 0, 10, 1], [50.1, 0, 0, 5, 2],
                    [5, 0, 0, 80, 3], [6, 0, 0, 90, 4], [0, 0, 0, 0, 5]], np.float32)
    records = {}
    for split in ("train", "val"):
        scan, label = tmp_path / f"{split}.bin", tmp_path / f"{split}.label"
        raw.tofile(scan)
        np.array([24, 24, 24, 10, 11, 24], np.uint8).tofile(label)
        records[split] = [dict(source="nuscenes", scene=split, log_token=split, token=split,
            sample_token=split, timestamp=0, frame=0, scan=str(scan), label=str(label),
            group="normal_nuscenes", subset=split, pose=np.eye(4).tolist())]
    monkeypatch.setattr(nuscenes, "sources", lambda root: (records, mapping))
    def reject_objects(*args):
        raise AssertionError("background construction must not extract or synthesize objects")
    monkeypatch.setattr(nuscenes, "_annotations", reject_objects)
    output = tmp_path / "background"
    nuscenes.build(tmp_path, output, 1, background_only=True)
    assert {p.name for p in output.iterdir()} == {"train.json", "val.json"}
    for split in ("train", "val"):
        manifest = load_manifest(output / f"{split}.json", split)
        assert manifest["summary"]["normal"] == 2
        assert manifest["summary"]["ignored_in_range"] == 2
        assert manifest["summary"]["outside_range"] == 1
        assert manifest["summary"]["empty_slots"] == 1
        assert not manifest["records"][0]["eligible"]
        sample = Scans(manifest)[0]
        np.testing.assert_array_equal(sample["xyzi"][:, :3], raw[:5, :3])
        np.testing.assert_array_equal(sample["targets"], [0, 0, -1, -1, -1])
    with pytest.raises(ValueError, match="empty output"):
        nuscenes.build(tmp_path, output, 1, background_only=True)


def test_instance_extraction_intersects_semantics_rotated_box_and_unique_identity():
    rotation = Rotation.from_euler("z", 90, degrees=True)
    center = np.array([3., -2., 1.])
    local = np.array([[1.5, 0., 0.], [0., 0., 0.], [0., 1.5, 0.],
                      [2.5, 0., 0.], [0., 0., 1.5], [-.5, 0., 0.]])
    world = rotation.apply(local) + center
    quaternion = rotation.as_quat()[[3, 0, 1, 2]].tolist()
    annotation = dict(instance_token="one-object", translation=center.tolist(), rotation=quaternion, size=[2., 4., 2.])
    other = dict(instance_token="another-object", translation=world[-1].tolist(), rotation=quaternion, size=[.4, .4, .4])
    labels = np.array([11, 24, 11, 11, 11, 11], dtype=np.uint8)
    # Swapping nuScenes width/length would incorrectly reject local x=1.5.
    np.testing.assert_array_equal(_instance_slots(world, labels, annotation, 11), [0, 5])
    np.testing.assert_array_equal(_instance_slots(world, labels, annotation, 11, [annotation, other]), [0])
    np.testing.assert_array_equal(labels, [11, 24, 11, 11, 11, 11])


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


def test_disconnected_observed_facets_do_not_fill_an_unobserved_angular_gap():
    azimuth, elevation = np.meshgrid([-.055, -.045, .045, .055], [-.015, .015])
    direction = np.column_stack((np.cos(elevation.ravel()) * np.cos(azimuth.ravel()),
                                 np.cos(elevation.ravel()) * np.sin(azimuth.ravel()), np.sin(elevation.ravel())))
    xyz, rings = 10 * direction, np.repeat([0, 1], 4)
    triangles = _triangles(xyz, rings, np.arange(32), .01)
    assert len(triangles) == 4
    distance, _ = _intersections(np.array([[1., 0., 0.]]), xyz, triangles, np.ones(len(xyz)))
    assert np.isinf(distance[0])


def test_near_measured_surface_can_be_sampled_far_away_without_resizing_or_new_rays():
    x, y = np.meshgrid([40.2, 40.4, 40.6], [-.2, 0., .2])
    road = np.column_stack((x.ravel(), y.ravel(), np.full(9, -1.5)))
    extra = np.array([[60., 0., -1.2], [60., 0., 8.], [2., 0., -.04]])
    raw = np.column_stack((np.vstack((road, extra)), np.full(12, 90.), np.zeros(12))).astype(np.float32)
    donor = dict(xyz=np.array([[-.2, -.5, .2], [-.2, .5, .2], [-.2, .5, 1.2], [-.2, -.5, 1.2]]),
                 triangles=np.array([[0, 1, 2], [0, 2, 3]]), intensity=np.full(4, .4), range=5.,
                 sensor_local=np.array([-5., 0., 1.5]))
    original = raw.copy()
    result = transplant(raw, np.r_[np.ones(9), np.zeros(3)], dict(pose=np.eye(4)),
                        donor, 1, np.random.default_rng(0), distance_bin=4)
    assert result is not None
    slots, xyzi, placement = result
    assert placement["scale"] == 1. and 40. <= placement["range"] <= 50.
    assert 9 in slots and 10 not in slots and 11 not in slots
    assert len(slots) == len(set(slots)) and np.all(slots < len(raw))
    local = (xyzi[:, :3] - placement["position_world"]) @ np.asarray(placement["basis_world"])
    np.testing.assert_allclose(local[:, 0], -.2, atol=5e-6)
    assert np.all(abs(local[:, 1]) <= .5 + 5e-6)
    assert np.all((local[:, 2] >= .2 - 5e-6) & (local[:, 2] <= 1.2 + 5e-6))
    distance = np.linalg.norm(xyzi[:, :3], axis=1)
    original_distance = np.linalg.norm(raw[slots, :3], axis=1)
    np.testing.assert_allclose(xyzi[:, :3] / distance[:, None],
                               raw[slots, :3] / original_distance[:, None], atol=1e-7)
    assert np.all(distance < original_distance)
    np.testing.assert_array_equal(raw, original)


def test_requests_respect_official_background_role_and_distance_budgets():
    from src.nuscenes import _requests

    for split, count, scene_count, per_role, per_bin in (("train", 28130, 700, 14065, 2813),
                                                        ("val", 6019, 150, 2000, 400)):
        records = [dict(token=f"{split}-{i}", sample_token=f"sample-{split}-{i}", frame=i,
                        scene=f"{split}-scene-{i % scene_count}", log_token=f"log-{i % 5}",
                        timestamp=i, subset=split) for i in range(count)]
        requests = _requests(records, split)
        assert Counter(role for role, _, _, _ in requests) == {
            "reference": per_role, "coverage": per_role, "control": per_role}
        by_role = {role: {record["token"] for r, record, _, _ in requests if r == role}
                   for role in ("reference", "coverage", "control")}
        assert all(len(tokens) == per_role for tokens in by_role.values())
        assert not (by_role["reference"] & by_role["coverage"])
        if split == "train":
            assert len(by_role["reference"] | by_role["coverage"]) == count
        else:
            assert not (by_role["control"] & (by_role["reference"] | by_role["coverage"]))
            assert len(set.union(*by_role.values())) == 6000
        coverage = [(distance, holdout) for role, _, distance, holdout in requests if role == "coverage"]
        assert Counter(distance for distance, _ in coverage) == {i: per_bin for i in range(5)}
        if split == "val":
            assert Counter(distance for distance, held in coverage if held) == {i: 120 for i in range(5)}
            assert sum(held is False for _, held in coverage) == 1400
        assert all(distance is None and holdout is None for role, _, distance, holdout in requests
                   if role in ("reference", "control"))
        assert {record["token"] for _, record, _, _ in requests} <= {record["token"] for record in records}
        assert _requests(records, split) == requests


def test_assignment_exhausts_observation_budgets_even_without_successful_placements():
    from src.nuscenes import _assign

    donors = [dict(id=f"view-{i}", instance="one-instance", geometry_group="one-shape",
                   kind="anomaly", scene="source", range=3., shape_holdout=False,
                   geometry_reliable=True, slots=np.arange(8)) for i in range(6)]
    requests = [("coverage", dict(token=f"background-{b}-{i}", scene="target",
                                   road_histogram=[1] * 5), b, False)
                for b in range(5) for i in range(13)]
    usage = Counter()
    assigned = _assign(requests, donors, usage)
    assert sum(request["donor"] is not None for request in assigned) == 60
    assert Counter(request["bin"] for request in assigned if request["donor"] is None) == dict.fromkeys(range(5), 1)
    assert set(usage.values()) == {2} and len(usage) == 6 * 5
    assert sum(usage.values()) == 60
    # Allocation spends the quota before any geometric placement is attempted.
    rejected = _assign(requests, donors, usage)
    assert all(request["donor"] is None for request in rejected)
    assert sum(usage.values()) == 60
    assert _assign(requests, donors, Counter()) == assigned


def test_assignment_requires_reliable_held_out_shapes_and_preserves_matched_controls():
    from src.nuscenes import _assign

    base = dict(kind="anomaly", scene="source", range=3., shape_holdout=True,
                geometry_reliable=True, instance="object", geometry_group="shape", slots=np.arange(8))
    donors = [dict(base, id="same-scene", scene="target"),
              dict(base, id="normal", kind="control"),
              dict(base, id="too-far", range=10.),
              dict(base, id="not-held", shape_holdout=False),
              dict(base, id="unreliable", geometry_reliable=False),
              dict(base, id="eligible")]
    record = dict(token="background", scene="target", road_histogram=[1, 0, 0, 0, 0])
    assigned = _assign([("coverage", record, 0, True)], donors, Counter())
    assert assigned[0]["donor"] == "eligible"
    assert _assign([("coverage", record, 0, True)], [], Counter())[0]["donor"] is None
    matched = dict(point_range_median=25., visible_points=20)
    controls = [dict(base, id="sparse-control", kind="control", range=10., slots=np.arange(10)),
                dict(base, id="matched-control", kind="control", range=10., slots=np.arange(125))]
    result = _assign([("control", record, None, None)], controls, Counter(), {record["token"]: matched})
    assert result[0]["donor"] == "matched-control" and result[0]["bin"] == 2
    assert result[0]["match"] is matched


def test_observed_surface_similarity_does_not_certify_shape_holdout():
    from src.nuscenes import _assign, group_geometry

    surface = np.array([[0., y, z] for y in (-.4, 0., .4) for z in (-.3, .1, .5)])
    donors = []
    for instance, offset in (("A", 0.), ("B", .08), ("C", .16)):
        for view in range(2):
            donors.append(dict(id=f"{instance}-{view}", instance=instance, timestamp=view,
                               subset="val", kind="anomaly", size=[2., 2., 2.],
                               source_view=[1., 0., 0.], object_xyz=surface + [offset, 0., .1 * view],
                               scene="source", range=3., slots=np.arange(len(surface))))
    summary = group_geometry(donors)
    # Repeat observations identify an official instance, not a complete shape family.
    assert all(donor["geometry_group"] is None for donor in donors)
    assert not any(donor["geometry_reliable"] or donor["shape_holdout"] for donor in donors)
    assert summary["groups"] is None and summary["development_reliable_groups"] == 0
    requests = [("coverage", dict(token=f"heldout-{i}", scene="target", road_histogram=[1] * 5),
                 i % 5, True) for i in range(600)]
    usage = Counter()
    assigned = _assign(requests, donors, usage)
    assert len(assigned) == 600 and all(request["donor"] is None for request in assigned)
    assert not usage
    ordinary = _assign([("reference", requests[0][1], 0, None)], donors, usage)
    assert ordinary[0]["donor"] is not None


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


def test_fixed_surface_world_hits_survive_sensor_translation_and_rotation():
    from src.nuscenes import _render_surface

    donor = dict(xyz=np.array([[0., -1., -1.], [0., 1., -1.], [0., 1., 1.], [0., -1., 1.]]),
                 triangles=np.array([[0, 1, 2], [0, 2, 3]]), intensity=np.full(4, .4),
                 sensor_local=np.array([-5., 0., 0.]))
    origin = np.array([12., 3., .5])
    basis = Rotation.from_euler("z", 25, degrees=True).as_matrix()
    original_basis, original_vertices = basis.copy(), donor["xyz"].copy()
    targets = np.array([[0., -.3, .2], [0., .4, -.4]]) @ basis.T + origin
    for yaw, translation in ((0., [0., -2., 0.]), (53., [3., 2., .5])):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        transform[:3, 3] = translation
        sensor_targets = (targets - transform[:3, 3]) @ transform[:3, :3]
        # Background returns lie behind known surface points on the same rays.
        xyz = np.vstack((2 * sensor_targets, -sensor_targets[0]))
        raw = np.column_stack((xyz, np.full(3, 90.), np.arange(3))).astype(np.float32)
        original_raw, original_transform = raw.copy(), transform.copy()
        slots, xyzi, reason = _render_surface(raw, transform, donor, origin, basis)
        assert reason == "visible"
        np.testing.assert_array_equal(slots, [0, 1])
        assert slots.dtype == np.int32 and xyzi.dtype == np.float32
        world_hits = xyzi[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(world_hits, targets, atol=2e-6)
        np.testing.assert_array_equal(raw, original_raw)
        np.testing.assert_array_equal(transform, original_transform)
    np.testing.assert_array_equal(basis, original_basis)
    np.testing.assert_array_equal(donor["xyz"], original_vertices)


def test_fixed_surface_distinguishes_occlusion_missing_rays_and_unknown_back():
    from src.nuscenes import _render_surface

    donor = dict(xyz=np.array([[0., -1., -1.], [0., 1., -1.], [0., -1., 1.]]),
                 triangles=np.array([[0, 1, 2]]), intensity=np.full(3, .4),
                 sensor_local=np.array([-5., 0., 0.]))
    origin, basis = np.array([10., 0., 0.]), np.eye(3)
    backside = np.eye(4)
    backside[0, 3] = 20.
    cases = (([5., -.15, -.15], np.eye(4), "surface_occluded"),
             ([0., 20., 0.], np.eye(4), "no_receiver_ray_in_cone"),
             ([20., .8, .8], np.eye(4), "no_surface_intersection"),
             ([-20., 0., 0.], backside, "unobserved_surface_side"))
    for xyz, transform, expected in cases:
        raw = np.array([[*xyz, 90., 0.]], dtype=np.float32)
        slots, xyzi, reason = _render_surface(raw, transform, donor, origin, basis)
        assert reason == expected
        assert slots.shape == (0,) and slots.dtype == np.int32
        assert xyzi.shape == (0, 4) and xyzi.dtype == np.float32


def test_fixed_surface_outside_supervision_still_replaces_only_existing_rays():
    from src.nuscenes import _render_surface

    donor = dict(xyz=np.array([[0., -1., -1.], [0., 1., -1.], [0., 1., 1.], [0., -1., 1.]]),
                 triangles=np.array([[0, 1, 2], [0, 2, 3]]), intensity=np.full(4, .4),
                 sensor_local=np.array([-5., 0., 0.]))
    raw = np.array([[80., 0., 0., 90., 0.], [20., .1, 0., 60., 1.],
                    [80., 0., 40., 50., 2.], [0., 0., 0., 0., 3.]], dtype=np.float32)
    original = raw.copy()
    slots, xyzi, reason = _render_surface(raw, np.eye(4), donor, np.array([55., 0., 0.]), np.eye(3))
    assert reason == "visible"
    np.testing.assert_array_equal(slots, [0])
    np.testing.assert_allclose(xyzi, [[55., 0., 0., .4]], atol=1e-7)
    hit_range = np.linalg.norm(xyzi[:, :3], axis=1)
    old_range = np.linalg.norm(raw[slots, :3], axis=1)
    assert np.all(hit_range > 50.) and np.all(hit_range < old_range)
    np.testing.assert_allclose(xyzi[:, :3] / hit_range[:, None], raw[slots, :3] / old_range[:, None])
    np.testing.assert_array_equal(raw, original)


def test_oriented_box_entry_handles_parallel_rays_and_returns_inside_box():
    from src.nuscenes import _box_entry

    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    # The rotated box spans world x in [-2,2], y in [-1,1].
    directions = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    entry = _box_entry(directions, np.array([-5., 0., 0.]), rotation, np.array([2., 4., 2.]))
    np.testing.assert_allclose(entry[0], 3., atol=1e-12)
    assert np.isinf(entry[1:]).all()
    assert np.isinf(_box_entry(directions[:1], np.array([-5., 2., 0.]), rotation, [2., 4., 2.])).all()
    np.testing.assert_array_equal(_box_entry(directions, np.zeros(3), rotation, [2., 4., 2.]), [0., 0., 0.])
    # A return inside the box is already uncertain; it need not pass its exit.
    return_ranges = np.array([2., 3.5, 8.])
    np.testing.assert_array_equal(entry[0] < return_ranges - 1e-4, [False, True, True])


def test_road_interior_requires_surrounding_measured_support():
    from src.nuscenes import _road_clearance

    road = np.array([[-1.5, -1.5, 0.], [-1.5, 1.5, 0.], [1.5, -1.5, 0.], [1.5, 1.5, 0.]])
    assert _road_clearance(road, [0., 0.]) == 1.5
    assert _road_clearance(road, [1.5, 0.]) < 1.
    assert _road_clearance(road[:2], [0., 0.]) == -np.inf



def test_sequence_keeps_supported_zero_hits_and_outside_changes_but_excludes_back(tmp_path, monkeypatch):
    import src.nuscenes as source

    mapping = [dict(raw=0, target=0, name="noise"), dict(raw=1, target=1, name="flat.driveable_surface")]
    donor = dict(id="one-view", instance="one-object", kind="anomaly", scene="source", range=5.,
                 xyz=np.array([[0., -1., -1.], [0., 1., -1.], [0., -1., 1.]]),
                 triangles=np.array([[0, 1, 2]]), intensity=np.full(3, .4),
                 sensor_local=np.array([-5., 0., 0.]), object_center_local=np.zeros(3),
                 box_rotation_local=np.eye(3), size=np.array([1., 2., 2.]))
    placement = dict(position_world=[10., 0., 0.], basis_world=np.eye(3).tolist(), range=10., scale=1.)
    clouds = (np.array([[80., -.2, -.2, 90., 0.]], np.float32),
              np.array([[0., 20., 0., 90., 0.]], np.float32),
              np.array([[10., .6, .6, 90., 0.], [5., 0., 0., 90., 1.]], np.float32),
              np.array([[-20., 0., 0., 90., 0.]], np.float32),
              np.array([[20., -.2, -.2, 90., 0.]], np.float32))
    records = []
    for index, sensor_x in enumerate((-50., 0., 0., 20., 0.)):
        pose = np.eye(4)
        pose[0, 3] = sensor_x
        records.append(dict(token=str(index), sample_token=str(index), timestamp=index,
                            scene="receiver", subset="train", pose=pose.tolist()))
    monkeypatch.setattr(source, "_read", lambda record:
                        (clouds[int(record["token"])], np.ones(len(clouds[int(record["token"])]), np.uint8)))
    monkeypatch.setattr(source, "transplant", lambda *args, **kwargs:
                        (np.empty(0, np.int32), np.empty((0, 4), np.float32), placement))
    # Placement validity is isolated here; collision rejection is tested below.
    monkeypatch.setattr(source, "_sequence_collision", lambda *args: None)
    (tmp_path / "train").mkdir()
    original, generated, report = source._sequence((records, donor, mapping, {}, str(tmp_path)))
    assert len(original) == len(report["frames"]) == 5
    assert [row["token"] for row in generated] == ["0", "2", "4"]
    assert [row["segment"] for row in generated] == [0, 0, 1]
    assert report["segments"] == 2
    assert all(row["placement"] == placement and row["donor"] == donor["id"] for row in generated)
    assert generated[0]["anomaly"] == 0 and generated[0]["visible_points"] == 1
    with np.load(generated[0]["delta"]) as delta:
        assert np.linalg.norm(delta["xyzi"][0, :3]) > 50.
        np.testing.assert_array_equal(delta["slots"], [0])
        np.testing.assert_array_equal(delta["labels"], [2])
    assert report["frames"][1]["anomaly"] == report["frames"][1]["visible_points"] == 0
    assert report["frames"][1]["status"] == "no_receiver_ray_in_cone"
    # Keep a possible unknown surface's old context, but never its normal label.
    assert generated[1]["uncertain_points"] == 1 and generated[1]["normal"] == 1
    assert report["frames"][2]["uncertain_supervised_points"] == 1
    with np.load(generated[1]["delta"]) as delta:
        np.testing.assert_array_equal(delta["slots"], [0])
        np.testing.assert_array_equal(delta["labels"], [0])
        np.testing.assert_array_equal(delta["xyzi"][:, :3], clouds[2][:1, :3])
    assert report["frames"][3]["status"] == "unobserved_surface_side"
    assert not report["frames"][3]["supported"]
    assert generated[2]["anomaly"] == 1
    assert not (tmp_path / "train" / "3.npz").exists()
    # Changing only donor supervision must not alter observation generation.
    _, controls, control_report = source._sequence((records, dict(donor, kind="control"), mapping, {}, str(tmp_path)))
    assert all(row["anomaly"] == 0 and row["group"] == "control_nuscenes" for row in controls)
    assert [f["supported"] for f in report["frames"]] == [f["supported"] for f in control_report["frames"]]
    for anomaly, control in zip(generated, controls):
        with np.load(anomaly["delta"]) as a, np.load(control["delta"]) as c:
            np.testing.assert_array_equal(a["slots"], c["slots"])
            np.testing.assert_array_equal(a["xyzi"], c["xyzi"])
            np.testing.assert_array_equal(np.where(a["labels"] == 2, 1, a["labels"]), c["labels"])


def test_sequence_rejects_fixed_placement_when_a_later_frame_collides(tmp_path, monkeypatch):
    import src.nuscenes as source

    mapping = [dict(raw=0, target=0, name="noise"), dict(raw=1, target=1, name="flat.driveable_surface")]
    donor = dict(id="one-view", instance="one-object", kind="anomaly", object_center_local=np.array([0., 0., .5]),
                 box_rotation_local=np.eye(3), size=np.ones(3))
    origin, basis = np.array([10., 0., -1.5]), np.eye(3)
    raw = np.array([[20., 0., -1.5, 90., 0.]], np.float32)
    labels = np.ones(1, np.uint8)
    records = [dict(token=str(i), sample_token=str(i), timestamp=i, scene="receiver", subset="train",
                    pose=np.eye(4).tolist()) for i in range(2)]
    frames = [(record, raw, labels, raw[:, :3]) for record in records]
    boxes = {"1": [dict(translation=[10., 0., -1.], rotation=[1., 0., 0., 0.], size=[1., 1., 1.])]}
    assert source._sequence_collision(frames, donor, origin, basis, {}, 1) is None
    assert source._sequence_collision(frames[:1], donor, origin, basis, boxes, 1) is None
    assert source._sequence_collision(frames, donor, origin, basis, boxes, 1) == "annotated_object_collision"
    monkeypatch.setattr(source, "_read", lambda record: (raw, labels))
    placement = dict(position_world=origin.tolist(), basis_world=basis.tolist(), range=10., scale=1.)
    monkeypatch.setattr(source, "transplant", lambda *args, **kwargs:
                        (np.empty(0, np.int32), np.empty((0, 4), np.float32), placement))
    def reject_render(*args):
        raise AssertionError("A conflicting fixed placement must be rejected before rendering")
    monkeypatch.setattr(source, "_render_surface", reject_render)
    original, generated, report = source._sequence((records, donor, mapping, boxes, str(tmp_path)))
    assert len(original) == 2 and not generated
    assert report["status"] == "no_sequence_placement"
    assert report["failures"] == {"annotated_object_collision": report["attempts"]}
    assert report["attempts"] > 0 and not list(tmp_path.iterdir())


def test_temporal_road_support_never_adds_receiver_rays():
    x, y = np.meshgrid([40.2, 40.4, 40.6], [-.2, 0., .2])
    support = np.column_stack((x.ravel(), y.ravel(), np.full(9, -1.5)))
    raw = np.array([[60., 0., -1.2, 90., 0.], [60., 0., 8., 90., 0.],
                    [2., 0., -.04, 90., 0.]], dtype=np.float32)
    donor = dict(xyz=np.array([[-.2, -.5, .2], [-.2, .5, .2], [-.2, .5, 1.2], [-.2, -.5, 1.2]]),
                 triangles=np.array([[0, 1, 2], [0, 2, 3]]), intensity=np.full(4, .4), range=5.,
                 sensor_local=np.array([-5., 0., 1.5]))
    inputs = (raw, np.zeros(3), dict(pose=np.eye(4)), donor, 1)
    assert transplant(*inputs, np.random.default_rng(0), distance_bin=4) is None
    result = transplant(*inputs, np.random.default_rng(0), distance_bin=4, support_road=support)
    assert result is not None
    slots, xyzi, placement = result
    np.testing.assert_array_equal(slots, [0])
    np.testing.assert_allclose(xyzi[:, :3] / np.linalg.norm(xyzi[:, :3], axis=1)[:, None],
                               raw[slots, :3] / np.linalg.norm(raw[slots, :3], axis=1)[:, None], atol=1e-7)
    assert placement["scale"] == 1. and 40. <= placement["range"] <= 50.



def test_dense_temporal_support_cannot_hide_disagreement_with_current_road():
    x, y = np.meshgrid([40.2, 40.4, 40.6], [-.2, 0., .2])
    current = np.column_stack((x.ravel(), y.ravel(), np.full(9, -1.5)))
    raw = np.column_stack((current, np.full(9, 90.), np.zeros(9))).astype(np.float32)
    # The pooled 90th percentile alone misses the displaced current scan.
    support = np.repeat(current + [0., 0., .3], 20, axis=0)
    donor = dict(xyz=np.array([[-.2, -.5, .2], [-.2, .5, .2], [-.2, .5, 1.2], [-.2, -.5, 1.2]]),
                 triangles=np.array([[0, 1, 2], [0, 2, 3]]), intensity=np.full(4, .4), range=5.)
    diagnostics = {}
    result = transplant(raw, np.ones(9), dict(pose=np.eye(4)), donor, 1, np.random.default_rng(0),
                        distance_bin=4, support_road=support, diagnostics=diagnostics)
    assert result is None
    assert sum(diagnostics["attempt_failures"].values()) == diagnostics["attempts"]
    assert set(diagnostics["attempt_failures"]) <= {"support_anchor_disagreement", "support_current_disagreement"}



def test_temporal_interior_support_does_not_require_a_current_road_return():
    x, y = np.meshgrid(np.linspace(38., 43., 26), np.linspace(-2., 2., 21))
    support = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.5)))
    raw = np.array([[60., 0., -1.2, 90., 0.], [60., 0., 8., 90., 0.],
                    [2., 0., -.04, 90., 0.]], np.float32)
    donor = dict(xyz=np.array([[-.2, -.5, .2], [-.2, .5, .2], [-.2, .5, 1.2], [-.2, -.5, 1.2]]),
                 triangles=np.array([[0, 1, 2], [0, 2, 3]]), intensity=np.full(4, .4), range=5.,
                 sensor_local=np.array([-5., 0., 1.5]))
    original = raw.copy()
    inputs = (raw, np.zeros(3), dict(pose=np.eye(4)), donor, 1)
    assert transplant(*inputs, np.random.default_rng(0), distance_bin=4, interior=True) is None
    slots, xyzi, placement = transplant(*inputs, np.random.default_rng(0), distance_bin=4,
                                        interior=True, support_road=support)
    np.testing.assert_array_equal(slots, [0])
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_allclose(xyzi[:, :3] / np.linalg.norm(xyzi[:, :3], axis=1)[:, None],
                               raw[slots, :3] / np.linalg.norm(raw[slots, :3], axis=1)[:, None], atol=1e-7)
    assert placement["scale"] == 1.



def test_reviewed_normals_are_point_specific_and_foreground_overrides_them(tmp_path):
    import pytest
    from src.data import read_nuscenes

    raw = np.array([[5., 0., 0., 100., 0.], [6., 0., 0., 90., 1.],
                    [7., 0., 0., 80., 2.]], np.float32)
    scan, label = tmp_path / 'scan.bin', tmp_path / 'label.bin'
    raw.tofile(scan)
    np.array([0, 0, 1], np.uint8).tofile(label)
    mapping = [dict(raw=0, name='static.manmade', target=0),
               dict(raw=1, name='flat.driveable_surface', target=1)]
    record = dict(scan=str(scan), label=str(label), token='native', frame=0, normal_slots=[0])
    np.testing.assert_array_equal(read_nuscenes(record, mapping).labels, [1, 0, 1])
    delta = tmp_path / 'delta.npz'
    np.savez(delta, token=np.asarray('native'), slots=np.array([0]),
             xyzi=np.array([[4., 0., 0., .3]], np.float32), labels=np.array([2], np.uint32))
    np.testing.assert_array_equal(read_nuscenes(dict(record, delta=str(delta)), mapping).labels, [2, 0, 1])
    for slots in ([2], [0, 0], [-1], [3]):
        with pytest.raises(ValueError, match='supplemental normal'):
            read_nuscenes(dict(record, normal_slots=slots), mapping)



def test_sequence_writer_discards_only_effectively_unchanged_observations(tmp_path):
    from src.nuscenes import _write_sequences

    raw = np.array([[5., 0., 0., 100., 0.], [6., 0., 0., 90., 1.]], np.float32)
    scan, label = tmp_path / 'scan.bin', tmp_path / 'label.bin'
    raw.tofile(scan)
    np.array([0, 1], np.uint8).tofile(label)
    mapping = [dict(raw=0, name='noise', target=0), dict(raw=1, name='flat.driveable_surface', target=1)]
    directory = tmp_path / 'train'
    directory.mkdir()
    original = dict(token='native', frame=0, scene='scene', subset='train', log_token='log',
                    scan=str(scan), label=str(label), role='original', normal=1, anomaly=0, eligible=False)
    rows = [original]
    for index, role in enumerate(('sequence', 'control')):
        delta = directory / f'{role}.npz'
        xyzi = raw[index:index + 1, :4].copy()
        xyzi[:, 3] /= 255.
        np.savez(delta, slots=np.array([index]), labels=np.array([0], np.uint32),
                 xyzi=xyzi, token=np.asarray('native'))
        rows.append(dict(original, role=role, normal=1 - index, delta=str(delta),
                         visible_points=0, uncertain_points=1, point_histogram=[0]*5, instance=role))
    # Extra already-ignored slots do not make a second effective observation.
    duplicate = directory / 'control_r3.npz'
    xyzi = raw[:, :4].copy()
    xyzi[:, 3] /= 255.
    np.savez(duplicate, slots=np.array([0, 1]), labels=np.array([0, 0], np.uint32),
             xyzi=xyzi, token=np.asarray('native'))
    rows.append(dict(rows[-1], delta=str(duplicate), uncertain_points=2))
    report = dict(scenes=[dict(scene='scene', subset='train', kind='anomaly', status='placed', frames=[])], surfaces={})
    result = _write_sequences(tmp_path, tmp_path, mapping, report, {'train': rows}, 0.)['train']
    assert [r['role'] for r in result['records']] == ['original', 'control']
    assert result['recipe']['summary']['discarded_unchanged'] == 1
    assert result['recipe']['summary']['discarded_repeated_ignore_observations'] == 1
    assert not (directory / 'sequence.npz').exists()
    assert not duplicate.exists()
    assert (directory / 'control.npz').exists()


def test_reviewed_identity_exclusions_survive_repeated_catalog_admission():
    from src.nuscenes import _object_admission

    common = dict(category='movable_object.pushable_pullable', raw_label=19,
                  appearance='container', object_group='container', points=8)
    catalog = dict(selected=[dict(common, review_id=1, subset='train'),
                            dict(common, review_id=2, subset='val',
                                 identity_review=dict(excluded=True, reason='Suspected shared physical object'))],
                   excluded=[], deferred=[], limits=[''])
    for _ in range(2):
        result = _object_admission(catalog)
        assert [r['review_id'] for r in result['selected']] == [1]
        assert result['excluded'][0]['admission_reason'] == 'Suspected shared physical object'
        assert result['summary']['admitted_instances'] == 1


def test_sequence_variants_preserve_native_identity_and_invalidate_changed_mapping(tmp_path, monkeypatch):
    import src.nuscenes as source

    mapping = [dict(raw=0, target=0, name='noise'), dict(raw=1, target=1, name='flat.driveable_surface')]
    record = dict(token='native', sample_token='native', timestamp=0, scene='receiver',
                  subset='train', pose=np.eye(4).tolist())
    donor = dict(id='surface', instance='object', kind='anomaly', scene='source', range=5.,
        xyz=np.array([[0., -1., -1.], [0., 1., -1.], [0., -1., 1.]]),
        triangles=np.array([[0, 1, 2]]), intensity=np.full(3, .4), sensor_local=np.array([-5., 0., 0.]),
        object_center_local=np.zeros(3), box_rotation_local=np.eye(3), size=np.array([1., 2., 2.]))
    reads = []
    def read(row):
        reads.append(row['token'])
        return np.array([[40., 0., 0., 90., 0.]], np.float32), np.ones(1, np.uint8)
    def place(*args, distance_bin=None, **kwargs):
        distance = 10. if distance_bin is None else 25.
        return np.array([0]), np.array([[distance, 0., 0., .4]], np.float32), dict(
            position_world=[distance, 0., 0.], basis_world=np.eye(3).tolist(), range=distance, scale=1.)
    monkeypatch.setattr(source, '_SEQUENCE_CACHE', None)
    monkeypatch.setattr(source, '_read', read)
    monkeypatch.setattr(source, 'transplant', place)
    monkeypatch.setattr(source, '_sequence_collision', lambda *args: None)
    monkeypatch.setattr(source, '_road_clearance', lambda *args: 1.)
    (tmp_path / 'train').mkdir()
    task = ([record], donor, mapping, {}, str(tmp_path))
    originals, base, _ = source._sequence((*task, None))
    _, far, report = source._sequence((*task, 2))
    assert reads == ['native'] and originals[0]['normal'] == 1
    assert base[0]['token'] == far[0]['token'] == 'native'
    assert base[0]['delta'] != far[0]['delta']
    assert report['variant'] == 'r2' and far[0]['inserted_point_histogram'] == [0, 0, 1, 0, 0]
    with np.load(base[0]['delta']) as a, np.load(far[0]['delta']) as b:
        np.testing.assert_array_equal(a['slots'], b['slots'])
        np.testing.assert_array_equal(a['xyzi'][:, 0], [10.])
        np.testing.assert_array_equal(b['xyzi'][:, 0], [25.])
    changed = [mapping[0], dict(mapping[1], target=0)]
    originals, _, _ = source._sequence(([record], donor, changed, {}, str(tmp_path), None))
    assert reads == ['native', 'native'] and originals[0]['normal'] == 0
