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
