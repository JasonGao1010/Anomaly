"""Diagnostic fixtures only: these choices are NOT V4 generation parameters."""

from dataclasses import replace
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from src.data import (Frame, Rays, STUSequence, point_targets, read_delta, read_rays,
                      restore_delta, supervision, validate_delta)
from src.shape import Shape, Trace, unresolved_penetration
from src.render import (Material, Object, Response, World, check_grounding,
                        ground_object, observed_collision, pair_collision, render_frame,
                        condition_cell, condition_coverage)


# Explicit numerical and response fixtures test semantics, not scientific coverage.
TRACE = Trace(96, 8, 24, 1e-5, 4., 1e-5, 1e-5, 1e-9)
MATERIAL = Material(.5, .1, 0.)
SUPPORT = dict(xy_resolution=33, z_steps=129, bisections=24, refinements=5)
FINE_SUPPORT = dict(xy_resolution=65, z_steps=257, bisections=24, refinements=5)
PENETRATION = dict(allowance_m=.05, gradient_step_m=1e-6, witness_fraction=1 - 1e-6)


def test_condition_coverage_counts_source_geometry_not_scaled_variants():
    descriptor = [math.log(15),1,0,0,math.log(30),0,0,0,0,0,0,math.log1p(.3),0,0,1,0,0,0,0,0,0,0]
    row = dict(domain="nuScenes",geometry="same-source",log="log-a",descriptor=descriptor)
    rows = [dict(row,variant="small"),dict(row,variant="large")]
    cells = condition_coverage(rows,{"nuScenes":[.1,.2]})
    assert condition_cell(descriptor,[.1,.2]) == (1,1,2)
    assert cells["nuScenes",1,1,2]["observations"] == 2
    assert cells["nuScenes",1,1,2]["geometries"] == {"same-source"}
    tiny = descriptor.copy();tiny[4] = math.log(4)
    assert condition_cell(tiny,[.1,.2]) is None  # Kept in a qualified multi-object scan, outside this >=5 view table.


@pytest.mark.parametrize("source", ["rendered_stu", "nuscenes"])
def test_object_relations_separate_worlds_and_check_scan_identity(tmp_path, source):
    from src.analyze import native_relations_scene
    records = []
    for number in range(2):
        pose = np.eye(4); pose[0, 3] = 5
        world = tmp_path / f"world-{number}.json"
        world.write_text(json.dumps(dict(seed=number, scene="same-scene", log_token="same-log",
            objects=[dict(object_id=1, geometry=f"variant-{number}", source_geometry="one-source",
                          anchor_frame=0, pose=pose.tolist())])))
        delta = tmp_path / f"frame-{number}.npz"
        record = dict(source=source, frame=0, token="one-token", source_identity="one-scan",
                      world=str(world), delta=str(delta), anomaly=5, pose=np.eye(4).tolist())
        payload = dict(frame=0, token="one-token", source_identity="one-scan", world=str(world),
                       xyzi=np.tile([4.,0,0,.2], (5,1)).astype(np.float32),
                       labels=np.full(5,2,np.uint32), object_ids=np.ones(5,np.int64))
        if source == "rendered_stu":
            record.pop("token"); payload.pop("token")
        np.savez(delta, **payload)
        records.append(record)
    rows = native_relations_scene(records)
    assert len(rows) == 2 and sum(row["views"] for row in rows) == 2
    assert {row["geometry"] for row in rows} == {"one-source"}
    field = "source_identity" if source == "rendered_stu" else "token"
    records[0][field] = "wrong-scan"
    with pytest.raises(ValueError, match="another scan"):
        native_relations_scene(records)


def test_ground_site_search_keeps_full_footprint_support():
    from scipy.spatial import cKDTree
    from src.render import _support_plane
    x, y = np.meshgrid(np.linspace(-2,2,31),np.linspace(-2,2,31))
    ground = np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    context = dict(ground=ground,ground_tree=cKDTree(ground[:,:2]))
    support = _support_plane(context,np.zeros(3),1.7)
    assert support is not None and support["residual"] == 0
    np.testing.assert_array_equal(support["normal"],[0,0,1])
    assert _support_plane(context,np.array([1.9,0,0]),1.7) is None


def sphere(radius=.5):
    return Shape(((radius, radius, radius),), ((0, 0, 0),), ((1, 1),), (0,),
                 ("union",), 0., (0., 0.), (0., 0.), 0., (1., 1., 1.), (0., 0., 0.))


def object_at(object_id, position, shape=None):
    pose = np.eye(4)
    pose[:3, 3] = position
    return Object(object_id, f"diagnostic-{object_id}", sphere() if shape is None else shape, MATERIAL, pose)


def rays_for(directions):
    directions = np.asarray(directions, np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return Rays(directions, np.zeros_like(directions), np.arange(len(directions)), np.array([[1., 0, 0]]))


def response(probabilities=(1.,), intensity=1.234567):
    ranges = np.array([0., 3.5, 100.]) if len(probabilities) == 2 else np.array([0., 100.])
    probability = np.asarray(probabilities).reshape(1, -1, 1)
    return Response(ranges, np.array([0., math.pi / 2]), np.array([0., 1.]), probability,
                    np.full(probability.shape + (2,), intensity), (0., 2.), 1 / 3500,
                    "Diagnostic constant response; not fitted data or a V4 default")


def test_native_column_beams_and_motion_shifted_empty_slots():
    directions = np.tile([1., 0, 0], (4, 1))
    rays = Rays(directions, np.tile([.1, 0, 0], (4, 1)), np.arange(4),
                np.tile([1., 0, 0], (2, 1)), np.array([0, 1, 0, 1]), np.array([True, True, False, False]))
    source = Frame(0, np.array([[7, 0, 0, .2], [7, 0, 0, .2], [.1, 0, 0, 0], [.1, 0, 0, 0]], np.float32),
                   np.eye(4), np.array([1, 1, 0, 0], np.uint32), sequence_id=0)
    response = Response(np.array([0., 100.]), np.array([0., math.pi/2]), np.array([0., 1.]),
        np.ones((2, 1, 1)), np.array([.1, .8])[:, None, None, None] * np.ones((2, 1, 1, 2)),
        (0., 1.), None, "Diagnostic beam identity fixture")
    world = World("diagnostic", 0, 19, (object_at(1, [3.5, 0, 0]),), 1e-6)
    result = render_frame(source, world, rays, response, TRACE)
    assert result.inserted.all() and (result.frame.labels == 2).all()
    np.testing.assert_allclose(result.frame.xyzi[:, 3], [.1, .8, .1, .8])
    np.testing.assert_allclose(result.frame.xyzi[:, 0], 3., atol=2e-7)
    assert result.sampling[0]["potential_surface_rays"] == 4
    assert result.sampling[0]["foreground_surface_rays"] == 4
    assert result.sampling[0]["returned_rays"] == 4


def test_observation_selection_preserves_base_and_real_view_changes_without_quotas():
    from src.data import observation_descriptor, select_observations
    cloud = np.array([[5,0,0,.2],[5,.1,0,.2],[5,0,.1,.2],[5,.1,.1,.2],[5,.2,0,.2]], np.float32)
    obj = np.eye(4)
    descriptors = []
    for angle in np.deg2rad(np.arange(0,180,20)):
        pose = np.eye(4)
        pose[:3,3] = [5*np.cos(angle),5*np.sin(angle),0]
        descriptors.append(observation_descriptor(cloud,obj,pose,[20,0,0,0,0,0,0,0]))
    descriptors.append(descriptors[0])
    kept, distance = select_observations(descriptors, [0,9])
    assert kept == list(range(10)) and distance.max() == 0
    kept, distance = select_observations(descriptors, [0])
    assert kept == list(range(9)) and distance.max() < 1e-6


@pytest.mark.parametrize("count", [0, 1, 4, 5])
def test_frame_selection_and_full_input(count):
    xyzi = np.array([[3., 0, 0, 1.2]] * 5 + [[0, 0, 0, 9], [2.5, 0, 0, .2],
                    [50, 0, 0, .2], [2.49, 0, 0, .2], [50.1, 0, 0, .2], [4, 0, 0, .2]], np.float32)
    packed = np.array([40] * 5 + [2, 40, 40, 2, 2, 0], np.uint32)
    packed[:count] = 2
    frame = Frame(0, xyzi, np.eye(4), packed)
    selected = supervision(frame)
    assert selected.anomaly_count == count
    assert selected.eligible == (count >= 5)
    assert selected.normal_count == 7 - count
    assert np.all(selected.targets == -1) if count < 5 else np.array_equal(selected.targets, point_targets(frame))
    assert np.array_equal(frame.return_slots, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
    np.testing.assert_array_equal(frame.xyzi, xyzi)
    # Equal coordinates remain distinct points, and input ownership is immutable.
    xyzi[0] = 0
    assert frame.xyzi[0, 0] == 3
    assert not frame.xyzi.flags.writeable


@pytest.mark.parametrize("eligible", [False, True])
def test_saved_206_reuse_preserves_full_scan_and_selection(eligible):
    data_root = Path(os.environ.get("STU_DATA_ROOT", "/home/jasongao/Data/STU"))
    pool = Path(os.environ.get("AJAE_SAMPLES_ROOT", Path(__file__).resolve().parents[2] / "AJAE/results/synthetic"))
    if not (data_root / "train/206").is_dir() or not (pool / "manifest.json").is_file():
        pytest.skip("real STU/206 and existing AJAE samples are required")
    entry = json.loads((pool / "manifest.json").read_text())["splits"]["train"]["worlds"][0]
    saved = json.loads((pool / entry["path"] / "manifest.json").read_text())
    record = next(row for row in saved["frames"] if row["in_range"] >= 5) if eligible else next(
        row for row in saved["frames"] if 1 <= row["in_range"] <= 4)
    original = STUSequence(data_root)[record["frame"]]
    path = pool / entry["path"] / "frames" / f"{record['frame']:06d}.npz"
    frame = restore_delta(path, original, entry["world_identity"])
    selected = supervision(frame)
    assert len(frame.xyzi) == len(original.xyzi) == 131072
    assert selected.anomaly_count == record["in_range"]
    assert selected.eligible == eligible
    delta = read_delta(path)
    untouched = np.ones(len(frame.xyzi), bool)
    untouched[delta["source_slot"]] = False
    np.testing.assert_array_equal(frame.xyzi[untouched], original.xyzi[untouched])
    np.testing.assert_array_equal(frame.labels[untouched], original.labels[untouched])
    assert selected.normal_count == int((point_targets(original) == 0).sum()) - int(
        (point_targets(original)[delta["occluded_slot"]] == 0).sum())
    if not eligible:
        assert np.all(selected.targets == -1)
    # Same-shaped data from another source must never be accepted as this scan.
    changed = original.xyzi.copy()
    changed[0, 3] += 1
    with pytest.raises(ValueError, match="different source or world"):
        validate_delta(delta, replace(original, xyzi=changed), entry["world_identity"])


def test_ellipsoid_intersection_matches_analytic_roots():
    shape = replace(sphere(), scales=((.7, .4, .2),))
    y, z = np.meshgrid(np.linspace(-.6, .6, 27), np.linspace(-.35, .35, 21))
    origins = np.column_stack((np.full(y.size, -3.), y.ravel(), z.ravel()))
    directions = np.broadcast_to([1., 0, 0], origins.shape)
    radicand = 1 - (origins[:, 1] / .4)**2 - (origins[:, 2] / .2)**2
    # Exclude exact tangencies: an isolated zero need not yield a sign-change bracket.
    selected = np.abs(radicand) > 1e-12
    distance, normal, valid = shape.intersect(origins[selected], directions[selected], TRACE)
    expected = radicand[selected] > 0
    np.testing.assert_array_equal(valid, expected)
    np.testing.assert_allclose(distance[expected], 3 - .7 * np.sqrt(radicand[selected][expected]), atol=2e-7, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(normal[valid], axis=1), 1., atol=1e-12)
    inside_distance, _, inside_valid = shape.intersect([0., 0, 0], [[1., 0, 0]], TRACE)
    assert inside_valid[0]
    np.testing.assert_allclose(inside_distance, [.7], atol=2e-7)


def test_thin_csg_and_deformed_bounds():
    shell = replace(sphere(1), scales=((1., 1., 1.), (.999, .999, .999)),
                    offsets=((0, 0, 0),) * 2, exponents=((1, 1),) * 2,
                    yaws=(0., 0.), operations=("union", "difference"))
    distance, _, valid = shell.intersect([-3., 0, 0], [[1., 0, 0]], TRACE)
    assert valid[0]
    np.testing.assert_allclose(distance, [2.], atol=2e-7)
    shape = replace(sphere(), offsets=((1.2, -.3, 1.5),), twist=.7, bend=(.1, -.08), taper=(.4, -.2))
    axes = np.linspace(-1, 1, 17)
    points = np.stack(np.meshgrid(axes, axes, axes), axis=-1).reshape(-1, 3)
    points = points[np.linalg.norm(points, axis=1) <= 1] * .5 + shape.offsets[0]
    deformed = shape.deform(points)
    np.testing.assert_allclose(shape.undeform(deformed), points, atol=1e-14)
    lo, hi = shape.bounds()
    assert ((deformed >= lo) & (deformed <= hi)).all()
    assert (np.linalg.norm(deformed, axis=1) <= shape.radius).all()
    target = shape.deform(np.array(shape.offsets))
    origins = np.broadcast_to([-4., 0, 1.5], target.shape)
    direction = target - origins
    first, _, hit = shape.intersect(origins, direction, TRACE)
    dense, _, dense_hit = shape.intersect(origins, direction, replace(TRACE, steps=1024, adaptive_depth=12))
    assert hit.all() and dense_hit.all()
    np.testing.assert_allclose(first, dense, atol=2e-7, rtol=0)


def test_grazing_stu_ray_converges_at_directed_generation_resolution():
    from scipy.optimize import brentq
    shape = replace(sphere(),scales=((1.515522358000158,.09499110760264762,.7640468477787957),),
                    exponents=((.3110289873700459,1.5780231404014864),))
    origin = np.array([[.6411699450929503,-.9937730547190656,1.1998403818035264]])
    direction = np.array([[.5360949500428284,.7513962861510644,-.38471525274800933]])
    direction /= np.linalg.norm(direction,axis=1,keepdims=True)
    expected = brentq(lambda t:float(shape.level(origin+t*direction)[0]),1.32255,1.32256,xtol=1e-14)
    for steps,depth in ((192,12),(384,14),(768,16)):
        distance,_,valid = shape.intersect(origin,direction,replace(TRACE,steps=steps,adaptive_depth=depth))
        assert valid[0]
        np.testing.assert_allclose(distance,[expected],rtol=0,atol=2e-12)


def test_joint_occlusion_missing_return_and_unchanged_background():
    rays = rays_for([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    xyzi = np.array([[7, 0, 0, .2], [2, 0, 0, .3], [0, 4, 0, .4], [0, 0, 0, .9], [0, 6, 0, .8]], np.float32)
    packed = np.array([40 | (8 << 16), 10 | (8 << 16), 0, 0, 1 | (9 << 16)], np.uint32)
    source = Frame(0, xyzi, np.eye(4), packed)
    front, rear = object_at(1, [3.5, 0, 0]), object_at(2, [5., 0, 0])
    world = World("diagnostic", 206, 13, (rear, front), 1e-6)
    missing = render_frame(source, world, rays, response((0., 1.)), TRACE)
    assert not missing.inserted.any()
    assert missing.occluded_original.tolist() == [True, False, False, False, False]
    np.testing.assert_array_equal(missing.frame.xyzi[0], np.zeros(4))
    assert missing.frame.labels[0] == 0
    np.testing.assert_array_equal(missing.frame.xyzi[1:], xyzi[1:])
    observed = render_frame(source, world, rays, response(), TRACE)
    assert observed.inserted.tolist() == [True, False, False, True, False]
    assert observed.object_ids.tolist() == [1, -1, -1, 1, -1]
    assert observed.frame.labels.tolist() == [2, int(packed[1]), 0, 2, int(packed[4])]
    assert observed.frame.xyzi[0, 3] > 1  # Do not clip the source intensity format to [0,1].
    np.testing.assert_allclose(observed.frame.xyzi[[0, 3], 0], 3., atol=2e-7)
    np.testing.assert_array_equal(observed.frame.xyzi[[1, 2, 4]], xyzi[[1, 2, 4]])
    np.testing.assert_array_equal(source.xyzi, xyzi)
    assert observed.visible_normal.tolist() == [False, True, False, False, True]
    repeated = render_frame(source, world, rays, response(), TRACE)
    np.testing.assert_array_equal(repeated.frame.xyzi, observed.frame.xyzi)


def test_world_is_fixed_across_poses_and_object_counts_follow_frame_selection():
    directions = [[1., 0, 0]] * 3 + [[0., 1, 0]] * 20
    rays = rays_for(directions)
    xyzi = np.column_stack((np.asarray(directions) * 10., np.full(23, .2))).astype(np.float32)
    source = Frame(0, xyzi, np.eye(4), np.full(23, 40, np.uint32))
    world = World("diagnostic-counts", 206, 7, (object_at(1, [4., 0, 0]), object_at(2, [0, 4., 0])), 1e-6)
    observation = render_frame(source, world, rays, response(), TRACE)
    assert observation.effective_counts() == {1: 3, 2: 20}
    assert supervision(observation.frame).anomaly_count == 23
    only_small = render_frame(source, replace(world, objects=world.objects[:1]), rays, response(), TRACE)
    assert only_small.effective_counts() == {1: 0}
    assert not supervision(only_small.frame).eligible
    pose = np.eye(4)
    pose[:3, :3] = [[0., -1., 0], [1., 0, 0], [0, 0, 1.]]
    pose[:3, 3] = [4., -5., 0]
    next_frame = Frame(1, xyzi, pose, source.labels)
    later = render_frame(next_frame, world, rays, response(), TRACE)
    assert (later.object_ids[:3] == 1).all()
    measured_world = later.frame.xyzi[:3, :3] @ pose[:3, :3].T + pose[:3, 3]
    np.testing.assert_allclose(measured_world, np.tile([4., -.5, 0], (3, 1)), atol=2e-7)
    np.testing.assert_array_equal(world.objects[0].pose[:3, 3], [4., 0, 0])


def test_grounding_and_euclidean_penetration_witnesses():
    shape = sphere()
    check = check_grounding(shape, coarse=SUPPORT, fine=FINE_SUPPORT, trace=TRACE,
                            surface_count=128, residual_tolerance=1e-6, convergence_m=1e-6,
                            buried_depth_m=1e-5, max_buried_fraction=0.)
    assert check.accepted
    normal = np.array([.1, -.2, 1.])
    normal /= np.linalg.norm(normal)
    item = ground_object(check, MATERIAL, object_id=1, geometry_id="diagnostic-sphere", anchor_world=[2., 3., 9.],
                         normal_world=normal, plane_offset=-1., yaw=.4)
    contact = item.pose[:3, 3] - .5 * normal
    np.testing.assert_allclose(contact @ normal, 1., atol=1e-8)
    world_points = np.array([[0., 0, 0], [.48, 0, 0], [.6, 0, 0]]) @ item.pose[:3, :3].T + item.pose[:3, 3]
    rejected, _ = observed_collision(item, world_points, **PENETRATION)
    np.testing.assert_array_equal(rejected, [0])
    # The flattened primitive's small implicit level is not a small physical depth.
    flat = replace(shape, scales=((1., 1., .01),))
    unresolved, levels = unresolved_penetration(flat, np.array([[0., 0, 0]]),
                                               **dict(PENETRATION, allowance_m=.005))
    assert unresolved[0] and levels[0] == -.01
    a = object_at(1, [0, 0, 0])
    b = object_at(2, [.8, 0, 0])
    options = dict(trace=TRACE, surface_count=128, residual_tolerance=1e-6, interior_power=7, **PENETRATION)
    assert pair_collision(a, b, **options)[0]
    assert not pair_collision(a, object_at(2, [1.1, 0, 0]), **options)[0]
    shell = replace(sphere(1.), scales=((1., 1., 1.), (.8, .8, .8)), offsets=((0, 0, 0),) * 2,
                    exponents=((1, 1),) * 2, yaws=(0., 0.), operations=("union", "difference"))
    # Empty subtracted centers must not falsely collide with a sphere in the cavity.
    assert not pair_collision(object_at(1, [0, 0, 0], shell), object_at(2, [0, 0, 0], sphere(.2)), **options)[0]


def test_parameters_and_geometry_identity_are_explicit():
    with pytest.raises(TypeError):
        Trace()
    with pytest.raises(TypeError):
        Response()
    a = object_at(1, [0, 0, 0])
    b = replace(object_at(2, [3, 0, 0]), geometry_id=a.geometry_id, shape=sphere(.8))
    with pytest.raises(ValueError, match="different shapes"):
        World("diagnostic", 206, 1, (a, b), 1e-6)
    with pytest.raises(ValueError, match="duplicate"):
        World("diagnostic", 206, 1, (a, a), 1e-6)


def test_response_bins_quantiles_and_probability_endpoints():
    probability = np.full((2, 2, 2), .5)
    probability[0, 0, 0], probability[1, 1, 0] = 0., 1.
    values = np.arange(8).reshape(2, 2, 2, 1) / 10 + np.array([0., .3, .7])
    sensor = Response(np.array([0., 5., 20.]), np.array([0., .5, math.pi / 2]),
                      np.array([0., .25, 1.]), probability, values, (0., 2.), 1 / 3500,
                      "Diagnostic table lookup and interpolation")
    beam, ranges, angles = np.array([0, 1, 0, 1]), np.array([1., 5., 10., 100.]), np.array([0., .3, .5, math.pi / 2])
    material = Material(.4, .8, .2)
    random = np.array([0., .25, .75, 1.])
    returned, intensity = sensor.sample(beam, ranges, angles, material,
                                        np.array([0., 1., .59, .60]), random)
    # logit(.5) + 2*.2 gives probability 0.59868766... in the last two cells.
    np.testing.assert_array_equal(returned, [False, True, True, False])
    cells = [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
    quantiles = np.clip(material.quantile + material.roughness * (random - .5), 0, 1)
    expected = np.array([np.interp(q, sensor.quantiles, values[cell]) for cell, q in zip(cells, quantiles)], np.float32)
    expected = np.rint(expected.astype(float) * 3500) / 3500
    np.testing.assert_allclose(intensity, expected, atol=1e-15, rtol=0)
    previous = None
    for quantile in (0., .25, .5, 1.):
        mask, values = sensor.sample(beam, ranges, angles, replace(material,quantile=quantile),
                                     np.array([0.,1.,.59,.60]), random)
        np.testing.assert_array_equal(mask, returned)
        if previous is not None:
            assert (values >= previous).all()
        previous = values


def test_real_206_observation():
    """Exercise original full scans with one explicitly diagnostic grounded sphere."""
    root = Path(os.environ.get("STU_DATA_ROOT", "/home/jasongao/Data/STU"))
    if not (root / "train/206").is_dir():
        pytest.skip("set STU_DATA_ROOT to run the real 206 integration check")
    sequence, rays = STUSequence(root), read_rays()
    source = sequence[100]
    road = source.xyzi[source.actual & (source.semantic == 40), :3].astype(np.float64)
    anchor_sensor = road[np.argmin(np.linalg.norm(road[:, :2] - [12., 0.], axis=1))]
    patch = road[np.linalg.norm(road[:, :2] - anchor_sensor[:2], axis=1) < 1.]
    # This local plane fit is a test fixture, not a migrated support-selection rule.
    patch_world = patch @ source.pose[:3, :3].T + source.pose[:3, 3]
    center = patch_world.mean(axis=0)
    _, _, axes = np.linalg.svd(patch_world - center, full_matrices=False)
    normal = axes[-1] * np.sign(axes[-1, 2])
    grounding = check_grounding(sphere(.4), coarse=SUPPORT, fine=FINE_SUPPORT, trace=TRACE,
                                surface_count=128, residual_tolerance=1e-6, convergence_m=1e-6,
                                buried_depth_m=1e-5, max_buried_fraction=0.)
    item = ground_object(grounding, MATERIAL, object_id=1, geometry_id="diagnostic-real-sphere",
                         anchor_world=anchor_sensor @ source.pose[:3, :3].T + source.pose[:3, 3],
                         normal_world=normal, plane_offset=-normal @ center, yaw=0.)
    world = World("diagnostic-real-206", 206, 13, (item,), 1e-6)
    constant = response()
    sensor = replace(constant, probability=np.repeat(constant.probability, 128, axis=0),
                     intensity=np.repeat(constant.intensity, 128, axis=0))
    no_return = replace(sensor, probability=np.zeros_like(sensor.probability))
    records = []
    for frame_id in (100, 101, 102):
        frame = sequence[frame_id]
        observed = render_frame(frame, world, rays, sensor, TRACE)
        missing = render_frame(frame, world, rays, no_return, TRACE)
        changed = observed.inserted | observed.occluded_original
        assert observed.inserted.any()
        np.testing.assert_array_equal(observed.frame.xyzi[~changed], frame.xyzi[~changed])
        np.testing.assert_array_equal(observed.frame.labels[~changed], frame.labels[~changed])
        assert not missing.inserted.any()
        assert not missing.frame.xyzi[missing.occluded_original].any()
        assert not missing.frame.labels[missing.occluded_original].any()
        points_world = observed.frame.xyzi[observed.inserted, :3] @ frame.pose[:3, :3].T + frame.pose[:3, 3]
        radii = np.linalg.norm(points_world - item.pose[:3, 3], axis=1)
        np.testing.assert_allclose(radii, .4, atol=2e-6, rtol=0)
        normal_points = frame.xyzi[frame.actual & (frame.semantic != 0) & (frame.semantic != 2), :3]
        normal_world = normal_points @ frame.pose[:3, :3].T + frame.pose[:3, 3]
        collisions, _ = observed_collision(item, normal_world, **PENETRATION)
        assert not len(collisions)
        selected = supervision(observed.frame)
        records.append(dict(frame=frame_id, slots=len(frame.xyzi), inserted=int(observed.inserted.sum()),
                            removed_background=int(observed.occluded_original.sum()),
                            valid_anomaly=selected.anomaly_count, eligible=selected.eligible))
    print("real 206 diagnostic:", records)
