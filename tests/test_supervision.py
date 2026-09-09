import heapq
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull

from src.supervision import (
    ScanGeometry, _conditions, _inside_hull, _surface_chunk, boundary_targets,
    sampling_targets, surface_targets, surface_probe,
)


CONFIG = json.loads(Path("protocol/v1.json").read_text())["supervision"]


def test_normal_manifest_does_not_require_anomaly_range():
    assert _conditions(dict(count=0, slots=131072), .1) == ["normal"]


def geometry(xyz, slots=None):
    if slots is None:
        slots = np.arange(len(xyz), dtype=np.int32)
    return ScanGeometry(np.column_stack((xyz, np.full(len(xyz), .3))).astype(np.float32),
                        slots, CONFIG["common"]["sampling_scale"])


def test_scale_uses_complete_distinct_geometry_and_slot_ties():
    xyz = np.array([[1, 0, 0], [1.125, 0, 0], [.875, 0, 0], [1, .125, 0],
                    [1, -.125, 0], [1, 0, .125], [1, 0, -.125], [1, 0, 0], [4, 0, 0]])
    g = geometry(xyz)
    center = g.inverse[0]
    np.testing.assert_array_equal(g.representative[g.neighbors[center]], np.arange(1, 7))
    assert g.delta[center] == .125 and not g.scale_valid[g.inverse[-1]]
    assert len(g.xyz) == len(xyz) - 1
    labels = np.zeros(len(xyz), np.int8)
    labels[7] = 1
    assert g.groups(labels)[center] == -1
    target = boundary_targets(g, labels, CONFIG["C1"]["parameters"])
    assert not target["boundary_valid"].any()
    np.testing.assert_array_equal(labels[[0, 7]], [0, 1])


def test_boundary_half_edges_and_same_label_shortest_paths():
    x, y = np.meshgrid(np.arange(10) * .04 + 1, np.arange(5) * .04)
    g = geometry(np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))))
    labels = (x.ravel() > 1.18).astype(np.int8)
    target = boundary_targets(g, labels, CONFIG["C1"]["parameters"])
    assert len(target["boundary_edges"]) > 0
    assert np.any(target["boundary_valid"] & (labels == 1) & (target["boundary_distance"] < 1))
    groups = g.groups(labels)
    adjacency = [[] for _ in groups]
    radius = np.minimum(.5, 3 * g.delta)
    distance = np.full(len(groups), np.inf)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            length = np.linalg.norm(g.xyz[i] - g.xyz[j])
            if j not in g.neighbors[i] or i not in g.neighbors[j] or length > min(radius[i], radius[j]):
                continue
            if groups[i] == groups[j]:
                adjacency[i].append((j, length))
                adjacency[j].append((i, length))
            else:
                distance[i] = min(distance[i], length / 2)
                distance[j] = min(distance[j], length / 2)
    queue = [(d, i) for i, d in enumerate(distance) if np.isfinite(d)]
    heapq.heapify(queue)
    while queue:
        d, i = heapq.heappop(queue)
        if d != distance[i]:
            continue
        for j, length in adjacency[i]:
            if d + length < distance[j]:
                distance[j] = d + length
                heapq.heappush(queue, (distance[j], j))
    np.testing.assert_allclose(target["_distance"], distance, rtol=0, atol=1e-12)
    labels[:] = 0
    labels[0] = 1
    isolated = boundary_targets(g, labels, CONFIG["C1"]["parameters"])
    assert not isolated["boundary_valid"].any()
    assert labels[0] == 1


def test_sampling_recomputes_scale_and_rejects_disappearing_anomaly_support():
    x, y = np.meshgrid(np.arange(12) * .03 + 1, np.arange(8) * .03)
    dense = geometry(np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))))
    rows = np.flatnonzero(np.tile(np.arange(12) % 2 == 0, 8))
    sparse = ScanGeometry(dense.xyzi[rows], dense.slots[rows], dense.parameters)
    pair = dict(dense_row=rows)
    labels = (x.ravel() > 1.16).astype(np.int8)
    target = sampling_targets(dense, sparse, labels, labels[rows], pair, CONFIG["C2"]["evidence_parameters"])
    assert np.any(target["sampling_consistency_valid"] & (labels[rows] == 1))
    assert np.median(sparse.delta) > np.median(dense.delta)
    labels[:] = 0
    labels[rows[0]] = 1
    target = sampling_targets(dense, sparse, labels, labels[rows], pair, CONFIG["C2"]["evidence_parameters"])
    assert not target["sampling_consistency_valid"].any()
    assert np.all(target["sampling_ignore_reason"] & 8)
    np.testing.assert_array_equal(sparse.xyzi, dense.xyzi[rows])


def test_hull_halfspaces_match_scipy_including_boundary_and_degenerate_support():
    rng = np.random.default_rng(731)
    for origin in ([0, 0], [2, 2], [1, 0], [1 + 2e-10, 0]):
        points = np.array(sorted(np.vstack((rng.uniform(-.9, .9, (60, 2)), [[-1, -1], [-1, 1], [1, -1], [1, 1]])).tolist()))
        shifted = points - origin
        expected = np.all(ConvexHull(shifted).equations[:, -1] <= 1e-10)
        assert _inside_hull(shifted, 1e-10) == expected
    assert not _inside_hull(np.array([[0., 0.], [0., 1.], [0., 2.]]), 1e-10)


def test_surface_uses_preinsertion_plane_and_each_views_visible_support():
    x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    xyzi = np.column_stack((x.ravel(), y.ravel(), -.8 + .02 * x.ravel(), np.full(x.size, .3))).astype(np.float32)
    slots = np.arange(len(xyzi), dtype=np.int32)
    original = SimpleNamespace(xyzi=xyzi, real_slots=slots, slot_count=len(slots),
                               labels=SimpleNamespace(semantic=np.full(len(slots), 40), semantic_target=np.zeros(len(slots))))
    post = xyzi.copy()
    center = len(post) // 2
    post[center, 2] += .1
    changed = slots == center
    source = SimpleNamespace(xyzi=post, zero_slot_mask=np.zeros(len(slots), bool))
    sample = SimpleNamespace(source=source, inserted_mask=changed, occluded_original_mask=changed)
    dense = ScanGeometry(post, slots, CONFIG["common"]["sampling_scale"])
    # Keep the anomaly but only 18 ground positions: dense validity cannot be copied.
    rows = np.r_[np.arange(18), center]
    sparse = ScanGeometry(post[rows], slots[rows], dense.parameters)
    pair = dict(dense_row=rows, ray_keep=np.isin(slots, rows))
    labels = changed.astype(np.int8)
    result = surface_targets(original, sample, [dense, sparse], [labels, labels[rows]], [pair], CONFIG["C3"]["parameters"])
    assert result[0]["surface_valid"][center]
    np.testing.assert_allclose(result[0]["surface_offset_z"][center], -.1, atol=1e-6)
    assert not result[1]["surface_valid"][-1]
    assert result[1]["surface_ignore_reason"][-1] == 64
    assert result[1]["surface_offset_z"][-1] == 0
    probe = surface_probe(post[center, :3].astype(float), original, sample, slots, CONFIG["C3"]["parameters"])
    assert probe["valid"] and probe["seen_positions"] == len(slots) - 1
    np.testing.assert_allclose(probe["offset_z_m"], result[0]["surface_offset_z"][center], atol=1e-6)


def test_surface_rejects_one_sided_support_even_with_twenty_positions():
    x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    ground = np.array(sorted(np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.))).tolist()))
    visible = np.column_stack((np.ones(len(ground), bool), ground[:, 0] > 0))
    p = np.array([20, .05, 3, 1.4826, .05, np.tan(np.deg2rad(20)), .01, 1e-10])
    result = _surface_chunk(np.array([[0., 0., -.9]]), ground, visible, np.array([0, len(ground)]),
                            np.arange(len(ground)), np.ones((1, 2), bool), p)
    np.testing.assert_array_equal(result[1], [[0, 128]])
    assert result[3][0, 1] >= 20
