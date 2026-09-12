import heapq
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree, ConvexHull

from src.geometry import (
    ScanGeometry, _inside_hull, _surface_chunk, boundary_targets,
    sampling_targets, surface_targets, surface_probe,
)
from src.profile import (
    Ledger,
    describe,
    frame_geometry,
    ground_relation,
    observed_geometry,
)
from src.scene import PointLabels, make_source_frame


def test_content_coverage_uses_official_count_and_distinct_worlds():
    from src.coverage import conditions, concentration

    world = {"height_m": .1}
    row = {"count": 9, "in_range": 4, "range": 40.}
    assert conditions(world, row)["far"]
    assert not conditions(world, row)["far_eligible"]
    row["in_range"] = 5
    assert conditions(world, row)["low_far_eligible"]
    counts = concentration({"one_object": 1000, "another_object": 10, "empty": 0})
    assert counts["worlds"] == 2 and counts["total"] == 1010
    assert counts["top_one_share"] == 1000 / 1010


def test_visibility_separates_replacement_loss_and_new_returns_after_ray_deduplication():
    from src.coverage import visibility_events
    from src.render import RayGrid

    angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    grid = RayGrid(np.c_[np.cos(angles), np.sin(angles), np.zeros(8)],
                   np.array([0.]), angles, beam_count=1)
    mapping = np.array([0, 1, 2, 3, 0, 1, 2, 3], np.int32)
    result = visibility_events([0, 1, 4, 5], [1, 2, 5, 6], mapping, grid)
    assert result["anomaly_rays"] == 2 and result["changed_native_rays"] == 2
    assert result["new_hit_rays"] == result["replaced_native_rays"] == result["lost_native_rays"] == 1
    assert result["changed_overlap_fraction"] == 2 / 3
    missing = visibility_events([], [], mapping, grid)
    assert missing["changed_overlap_fraction"] is None and missing["changed_components"] == 0


def test_content_selection_precedes_geometry_and_keeps_every_eligible_far_frame():
    from src.coverage import select_checks

    rows = [dict(frame=i, count=n, in_range=n, range=r, geometry_valid=False)
            for i, (n, r) in enumerate([(4, 40), (5, 40), (7, 42), (100, 6), (9, 20)])]
    world = dict(split="train", world="world_000", height_m=.1, rows=rows)
    selected = select_checks([world])
    assert [s["frame"] for s in selected] == [1, 2, 3]
    assert selected[1]["reasons"] == ["few_representative", "all_far_eligible"]
    for row in rows:
        row["geometry_valid"] = True
    assert selected == select_checks([world])


def test_candidate_collection_preserves_all_legal_worlds_before_balancing():
    from src.coverage import select_checks
    from src.generate import select_worlds

    rows = [dict(frame=i, count=n, in_range=n, range=r)
            for i, (n, r) in enumerate([(3, 40), (5, 36), (10, 40), (19, 49),
                                       (20, 36), (40, 40), (60, 49), (100, 6)])]
    world = dict(split="train", world="supplement_000", height_m=.1, rows=rows, content_check_frames=[5])
    selected = select_checks([world], far_limit=2)
    assert [r["frame"] for r in selected] == list(range(8))
    assert selected[0]["reasons"] == ["far_below_official_threshold"]
    assert selected[5]["reasons"] == ["declared_content_witness"]
    base = [dict(path="../candidates/train/candidate_009", world_identity="a" * 64),
            dict(path="../candidates/train/candidate_000", world_identity="b" * 64)]
    reports = [dict(index=2, status="qualified", purpose=dict(achieved=False), count=0,
                    world_identity="c" * 64, reason=None, family_id="train/2", paired=False, variant=0),
               dict(index=0, status="rejected", purpose=dict(achieved=False), reason="physical"),
               dict(index=1, status="rejected", purpose=dict(achieved=True), reason="physical")]
    chosen, selection = select_worlds(base, reports, "train")
    assert chosen[:2] == base
    assert len(chosen) == 3 and chosen[-1]["path"] == "train/world_002"
    assert chosen[-1]["cohort"] == "candidate"
    assert selection["accepted"] == [2]
    assert selection["rejected"] == {"0": "physical", "1": "physical"}


def test_balanced_world_assignment_enforces_joint_quotas_without_reusing_worlds():
    from collections import Counter
    from src.coverage import balanced_assignment

    candidates = [dict(identity=str(i), parent=str(i), regions=[str(i % 4), str((i+1) % 4)],
        eligible_cells={"first": 5, "second": 7}, states=["near/2/0", "far/4/0"] if i == 0 else ["near/2/0"])
        for i in range(8)]
    chosen, _ = balanced_assignment(candidates, {"first": 4, "second": 4}, 3)
    assert len({identity for identity, _ in chosen}) == 8
    assert Counter(cell for _, cell in chosen) == {"first": 4, "second": 4}
    for cell in ("first", "second"):
        selected = [candidates[int(identity)] for identity, name in chosen if name == cell]
        assert all(len({w["regions"][grid] for w in selected}) >= 3 for grid in (0, 1))
    impossible, _ = balanced_assignment(candidates[:-1], {"first": 4, "second": 4}, 3)
    assert impossible is None


def test_expansion_geometry_parents_stay_within_source_and_budget(tmp_path):
    from src.generate import expansion_schedule

    config = json.loads(Path("protocol/data.json").read_text())
    config["research_coverage"]["output"] = str(tmp_path)
    (tmp_path / "inventory.json").write_text(json.dumps(dict(worlds=[dict(split=s, regions=[f"{i}/0"],
        anchor_world_m=[10*i, 0, 0]) for s in ("train", "validation") for i in range(3)])))
    schedule = expansion_schedule(config)
    assert len({r["seed"] for r in schedule}) == len(schedule)
    assert len(schedule) == 78
    for split, count in (("train", 78), ("validation", 0)):
        groups = [r for r in schedule if r["split"] == split]
        indices = [i for r in groups for i in r["members"]]
        assert len(indices) == len(set(indices)) == count
        if not count:
            continue
        assert min(indices) >= config["proposals"]["index_start"]
        assert {r["profile"]["shape"] for r in groups} == ({"single", "cross", "bridge"} if split == "train" else {"single"})
        assert len({tuple(r["profile"]["target_region"]) for r in groups}) == 3
        from collections import Counter
        assert Counter(r["profile"]["combination"] for r in groups) == config["proposals"]["cell_candidates"][split]


def test_balanced_selection_preserves_recurring_core_evidence_and_concentration():
    from src.coverage import balanced_assignment

    candidates = [dict(identity=str(i), parent=str(i), regions=[str(i % 4), str((i+1) % 4)],
        eligible_cells={"first": 5, "second": 5}, states=["far/4/0"] if i == 0 else []) for i in range(8)]
    content = {"far_at_least_20": [dict(world_identity=str(i), frame=f,
                in_range_rays=1000 if i == 0 else 30) for i in range(8) for f in (0, 1)]}
    config = dict(core_cells=list(content), minimum_frames_per_parent=2,
                  maximum_parent_return_share=.35, maximum_region_return_share=.5)
    limits = dict(geometry_parents=3, support_regions=3, source_frames=2,
                  world_frames=8, anomaly_returns=240)
    chosen, _ = balanced_assignment(candidates, {"first": 2, "second": 2}, 2,
                                    (content, config, limits))
    assert chosen is not None and len(chosen) == 4
    assert "0" not in {identity for identity, _ in chosen}
    # Four worlds force the dominant contributor into the selection, which must fail.
    impossible, _ = balanced_assignment(candidates[:4], {"first": 2, "second": 2}, 2,
                                        (content, config, limits))
    assert impossible is None


def test_sparse_census_expands_only_worlds_triggered_by_fixed_observations():
    from src.coverage import research_geometry_selection

    config = json.loads(Path("protocol/data.json").read_text())["research_coverage"]
    config["geometry"]["sparse_world_census"] = True
    relations = {k: False for k in ("compact", "elongated", "sheet", "multi_branch", "multiple_contact")}
    record = dict(worlds=[dict(split="train", world=w, identity=w, height_m=.1, relations=relations)
                         for w in ("first", "second")])
    rows = [dict(split="train", world=w, world_identity=w, source_identity=str(f), frame=f,
                 in_range_rays=5, range=20., changed_native_rays=8)
            for w in ("first", "second") for f in range(10)]
    observations = [dict(world_identity=w, source_identity=str(f), anomaly_in_range=5,
                         native_context=dict(sparse=dict(positions=5)))
                    for w, f in (("first", 4), ("second", 3))]
    selected = research_geometry_selection(record, rows, config, observations)
    assert [r["frame"] for r in selected if r["world"] == "second"] == [0, 4, 9]
    assert {r["frame"] for r in selected if r["world"] == "first"} == set(range(10))
    config["geometry"]["sparse_world_census"] = False
    assert len(research_geometry_selection(record, rows, config, observations)) == 6


def test_sparse_witness_can_use_another_source_view_for_physical_support():
    from collections import Counter
    from src.data import source_identity
    from src.generate import sample_support

    def scan(frame, xyz, pose):
        sem = np.full(len(xyz), 40, np.uint16)
        labels = PointLabels(sem.astype(np.uint32), sem, np.zeros(len(xyz), np.uint16), np.full(len(xyz), 8, np.uint8))
        return make_source_frame(frame, np.c_[xyz, np.full(len(xyz), .2)].astype(np.float32), pose,
                                 labels, partition="train", sequence_id=206)
    witness = scan(0, np.c_[np.linspace(39.8, 40.2, 5), np.full(5, 5.), np.full(5, -1.5)], np.eye(4))
    x, y = np.meshgrid(np.linspace(4.4, 7.6, 17), np.linspace(3.4, 6.6, 17))
    pose = np.eye(4); pose[0, 3] = 34
    support = scan(1, np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.5)], pose)
    class Sequence(list):
        frame_ids = (0, 1)
        spec = SimpleNamespace(sequence_id=206)
        def lidar_pose(self, frame):
            return self[frame].lidar_pose
    sequence = Sequence([witness, support])
    config = json.loads(Path("protocol/data.json").read_text())["placement"]
    config.update(frame_interval=[0, 2], target_region=[4, 0], region_grid_m=10,
                  minimum_reference_positions=5, reference_cluster_radius_m=2, reference_clearance_m=[.05, 1.5])
    refs = [dict(frame=0, slots=list(range(5)), anchor_slot=2, kind="sparse", source_identity=source_identity(witness))]
    pool, record = next(sample_support(sequence, np.random.default_rng(8), config, .15, "any", Counter(), refs))
    assert pool.frames.tolist() == [1] and record["normal_reference"]["frame"] == 0
    assert record["normal_reference"]["source_identity"] == source_identity(witness)
    assert record["plane_support"] >= 20
    np.testing.assert_allclose(pool.anchors_world_m[0, 2], -1.5)


def test_target_opportunity_ignores_signal_realizations():
    from src.generate import opportunity_passes
    config = json.loads(Path("protocol/data.json").read_text())["proposals"]
    observations = [dict(distance_band="far", foreground_surface_rays=30,
        potential_changed_native_rays=4, final_anomaly_rays=0) for _ in range(5)]
    assert opportunity_passes(observations, dict(opportunity="far_dense"), config)
    for row in observations:
        row["final_anomaly_rays"] = 10000
    assert opportunity_passes(observations, dict(opportunity="far_dense"), config)
    assert not opportunity_passes(observations[:4], dict(opportunity="far_dense"), config)


def test_profile_weights_and_quantile_bounds(tmp_path):
    ledger = Ledger()
    ledger.add("C01", "intensity", np.zeros(100), unit="point", bins=(-1, 1, 11))
    ledger.add("C01", "intensity", [10], unit="point", bins=(-1, 1, 11))
    ledger.save(tmp_path / "hist.npz")
    with np.load(tmp_path / "hist.npz") as saved:
        import json

        meta = json.loads(str(saved["catalog"]))[0]
        observation = describe(meta, saved["x0"], saved["c0"])
        frame = describe(meta, saved["x0"], saved["w0"])
        assert observation["quantiles"]["0.95"]["value"] == 0
        assert frame["quantiles"]["0.95"]["value"] == 10
        assert saved["w0"].sum() == 2
    ledger = Ledger()
    values = np.array([-0.00203, 0.000001, 0.003234, 0.00412, np.nan])
    ledger.add("E01", "ground_height", values, resolution=0.0001, bins=(-1, 0, 1))
    item = next(iter(ledger.series.values()))
    x = np.array(sorted(item["counts"]))
    w = np.array([item["counts"][v] for v in x])
    result = describe(item, x, w)
    assert result["missing_fraction"] == 0.2
    for q, interval in result["quantiles"].items():
        exact = np.quantile(
            values[np.isfinite(values)], float(q), method="inverted_cdf"
        )
        assert interval["lower"] <= exact <= interval["upper"]
        assert interval["upper"] - interval["lower"] <= 0.000100000001


def test_ground_support_and_missing_height():
    x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    ground = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    tree = cKDTree(ground[:, :2])
    obj = np.array([[0, 0, 1], [0.1, 0.1, 1.2]])
    value = ground_relation(obj, ground, tree)
    assert value["ground_status"] == "local_plane_proxy"
    assert value["ground_height_median"] == pytest.approx(1.1)
    assert ground_relation(obj + 10, ground, tree)["ground_height_median"] is None


def test_report_sequence_weights_and_variance_use_full_distributions(tmp_path):
    import json

    from src.profile_report import saved_distributions, summarize

    ledger = Ledger()
    ledger.add(
        "C04", "background_intensity_std", [0.0] * 100, bins=(-np.inf, 0, 1, 11, np.inf)
    )
    ledger.add(
        "C04", "background_intensity_std", [10.0], bins=(-np.inf, 0, 1, 11, np.inf)
    )
    ledger.save(tmp_path / "hist.npz")
    with np.load(tmp_path / "hist.npz") as saved:
        std, variance = list(saved_distributions(saved))
        assert variance[0]["total"] == 100
        assert variance[0]["frame_total"] == 100
        np.testing.assert_array_equal(variance[1], [0, 100])
        meta = json.loads(str(saved["catalog"]))[0]
        # One sequence has 100 zeros; the other has one ten. Average CDFs, not quantiles.
        data = summarize(
            meta,
            *std[1:],
            sequence_weights=np.array([1.0, 1.0]),
            sequence_mean=10,
            sequence_bins=np.array([0.0, 1.0, 1.0, 0.0]),
            sequence_count=2,
        )
        assert data["observation_equal"]["quantiles"]["0.95"]["value"] == 0
        assert data["sequence_equal"]["quantiles"]["0.5"]["value"] == 0
        assert data["sequence_equal"]["quantiles"]["0.75"]["value"] == 10


def test_empty_scan_and_unknown_instance_keep_missing_values():
    xyzi = np.zeros((2, 4), np.float32)
    semantic = np.array([2, 40], np.uint16)
    labels = PointLabels(semantic.astype(np.uint32), semantic, np.zeros(2, np.uint16))
    source = make_source_frame(
        0, xyzi, np.eye(4), labels, partition="fixture", sequence_id=1
    )
    record, instances, points = frame_geometry(source, Ledger())
    assert record["visible"] == 0 and record["normal_fraction"] is None
    assert not instances and len(points["slot"]) == 0
    xyzi[0, 0] = 10
    source = make_source_frame(
        0, xyzi, np.eye(4), labels, partition="fixture", sequence_id=1
    )
    record, instances, points = frame_geometry(source, Ledger())
    assert record["unknown_instance_points"] == 1 and record["instance_count"] is None
    assert not instances and np.isnan(points["same_instance_neighbor_distance"]).all()


def test_observed_plane_residual_uses_unqueried_full_scan_neighbors():
    angles = np.arange(32) * (2 * np.pi / 32)
    xyz = np.column_stack((10 + .5 * np.cos(angles), .5 * np.sin(angles), np.ones(32)))
    xyzi = np.column_stack((np.vstack(([10, 0, 1.2], xyz)), np.zeros(33)))
    slots = np.arange(33, dtype=np.int32) * 3 + 7
    result = observed_geometry(xyzi, slots, slots[:1], block_size=5)
    assert result["source_slot"].tolist() == [7]
    assert result["neighbor_count"].tolist() == [32]
    assert result["valid"].all() and result["condition_valid"].all() and result["normal_valid"].all()
    assert result["surface_residual"][0] == pytest.approx(.2)
    assert result["roughness"][0] == pytest.approx(0, abs=1e-7)
    assert result["variation"][0] == pytest.approx(0, abs=1e-7)
    assert result["planarity"][0] == pytest.approx(1)
    assert result["scale"][0] == pytest.approx(np.sqrt(.5**2 + .2**2))
    assert result["residual_scaled"][0] == pytest.approx(.2 / np.sqrt(.5**2 + .2**2))
    assert np.isnan(observed_geometry(xyzi[:1], slots[:1])["surface_residual"]).all()
    whole = observed_geometry(xyzi, slots)
    for key in result:
        np.testing.assert_array_equal(result[key], whole[key][:1])


def test_observed_geometry_keeps_slot_aliases_without_inflating_support():
    # Exact repeated coordinates are file-row aliases, not additional physical support.
    angles = np.arange(40) * (2 * np.pi / 40)
    xyz = np.column_stack((10 + np.cos(angles), np.sin(angles), np.ones(40)))
    xyzi = np.column_stack((xyz, np.zeros(40)))
    slots = np.arange(40, dtype=np.int32) + 100
    reference = observed_geometry(xyzi, slots, workers=1)
    assert reference["normal_change_valid"].all()
    np.testing.assert_allclose(reference["normal_change"], 0, atol=1e-7)
    duplicated = np.vstack((xyzi, xyzi[3]))
    duplicated[-1, 3] = 99  # Intensity does not enter these geometric descriptors.
    duplicate_slots = np.r_[slots, 500].astype(np.int32)
    order = np.arange(41)[::-1]
    result = observed_geometry(duplicated[order], duplicate_slots[order], duplicate_slots,
                               workers=2, block_size=7)
    for key in reference:
        np.testing.assert_array_equal(result[key][:-1], reference[key])
        if key != "source_slot":
            np.testing.assert_array_equal(result[key][-1:], reference[key][3:4])
    assert result["source_slot"][-1] == 500


def test_observed_nonplanar_shape_survives_unreliable_normal_and_sparse_scale_is_missing():
    corners = np.array(np.meshgrid(*[[-.5, .5]] * 3)).reshape(3, -1).T
    xyz = np.vstack(([10, 0, 1], corners + [10, 0, 1], [100, 0, 1]))
    xyzi = np.column_stack((xyz, np.zeros(len(xyz))))
    slots = np.arange(len(xyz), dtype=np.int32)
    result = observed_geometry(xyzi, slots, np.array([0, 9], np.int32))
    assert result["neighbor_count"].tolist() == [8, 0]
    assert result["valid"].tolist() == [True, False]
    assert result["normal_valid"].tolist() == [False, False]
    assert result["normal_change_valid"].tolist() == [False, False]
    assert result["variation"][0] == pytest.approx(1 / 3)
    assert result["roughness"][0] == pytest.approx(.5)
    assert np.isnan(result["surface_residual"]).all()
    assert np.isnan(result["scale"][1]) and not result["condition_valid"][1]
    with pytest.raises(ValueError, match="absent"):
        observed_geometry(xyzi, slots, np.array([10], np.int32))


GEOMETRY_CONFIG = json.loads(Path("protocol/data.json").read_text())["geometry"]


def geometry(xyz, slots=None):
    if slots is None:
        slots = np.arange(len(xyz), dtype=np.int32)
    return ScanGeometry(np.column_stack((xyz, np.full(len(xyz), .3))).astype(np.float32),
                        slots, GEOMETRY_CONFIG["sampling_scale"])



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
    target = boundary_targets(g, labels, GEOMETRY_CONFIG["boundary"])
    assert not target["boundary_valid"].any()
    np.testing.assert_array_equal(labels[[0, 7]], [0, 1])



def test_boundary_half_edges_and_same_label_shortest_paths():
    x, y = np.meshgrid(np.arange(10) * .04 + 1, np.arange(5) * .04)
    g = geometry(np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))))
    labels = (x.ravel() > 1.18).astype(np.int8)
    target = boundary_targets(g, labels, GEOMETRY_CONFIG["boundary"])
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
    isolated = boundary_targets(g, labels, GEOMETRY_CONFIG["boundary"])
    assert not isolated["boundary_valid"].any()
    assert labels[0] == 1



def test_sampling_recomputes_scale_and_rejects_disappearing_anomaly_support():
    x, y = np.meshgrid(np.arange(12) * .03 + 1, np.arange(8) * .03)
    dense = geometry(np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))))
    rows = np.flatnonzero(np.tile(np.arange(12) % 2 == 0, 8))
    sparse = ScanGeometry(dense.xyzi[rows], dense.slots[rows], dense.parameters)
    pair = dict(dense_row=rows)
    labels = (x.ravel() > 1.16).astype(np.int8)
    target = sampling_targets(dense, sparse, labels, labels[rows], pair, GEOMETRY_CONFIG["sampling"])
    assert np.any(target["sampling_consistency_valid"] & (labels[rows] == 1))
    assert np.median(sparse.delta) > np.median(dense.delta)
    labels[:] = 0
    labels[rows[0]] = 1
    target = sampling_targets(dense, sparse, labels, labels[rows], pair, GEOMETRY_CONFIG["sampling"])
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
    dense = ScanGeometry(post, slots, GEOMETRY_CONFIG["sampling_scale"])
    # Keep the anomaly but only 18 ground positions: dense validity cannot be copied.
    rows = np.r_[np.arange(18), center]
    sparse = ScanGeometry(post[rows], slots[rows], dense.parameters)
    pair = dict(dense_row=rows, ray_keep=np.isin(slots, rows))
    labels = changed.astype(np.int8)
    result = surface_targets(original, sample, [dense, sparse], [labels, labels[rows]], [pair], GEOMETRY_CONFIG["surface"])[20]
    assert result[0]["surface_valid"][center]
    np.testing.assert_allclose(result[0]["surface_offset_z"][center], -.1, atol=1e-6)
    assert not result[1]["surface_valid"][-1]
    assert result[1]["surface_ignore_reason"][-1] == 64
    assert result[1]["surface_offset_z"][-1] == 0
    probe = surface_probe(post[center, :3].astype(float), original, sample, slots, GEOMETRY_CONFIG["surface"])
    assert probe["valid"] and probe["seen_positions"] == len(slots) - 1
    np.testing.assert_allclose(probe["offset_z_m"], result[0]["surface_offset_z"][center], atol=1e-6)



def test_surface_rejects_one_sided_support_even_with_twenty_positions():
    x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    ground = np.array(sorted(np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.))).tolist()))
    visible = np.column_stack((np.ones(len(ground), bool), ground[:, 0] > 0))
    p = np.array([20, .05, 3, 1.4826, .05, np.tan(np.deg2rad(20)), .01, 1e-10])
    result = _surface_chunk(np.array([[0., 0., -.9]]), ground, visible, np.array([0, len(ground)]),
                            np.arange(len(ground)), np.ones((1, 2), bool), p, np.array([20]))
    np.testing.assert_array_equal(result[1][..., 0], [[0, 128]])
    assert result[3][0, 1] >= 20



def test_visible_minimum_is_independent_of_reference_and_checks_remaining_geometry():
    x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    ground = np.array(sorted(np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.))).tolist()))
    # Thirteen well-spread positions, thirteen one-sided positions, nine positions.
    spread = np.zeros(len(ground), bool)
    spread[np.r_[np.arange(0, 81, 8), [4, 76]]] = True
    side = np.zeros(len(ground), bool)
    side[np.flatnonzero(ground[:, 0] > 0)[-13:]] = True
    visible = np.column_stack((spread, side, np.arange(len(ground)) < 9))
    p = np.array([20, .05, 3, 1.4826, .05, np.tan(np.deg2rad(20)), .01, 1e-10])
    result = _surface_chunk(np.array([[0., 0., -.9]]), ground, visible, np.array([0, len(ground)]),
                            np.arange(len(ground)), np.ones((1, 3), bool), p, np.array([20, 10]))
    np.testing.assert_array_equal(result[1][0], [[64, 0], [64, 128], [64, 64]])
    np.testing.assert_allclose(result[0][0, 0, 1], -.1, atol=1e-12)
    insufficient = _surface_chunk(np.array([[0., 0., -.9]]), ground, visible, np.array([0, 19]),
                                  np.arange(19), np.ones((1, 3), bool), p, np.array([20, 10]))
    assert np.all(insufficient[1] == 2)
    # Symmetric outliers leave exactly eighteen plane inliers from twenty-two references.
    reference = np.vstack((ground[10:28], [[-.9, -.9, -.5], [-.9, .9, -1.5],
                                          [.9, -.9, -1.5], [.9, .9, -.5]]))
    reference = np.array(sorted(reference.tolist()))
    too_few_inliers = _surface_chunk(np.array([[0., 0., -.9]]), reference, np.ones((22, 1), bool),
                                     np.array([0, 22]), np.arange(22), np.ones((1, 1), bool), p, np.array([20, 10]))
    assert too_few_inliers[2][0] == 18 and np.all(too_few_inliers[1] == 8)
