import numpy as np
import pytest
from scipy.spatial import cKDTree

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


def test_content_selection_precedes_supervision_and_keeps_every_eligible_far_frame():
    from src.coverage import select_checks

    rows = [dict(frame=i, count=n, in_range=n, range=r, auxiliary_valid=False)
            for i, (n, r) in enumerate([(4, 40), (5, 40), (7, 42), (100, 6), (9, 20)])]
    world = dict(split="train", world="world_000", height_m=.1, rows=rows)
    selected = select_checks([world])
    assert [s["frame"] for s in selected] == [1, 2, 3]
    assert selected[1]["reasons"] == ["few_representative", "all_far_eligible"]
    for row in rows:
        row["auxiliary_valid"] = True
    assert selected == select_checks([world])


def test_supplements_preserve_base_order_and_keep_content_failures_out():
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
    reports = [dict(index=2, status="qualified", purpose=dict(achieved=True), world_identity="c" * 64, reason=None),
               dict(index=0, status="rejected", purpose=dict(achieved=False), reason="content"),
               dict(index=1, status="rejected", purpose=dict(achieved=True), reason="physical")]
    chosen, selection = select_worlds(base, reports, "train")
    assert chosen[:2] == base
    assert len(chosen) == 3 and chosen[-1]["path"] == "train/supplement_002"
    assert selection["accepted_supplements"] == [2]
    assert selection["rejected_supplements"] == {"0": "content", "1": "physical"}


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
