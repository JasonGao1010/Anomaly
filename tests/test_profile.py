import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.profile import (
    Ledger,
    describe,
    frame_geometry,
    ground_relation,
)
from src.scene import PointLabels, make_source_frame


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
    ledger.add("F05", "shift", values, resolution=0.0001, bins=(-1, 0, 1))
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
