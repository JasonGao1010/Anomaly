from collections import Counter

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.model import joint_voxelize
from src.profile import (
    Ledger,
    describe,
    frame_geometry,
    ground_relation,
    stages,
    window_geometry,
)
from src.protocol import FrameSpan, SequenceSpec
from src.scene import PointLabels, assemble_window, make_source_frame


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


def test_profile_voxels_keep_all_members_and_history():
    sources = []
    ledger = Ledger()
    frame_rows = []
    for frame in range(5):
        points = np.array(
            [
                [1.011, 0.01, 0.01, 0.1],
                [1.012, 0.01, 0.01, 0.8],
                [1.013, 0.01, 0.01, 0.3],
                [2, 2, 2, 0.2],
                [0, 0, 0, 0],
            ],
            np.float32,
        )
        if frame == 4:
            points[0, 0] = 2.4
        semantic = np.array([40, 2 if frame % 2 == 0 else 40, 0, 40, 2], np.uint16)
        instance = np.where(semantic == 2, 1, 0).astype(np.uint16)
        packed = semantic.astype(np.uint32) + (instance.astype(np.uint32) << 16)
        labels = PointLabels(packed, semantic, instance)
        source = make_source_frame(
            frame, points, np.eye(4), labels, partition="train", sequence_id=206
        )
        sources.append(source)
        record, _, _ = frame_geometry(source, ledger, 206, frame)
        frame_rows.append(record)
        assert record["zero_slots"] == 1
        assert record["anomaly"] == int(frame % 2 == 0)
    spec = SequenceSpec("train", 206, "fixture", True, FrameSpan(0, 5))
    window = assemble_window(spec, 0, (0, 1, 2, 3, 4), sources)
    inputs = joint_voxelize(window)
    result, detail = window_geometry(window, inputs, frame_rows, ledger, 206, 4)
    assert result["visibility_pattern"] == "10101"
    assert result["history_visible_scans"] == 2
    assert result["current_anomaly_voxels"] == 1
    assert result["history_new_normal_voxels"] == 1
    assert result["history_new_normal_points"] == 1
    members = Counter()
    target = int(
        inputs.point_to_voxel[
            np.flatnonzero(window.current_mask & (window.labels.semantic == 2))[0]
        ]
    )
    for voxel, label in zip(
        inputs.point_to_voxel.tolist(), window.labels.semantic.tolist(), strict=True
    ):
        if voxel == target:
            members[label] += 1
    denominator = sum(members.values())
    assert detail["voxel_normal_fraction"][0] == members[40] / denominator
    assert detail["voxel_ignore_fraction"][0] == members[0] / denominator
    assert detail["voxel_anomaly_fraction"][0] == members[2] / denominator
    assert detail["voxel_mix"][0] == 3
    assert detail["voxel_hits"][0] == 31


def test_ground_support_and_observation_boundary_censoring():
    x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    ground = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    tree = cKDTree(ground[:, :2])
    obj = np.array([[0, 0, 1], [0.1, 0.1, 1.2]])
    value = ground_relation(obj, ground, tree)
    assert value["ground_status"] == "local_plane_proxy"
    assert value["ground_height_median"] == pytest.approx(1.1)
    assert ground_relation(obj + 10, ground, tree)["ground_height_median"] is None
    frames = [dict(anomaly=x) for x in (1, 1, 0, 1, 0, 0)]
    runs = stages(frames, 125)
    assert [r["kind"] for r in runs] == ["visible", "gap", "visible", "tail"]
    assert [r["length"] for r in runs] == [2, 1, 1, 2]
    assert runs[0]["left_censored"] and runs[-1]["right_censored"]
    assert frames[0]["first_in_visible_run"]


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


def test_synthetic_context_and_worlds_are_not_joined():
    from src.profile_report import joint_tables

    frames, windows = [], []
    for world, counts in (("000_00", [1, 1, 0, 0, 0]), ("000_01", [0, 0, 0, 0, 6])):
        rows = [
            dict(
                sequence=world,
                frame=i,
                anomaly=n,
                anomaly_in_range=n,
                anomaly_in_range_distance_median=12 if n else None,
                neighbor_road_fraction=None,
            )
            for i, n in enumerate(counts)
        ]
        episodes = stages(rows, world)
        assert episodes[0]["left_censored"] and episodes[-1]["right_censored"]
        frames.extend(rows)
        windows.append(
            dict(
                sequence=world,
                frame=4,
                scope="complete_windows",
                history_visible_scans=sum(n > 0 for n in counts[:4]),
                translation_m=1,
                normal_mix_fraction=0,
            )
        )
    result = joint_tables(frames, windows, [])
    assert sum(r["frames"] for r in result["stage_count"].values()) == 10
    assert sum(r["frames"] for r in result["count_distance_history"].values()) == 2
    assert result["count_distance_history"]["0|unseen|2"]["sequences"] == ["000_00"]
    assert result["count_distance_history"]["2|1|0"]["sequences"] == ["000_01"]
    assert result["official_count_distance"]["0|1"]["anomaly_in_range_points"] == 6
