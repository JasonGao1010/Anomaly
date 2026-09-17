"""Checks of V3's scientific identities, risk, query equation, and metric scope."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.data import detection_targets
from src.evaluate import (
    Prediction,
    VAL19,
    compare_reports,
    count_bin,
    distance_bin,
    evaluation_kind,
    instance_rows,
    operating_threshold,
    rank_metrics,
    stratified,
    summarize,
)
from src.model import V3, lr_factor, optimizer, paired_loss, prepare_scan
from src.scene import PointLabels, make_source_frame
from src.train import request_at
from vendor.litept.model import Block, Point, PointROPEAttention


def source(xyzi, raw, instance=None, sequence=206, frame=0):
    raw = np.array(raw, dtype=np.uint16)
    instance = (
        np.zeros_like(raw) if instance is None else np.array(instance, dtype=np.uint16)
    )
    packed = raw.astype(np.uint32) | (instance.astype(np.uint32) << 16)
    return make_source_frame(
        frame,
        np.array(xyzi, dtype=np.float32),
        np.eye(4),
        PointLabels(packed, raw, instance, None),
        partition="train",
        sequence_id=sequence,
    )


def test_targets_include_all_normal_labels_and_isolate_native_anomaly():
    scan = source(
        [[x, 0, 0, 4] for x in (2.49, 2.5, 50.0, 50.01, 4, 5, 6, 0)],
        [40, 1, 99, 40, 2, 0, 2, 0],
    )
    assert len(scan.observation_slots) == 7
    assert detection_targets(scan).tolist() == [-1, 0, 0, -1, -1, -1, -1]
    inserted = np.zeros(8, bool)
    inserted[6] = True
    assert detection_targets(scan, inserted=inserted).tolist() == [
        -1,
        0,
        0,
        -1,
        -1,
        -1,
        1,
    ]
    assert detection_targets(scan, real_anomalies=True).tolist() == [
        -1,
        0,
        0,
        -1,
        1,
        -1,
        1,
    ]


def test_known_201_aliases_preserve_denominator_and_distinct_intensities():
    xyzi = np.zeros((131072, 4), np.float32)
    xyzi[:3] = [[3.0, 0, 0, 0.1], [3.0, 0, 0, 0.2], [4.0, 0, 0, 0.1]]
    scan = source(
        np.tile(xyzi, (3, 1)),
        np.tile(np.r_[np.array([40] * 3), np.zeros(131069)], 3),
        sequence=201,
    )
    assert len(scan.observation_slots) == 3
    assert len(scan.real_slots) == 9
    assert np.array_equal(
        np.array([7.0, 8.0, 9.0])[scan.record_inverse], np.tile([7.0, 8.0, 9.0], 3)
    )
    assert (detection_targets(scan, records=True) == 0).sum() == 9
    anomaly = source(
        np.tile(xyzi, (3, 1)),
        np.tile(np.r_[np.array([2] * 3), np.zeros(131069)], 3),
        np.tile(np.r_[np.array([1] * 3), np.zeros(131069)], 3),
        sequence=201,
    )
    row = instance_rows(
        anomaly,
        detection_targets(anomaly, real_anomalies=True, records=True),
        records=True,
    )[0]
    assert row["points"] == 9 and row["observed_returns"] == 3
    assert row["returns"] == "1-4"


def test_voxels_keep_return_details_sensor_offsets_and_context_range():
    scan = prepare_scan(
        np.array(
            [[-0.001, 0, 0.01, 5.0], [-0.049, 0, 0.01, -2.0], [70, 0, 0, 9.0]],
            np.float32,
        )
    )
    assert len(scan.grid) == 2
    assert scan.inverse[0] == scan.inverse[1]
    assert np.allclose(scan.features[:2, 4], [0.024, -0.024])
    assert scan.features[:, 3].tolist() == [5.0, -2.0, 9.0]
    assert scan.features[2, 0] == 70
    assert np.allclose(scan.coord[scan.inverse[0]], [-0.025, 0, 0.01])


def test_instance_risk_and_zero_anomaly_request_mean():
    scores = torch.tensor([2.0, -1.0, -1.0, -1.0, 0.8], requires_grad=True)
    negative = torch.tensor([-0.5, 99.0], requires_grad=True)
    pt, nt = torch.tensor([1, 1, 1, 1, 0]), torch.tensor([0, -1])
    ids = torch.tensor([1, 2, 2, 2, 0])
    loss, parts = paired_loss(scores, negative, pt, nt, ids)
    expected_anomaly = (np.logaddexp(0, -2) + np.logaddexp(0, 1)) / 2
    expected = (
        0.5 * expected_anomaly
        + 0.25 * np.logaddexp(0, 0.8)
        + 0.25 * np.logaddexp(0, -0.5)
    )
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert scores.grad[1] == pytest.approx(float(scores.grad[2]))
    assert negative.grad[1] == 0
    zero, empty_parts = paired_loss(scores, negative, torch.full_like(pt, -1), nt, ids)
    assert float(empty_parts[0].detach()) == 0
    assert float(zero.detach()) == pytest.approx(0.25 * np.logaddexp(0, -0.5))
    assert float((loss + zero).detach() / 2) == pytest.approx(
        (expected + 0.25 * np.logaddexp(0, -0.5)) / 2
    )
    with pytest.raises(ValueError, match="reliable"):
        paired_loss(scores, negative, pt, nt, torch.zeros_like(ids))


def test_initialization_mechanisms_share_every_common_tensor_and_optimizer_roles():
    a, b, c = V3("A"), V3("B"), V3("C")
    assert all(
        torch.equal(value, b.state_dict()[name])
        for name, value in a.state_dict().items()
        if ".condition." not in name
    )
    assert all(
        torch.equal(value, c.state_dict()[name])
        for name, value in a.state_dict().items()
    )
    assert len([m for m in a.modules() if isinstance(m, PointROPEAttention)]) == 8
    assert (
        a.head[-1].weight.count_nonzero() > 0 and a.head[-1].bias.count_nonzero() == 0
    )
    for name, value in a.named_parameters():
        if ".condition.2.weight" in name:
            assert value.count_nonzero() == 0
    groups = optimizer(a).param_groups
    for group in groups:
        for name, parameter in zip(group["names"], group["params"]):
            assert group["weight_decay"] == (0.01 if parameter.ndim > 1 else 0)
            if ".condition." in name or "embedding" in name:
                assert group["peak_lr"] == 1e-4
    assert [lr_factor(n) for n in (1, 16, 128, 256)] == [0.1, 1.0, 1.0, 1.0]
    assert lr_factor(300, (257,)) == 0.5
    with pytest.raises(ValueError):
        lr_factor(1, (128,))


def test_conditional_query_matches_equation_before_rope_with_gradients():
    torch.manual_seed(3)
    block = PointROPEAttention(18, 1, 8, 100.0)
    torch.nn.init.normal_(block.condition[-1].weight, std=0.1)
    features = torch.randn(5, 18, requires_grad=True)
    coord = torch.randn(5, 3) * 20
    grid = torch.tensor([[1, 0, 0], [2, 1, 0], [4, 1, 1], [2, 4, 5], [0, 2, 3]])
    point = Point(
        feat=features,
        coord=coord,
        grid_coord=grid,
        offset=torch.tensor([5]),
        serialized_order=torch.arange(5)[None],
        serialized_inverse=torch.arange(5)[None],
    )
    actual = block(point).feat
    q, k, v = block.qkv(features).chunk(3, dim=-1)
    q = q + block.condition(coord / 50.0)
    # Independent axis matrices check the condition/rotation order and sqrt(18).
    matrices = []
    for position in grid:
        rotation = torch.zeros(18, 18)
        for axis in range(3):
            for j in range(3):
                phase = float(position[axis]) / 100.0 ** (2 * j / 6)
                a, b = axis * 6 + j, axis * 6 + j + 3
                rotation[a, a] = rotation[b, b] = np.cos(phase)
                rotation[a, b], rotation[b, a] = -np.sin(phase), np.sin(phase)
        matrices.append(rotation)
    rotation = torch.stack(matrices)
    rq = torch.einsum("nij,nj->ni", rotation, q)
    rk = torch.einsum("nij,nj->ni", rotation, k)
    expected = block.proj(torch.softmax(rq @ rk.T / np.sqrt(18), dim=-1) @ v)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual.square().sum().backward()
    assert features.grad.abs().sum() > 0
    assert block.condition[-1].weight.grad.abs().sum() > 0


def test_recomputed_attention_preserves_values_gradients_and_scan_boundaries():
    torch.manual_seed(9)
    block = Block(18, 1, patch_size=8, enable_conv=False, enable_attn=True)
    reference = deepcopy(block)
    reference.recompute = False
    features = torch.randn(20, 18, requires_grad=True)
    copied_features = features.detach().clone().requires_grad_(True)
    coord = torch.randn(20, 3)
    grid = torch.stack((torch.arange(20), torch.zeros(20), torch.ones(20)), 1).long()

    def point(values):
        result = Point(
            feat=values, coord=coord, grid_coord=grid, offset=torch.tensor([3, 20])
        )
        result.serialization(order=("z",))
        result.sparsify()
        return result

    actual, expected = (
        block(point(features)).feat,
        reference(point(copied_features)).feat,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(
        features.grad, copied_features.grad, atol=1e-6, rtol=1e-6
    )
    for (_, left), (_, right) in zip(
        block.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-6)
    # Changing the second scan must not affect any token in the first scan.
    changed = copied_features.detach().clone()
    changed[3:] += 9 * torch.randn_like(changed[3:])
    torch.testing.assert_close(
        reference(point(changed)).feat[:3], expected[:3], atol=1e-6, rtol=1e-6
    )


def test_threshold_ties_are_conservative_and_global_metrics_match_independent_reference():
    scores = np.array([3.0] * 2 + [1.0] * 98 + [4.0, 3.0, 1.0, -1.0], dtype=np.float32)
    labels = np.array([0] * 100 + [1] * 4)
    threshold = operating_threshold(scores, labels)
    assert threshold > 3.0
    assert np.sum(scores[labels == 0].astype(np.float64) >= threshold) == 0
    result = rank_metrics(scores, labels)
    assert result["R_at_1pct_FPR"] == 0.25
    rng = np.random.default_rng(7)
    scores, labels = rng.integers(-5, 8, 300), rng.integers(0, 2, 300)
    result = rank_metrics(scores, labels)
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    assert result["AP"] == pytest.approx(average_precision_score(labels, scores))
    assert result["AUROC"] == pytest.approx(roc_auc_score(labels, scores))
    assert result["FPR95"] == fpr[np.flatnonzero(tpr > 0.95)[0]]
    assert rank_metrics(scores, np.zeros(300))["AP"] is None


def test_groups_use_instances_and_target_union_never_double_counts():
    rows = [
        dict(
            instance=1,
            points=1,
            returns="1-4",
            distance="35-50",
            distance_m=40,
            height="<=0.30",
            height_m=0.2,
            object_id=None,
        ),
        dict(
            instance=2,
            points=3,
            returns="1-4",
            distance="2.5-10",
            distance_m=5,
            height=None,
            height_m=None,
            object_id=None,
        ),
    ]
    p = Prediction(
        125,
        1,
        np.array([2.0, 2.0, -1.0, -1.0, 2.0]),
        np.array([1, 1, 1, 1, 0]),
        np.array([1, 2, 2, 2, 0]),
        rows,
        {},
    )
    result = summarize([p], 0.0)
    group = result["anomaly_groups"]["target_union"]
    assert group["instance_observations"] == 2
    assert group["points"] == 4 and group["point_recall"] == 0.5
    assert group["instance_mean_point_recall"] == pytest.approx(2 / 3)
    assert group["distinct_objects"] is None
    assert result["normal_groups"]["all"]["false_positives_median"] == 1
    assert [distance_bin(x) for x in [2.5, 10, 20, 35, 50]] == [
        "2.5-10",
        "10-20",
        "20-35",
        "35-50",
        "35-50",
    ]
    assert [count_bin(x) for x in [1, 4, 5, 19, 20, 99, 100]] == [
        "1-4",
        "1-4",
        "5-19",
        "5-19",
        "20-99",
        "20-99",
        "100+",
    ]


def test_sampling_continuation_and_evaluation_scopes():
    short = [request_at(20260917, n, 8, 240, 449) for n in range(1, 129)]
    long = [request_at(20260917, n, 8, 240, 449) for n in range(1, 257)]
    assert all(np.array_equal(a, b) for a, b in zip(short, long))
    assert len(set(np.concatenate(long))) > 1900
    assert [
        evaluation_kind(n) for n in [0, 8, 16, 32, 64, 96, 128, 192, 256, 512, 1024]
    ] == [
        "micro",
        "micro",
        "panel",
        "panel",
        "panel",
        "panel",
        "full",
        "panel",
        "full",
        "full",
        "full",
    ]
    assert evaluation_kind(17) is None and evaluation_kind(17, final=True) == "full"
    assert stratified([0, 100, 101, 900], 3, np.random.default_rng(1))[-1] == 900


def test_candidate_comparison_preserves_tradeoffs_and_sequence_units(tmp_path):
    paths = []
    for route, ap, recall in (("P", 0.8, 0.6), ("T", 0.7, 0.8)):
        group = dict(instance_observations=2, instance_mean_point_recall=recall)
        report = dict(
            kind="full",
            candidate=dict(route=route, mechanism="A", seed=1, update=128),
            official_ranking=dict(
                points=1000, anomaly_points=20, AP=ap, FPR95=0.1, R_at_1pct_FPR=0.5
            ),
            official=dict(anomaly_groups={"returns/1-4": group}),
            normal201=dict(normal_groups={"all": dict(points=10000, fpr=0.02)}),
            sequences={
                str(seq): dict(anomaly_groups={"returns/1-4": group}) for seq in VAL19
            },
        )
        path = tmp_path / f"{route}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    compared = compare_reports(paths)
    assert all(not row["dominated_by"] for row in compared["candidates"])
    assert (
        compared["candidates"][0]["conditional_sequence_intervals"]["returns/1-4"][
            "sequences"
        ]
        == 19
    )
    assert not compared["seed_summary"]["P/A/128"]["at_least_three_seeds"]
    report["kind"] = "panel"
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="complete real"):
        compare_reports(paths)
