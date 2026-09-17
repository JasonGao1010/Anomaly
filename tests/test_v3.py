"""Checks of V3's scientific identities, risk, relation equation, and metric scope."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.data import detection_targets
from src.evaluate import (
    Prediction,
    WORKPOINTS,
    EXPOSURE_GROUPS,
    coverage_status,
    VAL19,
    compare_reports,
    count_bin,
    distance_bin,
    evaluation_kind,
    instance_rows,
    instance_predictions,
    operating_threshold,
    rank_metrics,
    prediction_inputs,
    low_support,
    stratified,
    summarize,
)
from src.model import (
    METHOD,
    RelationDecoder,
    V3,
    lr_factor,
    optimizer,
    paired_loss,
    prepare_scan,
    nearest_returns,
    normal_risk,
    tail_weights,
)
from src.scene import PointLabels, make_source_frame
from src.train import request_at, record_conditions, validate_continuation_recipe
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


def test_shared_budget_extension_preserves_scientific_settings_and_past_rates():
    fixed = dict(
        steps=128,
        seed=20260917,
        route="P",
        balance="instance",
        backbone_lr=1e-5,
        new_lr=1e-4,
        halve_at=[],
    )
    old = dict(
        balance="instance",
        tail_weight=0.5,
        backbone_lr=1e-5,
        new_lr=1e-4,
        halve_at=[],
        fixed_recipe=fixed,
    )
    new = deepcopy(old)
    new["fixed_recipe"]["steps"] = 256
    validate_continuation_recipe(old, new, 128, 256)
    assert old["fixed_recipe"]["steps"] == 128
    for key, value in (("tail_weight", 0.25), ("backbone_lr", 2e-5)):
        changed = deepcopy(new)
        changed[key] = value
        with pytest.raises(ValueError, match="risk or base rates"):
            validate_continuation_recipe(old, changed, 128, 256)
    changed = deepcopy(new)
    changed["fixed_recipe"]["seed"] += 1
    with pytest.raises(ValueError, match="scientific settings"):
        validate_continuation_recipe(old, changed, 128, 256)
    changed["fixed_recipe"] = dict(fixed, steps=64)
    with pytest.raises(ValueError, match="reduce"):
        validate_continuation_recipe(old, changed, 32, 64)
    new["fixed_recipe"].update(steps=512, halve_at=[257])
    new["halve_at"] = [257]
    validate_continuation_recipe(old, new, 128, 512)
    with pytest.raises(ValueError, match="past learning rates"):
        validate_continuation_recipe(old, new, 300, 512)


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


def test_parallel_preparation_preserves_order_geometry_and_bounded_lookahead():
    scans = [
        source([[3 + i, 0, 0, 1], [4 + i, 0, 0, 2]], [40, 40], frame=i)
        for i in range(6)
    ]
    consumed = []

    def inputs():
        for scan in scans:
            consumed.append(scan.frame_id)
            yield scan

    for i, (scan, prepared, support) in enumerate(prediction_inputs(inputs(), 2)):
        assert scan is scans[i]
        assert len(consumed) <= i + 3
        reference = prepare_scan(scan.xyzi[scan.observation_slots])
        for key in vars(reference):
            assert np.array_equal(getattr(reference, key), getattr(prepared, key))
        assert np.array_equal(support, low_support(scan))
    assert consumed == list(range(6))


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
    models = [V3(mode) for mode in ("D0", "D1", "D2", "D3")]
    reference = models[-1].state_dict()
    for model in models:
        for name, value in model.state_dict().items():
            if name in reference:
                assert torch.equal(value, reference[name]), name
        assert not any("condition" in name for name, _ in model.named_parameters())
        assert (
            len([m for m in model.modules() if isinstance(m, PointROPEAttention)]) == 8
        )
    assert models[-1].decoder.phi[-1].weight.count_nonzero() > 0
    assert models[-1].decoder.proj.weight.count_nonzero() > 0
    assert len({sum(p.numel() for p in m.parameters()) for m in models[1:]}) == 1
    assert not hasattr(models[0].decoder, "qkv")
    groups = optimizer(models[-1]).param_groups
    for group in groups:
        for name, parameter in zip(group["names"], group["params"]):
            assert group["weight_decay"] == (0.01 if parameter.ndim > 1 else 0)
            if name.startswith(("decoder.", "fusion.", "head.")) or "embedding" in name:
                assert group["peak_lr"] == 1e-4
    assert [lr_factor(n) for n in (1, 16, 128, 256)] == [0.1, 1.0, 1.0, 1.0]
    assert lr_factor(300, (257,)) == 0.5
    with pytest.raises(ValueError):
        lr_factor(1, (128,))


def test_backbone_attention_restores_unconditioned_rope_equation():
    torch.manual_seed(3)
    block = PointROPEAttention(18, 1, 8, 100.0)
    features = torch.randn(5, 18, requires_grad=True)
    grid = torch.tensor([[1, 0, 0], [2, 1, 0], [4, 1, 1], [2, 4, 5], [0, 2, 3]])
    point = Point(
        feat=features,
        coord=torch.randn(5, 3) * 20,
        grid_coord=grid,
        offset=torch.tensor([5]),
        serialized_order=torch.arange(5)[None],
        serialized_inverse=torch.arange(5)[None],
    )
    actual = block(point).feat
    q, k, v = block.qkv(features).chunk(3, dim=-1)
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


def test_neighbors_use_distinct_returns_self_and_fixed_identity_ties():
    xyz = np.r_[np.zeros((40, 3)), [[100, 0, 0], [-100, 0, 0], [0, 100, 0]]]
    index = nearest_returns(xyz)
    assert index.shape == (43, 32)
    for i in range(len(xyz)):
        candidates = [j for j in range(len(xyz)) if j != i]
        candidates.sort(key=lambda j: (float(np.square(xyz[j] - xyz[i]).sum()), j))
        assert index[i].tolist() == [i] + candidates[:31]
    for n in (1, 2, 7, 32):
        small = nearest_returns(xyz[:n])
        assert small.shape == (n, n)
        assert all(len(set(row)) == n for row in small)
    assert index[-1, 1] == 0  # No radius cap: 100-meter neighbors remain available.


@pytest.mark.parametrize("mode", ["D0", "D1", "D2", "D3"])
def test_relation_decoder_equations_and_chunked_gradients(mode):
    torch.manual_seed(42)
    decoder = RelationDecoder(mode, chunk_size=3).double()
    reference = deepcopy(decoder)
    features = torch.randn(7, 128, dtype=torch.float64, requires_grad=True)
    other = features.detach().clone().requires_grad_(True)
    xyz = torch.randn(7, 3, dtype=torch.float64) * 40
    index = torch.tensor(nearest_returns(xyz.numpy()).astype(np.int64))
    actual = decoder(features, xyz, index)
    normalized = reference.norm(other)
    if mode == "D0":
        expected = other + reference.mlp(normalized)
    else:
        q, k, v = reference.qkv(normalized).reshape(7, 3, 4, 32).unbind(1)
        messages = []
        for i in range(7):
            condition = (
                xyz[i] / 50 if mode == "D3" else torch.zeros(3, dtype=torch.float64)
            )
            base = reference.phi(torch.cat((condition, torch.zeros_like(condition))))
            geometry = torch.stack(
                [
                    reference.phi(torch.cat((condition, xyz[j] - xyz[i]))) - base
                    for j in index[i]
                ]
            )
            assert geometry[0].count_nonzero() == 0
            heads = []
            for h in range(4):
                e = geometry[:, h * 32 : (h + 1) * 32]
                weight = ((k[index[i], h] + e) @ q[i, h] / np.sqrt(32)).softmax(0)
                heads.append(weight @ (v[index[i], h] + e))
            message = torch.cat(heads)
            if mode == "D2":
                zero = torch.zeros(3, dtype=torch.float64)
                message = (
                    message
                    + reference.phi(torch.cat((xyz[i] / 50, zero)))
                    - reference.phi(torch.cat((zero, zero)))
                )
            messages.append(message)
        updated = other + reference.proj(torch.stack(messages))
        expected = updated + reference.ffn(reference.norm2(updated))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(features.grad, other.grad, atol=1e-12, rtol=1e-12)
    for left, right in zip(decoder.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-12)


def test_fractional_normal_tail_and_tied_boundary_share_gradient_mass():
    losses = torch.tensor([5.0, 4.0, 4.0, 1.0, 0.0], requires_grad=True)
    weight = tail_weights(losses, alpha=0.5)
    torch.testing.assert_close(weight, torch.tensor([0.4, 0.3, 0.3, 0.0, 0.0]))
    (losses * weight).sum().backward()
    torch.testing.assert_close(losses.grad, weight)
    torch.testing.assert_close(
        tail_weights(torch.tensor([9.0, 9.0, 1.0])), torch.tensor([0.5, 0.5, 0.0])
    )
    scores = torch.linspace(-3, 3, 1000, requires_grad=True)
    risk, mean, tail = normal_risk(scores)
    all_losses = torch.nn.functional.softplus(scores)
    torch.testing.assert_close(tail, all_losses[-10:].mean())
    torch.testing.assert_close(risk, (all_losses.mean() + all_losses[-10:].mean()) / 2)
    risk.backward()
    expected = scores.detach().sigmoid() * 0.5 / 1000
    expected[-10:] += scores.detach()[-10:].sigmoid() * 0.5 / 10
    torch.testing.assert_close(scores.grad, expected)
    torch.testing.assert_close(normal_risk(scores, 0)[0], mean)
    empty = torch.empty(0, requires_grad=True)
    assert all(float(x.detach()) == 0 for x in normal_risk(empty))


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
    assert result["R_at_1pct_FPR"] == result["R_at_0.1pct_FPR"] == 0.25
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
    diagnostic = instance_predictions([p], {"0.1pct": 2.0, "1pct": -1.0})
    assert [(r["sequence"], r["frame"], r["instance"]) for r in diagnostic] == [
        (125, 1, 1), (125, 1, 2)
    ]
    assert [r["detected"] for r in diagnostic] == [
        {"0.1pct": 1, "1pct": 1}, {"0.1pct": 1, "1pct": 3}
    ]
    assert sum(r["detected"]["0.1pct"] for r in diagnostic) == result["detected_points"]
    group = result["anomaly_groups"]["target_union"]
    assert group["instance_observations"] == 2
    assert group["points"] == 4 and group["point_recall"] == 0.5
    assert group["instance_mean_point_recall"] == pytest.approx(2 / 3)
    assert group["distinct_objects"] is None
    assert result["normal_groups"]["all"]["false_positives_median"] == 1
    assert result["segmentation"] == dict(
        TP=2, FP=1, FN=2, TN=0, Precision=2 / 3, Recall=0.5, IoU=0.4
    )
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


def test_coverage_requires_four_conditions_and_distinct_support_at_rounded_node():
    exposure = {}
    plan = dict(groups={name: dict(node=256) for name in EXPOSURE_GROUPS})
    row = dict(
        instance=60001,
        observed_returns=4,
        height_m=0.30,
        distance_m=50.0,
        disappeared=4,
        object_id="known",
    )
    for i in range(20):
        record_conditions(exposure, i % 5, f"world{i}", [row])
    assert not coverage_status(exposure, plan, 128)["eligible"]
    assert coverage_status(exposure, plan, 256)["eligible"]
    record_conditions(exposure, 0, "world0", [row])
    assert (
        coverage_status(exposure, plan, 256)["groups"]["few_returns"]["requests"] == 20
    )
    unknown = dict(row, height_m=None, disappeared=None)
    record_conditions(exposure, 99, "unknown", [unknown])
    assert (
        exposure["unknown"]["low_height"]
        == exposure["unknown"]["weak_disappearance"]
        == 1
    )
    assert (
        coverage_status(exposure, plan, 256)["groups"]["low_height"]["requests"] == 20
    )


def test_candidate_comparison_preserves_tradeoffs_and_insufficient_exposure(tmp_path):
    paths = []
    exposure = {}
    observation = dict(
        instance=1,
        observed_returns=1,
        height_m=0.2,
        distance_m=40.0,
        disappeared=0,
        object_id="known",
    )
    for i in range(20):
        record_conditions(exposure, i % 5, str(i), [observation])
    plan = dict(groups={name: dict(node=128) for name in EXPOSURE_GROUPS})
    for route, ap, recall in (("P", 0.8, 0.6), ("T", 0.7, 0.8)):
        group = dict(instance_observations=2, instance_mean_point_recall=recall)
        groups = {
            point: dict(
                anomaly_groups={"returns/1-4": group},
                normal_groups={"all": dict(points=980, fpr=0.01)},
            )
            for point in WORKPOINTS
        }
        report = dict(
            method=METHOD,
            kind="full",
            exposure=exposure,
            coverage_plan=plan,
            official_population=[[125, 1, 1000, 20]],
            candidate=dict(
                route=route, mechanism="D3", tail_weight=0.5, seed=1, update=128
            ),
            official_ranking=dict(
                points=1000,
                anomaly_points=20,
                AP=ap,
                FPR95=0.1,
                R_at_1pct_FPR=0.5,
                **{"R_at_0.1pct_FPR": 0.4},
            ),
            official=groups,
            normal201={
                point: dict(normal_groups={"all": dict(points=10000, fpr=0.02)})
                for point in WORKPOINTS
            },
            sequences={str(seq): groups for seq in VAL19},
        )
        path = tmp_path / f"{route}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    compared = compare_reports(paths)
    assert all(not row["dominated_by"] for row in compared["candidates"])
    assert (
        compared["candidates"][0]["conditional_sequence_intervals"]["1pct"][
            "returns/1-4"
        ]["sequences"]
        == 19
    )
    assert not compared["seed_summary"]["P/D3/lambda=0.5/128"]["at_least_three_seeds"]
    report["official_ranking"]["AP"] = 0.9
    report["coverage_plan"]["groups"]["few_returns"]["node"] = 256
    paths[1].write_text(json.dumps(report))
    compared = compare_reports(paths)
    assert compared["candidates"][0]["observed_dominated_by"] == [str(paths[1])]
    assert not compared["candidates"][0]["dominated_by"]
    assert not compared["candidates"][1]["selection_eligible"]
    report["official_population"][0][1] = 2
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="populations differ"):
        compare_reports(paths)
    report["kind"] = "panel"
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="complete real"):
        compare_reports(paths)
