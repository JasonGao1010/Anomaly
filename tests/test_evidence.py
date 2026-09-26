"""Paired SERVE comparisons retain official populations and exact decisions."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.evaluate import comparison_conditions, comparison_metrics, official_population
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


def test_voxel_sort_preserves_exact_point_order_means_and_origin_grid():
    from src.model import voxelize, GRID_SIZE
    rng = np.random.default_rng(107)
    cells = rng.integers(-5, 6, (400, 3))
    cells[1::3] = cells[::3][:len(cells[1::3])]
    cells[2::3] = cells[::3][:len(cells[2::3])]
    points = np.c_[(cells + rng.uniform(.05, .95, cells.shape)) * GRID_SIZE,
                   np.resize([1e20, 1., -1e20], len(cells))].astype(np.float32)
    # Large cancelling values expose changes in within-cell reduction order.
    samples = (points, points[:1], np.tile(points[:1], (7, 1)),
               np.array([[np.nextafter(np.float32(.05), np.float32(0)), -1., 2., .3],
                         [.05, -1., 2., .4], [-.05, -1., 2., .5]], np.float32))
    for xyzi in samples:
        grid = np.floor(xyzi[:, :3].astype(np.float64) / GRID_SIZE).astype(np.int64)
        unique, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind="stable")
        pointer = np.r_[0, np.cumsum(counts)].astype(np.int64)
        mean = (np.add.reduceat(xyzi[order].astype(np.float64), pointer[:-1], axis=0)
                / counts[:, None]).astype(np.float32)
        expected = dict(xyzi=xyzi, grid=unique - (unique.min(axis=0) // 16) * 16,
                        voxel_xyzi=mean, inverse=inverse, order=order, pointer=pointer,
                        offset=((xyzi[:, :3].astype(np.float64) - (grid + .5) * GRID_SIZE)
                                / GRID_SIZE).astype(np.float32))
        actual = voxelize(xyzi)
        for key, value in expected.items():
            assert actual[key].numpy().dtype == value.dtype
            np.testing.assert_array_equal(actual[key].numpy(), value, err_msg=key)


def test_normal_semantics_keeps_unseen_training_classes_and_false_positives():
    from src.normal import normal_semantic_metrics
    matrix = torch.zeros(19, 19, dtype=torch.long)
    matrix[0, 0], matrix[1, 1], matrix[9, 0] = 128, 128, 128
    measured = normal_semantic_metrics(matrix)
    assert measured["iou"][0] == .5 and measured["iou"][1] == 1
    assert measured["iou"][9] == 0 and measured["mean_iou_gt"] == .5


def point_records(scores, confidence):
    records = np.zeros(len(scores), dtype=[("score", "f4"), ("raw_score", "f4"),
                                         ("confidence", "f4"), ("semantic", "i2")])
    records["score"], records["raw_score"], records["confidence"] = scores, scores, confidence
    records["semantic"] = np.arange(len(scores)) % 19
    return records


def test_official_population_preserves_raw_slot_order_and_boundary_rules():
    distances = np.array([2.5, 50, 2.49, 50.01, 5, 5, 10, 10, 10, 10, 10, 10], dtype=np.float32)
    xyz = np.c_[distances, np.zeros((len(distances), 2), np.float32)]
    targets = np.array([0, 0, 0, 0, -1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    slots = np.array([40, 3, 97, 31, 2, 66, 13, 29, 75, 99, 1, 53])
    selected, labels = official_population(xyz, targets)
    assert selected.tolist() == [0, 1, 5, 6, 7, 8, 9, 10, 11]
    assert slots[selected].tolist() == [40, 3, 66, 13, 29, 75, 99, 1, 53]
    np.testing.assert_array_equal(labels, targets[selected])
    with pytest.raises(ValueError, match="five-anomaly"):
        official_population(xyz[:10], targets[:10])


def test_paired_rescue_counts_and_complete_ties_use_official_scores():
    labels = np.r_[np.zeros(10, np.int8), np.ones(5, np.int8)]
    confidence = [.95] * 10 + [.95, .99, .7, .95, .95]
    baseline = point_records([0, 0, 0, 0, .1, .1, .2, .2, .3, .3, .15, .25, .35, .45, .05], confidence)
    joint = point_records([0, 0, .1, .1, .2, .2, .2, .2, .4, .4, .5, .15, .6, .55, .45], [.8] * 15)
    result = comparison_metrics(labels, dict(semantic=baseline, joint=joint), fprs=(.1, .2, .4), confidence=.9)
    assert result["confident_unknown_points"] == 4
    rows = result["methods"]["joint"]["operating_points"]
    assert rows[0]["actual_fpr"] == 0
    assert rows[0]["unknown_recalled"] == 4
    assert rows[0]["confident_unknown_recalled"] == 3
    assert rows[0]["confident_unknown_rescued"] == 2
    assert rows[0]["unknown_lost"] == 0
    assert rows[1]["actual_fpr"] == .2
    assert rows[1]["confident_unknown_rescued"] == 2
    assert rows[1]["confident_unknown_lost"] == 1
    assert rows[1]["equal_actual_fpr"]
    assert rows[2]["actual_fpr"] == .2
    assert result["methods"]["semantic"]["individual_operating_points"][2]["actual_fpr"] == .4
    assert result["methods"]["semantic"]["operating_points"][2]["actual_fpr"] == .2
    for name, records in (("semantic", baseline), ("joint", joint)):
        reference = PointOODMetricsCalculator()
        reference.all_labels, reference.all_scores = [labels], [records["score"]]
        assert result["methods"][name]["metrics"] == reference.compute_metrics()
    # Every method shares one identity array; a common reorder cannot alter any
    # paired statistic, including confidence-selected rescue and loss counts.
    order = np.random.default_rng(7).permutation(len(labels))
    reordered = comparison_metrics(labels[order], dict(semantic=baseline[order], joint=joint[order]),
                                   fprs=(.1, .2, .4), confidence=.9)
    assert result == reordered


def test_requested_fpr_endpoints_and_missing_confident_points_are_explicit():
    labels = np.array([0, 0, 1, 1], np.int8)
    values = point_records([.1, .1, .05, .2], [.5] * 4)
    result = comparison_metrics(labels, dict(semantic=values, joint=values), fprs=(0., 1.))
    rows = result["methods"]["joint"]["operating_points"]
    assert [row["actual_fpr"] for row in rows] == [0., 1.]
    assert all(row["confident_unknown_recall"] is None for row in rows)
    invalid = values.copy()
    invalid["confidence"][0] = 1.1
    with pytest.raises(ValueError, match="confidence"):
        comparison_metrics(labels, dict(semantic=invalid, joint=values))


def matched_checkpoint(variant):
    config = dict(variant=variant, synthetic_anomalies=False, data_identity={"target": "same206"},
                  source_mapping={"24": [8, 9]}, target_mapping={"40": 8}, initial_sha256="official",
                  seed=206, source_epochs=1, target_epochs=8, batch=2, queries=4096,
                  source_replay_fraction=.2, eval_every=2000, target_eval_every=225,
                  selection="normal semantic quality first", budget={"source": {"visits": 100}, "target": {"visits": 200}})
    stage = dict(trained_frames=100, trained_updates=50, planned_frames=100, planned_updates=50,
                 budget_complete=True, selected_update=40)
    return dict(config=config, frozen=True, stages=dict(source=deepcopy(stage), target=deepcopy(stage)))


def test_independent_variants_must_have_matched_actual_training_exposure():
    checkpoints = {variant: matched_checkpoint(variant) for variant in ("semantic", "joint", "separate")}
    checkpoints["joint"]["stages"]["target"]["selected_update"] = 30
    result = comparison_conditions(checkpoints)
    assert result["matched"] and result["training_link_ablation_available"]
    checkpoints["joint"]["stages"]["target"].update(trained_frames=90, trained_updates=45, budget_complete=False)
    with pytest.raises(ValueError, match="trained_frames"):
        comparison_conditions(checkpoints)
    result = comparison_conditions(checkpoints, exploratory=True)
    assert not result["matched"] and "exploratory" in result["role"]
    checkpoints["semantic"]["config"]["variant"] = "joint"
    with pytest.raises(ValueError, match="independently trained"):
        comparison_conditions(checkpoints, exploratory=True)


def test_compare_records_one_shared_preparation_and_exact_point_identity(tmp_path, monkeypatch):
    import src.evaluate as evaluation
    import src.train as training
    from torch import nn

    calls = []
    targets = torch.tensor([-1] + [0] * 6 + [1] * 5, dtype=torch.int8)
    slots = torch.tensor([33, 1, 4, 8, 12, 17, 19, 21, 23, 27, 28, 30])

    class Prepared:
        def __getitem__(self, index):
            calls.append(index)
            xyzi = torch.zeros(12, 4)
            xyzi[:, 0] = 10
            return dict(xyzi=xyzi, slots=slots, targets=targets, index=index, slot_count=40)

    class Model(nn.Module):
        mode = "normal_hypothesis"

        def __init__(self, variant):
            super().__init__()
            self.variant = variant
            self.anchor = nn.Parameter(torch.zeros(()))

        def predict(self, sample):
            score = torch.arange(12, dtype=torch.float32) + int(sample["index"]) * 20
            if self.variant == "joint":
                score = score + .5
            return dict(score=score, raw_score=score + 1, confidence=torch.full((12,), .95),
                        semantic=torch.arange(12) % 19)

    monkeypatch.setattr(evaluation, "PreparedScans", lambda *args, **kwargs: Prepared())
    monkeypatch.setattr(evaluation, "load_model", lambda path, device: (Model(path), matched_checkpoint(path)))
    monkeypatch.setattr(evaluation, "memory_available", lambda: 10 ** 15)
    monkeypatch.setattr(training, "disk_check", lambda *args: None)
    monkeypatch.setattr(training, "runtime_snapshot", lambda: {"role": "unit fixture"})
    manifest = dict(kind="val", version="unit-fixture", sha256="unit-fixture", records=[
        dict(eligible=True, normal=6, anomaly=5, points=12, scan=f"unit-fixture/{i}.bin") for i in range(2)])
    result = evaluation.compare(dict(semantic="semantic", joint="joint"), manifest, tmp_path,
                                torch.device("cpu"), workers=0)
    assert calls == [0, 1]
    identities = np.load(tmp_path / "returns.npy")
    labels = np.load(tmp_path / "labels.npy")
    selected = identities[identities["official"]]
    np.testing.assert_array_equal(selected["target"], labels)
    np.testing.assert_array_equal(selected["slot"], np.tile(slots.numpy()[1:], 2))
    np.testing.assert_array_equal(selected["frame"], np.repeat([0, 1], 11))
    for name, offset in (("semantic", 0), ("joint", .5)):
        values = np.load(tmp_path / f"{name}.npy")
        expected = np.r_[np.arange(1, 12), np.arange(1, 12) + 20] + offset
        np.testing.assert_array_equal(values["score"], expected)
        np.testing.assert_array_equal(values["raw_score"], expected + 1)
    assert result["points"] == 22 and result["scans"] == 2
    assert result["conditions"]["matched"]


def test_observation_diagnostics_matches_independent_student_mixture_quantiles():
    from scipy.optimize import brentq
    from scipy.special import entr, logsumexp, softmax
    from scipy.stats import t
    from src.normal import observation_diagnostics

    rng = np.random.default_rng(410)
    means = np.log(rng.uniform(7., 18., (7, 19, 3)))
    scales = rng.uniform(.07, .3, (7, 19, 3))
    weights = rng.dirichlet([1., 2., 3.], (7, 19))
    beliefs = rng.normal(size=(7, 19))
    appearance = rng.dirichlet([1., 2., 3., 4.], (7, 19))
    means[0, 0], scales[0, 0], weights[0, 0] = np.log([8., 10., 15.]), [.1, .25, .4], [.2, .5, .3]
    # Coincident components reduce exactly to one t distribution, regardless of priors.
    means[2, 2], scales[2, 2], weights[2, 2] = np.log(12.), .15, [.1, .7, .2]
    appearance[2, 2] = [0., 0., 1., 0.]
    distances = np.array([11., 45., 12., 15., 10., 7., 20.])
    values = np.log(distances)
    indices = torch.tensor([7, 2, 4, 1, 0, 8, 6])
    full_values = torch.zeros(10, dtype=torch.float64)
    full_values[indices] = torch.from_numpy(values)
    allowed = torch.zeros(7, 19, dtype=torch.bool)
    for row, category in ((0, 0), (1, 0), (2, 2), (3, 3)):
        allowed[row, category] = True
    allowed[4, [4, 5]] = True
    allowed[5, [6, 7]] = True
    log_density = t.logpdf(values[:, None, None], 3, loc=means, scale=scales)
    prediction = {key: torch.from_numpy(value) for key, value in
                  dict(mean=means, scale=scales, weight=np.log(weights), log_prob=log_density, belief=beliefs).items()}
    prediction["supported"] = torch.tensor([True, True, True, False, True, False, False])
    observation = dict(log_distance=full_values)
    actual = observation_diagnostics(prediction, observation, indices, allowed, torch.from_numpy(appearance))

    expected = {key: np.zeros_like(value, dtype=float) for key, value in actual.items() if isinstance(value, np.ndarray)}
    for row, category in ((0, 0), (1, 0), (2, 2), (3, 3)):
        expected["query_count"][category] += 1
        expected["appearance_mode_mass"][category] += appearance[row, category]
        expected["appearance_mode_entropy_sum"][category] += entr(appearance[row, category]).sum()
        if not prediction["supported"][row]:
            continue
        mu, sigma, prior = means[row, category], scales[row, category], weights[row, category]

        def cdf(value):
            return np.dot(prior, t.cdf(value, 3, loc=mu, scale=sigma))

        quantiles = np.exp([brentq(lambda value: cdf(value) - probability, -50., 50., xtol=1e-13, rtol=1e-14)
                            for probability in (.05, .5, .95)])
        posterior = softmax(np.log(prior) + log_density[row, category])
        expected["prediction_count"][category] += 1
        expected["nll_sum"][category] -= logsumexp(np.log(prior) + log_density[row, category])
        expected["coverage90_count"][category] += .05 <= cdf(values[row]) <= .95
        expected["width90_m_sum"][category] += quantiles[2] - quantiles[0]
        expected["abs_median_error_m_sum"][category] += abs(quantiles[1] - distances[row])
        expected["finite_interval_count"][category] += 1
        expected["geometry_mode_mass"][category] += posterior
        expected["geometry_prior_mass"][category] += prior
        expected["geometry_scale_sum"][category] += sigma
        expected["geometry_mode_entropy_sum"][category] += entr(posterior).sum()
        expected["mean_pair_separation_sum"][category] += np.abs(mu[:, None] - mu[None, :])[np.triu_indices(3, 1)].sum()
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=2e-10, atol=2e-10, err_msg=key)
    assert actual["coarse_query_count"] == 1 and actual["unsupported_query_count"] == 2
    coarse_prior = softmax(beliefs[4, [4, 5]])
    coarse_nll = -logsumexp(np.log(coarse_prior)[:, None] + np.log(weights[4, [4, 5]]) + log_density[4, [4, 5]])
    assert actual["coarse_nll_sum"] == pytest.approx(coarse_nll, abs=1e-12)
    np.testing.assert_allclose(actual["geometry_mode_mass"].sum(1), actual["prediction_count"], atol=1e-12)
    np.testing.assert_allclose(actual["geometry_mode_mass"][2], weights[2, 2], atol=1e-12)
    assert actual["mean_pair_separation_sum"][2] == 0
    assert actual["coverage90_count"][0] == 1 and actual["coverage90_count"][2] == 1

    semantic = observation_diagnostics(None, observation, indices, allowed, torch.from_numpy(appearance))
    for key in ("query_count", "appearance_mode_mass", "appearance_mode_entropy_sum"):
        np.testing.assert_allclose(semantic[key], actual[key], atol=1e-12)
    for key, value in semantic.items():
        if key not in ("query_count", "appearance_mode_mass", "appearance_mode_entropy_sum"):
            assert np.all(np.asarray(value) == 0), key
