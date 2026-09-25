"""Paired mechanism comparisons retain official populations and exact decisions."""

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


def test_feature_support_matches_independent_conditional_gaussian_mixture():
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    from src.normal import FeatureSupport

    rng = np.random.default_rng(206)
    scorer = FeatureSupport(features=3, classes=3, modes=3)
    features = rng.normal(size=(7, 3)).astype(np.float32)
    conditions = np.column_stack((np.linspace(1., 4., 7), rng.normal(size=7))).astype(np.float32)
    location = np.array([.3, -.4, .7, -.8])
    scale = np.array([1.2, .7, 1.6, .4])
    coefficients = rng.normal(scale=.2, size=(3, 3, 4))
    log_variance_coefficients = np.array([[.45, -.35, .3], [0., 0., 0.], [-.4, .55, -.2]])
    centers = rng.normal(scale=.5, size=(3, 3, 4))
    scorer.location.copy_(torch.from_numpy(location))
    scorer.scale.copy_(torch.from_numpy(scale))
    scorer.range_location.fill_(2.1)
    scorer.range_scale.fill_(.8)
    scorer.coefficients.copy_(torch.from_numpy(coefficients))
    scorer.log_variance_coefficients.copy_(torch.from_numpy(log_variance_coefficients))
    scorer.centers.copy_(torch.from_numpy(centers))
    expected = np.full((len(features), 3), np.inf)
    values = (np.column_stack((features, conditions[:, 1])).astype(np.float64) - location) / scale
    distance = (conditions[:, 0].astype(np.float64) - 2.1) / .8
    basis = np.column_stack((np.ones(len(features)), distance, distance ** 2))
    for category, weights in ((0, [.2, .8, 0.]), (2, [.15, .35, .5])):
        # Off-diagonal covariance detects a transposed precision factor.
        factor = np.tril(rng.normal(scale=.3, size=(4, 4)))
        np.fill_diagonal(factor, [.7, 1.1, 1.4, .9])
        covariance = factor @ factor.T
        precision = np.linalg.solve(factor, np.eye(4)).T
        log_weights = np.full(3, -np.inf)
        active = np.asarray(weights) > 0
        log_weights[active] = np.log(np.asarray(weights)[active])
        scorer.present[category] = True
        scorer.precision_cholesky[category].copy_(torch.from_numpy(precision))
        scorer.log_volume[category] = np.log(np.diag(factor)).sum()
        scorer.log_weights[category].copy_(torch.from_numpy(log_weights))
        for row in range(len(features)):
            means = basis[row] @ coefficients[category] + centers[category]
            log_variance = 4 * np.tanh(basis[row] @ log_variance_coefficients[category] / 4)
            # SciPy includes the changing covariance volume as well as residual scaling.
            conditional_covariance = np.exp(log_variance) * covariance
            components = [multivariate_normal.logpdf(values[row], mean=mean, cov=conditional_covariance)
                          for mean in means]
            expected[row, category] = -logsumexp(log_weights + components)
    actual = scorer.class_energy(torch.from_numpy(features), torch.from_numpy(conditions))
    assert torch.isinf(actual[:, 1]).all()
    np.testing.assert_allclose(actual.numpy()[:, [0, 2]], expected[:, [0, 2]], rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(scorer(torch.from_numpy(features), torch.from_numpy(conditions)).numpy(),
                               expected.min(1), rtol=3e-6, atol=3e-6)


def test_feature_support_rejects_unfitted_mismatched_and_nonfinite_inputs():
    from src.normal import FeatureSupport

    scorer = FeatureSupport(features=3, classes=2, modes=1)
    features, conditions = torch.zeros(2, 3), torch.zeros(2, 2)
    with pytest.raises(ValueError, match="not been fitted"):
        scorer(features, conditions)
    scorer.present[0] = True
    scorer.log_weights[0, 0] = 0
    for wrong_features, wrong_conditions in ((torch.zeros(2, 4), conditions),
                                               (features, torch.zeros(3, 2)),
                                               (features, torch.zeros(2, 3))):
        with pytest.raises(ValueError, match="same points"):
            scorer(wrong_features, wrong_conditions)
    conditions[0, 1] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        scorer(features, conditions)


def test_feature_support_can_distinguish_joint_values_at_identical_class_confidence():
    from src.normal import FeatureSupport

    scorer = FeatureSupport(features=3, classes=2, modes=1)
    scorer.present[0] = True
    scorer.log_weights[0, 0] = 0
    features = torch.tensor([[0., 0., 0.], [3., 0., 0.], [0., 0., 0.]])
    conditions = torch.tensor([[2., 0.], [2., 0.], [2., 2.]])
    logits = torch.full((3, 16), -5.)
    logits[:, 3] = 5.
    confidence = logits.softmax(-1).amax(-1)
    assert torch.equal(confidence, confidence[0].expand_as(confidence))
    assert confidence[0] > .999
    # This arithmetic fixture verifies score dependence, not anomaly performance.
    scores = scorer(features, conditions)
    torch.testing.assert_close(scores[1:] - scores[0], torch.tensor([4.5, 2.]))


def test_support_conditions_preserves_raw_query_identity_and_repeated_indices():
    from src.normal import support_conditions

    xyz = np.array([[1 + i * .3, (i % 3) * .4, (i % 2) * .2] for i in range(13)], np.float32)
    xyzi = np.column_stack((xyz, np.arange(len(xyz), dtype=np.float32)))
    distances = np.linalg.norm(xyz.astype(np.float64), axis=1)
    pairwise = np.linalg.norm(xyz.astype(np.float64)[:, None] - xyz.astype(np.float64)[None], axis=-1)
    eighth_neighbor = np.sort(pairwise, axis=1)[:, 8]
    expected = np.column_stack((np.log(distances), np.log(eighth_neighbor))).astype(np.float32)
    actual = support_conditions(xyzi)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    indices = np.array([9, 2, 9, 0, 11])
    # Neighbors come from the complete raw scan, even for fewer than nine queries.
    np.testing.assert_array_equal(support_conditions(xyzi, indices), actual[indices])
    order = np.random.default_rng(43).permutation(len(xyzi))
    np.testing.assert_array_equal(support_conditions(xyzi[order], np.argsort(order)[indices]), actual[indices])
    coincident = support_conditions(np.repeat(xyzi[:1], 9, axis=0))
    np.testing.assert_allclose(coincident[:, 1], np.float32(np.log(.001)), rtol=0, atol=0)
    with pytest.raises(ValueError, match="nine finite"):
        support_conditions(xyzi[:8])


def test_cross_evidence_predictions_exclude_observed_targets_and_labels():
    from src.normal import CrossEvidence

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(206)
        model = CrossEvidence(modes=2)
    sample = dict(features=torch.zeros(3, 252), conditions=torch.zeros(3, 2),
                  labels=torch.tensor([0, 1, 2]))
    predicted = model.predict(sample["features"])
    assert set(predicted) == {"means", "log_scale", "log_weights", "deep_energy"}
    assert predicted["means"].shape == (3, 19, 2, 182)
    assert predicted["log_weights"].shape == (3, 19, 2)
    assert predicted["log_scale"].shape == (3, 182)
    assert predicted["deep_energy"].shape == (3, 19)
    assert not hasattr(model, "class_head")
    score = model.raw_score(sample["features"], sample["conditions"])
    sample["labels"] = torch.tensor([18, 17, 16])
    sample["conditions"] += 4
    changed = model.predict(sample["features"])
    for name in predicted:
        torch.testing.assert_close(changed[name], predicted[name], rtol=0, atol=0)
    assert (model.raw_score(sample["features"], sample["conditions"]) > score).all()
    # Early/middle targets are excluded too; only the last 72 values feed the heads.
    sample["features"][:, :180] += 3
    changed = model.predict(sample["features"])
    for name in predicted:
        torch.testing.assert_close(changed[name], predicted[name], rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = model(sample["features"], sample["conditions"])
        mixed_prediction = model.predict(sample["features"])
    assert mixed.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in mixed_prediction.values())
    torch.testing.assert_close(mixed, model.raw_score(sample["features"], sample["conditions"]), rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in model.parameters()) < 300_000


def test_cross_evidence_matches_independent_gaussian_mixture_and_inverse_transform():
    from scipy.special import logsumexp, softmax
    from scipy.stats import multivariate_normal
    from src.normal import CrossEvidence

    rng = np.random.default_rng(43)
    model = CrossEvidence(modes=2, hidden=2, latent=2)
    features = rng.normal(size=(3, 252)).astype(np.float32)
    conditions = rng.normal(size=(3, 2)).astype(np.float32)
    location = rng.normal(scale=.2, size=252).astype(np.float32)
    scale = rng.uniform(.7, 1.5, size=252).astype(np.float32)
    geo_location = np.array([2.4, -.8], np.float32)
    geo_scale = np.array([.9, .6], np.float32)
    linear = rng.normal(scale=.015, size=(73, 182)).astype(np.float32)
    whitener = np.eye(182, dtype=np.float32) + np.triu(
        rng.normal(scale=.01, size=(182, 182)).astype(np.float32), 1)
    means = rng.normal(scale=.2, size=(2, 182)).astype(np.float32)
    mean_offsets = rng.normal(scale=.15, size=(2, 182)).astype(np.float32)
    embedding = np.linspace(-1, 1, 19, dtype=np.float32)
    raw_scale = np.linspace(-.8, .5, 182, dtype=np.float32)
    raw_weights = np.array([-.7, .3], np.float32)
    weight_offsets = np.array([.4, -.2], np.float32)
    prior_logits = np.linspace(-.6, .8, 19, dtype=np.float32)
    prior_logits[1] = 100  # An absent class must stay excluded despite its logit.
    centers = np.zeros((19, 72), np.float32)
    centers[:, 0] = np.linspace(-1.5, 1.5, 19)
    present = [0, 7, 18]
    with torch.no_grad():
        for name, value in (("feat_location", location), ("feat_scale", scale),
                            ("geo_location", geo_location), ("geo_scale", geo_scale),
                            ("linear", linear), ("whitener", whitener)):
            getattr(model, name).copy_(torch.from_numpy(value))
        # A tiny deterministic head makes class-dependent parameters independently
        # reproducible; this artificial fixture verifies implementation only.
        for parameter in model.parameters():
            parameter.zero_()
        model.class_embedding.weight[:, 0].copy_(torch.from_numpy(embedding))
        model.density_head[0].weight[0, 0] = 1
        model.density_head[-1].weight[:, 0].copy_(torch.from_numpy(
            np.concatenate((mean_offsets.ravel(), weight_offsets))))
        model.density_head[-1].bias.copy_(torch.from_numpy(np.concatenate((means.ravel(), raw_weights))))
        model.scale_head.bias.copy_(torch.from_numpy(raw_scale))
        model.prior_head.bias.copy_(torch.from_numpy(prior_logits))
        model.deep_centers.copy_(torch.from_numpy(centers))
        model.deep_present.zero_()
        model.deep_present[present] = True
    values = (features.astype(np.float64) - location) / scale
    geometry = (conditions.astype(np.float64) - geo_location) / geo_scale
    base = np.column_stack((np.ones(3), values[:, 180:])) @ linear.astype(np.float64)
    observed = (np.column_stack((values[:, :180], geometry)) - base) @ whitener.astype(np.float64)
    log_scale = 3 * np.tanh(raw_scale.astype(np.float64) / 3)
    covariance = np.diag(np.exp(2 * log_scale))
    activation = embedding.astype(np.float64) / (1 + np.exp(-embedding.astype(np.float64)))
    class_means = means.astype(np.float64)[None] + activation[:, None, None] * mean_offsets
    class_weights = raw_weights.astype(np.float64)[None] + activation[:, None] * weight_offsets
    log_weights = class_weights - logsumexp(class_weights, axis=1, keepdims=True)
    # SciPy evaluates the full normalized density, including its scale-dependent volume.
    components = np.stack([np.column_stack([
        multivariate_normal.logpdf(observed, mean=mean, cov=covariance) for mean in class_mean])
        for class_mean in class_means], axis=1)
    expected_conditional = -logsumexp(components + log_weights[None], axis=2)
    expected_base = np.full((len(features), 19), np.inf)
    for category in present:
        expected_base[:, category] = -multivariate_normal.logpdf(
            values[:, 180:], mean=centers[category], cov=np.eye(72)) + np.log(len(present))
    # The Gaussian mixture supplies only p0(D). Learned rho(c|D) independently
    # partitions that fixed marginal; it is not the Gaussian class posterior.
    log_rho = np.full(19, -np.inf)
    log_rho[present] = prior_logits[present].astype(np.float64)
    log_rho -= logsumexp(log_rho)
    expected_deep = -logsumexp(-expected_base, axis=1)[:, None] - log_rho[None]
    expected_joint = expected_deep + expected_conditional
    tensor_features = torch.from_numpy(features)
    tensor_conditions = torch.from_numpy(conditions)
    predicted_parameters = model.predict(tensor_features)
    np.testing.assert_allclose(predicted_parameters["deep_energy"].detach().numpy(),
                               expected_deep, rtol=2e-6, atol=2e-5)
    conditional = model.negative_log_likelihood(predicted_parameters,
                                                model.observations(tensor_features, tensor_conditions))
    np.testing.assert_allclose(conditional.detach().numpy(), expected_conditional, rtol=2e-6, atol=2e-5)
    joint = model.class_energy(tensor_features, tensor_conditions, predicted_parameters)
    np.testing.assert_allclose(joint.detach().numpy(), expected_joint, rtol=2e-6, atol=3e-5)
    expected_score = -logsumexp(-expected_joint, axis=1)
    np.testing.assert_allclose(model.raw_score(tensor_features, tensor_conditions).detach().numpy(),
                               expected_score, rtol=2e-6, atol=3e-5)
    mean_by_class = (softmax(class_weights, axis=1)[..., None] * class_means).sum(1)
    mean = softmax(-expected_deep, axis=1) @ mean_by_class
    expected = base + np.linalg.solve(whitener.astype(np.float64).T, mean.T).T
    predicted = model.prediction(tensor_features)
    np.testing.assert_allclose(predicted["features"].detach().numpy(),
                               expected[:, :180] * scale[:180] + location[:180], rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(predicted["conditions"].detach().numpy(),
                               expected[:, 180:] * geo_scale + geo_location, rtol=3e-6, atol=2e-6)
    model.point_chunk = 1
    np.testing.assert_allclose(model.raw_score(tensor_features, tensor_conditions).detach().numpy(),
                               expected_score, rtol=2e-6, atol=3e-5)
    for name, value in model.prediction(tensor_features).items():
        torch.testing.assert_close(value, predicted[name], rtol=3e-6, atol=2e-6)


def test_cross_evidence_deep_support_matches_shared_covariance_class_mixture():
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    from src.normal import CrossEvidence

    rng = np.random.default_rng(827)
    model = CrossEvidence(modes=1)
    features = rng.normal(size=(7, 252)).astype(np.float32)
    location = rng.normal(scale=.3, size=252).astype(np.float32)
    scale = rng.uniform(.5, 2., size=252).astype(np.float32)
    factor = np.tril(rng.normal(scale=.02, size=(72, 72)))
    np.fill_diagonal(factor, np.linspace(.7, 1.3, 72))
    covariance = factor @ factor.T
    centers = rng.normal(scale=.4, size=(3, 72)).astype(np.float32)
    with torch.no_grad():
        model.feat_location.copy_(torch.from_numpy(location))
        model.feat_scale.copy_(torch.from_numpy(scale))
        model.deep_present.zero_()
        model.deep_present[[0, 7, 18]] = True
        model.deep_centers[[0, 7, 18]] = torch.from_numpy(centers)
        # Absent centers must never enter the likelihood or its prior normalization.
        model.deep_centers[1].fill_(100.)
        model.deep_whitener.copy_(torch.from_numpy(np.linalg.inv(factor).T))
        model.deep_log_volume.fill_(np.log(np.diag(factor)).sum())
    deep = ((features.astype(np.float64) - location) / scale)[:, 180:]
    components = np.stack([multivariate_normal.logpdf(deep, mean=center, cov=covariance)
                           for center in centers], axis=1)
    expected_classes = np.full((len(features), 19), np.inf)
    expected_classes[:, [0, 7, 18]] = -components + np.log(len(centers))
    actual = model.base_class_energy(torch.from_numpy(features))
    np.testing.assert_allclose(actual.numpy(), expected_classes, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(model.deep_energy(torch.from_numpy(features)).numpy(),
                               -logsumexp(-expected_classes, axis=1), rtol=2e-6, atol=2e-5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(model.base_class_energy(torch.from_numpy(features)), actual, rtol=0, atol=0)
    model.deep_present.zero_()
    with pytest.raises(ValueError, match="observed normal class"):
        model.base_class_energy(torch.from_numpy(features))


def test_cross_evidence_learned_class_prior_preserves_fixed_deep_marginal():
    from scipy.special import logsumexp
    from src.normal import CrossEvidence

    rng = np.random.default_rng(746)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(206)
        model = CrossEvidence(hidden=4, latent=2)
    features = torch.from_numpy(rng.normal(size=(5, 252)).astype(np.float32))
    present = [0, 7, 18]
    with torch.no_grad():
        model.deep_present.zero_()
        model.deep_present[present] = True
        model.deep_centers[7, 0] = 3
        model.deep_centers[18, 1] = -2
    base = model.base_class_energy(features)
    marginal = model.deep_energy(features)
    initial = model.predict(features)["deep_energy"]
    assert torch.count_nonzero(model.prior_head.weight) == torch.count_nonzero(model.prior_head.bias) == 0
    torch.testing.assert_close(initial[:, present], marginal[:, None].expand(-1, 3) + np.log(3),
                               rtol=0, atol=1e-5)
    assert torch.isinf(initial[:, ~model.deep_present]).all()
    assert not torch.allclose((-base).softmax(-1)[:, present], torch.full((5, 3), 1 / 3))

    weight = rng.normal(scale=1.7, size=(19, 2)).astype(np.float32)
    bias = rng.normal(scale=.7, size=19).astype(np.float32)
    bias[1] = 1000
    with torch.no_grad():
        model.prior_head.weight.copy_(torch.from_numpy(weight))
        model.prior_head.bias.copy_(torch.from_numpy(bias))
    latent = model.encoder(model._features(features)[:, 180:]).detach().numpy().astype(np.float64)
    logits = latent @ weight.astype(np.float64).T + bias.astype(np.float64)
    logits[:, ~model.deep_present.numpy()] = -np.inf
    log_rho = logits - logsumexp(logits, axis=1, keepdims=True)
    learned = model.deep_class_energy(features)
    np.testing.assert_allclose(learned.detach().numpy(), marginal[:, None].numpy() - log_rho,
                               rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose((-learned).softmax(-1).detach().numpy(), np.exp(log_rho), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.exp(log_rho).sum(1), np.ones(5), rtol=0, atol=1e-14)
    torch.testing.assert_close(model.predict(features)["deep_energy"], learned, rtol=0, atol=0)
    torch.testing.assert_close(model.base_class_energy(features), base, rtol=0, atol=0)
    torch.testing.assert_close(model.deep_energy(features), marginal, rtol=0, atol=0)
    # Marginalize the learned joint in log space to avoid underflow in 72D.
    torch.testing.assert_close(-torch.logsumexp(-learned, -1), marginal, rtol=0, atol=2e-5)
    assert not torch.allclose(learned[:, present], initial[:, present])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(model.deep_class_energy(features), learned, rtol=0, atol=0)


def test_cross_evidence_observations_change_joint_class_support_and_partial_label_gradients():
    from src.normal import CrossEvidence

    # Artificial mechanism fixture: identical deep input, distinct observed detail.
    # Other normal classes are alternative explanations, never anomaly labels.
    model = CrossEvidence(modes=1, hidden=4, latent=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.deep_present.zero_()
        model.deep_present[:2] = True
        model.class_embedding.weight[0, 0] = 1
        model.class_embedding.weight[1, 0] = 3
        model.encoder[2].bias[0] = .5
        model.density_head[0].weight[0, 0] = 1
        model.density_head[-1].weight[0, 0] = 1
        model.density_head[-1].weight[180, 0] = 1
    features = torch.zeros(2, 252)
    conditions = torch.zeros(2, 2)
    parameters = model.predict(features)
    means = parameters["means"][0, :2, 0].detach()
    features[:, :180] = means[:, :180]
    conditions[:] = means[:, 180:]
    for name, value in model.predict(features).items():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
        torch.testing.assert_close(value[0], value[1], rtol=0, atol=0)
    deep = model.deep_class_energy(features)
    torch.testing.assert_close(deep[0], deep[1], rtol=0, atol=0)
    energy = model.class_energy(features, conditions)
    assert energy.argmin(1).tolist() == [0, 1]
    assert energy[0, 0] < energy[0, 1] and energy[1, 1] < energy[1, 0]
    torch.testing.assert_close(model.raw_score(features, conditions),
                               -torch.logsumexp(-energy, -1), rtol=0, atol=0)
    allowed = torch.zeros(2, 19, dtype=torch.bool)
    allowed[0, 0], allowed[1, 1] = True, True
    logits = -energy
    normalizer = torch.logsumexp(logits, -1)
    supported = torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), -1)
    partial_ce = normalizer - supported
    torch.testing.assert_close(partial_ce, -torch.log(logits.softmax(-1)[allowed]), rtol=1e-4, atol=1e-5)
    # Admitting both normal explanations makes their summed posterior one; no
    # separate binary rejection target is assigned to either class.
    allowed[:, :2] = True
    coarse_ce = normalizer - torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), -1)
    torch.testing.assert_close(coarse_ce, torch.zeros_like(coarse_ce), rtol=0, atol=0)
    assert (coarse_ce <= partial_ce).all()
    # One shared joint likelihood supplies both observation and prior gradients;
    # unequal fixture weights prevent exact cancellation between the two classes.
    loss = ((-supported / model.dimensions + .1 * partial_ce) * torch.tensor([.25, .75])).sum()
    loss.backward()
    assert model.class_embedding.weight.grad[:2].abs().sum() > 0
    for layer in (model.density_head[0], model.density_head[-1]):
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    for parameter in model.prior_head.parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        assert torch.count_nonzero(parameter.grad[~model.deep_present]) == 0


def test_score_calibration_uses_frozen_groups_and_preserves_unbounded_tail_order():
    from src.normal import ScoreCalibration

    calibration = ScoreCalibration()
    scores = torch.cat((torch.linspace(0, 1, 2048), torch.linspace(10, 20, 2048), torch.ones(7)))
    predicted = torch.cat((torch.zeros(2048), torch.ones(2048), torch.full((7,), 2))).long()
    conditions = torch.zeros(len(scores), 2)
    report = calibration.fit(scores, predicted, conditions)
    assert report["class_counts"][:3] == [2048, 2048, 7]
    assert report["fallback_classes"] == list(range(2, 16))
    assert calibration(scores, None, conditions) is scores
    calibration.enabled.fill_(True)
    probes = torch.tensor([30., 40., 50.])
    reference = calibration(probes, torch.zeros(3, dtype=torch.long), conditions[:3])
    assert torch.isfinite(reference).all() and (reference.diff() > 0).all()
    assert reference[0] > -np.log(.001)
    same_score = torch.full((3,), .5)
    rescaled = calibration(same_score, torch.tensor([0, 1, 2]), conditions[:3])
    assert rescaled[0] != rescaled[1]
    global_reference = calibration(same_score, torch.full((3,), 15), conditions[:3])
    assert rescaled[2] == global_reference[2]
    torch.testing.assert_close(reference, calibration(
        probes, torch.zeros(3, dtype=torch.long), torch.tensor([[100., -100.]]).repeat(3, 1)),
        rtol=0, atol=0)
    with pytest.raises(ValueError, match="16-class"):
        calibration(probes, torch.tensor([0, 1, 18]), conditions[:3])


def test_score_calibration_ties_fallback_and_state_roundtrip():
    import io
    from src.normal import ScoreCalibration

    calibration = ScoreCalibration()
    scores = torch.cat((torch.zeros(2048), torch.arange(2048).float().div(8).floor()))
    predicted = torch.cat((torch.zeros(2048), torch.ones(2048))).long()
    conditions = torch.zeros(len(scores), 2)
    report = calibration.fit(scores, predicted, conditions)
    assert 0 in report["fallback_classes"] and 1 not in report["fallback_classes"]
    assert calibration.lengths[0] < 11
    calibration.enabled.fill_(True)
    probes = torch.tensor([-1., 0., 1., 300., 600.])
    classes = torch.tensor([0, 0, 1, 1, 15])
    expected = calibration(probes, classes, conditions[:5])
    checkpoint = io.BytesIO()
    torch.save(calibration.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = ScoreCalibration()
    state = torch.load(checkpoint, weights_only=True)
    assert set(state) == {"probabilities", "knots", "levels", "lengths", "counts", "groups",
                          "minimum_points", "fitted", "enabled"}
    restored.load_state_dict(state, strict=True)
    ScoreCalibration(range_bandwidth=0.).load_state_dict(state, strict=True)
    torch.testing.assert_close(restored(probes, classes, conditions[:5]), expected, rtol=0, atol=0)
    if torch.cuda.is_available():
        actual = restored.cuda()(probes.cuda(), classes.cuda(), conditions[:5].cuda()).cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_range_calibration_matches_weighted_quantiles_and_is_continuous():
    from src.normal import ScoreCalibration

    rng = np.random.default_rng(73)
    distance = np.linspace(np.log(2.5), np.log(50), 16384)
    scores = 6 * distance + rng.normal(0, .5, len(distance))
    conditions = torch.from_numpy(np.column_stack((distance, np.zeros(len(distance)))))
    classes = torch.zeros(len(distance), dtype=torch.long)
    calibration = ScoreCalibration(range_bandwidth=.5)
    report = calibration.fit(scores, classes, conditions)
    assert report["range_fallback_anchors"][1] == 0
    calibration.enabled.fill_(True)

    # Independent empirical inverse CDF verifies the scientific q99 threshold.
    order = np.argsort(scores)
    quantiles = []
    for anchor in calibration.range_anchors.numpy():
        weights = np.exp(-.5 * ((distance[order] - anchor) / .5) ** 2)
        indices = np.searchsorted(np.cumsum(weights), calibration.probabilities.numpy() * weights.sum())
        quantiles.append(scores[order[indices]])
    quantiles = np.asarray(quantiles, dtype=np.float32)
    np.testing.assert_array_equal(calibration.range_knots[1].numpy(), quantiles)
    anchor = calibration.range_anchors.numpy()
    probe_range = anchor[9] * .65 + anchor[10] * .35
    threshold = quantiles[9, 8] * .65 + quantiles[10, 8] * .35
    at_threshold = calibration(torch.tensor([threshold]), torch.zeros(1, dtype=torch.long),
                               torch.tensor([[probe_range, 0.]]))
    assert float(at_threshold[0]) == pytest.approx(-np.log(.01), abs=2e-5)

    probes = torch.linspace(-10, 50, 128)
    context = torch.tensor([[probe_range, 0.]]).repeat(len(probes), 1)
    output = calibration(probes, classes[:len(probes)], context)
    assert (output.diff() > 0).all() and output[-1] > -np.log(.001)
    shifted_spacing = context.clone()
    shifted_spacing[:, 1] = float("nan")
    torch.testing.assert_close(calibration(probes, classes[:len(probes)], shifted_spacing), output, rtol=0, atol=0)
    for position in anchor[1:-1]:
        near = torch.tensor([[position - 1e-6, 0.], [position, 0.], [position + 1e-6, 0.]])
        values = calibration(torch.full((3,), 16.), classes[:3], near)
        assert float((values - values[1]).abs().max()) < 1e-4
    endpoints = torch.tensor([[anchor[0], 0.], [anchor[-1], 0.]])
    assert calibration(torch.full((2,), 16.), classes[:2], endpoints).diff().abs().item() > .1
    invalid = context.clone()
    invalid[0, 0] = float("inf")
    with pytest.raises(ValueError, match="log range"):
        calibration(probes, classes[:len(probes)], invalid)
    with pytest.raises(ValueError, match="log range"):
        calibration.fit(probes, classes[:len(probes)], invalid)


def test_range_calibration_effective_sample_fallback_and_rare_class_global():
    from src.normal import ScoreCalibration

    first = np.linspace(np.log(2.5), np.log(50), 3072)
    second = np.linspace(np.log(2.5), np.log(50), 12288)
    distance = np.concatenate((first, second, first[:7]))
    scores = np.concatenate((np.linspace(0, 1, len(first)), 8 + 2 * second, np.zeros(7)))
    classes = torch.tensor([0] * len(first) + [1] * len(second) + [2] * 7)
    conditions = torch.from_numpy(np.column_stack((distance, np.zeros(len(distance)))))
    calibration = ScoreCalibration(range_bandwidth=.25)
    report = calibration.fit(scores, classes, conditions)
    assert report["range_fallback_anchors"][1] == 16
    assert calibration.range_groups_enabled[1]
    torch.testing.assert_close(calibration.range_knots[1], calibration.knots[1].expand(16, -1), rtol=0, atol=0)
    assert calibration.groups[3] == 0 and not calibration.range_groups_enabled[0]
    calibration.enabled.fill_(True)
    probes = torch.full((2,), 12.)
    context = torch.tensor([[np.log(10), 0.], [np.log(20), 0.]])
    rare = calibration(probes, torch.full((2,), 2), context)
    absent = calibration(probes, torch.full((2,), 15), context)
    torch.testing.assert_close(rare, absent, rtol=0, atol=0)
    original = ScoreCalibration()
    original.fit(scores, classes, conditions)
    original.enabled.fill_(True)
    torch.testing.assert_close(rare, original(probes, torch.full((2,), 2), context), rtol=0, atol=0)
    assert rare[0] == rare[1]
    supported = calibration(probes, torch.ones(2, dtype=torch.long), context)
    assert calibration.range_groups_enabled[2] and supported.diff().abs().item() > .1

    # Loading an older fitted state must not silently change its scoring rule.
    state = deepcopy(calibration.state_dict())
    state["range_groups_enabled"][0] = True
    state["range_knots"][0] = state["range_knots"][2]
    historical = ScoreCalibration(range_bandwidth=.25)
    historical.load_state_dict(state, strict=True)
    torch.testing.assert_close(historical(probes, torch.full((2,), 2), context), supported, rtol=0, atol=0)


def test_range_calibration_atoms_keep_class_cdf_and_local_atoms_fall_back():
    from src.normal import ScoreCalibration

    size = 16384
    continuous = np.linspace(0, 100, size)
    local_atom = (continuous > 28) & (continuous < 48)
    continuous[local_atom] = 38
    scores = np.concatenate((np.repeat([0., 1.], 4096), continuous))
    distance = np.full(len(scores), np.log(50))
    distance[8192:][local_atom] = np.log(2.5)
    conditions = np.column_stack((distance, np.zeros(len(scores))))
    classes = torch.tensor([0] * 8192 + [1] * size)
    calibration = ScoreCalibration(range_bandwidth=.25)
    calibration.fit(scores, classes, conditions)
    assert not calibration.range_groups_enabled[1]
    assert calibration.lengths[1] == 3  # Linear empirical median lies between the two atoms.
    assert calibration.range_groups_enabled[2]
    # The local atom has >2048 effective points; duplicate quantiles cause fallback.
    weights = np.exp(-.5 * ((distance[8192:] - np.log(2.5)) / .25) ** 2)
    assert weights.sum() ** 2 / np.square(weights).sum() > 2048
    torch.testing.assert_close(calibration.range_knots[2, 0], calibration.knots[2], rtol=0, atol=0)
    assert not torch.equal(calibration.range_knots[2, -1], calibration.knots[2])
    calibration.enabled.fill_(True)
    original = ScoreCalibration()
    original.fit(scores, classes, conditions)
    original.enabled.fill_(True)
    probes = torch.tensor([-1., .5, 3.])
    context = torch.tensor([[np.log(2.5), 0.], [np.log(20), 0.], [np.log(50), 0.]])
    torch.testing.assert_close(calibration(probes, torch.zeros(3, dtype=torch.long), context),
                               original(probes, torch.zeros(3, dtype=torch.long), context), rtol=0, atol=0)
    restored = ScoreCalibration(range_bandwidth=1.)
    restored.load_state_dict(calibration.state_dict(), strict=True)
    assert float(restored.range_bandwidth) == .25
    torch.testing.assert_close(restored(probes, torch.ones(3, dtype=torch.long), context),
                               calibration(probes, torch.ones(3, dtype=torch.long), context), rtol=0, atol=0)


def test_cross_evidence_calibrates_one_raw_score_using_supplied_official_prediction():
    from src.normal import CrossEvidence

    model = CrossEvidence()
    features, conditions = torch.zeros(3, 252), torch.zeros(3, 2)
    with torch.no_grad():
        model.deep_present.zero_()
        model.deep_present[18] = True
    predicted = model.predict(features)
    energy = model.class_energy(features, conditions)
    raw = model.raw_score(features, conditions)
    torch.testing.assert_close(model.negative_log_likelihood(
        predicted, model.observations(features, conditions)) + predicted["deep_energy"],
        energy, rtol=0, atol=0)
    torch.testing.assert_close(-torch.logsumexp(-energy, -1), raw, rtol=0, atol=0)
    scores = torch.cat((torch.linspace(0, 200, 2048), torch.linspace(300, 500, 2048)))
    official_classes = torch.cat((torch.zeros(2048), torch.ones(2048))).long()
    model.calibration.fit(scores, official_classes, torch.zeros(len(scores), 2))
    model.calibration.enabled.fill_(True)
    # The learned 19-class explanation cannot replace the supplied official
    # 16-class prediction that selects a frozen calibration group.
    assert (energy.argmin(1) == 18).all()
    with pytest.raises(ValueError, match="16-class"):
        model(features, conditions)
    supplied = torch.tensor([0, 1, 0])
    expected = model.calibration(raw, supplied, conditions)
    torch.testing.assert_close(model(features, conditions, supplied), expected, rtol=0, atol=0)
    assert expected[0] != expected[1]


def test_support_queries_and_cache_preserve_coarse_normal_candidate_sets(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src import data
    from src.train import SupportScans, support_cache

    points = np.zeros((12, 4), dtype=np.float32)
    points[:, 0] = np.arange(4, 16)
    points[:, 3] = .2
    points[[0, 11], 0] = [2., 55.]
    allowed = np.zeros((len(points), 19), dtype=bool)
    allowed[1, 0], allowed[4, 3], allowed[7, 12] = True, True, True
    allowed[2, [5, 6]] = True
    allowed[3, [8, 9, 10, 11]] = True
    allowed[6, [14, 15]] = True
    slots = np.array([1, 4, 6, 8, 12, 15, 17, 20, 25, 27, 30, 31])
    # read_normal_record already clears out-of-range and untrusted label sets;
    # every returned row still represents an actual return in the input context.
    raw = dict(xyzi=points, allowed=allowed, slots=slots, slot_count=32)
    monkeypatch.setattr(data, "read_normal_record", lambda record: raw)
    records = [dict(source="nuscenes", scene="normal-fixture")]
    expected = np.flatnonzero(allowed.any(1))
    semantic = np.array([0, -1, -1, 3, -1, 12], dtype=np.int16)
    for development in (False, True):
        sample = SupportScans(records, development=development)[0]
        np.testing.assert_array_equal(sample["queries"].numpy(), expected)
        np.testing.assert_array_equal(sample["slots"].numpy(), slots[expected])
        np.testing.assert_array_equal(sample["allowed"].numpy(), allowed[expected])
        np.testing.assert_array_equal(sample["semantic"].numpy(), semantic)
        assert sample["allowed"].dtype == torch.bool
        assert len(sample["inverse"]) == len(points)
        assert np.all(np.any(points[expected, :3] != 0, axis=1))
        assert np.all((np.linalg.norm(points[expected, :3], axis=1) >= 2.5)
                      & (np.linalg.norm(points[expected, :3], axis=1) <= 50))

    def encode(sample, indices):
        return dict(features=indices[:, None].float().expand(-1, 252))

    model = SimpleNamespace(perception=SimpleNamespace(encode=encode))
    info = support_cache(model, records, tmp_path, torch.device("cpu"), workers=0)
    assert info["labels"] == "normal_candidate_sets_v1" and info["count"] == len(expected)
    np.testing.assert_array_equal(np.load(info["paths"]["allowed"])[:info["count"]], allowed[expected])
    np.testing.assert_array_equal(np.load(info["paths"]["semantic"])[:info["count"]], semantic)
    np.testing.assert_array_equal(np.load(info["paths"]["slot"])[:info["count"]], slots[expected])
    np.testing.assert_array_equal(np.load(info["paths"]["features"])[:info["count"], 0], expected)
    with pytest.raises(ValueError, match="will not overwrite"):
        support_cache(model, records, tmp_path, torch.device("cpu"), workers=0)


def test_partial_label_normal_em_matches_independent_penalized_likelihood(tmp_path):
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    from src.normal import CrossEvidence
    from src.train import cross_initialize

    # Artificial implementation fixture, not measured scientific training data.
    rng = np.random.default_rng(206)
    latent = np.r_[np.zeros(60), np.ones(60), np.arange(120) % 2].astype(int)
    features = rng.normal(size=(240, 252)).astype(np.float32)
    features[:, 180:188] += (latent[:, None] * 2 - 1) * .45
    semantic = np.r_[np.full(60, 3), np.full(60, 8), np.full(120, -1)].astype(np.int16)
    allowed = np.zeros((240, 19), dtype=bool)
    allowed[:60, 3], allowed[60:120, 8] = True, True
    allowed[120:, [3, 8, 14]] = True  # Class 14 has no singleton anchor.
    values = dict(features=features, conditions=rng.normal(size=(240, 2)).astype(np.float32),
                  semantic=semantic, allowed=allowed)
    paths = {key: str(tmp_path / (key + ".npy")) for key in values}
    for key, value in values.items():
        np.save(paths[key], value)
    indices = [np.flatnonzero(semantic == category) for category in range(19)]
    indices.append(np.flatnonzero(semantic == -1))
    model = CrossEvidence()
    report = cross_initialize(model, dict(paths=paths, count=240), indices)
    assert report["singleton_points"] == report["coarse_points"] == 120
    assert 1 <= report["deep_em_iterations"] <= 20
    assert np.all(np.diff(report["deep_penalized_objectives"]) >= -1e-6)
    np.testing.assert_array_equal(model.deep_present.nonzero().flatten().numpy(), [3, 8])

    # Reconstruct the covariance from saved FP32 parameters; SciPy evaluates the
    # class densities independently of the implementation's whitened-distance E-step.
    deep = ((features.astype(float) - model.feat_location.numpy())
            / model.feat_scale.numpy())[:, 180:]
    inverse = np.linalg.inv(model.deep_whitener.numpy().astype(float))
    covariance = inverse.T @ inverse
    component = np.column_stack([multivariate_normal.logpdf(
        deep, mean=model.deep_centers[category].numpy(), cov=covariance) for category in (3, 8)]) - np.log(2)
    unrestricted = logsumexp(component, axis=1).sum()
    admitted = allowed[:, [3, 8]]
    component[~admitted] = -np.inf
    marginal = logsumexp(component, axis=1)
    responsibility = np.exp(component - marginal[:, None])
    assert np.all(responsibility[~admitted] == 0)
    np.testing.assert_array_equal(responsibility[:120], np.eye(2)[latent[:120]])
    assert unrestricted - marginal.sum() > .01  # The label constraint changes this fixture.
    objective = marginal.sum() - .5 * len(deep) * .01 * np.trace(np.linalg.inv(covariance))
    assert abs(objective - report["deep_penalized_objective"]) / len(deep) < 2e-5
    assert abs(marginal.sum() - report["deep_partial_log_likelihood"]) / len(deep) < 2e-5

    semantic[120] = allowed[120].argmax()
    np.save(paths["semantic"], semantic)
    with pytest.raises(ValueError, match="singleton identities disagree"):
        cross_initialize(CrossEvidence(), dict(paths=paths, count=240), indices)
