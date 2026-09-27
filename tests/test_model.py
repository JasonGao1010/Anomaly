"""Paper controls: matched initialization, exact readouts, and normal CSSR fitting."""

import numpy as np
import pytest
import torch
from torch import nn

from src.model import NormalHypothesis, NORMAL_VARIANTS
from src.normal import allowed_loss, geometry_energy, LOG_RETURN_PEAK


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr("src.model.LitePT", lambda **kwargs: nn.Identity())
    return lambda variant="joint": NormalHypothesis(variant)


def test_every_control_preserves_common_initialization_and_random_stream(models):
    states, streams = {}, []
    for variant in NORMAL_VARIANTS:
        torch.manual_seed(206)
        model = models(variant)
        states[variant] = model.state_dict()
        streams.append(torch.rand(9))
    for variant, state in states.items():
        for key in states["joint"]:
            if key == "hypotheses.surface.weight" or key == "hypotheses.surface.bias":
                if variant == "single_component":
                    continue
            torch.testing.assert_close(state[key], states["joint"][key], rtol=0, atol=0)
    for stream in streams:
        torch.testing.assert_close(stream, streams[0], rtol=0, atol=0)
    assert models("single_component").hypotheses.components == 1
    assert models("target_available").hypotheses.target_available


def test_fixed_readouts_match_density_arithmetic_and_share_one_forward(models, monkeypatch):
    model = models()
    appearance = torch.full((3, 19), 10., dtype=torch.float64)
    appearance[:, :2] = torch.tensor([[0., 1.], [1., 0.], [0., 1.]])
    measurement = torch.full_like(appearance, 8.)
    measurement[:, :2] = torch.tensor([[5., 0.], [0., 6.], [0., 0.]])
    measurement[-1] = 0
    energy = appearance + measurement
    parts = dict(energy=energy, semantic_energy=appearance, geometry_energy=measurement,
                 raw_score=energy.amin(-1))
    calls = []

    def components(sample):
        calls.append(1)
        return parts

    monkeypatch.setattr(model, "components", components)
    outputs = model.predict_readouts(dict(xyzi=torch.zeros(3, 4)))
    assert len(calls) == 1
    common = -measurement.neg().exp().mean(-1).log()
    torch.testing.assert_close(outputs["common_density"]["raw_score"], appearance.amin(-1) + common)
    assert torch.equal(outputs["common_density"]["semantic"], appearance.argmin(-1))
    assert torch.equal(outputs["appearance"]["semantic"], appearance.argmin(-1))
    torch.testing.assert_close(outputs["independent_minima"]["raw_score"],
                               appearance.amin(-1) + measurement.amin(-1))
    assert torch.equal(outputs["independent_minima"]["semantic"], energy.argmin(-1))
    assert bool((outputs["joint"]["raw_score"] >= outputs["independent_minima"]["raw_score"]).all())
    for output in outputs.values():
        torch.testing.assert_close(output["score"], output["raw_score"])


def test_standard_and_cssr_classification_follow_their_paper_objectives(models, monkeypatch):
    torch.manual_seed(3)
    features = torch.randn(5, 48)
    allowed = torch.zeros(5, 19, dtype=torch.bool)
    allowed[0:2, 0], allowed[2:4, 1], allowed[4, 2:4] = True, True, True
    sample = dict(xyzi=torch.zeros(5, 4), allowed=allowed, queries=torch.tensor([0, 2]))
    for variant in ("standard", "cssr"):
        model = models(variant)
        monkeypatch.setattr(model, "features", lambda sample: features)
        energy = model.feature_energy(features)
        expected = allowed_loss(-energy, allowed)
        if variant == "cssr":
            expected = allowed_loss(-energy[sample["queries"]], allowed[sample["queries"]]) + .5 * expected
            errors = torch.stack([(features - ae(features)).abs().sum(-1) for ae in model.autoencoders], -1)
            torch.testing.assert_close(energy, .1 * errors)
            with pytest.raises(ValueError, match="statistics"):
                model.predict(sample)
        loss, _ = model.train().loss(sample)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        parameters = model.classifier.parameters() if variant == "standard" else model.autoencoders.parameters()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        if variant == "standard":
            outputs = model.predict_readouts(sample)
            torch.testing.assert_close(outputs["energy"]["raw_score"], -torch.logsumexp(-energy, -1))
            torch.testing.assert_close(outputs["softmax"]["raw_score"], 1 - (-energy).softmax(-1).amax(-1))
            assert torch.equal(outputs["energy"]["semantic"], outputs["softmax"]["semantic"])
        model.eval().loss(sample)
        assert model.development_prediction.shape == (5,)
        assert model.development_diagnostics == {}


def test_cssr_reference_matches_independent_allowed_point_moments(models, monkeypatch):
    model = models("cssr")
    rng = np.random.default_rng(27)
    plain = [rng.normal(size=(6, 48)), rng.normal(size=(5, 48))]
    augmented = [rng.normal(size=(7, 48)), rng.normal(size=(4, 48))]

    def samples(arrays):
        for values in arrays:
            allowed = torch.zeros(len(values), 19, dtype=torch.bool)
            allowed[1:, 0] = True
            yield dict(features=torch.from_numpy(values).float(), allowed=allowed)

    monkeypatch.setattr(model, "features", lambda sample: sample["features"])

    def errors(features):
        return (features[:, :1] - torch.arange(19, device=features.device)[None] / 3).abs() + .2

    monkeypatch.setattr(model, "reconstruction_error", errors)
    result = model.fit_cssr_reference(samples(plain), samples(augmented), torch.device("cpu"))
    x = np.concatenate([x[1:].astype(np.float32) for x in plain]).astype(np.float64)
    error = errors(torch.from_numpy(x).float()).numpy().astype(np.float64)
    classes = error.argmin(-1)
    means, grams = np.zeros((19, 48)), np.zeros((19, 48, 48))
    for category in range(19):
        group = np.abs(x[classes == category])
        if len(group):
            means[category] = group.mean(0)
            grams[category] = group.T @ group / len(group)
    means /= means.sum(0) + 1e-8
    np.testing.assert_allclose(model.cssr_mean.numpy(), means, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.cssr_gram.numpy(), grams, rtol=1e-12, atol=1e-12)
    x = np.concatenate([x[1:].astype(np.float32) for x in augmented]).astype(np.float64)
    error = errors(torch.from_numpy(x).float()).numpy().astype(np.float64)
    classes, values = error.argmin(-1), np.abs(x)
    support = np.column_stack((-error.min(-1) / (values.sum(-1) ** 2 + 1e-8),
        (values * means[classes]).sum(-1), np.einsum("ni,nij,nj->n", values, grams[classes], values)))
    np.testing.assert_allclose(model.cssr_location.numpy(), support.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.cssr_scale.numpy(), support.std(0), rtol=1e-12, atol=1e-12)
    assert result["unaugmented_points"] == 9 and result["augmented_points"] == 9
    actual = model.cssr_score(torch.from_numpy(x).float(), torch.from_numpy(error).float())
    expected = -((support - support.mean(0)) / (support.std(0) + 1e-8)).sum(-1)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-6)


def test_semantic_competition_density_derivative_matches_allowed_set_formula():
    log_density = torch.linspace(-2, 1, 19, dtype=torch.float64)[None].requires_grad_()
    appearance = torch.linspace(0, 2, 19, dtype=torch.float64)[None]
    allowed = torch.zeros(1, 19, dtype=torch.bool)
    allowed[:, [2, 6]] = True
    prediction = dict(weight=torch.zeros(1, 19, 1, dtype=torch.float64),
                      log_prob=log_density[..., None], supported=torch.ones(1, dtype=torch.bool))
    energy = appearance + geometry_energy(prediction)
    allowed_loss(-energy, allowed).backward()
    probability = (-energy).softmax(-1)
    conditional = probability * allowed / (probability * allowed).sum(-1, keepdim=True)
    torch.testing.assert_close(log_density.grad, probability - conditional, rtol=1e-12, atol=1e-12)
    assert float(LOG_RETURN_PEAK) > 1


def test_retrained_loss_ablations_remove_only_the_specified_term(models):
    reference = models().loss_weights()
    for variant, term in (("no_nll", "geometry"), ("no_compactness", "normal")):
        expected = dict(reference)
        expected[term] = 0.
        assert models(variant).loss_weights() == expected
    for variant in ("target_available", "single_component", "separate"):
        assert models(variant).loss_weights() == reference
