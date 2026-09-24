"""Gaussian ray mathematics, target exclusion and full-batch gradient semantics."""

from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfcx, log_ndtr
import torch
from torch import nn
from torch.nn import functional as F

from src.normal import (NormalField, Compatibility, angular_observation, joint_nll,
                        ray_parameters, ray_log_prob, log_normal_mass, SCALES, LOWER, UPPER,
                        HYPOTHESES, KERNELS, LogNormalCDF, log_event_ratio)
from src.model import Segmentor, balanced_loss, ranking_loss, to_device
from src.train import cached_backward, seed_all, rng_state, restore_rng


def angular_scan():
    az, el = np.meshgrid(np.arange(-9.7, 10., .7), np.arange(-8.3, 3., .8))
    az, el = np.deg2rad(az.ravel()), np.deg2rad(el.ravel())
    rays = np.column_stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))
    ranges = np.linspace(8., 24., len(az))
    return np.column_stack((rays * ranges[:, None], np.full(len(az), .4))).astype(np.float32), rays


@pytest.mark.parametrize("size", SCALES)
def test_field_excludes_entire_target_block_values_counts_and_gradients(size):
    xyzi, rays = angular_scan()
    first = angular_observation(xyzi, directions=rays)
    grid = first["grids"][str(size)]
    group = int(grid["group"][len(xyzi) // 2])
    target = (grid["group"] == group).numpy()
    changed = xyzi.copy()
    changed[target, :3] *= np.linspace(.2, 7., target.sum())[:, None]
    changed[target, 3] = 1000.
    second = angular_observation(changed, directions=rays)
    seed_all(7)
    model = NormalField(recompute=False)
    a, b = model(first)[str(size)], model(second)[str(size)]
    for name in a:
        torch.testing.assert_close(a[name][group], b[name][group], atol=0, rtol=0)
    # Repeating target points changes its population but cannot change its field.
    repeated = angular_observation(np.concatenate((changed, changed[target])),
                                   directions=np.concatenate((rays, rays[target])))
    c = model(repeated)[str(size)]
    for name in a:
        # A changed matrix batch shape can change FP32 GEMM rounding. The value
        # intervention above is bitwise invariant; the gradient check is exact.
        torch.testing.assert_close(a[name][group], c[name][group], atol=1e-5, rtol=2e-6)
    first["features"].requires_grad_()
    predicted = model(first)[str(size)]
    sum(value[group].square().sum() for value in predicted.values()).backward()
    assert torch.count_nonzero(first["features"].grad[target]) == 0
    assert first["features"].grad[~target].abs().sum() > 0


def test_nested_grids_wrap_missing_context_and_augmentation_consistency():
    angle = np.deg2rad([179.7, -179.7])
    xyzi = np.column_stack((10 * np.cos(angle), 10 * np.sin(angle), [0., 0.], [.1, .2])).astype(np.float32)
    o = angular_observation(xyzi)
    for grid in o["grids"].values():
        assert grid["neighbors"].shape == (2, 24)
        assert not (grid["neighbors"] == torch.arange(2)[:, None]).any()
        assert (grid["neighbors"][0] == 1).any()
    xyzi[:, :3] = xyzi[:, [1, 0, 2]]
    with pytest.raises(ValueError, match="rebuild after augmentation"):
        angular_observation(xyzi, directions=o["directions"].numpy())
    lone = angular_observation(xyzi[:1])
    field = NormalField()
    outputs = field(lone)
    sum(value.square().sum() for f in outputs.values() for value in f.values()).backward()
    assert field.empty.grad.abs().sum() > 0
    assert all(torch.isfinite(value).all() for f in outputs.values() for value in f.values())


def test_analytic_ray_field_matches_independent_3d_and_quadrature():
    torch.manual_seed(12)
    center = torch.randn(1, 2, 3, 3, dtype=torch.float64) * 4 + torch.tensor([12., 0., 0.])
    a = torch.randn(1, 2, 3, 3, 3, dtype=torch.float64)
    covariance = a @ a.transpose(-1, -2) + 2 * torch.eye(3)
    chol = torch.linalg.cholesky(covariance)
    inverse = torch.linalg.inv(chol)
    log_amp = torch.randn(1, 2, 3, dtype=torch.float64) - 3
    field = dict(center=center, inverse=inverse, log_amplitude=log_amp)
    origin = torch.tensor([[.5, -.2, .1]], dtype=torch.float64)
    direction = F.normalize(torch.tensor([[1., .1, -.04]], dtype=torch.float64), dim=1)
    mu, tau, log_h = ray_parameters(field, torch.zeros(1, dtype=torch.long), origin, direction)
    r = torch.tensor([11.3], dtype=torch.float64)
    delta = origin[:, None, None] + r[:, None, None, None] * direction[:, None, None] - center
    direct = log_amp - .5 * torch.einsum('bkmi,bkmij,bkmj->bkm', delta, torch.linalg.inv(covariance), delta)
    torch.testing.assert_close(log_h - .5 * ((r[:, None, None] - mu) / tau).square(), direct, rtol=1e-12, atol=1e-12)
    expected = []
    for k in range(2):
        def rate(t):
            return sum(math.exp(float(log_h[0, k, m]) - .5 * ((t - float(mu[0, k, m])) / float(tau[0, k, m]))**2)
                       for m in range(3))
        total = quad(rate, LOWER, UPPER, epsabs=1e-12)[0]
        partial = quad(rate, LOWER, float(r), epsabs=1e-12)[0]
        expected.append(math.log(rate(float(r))) - partial - math.log(-math.expm1(-total)))
    actual = ray_log_prob(mu, tau, log_h, r)[0]
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(actual, ray_log_prob(mu.flip(-1), tau.flip(-1), log_h.flip(-1), r)[0])


@pytest.mark.parametrize("log_amplitude", [-1000., -20., -4., 1.])
def test_conditional_density_normalizes_with_overlap_and_tiny_amplitude(log_amplitude):
    mu = torch.tensor([[[10., 20., 22.], [6., 25., 40.]]], dtype=torch.float64)
    tau = torch.tensor([[[2., 4., 3.], [1., 10., 2.]]], dtype=torch.float64)
    h = torch.full_like(mu, log_amplitude)
    weights = np.array([.3, .7])
    def density(r):
        value = ray_log_prob(mu, tau, h, torch.tensor([r], dtype=torch.float64))[0].exp().numpy()
        return value @ weights
    assert abs(quad(density, LOWER, UPPER, epsabs=2e-9, limit=200)[0] - 1) < 2e-8


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_fp32_tail_mass_density_and_gradients_do_not_collapse(device):
    mu = torch.tensor([[[-100., 130.], [12., 15.]]], device=device, requires_grad=True)
    tau = torch.tensor([[[1.3, .8], [4., 2.]]], device=device, requires_grad=True)
    h = torch.tensor([[[-1000., -1002.], [-1000., -1001.]]], device=device, requires_grad=True)
    values = ray_log_prob(mu, tau, h, torch.tensor([20.], device=device))
    reference = ray_log_prob(mu.detach().double(), tau.detach().double(), h.detach().double(),
                             torch.tensor([20.], device=device, dtype=torch.float64)).float()
    torch.testing.assert_close(values, reference, atol=.008, rtol=2e-5)
    assert torch.isfinite(values).all() and values[0, 0] != values[0, 1]
    values.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in (mu, tau, h))
    assert mu.grad.abs().sum() > 0 and tau.grad.abs().sum() > 0
    # Very narrow far-tail intervals require a local integral, not CDF subtraction.
    mass = log_normal_mass(torch.tensor([0.], device=device), torch.tensor([1.], device=device), 100., 100.00001)
    expected = -.5 * 100.000005**2 - .5 * math.log(2 * math.pi) + math.log(.00001)
    assert abs(float(mass) - expected) < .002


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_logcdf_extreme_tail_keeps_values_and_correct_derivatives(device):
    x = torch.tensor([-1e8, -1e5, -1e4, -100., -20., -3., 0., 3., 10.], device=device, requires_grad=True)
    actual = LogNormalCDF.apply(x)
    torch.testing.assert_close(actual, torch.special.log_ndtr(x), atol=0, rtol=0)
    actual.sum().backward()
    reference = x.detach().cpu().double().numpy()
    expected = np.empty_like(reference)
    negative = reference < 0
    expected[negative] = math.sqrt(2 / math.pi) / erfcx(-reference[negative] / math.sqrt(2))
    expected[~negative] = np.exp(-.5 * reference[~negative]**2 - log_ndtr(reference[~negative]) - .5 * math.log(2 * math.pi))
    np.testing.assert_allclose(x.grad.cpu().numpy(), expected, rtol=2e-6, atol=0)
    assert torch.autograd.gradcheck(LogNormalCDF.apply,
        (torch.tensor([-20., -3., 0., 3.], dtype=torch.float64, device=device, requires_grad=True),),
        eps=1e-5, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_interval_mass_extreme_tail_has_finite_analytic_gradients(device):
    mu = torch.tensor([100., 1e6], device=device, requires_grad=True)
    tau = torch.tensor([1e-4, 1.], device=device, requires_grad=True)
    mass = log_normal_mass(mu, tau, LOWER, UPPER)
    mass.sum().backward()
    a = (LOWER - mu.detach().cpu().double().numpy()) / tau.detach().cpu().double().numpy()
    b = (UPPER - mu.detach().cpu().double().numpy()) / tau.detach().cpu().double().numpy()
    ratio = np.exp(log_ndtr(a) - log_ndtr(b))
    ra, rb = (math.sqrt(2 / math.pi) / erfcx(-z / math.sqrt(2)) for z in (a, b))
    scale = tau.detach().cpu().double().numpy() * (1 - ratio)
    np.testing.assert_allclose(mu.grad.cpu().numpy(), (ratio * ra - rb) / scale, rtol=2e-6)
    np.testing.assert_allclose(tau.grad.cpu().numpy(), (a * ratio * ra - b * rb) / scale, rtol=2e-6)
    assert torch.isfinite(mass).all()
    # A distant narrow interval must not form middle**4 * width**4 (inf * 0).
    center = torch.tensor([-1e30], device=device, requires_grad=True)
    spread = torch.tensor([1e15], device=device, requires_grad=True)
    value = log_normal_mass(center, spread, 0., 1e-5)
    value.sum().backward()
    assert all(torch.isfinite(t).all() for t in (value, center.grad, spread.grad))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_event_normalizer_large_rate_retains_value_and_gradient(device):
    values = [-1000., -20., -5., -1., 0., 1., 5., 89., 100., 1000.]
    x = torch.tensor(values, device=device, requires_grad=True)
    result = log_event_ratio(x)
    result.sum().backward()
    expected, derivatives = [], []
    for value in values:
        if value > math.log(50):
            expected.append(-value); derivatives.append(-1.)
        else:
            rate = math.exp(value)
            expected.append(math.log(-math.expm1(-rate)) - value if rate else 0.)
            derivatives.append(-rate / 2 + rate**2 / 12 if rate < .01 else rate / math.expm1(rate) - 1)
    np.testing.assert_allclose(result.detach().cpu().numpy(), expected, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(x.grad.cpu().numpy(), derivatives, atol=1e-7, rtol=2e-6)


def test_joint_nll_shares_hypothesis_and_weights_blocks_equally():
    prob = torch.tensor([[.9, .01], [.01, .9], [.7, .3]]).log().requires_grad_()
    weights = torch.tensor([[.4, .6], [.2, .8]]).log()
    grid = dict(order=torch.arange(3), pointer=torch.tensor([0, 2, 3]))
    selected = torch.ones(3, dtype=torch.bool)
    expected = (-math.log(.009) / 2 - math.log(.2 * .7 + .8 * .3)) / 2
    actual = joint_nll(prob, weights, grid, selected)
    assert abs(float(actual.detach()) - expected) < 1e-6
    actual.backward()
    assert (prob.grad.abs().sum(1) > 0).all()
    assert actual > -(prob.exp() * weights.exp()[torch.tensor([0, 0, 1])]).sum(1).log().mean()


def test_point_compatibility_never_reweights_prior_with_target_peer_ranges():
    xyzi, _ = angular_scan()
    observation = angular_observation(xyzi)
    fields = NormalField()(observation)
    decoder = Compatibility()
    state = torch.randn(len(xyzi), 64)
    before, _ = decoder(state, observation, fields, 0, len(xyzi))
    changed = deepcopy(observation)
    changed["distance"][0] *= 1.8
    after, _ = decoder(state, changed, fields, 0, len(xyzi))
    torch.testing.assert_close(before[1:], after[1:], atol=0, rtol=0)
    assert not torch.equal(before[0], after[0])
    after.square().mean().backward()
    gradients = [p.grad for p in decoder.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


@pytest.mark.parametrize("permuted", ["kernels", "hypotheses"])
def test_compatibility_is_invariant_to_kernel_and_hypothesis_names(permuted):
    seed_all(43)
    xyzi, _ = angular_scan()
    observation = angular_observation(xyzi[:32])
    fields = NormalField(recompute=False)(observation)
    decoder = Compatibility().eval()
    state = torch.randn(32, 64)
    permutation = torch.randperm(KERNELS if permuted == "kernels" else HYPOTHESES)
    reordered = {}
    for size, field in fields.items():
        reordered[size] = {
            name: value[:, permutation] if permuted == "hypotheses"
            else value if name == "log_weights" else value[:, :, permutation]
            for name, value in field.items()}
    before, probability = decoder(state, observation, fields, 0, len(state))
    after, changed_probability = decoder(state, observation, reordered, 0, len(state))
    torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-6)
    expected = probability[..., permutation] if permuted == "hypotheses" else probability
    torch.testing.assert_close(changed_probability, expected, atol=2e-6, rtol=2e-6)


def test_compatibility_distinguishes_hypothesis_grouping_at_equal_marginal_density():
    seed_all(29)
    middle = (LOWER + UPPER) / 2
    observation = angular_observation(np.array([[middle, 0., 0., .4]], dtype=np.float32))
    center = torch.zeros(1, HYPOTHESES, KERNELS, 3)
    center[:, :HYPOTHESES // 2, :, 0] = middle - 8
    center[:, HYPOTHESES // 2:, :, 0] = middle + 8
    field = dict(center=center, inverse=torch.eye(3).expand(1, HYPOTHESES, KERNELS, 3, 3) / 4,
                 log_amplitude=torch.full((1, HYPOTHESES, KERNELS), -20.),
                 log_weights=torch.full((1, HYPOTHESES), -math.log(HYPOTHESES)))
    count = HYPOTHESES * KERNELS
    order = torch.stack((torch.arange(count // 2), torch.arange(count // 2, count)), 1).flatten()
    # The same kernel multiset becomes four identical near/far hypotheses.
    regrouped = {name: value if name == "log_weights"
                 else value.flatten(1, 2)[:, order].reshape_as(value) for name, value in field.items()}
    fields = {str(size): field for size in SCALES}
    changed = {str(size): regrouped for size in SCALES}
    state, decoder = torch.zeros(1, 64), Compatibility().eval()
    before, probability = decoder(state, observation, fields, 0, 1)
    after, changed_probability = decoder(state, observation, changed, 0, 1)
    # Symmetry and the low-rate limit keep density evidence equal to FP32 tolerance.
    # A flat kernel pool cannot distinguish these sets; the learned features must.
    torch.testing.assert_close(probability, changed_probability, atol=1e-6, rtol=0)
    torch.testing.assert_close(before[:, -len(SCALES):], after[:, -len(SCALES):], atol=1e-6, rtol=0)
    assert (before[:, 64:-len(SCALES)] - after[:, 64:-len(SCALES)]).abs().max() > 1e-5


def test_compatibility_chunks_preserve_scores_and_outside_support_has_no_density(monkeypatch):
    seed_all(61)
    distances = np.array([1., LOWER, 8., 15., UPPER, 60.], dtype=np.float32)
    xyzi = np.column_stack((distances, np.zeros((len(distances), 2)), np.full(len(distances), .4)))
    observation = angular_observation(xyzi.astype(np.float32))
    fields = NormalField(recompute=False)(observation)
    decoder, state = Compatibility().eval(), torch.randn(len(distances), 64)
    whole, probability = decoder(state, observation, fields, 0, len(state))
    parts = [decoder(state[start:start + 2], observation, fields, start, start + 2)
             for start in range(0, len(state), 2)]
    torch.testing.assert_close(torch.cat([part[0] for part in parts]), whole, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(torch.cat([part[1] for part in parts]), probability, atol=2e-6, rtol=2e-6)
    assert whole.shape == (len(distances), 163) and probability.shape == (len(distances), len(SCALES), HYPOTHESES)
    torch.testing.assert_close(whole[:, :64], state, atol=0, rtol=0)
    assert torch.isfinite(whole).all() and torch.isfinite(probability).all()
    assert torch.count_nonzero(whole[[0, -1], -len(SCALES):]) == 0
    def changed_boundary_density(mu, tau, log_h, measured):
        probability = ray_log_prob(mu, tau, log_h, measured)
        boundary = (measured == LOWER) | (measured == UPPER)
        return probability + boundary[:, None] * (7 + 2 * torch.arange(HYPOTHESES))
    monkeypatch.setattr("src.normal.ray_log_prob", changed_boundary_density)
    altered, _ = decoder(state, observation, fields, 0, len(state))
    # An out-of-range point is internally evaluated at a clipped endpoint. Even
    # changing that density must leave both its learned and scalar evidence intact.
    torch.testing.assert_close(altered[[0, -1]], whole[[0, -1]], atol=0, rtol=0)
    assert not torch.equal(altered[[1, -2], -len(SCALES):], whole[[1, -2], -len(SCALES):])


@pytest.mark.parametrize("sharp_and_low_prior", [False, True])
def test_detection_gradient_through_hypothesis_features_reaches_field_and_prior(sharp_and_low_prior):
    seed_all(83)
    xyzi, _ = angular_scan()
    observation = angular_observation(xyzi[:32])
    model, decoder = NormalField(recompute=False), Compatibility()
    fields = model(observation)
    for field in fields.values():
        if sharp_and_low_prior:
            field["inverse"] = field["inverse"] * 1000
            field["log_weights"] = (field["log_weights"] - 1000 * torch.arange(HYPOTHESES)).log_softmax(-1)
        for value in field.values():
            value.retain_grad()
    hidden, _ = decoder(torch.randn(32, 64), observation, fields, 0, 32)
    assert torch.isfinite(hidden).all()
    # Isolate the learned hypothesis path: neither actual features nor marginal
    # log densities can supply this classification gradient.
    score = nn.Linear(32 * len(SCALES), 1)(hidden[:, 64:-len(SCALES)]).flatten()
    loss = F.binary_cross_entropy_with_logits(score, (torch.arange(32) % 2).float())
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("center", "inverse", "log_amplitude", "log_weights"):
        gradients = [field[name].grad for field in fields.values()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients), name
        assert sum(float(g.abs().sum()) for g in gradients) > 0, name
    for module in (model.encoder, model.parameters_out, model.weights, decoder):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_clean_companion_likelihood_matches_shared_prediction_and_gradients():
    xyzi, _ = angular_scan()
    observation = angular_observation(xyzi)
    targets = torch.zeros(len(xyzi), dtype=torch.long)
    targets[::5] = -1
    seed_all(17)
    model = NormalField()
    reference = deepcopy(model)
    reference.recompute = False
    fields = reference(observation)
    _, probabilities = Compatibility()(torch.zeros(len(xyzi), 64), observation, fields, 0, len(xyzi))
    losses = []
    for scale, size in enumerate(SCALES):
        group = observation["grids"][str(size)]["group"]
        blocks = []
        for index in torch.unique(group[targets == 0]):
            selected = (group == index) & (targets == 0)
            # One hypothesis explains all selected rays before marginalization.
            joint = fields[str(size)]["log_weights"][index] + probabilities[selected, scale].sum(0)
            blocks.append(-joint.logsumexp(0) / selected.sum())
        losses.append(torch.stack(blocks).mean())
    expected = torch.stack(losses).mean()
    actual = model.likelihood(observation, targets)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    expected.backward()
    actual.backward()
    for parameter, original in zip(model.parameters(), reference.parameters()):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, original.grad, atol=2e-6, rtol=2e-4)
    assert model.encoder[0].weight.grad.abs().sum() > 0


def test_field_preserves_actual_path_initialization_and_has_fresh_fp32_head():
    seed_all(31)
    baseline = Segmentor("conditional")
    random = torch.get_rng_state()
    seed_all(31)
    candidate = Segmentor()
    assert torch.equal(torch.get_rng_state(), random)
    for key, value in baseline.state_dict().items():
        if not key.startswith("head."):
            assert torch.equal(value, candidate.state_dict()[key]), key
    assert candidate.head[0].in_features == 163
    extra = sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in baseline.parameters())
    assert 0 < extra < 2_000_000
    with pytest.raises(ValueError, match="retired"):
        Segmentor("geometry")


class CacheModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 12), nn.BatchNorm1d(12), nn.GELU(), nn.Dropout(.25))
        self.normal = nn.Linear(12, 4)
        self.head = nn.Linear(16, 1)

    def forward(self, sample, *, normal_loss=False):
        x = self.encoder(sample["xyzi"])
        normal = self.normal(x)
        score = self.head(torch.cat((x, normal), 1)).flatten()
        if "normal_reference" in sample:
            auxiliary = self.normal(F.pad(sample["normal_reference"]["xyzi"], (0, 8))).square().mean()
        else:
            auxiliary = normal.square().mean() if sample["normal_training"] else normal.sum() * 0
        return (score, auxiliary) if normal_loss else score


@pytest.mark.parametrize("scans,all_normal,paired", [(8, False, False), (3, False, False),
                                                   (8, True, False), (8, False, True)])
def test_gradient_cache_equals_joint_graph_with_bn_dropout_and_one_rng_advance(scans, all_normal, paired):
    seed_all(71)
    model = CacheModel().train()
    reference = deepcopy(model)
    samples = [dict(xyzi=torch.randn(15 + i, 4), targets=torch.zeros(15 + i, dtype=torch.long),
                    normal_training=i % 3 == 0) for i in range(scans)]
    if not all_normal:
        for i, sample in enumerate(samples):
            if i % 3:
                sample["targets"][i % 5] = 1
            sample["targets"][-1] = -1
    if paired:
        for sample in samples:
            sample["normal_reference"] = dict(xyzi=sample["xyzi"] + .3, normal_training=True,
                targets=torch.zeros_like(sample["targets"]))
            sample["normal_training"] = False
    device = torch.device("cpu")
    start = rng_state(device)
    outputs = [reference(sample, normal_loss=True) for sample in samples]
    scores = torch.cat([value[0] for value in outputs])
    targets = torch.cat([s["targets"] for s in samples])
    counts = torch.stack([(targets == label).sum() for label in (0, 1)])
    rank, detail = ranking_loss(scores, targets, 19)
    normal_count = sum(s.get("normal_reference", s)["normal_training"] for s in samples)
    normal = sum(v[1] for v in outputs) / normal_count
    loss = balanced_loss(scores, targets, counts) + rank + .1 * normal
    loss.backward()
    end = rng_state(device)
    restore_rng(start, device)
    cached, info = cached_backward(model, samples, device, rank_weight=1., rank_seed=19)
    torch.testing.assert_close(cached, loss.detach())
    assert info["ranking_scans"] == scans and info["replay_max_abs"] == 0
    assert info["normal_scans"] == normal_count
    if not all_normal:
        torch.testing.assert_close(info["threshold"], detail["threshold"])
    else:
        assert info["threshold"] is None
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-7, rtol=3e-5)
    for a, b in zip(model.buffers(), reference.buffers()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), end["torch"])


def test_infer_preserves_actual_slots_and_builds_multiscale_observation(monkeypatch):
    import src.evaluate as evaluation
    xyzi, _ = angular_scan()
    raw = np.zeros((2 * len(xyzi), 4), np.float32)
    raw[::2] = xyzi
    frame = SimpleNamespace(xyzi=raw, actual=np.arange(len(raw)) % 2 == 0,
                            return_slots=np.arange(0, len(raw), 2), frame_id=0)
    def read_scan(path, *, io_timing):
        io_timing["seconds"] = 0.
        return frame
    monkeypatch.setattr(evaluation, "read_scan", read_scan)
    class Score(nn.Module):
        normal = True
        def forward(self, sample):
            assert set(sample["observation"]["grids"]) == {"1", "2", "4"}
            return sample["observation"]["distance"]
    prediction, _ = evaluation.infer(Score(), "unused", torch.device("cpu"))
    assert not prediction[1::2].any()
    np.testing.assert_allclose(prediction[::2], np.linalg.norm(xyzi[:, :3], axis=1), rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="full LitePT requires CUDA")
def test_real_complete_model_gradients_maximum_pair_and_checkpoint_reload(tmp_path):
    """Implementation evidence on real scans, not a trained detection benchmark."""
    from pathlib import Path
    import subprocess
    import sys
    import time
    from src.data import load_manifest, SOURCE_VERSION
    from src.evaluate import PreparedScans, autocast, load_model
    from src.model import balanced_loss
    from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator
    manifest_path = Path("results/data/nuscenes/train.json")
    if not manifest_path.exists() or not Path("assets/nuscenes.pth").exists():
        pytest.skip("local real data and official initialization are required")
    device = torch.device("cuda")
    manifest = load_manifest(manifest_path, "train")
    assert manifest["version"] == SOURCE_VERSION
    data = PreparedScans(manifest, normal=True)
    indices = [max((i for i, row in enumerate(manifest["records"]) if row["group"] == group
                    and (group == "normal_nuscenes" or row["anomaly"] >= 5)),
                   key=lambda i: manifest["records"][i]["points"])
               for group in ("anomaly_nuscenes", "normal_nuscenes")]
    samples = [data[i] for i in indices]
    assert [s["normal_training"] for s in samples] == [False, True]
    assert samples[0]["normal_reference"]["normal_training"] and "normal_reference" not in samples[1]
    assert all((s.get("normal_reference", s)["targets"] == 0).any() for s in samples)
    seed_all(17)
    model = Segmentor().to(device).train()
    model.load_pretrained("assets/nuscenes.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    sample = to_device(samples[0], device)
    with autocast(device):
        score = model(sample)
    assert score.dtype == torch.float32
    counts = torch.stack([(sample["targets"] == label).sum() for label in (0, 1)])
    balanced_loss(score, sample["targets"], counts).backward()
    for module in (model.normal.encoder, model.normal.parameters_out, model.normal.weights,
                   model.compatibility, model.head, model.backbone):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    optimizer.zero_grad(set_to_none=True)
    del score, sample
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss, detail = cached_backward(model, samples, device, rank_weight=1., rank_seed=73)
    assert detail["normal_scans"] == 2 and detail["replay_max_abs"] <= 1e-5
    assert torch.isfinite(loss)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(norm) and norm > 0
    previous = model.normal.parameters_out.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(previous, model.normal.parameters_out.weight)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    del optimizer, detail
    model.eval()
    sample = to_device(samples[0], device)
    with torch.no_grad(), autocast(device):
        before = model(sample).cpu()
    assert torch.isfinite(before).all() and len(before) == len(sample["xyzi"])
    path = tmp_path / "field.pt"
    torch.save(dict(version=manifest["version"], mode="field", model=model.state_dict()), path)
    del model
    torch.cuda.empty_cache()
    restored, _ = load_model(path, device)
    with torch.no_grad(), autocast(device):
        after = restored(sample).cpu()
    # Bounds are set before this real execution: absolute logits 1e-5 + rtol 1e-5,
    # and each official percentage-valued metric within 1e-5 percentage points.
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    del restored, sample, module
    torch.cuda.empty_cache()
    cold_path = tmp_path / "cold.npy"
    subprocess.run([sys.executable, "-c", '''
import sys, numpy as np, torch
from pathlib import Path
from src.data import load_manifest
from src.evaluate import PreparedScans, autocast, load_model
from src.model import to_device
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
device = torch.device("cuda")
model, _ = load_model(Path(sys.argv[1]), device)
data = PreparedScans(load_manifest(Path(sys.argv[2]), "train"), normal=True, normal_reference=False)
sample = to_device(data[int(sys.argv[3])], device)
with torch.no_grad(), autocast(device):
    scores = model(sample).cpu().numpy()
np.save(sys.argv[4], scores)
''', str(path), str(manifest_path), str(indices[0]), str(cold_path)], check=True)
    cold = torch.from_numpy(np.load(cold_path))
    torch.testing.assert_close(before, cold, atol=1e-5, rtol=1e-5)
    targets = samples[0]["targets"].numpy()
    valid = targets >= 0
    metrics = []
    for scores in (before, after, cold):
        calculator = PointOODMetricsCalculator()
        calculator.all_scores = [scores.numpy()[valid]]
        calculator.all_labels = [targets[valid]]
        metrics.append(calculator.compute_metrics())
    delta = {key: max(abs(metrics[0][key] - other[key]) for other in metrics[1:])
             for key in ("AP", "AUROC", "FPR95")}
    assert max(delta.values()) <= 1e-5
    print(dict(real_indices=indices, real_points=[len(s["xyzi"]) for s in samples],
               maximum_pair_seconds=seconds, peak_vram_bytes=peak,
               reload_max_abs=max(float((before - other).abs().max()) for other in (after, cold)), metric_deltas=delta))
    path.unlink()
    cold_path.unlink()


def test_predictive_quantiles_match_truncated_exponential_and_low_rate_limit():
    from src.normal import ray_quantiles
    probabilities = np.array([.05, .5, .95])
    mu, tau = torch.zeros(2, 1, 1), torch.full((2, 1, 1), 1e8)
    h = torch.tensor([[[math.log(.04)]], [[-1000.]]])
    intervals = ray_quantiles(mu, tau, h, torch.zeros(2, 1))
    expected = np.stack((LOWER - np.log1p(-probabilities * -np.expm1(-.04 * (UPPER - LOWER))) / .04,
                         LOWER + probabilities * (UPPER - LOWER)))
    np.testing.assert_allclose(intervals.numpy(), expected, atol=1e-5, rtol=2e-6)


def test_normal_diagnostics_use_all_valid_normals_and_reject_synthetic_targets():
    from src.normal import prediction_metrics
    xyzi, _ = angular_scan()
    xyzi = xyzi[:12]
    sample = dict(observation=angular_observation(xyzi), targets=torch.tensor([0] * 10 + [-1, -1]), normal_training=True)
    result = prediction_metrics(NormalField().eval(), sample)
    assert result["points"] == 10 and set(result["scales"]) == {"1", "2", "4"}
    for values in result["scales"].values():
        assert all(math.isfinite(v) for v in values.values())
        assert 0 <= values["coverage90"] <= 1 and 0 < values["width90_m"] < UPPER - LOWER
    sample["normal_training"] = False
    with pytest.raises(ValueError, match="unmodified real-normal"):
        prediction_metrics(NormalField(), sample)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sparse indices require CUDA")
@pytest.mark.parametrize("kernel", [3, 5])
@pytest.mark.parametrize("isolated", [False, True])
def test_ordered_sparse_convolution_matches_dense_values_and_gradients(kernel, isolated):
    from vendor.litept.model import SubMConv3d
    import spconv.pytorch as spconv

    generator = torch.Generator().manual_seed(912)
    if isolated:
        coordinates = [[0, 2, 2, 2]]
    else:
        # Holes, boundaries, mirrored offsets and separate batches expose index errors.
        coordinates = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                       [0, 1, 1, 1], [0, 2, 0, 0], [0, 0, 0, 2], [0, 3, 2, 1],
                       [0, 3, 2, 3], [0, 4, 4, 4], [1, 0, 0, 0], [1, 1, 0, 0],
                       [1, 3, 3, 3], [1, 4, 3, 3]]
    indices = torch.tensor(coordinates, dtype=torch.int32, device="cuda")
    batches, width = 1 if isolated else 2, 5
    features = torch.randn(len(indices), 3, generator=generator).cuda().requires_grad_()
    # Submanifold indexing centers the kernel, including the backbone's original padding values.
    padding = 1 if kernel == 5 else 0
    layers = [SubMConv3d(3, 4, kernel, padding=padding, bias=True, indice_key="shared").cuda(),
              SubMConv3d(4, 2, kernel, padding=padding, bias=True, indice_key="shared").cuda()]
    for layer in layers:
        with torch.no_grad():
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator).to("cuda") * .1)
            layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator).to("cuda") * .1 + .2)
    reference_input = features.detach().double().requires_grad_()
    reference_features = reference_input
    reference_parameters = [(layer.weight.detach().double().requires_grad_(),
                             layer.bias.detach().double().requires_grad_()) for layer in layers]
    sparse = spconv.SparseConvTensor(features, indices, [width] * 3, batches)
    flat = ((indices[:, 0].long() * width + indices[:, 1]) * width + indices[:, 2]) * width + indices[:, 3]

    def dense_reference(values, weight, bias):
        # Reset inactive sites between layers: submanifold convolution keeps only active outputs.
        dense = values.new_zeros((batches * width ** 3, values.shape[1])).index_copy(0, flat, values)
        dense = dense.reshape(batches, width, width, width, values.shape[1]).permute(0, 4, 1, 2, 3)
        result = F.conv3d(dense, weight.permute(0, 4, 1, 2, 3).contiguous(), bias, padding=kernel // 2)
        return result.permute(0, 2, 3, 4, 1).reshape(-1, weight.shape[0])[flat]

    actual, expected = [], []
    cached = None
    for layer, (weight, bias) in zip(layers, reference_parameters):
        sparse = layer(sparse)
        torch.testing.assert_close(sparse.indices, indices, atol=0, rtol=0)
        if cached is None:
            cached = dict(sparse.indice_dict)
            assert cached
        else:
            assert sparse.indice_dict.keys() == cached.keys()
            assert all(sparse.indice_dict[key] is value for key, value in cached.items())
        reference_features = dense_reference(reference_features, weight, bias)
        torch.testing.assert_close(sparse.features.double(), reference_features, atol=1e-5, rtol=1e-5)
        probe = torch.randn(sparse.features.shape, generator=generator).cuda()
        actual.append((sparse.features * probe).sum())
        expected.append((reference_features * probe.double()).sum())
    # The independent dense graph uses FP64; gradients verify both index direction and weight layout.
    sum(actual).backward()
    sum(expected).backward()
    assert torch.isfinite(features.grad).all()
    torch.testing.assert_close(features.grad.double(), reference_input.grad, atol=1e-5, rtol=1e-5)
    for layer, (weight, bias) in zip(layers, reference_parameters):
        for observed, reference in ((layer.weight.grad, weight.grad), (layer.bias.grad, bias.grad)):
            assert torch.isfinite(observed).all()
            torch.testing.assert_close(observed.double(), reference, atol=1e-5, rtol=1e-5)
