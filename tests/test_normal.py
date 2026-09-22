"""Leakage, shared-hypothesis likelihood, and point-local evidence regressions."""

from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from src.normal import (NORMAL_MODES, NormalDistribution, angular_observation, component_log_prob,
                        joint_nll, point_evidence, quantiles, ray_planes)
from src.model import Segmentor, to_device
from src.train import seed_all


def angular_scan():
    az = np.deg2rad(np.array([.3, .7, 1.3, 1.7, 2.3, 2.7, 3.3, 3.7, -1.7, -1.3, -.7, -.3]))
    el = np.deg2rad(np.full(len(az), -5.))
    rays = np.column_stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))
    ranges = np.linspace(8., 14., len(az))
    xyzi = np.column_stack((rays * ranges[:, None], np.full(len(az), .4))).astype(np.float32)
    return xyzi, rays


@pytest.mark.parametrize("mode", NORMAL_MODES)
def test_prediction_excludes_all_ranges_coordinates_and_intensities_in_target_block(mode):
    xyzi, rays = angular_scan()
    first = angular_observation(xyzi, directions=rays)
    group = first["group"][0]
    target = first["group"] == group
    changed = xyzi.copy()
    changed[target, :3] *= np.array([.2, 7., 3., .1])[:, None]
    changed[target, 3] = 1000.
    second = angular_observation(changed, directions=rays)
    seed_all(7)
    model = NormalDistribution(mode)
    # Activate every path; zero-initialized output weights could hide leakage.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * .03)
    a, b = model(first), model(second)
    for name in ("mu", "scale", "valid"):
        assert torch.equal(a[name][target], b[name][target])
    assert torch.equal(a["log_weights"][group], b["log_weights"][group])
    first["features"].requires_grad_()
    first["log_range"].requires_grad_()
    predicted = model(first)
    objective = predicted["mu"][target].sum() + predicted["scale"][target].sum()
    objective = objective + predicted["log_weights"][group].square().sum()
    objective.backward()
    assert torch.count_nonzero(first["features"].grad[target]) == 0
    assert torch.count_nonzero(first["log_range"].grad[target]) == 0
    assert first["features"].grad[~target].abs().sum() > 0
    assert first["log_range"].grad[~target].abs().sum() > 0


def test_angular_blocks_use_only_rays_and_keep_azimuth_wrap_neighbors():
    xyzi, rays = angular_scan()
    a = angular_observation(xyzi, directions=rays)
    xyzi[:, :3] *= np.linspace(.01, 100., len(xyzi))[:, None]
    b = angular_observation(xyzi, directions=rays)
    for key in ("cells", "group", "order", "pointer", "neighbors", "relative"):
        assert torch.equal(a[key], b[key])
    assert not (a["neighbors"] == a["cells"][:, None]).any()
    # Changing the target's size cannot alter which outside rays provide context.
    keep = np.arange(len(xyzi)) != 0
    small = angular_observation(xyzi[keep], directions=rays[keep])
    assert torch.equal(a["neighbors"], small["neighbors"])


def distribution(mu, scale=.1, weights=None, group=None):
    mu = torch.as_tensor(mu, dtype=torch.float32)
    if weights is None:
        weights = torch.ones((1, mu.shape[1])) / mu.shape[1]
    return dict(mu=mu, scale=torch.full_like(mu, scale), valid=torch.ones_like(mu, dtype=torch.bool),
                log_weights=torch.as_tensor(weights).log(),
                group=torch.zeros(len(mu), dtype=torch.long) if group is None else group)


def test_joint_likelihood_marginalizes_one_shared_hypothesis_after_all_rays():
    prediction = distribution([[0., 2.], [0., 2.]], weights=[[.3, .7]])
    observations = torch.tensor([0., 2.])
    value = joint_nll(prediction, observations, torch.ones(2, dtype=torch.bool))
    # Each point fits a different mode. Independent pointwise mode selection
    # would falsely give this incompatible patch a high joint likelihood.
    p = torch.exp(component_log_prob(prediction, observations))
    expected = -torch.log(.3 * p[0, 0] * p[1, 0] + .7 * p[0, 1] * p[1, 1]) / 2
    independent = -torch.log((p * torch.tensor([.3, .7])).sum(1)).mean()
    torch.testing.assert_close(value, expected)
    assert value > independent + 5
    selected = torch.tensor([True, False])
    torch.testing.assert_close(joint_nll(prediction, observations, selected),
                               -torch.log((p[0] * torch.tensor([.3, .7])).sum()))


def test_physical_intersections_share_one_plane_across_directions_and_origins():
    normal = torch.tensor([1., .2, -.1]).expand(3, 1, 3)
    origin = torch.tensor([[0., 0., 0.], [.3, .1, 0.], [-.1, .2, .05]])
    direction = torch.nn.functional.normalize(torch.tensor([[1., 0., 0.], [1., .2, 0.], [1., -.1, .1]]), dim=1)
    mu, valid = ray_planes(torch.full((3, 1), 8.), normal, origin, direction)
    hits = origin + mu.exp() * direction
    assert valid.all()
    torch.testing.assert_close((hits * normal[:, 0]).sum(1), torch.full((3,), 8.))


def test_single_bad_return_is_scored_locally_not_by_broadcasting_patch_loss():
    prediction = distribution([[0., 2.]] * 20, scale=.1, weights=[[.8, .2]])
    normal = torch.zeros(20)
    abnormal = normal.clone()
    abnormal[-1] = 1.
    before, after = point_evidence(prediction, normal), point_evidence(prediction, abnormal)
    torch.testing.assert_close(after[:-1], before[:-1], atol=2e-5, rtol=2e-5)
    assert after[-1, 1] > before[-1, 1] + 9
    assert after[-1, 3] > after[:-1, 3].max() + 5


def test_predictive_intervals_use_prior_distribution_and_known_laplace_quantiles():
    prediction = distribution([[math.log(10.)]], scale=.2)
    intervals = quantiles(prediction)[0]
    expected = torch.tensor([10 * math.exp(-.2 * math.log(10)), 10., 10 * math.exp(.2 * math.log(10))])
    torch.testing.assert_close(intervals, expected, atol=3e-5, rtol=3e-6)
    # The complete likelihood penalizes wider normal predictions at zero error.
    narrow = joint_nll(prediction, torch.tensor([math.log(10.)]), torch.tensor([True]))
    wide = deepcopy(prediction)
    wide["scale"] *= 3
    assert joint_nll(wide, torch.tensor([math.log(10.)]), torch.tensor([True])) > narrow


@pytest.mark.parametrize("mode", NORMAL_MODES)
def test_new_branches_preserve_every_n1_parameter_and_rng(mode):
    seed_all(31)
    original = Segmentor("conditional")
    random = torch.get_rng_state()
    seed_all(31)
    candidate = Segmentor(mode)
    assert torch.equal(torch.get_rng_state(), random)
    for key, value in original.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
    assert torch.count_nonzero(candidate.normal_evidence.weight) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA comparison")
@pytest.mark.parametrize("mode", NORMAL_MODES)
def test_cpu_cuda_prediction_and_joint_likelihood_agree(mode):
    xyzi, rays = angular_scan()
    observation = angular_observation(xyzi, directions=rays)
    seed_all(5)
    cpu = NormalDistribution(mode)
    gpu = deepcopy(cpu).cuda()
    first, second = cpu(observation), gpu(to_device(observation, torch.device("cuda")))
    for key in ("mu", "scale", "log_weights"):
        torch.testing.assert_close(first[key], second[key].cpu(), atol=2e-5, rtol=2e-5)
    selected = torch.ones(len(xyzi), dtype=torch.bool)
    a = joint_nll(first, observation["log_range"], selected)
    b = joint_nll(second, observation["log_range"].cuda(), selected.cuda())
    torch.testing.assert_close(a, b.cpu(), atol=2e-5, rtol=2e-5)
