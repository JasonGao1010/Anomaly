"""Normal evidence, ray mathematics, target exclusion and gradient semantics."""

from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfcx, log_ndtr
from scipy.stats import t as student_t
import torch
from torch import nn
from torch.nn import functional as F

from src.normal import (NormalField, Compatibility, angular_observation, joint_nll,
                        ray_parameters, ray_log_prob, log_normal_mass, SCALES, LOWER, UPPER,
                        HYPOTHESES, KERNELS, LogNormalCDF, log_event_ratio)
from src.model import Segmentor, balanced_loss, ranking_loss, to_device
from src.train import cached_backward, seed_all, rng_state, restore_rng


def test_semantic_hypotheses_exclude_entire_target_cell_values_counts_and_gradients():
    from src.normal import hypothesis_observation, SemanticHypotheses
    torch.manual_seed(71)
    xyzi, _ = angular_scan()
    observation = hypothesis_observation(xyzi)
    model = SemanticHypotheses().eval()
    group = observation["group"][len(xyzi) // 2].reshape(1)
    first = model.propose(observation, model.encode(observation), group)
    second_observation = deepcopy(observation)
    target = observation["group"] == group.item()
    second_observation["features"][target] *= 3
    second_observation["log_distance"][target] += 1
    second = model.propose(second_observation, model.encode(second_observation), group)
    for name in first:
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)

    # Multiple returns in the withheld cell must not reveal its occupancy or depth.
    extra = np.repeat(xyzi[target.numpy()], 7, axis=0)
    extra[:, :3] *= np.linspace(.7, 1.4, len(extra))[:, None]
    extra[:, 3] = 1
    more = hypothesis_observation(np.concatenate((xyzi, extra)))
    assert torch.equal(more["cells"], observation["cells"])
    third = model.propose(more, model.encode(more), group)
    for name in first:
        torch.testing.assert_close(first[name], third[name], atol=0, rtol=0)

    observation["features"].requires_grad_()
    observation["log_distance"].requires_grad_()
    proposed = model.propose(observation, model.encode(observation), group)
    sum(value.square().sum() for key, value in proposed.items() if key != "supported").backward()
    for key in ("features", "log_distance"):
        gradient = observation[key].grad
        assert torch.equal(gradient[target], torch.zeros_like(gradient[target]))
        assert bool(gradient[~target].abs().sum() > 0)


def test_normal_label_sets_do_not_invent_fine_source_labels():
    from src.data import NUSCENES_NORMAL_SETS, STU_NORMAL_SEMANTICS
    assert NUSCENES_NORMAL_SETS[24] == (8, 9)
    assert NUSCENES_NORMAL_SETS[30] == (14, 15)
    assert NUSCENES_NORMAL_SETS[14] == (1, 6)
    assert not ({0, 1, 9, 10, 11, 12, 25, 28, 29, 31} & NUSCENES_NORMAL_SETS.keys())
    assert not ({0, 1, 2, 52, 99} & STU_NORMAL_SEMANTICS.keys())


def test_hypothesis_density_matches_independent_student_t_in_log_distance():
    from src.normal import hypothesis_observation, SemanticHypotheses, geometry_energy
    torch.manual_seed(13)
    xyzi, _ = angular_scan()
    observation = hypothesis_observation(xyzi)
    model = SemanticHypotheses().double().eval()
    observation = {key: value.double() if value.is_floating_point() else value
                   for key, value in observation.items()}
    indices = torch.tensor([18, 73, 129])
    prediction = model(observation, indices)
    mean = prediction["mean"].detach().numpy()
    scale = prediction["scale"].detach().numpy()
    value = observation["log_distance"][indices].numpy()[:, None, None]
    expected = student_t.logpdf(value, df=3, loc=mean, scale=scale)
    peak = student_t.logpdf(mean, df=3, loc=mean, scale=scale)
    np.testing.assert_allclose(prediction["log_prob"].detach(), expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(prediction["log_compatibility"].detach(), expected - peak,
                               atol=2e-12, rtol=2e-12)
    weights = prediction["weight"].detach().exp().numpy()
    np.testing.assert_allclose(weights.sum(-1), 1., atol=2e-15)
    # Integration is over log distance: no metre Jacobian or range truncation.
    mass, error = quad(lambda x: np.dot(weights[0, 0], student_t.pdf(
        x, df=3, loc=mean[0, 0], scale=scale[0, 0])), -np.inf, np.inf, epsabs=1e-10)
    assert abs(mass - 1) < 1e-9 and error < 1e-8
    expected_energy = -np.log((weights * np.exp(expected - peak)).sum(-1))
    expected_energy[~prediction["supported"].numpy()] = 0
    np.testing.assert_allclose(geometry_energy(prediction).detach(), expected_energy,
                               atol=2e-12, rtol=2e-12)


def test_coarse_normal_labels_admit_different_fine_classes_in_one_cell():
    from src.normal import hypothesis_loss
    allowed = torch.zeros(2, 19, dtype=torch.bool)
    allowed[:, :2] = True
    prediction = dict(log_prob=torch.full((2, 19, 3), math.log(.1)),
                      weight=torch.full((2, 19, 3), -math.log(3)),
                      belief=torch.zeros(2, 19), group=torch.zeros(2, dtype=torch.long),
                      supported=torch.ones(2, dtype=torch.bool))
    prediction["belief"][:, 0] = math.log(3)
    prediction["log_prob"][0, 0] = math.log(4)
    prediction["log_prob"][1, 1] = math.log(4)
    geometry, context = hypothesis_loss(prediction, allowed)
    expected = -(math.log(.75 * 4 + .25 * .1) + math.log(.75 * .1 + .25 * 4)) / 2
    torch.testing.assert_close(geometry, torch.tensor(expected))
    torch.testing.assert_close(context, torch.tensor(-math.log(4 / 21)))
    # Unadmitted classes affect context classification, never the conditional density.
    prediction["belief"][:, 2:] += 10
    changed_geometry, changed_context = hypothesis_loss(prediction, allowed)
    torch.testing.assert_close(changed_geometry, geometry)
    assert changed_context > context + 9
    extended = {key: torch.cat((value, value)) for key, value in prediction.items()}
    extended["supported"][2] = False
    extended["log_prob"][2:] = -1e6
    labels = torch.cat((allowed, allowed))
    labels[3].zero_()
    filtered_geometry, filtered_context = hypothesis_loss(extended, labels)
    torch.testing.assert_close(filtered_geometry, changed_geometry)
    torch.testing.assert_close(filtered_context, changed_context)


def test_allowed_set_balancing_and_missing_supervision_have_finite_gradients():
    from src.normal import allowed_loss, hypothesis_loss, geometry_energy
    logits = torch.tensor([[2., -.5, 0.], [-.7, .1, 1.]], requires_grad=True)
    allowed = torch.tensor([[True, False, False], [False, True, True]])
    original = allowed_loss(logits, allowed)
    repeated = torch.tensor([0, 0, 0, 0, 1])
    torch.testing.assert_close(allowed_loss(logits[repeated], allowed[repeated]), original)
    blank = torch.zeros_like(allowed)
    prediction = dict(log_prob=torch.randn(2, 3, 3, requires_grad=True),
                      belief=logits, weight=torch.full((2, 3, 3), -math.log(3)),
                      log_compatibility=torch.full((2, 3, 3), -5., requires_grad=True),
                      supported=torch.zeros(2, dtype=torch.bool))
    empty = allowed_loss(logits, blank)
    for labels in (allowed, blank):
        density, context = hypothesis_loss(prediction, labels)
        assert density.item() == context.item() == 0
        empty = empty + density + context
    energy = geometry_energy(prediction)
    assert torch.equal(energy, torch.zeros_like(energy))
    (empty + energy.sum()).backward()
    for value in (logits, prediction["log_prob"], prediction["log_compatibility"]):
        assert torch.equal(value.grad, torch.zeros_like(value))


def test_joint_semantic_supervision_reaches_the_held_out_predictor(hypothesis_model, monkeypatch):
    from src.normal import hypothesis_observation
    torch.manual_seed(51)
    xyzi, _ = angular_scan()
    observation = hypothesis_observation(xyzi)
    model = hypothesis_model.train()
    indices = torch.tensor([40, 74, 165])
    semantic = torch.full((len(xyzi), 19), 5., requires_grad=True)
    allowed = torch.zeros(len(xyzi), 19, dtype=torch.bool)
    allowed[indices] = F.one_hot(torch.tensor([0, 5, 8]), 19).bool()
    sample = dict(xyzi=torch.from_numpy(xyzi), observation=observation,
                  queries=indices, allowed=allowed)
    monkeypatch.setattr(model, "semantic", lambda sample: semantic)
    monkeypatch.setattr("src.model.NORMAL_LOSS_WEIGHTS",
                        dict(joint=1., semantic=0., normal=0., geometry=0., context=0.))
    loss, detail = model.loss(sample)
    torch.testing.assert_close(loss.detach(), detail["joint"])
    loss.backward()
    assert semantic.grad.abs().sum() > 0
    assert torch.equal(semantic.grad[0], torch.zeros(19))
    for parameter in (model.hypotheses.surface.weight, model.hypotheses.queries,
                      model.hypotheses.encoder[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        assert torch.isfinite(parameter.grad).all()
    with torch.no_grad():
        development_loss, _ = model.eval().loss(sample)
        expected = model.components(sample)
    torch.testing.assert_close(development_loss, loss.detach(), atol=2e-5, rtol=2e-5)
    assert torch.equal(model.development_prediction, expected["logits"].argmax(-1))
    assert torch.equal(model.development_semantic_prediction, semantic.argmin(-1))


@pytest.fixture
def hypothesis_model(monkeypatch):
    from src.model import NormalHypothesis
    # Sparse-backbone execution is covered separately; these tests isolate the head.
    monkeypatch.setattr("src.model.LitePT", lambda **kwargs: nn.Identity())
    return NormalHypothesis().eval()


def test_joint_components_use_matching_classes_and_preserve_point_subsets(hypothesis_model, monkeypatch):
    from src.normal import hypothesis_observation
    xyzi, _ = angular_scan()
    sample = dict(xyzi=torch.from_numpy(xyzi), observation=hypothesis_observation(xyzi))
    semantic = torch.rand(len(xyzi), 19)
    monkeypatch.setattr(hypothesis_model, "semantic", lambda sample: semantic)
    with torch.no_grad():
        full = hypothesis_model.components(sample)
        indices = torch.tensor([180, 4, 119, 4, 51])
        subset = hypothesis_model.components(sample, indices)
        monkeypatch.setattr("src.normal.BLOCK_CHUNK", 2)
        chunked = hypothesis_model.components(sample, indices)
    for key in ("energy", "logits", "semantic_energy", "geometry_energy", "raw_score"):
        assert len(subset[key]) == len(indices)
        torch.testing.assert_close(subset[key], full[key][indices], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(subset[key], chunked[key], atol=2e-5, rtol=2e-5)
    for key in full["prediction"]:
        torch.testing.assert_close(subset["prediction"][key], full["prediction"][key][indices],
                                   atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(subset["energy"], subset["semantic_energy"] + subset["geometry_energy"])
    torch.testing.assert_close(subset["logits"], -subset["energy"])

    # A low semantic cost and a low geometric cost must belong to the same class.
    semantic.fill_(20)
    semantic[:, :2] = torch.tensor([0., 1.])
    prediction = dict(weight=torch.full((len(xyzi), 19, 3), -math.log(3)),
                      log_compatibility=torch.zeros(len(xyzi), 19, 3),
                      supported=torch.ones(len(xyzi), dtype=torch.bool))
    prediction["log_compatibility"][:, 0] = -5
    monkeypatch.setattr(hypothesis_model.hypotheses, "forward", lambda observation, indices:
                        {key: value[indices] for key, value in prediction.items()})
    changed = hypothesis_model.components(sample, indices)
    assert bool((semantic[indices].argmin(-1) == 0).all())
    assert bool((changed["logits"].argmax(-1) == 1).all())
    hypothesis_model.calibrated.fill_(True)
    hypothesis_model.calibration.copy_(torch.linspace(0., 10., 129))
    output = hypothesis_model.predict(sample)
    assert bool((output["semantic"] == 1).all())
    assert bool((output["semantic_only"] == 0).all())
    torch.testing.assert_close(output["raw_score"][indices], changed["energy"].amin(-1))
    torch.testing.assert_close(output["score"], hypothesis_model.calibrate_score(output["raw_score"]))
    torch.testing.assert_close(hypothesis_model(sample), output["score"])
    prediction["supported"].zero_()
    unsupported = hypothesis_model.components(sample, indices)
    torch.testing.assert_close(unsupported["energy"], semantic[indices])


def test_global_calibration_preserves_order_with_repeated_quantiles(hypothesis_model):
    from src.model import CALIBRATION_LEVELS
    raw = torch.linspace(-2., 6., 801)
    with pytest.raises(ValueError, match="calibrat"):
        hypothesis_model.calibrate_score(raw)
    assert hypothesis_model.calibration.shape == (129,)
    hypothesis_model.calibrated.fill_(True)
    hypothesis_model.calibration.copy_(torch.linspace(0., 4., 129))
    expected = torch.tensor(CALIBRATION_LEVELS, dtype=torch.float32)
    torch.testing.assert_close(hypothesis_model.calibrate_score(hypothesis_model.calibration), expected)
    hypothesis_model.calibration[40:61] = hypothesis_model.calibration[40]
    calibrated = hypothesis_model.calibrate_score(raw)
    assert torch.isfinite(calibrated).all()
    assert bool((calibrated[1:] >= calibrated[:-1]).all())
    selected = torch.tensor([540, 2, 211, 540, 700])
    torch.testing.assert_close(hypothesis_model.calibrate_score(raw[selected]), calibrated[selected])
    hypothesis_model.calibration.fill_(2.)
    constant_reference = hypothesis_model.calibrate_score(raw)
    assert torch.isfinite(constant_reference).all()
    assert bool((constant_reference[1:] > constant_reference[:-1]).all())


def test_normal_checkpoint_rejects_previous_scientific_definition():
    from src.model import (NormalHypothesis, NORMAL_VERSION, NORMAL_ARCHITECTURE,
                           NORMAL_SCORE_VERSION)
    saved = dict(version=NORMAL_VERSION,
                 config=dict(architecture=NORMAL_ARCHITECTURE, score_version=NORMAL_SCORE_VERSION),
                 model=dict(calibration=torch.zeros(129), calibrated=torch.tensor(True)))
    NormalHypothesis.validate_checkpoint(saved, require_calibrated=True)
    for location, key in (("", "version"), ("config", "architecture"), ("config", "score_version")):
        previous = deepcopy(saved)
        (previous[location] if location else previous)[key] = "previous-method"
        with pytest.raises(ValueError, match="incompatible"):
            NormalHypothesis.validate_checkpoint(previous)
    previous = deepcopy(saved)
    previous["model"]["calibration"] = torch.zeros(19, 2, 129)
    with pytest.raises(ValueError, match="shape"):
        NormalHypothesis.validate_checkpoint(previous)
    saved["model"]["calibrated"].fill_(False)
    NormalHypothesis.validate_checkpoint(saved)
    with pytest.raises(ValueError, match="calibration"):
        NormalHypothesis.validate_checkpoint(saved, require_calibrated=True)


def test_semantic_context_projection_reuse_matches_original_attention_and_gradients():
    from src.normal import SemanticHypotheses
    torch.manual_seed(29)
    model = SemanticHypotheses().double().eval()
    reference = deepcopy(model)
    tokens = torch.randn(7, 48, dtype=torch.double, requires_grad=True)
    original_tokens = tokens.detach().clone().requires_grad_()
    depth = torch.randn(7, dtype=torch.double, requires_grad=True)
    original_depth = depth.detach().clone().requires_grad_()
    neighbors = torch.full((4, 16), 7, dtype=torch.long)
    neighbors[0, :3], neighbors[1, :2], neighbors[2, :3] = (
        torch.tensor(row) for row in ([1, 2, 5], [0, 6], [1, 3, 5]))
    observation = dict(neighbors=neighbors, position=torch.randn(4, 3, dtype=torch.double))
    present = neighbors < len(tokens)
    context = F.pad(original_tokens, (0, 0, 0, 1))[neighbors]
    context = torch.cat((context, reference.empty.expand(4, 1, -1)), 1)
    mask = torch.cat((~present, present.any(1, keepdim=True)), 1)
    state = F.normalize(reference.queries, dim=-1)[None] * math.sqrt(48)
    state = state + reference.position(observation["position"])[:, None]
    for layer in reference.layers:
        update = layer["attention"](state, context, context, key_padding_mask=mask, need_weights=False)[0]
        state = layer["norm"](state + update)
        state = layer["final_norm"](state + layer["feedforward"](state))
    raw = reference.surface(state).reshape(4, 19, 3, 8)
    base = (F.pad(original_depth, (0, 1))[neighbors] * present).sum(1) / present.sum(1).clamp_min(1)
    base = torch.where(present.any(1), base, torch.full_like(base, math.log(20)))
    expected = dict(mean=base[:, None, None] + raw[..., 0], slope=raw[..., 1:6],
                    scale=.001 + F.softplus(raw[..., 6]), weight=raw[..., 7].log_softmax(-1),
                    belief=reference.belief(state).squeeze(-1), supported=present.any(1))
    actual = model.propose(observation, (tokens, depth), torch.arange(4), model.project_context(tokens))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-12, rtol=2e-12)
    sum(value.square().sum() for key, value in actual.items() if key != "supported").backward()
    sum(value.square().sum() for key, value in expected.items() if key != "supported").backward()
    torch.testing.assert_close(tokens.grad, original_tokens.grad, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(depth.grad, original_depth.grad, atol=2e-11, rtol=2e-11)
    for (name, parameter), (reference_name, original) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == reference_name
        if parameter.grad is None:
            assert original.grad is None
        else:
            torch.testing.assert_close(parameter.grad, original.grad, atol=2e-11, rtol=2e-11)


def test_distant_ground_requires_unbounded_and_curved_angular_prediction():
    # A flat road at 43--49 m already exceeds the retired slope cap of 20.
    elevation = np.deg2rad(np.linspace(-2., -1.75, 100))
    distance = 1.5 / -np.sin(elevation)
    target = np.log(distance)
    assert (target[-1] - target[0]) / (elevation[-1] - elevation[0]) > 20
    position = (elevation - elevation.mean()) / np.deg2rad(.25)
    predicted = np.polyval(np.polyfit(position, target, 2), position)
    assert np.max(np.abs(predicted - target)) < 1e-4


def angular_scan():
    az, el = np.meshgrid(np.arange(-9.7, 10., .7), np.arange(-8.3, 3., .8))
    az, el = np.deg2rad(az.ravel()), np.deg2rad(el.ravel())
    rays = np.column_stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))
    ranges = np.linspace(8., 24., len(az))
    return np.column_stack((rays * ranges[:, None], np.full(len(az), .4))).astype(np.float32), rays


@pytest.mark.parametrize("chunk", [1, 4])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_shared_context_projections_match_original_mha_values_and_gradients(chunk, device):
    seed_all(113)
    model = NormalField(recompute=False).to(device)
    reference = deepcopy(model)
    tokens = torch.randn(7, 128, device=device, requires_grad=True)
    original_tokens = tokens.detach().clone().requires_grad_()
    neighbors = torch.full((4, 24), len(tokens), dtype=torch.long, device=device)
    neighbors[0, :3], neighbors[1, :2], neighbors[2, :3] = (torch.tensor(row, device=device)
        for row in ([1, 2, 5], [0, 6], [1, 3, 5]))
    position, basis = torch.randn(4, 4, device=device), torch.eye(3, device=device).expand(4, 3, 3)
    # The final row has no neighbor: only its separate learned empty token is visible.
    present = neighbors < len(tokens)
    context = F.pad(original_tokens, (0, 0, 0, 1))[neighbors]
    context = torch.cat((context, reference.empty.expand(4, 1, -1)), 1)
    mask = torch.cat((~present, present.any(1, keepdim=True)), 1)
    state = reference.queries[None] + reference.position(position)[:, None]
    for layer in reference.layers:
        update = layer["attention"](state, context, context, key_padding_mask=mask, need_weights=False)[0]
        state = layer["norm"](state + update)
        state = layer["final_norm"](state + layer["feedforward"](state))
    expected_raw = reference.parameters_out(state)
    expected_weights = reference.weights(state).squeeze(-1).log_softmax(-1)
    projected, raw = model.project_context(tokens), []
    hook = model.parameters_out.register_forward_hook(lambda module, inputs, output: raw.append(output))
    parts = [model.decode(projected, neighbors[start:start + chunk], position[start:start + chunk], basis[start:start + chunk])
             for start in range(0, len(neighbors), chunk)]
    hook.remove()
    actual_raw, actual_weights = torch.cat(raw), torch.cat([part[-1] for part in parts])
    torch.testing.assert_close(actual_raw, expected_raw, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_weights, expected_weights, atol=1e-5, rtol=1e-5)
    raw_probe, weight_probe = torch.randn_like(actual_raw), torch.randn_like(actual_weights)
    ((actual_raw * raw_probe).mean() + (actual_weights * weight_probe).mean()).backward()
    ((expected_raw * raw_probe).mean() + (expected_weights * weight_probe).mean()).backward()
    torch.testing.assert_close(tokens.grad, original_tokens.grad, atol=2e-6, rtol=2e-4)
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        if actual.grad is None or expected.grad is None:
            assert actual.grad is expected.grad, name
        else:
            assert torch.isfinite(actual.grad).all(), name
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-4, msg=name)


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
    indices = torch.arange(len(xyzi))
    before, _ = decoder(state, observation, fields, indices)
    changed = deepcopy(observation)
    changed["distance"][0] *= 1.8
    after, _ = decoder(state, changed, fields, indices)
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
    indices = torch.arange(len(state))
    before, probability = decoder(state, observation, fields, indices)
    after, changed_probability = decoder(state, observation, reordered, indices)
    torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-6)
    expected = probability[..., permutation] if permuted == "hypotheses" else probability
    torch.testing.assert_close(changed_probability, expected, atol=2e-6, rtol=2e-6)


def test_compatibility_distinguishes_hypothesis_grouping_at_equal_marginal_density():
    seed_all(29)
    middle = (LOWER + UPPER) / 2
    observation = angular_observation(np.array([[middle - 4, 0., 0., .4]], dtype=np.float32))
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
    before, probability = decoder(state, observation, fields, torch.arange(1))
    after, changed_probability = decoder(state, observation, changed, torch.arange(1))
    # Identical marginal densities can arise from different complete hypotheses.
    # The compact reader must retain their individual evidence before pooling.
    assert not torch.allclose(probability, changed_probability)
    torch.testing.assert_close(before[:, -len(SCALES):], after[:, -len(SCALES):], atol=1e-6, rtol=0)
    assert (before[:, 64:-len(SCALES)] - after[:, 64:-len(SCALES)]).abs().max() > 1e-5


def test_compatibility_chunks_preserve_scores_and_outside_support_has_no_density(monkeypatch):
    seed_all(61)
    distances = np.array([1., LOWER, 8., 15., UPPER, 60.], dtype=np.float32)
    xyzi = np.column_stack((distances, np.zeros((len(distances), 2)), np.full(len(distances), .4)))
    observation = angular_observation(xyzi.astype(np.float32))
    fields = NormalField(recompute=False)(observation)
    decoder, state = Compatibility().eval(), torch.randn(len(distances), 64)
    whole, probability = decoder(state, observation, fields, torch.arange(len(state)))
    parts = [decoder(state[start:start + 2], observation, fields, torch.arange(start, start + 2))
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
    altered, _ = decoder(state, observation, fields, torch.arange(len(state)))
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
    hidden, _ = decoder(torch.randn(32, 64), observation, fields, torch.arange(32))
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
    xyzi[1, :3] *= .1
    xyzi[2, :3] *= 10
    observation = angular_observation(xyzi)
    targets = torch.zeros(len(xyzi), dtype=torch.long)
    targets[::5] = -1
    targets[::7] = 1
    eligible = ((targets == 0) & (observation["distance"] >= LOWER) & (observation["distance"] <= UPPER))
    seed_all(17)
    model = NormalField()
    reference = deepcopy(model)
    reference.recompute = False
    fields = reference(observation)
    _, probabilities = Compatibility()(torch.zeros(len(xyzi), 64), observation, fields, torch.arange(len(xyzi)))
    losses = []
    for scale, size in enumerate(SCALES):
        group = observation["grids"][str(size)]["group"]
        blocks = []
        for index in torch.unique(group[eligible]):
            selected = (group == index) & eligible
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


def test_sparse_compatibility_queries_preserve_point_order_values_and_gradients():
    seed_all(101)
    xyzi, _ = angular_scan()
    observation = angular_observation(xyzi[:24])
    model, decoder = NormalField(recompute=False), Compatibility()
    reference_model, reference_decoder = deepcopy(model), deepcopy(decoder)
    state = torch.randn(24, 64, requires_grad=True)
    reference_state = state.detach().clone().requires_grad_()
    selected = torch.tensor([21, 3, 10, 1, 18])
    full, full_probability = reference_decoder(reference_state, observation, reference_model(observation), torch.arange(24))
    sparse, probability = decoder(state[selected], observation, model(observation), selected)
    torch.testing.assert_close(sparse, full[selected], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(probability, full_probability[selected], atol=2e-6, rtol=2e-6)
    probe = torch.randn_like(sparse)
    (sparse * probe).mean().backward()
    (full[selected] * probe).mean().backward()
    torch.testing.assert_close(state.grad, reference_state.grad, atol=2e-6, rtol=2e-4)
    for actual, expected in zip((*model.parameters(), *decoder.parameters()),
                                (*reference_model.parameters(), *reference_decoder.parameters())):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-4)


def test_clean_companion_without_valid_normal_targets_has_zero_loss_and_gradients():
    xyzi, _ = angular_scan()
    observation = angular_observation(xyzi[:8])
    model = NormalField(recompute=False)
    loss = model.likelihood(observation, torch.full((8,), -1, dtype=torch.long))
    assert loss.item() == 0 and loss.requires_grad
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0


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

    def forward(self, sample, *, normal_loss=False, query_indices=None):
        x = self.encoder(sample["xyzi"])
        normal = self.normal(x)
        score = self.head(torch.cat((x, normal), 1)).flatten()
        if query_indices is not None:
            score = score[query_indices]
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
    for sample in samples:
        sample["control_mask"] = torch.zeros_like(sample["targets"], dtype=torch.bool)
        sample["control_mask"][(sample["targets"] == 0).nonzero()[0]] = True
    device = torch.device("cpu")
    start = rng_state(device)
    outputs = [reference(sample, normal_loss=True) for sample in samples]
    scores = torch.cat([value[0] for value in outputs])
    targets = torch.cat([s["targets"] for s in samples])
    counts = torch.stack([(targets == label).sum() for label in (0, 1)])
    rank, detail = ranking_loss(scores, targets, 19)
    normal_count = sum(s.get("normal_reference", s)["normal_training"] for s in samples)
    normal = sum(v[1] for v in outputs) / normal_count
    controls = torch.cat([s["control_mask"] for s in samples])
    loss = balanced_loss(scores, targets, counts, control_mask=controls) + rank + .1 * normal
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


def test_control_balancing_preserves_object_weight_when_background_is_repeated():
    def compute(repeats):
        values = torch.tensor([-.7, .4, 1.2, 99.], requires_grad=True)
        scores = torch.cat((values[:2], values[2:3].expand(repeats), values[3:]))
        targets = torch.tensor([1, 0] + [0] * repeats + [-1])
        controls = torch.tensor([False, True] + [False] * (repeats + 1))
        loss = balanced_loss(scores, targets, torch.tensor([repeats + 1, 1]), controls)
        loss.backward()
        return loss.detach(), values.grad
    loss, gradient = compute(1)
    repeated, repeated_gradient = compute(1000)
    expected = .5 * F.softplus(torch.tensor(.7)) + .25 * F.softplus(torch.tensor(.4)) + .25 * F.softplus(torch.tensor(1.2))
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(repeated, loss)
    torch.testing.assert_close(repeated_gradient, gradient)
    assert gradient[-1] == 0
    with pytest.raises(ValueError, match="verified normal"):
        balanced_loss(torch.zeros(2), torch.tensor([0, 1]), torch.tensor([1, 1]), torch.tensor([False, True]))


def test_control_identity_uses_delta_slots_and_excludes_ignored_returns(tmp_path, monkeypatch):
    from src.data import Scans, SOURCE_VERSION
    from src.evaluate import PreparedScans
    delta = tmp_path / "control.npz"
    np.savez(delta, slots=np.array([4, 7, 9]), labels=np.array([1, 1, 2], dtype=np.uint32))
    sample = dict(xyzi=np.ones((4, 4), np.float32), slots=np.array([1, 4, 7, 9]),
                  targets=np.array([0, 0, -1, 1]), slot_count=10, index=0)
    monkeypatch.setattr(Scans, "__getitem__", lambda self, index: sample)
    record = dict(source="nuscenes", group="control_nuscenes", delta=str(delta), inserted_points=1)
    data = PreparedScans(dict(version=SOURCE_VERSION, kind="train", records=[record]), voxel=False)
    assert data[0]["control_mask"].tolist() == [False, True, False, False]
    record["inserted_points"] = 2
    with pytest.raises(ValueError, match="control points differ"):
        data[0]


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
def test_real_selected_queries_preserve_complete_model_scores_likelihood_and_gradients():
    """Sparse final queries preserve full-context training on a real normal scan."""
    from pathlib import Path
    import time
    from src.data import load_manifest
    from src.evaluate import PreparedScans

    manifest_path = Path("results/data/sequence/train.json")
    if not manifest_path.exists() or not Path("assets/nuscenes.pth").exists():
        pytest.skip("local nuScenes data and official initialization are required")
    manifest = load_manifest(manifest_path, "train")
    index = next(i for i, row in enumerate(manifest["records"])
                 if row["group"] == "normal_nuscenes" and not row.get("delta") and row["normal"] > 0)
    data = PreparedScans(manifest, normal=True)
    device = torch.device("cuda")
    sample = to_device(data[index], device)
    assert sample["normal_training"] and "normal_reference" not in sample
    selected = (sample["targets"] >= 0).nonzero().flatten()
    assert 0 < len(selected) < len(sample["xyzi"])
    seed_all(107)
    model = Segmentor(recompute=False).to(device).train()
    model.load_pretrained("assets/nuscenes.pth")
    initial_rng = rng_state(device)
    initial_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    results = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for sparse in (False, True):
        model.zero_grad(set_to_none=True)
        restore_rng(initial_rng, device)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(initial_buffers[name])
        logits, normal = model(sample, normal_loss=True, query_indices=selected if sparse else None)
        scored = logits if sparse else logits[selected]
        # Both paths optimize exactly the same points and clean normal targets.
        loss = F.softplus(scored).mean() + .1 * normal
        loss.backward()
        torch.cuda.synchronize(device)
        results.append(dict(scores=scored.detach().cpu(), normal=normal.detach().cpu(),
            gradients={name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                       for name, parameter in model.named_parameters()},
            buffers={name: value.detach().cpu().clone() for name, value in model.named_buffers()},
            rng=rng_state(device)))
        del logits, normal, scored, loss
    full, sparse = results
    # Bounds are fixed before execution; changed GEMM batch shapes can round differently.
    torch.testing.assert_close(sparse["scores"], full["scores"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sparse["normal"], full["normal"], atol=1e-5, rtol=1e-5)
    gradient_error = 0.
    for name, actual in sparse["gradients"].items():
        reference = full["gradients"][name]
        if actual is None or reference is None:
            assert actual is reference, name
            continue
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), name
        gradient_error = max(gradient_error, float((actual - reference).abs().max()))
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-4, msg=name)
    for name, actual in sparse["buffers"].items():
        torch.testing.assert_close(actual, full["buffers"][name], atol=0, rtol=0, msg=name)
    assert torch.equal(sparse["rng"]["torch"], full["rng"]["torch"])
    assert torch.equal(sparse["rng"]["cuda"], full["rng"]["cuda"])
    print(dict(real_index=index, original_points=len(sample["xyzi"]), queried_points=len(selected),
        parameters=sum(p.numel() for p in model.parameters()),
        logits_max_abs=float((sparse["scores"] - full["scores"]).abs().max()),
        normal_nll_abs=float((sparse["normal"] - full["normal"]).abs()), gradient_max_abs=gradient_error,
        seconds=time.perf_counter() - started, peak_vram_bytes=torch.cuda.max_memory_allocated(device)))


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
    manifest_path = Path("results/data/sequence/train.json")
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
