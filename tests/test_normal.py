"""SERVE target exclusion, predictive density, gradients and normal supervision."""

from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t as student_t
import torch
from torch import nn
from torch.nn import functional as F


@pytest.mark.parametrize("maximum", [31, 57, 1299])
def test_rotary_serialization_bound_preserves_values_and_gradients(maximum):
    from vendor.litept.pointrope import PointROPE
    torch.manual_seed(84)
    positions = torch.randint(0, maximum + 1, (1, 127, 3))
    positions[0, 0, 0] = maximum
    tokens = torch.randn(1, 4, 127, 18, requires_grad=True)
    rope = PointROPE()
    expected = rope(tokens, positions)
    expected_gradient = torch.autograd.grad(expected.square().sum(), tokens)[0]
    depth = (maximum + 1).bit_length()
    actual = rope(tokens, positions, max_seqlen=(1 << depth) - 1)
    actual_gradient = torch.autograd.grad(actual.square().sum(), tokens)[0]
    assert torch.equal(actual, expected)
    assert torch.equal(actual_gradient, expected_gradient)


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
    assert NUSCENES_NORMAL_SETS[24] == (8, 9, 10, 11)
    assert NUSCENES_NORMAL_SETS[30] == (14, 15)
    assert NUSCENES_NORMAL_SETS[14] == (1, 6)
    assert not ({0, 1, 9, 10, 11, 12, 25, 28, 29, 31} & NUSCENES_NORMAL_SETS.keys())
    assert not ({0, 1, 2, 52, 99} & STU_NORMAL_SEMANTICS.keys())


def test_hypothesis_density_matches_independent_student_t_in_log_distance():
    from src.normal import hypothesis_observation, SemanticHypotheses, geometry_energy, LOG_RETURN_PEAK
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
    expected_energy = LOG_RETURN_PEAK - np.log((weights * np.exp(expected)).sum(-1))
    expected_energy[~prediction["supported"].numpy()] = 0
    np.testing.assert_allclose(geometry_energy(prediction).detach(), expected_energy,
                               atol=2e-12, rtol=2e-12)


def test_density_score_retains_width_cost_for_broad_mixture_modes():
    from src.normal import geometry_energy, LOG_RETURN_PEAK
    scale = torch.tensor([[.02, .02, .02], [.002, 1., 1.]], dtype=torch.float64)
    weight = torch.tensor([[1/3, 1/3, 1/3], [.1, .45, .45]], dtype=torch.float64).log()
    for residual in (0., 1.):
        log_prob = torch.from_numpy(student_t.logpdf(residual, 3, scale=scale.numpy()))
        prediction = dict(log_prob=log_prob[:, None], weight=weight[:, None],
                          supported=torch.ones(2, dtype=torch.bool))
        energy = geometry_energy(prediction).squeeze(-1)
        nll = -torch.logsumexp(weight + log_prob, -1)
        torch.testing.assert_close(energy, LOG_RETURN_PEAK + nll)
        torch.testing.assert_close(energy[1] - energy[0], nll[1] - nll[0])
    # At the predicted mean, tenfold broadening pays log(10), not zero cost.
    prediction["log_prob"] = torch.from_numpy(student_t.logpdf(
        0., 3, scale=np.array([.01, .1])[:, None, None])).expand(2, 1, 3)
    prediction["weight"] = torch.full((2, 1, 3), -math.log(3), dtype=torch.float64)
    energy = geometry_energy(prediction).squeeze(-1)
    torch.testing.assert_close(energy[1] - energy[0], torch.tensor(math.log(10), dtype=torch.float64))


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
    for value in (logits, prediction["log_prob"]):
        assert torch.equal(value.grad, torch.zeros_like(value))
    assert prediction["log_compatibility"].grad is None


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
    monkeypatch.setattr(model, "semantic", lambda sample, modes=False:
                        (semantic, torch.full((len(indices), 19, 4), .25)) if modes else semantic)
    monkeypatch.setattr("src.model.NORMAL_LOSS_WEIGHTS",
                        dict(classification=1., semantic=0., normal=0., geometry=0., context=0.))
    loss, detail = model.loss(sample)
    torch.testing.assert_close(loss.detach(), detail["classification"])
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


def test_variants_match_initialization_and_isolate_observation_class_gradients(hypothesis_model, monkeypatch):
    from src.model import NormalHypothesis
    from src.normal import hypothesis_observation
    xyzi, _ = angular_scan()
    indices = torch.tensor([40, 74, 165])
    allowed = torch.zeros(len(xyzi), 19, dtype=torch.bool)
    allowed[indices] = F.one_hot(torch.tensor([0, 5, 8]), 19).bool()
    sample = dict(xyzi=torch.from_numpy(xyzi), observation=hypothesis_observation(xyzi),
                  queries=indices, allowed=allowed)
    states, outputs = [], {}
    for variant in ("joint", "semantic", "separate"):
        torch.manual_seed(87)
        model = NormalHypothesis(variant=variant).train()
        states.append(deepcopy(model.state_dict()))
        semantic = torch.full((len(xyzi), 19), 5., requires_grad=True)
        monkeypatch.setattr(model, "semantic", lambda sample: semantic)
        monkeypatch.setattr("src.model.NORMAL_LOSS_WEIGHTS",
            dict(classification=1., semantic=0., normal=0., geometry=0., context=0.))
        loss, _ = model.loss(sample)
        loss.backward()
        gradient = model.hypotheses.surface.weight.grad
        assert semantic.grad.abs().sum() > 0
        if variant == "joint":
            assert gradient is not None and gradient.abs().sum() > 0
        else:
            assert gradient is None or torch.equal(gradient, torch.zeros_like(gradient))
        outputs[variant] = model.components(sample)["energy"].detach()
        if variant == "separate":
            model.zero_grad(set_to_none=True)
            monkeypatch.setattr("src.model.NORMAL_LOSS_WEIGHTS",
                dict(classification=0., semantic=0., normal=0., geometry=1., context=1.))
            model.loss(sample)[0].backward()
            assert model.hypotheses.surface.weight.grad.abs().sum() > 0
        if variant == "semantic":
            monkeypatch.setattr(model.hypotheses, "forward", lambda *args: pytest.fail("semantic baseline must skip predictor"))
            torch.testing.assert_close(model.components(sample)["energy"], semantic)
    for state in states[1:]:
        for key in states[0]:
            torch.testing.assert_close(state[key], states[0][key], rtol=0, atol=0)
    torch.testing.assert_close(outputs["joint"], outputs["separate"], rtol=0, atol=0)


def test_joint_components_use_matching_classes_and_preserve_point_subsets(hypothesis_model, monkeypatch):
    from src.normal import hypothesis_observation, LOG_RETURN_PEAK
    xyzi, _ = angular_scan()
    sample = dict(xyzi=torch.from_numpy(xyzi), observation=hypothesis_observation(xyzi))
    semantic = torch.rand(len(xyzi), 19)
    monkeypatch.setattr(hypothesis_model, "semantic", lambda sample: semantic)
    with torch.no_grad():
        full = hypothesis_model.components(sample)
        indices = torch.tensor([180, 4, 119, 4, 51])
        subset = hypothesis_model.components(sample, indices)
        monkeypatch.setattr("src.normal.HYPOTHESIS_CHUNK", 2)
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
                      log_prob=torch.full((len(xyzi), 19, 3), LOG_RETURN_PEAK),
                      supported=torch.ones(len(xyzi), dtype=torch.bool))
    prediction["log_prob"][:, 0] -= 5
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
                 config=dict(architecture=NORMAL_ARCHITECTURE, score_version=NORMAL_SCORE_VERSION, variant="joint"),
                 model=dict(calibration=torch.zeros(129), calibrated=torch.tensor(True)))
    NormalHypothesis.validate_checkpoint(saved, require_calibrated=True)
    for location, key in (("", "version"), ("config", "architecture"), ("config", "score_version"), ("config", "variant")):
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


def test_cycle_refinement_uses_rider_attributes_unique_boxes_and_world_pose():
    from scipy.spatial.transform import Rotation
    from src.data import refine_normal_labels
    rotation = Rotation.from_euler("z", 90, degrees=True)
    pose = np.eye(4)
    pose[:3, :3] = rotation.as_matrix()
    pose[:3, 3] = [100., 200., 1.]
    xyzi = np.array([[5., 0., 0., .5], [5.8, 0., 0., .5], [8., 0., 0., .5],
                     [11., 0., 0., .5], [14., 0., 0., .5]], np.float32)
    labels = np.full(5, 14)
    allowed = np.zeros((5, 19), bool)
    allowed[:, [1, 6]] = True
    boxes = []
    for token, center, rider in [("rider", 5., True), ("parked", 8., False),
                                  ("overlap-a", 11., True), ("overlap-b", 11., False)]:
        boxes.append(dict(token=token, instance_token=token, raw_class=14, with_rider=rider,
            translation=(rotation.apply([center, 0, 0]) + pose[:3, 3]).tolist(),
            rotation=rotation.as_quat()[[3, 0, 1, 2]].tolist(), size=[.5, 2., 2.]))
    refine_normal_labels(dict(pose=pose, normal_cycles=boxes), xyzi, labels, allowed, np.arange(5))
    assert np.array_equal(allowed.sum(1), [1, 1, 2, 2, 2])
    assert allowed[:2, 6].all() and not allowed[:2, 1].any()
    assert allowed[2:, [1, 6]].all()
    # nuScenes motorcycles also include light three-wheel vehicles.
    for box in boxes:
        box["raw_class"] = 21
    allowed[:] = False
    allowed[:, [2, 4, 7]] = True
    refine_normal_labels(dict(pose=pose, normal_cycles=boxes), xyzi,
                         np.full(5, 21), allowed, np.arange(5))
    assert np.array_equal(allowed.sum(1), [2, 2, 3, 3, 3])
    assert allowed[:2, [4, 7]].all() and not allowed[:2, 2].any()


def test_reviewed_point_labels_preserve_slots_source_labels_and_range(tmp_path):
    from src.data import read_normal_record, file_sha256
    raw = np.array([[0, 0, 0, 0, 0], [8, 0, 0, 127.5, 1], [9, 0, 0, 255, 2],
                    [60, 0, 0, 255, 3], [10, 0, 0, 255, 4]], np.float32)
    labels = np.array([24, 24, 28, 28, 28], np.uint8)
    scan, label = tmp_path / "scan.bin", tmp_path / "label.bin"
    raw.tofile(scan)
    labels.tofile(label)
    record = dict(source="nuscenes", scan=str(scan), label=str(label), pose=np.eye(4).tolist(),
        normal_fine=[dict(semantic="parking", raw_class=24, point_slots=[1],
            scan_sha256=file_sha256(scan), label_sha256=file_sha256(label)),
            dict(semantic="building", raw_class=28, point_slots=[2, 3])])
    loaded = read_normal_record(record)
    np.testing.assert_array_equal(loaded["slots"], [1, 2, 3, 4])
    np.testing.assert_array_equal(loaded["allowed"].sum(1), [1, 1, 0, 0])
    assert loaded["allowed"][0, 9] and loaded["allowed"][1, 12]
    assert loaded["xyzi"][0, 3] == .5
    invalid = deepcopy(record)
    invalid["normal_fine"][0]["point_slots"] = [2]
    with pytest.raises(ValueError, match="source labels"):
        read_normal_record(invalid)
    invalid = deepcopy(record)
    invalid["normal_fine"][0]["scan_sha256"] = "changed"
    with pytest.raises(ValueError, match="source file changed"):
        read_normal_record(invalid)
    invalid = deepcopy(record)
    invalid["normal_fine"][0]["point_slots"] = [0]
    with pytest.raises(ValueError, match="absent"):
        read_normal_record(invalid)


def test_reviewed_normal_annotations_cannot_cross_scan_partition(tmp_path, monkeypatch):
    import json
    from src import data
    annotation = dict(token="scan", sample_token="sample", scene="scene-a", subset="val",
                      semantic="parking", raw_class=24, point_slots=[1])
    path = tmp_path / "normal.json"
    path.write_text(json.dumps(dict(records=[annotation])))
    monkeypatch.setattr(data, "NORMAL_ANNOTATIONS", path)
    monkeypatch.setattr(data, "file_sha256", lambda path: "test-source")
    monkeypatch.setattr(data, "normal_cycle_annotations", lambda root, hashes: {})
    record = dict(token="scan", sample_token="sample", scene="scene-a", subset="train")
    with pytest.raises(ValueError, match="another scan or partition"):
        data.attach_normal_annotations([record], tmp_path)


def test_normal_reader_rejects_same_path_training_input_replacement(tmp_path):
    from src.data import read_normal_record, file_sha256
    scan, label = tmp_path / "scan.bin", tmp_path / "scan.label"
    raw = np.array([[5., 0., 0., .5], [6., 0., 0., .5]], np.float32)
    labels = np.array([40, 10], np.uint32)
    raw.tofile(scan)
    labels.tofile(label)
    record = dict(source="normal_stu", scan=str(scan), label=str(label), pose=np.eye(4).tolist(),
                  scan_sha256=file_sha256(scan), label_sha256=file_sha256(label))
    assert read_normal_record(record)["allowed"].sum() == 2
    raw[0, 0] = 7.
    raw.tofile(scan)
    with pytest.raises(ValueError, match="source file changed: scan"):
        read_normal_record(record)
    raw[0, 0] = 5.
    raw.tofile(scan)
    labels[0] = 10
    labels.tofile(label)
    with pytest.raises(ValueError, match="source file changed: label"):
        read_normal_record(record)


@pytest.mark.parametrize("changed_file", ["scan", "label"])
def test_normal_source_identity_tracks_bytes_and_preserves_existing_hashes(tmp_path, monkeypatch, changed_file):
    import json
    from src import data
    scan, label = tmp_path / "scan.bin", tmp_path / "label.bin"
    np.array([[5., 0., 0., 127.5, 0]], np.float32).tofile(scan)
    np.array([17], np.uint8).tofile(label)
    row = dict(source="nuscenes", scan=str(scan), label=str(label), pose=np.eye(4).tolist())
    directory = tmp_path / "results/data/background"
    directory.mkdir(parents=True)
    manifest = directory / "train.json"
    manifest.write_text(json.dumps(dict(root=str(tmp_path), records=[row])))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "attach_normal_annotations", lambda records, root: records)
    records = data.normal_records("nuscenes")
    initial_identity = data.identity(records)
    assert records[0]["scan_sha256"] == data.file_sha256(scan)
    assert records[0]["label_sha256"] == data.file_sha256(label)
    assert data.read_normal_record(records[0])["allowed"][0, 0]
    if changed_file == "scan":
        np.array([[6., 0., 0., 127.5, 0]], np.float32).tofile(scan)
    else:
        np.array([23], np.uint8).tofile(label)
    with pytest.raises(ValueError, match="source file changed: " + changed_file):
        data.read_normal_record(records[0])
    # An unhashed source manifest can start a new run, whose actual input identity differs.
    assert data.identity(data.normal_records("nuscenes")) != initial_identity
    manifest.write_text(json.dumps(dict(root=str(tmp_path), records=records)))
    # Existing identities may never be silently replaced with the changed file's digest.
    with pytest.raises(ValueError, match="source file changed: " + changed_file):
        data.normal_records("nuscenes")
