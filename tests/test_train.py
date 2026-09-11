"""Training regressions: supervision weighting, valid updates and exact continuation."""

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from src.supervision import boundary_loss, detection_loss, sampling_loss, surface_loss
from src.train import (assert_state_close, backward_sample, check_summary, loss_view_weights,
                       optimizer_for, restore_checkpoint, save_checkpoint, seed_all, to_device, update)


SETTINGS = json.loads(Path("protocol/v1.json").read_text())["training"]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the training loop uses CUDA")


@CUDA
def test_unpooling_normalizes_overflowing_projections_before_half_rounding():
    import spconv.pytorch as spconv
    from vendor.litept.model import GridUnpooling, Point

    model = GridUnpooling(2, 2, 3,
        norm_layer=lambda channels: torch.nn.BatchNorm1d(channels, track_running_stats=False),
        act_layer=torch.nn.GELU).cuda()
    with torch.no_grad():
        for branch in (model.proj, model.proj_skip):
            branch[0].weight.copy_(torch.tensor([[128., 64.], [-64., 128.], [96., -96.]], device="cuda"))
            branch[0].bias.zero_()
    reference = copy.deepcopy(model)
    features = torch.tensor([[256., 512.], [512., -128.], [-256., 256.], [128., -512.]],
                            dtype=torch.float16, device="cuda")
    inverse = torch.tensor([2, 0, 3, 1], device="cuda")
    coarse = features.clone().requires_grad_()
    skip = features[inverse].clone().requires_grad_()
    coarse_reference = coarse.detach().float().requires_grad_()
    skip_reference = skip.detach().float().requires_grad_()

    def point(coarse_features, skip_features):
        indices = torch.zeros((4, 4), dtype=torch.int32, device="cuda")
        indices[:, 1] = torch.arange(4, device="cuda")
        parent = Point(feat=skip_features, sparse_conv_feat=spconv.SparseConvTensor(
            skip_features, indices, [4, 1, 1], 1))
        return Point(feat=coarse_features, pooling_parent=parent, pooling_inverse=inverse)

    # Both raw projections overflow half precision, although normalization is well-defined.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        assert torch.isinf(model.proj[0](coarse)).any()
        assert torch.isinf(model.proj_skip[0](skip)).any()
    with torch.autocast("cuda", enabled=False):
        assert reference.proj[0](coarse_reference).abs().max() > torch.finfo(torch.float16).max
        expected = reference(point(coarse_reference, skip_reference)).feat
    with torch.autocast("cuda", dtype=torch.float16):
        actual = model(point(coarse, skip))
    assert actual.feat.dtype == torch.float16
    assert torch.isfinite(actual.feat).all() and torch.isfinite(expected).all()
    # Matching branches make their sum exactly twice one output, isolating half rounding.
    torch.testing.assert_close(actual.feat, expected.half(), atol=0., rtol=0.)
    torch.testing.assert_close(actual.sparse_conv_feat.features, actual.feat, atol=0., rtol=0.)
    weights = torch.arange(1, 13, dtype=torch.float32, device="cuda").reshape(4, 3) / 16
    (actual.feat.float() * weights).sum().backward()
    (expected * weights).sum().backward()
    for tensor in (coarse, skip, *model.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    for actual_input, expected_input in ((coarse, coarse_reference), (skip, skip_reference)):
        torch.testing.assert_close(actual_input.grad, expected_input.grad.half(), atol=0., rtol=0.)
    for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, atol=1e-7, rtol=1e-5)


class PointHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.25, .5, -.125], device="cuda"))

    def forward(self, scan):
        x = scan["x"]
        a, b, c = self.weight
        return dict(logits=a*x+b, boundary=(b*x+c).sigmoid(), surface=c*x-a)


@pytest.mark.parametrize("auxiliary_scale", [0., 1.])
def test_auxiliary_scale_preserves_three_view_detection_graph(monkeypatch, auxiliary_scale):
    import src.train as training

    class LinkedHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context = torch.nn.Linear(2, 3)
            self.boundary = torch.nn.Linear(3, 1)
            self.surface = torch.nn.Linear(3, 1)
            self.anomaly = torch.nn.Linear(5, 1)
            self.forward_inputs = []

        def forward(self, scan):
            self.forward_inputs.append(scan["x"].detach().clone())
            features = self.context(scan["x"]).tanh()
            boundary = self.boundary(features).sigmoid()
            surface = self.surface(features)
            logits = self.anomaly(torch.cat((features, boundary, surface), dim=1))
            return dict(logits=logits[:, 0], boundary=boundary[:, 0], surface=surface[:, 0])

    # CPU arithmetic isolates objective semantics from CUDA reduction variability.
    monkeypatch.setattr(training, "to_device", lambda values: values)
    monkeypatch.setattr(training.torch, "autocast", lambda *args, **kwargs: nullcontext())
    torch.manual_seed(981)
    model = LinkedHeads()
    initial = copy.deepcopy(model.state_dict())
    reference = copy.deepcopy(model)
    views = []
    for view in range(3):
        target = dict(labels=torch.tensor([0, 1, -1, 0]),
                      boundary_valid=torch.tensor([True, True, False, True]),
                      surface_valid=torch.tensor([True, True, False, True]),
                      boundary_distance=torch.tensor([.2, .5, 0., 1.]),
                      surface_offset_z=torch.tensor([-.3, .4, 0., -.1]))
        if view:
            target.update(dense_row=torch.arange(4),
                          sampling_consistency_valid=torch.tensor([True, True, False, True]))
        views.append(dict(scan=dict(x=torch.tensor([[.2, -.5], [.8, .1], [1., .4], [-.4, .7]]) + .1*view),
                          target=target))
    sample = dict(world="objective_fixture", frame=0, views=views)
    input_before = copy.deepcopy(sample)
    outputs = [reference(view["scan"]) for view in views]
    # Independent reference explicitly averages the same three detection forwards.
    terms = {"detection": torch.stack([detection_loss(output["logits"], view["target"]["labels"])
                                       for output, view in zip(outputs, views)]).mean()}
    for name, loss, output_key, target_key in (
        ("boundary", boundary_loss, "boundary", "boundary_distance"),
        ("surface", surface_loss, "surface", "surface_offset_z"),
    ):
        terms[name] = torch.stack([loss(output[output_key], view["target"][target_key],
                                        view["target"]["labels"], view["target"][name+"_valid"])
                                   for output, view in zip(outputs, views)]).mean()
    terms["sampling"] = torch.stack([
        sampling_loss(outputs[v]["logits"], outputs[0]["logits"], views[v]["target"]["dense_row"],
                      views[v]["target"]["labels"], views[v]["target"]["sampling_consistency_valid"])
        for v in (1, 2)]).mean()
    objective = terms["detection"]
    if auxiliary_scale:
        objective = objective + sum(SETTINGS["loss_coefficients"][name]*terms[name]
                                    for name in SETTINGS["loss_coefficients"])
    (objective / 4).backward()
    logs = backward_sample(model, sample, torch.amp.GradScaler("cpu", enabled=False),
                           SETTINGS["loss_coefficients"], 4, auxiliary_scale=auxiliary_scale)
    assert len(model.forward_inputs) == 3
    assert_state_close(model.forward_inputs, [view["scan"]["x"] for view in views])
    assert_state_close(sample, input_before)
    assert_state_close(model.state_dict(), initial)
    assert set(dict(model.named_parameters())) == set(dict(reference.named_parameters()))
    for (name, actual), (reference_name, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == reference_name and actual.grad is not None and expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-6)
    for head in (model.boundary, model.surface):
        assert head.weight.grad.abs().sum() > 0
    for name, value in terms.items():
        assert logs[name] == pytest.approx(float(value.detach()), abs=1e-7, rel=1e-6)


@CUDA
@pytest.mark.parametrize("case", ["all", "one_view", "one_pair", "none", "zero_error"])
def test_view_average_matches_reference_loss_gradients_and_summary(case):
    model = PointHead()
    active = {
        "all": ([1,1,1], [0,1,1]),
        "one_view": ([1,0,0], [0,0,0]),
        "one_pair": ([0,0,0], [0,1,0]),
        "none": ([0,0,0], [0,0,0]),
        "zero_error": ([1,1,1], [0,1,1]),
    }
    surface_views, pairs = active[case]
    views = []
    for v in range(3):
        x = torch.tensor([.2, .8, 1.2, -.4]) + (.2*v if not (case == "zero_error" and v == 1) else 0.)
        # An ignored row with a true mask cannot make a view supervised.
        mask = torch.full((4,), bool(surface_views[v]))
        mask[2] = True
        target = dict(labels=torch.tensor([0,1,-1,0]), boundary_valid=mask,
                      surface_valid=mask.clone(), boundary_distance=torch.full((4,), .2),
                      surface_offset_z=torch.full((4,), -.5))
        if v:
            target.update(dense_row=torch.arange(4), sampling_consistency_valid=torch.tensor([bool(pairs[v])]*4))
            target["sampling_consistency_valid"][2] = True
        if case == "zero_error" and v == 0:
            with torch.no_grad():
                output = model(dict(x=x.cuda()))
            target["boundary_distance"] = output["boundary"].cpu()
            target["surface_offset_z"] = output["surface"].cpu()
        views.append(dict(scan=dict(x=x), target=target))
    sample = dict(world="loss_fixture", frame=0, views=views)
    weights = loss_view_weights(views)
    for v, weight in enumerate(weights):
        assert weight["detection"] == 1/3
        assert weight["boundary"] == weight["surface"] == surface_views[v]/max(sum(surface_views), 1)
        assert weight["sampling"] == pairs[v]/max(sum(pairs), 1)

    reference = copy.deepcopy(model)
    outputs = [reference(to_device(view["scan"])) for view in views]
    terms = {name:[] for name in ("detection", "boundary", "surface", "sampling")}
    for v, (view, output) in enumerate(zip(views, outputs)):
        target = to_device(view["target"])
        labels = target["labels"]
        terms["detection"].append(detection_loss(output["logits"], labels))
        if surface_views[v]:
            terms["boundary"].append(boundary_loss(output["boundary"], target["boundary_distance"], labels, target["boundary_valid"]))
            terms["surface"].append(surface_loss(output["surface"], target["surface_offset_z"], labels, target["surface_valid"]))
        if pairs[v]:
            terms["sampling"].append(sampling_loss(output["logits"], outputs[0]["logits"],
                                      target["dense_row"], labels, target["sampling_consistency_valid"]))
    if case == "zero_error":
        assert all(terms[name][0].item() == 0 for name in ("boundary", "surface", "sampling"))
    # Independent reference: explicitly average the nonempty lists, including zero errors.
    expected = {name:torch.stack(values).mean() if values else outputs[0]["logits"].sum()*0
                for name, values in terms.items()}
    objective = expected["detection"] + sum(SETTINGS["loss_coefficients"][name]*expected[name]
                                             for name in SETTINGS["loss_coefficients"])
    (objective / SETTINGS["gradient_accumulation"]).backward()
    logs = backward_sample(model, sample, torch.amp.GradScaler("cuda", enabled=False),
                           SETTINGS["loss_coefficients"], SETTINGS["gradient_accumulation"])
    torch.testing.assert_close(model.weight.grad, reference.weight.grad, atol=1e-7, rtol=1e-6)
    summary = check_summary(model, [sample])[0]["losses"]
    for name, value in expected.items():
        assert logs[name] == pytest.approx(float(value.detach()), abs=1e-7, rel=1e-6)
        assert summary[name] == pytest.approx(float(value.detach()), abs=1e-7, rel=1e-6)


@CUDA
@pytest.mark.parametrize("gradient", ["ordinary", "finite_overflow", "inf", "nan"])
def test_update_counts_only_numerically_valid_optimizer_steps(gradient):
    model = torch.nn.Linear(2, 1, bias=False).cuda()
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer, scaler = optimizer_for(model, SETTINGS)
    scaler.scale(model.weight.sum()).backward()
    assert update(model, optimizer, scaler, SETTINGS, 1)["success"]
    before = model.weight.detach().clone()
    adam_before = copy.deepcopy(optimizer.state_dict())
    scale_before = scaler.get_scale()
    scaler.scale(model.weight.sum()).backward()
    values = dict(ordinary=[3.,4.], finite_overflow=[1e20,1e20], inf=[float("inf"),1.], nan=[float("nan"),1.])
    model.weight.grad.copy_(torch.tensor([values[gradient]], device="cuda") * scale_before)
    if gradient == "finite_overflow":
        with pytest.raises(FloatingPointError, match="finite gradient elements produced a nonfinite total norm"):
            update(model, optimizer, scaler, SETTINGS, 2)
        assert torch.equal(model.weight, before)
        assert bool(torch.isfinite(model.weight.grad).all()) and bool((model.weight.grad != 0).all())
        assert_state_close(optimizer.state_dict(), adam_before)
        assert scaler.get_scale() == scale_before
        return
    result = update(model, optimizer, scaler, SETTINGS, 2)
    if gradient == "ordinary":
        assert result["success"] and result["gradient_norm"] == 5. and result["skip_reason"] is None
        assert not torch.equal(model.weight, before) and scaler.get_scale() == scale_before
        old = adam_before["state"][0]
        clipped = torch.tensor([[3.,4.]], device="cuda") / (5. + 1e-6)
        actual = optimizer.state_dict()["state"][0]
        torch.testing.assert_close(actual["exp_avg"], .9*old["exp_avg"] + .1*clipped)
        torch.testing.assert_close(actual["exp_avg_sq"], .999*old["exp_avg_sq"] + .001*clipped.square())
    else:
        assert not result["success"] and result["gradient_norm"] is None
        assert result["skip_reason"] == "nonfinite_gradient_elements"
        assert torch.equal(model.weight, before)
        assert_state_close(optimizer.state_dict()["state"], adam_before["state"])
        assert scaler.get_scale() == scale_before * scaler.get_backoff_factor()


def test_next_accumulated_update_after_restore_is_exact_on_cpu(tmp_path):
    def construct():
        model = torch.nn.Sequential(torch.nn.Linear(4,6), torch.nn.Dropout(.25), torch.nn.Linear(6,2)).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"],
                                      weight_decay=SETTINGS["weight_decay"], betas=SETTINGS["betas"])
        return model, optimizer, torch.amp.GradScaler("cpu", init_scale=SETTINGS["initial_loss_scale"])

    order = [7,2,8,5,11,4,9,3]
    def group(model, optimizer, scaler, visited):
        optimizer.zero_grad(set_to_none=True)
        identities = order[visited:visited+4]
        for index in identities:
            x = torch.tensor([[index/20, random.random(), np.random.rand(), 1.]], dtype=torch.float64)
            scaler.scale(model(x).square().mean()/4).backward()
        return dict(indices=identities, **update(model, optimizer, scaler, SETTINGS, visited//4+1))

    seed_all(7654)
    model, optimizer, scaler = construct()
    first = group(model, optimizer, scaler, 0)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, optimizer, scaler, SETTINGS, order, 4, int(first["success"]), [first])
    second = group(model, optimizer, scaler, 4)
    expected_model = copy.deepcopy(model.state_dict())
    expected_adam = copy.deepcopy(optimizer.state_dict())
    expected_scaler = scaler.state_dict()
    expected_rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state_all())
    model, optimizer, scaler = construct()
    random.random()
    np.random.rand()
    torch.rand(8)
    if torch.cuda.is_available():
        torch.rand(8, device="cuda")
    state = restore_checkpoint(path, model, optimizer, scaler, SETTINGS, order)
    assert state["visited"] == 4 and state["attempted_updates"] == 1 and not state["complete"]
    resumed = group(model, optimizer, scaler, state["visited"])
    assert resumed == second and resumed["success"] and first["success"]
    assert_state_close(model.state_dict(), expected_model)
    assert_state_close(optimizer.state_dict(), expected_adam)
    assert_state_close(scaler.state_dict(), expected_scaler)
    assert_state_close(dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                           cuda=torch.cuda.get_rng_state_all()), expected_rng)


def test_paired_initial_restores_complete_state_and_rejects_trained_start(tmp_path):
    seed_all(483)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.)
    config = dict(training=dict(auxiliary_scale=1., run_directory="results/paired"), identity="same_inputs")
    order = [8, 3, 11, 2]
    path = tmp_path / "initial.pt"
    save_checkpoint(path, model, optimizer, scaler, config, order, 0, 0, [])
    initial = torch.load(path, map_location="cpu", weights_only=False)

    def current():
        return copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  scaler=scaler.state_dict(), rng=dict(python=random.getstate(),
                                  numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                  cuda=torch.cuda.get_rng_state_all())))

    for auxiliary_scale in (0., 1.):
        # Change every restorable component; both arms must recover the same actual state.
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model(torch.rand(5, 3)).square().mean()).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        random.random()
        np.random.rand()
        if torch.cuda.is_available():
            torch.rand(4, device="cuda")
        arm = copy.deepcopy(config)
        arm["training"]["auxiliary_scale"] = auxiliary_scale
        state = restore_checkpoint(path, model, optimizer, scaler, arm, order, initial=True)
        assert state["visited"] == state["attempted_updates"] == state["successful_updates"] == 0
        assert state["order"] == order and state["history"] == [] and state["complete"] is False
        assert_state_close(current(), {key: initial[key] for key in ("model", "optimizer", "scaler", "rng")})

    unchanged = current()
    for key, value in (("visited", 1), ("attempted_updates", 1), ("successful_updates", 1),
                       ("history", [{"attempt": 1}]), ("complete", True)):
        invalid = copy.deepcopy(initial)
        invalid[key] = value
        # A failed initial-state guard must not partially replace live model weights.
        next(iter(invalid["model"].values())).add_(10.)
        torch.save(invalid, tmp_path / "invalid.pt")
        with pytest.raises(ValueError):
            restore_checkpoint(tmp_path / "invalid.pt", model, optimizer, scaler, config, order, initial=True)
        assert_state_close(current(), unchanged)
    wrong = copy.deepcopy(config)
    wrong["training"]["run_directory"] = "another_run"
    with pytest.raises(ValueError):
        restore_checkpoint(path, model, optimizer, scaler, wrong, order, initial=True)
    with pytest.raises(ValueError):
        restore_checkpoint(path, model, optimizer, scaler, config, list(reversed(order)), initial=True)
    # Ordinary continuation still forbids even the auxiliary-scale-only difference.
    wrong = copy.deepcopy(config)
    wrong["training"]["auxiliary_scale"] = 0.
    with pytest.raises(ValueError):
        restore_checkpoint(path, model, optimizer, scaler, wrong, order)
    assert_state_close(current(), unchanged)
