"""Training regressions: supervision weighting, valid updates and exact continuation."""

import copy
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


class PointHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.25, .5, -.125], device="cuda"))

    def forward(self, scan):
        x = scan["x"]
        a, b, c = self.weight
        return dict(logits=a*x+b, boundary=(b*x+c).sigmoid(), surface=c*x-a)


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
