from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn

from src.model import load_config
from src.train import (_gradient_comparison, batch_loss, keep_loss, load_checkpoint,
                       tail_loss, Requests, evaluation_state)


def test_micro_passes_balance_exactly_and_resume_keeps_draws():
    config = load_config()
    samples = np.arange(32) * 13
    requests = list(Requests(None, config, 256, samples=samples))
    assert len(requests) == 512
    for start in range(0, len(requests), 32):
        assert sorted(r[0] for r in requests[start:start + 32]) == list(samples)
    assert all(sum(r[0] == sample for r in requests) == 16 for sample in samples)
    assert [r[1] for r in requests] == list(range(512))
    assert not any(r[2] for r in requests[:202])
    assert all(r[2] for r in requests[202:])
    assert list(Requests(None, config, 256, start=117, samples=samples)) == requests[234:]


def test_evaluation_restores_random_streams_buffers_and_training_mode():
    model = nn.BatchNorm1d(2).train()
    cpu, numpy, python = torch.get_rng_state(), np.random.get_state(), random.getstate()
    buffers = [b.clone() for b in model.buffers()]
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        with evaluation_state(model):
            assert not model.training and not torch.is_grad_enabled()
            model.running_mean.add_(7)
            torch.rand(3), np.random.rand(3), random.random()
            raise RuntimeError("evaluation interrupted")
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), cpu, rtol=0, atol=0)
    np.testing.assert_equal(np.random.get_state(), numpy)
    assert random.getstate() == python
    for actual, expected in zip(model.buffers(), buffers, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class SmallModel(nn.Module):
    """Shared graph nodes and BatchNorm mimic the diagnostic interface on CPU."""

    def __init__(self):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.GELU())
        self.point = nn.Linear(4, 6)
        self.base_head = nn.Linear(12, 1)
        self.relation_head = nn.Linear(6, 1)
        self.calls = 0

    def forward(self, scan, query, *, return_features=False):
        self.calls += 1
        h, z = self.context(scan["xyzi"]), self.point(scan["xyzi"])
        score = (self.base_head(torch.cat((h[query], z[query]), -1))
                 + self.relation_head(torch.tanh(h[query] + z[query]))).squeeze(-1)
        return dict(score=score, context=h, point=z) if return_features else score


def batch_fixture():
    rows = []
    for frame in (11, 12):
        original = torch.randn(10, 4)
        inserted = original.clone()
        inserted[[1, 5]] += .5
        rows.append(dict(scan=dict(xyzi=inserted), original=dict(xyzi=original),
            query=torch.tensor([0, 1, 3, 5, 7]), detection_index=torch.tensor([0, 1, 2, 3]),
            target=torch.tensor([0, 1, 0, 1]), keep_index=torch.tensor([0, 4]),
            original_query=torch.tensor([0, 7]), frame=frame))
    return rows


def test_loss_details_preserve_loss_tail_selection_gradients_and_batchnorm():
    torch.manual_seed(19)
    config = load_config()
    config["loss"]["pairs_per_tail"] = 17
    rows, model = batch_fixture(), SmallModel().train()
    reference = deepcopy(model)
    expected, expected_stats = batch_loss(reference, rows, config, 200)
    actual, stats, observed = batch_loss(model, rows, config, 200, details=True)
    assert model.calls == reference.calls == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stats == expected_stats

    scores = torch.cat([observed["scores"][2 * i][row["detection_index"]]
                        for i, row in enumerate(rows)])
    frames = torch.cat([torch.full_like(row["target"], row["frame"]) for row in rows])
    generator = torch.Generator().manual_seed(config["training"]["seed"] + 31 + 200)
    selected = {}
    tail, tail_stats = tail_loss(scores, observed["target"], frames, config["loss"],
                                 generator, selections=selected)
    torch.testing.assert_close(tail, observed["components"]["tail"], rtol=0, atol=0)
    assert tail_stats == {key: stats[key] for key in tail_stats}
    for name in ("high_normal", "low_anomaly"):
        torch.testing.assert_close(selected[name], observed["tail"][name], rtol=0, atol=0)
    for actual_pair, expected_pair in zip(observed["tail"]["pairs"], selected["pairs"], strict=True):
        for actual_index, expected_index in zip(actual_pair, expected_pair, strict=True):
            torch.testing.assert_close(actual_index, expected_index, rtol=0, atol=0)

    actual.backward()
    expected.backward()
    for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)
    for actual_buffer, expected_buffer in zip(model.buffers(), reference.buffers(), strict=True):
        torch.testing.assert_close(actual_buffer, expected_buffer, rtol=0, atol=0)
    assert model.context[1].num_batches_tracked == 4


def test_shared_feature_diagnostics_and_mean_worst_use_the_same_forwards():
    torch.manual_seed(23)
    config, rows, model = load_config(), batch_fixture(), SmallModel().train()
    config["loss"]["pairs_per_tail"] = 17
    _, _, observed = batch_loss(model, rows, config, 200, details=True)
    assert model.calls == 4
    before = torch.cat(observed["scores"][1::2])
    after = torch.cat([observed["scores"][2 * i][row["keep_index"]] for i, row in enumerate(rows)])
    for mode in ("mean", "worst"):
        component = "keep_mean" if mode == "mean" else "keep"
        torch.testing.assert_close(observed["components"][component], keep_loss(before, after, mode), rtol=0, atol=0)
    buffers = [buffer.clone() for buffer in model.buffers()]
    shared = [*observed["context"], *observed["point"]]
    for component in observed["components"].values():
        gradient = torch.autograd.grad(component, [*model.parameters(), *shared],
                                      allow_unused=True, retain_graph=True)
        assert all(torch.isfinite(value).all() for value in gradient if value is not None)
        assert any(value is not None and torch.count_nonzero(value) for value in gradient[-len(shared):])
    for current, saved in zip(model.buffers(), buffers, strict=True):
        torch.testing.assert_close(current, saved, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_comparison_has_no_direction_for_zero_vectors():
    result = _gradient_comparison(dict(zero=torch.zeros(2), a=torch.tensor([3., 4.]),
                                       opposite=torch.tensor([-3., -4.])))
    assert result["norm"] == dict(zero=0., a=5., opposite=5.)
    assert result["cosine"] == {"zero:a": None, "zero:opposite": None, "a:opposite": -1.}


def test_checkpoint_rejects_missing_preprocessing_before_model_construction(tmp_path, monkeypatch):
    import src.train as train

    def forbidden_model(*args, **kwargs):
        raise AssertionError("invalid checkpoint must fail before creating a model")

    monkeypatch.setattr(train, "AJAE", forbidden_model)
    path = tmp_path / "incomplete.pt"
    torch.save(dict(format="ajae-v1-checkpoint"), path)
    with pytest.raises(ValueError, match="saved preprocessing"):
        load_checkpoint(path, device="cpu")


def test_disabled_tail_preserves_mean_objective_and_gradients():
    torch.manual_seed(91)
    config, rows, model = load_config(), batch_fixture(), SmallModel().train()
    config["loss"].update(keep_mode="mean", tail_weight=0.)
    reference = deepcopy(model)
    expected, _, observed = batch_loss(reference, rows, config, 128, details=True)
    actual, stats = batch_loss(model, rows, config, 128)
    torch.testing.assert_close(actual, observed["components"]["detection"]
        + .28 * observed["components"]["keep_mean"], rtol=0, atol=0)
    assert stats["pairs"] == 0 and stats["tail"] == 0
    actual.backward()
    expected.backward()
    for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)
