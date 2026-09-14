from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn

from src.model import load_config
from src.train import (_gradient_comparison, batch_loss, keep_loss, load_checkpoint,
                       tail_loss, Requests, evaluation_state, validate_initial_state, validate_resume_state)


def test_pretrained_transfer_is_complete_and_does_not_touch_new_modules():
    from src.model import transfer_backbone
    model = nn.Module()
    model.backbone = nn.Sequential(nn.Linear(4, 72), nn.BatchNorm1d(72))
    model.head = nn.Linear(72, 1)
    fresh = deepcopy(model.head.state_dict())
    weights = {"module.backbone." + k: v.clone() + 1 for k, v in model.backbone.state_dict().items()}
    weights.update({"module.seg_head.weight": torch.ones(16, 72), "module.seg_head.bias": torch.ones(16)})
    report = transfer_backbone(model, weights)
    assert report["exact_tensor_equality"] and report["loaded_keys"] == len(model.backbone.state_dict())
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, weights["module.backbone." + name], rtol=0, atol=0)
    for name, value in model.head.state_dict().items():
        torch.testing.assert_close(value, fresh[name], rtol=0, atol=0)
    missing = dict(weights)
    missing.pop("module.backbone.1.running_mean")
    with pytest.raises(ValueError, match="state keys differ"):
        transfer_backbone(model, missing)
    with pytest.raises(ValueError, match="state keys differ"):
        transfer_backbone(model, dict(weights, extra=torch.ones(1)))
    invalid = dict(weights)
    invalid["module.backbone.0.weight"] = torch.zeros(72, 5)
    with pytest.raises(ValueError, match="invalid official tensor"):
        transfer_backbone(model, invalid)


def test_pretrained_optimizer_starts_empty_with_disjoint_learning_rates():
    from src.train import make_optimizer
    from src.model import validate_config
    model = nn.Module()
    model.backbone, model.head = nn.Linear(4, 72), nn.Linear(72, 1)
    config = load_config("protocol/pretrain.json")
    optimizer = make_optimizer(model, config)
    assert not optimizer.state
    assert [g["lr"] for g in optimizer.param_groups] == [2e-5, 2e-4]
    grouped = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(grouped) == len(set(grouped)) and set(grouped) == {id(p) for p in model.parameters()}
    assert all(p.requires_grad for p in model.parameters())
    assert len(make_optimizer(model, load_config()).param_groups) == 1
    config["model"]["intensity_transform"] = "divide_by_255"
    with pytest.raises(ValueError, match="input rule"):
        validate_config(config)


def test_raw_intensity_statistics_keep_values_above_one_and_ignore_labels():
    from types import SimpleNamespace
    from src.train import intensity_summary
    values = np.array([.1, .2, 1.6, .4], dtype=np.float32)
    sequence = [SimpleNamespace(xyzi=np.column_stack((np.ones((4, 3)), values)), real_slots=np.arange(4), frame_id=7)]
    result = intensity_summary(sequence)
    assert result["returns"] == 4 and result["above_1"] == 1
    assert result["max"] == float(values.max())
    assert result["quantiles"]["q50"] == np.quantile(values.astype(np.float64), .5)


def test_pretrained_fit_cannot_silently_start_from_random_weights(tmp_path):
    from src.train import fit
    with pytest.raises(ValueError, match="saved initialized state"):
        fit(load_config("protocol/pretrain.json"), tmp_path / "no_data", 1, tmp_path / "run")


def test_short_initialization_allows_only_sampling_scope_change():
    config = load_config()
    saved = dict(step=0, optimizer=dict(state={}), config=deepcopy(config), samples=[["world", 11]])
    config["scope"] = "finite full-pool learning"
    validate_initial_state(saved, config, [["world", 11]])
    for altered in (dict(saved, step=256), dict(saved, optimizer=dict(state={0: {}}))):
        with pytest.raises(ValueError, match="untrained step-zero"):
            validate_initial_state(altered, config, [["world", 11]])
    config["loss"]["keep_mode"] = "worst"
    with pytest.raises(ValueError, match="definition"):
        validate_initial_state(saved, config, [["world", 11]])


def test_paired_initialization_exception_is_explicit_and_cannot_hide_other_changes():
    config = load_config()
    config["loss"]["keep_mode"] = "worst"
    saved_config = deepcopy(config)
    saved_config["loss"]["keep_mode"] = "mean"
    saved = dict(step=0, optimizer=dict(state={}), config=saved_config, samples=[["world", 11]])
    changes = {"loss.keep_mode": {"from": "mean", "to": "worst"}}
    validate_initial_state(saved, config, saved["samples"], changes)
    assert saved["config"]["loss"]["keep_mode"] == "mean"
    with pytest.raises(ValueError, match="definition"):
        validate_initial_state(saved, config, saved["samples"])
    for section, key, value in (("loss", "tail_weight", 1), ("training", "seed", 1),
                                 ("model", "condition_modulation", False)):
        changed = deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError, match="definition"):
            validate_initial_state(saved, changed, saved["samples"], changes)
    with pytest.raises(ValueError, match="only the declared"):
        validate_initial_state(saved, config, saved["samples"], dict(changes, seed=1))
    with pytest.raises(ValueError, match="untrained step-zero"):
        validate_initial_state(dict(saved, step=1024), config, saved["samples"], changes)


def test_full_pool_requests_preserve_original_draw_stream_and_resume():
    config = load_config()
    probabilities = np.array([.03, .11, .36, .5])
    actual = list(Requests(probabilities, config, 1024))
    cdf = np.cumsum(probabilities)
    expected = [int(np.searchsorted(cdf, np.random.default_rng(
        np.random.SeedSequence([config["training"]["seed"], 7, draw])).random(), side="right"))
        for draw in range(2048)]
    assert [r[0] for r in actual] == expected
    assert [r[1] for r in actual] == list(range(2048))
    assert list(Requests(probabilities, config, 1024, start=256)) == actual[512:]
    cumulative = list(Requests(probabilities, config, 8000))
    assert list(Requests(probabilities, config, 8000, start=4000)) == cumulative[8000:]
    assert all(r[2] for r in cumulative[8000:])


def test_resume_rejects_partial_buffers_and_any_recipe_or_probability_change():
    config = load_config()
    probabilities = np.array([.25, .75])
    saved = dict(step=4000, config=deepcopy(config), samples=[["a", 1], ["b", 2]],
                 probabilities=torch.from_numpy(probabilities), experiment={"format": "ajae-staged-learning"})
    assert validate_resume_state(saved, config, saved["samples"], probabilities, saved["experiment"]) == 4000
    with pytest.raises(ValueError, match="partial failure"):
        validate_resume_state(dict(saved, failure=""), config, saved["samples"], probabilities, saved["experiment"])
    changed = deepcopy(config)
    changed["loss"]["keep_mode"] = "worst"
    for candidate, distribution in ((changed, probabilities), (config, probabilities[::-1].copy())):
        with pytest.raises(ValueError, match="resume configuration"):
            validate_resume_state(saved, candidate, saved["samples"], distribution, saved["experiment"])


def test_resume_201_removal_is_explicit_and_cannot_change_training_or_val19():
    config = load_config()
    probabilities = np.array([.25, .75])
    declaration = dict(format="ajae-staged-learning", maximum_updates=32000,
        evaluation=dict(full_synthetic_steps=[4000, 32000], full_val19_steps=[4000, 8000, 16000, 32000],
                        real_steps=[4000, 8000, 16000, 32000], synthetic="all 201 worlds"))
    saved = dict(step=4000, config=deepcopy(config), samples=[["a", 1], ["b", 2]],
                 probabilities=torch.from_numpy(probabilities), experiment=declaration)
    requested = deepcopy(declaration)
    requested["evaluation"].update(full_synthetic_steps=[], synthetic_splits=["train"], synthetic="206 only")
    with pytest.raises(ValueError, match="resume configuration"):
        validate_resume_state(saved, config, saved["samples"], probabilities, requested)
    assert validate_resume_state(saved, config, saved["samples"], probabilities, requested, without_201=True) == 4000
    assert saved["experiment"] == declaration and "synthetic_splits" not in declaration["evaluation"]
    for key, value in (("full_val19_steps", [8000]), ("real_steps", []), ("threshold", .5)):
        changed = deepcopy(requested)
        changed["evaluation"][key] = value
        with pytest.raises(ValueError, match="resume configuration"):
            validate_resume_state(saved, config, saved["samples"], probabilities, changed, without_201=True)
    changed_config = deepcopy(config)
    changed_config["loss"]["keep_mode"] = "worst"
    for candidate, distribution in ((changed_config, probabilities), (config, probabilities[::-1].copy())):
        with pytest.raises(ValueError, match="resume configuration"):
            validate_resume_state(saved, candidate, saved["samples"], distribution, requested, without_201=True)
    with pytest.raises(ValueError, match="no 201 evaluation"):
        validate_resume_state(saved, config, saved["samples"], probabilities, declaration, without_201=True)


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
    config["loss"].update(keep_mode="worst", tail_weight=1.)
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
    config["loss"].update(keep_mode="worst", tail_weight=1.)
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


def test_normalization_control_changes_only_batchnorm_and_restores_state():
    from src.train import normalization_mode
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(.8)).eval()
    saved = deepcopy(model.state_dict())
    x = torch.tensor([[5., 8.], [7., 12.], [9., 16.]])
    with normalization_mode(model):
        ordinary = model(x)
    with normalization_mode(model, current_scan=True):
        assert not model.training and model[0].training and not model[1].training
        current = model(x)
    assert not torch.allclose(ordinary, current)
    torch.testing.assert_close(current.mean(0), torch.zeros(2), atol=1e-6, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def test_detached_numerics_do_not_change_gradients_or_count_recomputation():
    from src.train import _Numerics
    torch.manual_seed(53)
    model = SmallModel().train()
    reference = deepcopy(model)
    rows, config = batch_fixture(), load_config()
    expected, _ = batch_loss(reference, rows, config, 201)
    expected.backward()
    with _Numerics(model) as trace:
        actual, _ = batch_loss(model, rows, config, 201)
        trace.enabled = False
        before = trace.summary()
        actual.backward()
        assert trace.summary() == before
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_diagnostic_selection_keeps_saved_state_and_log_scopes_separate():
    from src.train import diagnostic_batches, optimization_log_summary
    rows = [dict(step=i + 1, auxiliary_fraction=1., gradient_norm=float(i + 1),
                 anomaly_queries=0 if i == 0 else i, detection=.1, keep=.2)
            for i in range(10)]
    selected = diagnostic_batches(rows, 8)
    assert max(c["record"]["step"] for c in selected) == 9
    assert all(c["record"]["step"] != 10 for c in selected)
    assert selected[-1]["reasons"] == ["next_update_with_saved_preupdate_state"]
    assert optimization_log_summary(rows)["by_anomaly_count"]["zero"]["count_gradient_spearman"] is None


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
