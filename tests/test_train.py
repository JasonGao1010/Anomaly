from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn

from src.model import load_config
from src.train import (_gradient_comparison, batch_loss, keep_loss, load_checkpoint,
                       tail_loss, Requests, evaluation_state, validate_initial_state, validate_resume_state)


def test_group_queries_separate_spare_seats_empty_weights_and_overlap():
    from src.train import grouped_queries
    rng = np.random.default_rng(21)
    base, small, empty = np.arange(20), np.array([0]), np.array([], dtype=int)
    groups, weights = grouped_queries(base, small, empty, [4, 2, 2], [.5, .25, .25], rng)
    assert list(map(len, groups)) == [7, 1, 0] and weights == [.75, .25, 0.]
    groups, weights = grouped_queries(np.arange(3), np.array([0]), np.array([0]),
                                      [4, 2, 2], [.5, .25, .25], rng)
    assert list(map(len, groups)) == [3, 1, 1] and weights == [.5, .25, .25]
    assert [g.tolist().count(0) for g in groups] == [1, 1, 1]
    groups, weights = grouped_queries(empty, empty, empty, [4, 2, 2], [.5, .25, .25], rng)
    assert not any(map(len, groups)) and weights == [0., 0., 0.]


@pytest.mark.parametrize("positive_counts", [(2, 0), (1, 3), (0, 0)])
def test_v2_frame_risk_and_gradients_match_explicit_formula(positive_counts):
    from src.train import experiment_config, load_experiment
    config = experiment_config(load_experiment("protocol/v2.json"))
    rows = batch_fixture()
    for row, count in zip(rows, positive_counts, strict=True):
        row["target"][:] = 0
        row["target"][torch.tensor([1, 2, 3])[:count]] = 1
        normal = row["detection_index"][row["target"] == 0]
        row.update(normal_groups=[normal, normal[:1], normal[:0]], normal_weights=[.75, .25, 0.],
                   keep_groups=[torch.tensor([0, 1]), torch.tensor([0]), torch.tensor([0])],
                   keep_weights=[.5, .25, .25])
    torch.manual_seed(39)
    model = SmallModel().double().train()
    for row in rows:
        row["scan"]["xyzi"] = row["scan"]["xyzi"].double()
        row["original"]["xyzi"] = row["original"]["xyzi"].double()
    loss, stats, details = batch_loss(model, rows, config, 0, details=True)
    assert model.calls == 4 and stats["positive_frames"] == sum(n > 0 for n in positive_counts)
    assert stats["auxiliary_fraction"] == 1.
    score = details["scores"]
    normal = []
    kept = []
    for row, after, before in zip(rows, score[::2], score[1::2], strict=True):
        values = torch.nn.functional.softplus(after)
        normal.append(.75 * values[row["normal_groups"][0]].mean() + .25 * values[row["normal_groups"][1]].mean())
        pairs = .5 * (torch.nn.functional.softplus(before) + values[row["keep_index"]])
        kept.append(.5 * pairs.mean() + .5 * pairs[0])
    positives = [torch.nn.functional.softplus(-after[row['detection_index']][row['target'] == 1]).mean()
                 for row, after in zip(rows, score[::2], strict=True) if (row['target'] == 1).any()]
    anomaly = sum(positives) / len(positives) if positives else score[0].sum() * 0
    expected = .25 * sum(normal) + .5 * anomaly + .5 * sum(kept)
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-14)
    parameters = list(model.parameters())
    actual_grad = torch.autograd.grad(loss, parameters, retain_graph=True)
    reference_grad = torch.autograd.grad(expected, parameters)
    for actual, reference in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-14)


def test_v2_schedule_and_request_resume_use_local_updates():
    from src.train import auxiliary_fraction, experiment_config, learning_rate_factor, load_experiment
    config = experiment_config(load_experiment("protocol/v2.json"))
    schedule = config["training"]["learning_rate_schedule"]
    assert [learning_rate_factor(i, schedule) for i in (1, 50, 2048)] == [.1, 1., .1]
    probabilities = np.array([.2, .3, .5])
    whole = list(Requests(probabilities, config, 2048))
    assert whole[2048:] == list(Requests(probabilities, config, 2048, start=1024))
    assert all(need for _, _, need in whole) and auxiliary_fraction(0, config["training"]) == 1.


def test_v2_queries_keep_identity_overlap_and_missing_return_neighborhood():
    from src.data import FrozenFrame, low_support_slots
    from src.scene import PointLabels, make_source_frame
    from src.train import experiment_config, load_experiment, query_rows
    xyzi = np.zeros((13, 4), np.float32)
    xyzi[:, 0], xyzi[:, 3] = np.arange(10, 23), .3
    xyzi[1] = xyzi[0]
    xyzi[12, :3] = 0
    packed = np.full(13, 40, np.uint32)
    targets = np.full(13, 8, np.uint8)
    targets[[11, 12]] = 255
    def source(points, labels, target):
        return make_source_frame(7, points, np.eye(4), PointLabels(labels,
            (labels & 65535).astype(np.uint16), (labels >> 16).astype(np.uint16), target),
            partition="train", sequence_id=206)
    original = source(xyzi, packed, targets)
    current, labels, current_target = xyzi.copy(), packed.copy(), targets.copy()
    current[[2, 3], 0] = [11., 11.5]
    labels[[2, 3]] = 2 | (60001 << 16)
    current[4], labels[4], current_target[[2, 3, 4]] = 0, 0, 255
    inserted, occluded = np.zeros(13, bool), np.zeros(13, bool)
    inserted[[2, 3]], occluded[[2, 3, 4]] = True, True
    frozen = FrozenFrame(source(current, labels, current_target), 'a' * 64, inserted, occluded)
    t = experiment_config(load_experiment('protocol/v2.json'))['training']
    sparse = low_support_slots(original, 2., 8)
    queries = query_rows(frozen, original, t, np.random.default_rng(5), sparse_slots=sparse)
    slots = frozen.source.real_slots[queries['query'].numpy()]
    assert len(slots) == len(set(slots)) and 12 not in slots and 11 not in slots
    np.testing.assert_array_equal(slots[queries['detection_index']][queries['target'] == 1], [2, 3])
    assert 5 in slots[queries['normal_groups'][2]]  # Its nearby changed point is the lost return at slot4.
    for idx in queries['normal_groups']:
        assert np.all(frozen.anomaly_target[slots[idx]] == 0)
    np.testing.assert_array_equal(original.real_slots[queries['original_query']], queries['keep_slot'])
    np.testing.assert_array_equal(slots[queries['keep_index']], queries['keep_slot'])
    assert 0 in slots[queries['normal_groups'][0]] and 0 in slots[queries['normal_groups'][1]]
    assert queries['normal_weights'] == [.5, .25, .25]


def test_v2_resume_rejects_changed_condition_slots_and_parent_recipe(monkeypatch):
    from src.train import (experiment_config, initialize_stage, load_experiment,
                           make_optimizer, schedule_state, set_learning_rates)
    experiment = load_experiment('protocol/v2.json')
    config = experiment_config(experiment)
    parent = experiment_config(load_experiment('protocol/finetune.json'))
    parent_experiment = load_experiment('protocol/keep.json', 'mean')
    model = nn.Module()
    model.backbone = nn.Sequential(nn.Linear(1, 2), nn.BatchNorm1d(2))
    model.head = nn.Linear(2, 1)
    optimizer = make_optimizer(model, parent)
    set_learning_rates(optimizer, parent, 1152)
    identities, probabilities = [['a', 1]], np.array([1.])
    saved = dict(format='ajae-v1-checkpoint', step=1152, config=parent, experiment=parent_experiment,
        model=deepcopy(model.state_dict()), optimizer=optimizer.state_dict(), samples=identities,
        probabilities=torch.from_numpy(probabilities), preprocessing={},
        scheduler_state=schedule_state(optimizer, parent, 1152))
    # Calibration validation is covered by the real parent check; this fixture isolates inheritance.
    monkeypatch.setattr('src.train.ScanTransform', lambda *args, **kwargs: None)
    fresh = initialize_stage(model, saved, config, experiment)
    assert not fresh.state and [g['lr'] for g in fresh.param_groups] == pytest.approx([5e-7, 5e-6])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved['model'][key], rtol=0, atol=0)
    conditions = {'1': ['source', [1, 3]]}
    local = dict(config=config, step=0, optimizer=fresh.state_dict(), experiment=experiment, samples=identities,
        probabilities=torch.from_numpy(probabilities), condition_sources=conditions,
        scheduler_state=schedule_state(fresh, config, 0))
    assert validate_resume_state(local, config, identities, probabilities, experiment, condition_sources=conditions) == 0
    with pytest.raises(ValueError, match='slot sets changed'):
        validate_resume_state(local, config, identities, probabilities, experiment, condition_sources={'1': ['source', [1]]})
    changed = deepcopy(config)
    changed['model']['condition_modulation'] = False
    with pytest.raises(ValueError, match='only its declared'):
        initialize_stage(model, saved, changed, experiment)
    with pytest.raises(ValueError, match='mean1152'):
        initialize_stage(model, dict(saved, step=1024), config, experiment)


def test_v2_fit_rejects_unresolved_physical_pool_before_creating_model(tmp_path, monkeypatch):
    import json
    from src.train import experiment_config, fit, load_experiment
    experiment = load_experiment('protocol/v2.json')
    config = experiment_config(experiment)
    config['training']['sampling'] = str(tmp_path / 'conditions.npz')
    (tmp_path / 'conditions.json').write_text(json.dumps(dict(parameters=config['training']['conditions'],
        regions_satisfied=True, collision=dict(certified=False))))
    monkeypatch.setattr('src.train.TrainingFrames', lambda *a, **kw: None)
    monkeypatch.setattr('src.train.torch.load', lambda *a, **kw: dict(format='ajae-v1-checkpoint', preprocessing={}))
    def forbidden_model(*a, **kw):
        raise AssertionError('invalid physical pool reached model construction')
    monkeypatch.setattr('src.train.AJAE', forbidden_model)
    with pytest.raises(ValueError, match='unresolved Euclidean placement'):
        fit(config, tmp_path, 2048, tmp_path / 'fit', experiment=experiment)
    assert not (tmp_path / 'fit').exists()


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


def test_finetuning_schedule_boundaries_and_strict_continuation():
    from src.train import (experiment_config, learning_rate_factor, learning_rates, load_experiment,
                           make_optimizer, schedule_state, set_learning_rates)
    experiment = load_experiment("protocol/finetune.json")
    config = experiment_config(experiment)
    schedule = config["training"]["learning_rate_schedule"]
    assert learning_rate_factor(1, schedule) == .1
    assert learning_rate_factor(200, schedule) == 1.
    assert learning_rate_factor(32000, schedule) == .1
    rates = np.array([learning_rates(config, update) for update in range(1, 32001)])
    assert np.all(np.diff(rates[:200, 0]) > 0) and np.all(np.diff(rates[199:, 0]) < 0)
    np.testing.assert_allclose(rates[:, 0] / rates[:, 1], .1, rtol=1e-15)
    with pytest.raises(ValueError, match="outside"):
        learning_rate_factor(32001, schedule)

    model = nn.Module()
    model.backbone, model.head = nn.Linear(1, 1), nn.Linear(1, 1)
    reference = deepcopy(model)
    optimizer, uninterrupted = make_optimizer(model, config), make_optimizer(reference, config)
    identities, probabilities = [["a", 1]], np.array([1.])
    for update in range(1, 4002):
        for module, opt in ((model, optimizer), (reference, uninterrupted)):
            set_learning_rates(opt, config, update)
            opt.zero_grad()
            sum(p.square().sum() for p in module.parameters()).backward()
            opt.step()
        if update in (199, 4000):
            saved = dict(config=config, experiment=experiment, step=update, samples=identities,
                probabilities=torch.from_numpy(probabilities), optimizer=deepcopy(optimizer.state_dict()),
                scheduler_state=schedule_state(optimizer, config, update))
            assert validate_resume_state(saved, config, identities, probabilities, experiment) == update
            optimizer = make_optimizer(model, config)
            optimizer.load_state_dict(saved["optimizer"])
            broken = deepcopy(saved)
            broken["optimizer"]["param_groups"][0]["lr"] = 2e-5
            with pytest.raises(ValueError, match="global update"):
                validate_resume_state(broken, config, identities, probabilities, experiment)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pretrained_schedule_exception_cannot_modify_initial_weights_recipe_or_resume():
    from src.train import experiment_config, load_experiment
    experiment = load_experiment("protocol/finetune.json")
    config = experiment_config(experiment)
    saved = dict(config=load_config("protocol/pretrain.json"), step=0,
                 optimizer=dict(state={}), samples=[["a", 1]])
    before = deepcopy(saved)
    validate_initial_state(saved, config, saved["samples"], experiment["initial_changes"])
    assert saved == before
    with pytest.raises(ValueError, match="definition"):
        validate_initial_state(saved, config, saved["samples"])
    for section, key, value in (("training", "learning_rate", 1e-3), ("loss", "keep_mode", "worst"),
                                 ("model", "condition_modulation", False)):
        candidate = deepcopy(config)
        candidate[section][key] = value
        with pytest.raises(ValueError, match="definition"):
            validate_initial_state(saved, candidate, saved["samples"], experiment["initial_changes"])
    with pytest.raises(ValueError, match="untrained step-zero"):
        validate_initial_state(dict(saved, step=1), config, saved["samples"], experiment["initial_changes"])


def test_optimization_records_preserve_clipping_updates_and_pre_forward_evidence(tmp_path):
    from src.train import forward_state, gradient_groups, parameter_change, make_optimizer
    torch.manual_seed(83)
    model = nn.Module()
    model.backbone = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Dropout(.2))
    model.head = nn.Linear(3, 1)
    reference = deepcopy(model)
    x = torch.randn(8, 3)
    config = load_config("protocol/pretrain.json")
    opt, old_opt = make_optimizer(model, config), make_optimizer(reference, config)
    state = forward_state(model)
    before = deepcopy(model.state_dict())
    initial_rng = torch.get_rng_state()
    loss = model.head(model.backbone(x)).square().mean()
    loss.backward()
    norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters()], error_if_nonfinite=True)
    observed = gradient_groups(model)
    assert set(observed) == {"backbone", "new_modules"}
    # Parameters have not changed, but BN buffers have; the captured prefix restores the actual forward input state.
    alert = dict(model=model.state_dict(), **{k: v for k, v in state.items() if k != "buffers"})
    alert["model"].update(state["buffers"])
    torch.save(alert, tmp_path / "alert.pt")
    loaded = torch.load(tmp_path / "alert.pt", weights_only=True)
    for key, value in before.items():
        torch.testing.assert_close(loaded["model"][key], value, rtol=0, atol=0)
    before_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    torch.nn.utils.clip_grads_with_norm_(model.parameters(), 1., norm)
    opt.step()
    change = parameter_change(model, before_parameters)
    assert all(0 < v["relative_update"] < .1 for v in change.values())
    measured_rng = torch.get_rng_state()
    torch.testing.assert_close(loaded["torch_rng"], initial_rng, rtol=0, atol=0)
    torch.set_rng_state(loaded["torch_rng"])
    expected = reference.head(reference.backbone(x)).square().mean()
    expected.backward()
    old_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
    old_opt.step()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(norm, old_norm, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), measured_rng, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)


def test_detection_class_records_keep_missing_anomalies_distinct_from_zero_loss():
    from src.train import detection_loss
    scores = torch.tensor([-2., 1.], requires_grad=True)
    target = torch.zeros(2, dtype=torch.long)
    loss, details = detection_loss(scores, target, details=True)
    assert details["anomaly_loss"] is None
    assert float(loss.detach()) == .5 * details["normal_loss"]
    torch.testing.assert_close(loss, .5 * torch.nn.functional.softplus(scores).mean(), rtol=0, atol=0)


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


def test_insertion_protection_keeps_baseline_gradient_and_gates_only_after():
    before = torch.tensor([-2., 1., 0., -40.], dtype=torch.float64, requires_grad=True)
    after = torch.tensor([1., -2., 0., -30.], dtype=torch.float64, requires_grad=True)
    loss = keep_loss(before, after, "increase")
    expected = .5 * torch.maximum(torch.nn.functional.softplus(before),
                                  torch.nn.functional.softplus(after)).mean()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    original_grad, inserted_grad = torch.autograd.grad(loss, (before, after))
    torch.testing.assert_close(original_grad, .5 * before.sigmoid() / len(before))
    torch.testing.assert_close(inserted_grad, .5 * after.sigmoid() * (after > before) / len(before))
    mean_grad = torch.autograd.grad(keep_loss(before, after, "mean"), (before, after))
    torch.testing.assert_close(original_grad, mean_grad[0])
    torch.testing.assert_close(inserted_grad[[0, 3]], mean_grad[1][[0, 3]])
    assert inserted_grad[1] == inserted_grad[2] == 0
    assert 0 < inserted_grad[3] < 1e-12
    # Detaching the whole baseline would lose its supervision; a maximum cancels it when after is worse.
    maximum_grad = torch.autograd.grad(expected, before)[0]
    assert maximum_grad[0] == 0 and original_grad[0] > 0
    empty = torch.empty(0, requires_grad=True)
    keep_loss(empty, empty, "increase").backward()
    assert empty.grad is not None and empty.grad.numel() == 0


def test_protection_branch_preserves_resume_rules_and_draws():
    from src.train import (experiment_config, load_experiment, make_optimizer, schedule_state,
                           set_learning_rates, validate_branch_state)
    source_experiment = load_experiment("protocol/finetune.json")
    config = experiment_config(source_experiment)
    model = nn.Module()
    model.backbone, model.head = nn.Linear(1, 1), nn.Linear(1, 1)
    optimizer = make_optimizer(model, config)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    # This small fixture checks state validation, not real trained-model compatibility.
    for state in optimizer.state.values():
        state["step"].fill_(1024)
    set_learning_rates(optimizer, config, 1024)
    identities, probabilities = [["a", 1], ["b", 2]], np.array([.3, .7])
    saved = dict(step=1024, config=config, experiment=source_experiment, samples=identities,
        probabilities=torch.from_numpy(probabilities), optimizer=optimizer.state_dict(),
        scheduler_state=schedule_state(optimizer, config, 1024), model=model.state_dict(),
        preprocessing={}, torch_rng=torch.get_rng_state(), cuda_rng=[],
        python_rng=random.getstate(), numpy_rng=np.random.get_state())
    expected_requests = list(Requests(probabilities, config, 1152, start=1024))
    assert len(expected_requests) == 256 and [r[1] for r in expected_requests] == list(range(2048, 2304))
    for arm in ("mean", "increase"):
        experiment = load_experiment("protocol/keep.json", arm)
        candidate = deepcopy(config)
        candidate["scope"] = experiment["scope"]
        candidate["loss"].update(experiment["loss_overrides"])
        assert validate_branch_state(saved, candidate, identities, probabilities, experiment) == 1024
        assert list(Requests(probabilities, candidate, 1152, start=1024)) == expected_requests
        with pytest.raises(ValueError, match="resume configuration"):
            validate_resume_state(saved, candidate, identities, probabilities, experiment)
        for section, key, value in (("training", "learning_rate", 1e-3), ("loss", "keep_weight", .5),
                                    ("model", "condition_modulation", False)):
            changed = deepcopy(candidate)
            changed[section][key] = value
            with pytest.raises(ValueError, match="only loss.keep_mode"):
                validate_branch_state(saved, changed, identities, probabilities, experiment)
        changed = deepcopy(experiment)
        changed["evaluation"]["full_val19_steps"] = [1152]
        with pytest.raises(ValueError, match="fixed206/152"):
            validate_branch_state(saved, candidate, identities, probabilities, changed)
        incomplete = dict(saved)
        incomplete.pop("cuda_rng")
        with pytest.raises(ValueError, match="complete trained state"):
            validate_branch_state(incomplete, candidate, identities, probabilities, experiment)
    assert saved["config"]["loss"]["keep_mode"] == "mean"
    with pytest.raises(ValueError, match="choose an explicitly declared"):
        load_experiment("protocol/keep.json")


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


def test_protection_control_keeps_four_forwards_and_normalization_inputs():
    torch.manual_seed(73)
    rows, model = batch_fixture(), SmallModel().train()
    config = load_config()
    config["loss"].update(keep_mode="mean", tail_weight=0.)
    candidate = deepcopy(model)
    reference_loss, reference_stats, reference = batch_loss(model, rows, config, 1024, details=True)
    random_state = torch.get_rng_state()
    config["loss"]["keep_mode"] = "increase"
    loss, stats, actual = batch_loss(candidate, rows, config, 1024, details=True)
    assert model.calls == candidate.calls == 4
    for observed, expected in zip(actual["scores"], reference["scores"], strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual["components"]["detection"], reference["components"]["detection"], rtol=0, atol=0)
    reference_loss.backward()
    loss.backward()
    for observed, expected in zip(candidate.buffers(), model.buffers(), strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), random_state, rtol=0, atol=0)
    assert candidate.context[1].num_batches_tracked == 4
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())


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
