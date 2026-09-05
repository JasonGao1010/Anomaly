from collections import OrderedDict
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.protocol import load_protocol
from src.train import (
    CURRENT_FRAMES,
    SEGMENTS,
    balanced_loss,
    current_metrics,
    fixed_check,
    optimizer_update,
    parameter_changes,
    random_state,
    restore_random_state,
    select_windows,
    shuffled_schedule,
    training_samples,
)


@pytest.mark.parametrize("labels", [(0, 0, 1, -1, 1), (0, -1, 0), (1, -1, 1)])
def test_balanced_point_loss_and_ignore_gradient(labels):
    logits = torch.linspace(-2, 2, len(labels), requires_grad=True)
    target = torch.tensor(labels)
    loss, parts = balanced_loss(logits, target)
    means = [
        F.softplus(logits[target == 0]).mean(),
        F.softplus(-logits[target == 1]).mean(),
    ]
    expected = torch.stack([mean for mean in means if mean.isfinite()]).mean()
    torch.testing.assert_close(loss, expected)
    assert len(parts) == len(set(labels) - {-1})
    loss.backward()
    assert torch.all(logits.grad[target == -1] == 0)
    assert torch.all(logits.grad[target != -1] != 0)
    with pytest.raises(ValueError, match="no valid"):
        balanced_loss(logits, torch.full_like(target, -1))


def test_evaluation_restores_rng_modes_and_batchnorm():
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout()).train()
    initial = random_state()
    before = {name: value.clone() for name, value in model.named_buffers()}
    results = []
    for _ in range(2):
        with fixed_check(model, 94):
            results.append(
                (random.random(), np.random.rand(), model(torch.randn(8, 3)))
            )
            assert not model.training
            if torch.cuda.is_available():
                torch.rand(5, device="cuda")
        assert model.training
    assert results[0][:2] == results[1][:2]
    torch.testing.assert_close(results[0][2], results[1][2])
    actual = (random.random(), np.random.rand(), torch.rand(5))
    restore_random_state(initial)
    expected = (random.random(), np.random.rand(), torch.rand(5))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])
    if initial["cuda"]:
        for actual_cuda, expected_cuda in zip(
            torch.cuda.get_rng_state_all(), initial["cuda"], strict=True
        ):
            assert torch.equal(actual_cuda, expected_cuda)
    for name, value in model.named_buffers():
        assert torch.equal(value, before[name])


def test_fixed_selection_and_without_replacement_passes():
    pool = load_protocol().training_pool
    all_windows = [
        SimpleNamespace(
            observation_sequence_id=pool.synthetic_sequence_id(0),
            current_frame_id=start + 4,
            frame_ids=tuple(range(start, start + 5)),
        )
        for segment in range(16)
        for start in pool.window_starts(segment)
    ]

    class Dataset:
        gradient_updates_allowed = True

        def __getitem__(self, index):
            return all_windows[index]

    dataset = Dataset()
    dataset.pool = pool
    selected = select_windows(dataset)
    assert tuple(w.current_frame_id for w in selected) == CURRENT_FRAMES
    assert all(
        w.current_frame_id == pool.segments[s].stop - 1
        for w, s in zip(selected, SEGMENTS, strict=True)
    )
    schedule = shuffled_schedule()
    assert len(schedule) == 200
    for start in range(0, 200, 8):
        assert sorted(schedule[start : start + 8]) == list(range(8))
    assert all(schedule.count(index) == 25 for index in range(8))
    dataset.gradient_updates_allowed = False
    with pytest.raises(ValueError, match="only the frozen 206"):
        select_windows(dataset)


def test_coverage_selection_and_equal_budget_schedules():
    pool = load_protocol().training_pool
    narrow, broad = (training_samples(pool, expanded=x) for x in (False, True))
    assert len(narrow) == 8 and len(broad) == 128
    assert tuple(s["current_frame"] for s in narrow) == CURRENT_FRAMES
    assert all(sample in broad for sample in narrow)
    assert len({(s["sequence_id"], s["segment_index"]) for s in broad}) == 128
    for sample in broad:
        sequence, segment = sample["synthetic_sequence_index"], sample["segment_index"]
        assert sample["current_frame"] == pool.segments[segment].stop - 1
        assert (
            sample["dataset_index"]
            == sequence * 385
            + sum(len(pool.window_starts(s)) for s in range(segment + 1))
            - 1
        )
    for count, visits in ((8, 160), (128, 10)):
        schedule = shuffled_schedule(23, count, 1280)
        assert all(schedule.count(i) == visits for i in range(count))
        for start in range(0, 1280, count):
            assert sorted(schedule[start : start + count]) == list(range(count))
        assert schedule == shuffled_schedule(23, count, 1280)
    assert shuffled_schedule(23, 8, 1280)[:200] == shuffled_schedule()
    with pytest.raises(ValueError, match="complete passes"):
        shuffled_schedule(23, 128, 200)
    with pytest.raises(ValueError, match="206"):
        training_samples(load_protocol().validation_pool, True)


def test_coverage_runs_groups_then_predeclared_paired_checkpoints(
    tmp_path, monkeypatch
):
    import src.train as train
    import src.evaluate as evaluation

    learning, transfer = tmp_path / "learn", tmp_path / "transfer"
    learning.mkdir()
    transfer.mkdir()
    initial = learning / "initial.pt"
    initial.write_bytes(b"fixture checkpoint, never loaded by this orchestration test")
    (learning / "final.pt").write_bytes(b"preserved historical checkpoint")
    manifest = transfer / "samples.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": evaluation.select_samples(load_protocol().validation_pool),
                "checkpoints": {"initial": {"sha256": evaluation.file_hash(initial)}},
            }
        )
    )
    for name in ("summary.json", "results.jsonl"):
        (transfer / name).write_text("{}")
    monkeypatch.setattr(
        train,
        "host_disk",
        lambda: {"SizeRemaining": 100 * 2**30, "reserve_bytes": 10 * 2**30},
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []

    def fit(data_root, output, *, group, initial, workers):
        assert initial == learning / "initial.pt"
        assert output.name == group
        plan = json.loads((output.parent / "plan.json").read_text())
        assert plan["primary_step"] == 1280 and plan["check_steps"] == [640, 1280]
        assert len(plan["groups"][group]["schedule"]) == 1280
        calls.append(group)
        return {"status": "completed", "successful_updates": 1280}

    def compare(
        data_root,
        checkpoints,
        output,
        *,
        checkpoint_paths,
        samples_file,
        expected_attempts,
    ):
        assert samples_file == manifest
        assert tuple(checkpoint_paths) == ("A", "B")
        filename = "final.pt" if expected_attempts == 1280 else "step_0640.pt"
        assert all(path.name == filename for path in checkpoint_paths.values())
        calls.append(expected_attempts)
        return {"status": "completed"}

    monkeypatch.setattr(train, "run", fit)
    monkeypatch.setattr(evaluation, "run", compare)
    train.run_coverage(tmp_path, tmp_path / "coverage", initial, manifest, 1)
    assert calls == ["A", "B", 640, 1280]
    result = json.loads((tmp_path / "coverage" / "summary.json").read_text())
    assert result["equal_successful_updates"] and result["prior_evidence_unchanged"]


def test_official_ap_distance_ignore_and_eligibility():
    points = np.zeros((9, 3), dtype=np.float32)
    points[:, 0] = (2.5, 50, 10, 10, 10, 10, 10, 2.49, 50.01)
    scores = np.array((0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 1, 1, 1), np.float32)
    semantic = np.array((2, 2, 2, 2, 2, 40, 0, 2, 2), np.uint16)
    result = current_metrics(points, scores, semantic)
    assert result["anomaly_count"] == 5 and result["normal_count"] == 1
    assert result["AP"] == pytest.approx(
        np.mean(np.arange(1, 6) / np.arange(2, 7)) * 100
    )
    semantic[0] = 0
    assert current_metrics(points, scores, semantic)["AP"] is None


def test_real_optimizer_update_and_overflow_skip_are_distinct():
    model = nn.Sequential(
        OrderedDict(
            (
                ("backbone", nn.Linear(3, 4)),
                ("head", nn.Linear(4, 1)),
            )
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    update = optimizer_update(
        model(torch.ones(6, 3)).square().mean(), model, optimizer, scaler
    )
    assert update["updated"] and update["scale_after"] == 128
    changes = parameter_changes(model, before)
    assert all(changes[key]["changed_elements"] > 0 for key in ("backbone", "head"))
    assert (
        torch.nn.utils.get_total_norm([p.grad for p in model.parameters()]) <= 1.00001
    )

    optimizer.zero_grad(set_to_none=True)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    handle = next(model.parameters()).register_hook(
        lambda grad: torch.full_like(grad, torch.inf)
    )
    update = optimizer_update(
        model(torch.ones(6, 3)).square().mean(), model, optimizer, scaler
    )
    handle.remove()
    assert not update["updated"] and update["scale_after"] == 64
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
