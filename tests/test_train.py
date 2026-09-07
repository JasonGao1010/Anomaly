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


def test_full_pool_visits_include_all_early_and_terminal_windows():
    pool = load_protocol().training_pool
    samples = training_samples(pool, full=True)
    assert len(samples) == 3080
    assert [s["dataset_index"] for s in samples] == list(range(3080))
    assert len({(s["sequence_id"], s["segment_index"]) for s in samples}) == 128
    for sequence in range(8):
        for segment in range(16):
            selected = [
                s
                for s in samples
                if s["synthetic_sequence_index"] == sequence
                and s["segment_index"] == segment
            ]
            assert [s["current_frame"] - 4 for s in selected] == list(
                pool.window_starts(segment)
            )
            assert all(
                s["frame_ids"]
                == list(range(s["current_frame"] - 4, s["current_frame"] + 1))
                for s in selected
            )
    schedule = shuffled_schedule(23, 3080, 30800)
    for start in range(0, 30800, 3080):
        assert sorted(schedule[start : start + 3080]) == list(range(3080))


def test_full_schedule_successful_warmup_and_two_complete_low_rate_epochs():
    from src.train import advance_full_schedule, full_learning_rate

    state = {
        "successful_updates": 0,
        "lr_level": 0,
        "reference_ap": None,
        "bad_epochs": 0,
        "low_lr_epochs": 0,
        "low_lr_bad_epochs": 0,
        "completed_epochs": 0,
    }
    assert full_learning_rate(state) == pytest.approx(3e-5)
    state["planned_attempts"] = 40  # Skipped attempts never advance warmup.
    assert full_learning_rate(state) == pytest.approx(3e-5)
    state["successful_updates"] = 199
    assert full_learning_rate(state) == pytest.approx(3e-4)
    state["successful_updates"] = 200
    assert full_learning_rate(state) == pytest.approx(3e-4)
    for epoch, ap in enumerate((94, 94.04, 94.1, 94.02, 94.07, 94.09, 94.05), 1):
        state["completed_epochs"] = epoch
        improved, stopped = advance_full_schedule(state, ap)
        assert improved == (epoch == 1)
        assert stopped == (epoch == 7)
        assert state["reference_ap"] == 94
        assert state["lr_level"] == (0 if epoch < 3 else 1 if epoch < 5 else 2)
    state["completed_epochs"] = 8
    assert advance_full_schedule(state, 94.12) == (True, False)
    assert state["low_lr_bad_epochs"] == 0


def test_nre_learning_rate_uses_visits_after_successful_update_warmup():
    from src.train import nre_learning_rate

    state = dict(successful_updates=0, planned_attempts=0)
    assert nre_learning_rate(state) == pytest.approx(3e-5)
    # An overflow consumes a visit but does not advance successful-update warmup.
    state["planned_attempts"] = 1
    assert nre_learning_rate(state) == pytest.approx(3e-5)
    state.update(successful_updates=199, planned_attempts=200)
    assert nre_learning_rate(state) == pytest.approx(3e-4)
    state["successful_updates"] = 200
    for completed, expected in (
        (14239, 3e-4),
        (14240, 1e-4),
        (21359, 1e-4),
        (21360, 3e-5),
        (28479, 3e-5),
    ):
        state["planned_attempts"] = completed
        assert nre_learning_rate(state) == expected
    state["planned_attempts"] = 28480
    with pytest.raises(ValueError, match="visit budget"):
        nre_learning_rate(state)


def test_explicit_recovery_keeps_training_state_and_rejects_damaged_buffers(tmp_path):
    from src.evaluate import file_hash
    from src.train import recovery_payload

    plan_path, checkpoint = tmp_path / "plan.json", tmp_path / "visit_14240.pt"
    plan = dict(
        config={"seed": 23}, schedule=[0, 1], sampler_random_state={"fixed": 23}
    )
    plan_path.write_text(json.dumps(plan))
    payload = dict(
        **plan,
        plan_sha256=file_hash(plan_path),
        next_schedule_index=14240,
        state=dict(
            status="running",
            planned_attempts=14240,
            successful_updates=14236,
            next_position=0,
            completed_monitors=2,
        ),
        model={"running_mean": torch.ones(2)},
        optimizer={"state": {0: {"exp_avg": torch.tensor([0.125])}}},
        scaler={"scale": 128.0},
        random_state=random_state(),
    )
    torch.save(payload, checkpoint)
    restored = recovery_payload(checkpoint, plan_path)
    assert restored["state"] == payload["state"]
    assert restored["scaler"] == payload["scaler"]
    assert restored["schedule"] == payload["schedule"]
    torch.testing.assert_close(
        restored["optimizer"]["state"][0]["exp_avg"],
        payload["optimizer"]["state"][0]["exp_avg"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        restored["random_state"]["torch"],
        payload["random_state"]["torch"],
        rtol=0,
        atol=0,
    )
    payload["model"]["running_mean"][0] = float("inf")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="nonfinite"):
        recovery_payload(checkpoint, plan_path)
    payload["model"]["running_mean"].zero_()
    payload["state"]["status"] = "numerical_error"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="healthy recovery source"):
        recovery_payload(checkpoint, plan_path)


def test_monitor_corrections_replace_working_metrics_without_changing_checkpoints(
    tmp_path,
):
    from src.evaluate import file_hash
    from src.train import apply_monitor_corrections, nre_monitor_candidate

    plan = {
        "monitor_samples": [{"view": "real"}, {"view": "synthetic"}, {"view": "normal"}]
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    state = dict(
        planned_attempts=14240,
        successful_updates=14236,
        monitor_candidates=[],
        monitor_results=[],
    )
    entries = []
    for index, visit in enumerate((7120, 14240), 1):
        checkpoint = tmp_path / f"visit_{visit:05d}.pt"
        checkpoint.write_bytes(b"immutable original model and training state")
        directory = tmp_path / f"monitor_{index:02d}_corrected"
        results = {}
        for domain in ("real", "synthetic"):
            path = directory / domain
            path.mkdir(parents=True)
            samples = [
                s
                for s in plan["monitor_samples"]
                if (s["view"] == "real") == (domain == "real")
            ]
            manifest = dict(
                identity=dict(
                    plan_sha256=file_hash(plan_path),
                    visit=visit,
                    checkpoint_sha256=file_hash(checkpoint),
                    rotary_cache_precision="autocast_separated",
                ),
                samples=samples,
            )
            (path / "samples.json").write_text(json.dumps(manifest))
            results[domain] = dict(
                status="completed",
                completed_windows=len(samples),
                optimizer_updates=0,
                model_parameters_and_buffers_unchanged=True,
                all_frames=dict(AP=50 + index, AUROC=99, FPR95=5),
                normal_without_anomaly_returns=dict(fraction_ge_0=0.001),
            )
            (path / "summary.json").write_text(json.dumps(results[domain]))
        candidate = nre_monitor_candidate(checkpoint, directory, results)
        state["monitor_candidates"].append(
            {
                **candidate,
                "AP": 1.0,
                "evaluation": str(tmp_path / f"monitor_{index:02d}"),
            }
        )
        state["monitor_results"].append({"AP": 1.0, "successful_updates": visit})
        entries.append(
            dict(
                visit=visit,
                checkpoint_sha256=file_hash(checkpoint),
                corrected=candidate,
            )
        )
    record = dict(status="completed", plan_sha256=file_hash(plan_path), entries=entries)
    (tmp_path / "monitor_corrections.json").write_text(json.dumps(record))
    apply_monitor_corrections(state, tmp_path, plan)
    assert [c["AP"] for c in state["monitor_candidates"]] == [51, 52]
    assert [c["AP"] for c in state["monitor_results"]] == [51, 52]
    assert (state["planned_attempts"], state["successful_updates"]) == (14240, 14236)
    assert all(
        file_hash(tmp_path / f"visit_{e['visit']:05d}.pt") == e["checkpoint_sha256"]
        for e in entries
    )
    apply_monitor_corrections(
        state, tmp_path, plan
    )  # Resume uses the same corrected evidence.
    summary_path = tmp_path / "monitor_01_corrected/synthetic/summary.json"
    damaged = json.loads(summary_path.read_text())
    damaged["completed_windows"] -= 1
    summary_path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="complete corrected monitor"):
        apply_monitor_corrections(state, tmp_path, plan)


@pytest.mark.parametrize("position", ["epoch", "visit"])
def test_full_selection_uses_global_ap_band_and_one_scope(position):
    from src.train import choose_candidate

    def candidate(name, ap, fpr, normal, epoch):
        return {
            "name": name,
            "AP": ap,
            "FPR95": fpr,
            "normal_fraction": normal,
            position: epoch,
            "scope": "complete_201",
        }

    a = candidate("A", 94, 0.1, 0.01, 1)
    b = candidate("B", 94.09, 0.2, 0.01, 2)
    c = candidate("C", 94.18, 0.3, 0.01, 3)
    assert choose_candidate([a, b, c]) is b  # A is outside the maximum's AP band.
    c.update(FPR95=0.2, normal_fraction=0.005)
    assert choose_candidate([a, b, c]) is c
    b.update(normal_fraction=0.005)
    assert choose_candidate([a, b, c]) is b
    with pytest.raises(ValueError, match="common evaluation scope"):
        choose_candidate([a, {**b, "scope": "fixed_345"}])


def test_atomic_progress_replaces_only_mutable_file(tmp_path):
    from src.train import write_progress

    path = tmp_path / "summary.json"
    write_progress(path, {"next_position": 500})
    write_progress(path, {"next_position": 1000})
    assert json.loads(path.read_text()) == {"next_position": 1000}
    assert list(tmp_path.iterdir()) == [path]


def test_user_finish_keeps_complete_epoch_and_rejects_partial_training_or_monitoring():
    from src.train import finish_full_training

    state = {
        "phase": "training",
        "status": "running",
        "completed_epochs": 7,
        "next_position": 0,
        "monitor_candidates": list(range(7)),
        "planned_attempts": 21560,
        "successful_updates": 21559,
    }
    for change in ({"next_position": 1}, {"phase": "monitor"}, {"completed_epochs": 8}):
        with pytest.raises(ValueError, match="complete trained and monitored epoch"):
            finish_full_training({**state, **change})
    finish_full_training(state)
    assert state["phase"] == "selection"
    assert state["status"] == "user_requested_epoch_stop"
    assert state["user_stop_after_epoch"] == 7
    assert (state["planned_attempts"], state["successful_updates"]) == (21560, 21559)
    finish_full_training(state)  # Resuming final evaluation cannot restart training.
    assert state["status"] == "user_requested_epoch_stop"
