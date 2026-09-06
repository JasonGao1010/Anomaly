import numpy as np
import pytest
import torch
from torch import nn

from src.evaluate import (
    CURRENT_FRAMES,
    assert_unchanged,
    evaluation_targets,
    normal_statistics,
    official_metrics,
    select_samples,
    synthetic_metrics,
    anomaly_losses,
    exact_metrics,
    full_samples,
    packed_scores,
    pooled_files,
    diagnostic_bin,
)
from src.protocol import load_protocol
from src.train import fixed_check
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


def test_earlier_middle_selection_and_paired_frames():
    pool = load_protocol().validation_pool
    samples = select_samples(pool)
    assert len(samples) == 23
    assert tuple(sample["current_frame"] for sample in samples) == CURRENT_FRAMES
    all_starts = [
        start for segment in range(23) for start in pool.window_starts(segment)
    ]
    for sample in samples:
        starts = pool.window_starts(sample["segment_index"])
        assert sample["window_start"] == starts[(len(starts) - 1) // 2]
        assert sample["window_start"] == all_starts[sample["dataset_index"]]
        assert sample["frame_ids"] == list(
            range(sample["current_frame"] - 4, sample["current_frame"] + 1)
        )
        assert sample["synthetic_sequence_id"] == "synthetic/validation/000"
        assert sample["normal_sequence_id"] == "train/201"
    assert samples[-1]["current_frame"] == 650
    with pytest.raises(ValueError, match="201 validation"):
        select_samples(load_protocol().training_pool)


def test_weighted_ap_and_realizable_recall_keep_score_ties(tmp_path):
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    # The tie at .8 exceeds the 1% FPR budget and cannot be partially selected.
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.7] + [0.1] * 98, dtype=np.float32)
    target = np.array([1, 1, 0, 0, 1] + [0] * 98)
    positive, negative = int(target.sum()), int((target == 0).sum())
    pi = 87398 / 193470656
    weights = np.where(target, pi / positive, (1 - pi) / negative)
    reference = average_precision_score(target, scores, sample_weight=weights) * 100
    fpr, tpr, thresholds = roc_curve(target, scores, drop_intermediate=False)
    best = np.flatnonzero(tpr == max(tpr[fpr <= 0.01]))[0]
    records = packed_scores(scores, target)
    for chunk in (1, 2, 7, 1 << 20):
        result = exact_metrics(np.sort(records), chunk_size=chunk, prevalence=pi)
        assert result["standardized_AP"] == pytest.approx(reference, abs=1e-11)
        assert result["AUROC"] == pytest.approx(roc_auc_score(target, scores) * 100)
        point = result["recall_at_fpr_limit"]
        assert point["recall"] == pytest.approx(tpr[best] * 100)
        assert point["FPR"] == fpr[best] * 100
        assert point["threshold"] == thresholds[best]
        assert point["tp"] == 1 and point["fp"] == 0
    path = tmp_path / "records.bin"
    np.r_[records[:5], np.uint64(0), records[5:]].tofile(path)
    result = pooled_files(
        [path, path], ranges=[(0, 5), (6, len(records) - 5)], prevalence=pi
    )
    assert result["standardized_AP"] == pytest.approx(reference, abs=1e-11)
    # At the pool's own prevalence, standardized AP equals ordinary AP.
    result = exact_metrics(np.sort(records), prevalence=positive / len(target))
    assert result["standardized_AP"] == pytest.approx(result["AP"], abs=1e-11)
    blocked = exact_metrics(np.sort(packed_scores([0.9, 0.9], [0, 1])), prevalence=pi)
    assert blocked["recall_at_fpr_limit"] == dict(
        recall=0.0, FPR=0.0, threshold=None, tp=0, fp=0
    )


def test_diagnostic_strata_use_fixed_half_open_boundaries():
    assert diagnostic_bin(4, 10) is None
    assert diagnostic_bin(5, 2.5) == "0_0"
    assert diagnostic_bin(19, 9.999) == "0_0"
    assert diagnostic_bin(20, 10) == "1_1"
    assert diagnostic_bin(100, 20) == "2_2"
    assert diagnostic_bin(500, 35) == "3_3"
    assert diagnostic_bin(500, 50) == "3_3"
    with pytest.raises(ValueError, match="official-range"):
        diagnostic_bin(5, 50.001)


def test_intensity_distribution_matches_exact_weighted_order_statistics():
    from src.intensity import distribution, QUANTILES

    values = np.array([0, 0, 1 / 3500, 1 / 3500, 0.1234567, 2], np.float32)
    result = distribution(values, (0.0, 2.0))
    np.testing.assert_allclose(
        list(result["quantiles"].values()),
        np.quantile(values.astype(np.float64), QUANTILES),
        rtol=0,
        atol=1e-15,
    )
    assert result["unique_count"] == 4
    assert result["zero_fraction"] == 2 / 6
    assert result["repeated_point_fraction"] == 4 / 6
    assert result["grid_1_over_3500_fraction"] == 5 / 6


def test_normal_filter_ignores_frame_eligibility_and_keeps_fixed_threshold():
    points = np.zeros((6, 3), dtype=np.float32)
    points[:, 0] = (2.5, 50, 2.49, 50.01, 10, 10)
    target = evaluation_targets(points, np.array((40, 48, 40, 40, 0, 2)))
    np.testing.assert_array_equal(target, (0, 0, -1, -1, -1, 1))
    scores = np.array((0.49, 0.5, 1, 1, 1, 1), dtype=np.float32)
    result = normal_statistics(scores[target == 0])
    assert result["point_count"] == 2 and result["count_ge_0_5"] == 1
    assert result["fraction_ge_0_5"] == 0.5
    assert result["median"] == float(np.median(scores[:2]))
    assert normal_statistics(np.empty(0))["fraction_ge_0_5"] is None


def test_official_pooling_is_not_mean_window_ap_and_skips_ineligible_frames():
    pooled = PointOODMetricsCalculator()
    inputs = [
        (np.array([0.9] * 5 + [0.8] * 5), np.array([2] * 5 + [40] * 5)),
        (np.array([0.2] * 5 + [0.1] * 45), np.array([2] * 5 + [40] * 45)),
    ]
    for scores, semantic in inputs:
        points = np.tile([10.0, 0, 0], (len(scores), 1))
        result = synthetic_metrics(points, scores, semantic, pooled)
        assert result["AP"] == 100 and result["eligible"]
    result = synthetic_metrics(
        np.tile([10.0, 0, 0], (10, 1)),
        np.ones(10),
        np.array([2] * 4 + [40] * 6),
        pooled,
    )
    assert result["AP"] is None and result["anomaly_count"] == 4
    assert result["ineligible_reason"] == "fewer_than_5_official_anomaly_points"
    assert len(pooled.all_labels) == 2
    reference = PointOODMetricsCalculator()
    scores, semantic = (np.concatenate([item[k] for item in inputs]) for k in (0, 1))
    reference.update(np.tile([10.0, 0, 0], (len(scores), 1)), scores, semantic)
    assert official_metrics(pooled) == official_metrics(reference)
    assert official_metrics(pooled)["AP"] < 100

    # At exactly 95% recall the upstream routine advances to the next ROC point.
    strict = PointOODMetricsCalculator()
    scores = np.array([0.9] * 19 + [0.1, 0.2, 0.05])
    strict.update(np.tile([10.0, 0, 0], (22, 1)), scores, np.array([2] * 20 + [40] * 2))
    assert official_metrics(strict)["FPR95"] == 50


def test_zero_update_checks_parameters_and_batchnorm_buffers():
    model = (
        nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 1)).eval().requires_grad_(False)
    )
    reference = {name: value.clone() for name, value in model.state_dict().items()}
    with fixed_check(model, 23):
        logits = model(torch.randn(8, 3))
        assert not logits.requires_grad
    assert_unchanged(model, reference)
    with torch.no_grad():
        model[1].weight.add_(0.1)
    with pytest.raises(RuntimeError, match="changed model"):
        assert_unchanged(model, reference)


def test_full_sample_coverage_preserves_old_view_and_boundaries():
    pool = load_protocol().validation_pool
    samples = full_samples(pool)
    assert len(samples) == 3038
    assert sum(row["scope"] == "selected_23" for row in samples) == 23
    assert sum(row["scope"] == "sequence_0_remaining" for row in samples) == 567
    assert sum(row["scope"] == "sequences_1_3" for row in samples) == 1770
    synthetic = samples[:2360]
    assert [row["dataset_index"] for row in synthetic] == list(range(2360))
    for row in synthetic:
        span = pool.segments[row["segment_index"]]
        assert span.start <= row["frame_ids"][0] < row["current_frame"] < span.stop
    for old, row in zip(
        select_samples(pool),
        [r for r in samples if r["scope"] == "selected_23"],
        strict=True,
    ):
        assert row["current_frame"] == old["current_frame"]
        assert row["check_seed"] == old["check_seed"]
    assert [row["current_frame"] for row in samples[2360:]] == list(range(4, 682))
    assert len({(row["sequence_id"], row["current_frame"]) for row in samples}) == 3038


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 1000000])
def test_exact_disk_metrics_match_official_ties_and_roc_pruning(tmp_path, chunk_size):
    generator = np.random.default_rng(91)
    cases = [
        (generator.random(2000, dtype=np.float32), generator.integers(0, 2, 2000)),
        (
            generator.integers(0, 8, 2000).astype(np.float32) / 8,
            generator.integers(0, 2, 2000),
        ),
        (np.full(100, 0.5, dtype=np.float32), np.tile([0, 1], 50)),
        (
            np.array([0.9] * 19 + [0.1, 0.2, 0.05], np.float32),
            np.array([1] * 20 + [0, 0]),
        ),
        # Collinear ROC nodes crossing 95% must be dropped before strict FPR95.
        (np.arange(100, 0, -1, dtype=np.float32).repeat(2) / 101, np.tile([0, 1], 100)),
    ]
    for index, (scores, target) in enumerate(cases):
        calculator = PointOODMetricsCalculator()
        calculator.all_scores = [scores]
        calculator.all_labels = [target]
        expected = official_metrics(calculator)
        keys = packed_scores(scores, target)
        result = exact_metrics(np.sort(keys), chunk_size=chunk_size)
        for name, value in expected.items():
            assert result[name] == pytest.approx(value, abs=1e-10, rel=0)
        paths = [tmp_path / f"{index}_{part}.bin" for part in range(2)]
        for path, block in zip(paths, np.array_split(keys, 2), strict=True):
            block.tofile(path)
        pooled = pooled_files(paths)
        for name, value in expected.items():
            assert pooled[name] == pytest.approx(value, abs=1e-10, rel=0)
        assert pooled["normal_count"] == int((target == 0).sum())


@pytest.mark.parametrize("count", [103, 104])
def test_disk_normal_quantiles_are_exact_and_loss_scopes_count_points(tmp_path, count):
    scores = np.random.default_rng(6).random(count, dtype=np.float32)
    path = tmp_path / "normal.bin"
    packed_scores(scores, np.zeros(len(scores))).tofile(path)
    assert pooled_files([path], normal=True) == normal_statistics(scores)
    logits = torch.tensor([-100.0, 100.0, -3.0, 4.0, 6.0, -9.0])
    target = torch.tensor([1, 1, 1, 0, -1, 1])
    current = np.array([False, False, True, True, True, True])
    result = anomaly_losses(logits, target, current, np.array([1, 0, -1, -1]))
    assert [result[key]["point_count"] for key in result] == [4, 2, 2, 1]
    assert (
        result["all"]["loss_sum"]
        == result["history"]["loss_sum"] + result["current"]["loss_sum"]
    )
    assert (
        result["all"]["loss_sum"] > 100
    )  # Do not recover this from saturated sigmoid.


def test_fixed_monitor_covers_all_worlds_and_preserves_full_inference_seeds():
    from src.evaluate import monitor_samples

    pool = load_protocol().validation_pool
    samples = monitor_samples(pool)
    assert len(samples) == 345
    full = full_samples(pool)
    assert all(sample in full for sample in samples)
    synthetic = [s for s in samples if s["view"] == "synthetic"]
    normal = [s for s in samples if s["view"] == "normal"]
    assert len(synthetic) == 276 and len(normal) == 69
    assert len({s["current_frame"] for s in normal}) == 69
    for sequence in range(4):
        for segment in range(23):
            rows = [
                s
                for s in synthetic
                if s["sequence_index"] == sequence and s["segment_index"] == segment
            ]
            expected = (
                [segment * 28 + i for i in (4, 15, 27)]
                if segment < 22
                else [620, 650, 681]
            )
            assert [s["current_frame"] for s in rows] == expected
            assert [s["check_seed"] for s in rows] == [23 + 23 * sequence + segment] * 3


def test_monitor_resume_removes_only_uncommitted_prediction_and_metric_tail(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    from types import SimpleNamespace

    import src.evaluate as evaluation

    model = nn.Linear(2, 1)
    dataset = SimpleNamespace(
        gradient_updates_allowed=False, manifest={"segments": []}, source_sequence=None
    )
    sample = {"view": "normal", "current_frame": 4}
    identity = {"model": evaluation.model_digest(model)}
    monkeypatch.setattr(evaluation, "WindowPartition", lambda *args: None)
    (tmp_path / "samples.json").write_text(
        json.dumps({"identity": identity, "samples": [sample], "worlds": []})
    )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    committed = predictions / "frame_000004.npz"
    committed.write_bytes(b"transaction fixture, not scientific prediction evidence")
    dangling = predictions / "frame_000005.npz"
    dangling.write_bytes(b"uncommitted writer output")
    current = tmp_path / "current"
    current.mkdir()
    keys = packed_scores(np.array([0.2, 0.7], np.float32), np.zeros(2))
    records = current / "normal.bin"
    records.write_bytes(keys.tobytes() + b"partial writer tail")
    row = {
        **sample,
        "prediction": {
            "file": "predictions/frame_000004.npz",
            "file_sha256": evaluation.file_hash(committed),
        },
        "evaluation_records": {
            "file": "current/normal.bin",
            "offset": 0,
            "count": 2,
            "sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
        },
    }
    (tmp_path / "results.jsonl").write_bytes(
        (json.dumps(row) + "\n").encode() + b'{"incomplete":'
    )
    summary = {"status": "completed", "completed_windows": 1}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert (
        evaluation.evaluate_samples(
            model,
            dataset,
            [sample],
            tmp_path,
            identity=identity,
            check_resources=lambda: None,
        )
        == summary
    )
    assert committed.exists() and not dangling.exists()
    assert records.read_bytes() == keys.tobytes()
    assert (tmp_path / "results.jsonl").read_text() == json.dumps(row) + "\n"


@pytest.mark.parametrize("count", [0, 1, 2, 103, 104, 1048579])
def test_float32_normal_records_keep_exact_quantiles_without_sort_copy(tmp_path, count):
    values = np.random.default_rng(66).random(count, dtype=np.float32)
    values[::3] = 0.5
    paths = [tmp_path / f"part_{i}.bin" for i in range(2)]
    for path, block in zip(paths, np.array_split(values, 2)):
        block.tofile(path)
    assert pooled_files(paths, normal=True, normal_float32=True) == normal_statistics(
        values
    )


def test_real_pooling_includes_startup_and_keeps_normal_only_frames(tmp_path):
    from src.evaluate import save_window, summarize_real
    from src.protocol import SequenceSpec, FrameSpan
    from src.scene import PointLabels, make_source_frame, assemble_window

    spec = SequenceSpec("val", 125, "fixture", True, FrameSpan(0, 5))
    sources, rows = [], []
    reference = PointOODMetricsCalculator()
    full_reference = PointOODMetricsCalculator()
    for current, anomaly_count in enumerate((6, 0, 4, 0, 5)):
        points = np.tile(np.array([10, 0, 0, 0.2], np.float32), (12, 1))
        points[-1, :3] = 0  # A labelled file slot still has no observed return.
        semantic = np.array(
            [2] * anomaly_count + [40] * (12 - anomaly_count), np.uint16
        )
        labels = PointLabels(
            semantic.astype(np.uint32), semantic, np.zeros(12, np.uint16), None
        )
        sources.append(
            make_source_frame(
                current, points, np.eye(4), labels, partition="val", sequence_id=125
            )
        )
        window = assemble_window(
            spec, 0, tuple(range(current + 1)), sources, startup=current < 4
        )
        scores = np.linspace(0, 1, window.points.count, dtype=np.float32)
        row = save_window(
            tmp_path,
            dict(
                view="real",
                sequence_index=125,
                current_frame=current,
                scope="startup" if current < 4 else "full",
            ),
            window,
            scores,
            {},
            None,
            {},
        )
        rows.append(row)
        raw = window.current_frame.source.restore_real(scores[window.current_mask])
        reference.update(points[:, :3], raw, semantic)
        if current == 4:
            full_reference.update(points[:, :3], raw, semantic)
    result = summarize_real(rows, tmp_path, check_resources=lambda: None)
    for name, value in official_metrics(reference).items():
        assert result["all_frames"][name] == pytest.approx(value, abs=1e-10)
    for name, value in official_metrics(full_reference).items():
        assert result["full_history"][name] == pytest.approx(value, abs=1e-10)
    assert result["all_frames"]["eligible_frames"] == 2
    assert result["startup"]["eligible_frames"] == 1
    assert result["normal_without_anomaly_returns"]["frame_count"] == 2
    assert result["normal_without_anomaly_returns"]["point_count"] == 22
    assert result["ineligible_frames"]["one_to_four_official_anomaly_points"] == 1
