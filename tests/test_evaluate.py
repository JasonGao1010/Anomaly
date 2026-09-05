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
