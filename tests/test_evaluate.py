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
