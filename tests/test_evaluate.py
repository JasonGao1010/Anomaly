import numpy as np
import pytest

from src.data import FramePrediction
from src.evaluate import (
    official_metrics,
    exact_metrics,
    packed_scores,
    pooled_files,
    diagnostic_bin,
    APAttribution,
    evaluate_frames,
)
from src.scene import PointLabels, make_source_frame
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


def test_signed_logit_pooling_preserves_unsaturated_order_and_zero_threshold(tmp_path):
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.evaluate import bits_score

    scores = np.array([-80, -4, -0.0, 0.0, 4, 80, 81, 81], np.float32)
    target = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    packed = packed_scores(scores, target, score_kind="logit")
    np.testing.assert_array_equal(
        bits_score((packed >> 1).astype(np.uint32), "logit"), scores
    )
    for chunk in (1, 3, 100):
        result = exact_metrics(np.sort(packed), chunk_size=chunk, score_kind="logit")
        assert result["AP"] == pytest.approx(
            100 * average_precision_score(target, scores)
        )
        assert result["AUROC"] == pytest.approx(100 * roc_auc_score(target, scores))


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
        high = result["official_high_recall"]
        accepted = scores >= high["threshold"]
        assert high["tp"] == int(np.sum(accepted & (target == 1)))
        assert high["fp"] == int(np.sum(accepted & (target == 0)))
        assert high["recall"] > 95
        assert high["FPR"] == pytest.approx(expected["FPR95"], abs=1e-10)
        paths = [tmp_path / f"{index}_{part}.bin" for part in range(2)]
        for path, block in zip(paths, np.array_split(keys, 2), strict=True):
            block.tofile(path)
        pooled = pooled_files(paths)
        for name, value in expected.items():
            assert pooled[name] == pytest.approx(value, abs=1e-10, rel=0)
        assert pooled["normal_count"] == int((target == 0).sum())


def test_global_ap_deficit_uses_complete_ties():
    from src.evaluate import APAttribution
    from src.evaluate import score_bits

    scores = np.array([2, 2, 1, -1, -1, -3], dtype=np.float32)
    labels = np.array([1, 0, 1, 1, 0, 0])
    observer = APAttribution()
    result = exact_metrics(
        np.sort(packed_scores(scores, labels, score_kind="logit")),
        chunk_size=2,
        score_kind="logit",
        observe=observer,
    )
    precision = dict(
        zip(
            np.concatenate(observer.bits),
            np.concatenate(observer.precision),
            strict=True,
        )
    )
    deficit = np.array(
        [1 - precision[b] for b in score_bits(scores[labels == 1], "logit")]
    )
    assert deficit.sum() / 3 == pytest.approx(1 - result["AP"] / 100, abs=1e-12)
    assert precision[score_bits([2], "logit")[0]] == 0.5


def test_single_scan_pooling_preserves_official_scope_and_point_identity(tmp_path):
    pairs, official = [], PointOODMetricsCalculator()
    for frame_id, anomaly_count in enumerate((5, 7, 4)):
        ranges = np.array(
            [2.5, 50, 10, 11, 12, 13, 14, 15, 2.49, 50.01, 0, 16], np.float32
        )
        xyzi = np.zeros((len(ranges), 4), np.float32)
        xyzi[:, 0] = ranges
        semantic = np.full(len(ranges), 40, np.uint16)
        semantic[:anomaly_count] = 2
        semantic[8:11] = 2
        semantic[-1] = 0
        labels = PointLabels(
            semantic.astype(np.uint32), semantic, np.zeros(len(ranges), np.uint16)
        )
        source = make_source_frame(
            frame_id,
            xyzi,
            np.eye(4),
            labels,
            partition="fixture",
            sequence_id=1,
        )
        slots = source.real_slots[::-1]
        prediction = FramePrediction(
            "fixture",
            1,
            frame_id,
            slots,
            (np.sin(slots + frame_id) * 10).astype(np.float32),
        )
        pairs.append((source, prediction))
        official.update(source.xyzi[:, :3], prediction.restore(source), semantic)
    observer = APAttribution()
    result, rows = evaluate_frames(iter(pairs), directory=tmp_path, observe=observer)
    assert result["frames"] == 3 and result["eligible_frames"] == 2
    assert [r["anomaly_points"] for r in rows] == [5, 7, 4]
    for key, value in official_metrics(official).items():
        assert result[key] == pytest.approx(value, abs=1e-10)
    all_scores = np.concatenate(official.all_scores)
    all_labels = np.concatenate(official.all_labels)
    anomaly_scores = all_scores[all_labels == 1]
    precision, required = observer.values(anomaly_scores)
    assert np.mean(1 - precision) == pytest.approx(1 - result["AP"] / 100)
    for score, p, q in zip(anomaly_scores, precision, required, strict=True):
        accepted = all_scores >= score
        assert p == pytest.approx(all_labels[accepted].mean())
        assert q == pytest.approx(np.mean(all_scores[all_labels == 0] >= score))


def test_prediction_duplicate_scan_is_rejected(tmp_path):
    xyzi = np.array([[10, 0, 0, 0.2]], np.float32)
    semantic = np.array([40], np.uint16)
    source = make_source_frame(
        0,
        xyzi,
        np.eye(4),
        PointLabels(semantic.astype(np.uint32), semantic, np.zeros(1, np.uint16)),
        partition="fixture",
        sequence_id=1,
    )
    prediction = FramePrediction(
        "fixture", 1, 0, source.real_slots, np.array([0], np.float32)
    )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_frames(
            [(source, prediction), (source, prediction)], directory=tmp_path
        )
