import numpy as np
import pytest

from src.data import FramePrediction
from src.evaluate import (
    official_metrics,
    exact_metrics,
    metrics_from_groups,
    packed_scores,
    score_bits,
    pooled_files,
    diagnostic_bin,
    APAttribution,
    evaluate_frames,
)
from src.scene import PointLabels, make_source_frame
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


def test_paired_scores_preserve_ties_and_ignore_uniform_logit_shift(tmp_path):
    from src.evaluate import paired_score_summary, paired_transitions
    target = np.array([0, 0, 1, 1], np.int8)
    base = np.array([-3, 1, 1, 2], np.float32)
    modes = ["parent", "v2", "v2_parent_bn"]
    frames = []
    for i in range(2):
        file = f"{i}.npz"
        parts = np.array([[base, np.zeros(4), base],
                          [base - 10, np.zeros(4), base - 10],
                          [base, np.zeros(4), base]], np.float32)
        np.savez_compressed(tmp_path / file, scores=parts, target=target,
            source_slot=np.arange(4, dtype=np.int32), instance=target.astype(np.uint16),
            xyzi=np.ones((4, 4), np.float32), semantic=np.where(target, 2, 40).astype(np.uint16),
            shell_counts=np.full((4, 3), 8, np.uint8))
        frames.append(dict(sequence=125, frame=i, file=file, historical=i == 0))
    manifest = dict(frames=frames, modes=modes, components=["base", "relation", "final"],
                    full_validation_thresholds=[1., -9.])
    result = paired_score_summary(tmp_path, manifest)
    assert result["curves"]["all"]["parent"]["final"]["AP"] == pytest.approx(5 / 6 * 100)
    for row in result["frames"]:
        assert row["AP_change_pp"] == 0
        assert row["decisions"]["normal"]["added"] == row["decisions"]["normal"]["removed"] == 0
    transitions = paired_transitions(target, base, base[::-1], [1, 1])
    assert transitions["normal"]["added"] == transitions["anomaly"]["removed"] == 1


def test_normal_pairs_use_unchanged_slots_and_one_threshold():
    from src.data import FrozenFrame
    from src.evaluate import retained_witness_slots, paired_normal_summary
    xyzi = np.array([[1, 0, 0, .2], [2, 0, 0, .3], [3, 0, 0, .4], [4, 0, 0, .5]], np.float32)
    semantic = np.full(4, 40, np.uint16)
    labels = PointLabels(semantic.astype(np.uint32), semantic, np.zeros(4, np.uint16), np.zeros(4, np.uint8))
    original = make_source_frame(11, xyzi, np.eye(4), labels, partition="train", sequence_id=206)
    frozen = FrozenFrame(original, "a" * 64, np.zeros(4, bool), np.zeros(4, bool))
    measured = dict(contrasts=dict(sparse=dict(normal_source_slots=[0, 1, 1])), changed_normal_source_slots=[1, 2, 3])
    groups = retained_witness_slots(frozen, original, measured)
    np.testing.assert_array_equal(groups["sparse"], [0, 1])
    altered = xyzi.copy()
    altered[1, 3] += .1
    changed = make_source_frame(11, altered, np.eye(4), labels, partition="train", sequence_id=206)
    with pytest.raises(ValueError, match="physical return"):
        retained_witness_slots(FrozenFrame(changed, "a" * 64, np.zeros(4, bool), np.zeros(4, bool)), original, measured)
    rows = [dict(before=[-2., -1., 0., 1.], after=[-1., 1., -2., 2.])]
    result = paired_normal_summary(rows, 0.)
    assert {k: result[k] for k in ("both_low", "low_to_high", "high_to_low", "both_high")} == dict.fromkeys(
        ("both_low", "low_to_high", "high_to_low", "both_high"), 1)
    assert result["before_fp"] == result["after_fp"] == 2
    assert result["delta"]["mean"] == .5
    assert paired_normal_summary(rows, None)["both_low"] == 4
    assert paired_normal_summary([], 0.)["delta"] is None
    with pytest.raises(FloatingPointError):
        paired_normal_summary([dict(before=[0.], after=[float("nan")])], 0.)


def test_fixed_training_only_does_not_access_201(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.data import FrozenFrame
    from src.evaluate import evaluate_fixed
    monkeypatch.setattr("src.evaluate._evaluation_space", lambda required: None)
    xyzi = np.array([[10, 0, 0, .2], [11, 0, 0, .3], [12, 0, 0, .4]], np.float32)
    semantic = np.array([40, 40, 2], np.uint16)
    instance = np.array([0, 0, 60001], np.uint16)
    labels = PointLabels((instance.astype(np.uint32) << 16) | semantic, semantic, instance,
                         np.array([0, 0, 255], np.uint8))
    source = make_source_frame(11, xyzi, np.eye(4), labels, partition="train", sequence_id=206)
    frozen = FrozenFrame(source, "a" * 64, np.array([False, False, True]), np.zeros(3, bool))
    record = dict(identity="a" * 64, frame=11, role="few_returns")
    prepared = dict(datasets={"train": [frozen]}, selection={"train": [record]},
                    indices={"train": [0]}, geometry={})
    scores = np.array([-3., -2., 2.], np.float32)
    model = SimpleNamespace(training=False, predict=lambda scan, transform: SimpleNamespace(restore=lambda scan: scores))
    result = evaluate_fixed(model, None, prepared, directory=tmp_path)
    assert "validation" not in result
    assert result["train"]["full"]["normal_count"] == 2
    assert result["train"]["full"]["anomaly_count"] == 1
    assert result["train"]["full"]["AP"] == 100.
    assert result["train"]["official"]["frames"] == 0
    import torch
    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(1)
            self.seen = []

        def predict(self, source, transform=None, *, prepared=None):
            self.seen.append(prepared)
            values = self.bn(prepared).detach().numpy()[:, 0]
            return SimpleNamespace(restore=lambda source: values)

    scans = []

    def transform(source):
        scans.append(source)
        return torch.from_numpy(scores.copy()).reshape(-1, 1)

    model = Model().eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = torch.get_rng_state()
    result = evaluate_fixed(model, transform, prepared, directory=tmp_path, include_normalization=True)
    assert len(scans) == 1 and len(model.seen) == 2 and model.seen[0] is model.seen[1]
    controls = result["train"]["normalization"]
    for mode in ("saved_running_statistics", "current_scan_statistics"):
        assert controls[mode]["full"]["normal_count"] == 2 and controls[mode]["full"]["anomaly_count"] == 1
    assert controls["saved_running_statistics"]["full"] is result["train"]["full"]
    assert controls["score_changes"][0]["by_label"]["0"]["absolute_mean"] > 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)


def test_fixed_full_labels_keep_zero_and_few_return_frames_and_reuse_threshold(tmp_path, monkeypatch):
    from src.evaluate import fixed_summary, threshold_counts
    monkeypatch.setattr("src.evaluate._evaluation_space", lambda required: None)
    zero = dict(scores=np.array([-4., -2., 99.], np.float32), target=np.array([0, 0, -1], np.int8))
    few = dict(scores=np.array([-3., 2., 3.], np.float32), target=np.array([0, 1, 1], np.int8))
    summary = fixed_summary([zero, few], directory=tmp_path)
    assert summary["frames"] == 2
    assert (summary["normal_count"], summary["anomaly_count"]) == (3, 2)
    assert summary["AP"] == 100. and summary["FPR95"] == 0.
    expected = .5 * (np.logaddexp(0., [-4., -2., -3.]).mean() + np.logaddexp(0., [-2., -3.]).mean())
    assert summary["detection_loss"] == pytest.approx(expected)
    threshold = summary["recall_at_fpr_limit"]["threshold"]
    shifted_zero = dict(zero, scores=zero["scores"] + 5)
    counts = threshold_counts([shifted_zero], threshold)
    assert counts["anomaly"] == 0 and counts["recall"] is None
    assert counts["fp"] == 1 and counts["FPR"] == 50.
    assert threshold_counts([few], None)["tp"] == 0


def test_external_exact_counts_merge_sparse_float32_ranges_and_cross_run_ties(tmp_path):
    from src.evaluate import ScoreCounts, score_groups
    random = np.random.default_rng(193)
    # Thousands of unrelated exponent/mantissa pages must not allocate dense pages.
    broad = random.integers(1, 0x7f800000, 3000, dtype=np.uint32)
    broad[::2] |= np.uint32(0x80000000)
    scores = np.r_[broad.view(np.float32), np.array([-80, -4, -0., 0., 4, 80, 81, 81] * 123, np.float32)]
    target = (np.arange(len(scores)) % 7 < 3).astype(np.int64)
    ordered = np.sort(packed_scores(scores, target, score_kind="logit"))
    expected_groups = tuple(np.concatenate(parts) for parts in zip(*score_groups(ordered), strict=True))
    requests = []
    with ScoreCounts(max_bytes=4096, directory=tmp_path, check_resources=requests.append) as counts:
        for rows in np.array_split(random.permutation(len(scores)), 13):
            counts.add(scores[rows], target[rows])
        counts.add(np.array([], np.float32), np.array([], np.int64))
        groups = tuple(np.concatenate(parts) for parts in zip(*counts.groups(), strict=True))
        for actual, expected in zip(groups, expected_groups, strict=True):
            np.testing.assert_array_equal(actual, expected)
        actual = counts.metrics()
        expected = exact_metrics(ordered, score_kind="logit")
        for name in ("AP", "AUROC", "FPR95"):
            assert actual.pop(name) == pytest.approx(expected.pop(name), abs=1e-12, rel=0)
        assert actual == expected
        assert counts.spills > 10 and counts.merges > 10 and len(requests) > 20
        assert counts.disk_bytes == sum(path.stat().st_size for path in counts.directory.iterdir())
        assert counts.peak_disk_bytes >= counts.disk_bytes > 0
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="4096"):
        ScoreCounts(max_bytes=1)


def test_repeated_background_counts_equal_literal_world_copies(tmp_path):
    from src.evaluate import ScoreCounts
    scores = np.array([-5, -0., 0., 2, 2, 9], np.float32)
    target = np.array([0, 0, 1, 1, 0, 1])
    with ScoreCounts(max_bytes=4096, directory=tmp_path) as weighted, \
            ScoreCounts(max_bytes=4096, directory=tmp_path) as literal:
        for copies in (7, 1, 11):
            weighted.add(scores, target, copies=copies)
            for _ in range(copies):
                literal.add(scores, target)
        assert weighted.metrics() == literal.metrics()
        for threshold in (None, -5, 0, 2, 9, 10):
            assert weighted.at_threshold(threshold) == literal.at_threshold(threshold)
        assert weighted.at_threshold(2)["tp"] == 38
        assert weighted.at_threshold(2)["fp"] == 19
        with pytest.raises(ValueError, match="positive integer"):
            weighted.add(scores, target, copies=0)


def test_external_count_disk_failure_cleans_runs_and_respects_host_reserve(tmp_path, monkeypatch):
    from src.evaluate import ScoreCounts, _evaluation_space
    reserve = 10_000_000_000
    monkeypatch.setattr("src.evaluate.host_disk", lambda: dict(SizeRemaining=reserve + 2000, reserve_bytes=reserve))
    _evaluation_space(2000)
    with pytest.raises(OSError, match="10 GB reserve"):
        _evaluation_space(2001)

    requests = []
    def space(required):
        requests.append(required)
        if len(requests) == 3:
            raise OSError("fixture disk exhausted")

    with pytest.raises(OSError, match="fixture disk"):
        with ScoreCounts(max_bytes=4096, directory=tmp_path, check_resources=space) as counts:
            counts.add(np.arange(500, dtype=np.float32), np.arange(500) % 2)
    assert len(requests) == 3 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("last_distance", [50.0, 50.01])
def test_synthetic_official_filter_preserves_insertion_and_ignore_targets(last_distance):
    from src.data import FrozenFrame
    from src.evaluate import official_frame, synthetic_targets
    ranges = np.array([2.5, 50, 10, 20, 10, 10, 10, last_distance], np.float32)
    xyzi = np.zeros((len(ranges), 4), np.float32)
    xyzi[:, 0] = ranges
    semantic = np.array([2, 2, 2, 2, 2, 52, 40, 2], np.uint16)
    inserted = np.array([1, 1, 1, 1, 0, 0, 0, 1], bool)
    instance = np.where(inserted, 60001, 0).astype(np.uint16)
    packed = (instance.astype(np.uint32) << 16) | semantic.astype(np.uint32)
    mapped = np.array([255, 255, 255, 255, 255, 255, 0, 255], np.uint8)
    source = make_source_frame(0, xyzi, np.eye(4), PointLabels(packed, semantic, instance, mapped),
                               partition="fixture", sequence_id=201)
    frozen = FrozenFrame(source, "a" * 64, inserted, np.zeros(len(ranges), bool))
    full, _ = synthetic_targets(frozen)
    np.testing.assert_array_equal(full, [1, 1, 1, 1, -1, -1, 0, 1])
    filtered, eligible = synthetic_targets(frozen, official=True)
    expected = full.copy()
    if last_distance > 50:
        expected[-1] = -1
    np.testing.assert_array_equal(filtered, expected)
    assert eligible == (last_distance <= 50)
    # Real validation still follows raw STU semantics, including the native class 2.
    prediction = FramePrediction("fixture", 201, 0, source.real_slots, np.arange(len(ranges), dtype=np.float32))
    _, real_target, real_eligible = official_frame(source, prediction)
    assert real_target[4] == 1 and real_target[5] == 0 and real_eligible


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


def test_multiple_global_fpr_limits_match_individual_complete_tie_reductions(tmp_path):
    scores = np.array([9, 8, 8, 7, 6, 6, 5] + [-4] * 9998, np.float32)
    target = np.array([1, 1, 0, 1, 1, 0, 0] + [0] * 9998)
    records = packed_scores(scores, target, score_kind="logit")
    limits = (0.01, 0.001, 0.0001)
    path = tmp_path / "scores.bin"
    records.tofile(path)
    pooled = pooled_files([path], score_kind="logit", fpr_limits=limits)
    for chunk in (2, 100000):
        result = exact_metrics(np.sort(records), chunk_size=chunk, score_kind="logit", fpr_limits=limits)
        for limit in limits:
            reference = exact_metrics(np.sort(records), chunk_size=chunk, score_kind="logit", fpr_limit=limit)
            point = result["operating_points"][f"{limit:g}"]
            assert {key: point[key] for key in reference["recall_at_fpr_limit"]} == reference["recall_at_fpr_limit"]
            assert pooled["operating_points"][f"{limit:g}"] == point
            accepted = scores >= point["threshold"]
            assert point["fp"] == np.count_nonzero(accepted & (target == 0))
            assert point["tp"] == np.count_nonzero(accepted & (target == 1))
            assert point["precision"] == pytest.approx(100 * target[accepted].mean(), abs=1e-12)


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


@pytest.mark.parametrize("block_size", [1, 3, 1000000])
def test_merged_frame_ties_match_exact_point_metrics(block_size):
    scores = np.array(
        [-80, -4, -0.0, 0.0, 4, 80, 81, 81] * 13, dtype=np.float32
    )
    labels = (np.arange(len(scores)) % 7 < 3).astype(np.int64)
    bits = score_bits(scores, "logit")
    merged = {}
    # Merge per-frame unique scores without expanding or rounding any tie.
    for rows in np.array_split(np.arange(len(scores)), 5):
        unique, inverse, counts = np.unique(
            bits[rows], return_inverse=True, return_counts=True
        )
        positives = np.bincount(inverse, weights=labels[rows]).astype(np.int64)
        for key, count, positive in zip(unique, counts, positives, strict=True):
            old_count, old_positive = merged.get(key, (0, 0))
            merged[key] = (old_count + count, old_positive + positive)
    unique = np.array(sorted(merged, reverse=True), dtype=np.uint32)
    counts, positives = np.array([merged[key] for key in unique], np.int64).T
    groups = (
        (unique[start:start + block_size], counts[start:start + block_size],
         positives[start:start + block_size])
        for start in range(0, len(unique), block_size)
    )
    kwargs = dict(score_kind="logit", prevalence=0.03, fpr_limits=(0.1, 0.5))
    grouped_observer, point_observer = APAttribution(), APAttribution()
    result = metrics_from_groups(
        groups, positive=int(labels.sum()), negative=int((labels == 0).sum()),
        observe=grouped_observer, **kwargs,
    )
    expected = exact_metrics(
        np.sort(packed_scores(scores, labels, score_kind="logit")),
        chunk_size=5, observe=point_observer, **kwargs,
    )
    # Only floating summation grouping may differ; thresholds and counts are exact.
    for name in ("AP", "AUROC", "standardized_AP"):
        assert result.pop(name) == pytest.approx(expected.pop(name), abs=1e-12, rel=0)
    assert result == expected
    for name in ("bits", "precision", "required_fpr"):
        np.testing.assert_array_equal(
            np.concatenate(getattr(grouped_observer, name)),
            np.concatenate(getattr(point_observer, name)),
        )


def test_group_metrics_reject_incomplete_ties_and_wrong_totals():
    bits = score_bits([0.9, 0.8], "probability")
    one = np.array([1], np.int64)
    zero = np.array([0], np.int64)
    with pytest.raises(ValueError, match="complete ties"):
        metrics_from_groups(
            [(bits[:1], one, one), (bits[:1], one, zero)], positive=1, negative=1
        )
    with pytest.raises(ValueError, match="match class totals"):
        metrics_from_groups([(bits[:1], one, one)], positive=1, negative=1)


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
    captured = {(1, 0): None}
    result, rows = evaluate_frames(iter(pairs), directory=tmp_path, observe=observer,
                                  per_sequence=True, capture=captured)
    assert result["frames"] == 3 and result["eligible_frames"] == 2
    assert [r["anomaly_points"] for r in rows] == [5, 7, 4]
    for key, value in official_metrics(official).items():
        assert result[key] == pytest.approx(value, abs=1e-10)
        assert result["per_sequence"]["1"]["curve"][key] == pytest.approx(value, abs=1e-10)
    np.testing.assert_array_equal(captured[1, 0], pairs[0][1].restore(pairs[0][0]))
    for key in ("tp", "fp"):
        assert result["per_sequence"]["1"]["at_global_threshold"][key] == result["recall_at_fpr_limit"][key]
    skipped, skipped_rows = evaluate_frames(iter([*pairs[:2], (pairs[2][0], None)]), directory=tmp_path)
    assert rows == skipped_rows
    for key in ("AP", "FPR95", "AUROC", "recall_at_fpr_limit"):
        assert skipped[key] == result[key]
    with pytest.raises(ValueError, match="eligible scan requires"):
        evaluate_frames(iter([(pairs[0][0], None)]), directory=tmp_path)
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
