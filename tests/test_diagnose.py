import numpy as np
import pytest

from src.data import FrozenFrame
from src.diagnose import scope_masks, range_ids, recurrence
from src.scene import PointLabels, make_source_frame


def test_synthetic_scope_preserves_inserted_identity_ignore_and_official_boundaries():
    distance = np.array([2.5, 50, 10, 20, 35, 2.49, 50.01, 15, 16, 17], np.float32)
    xyzi = np.zeros((len(distance), 4), np.float32)
    xyzi[:, 0] = distance
    inserted = np.array([True] * 7 + [False] * 3)
    raw = np.array([2] * 8 + [40, 0], np.uint16)
    instance = np.where(inserted, 60001, 0).astype(np.uint16)
    semantic_target = np.where(raw == 40, 8, 255).astype(np.uint8)
    packed = raw.astype(np.uint32) | (instance.astype(np.uint32) << 16)
    source = make_source_frame(0, xyzi, np.eye(4), PointLabels(packed, raw, instance, semantic_target),
                               partition="train", sequence_id=201)
    frozen = FrozenFrame(source, "a" * 64, inserted, np.zeros(len(raw), bool))
    target = frozen.anomaly_target
    masks, actual_distance, eligible = scope_masks(source.xyzi, target)
    assert eligible
    assert np.count_nonzero(masks[0]) == 8
    assert np.count_nonzero(masks[1] & (target == 1)) == 5
    # A source semantic 2 without an inserted return is ignored, never made positive.
    assert target[7] == -1 and not masks[1][7] and target[8] == 0
    changed = target.copy()
    changed[4] = -1
    assert not scope_masks(source.xyzi, changed)[2]
    np.testing.assert_array_equal(range_ids(actual_distance[:7]), [1, 4, 2, 3, 4, 0, 5])


def test_background_identity_counts_worlds_and_coincident_slots_separately():
    # The first two original slots coincide; each world may contribute both rows.
    slot_seen = np.array([3, 3, 1], np.uint32)
    slot_fp = np.array([[3, 2, 0], [1, 1, 0], [0, 0, 0]], np.uint32)
    position_seen = np.zeros(2, np.uint32)
    position_fp = np.zeros((3, 2), np.uint32)
    positions = [0, 0, 1]
    np.bitwise_or.at(position_seen, positions, slot_seen)
    for k in range(3):
        np.bitwise_or.at(position_fp[k], positions, slot_fp[k])
    slots = recurrence(slot_seen, slot_fp)
    unique = recurrence(position_seen, position_fp)
    assert slots["observed"] == 3 and unique["observed"] == 2
    assert slots["false_positive"] == [2, 2, 0]
    assert unique["false_positive"] == [1, 1, 0]
    assert unique["fp_world_histograms"][0][2] == 1
    assert unique["fp_world_histograms"][1][1] == 1


def test_real_required_fpr_counts_complete_ties_and_separates_frame_from_global():
    from src.diagnose import required_frame_fpr, anomaly_summary, POINT_DTYPE
    from src.evaluate import APAttribution, exact_metrics, packed_scores

    scores = np.array([2, 2, -1, -2, 4, 3, 1, 0], np.float32)
    target = np.array([1, 0, 1, 0, 0, 0, 1, 0])
    frame = np.array([0] * 4 + [1] * 4)
    observer = APAttribution()
    metrics = exact_metrics(np.sort(packed_scores(scores, target, score_kind="logit")),
                            score_kind="logit", observe=observer, chunk_size=2)
    positive = target == 1
    points = np.zeros(positive.sum(), POINT_DTYPE)
    points["score"] = scores[positive]
    points["frame"] = frame[positive]
    points["precision"], points["q_global"] = observer.values(points["score"])
    for fid in (0, 1):
        normals = scores[(frame == fid) & (target == 0)]
        use = points["frame"] == fid
        points["q_frame"][use] = required_frame_fpr(normals, points["score"][use])
        for score, actual in zip(points["score"][use], points["q_frame"][use], strict=True):
            assert actual == np.mean(normals >= score)
    np.testing.assert_allclose(points["q_frame"], [.5, .5, 2/3])
    np.testing.assert_allclose(points["q_global"], [.6, .8, .6])
    parts = [anomaly_summary(points[points["frame"] == fid], len(points), [2., -1.]) for fid in (0, 1)]
    assert abs(sum(r["ap_deficit_pp"] for r in parts) - (100-metrics["AP"])) < 1e-12
    assert sum(r["tp_1"] for r in parts) == 1
    assert sum(r["tp_95"] for r in parts) == 3


def test_compact_synthetic_replay_preserves_slots_masks_and_cross_frame_logit_ties(tmp_path):
    from src.diagnose import _paired_cached_synthetic, _paired_metrics
    from src.evaluate import APAttribution, exact_metrics, packed_scores

    rows, expected, all_scores, all_labels = [], [], [], []
    with (tmp_path / "scores").open("w+b") as scores, (tmp_path / "flags").open("w+b") as flags:
        for frame, n in enumerate((23, 31)):
            xyzi = np.zeros((n, 4), np.float32)
            xyzi[:, 0] = np.linspace(1, 55, n)
            raw = np.full(n, 40, np.uint16)
            raw[[1, 4]] = 0
            inserted = np.zeros(n, bool)
            inserted[6:12] = True
            raw[inserted] = 2
            instance = np.where(inserted, 60001, 0).astype(np.uint16)
            semantic_target = np.where(raw == 40, 8, 255).astype(np.uint8)
            packed_label = raw.astype(np.uint32) | (instance.astype(np.uint32) << 16)
            source = make_source_frame(frame, xyzi, np.eye(4),
                PointLabels(packed_label, raw, instance, semantic_target), partition="train", sequence_id=201)
            frozen = FrozenFrame(source, "a" * 64, inserted, np.zeros(n, bool))
            target = frozen.anomaly_target
            slots = np.flatnonzero(target >= 0)
            score = ((slots % 7)-3).astype(np.float32)  # Negative logits and ties cross both frames.
            masks = np.stack((slots % 2 == 0, slots % 3 == 0, slots % 5 != 0))
            scopes, _, eligible = scope_masks(source.xyzi, target)
            row = dict(offset=scores.tell()//4, flag_offset=flags.tell(), eligible=eligible)
            for scope, use in zip(("all", "stu_filtered"), scopes, strict=True):
                row[scope] = dict(normal=int(np.sum(use & (target == 0))), anomaly=int(np.sum(use & (target == 1))))
            score.tofile(scores)
            np.packbits(masks, axis=1).tofile(flags)
            rows.append((row, frozen))
            expected.append((slots, score, masks))
            all_scores.append(score)
            all_labels.append(target[slots])
        scores.flush()
        flags.flush()
        with (tmp_path / "packed").open("w+b") as stream:
            for (row, frozen), (slots, score, masks) in zip(rows, expected, strict=True):
                restored_slots, restored_score, labels, restored_masks, _, _ = _paired_cached_synthetic(row, frozen, scores, flags)
                np.testing.assert_array_equal(restored_slots, slots)
                np.testing.assert_array_equal(restored_score, score)
                np.testing.assert_array_equal(restored_masks, masks)
                packed_scores(restored_score, labels, score_kind="logit").tofile(stream)
            observer = APAttribution()
            result = _paired_metrics(stream, sum(map(len, all_scores)), observer)
        ordered = np.sort(packed_scores(np.concatenate(all_scores), np.concatenate(all_labels), score_kind="logit"))
        expected_metrics = exact_metrics(ordered, score_kind="logit", fpr_limits=(.01, .001, .0001))
        assert result == expected_metrics
        positives = np.concatenate(all_scores)[np.concatenate(all_labels) == 1]
        normal_scores = np.concatenate(all_scores)[np.concatenate(all_labels) == 0]
        _, required = observer.values(positives)
        np.testing.assert_array_equal(required, [np.mean(normal_scores >= value) for value in positives])
        broken = dict(rows[0][0], offset=1000)
        with pytest.raises(ValueError, match="truncated"):
            _paired_cached_synthetic(broken, rows[0][1], scores, flags)


def test_paired_groups_use_each_arm_threshold_and_full_ties():
    from src.diagnose import _paired_add, _paired_group_rows, _paired_thresholds
    from src.evaluate import exact_metrics, packed_scores

    score = np.r_[np.linspace(-5, 0, 200), [2, 1, 0, -1]].astype(np.float32)
    labels = np.r_[np.zeros(200, np.int8), np.ones(4, np.int8)]
    results, thresholds = [], []
    for shifted in (score, score+20):
        metrics = exact_metrics(np.sort(packed_scores(shifted, labels, score_kind="logit")),
                                score_kind="logit", fpr_limits=(.01, .001, .0001))
        threshold = _paired_thresholds(metrics)
        thresholds.append(threshold)
        groups = {}
        _paired_add(groups, "official", "all", "all", shifted, labels, threshold)
        _paired_add(groups, "official", "parts", np.arange(len(score)) % 2, shifted, labels, threshold)
        result = _paired_group_rows(groups)
        for index, name in enumerate(("1", "0.1", "0.01", "95")):
            expected_fp = int(np.sum((labels == 0) & (shifted >= threshold[index])))
            expected_tp = int(np.sum((labels == 1) & (shifted >= threshold[index])))
            assert result[0]["fp_"+name] == expected_fp
            assert result[0]["tp_"+name] == expected_tp
            assert sum(r["fp_"+name] for r in result[1:]) == expected_fp
        results.append(result)
    assert thresholds[0][0] != thresholds[1][0]
    assert results[0] == results[1]


def test_existing_normal_context_excludes_unknowns_and_missing_surface_targets():
    from src.diagnose import _paired_synthetic_masks

    xyz = np.array([[0, 0, 0], [2, 0, 0], [2.0001, 0, 0], [1, 0, 0], [.5, 0, 0]])
    labels = np.array([1, 0, 0, -1, 0])
    target = dict(surface_valid=np.array([True, True, False, True, True]),
                  surface_offset_z=np.array([-.1, -.05, -.1, -.1, np.nan]))
    near, raised, valid = _paired_synthetic_masks(xyz, labels, target)
    np.testing.assert_array_equal(near, [False, True, False, False, True])
    np.testing.assert_array_equal(raised, [False, True, False, False, False])
    np.testing.assert_array_equal(valid, target["surface_valid"])


def test_paired_anomalies_match_slots_not_rank_and_reject_changed_identity():
    from src.diagnose import POINT_DTYPE, _paired_compare_points

    first = np.zeros(3, POINT_DTYPE)
    first["sequence"] = 125
    first["frame"] = [2, 1, 1]
    first["slot"] = [5, 7, 3]
    first["distance"] = 4
    first["q_frame"] = [.3, .1, .2]
    second = first[[2, 0, 1]].copy()
    second["q_frame"] += .05
    rows = _paired_compare_points(first, second)
    assert [(r["frame"], r["slot"]) for r in rows] == [(1, 3), (1, 7), (2, 5)]
    np.testing.assert_allclose([r["q_frame_delta_pp"] for r in rows], -5)
    second["slot"][0] += 1
    with pytest.raises(ValueError, match="identities differ"):
        _paired_compare_points(first, second)


def test_complete_comparison_keeps_independent_work_points_and_rejects_denominator_change(tmp_path):
    import json
    from src.data import _atomic_json
    from src.diagnose import (POINT_DTYPE, _paired_add, _paired_group_rows, _paired_thresholds,
                              compare_paired_runs)
    from src.evaluate import APAttribution, exact_metrics, packed_scores

    base = np.r_[np.linspace(-5, 0, 200), [2, 1, 0, -1]].astype(np.float32)
    labels = np.r_[np.zeros(200, np.int8), np.ones(4, np.int8)]
    for arm, scale, shift in (("joint", 1., 0.), ("detection", 0., 20.)):
        run = tmp_path / arm
        run.mkdir()
        _atomic_json(run / "protocol.json", dict(training=dict(auxiliary_scale=scale)))
        score = base + shift
        observer = APAttribution()
        metrics = exact_metrics(np.sort(packed_scores(score, labels, score_kind="logit")),
                                score_kind="logit", fpr_limits=(.01, .001, .0001), observe=observer)
        groups = {}
        _paired_add(groups, "official", "all", "all", score, labels, _paired_thresholds(metrics))
        group_rows = _paired_group_rows(groups)
        counts = dict(normal=200, anomaly=4)
        synthetic = dict(metrics={scope: metrics for scope in ("all", "stu_filtered")},
            frames=[dict(world="test_world", world_identity="test", frame=0, eligible=True,
                         actual_points=204, all=counts, stu_filtered=counts)], groups=group_rows, detection_loss=1.)
        real = dict(metrics=metrics, groups=group_rows, sequences={"125": dict(metrics=metrics)},
            frames=[dict(sequence=125, frame=0, eligible=True, actual_points=204, normal_points=200, anomaly_points=4)])
        result = dict(binding=dict(dataset_sha256="fixture", checkpoint_sha256=arm), fixed_cases={}, definitions={},
                      synthetic=synthetic, real=real, original_normal=dict(groups=group_rows))
        _atomic_json(run / "evaluation.json", result)
        points = np.zeros(4, POINT_DTYPE)
        points["sequence"], points["slot"], points["distance"] = 125, np.arange(200, 204), 5
        points["score"] = score[labels == 1]
        points["precision"], points["q_global"] = observer.values(points["score"])
        points["q_frame"] = points["q_global"]
        np.savez(run / "anomalies.npz", **{key: points[key] for key in POINT_DTYPE.names})
    compared = compare_paired_runs(tmp_path)
    assert compared["denominator_check"]
    assert compared["sequence125"]["q_frame"]["equal"] == 4
    r1 = next(row for row in compared["metric_rows"] if row["domain"] == "real" and row["metric"] == "R1")
    assert r1["threshold_joint"] != r1["threshold_detection"]
    assert r1["delta_pp"] == 0
    path = tmp_path / "detection" / "evaluation.json"
    broken = json.loads(path.read_text())
    broken["real"]["frames"][0]["normal_points"] += 1
    _atomic_json(path, broken)
    with pytest.raises(ValueError, match="membership or point denominators"):
        compare_paired_runs(tmp_path)
