import numpy as np

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
