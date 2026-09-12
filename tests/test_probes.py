import copy

import numpy as np

from src.probes import FEATURES, GEOMETRY, descriptive_matches, fit_reference, score_reference, tail_score


def points(n=512):
    data = {name: np.linspace(0, 1, n, dtype=np.float32) for name in FEATURES}
    data.update(frame=np.arange(n) % 32, source_slot=np.arange(n),
                range=np.full(n, 12., np.float32), ray_z=np.zeros(n, np.float32),
                scale=np.full(n, .2, np.float32), neighbor_count=np.full(n, 24, np.float32))
    return data


def test_empirical_two_tails_include_ties_and_share_tail_ceiling():
    reference = np.array([0., 1., 1., 2.])
    np.testing.assert_allclose(tail_score([0., 1., 2., -1., 3.], reference), [.5, 0, .5, 1, 1])
    np.testing.assert_array_equal(tail_score([-1., 3.], np.tile(reference, 100)), [1, 1])
    assert tail_score([1.], np.ones(512))[0] == 0
    assert np.isnan(tail_score([np.nan], reference)[0])


def test_missing_reference_and_unreliable_geometry_remain_missing():
    train = points()
    reference = fit_reference(train)
    observed = points(3)
    observed["range"][:] = [12, 45, 12]
    observed["normal_change"][2] = np.nan
    scores, individual, _ = score_reference(observed, reference)
    assert np.isnan(scores["C_range"][1:]).all()
    assert np.isfinite(individual["C_range"]["roughness"][2])
    assert np.isnan(scores["B_geometry"][2])
    train["frame"][:] = 0
    unsupported = fit_reference(train)
    assert np.isnan(score_reference(observed, unsupported)[0]["B_geometry"]).all()


def test_held_out_scoring_never_changes_normal_reference():
    reference = fit_reference(points())
    saved = copy.deepcopy(reference)
    held_out = points(5)
    held_out["scale"][:] = [.1, .2, .3, .2, .2]
    held_out["roughness"][:] = 1000
    scores, _, cells = score_reference(held_out, reference)
    assert np.isnan(scores["C_sampling"][[0, 2]]).all()
    assert (cells["sampling"][[0, 2]] == -1).all()
    assert np.isfinite(scores["C_sampling"][[1, 3, 4]]).all()
    assert reference["metadata"] == saved["metadata"]
    for name in FEATURES:
        np.testing.assert_array_equal(reference["global_fields"][name], saved["global_fields"][name])
    for mode, group in reference["conditional"].items():
        for cell, fields in group.items():
            for name in GEOMETRY:
                np.testing.assert_array_equal(fields[name], saved["conditional"][mode][cell][name])


def test_conditioned_scores_reduce_to_identical_rule_for_one_cell():
    train = points()
    reference = fit_reference(train)
    scores, individual, cells = score_reference(points(10), reference)
    for key in ("C_range", "C_direction", "C_sampling"):
        np.testing.assert_array_equal(scores[key], scores["B_geometry"])
        for name in GEOMETRY:
            np.testing.assert_array_equal(individual[key][name], individual["B_geometry"][name])
    assert reference["metadata"]["frames"] == 32
    assert all(len(np.unique(value)) == 1 for value in cells.values())


def test_existing_world_check_preserves_original_field_scores_and_missing_support(monkeypatch):
    from src import coverage

    fitted = fit_reference(points())
    fields = coverage.GEOMETRY_FIELDS
    conditional = {cell: {name: values[name] for name in fields}
                   for cell, values in fitted["conditional"]["direction"].items()}
    thresholds = {name: .99 for name in fields}
    reference = dict(edges=fitted["edges"], conditional=conditional, thresholds=thresholds,
                     quantiles={cell: {name: np.quantile(values[name], [.1, .9]) for name in fields}
                                for cell, values in conditional.items()})
    monkeypatch.setattr(coverage, "_geometry_reference", reference, raising=False)
    sample = points(7)
    sample["normal_change"][:] = [-1., 0., .5, 1., 2., np.nan, .5]
    sample["surface_residual"][:] = [.2, .2, .2, .2, .2, .2, .2]
    sample["range"][-1] = 45  # Geometry exists, but this condition has no reference.
    actual = coverage._geometry_condition(sample)
    _, individual, _ = score_reference(sample, fitted)
    for name in fields:
        expected = individual["C_direction"][name]
        np.testing.assert_array_equal(actual[name]["covered"], np.isfinite(expected))
        np.testing.assert_array_equal(actual[name]["hit"], expected >= thresholds[name])
        np.testing.assert_array_equal(actual[name]["lower"] | actual[name]["central"] | actual[name]["upper"],
                                      actual[name]["covered"])
    assert actual["surface_residual"]["covered"][5]
    assert not actual["normal_change"]["covered"][5]


def test_descriptive_weighted_ties_have_exact_ks_and_directional_auc():
    sample = points(6)
    sample.update(sequence=np.ones(6, int), target=np.array([0, 0, 0, 1, 1, 1]),
                  weight=np.array([1., 1., 2., 1., 2., 1.]),
                  roughness=np.array([0., 1., 1., 0., 1., 2.]))
    rows = descriptive_matches(sample, fit_reference(points()))
    row = next(row for row in rows if row["conditioning"] == "all" and row["feature"] == "roughness")
    assert row["ks"] == .25
    assert row["auc_greater"] == 9.5 / 16
    assert row["normal"] == dict(sampled=3, weighted_total=4., frames=3, valid=3,
                                 weighted_valid=4., valid_frames=3, missing=0, weighted_missing=0.,
                                 p10=0., median=1., p90=1.)
    assert row["anomaly"]["p90"] == 2.


def test_descriptive_missing_denominators_and_sequence_frame_identity():
    sample = points(5)
    sample.update(sequence=np.array([101, 101, 102, 101, 102]), frame=np.array([0, 1, 0, 0, 0]),
                  target=np.array([0, 0, 0, 1, 1]), weight=np.array([1., 2., 3., 4., 5.]),
                  roughness=np.array([np.nan, 1., 3., np.nan, 4.]))
    rows = descriptive_matches(sample, fit_reference(points()))
    row = next(row for row in rows if row["conditioning"] == "all" and row["feature"] == "roughness")
    for label, sampled, total, frames, missing_weight in (("normal", 3, 6, 3, 1), ("anomaly", 2, 9, 2, 4)):
        stats = row[label]
        assert stats["sampled"] == sampled and stats["weighted_total"] == total
        assert stats["valid"] == sampled - 1 and stats["weighted_valid"] == total - missing_weight
        assert stats["frames"] == frames and stats["valid_frames"] == frames - 1
        assert stats["missing"] == 1 and stats["weighted_missing"] == missing_weight
    assert row["ks"] == 1 and row["auc_greater"] == 1


def test_exact_score_group_merge_and_sequence_removal_match_expanded_metrics():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from src.evaluate import exact_metrics, packed_scores
    from src.geometry import GROUP_DTYPE, group_metrics, merge_count_archives, merge_groups, score_groups, subtract_groups

    rng = np.random.default_rng(104)
    scores = rng.choice(np.array([0., .1, .2, .5, .9, 1.], np.float32), 517)
    labels = (np.arange(len(scores)) % 4 == 0).astype(np.int8)
    sequence = np.arange(len(scores)) % 3
    parts = [score_groups(scores[sequence == s], labels[sequence == s]) for s in range(3)]
    pooled = merge_groups(parts)
    np.testing.assert_array_equal(pooled, score_groups(scores, labels))
    assert group_metrics(pooled) == exact_metrics(np.sort(packed_scores(scores, labels)))
    for s, removed in enumerate(parts):
        remaining = subtract_groups(pooled, removed)
        keep = sequence != s
        np.testing.assert_array_equal(remaining, score_groups(scores[keep], labels[keep]))
        assert group_metrics(remaining) == exact_metrics(np.sort(packed_scores(scores[keep], labels[keep])))
    assert len(subtract_groups(pooled, pooled)) == 0
    # Exercise real compressed archives, including cross-block ties and an empty comparison.
    empty = np.empty(0, GROUP_DTYPE)
    with TemporaryDirectory(prefix="ajae-count-test-") as temporary:
        directory = Path(temporary)
        chunks = []
        for index, part in enumerate(parts):
            path = directory / f"{index}.npz"
            np.savez_compressed(path, **{"0": part, "1": part[::2], "2": empty})
            chunks.append(path)
        destination = directory / "merged.npz"
        merge_count_archives(chunks, destination, ["all", "subset", "empty"])
        with np.load(destination, allow_pickle=False) as saved:
            assert saved.files == ["0", "1", "2"]
            for key, expected in enumerate((pooled, merge_groups([p[::2] for p in parts]), empty)):
                np.testing.assert_array_equal(saved[str(key)], expected)


def test_comparison_cohorts_share_finite_identities_without_discarding_other_features():
    from src.geometry import comparisons

    observed = points(4)
    observed["normal_change"][1] = np.nan
    observed["scale"][2] = np.nan
    observed["range"][3] = 45  # No fitted conditional support for this range cell.
    scores, individual, _ = score_reference(observed, fit_reference(points()))
    groups = {name: (methods, finite) for name, methods, finite in comparisons(scores, individual)}
    expected = dict(range=[True, False, True, False], direction=[True, False, True, False],
                    sampling=[True, False, False, False], normalization=[True, False, True, True])
    for cohort, finite in expected.items():
        methods, mask = groups[cohort]
        np.testing.assert_array_equal(mask, finite)
        np.testing.assert_array_equal(mask, np.logical_and.reduce([np.isfinite(a) for a in methods.values()]))
    # A missing normal-change estimate must not erase the separate roughness comparison.
    np.testing.assert_array_equal(groups["feature/roughness/range"][1], [True, True, True, False])
    np.testing.assert_array_equal(groups["feature/normal_change/range"][1], [True, False, True, False])


def test_subgroup_weights_missing_coverage_and_fixed_threshold_rejection():
    from src.geometry import comparisons
    from src.probes import subgroup_diagnostics

    reference, sample = fit_reference(points()), points(24)
    # Two sequences share frame zero; each has ten complete anomalies and two normal samples.
    sample.update(sequence=np.repeat([101, 102], 12), frame=np.zeros(24, int),
                  target=np.ones(24, np.int8), weight=np.ones(24))
    sample["target"][[0, 1, 12, 13]] = 0
    sample["weight"][[0, 1]], sample["weight"][[12, 13]] = 2, 3
    for name in (*GEOMETRY, "residual_scaled", "roughness_scaled"):
        sample[name][:] = .5
    sample["surface_residual"][[1, 13, 2, 3]] = 2
    sample["roughness"][[0, 14]] = np.nan
    sample["roughness_scaled"][[0, 14]] = np.nan
    sample["scale"][[0, 14]] = np.nan
    scores, individual, _ = score_reference(sample, reference)
    thresholds = {cohort + "|" + method: .5 for cohort, methods, _ in comparisons(scores, individual)
                  if not cohort.startswith("feature/") for method in methods}
    thresholds["range|A_range"] = None
    rows = subgroup_diagnostics(sample, reference, thresholds)
    row = next(r for r in rows if (r["axis"], r["group"], r["cohort"], r["model"])
               == ("range", "1", "range", "B_geometry"))
    for key, expected in dict(normal_sampled_total=4, normal_weighted_total=10,
                              normal_sampled_covered=3, normal_weighted_covered=8,
                              normal_sampled_fp=2, normal_weighted_fp=5,
                              anomaly_total=20, anomaly_covered=19, anomaly_tp=2).items():
        assert row[key] == expected
    np.testing.assert_allclose([row["normal_coverage_estimate_percent"], row["anomaly_coverage_percent"],
                                row["normal_fpr_estimate_percent"], row["anomaly_recall_covered_percent"],
                                row["anomaly_detected_fraction_all_percent"]], [80, 95, 62.5, 200 / 19, 10])
    rejected = next(r for r in rows if (r["axis"], r["cohort"], r["model"])
                    == ("range", "range", "A_range"))
    assert rejected["normal_weighted_covered"] == 8 and rejected["anomaly_covered"] == 19
    assert rejected["normal_weighted_fp"] == rejected["anomaly_tp"] == 0
    returns = [r for r in rows if r["axis"] == "returns"]
    assert {r["group"] for r in returns} == {"0"}
    assert all(r["anomaly_total"] == 20 for r in returns)
    missing = next(r for r in rows if (r["axis"], r["group"], r["cohort"], r["model"])
                   == ("scale", "missing_or_outside_reference", "range", "B_geometry"))
    assert missing["normal_weighted_total"] == 2 and missing["anomaly_total"] == 1
    assert missing["normal_sampled_covered"] == missing["anomaly_covered"] == 0
    assert missing["normal_fpr_estimate_percent"] is missing["anomaly_recall_covered_percent"] is None
