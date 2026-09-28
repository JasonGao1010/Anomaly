"""Complete-result summaries preserve seed identities and statistical populations."""

import json

import numpy as np
import pytest

from src.evaluate import summarize_experiments
from src.model import NORMAL_VARIANTS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment_directory(tmp_path):
    present = [0, 1, 3, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]
    truth = [10 if category in present else 0 for category in range(19)]
    fixed = ["joint_appearance", "joint_common_density", "joint_independent_minima"]
    methods = ["semantic", "joint", "separate", *fixed]
    for index, seed in enumerate((206, 307, 409)):
        def metrics(method):
            return dict(AP=20. + index + (2 if method == "joint" else 0), AUROC=96. + index / 2, FPR95=10. - index)

        def iou(method):
            return .4 + .01 * index + (.02 if method in ("joint", "joint_independent_minima") else 0)

        prediction = dict(queries=10 + index, nll=-2. + .1 * index, coverage90=.9 - .01 * index,
                          width90_m=2. + .2 * index, median_mae_m=.5 + .1 * index)
        prediction["by_range"] = [dict(range_m=interval, query_share=share, **prediction)
            for interval, share in zip(("[2.5,10)", "[10,20)", "[20,35)", "[35,50]"), (.1, .2, .3, .4))]
        for variant in NORMAL_VARIANTS:
            run = tmp_path / f"{variant}-{seed}"
            normal = dict(mean_iou_gt=iou(variant), ground_truth_points=truth,
                          iou=[iou(variant) if category in present else None for category in range(19)], predictive=prediction)
            write(run / "normal201.json", normal)
            for readout in (("energy", "softmax") if variant == "standard" else ("joint",)):
                row = dict(seed=seed, variant=variant, readout=readout, split="val", complete=True,
                           manifest_sha256="measured-fixture", architecture="same", score_version="same",
                           points=100, scans=2, metrics=metrics(variant))
                write(run / (f"val_{readout}.json" if variant == "standard" else "val.json"), row)
            if variant in ("semantic", "joint"):
                write(run / "runtime.json", dict(scans=["scan-a", "scan-b"], hardware="CPU fixture", precision="FP32",
                    batch_size=1, warmup_scans=50, mean_ms=20. + index + (3 if variant == "joint" else 0),
                    peak_vram_bytes=(3. + index / 10) * 1e9, timing_scope="fixture", memory_scope="fixture"))
        normal = {method: dict(mean_iou_gt=iou(method), corrected_fraction=.02 + index / 100,
            introduced_fraction=.01, net_accuracy_change=.01 + index / 100) for method in methods}
        paired = dict(seed=seed, conditions=dict(matched=True, differences=[]), split="val",
            manifest_sha256="measured-fixture", points=100, scans=2, baseline_confidence_threshold=.9,
            confident_unknown_points=10 + index, methods={}, instance_coverage=dict(
                size_bins=["1-4", "5-9", "10-19", "20-49", "50-99", "100+"], instance_counts=[2] * 6,
                unassigned_anomaly_points=0, methods={}))
        for method in methods:
            operating = [dict(requested_fpr=fpr, actual_fpr=fpr, confident_unknown_recall=.5 + index / 10,
                confident_unknown_recovered_fraction=.2, confident_unknown_lost_fraction=.1,
                confident_unknown_net_recall_change=.1) for fpr in (.001, .005, .01, .02, .05)]
            paired["methods"][method] = dict(metrics=metrics(method), operating_points=operating)
            paired["instance_coverage"]["methods"][method] = [dict(requested_fpr=row["requested_fpr"],
                actual_fpr=row["actual_fpr"], recall=[.5] * 6) for row in operating]
        write(tmp_path / f"paired-{seed}/comparison.json", paired)
        write(tmp_path / f"paired-{seed}/normal.json", dict(seed=seed, conditions=dict(matched=True, differences=[]), methods=normal))
    return tmp_path


def test_summary_reports_every_table_and_uses_seed_level_sample_deviations(experiment_directory):
    result = summarize_experiments(experiment_directory)
    assert len(result["main"]) == 6 and len(result["ablations"]) == 9 and len(result["per_class_iou"]) == 14
    assert result["main"]["joint"]["validation"]["AP"] == dict(mean=23., std=1., per_seed={"206": 22., "307": 23., "409": 24.})
    assert len(result["missing_test"]) == 6 and all(row["test"] is None for row in result["main"].values())
    assert all(len(row["missing_files"]) == 3 and row["reason"] for row in result["missing_test"])
    np.testing.assert_allclose(result["per_class_iou"][0]["difference"]["mean"], 2.)
    np.testing.assert_allclose(result["per_class_iou"][0]["appearance"]["std"], 1.)
    assert result["ablations"]["joint_appearance"]["normal_mIoU"] == result["ablations"]["joint_common_density"]["normal_mIoU"]
    assert result["ablations"]["joint_independent_minima"]["normal_mIoU"] == result["ablations"]["joint"]["normal_mIoU"]
    assert result["runtime"]["additional_ms"]["mean"] == 3
    assert result["predictive"]["joint"]["coverage90"]["mean"] == 89
    assert result["predictive"]["joint"]["by_range"][3]["query_share"]["mean"] == 40
    assert result["confident_unknown"]["methods"]["joint"][2]["confident_unknown_recall"]["mean"] == 60
    assert result["instance_coverage"]["complete"] is True
    json.dumps(result, allow_nan=False)


def test_missing_instance_ids_leave_point_results_available_and_mark_object_population(experiment_directory):
    for seed in (206, 307, 409):
        path = experiment_directory / f"paired-{seed}/comparison.json"
        paired = json.loads(path.read_text())
        paired["instance_coverage"]["unassigned_anomaly_points"] = 3
        write(path, paired)
    result = summarize_experiments(experiment_directory)
    coverage = result["instance_coverage"]
    assert coverage["complete"] is False and coverage["unassigned_anomaly_points"] == 3
    assert "positive instance ID" in coverage["population"]
    assert result["main"]["joint"]["validation"]["AP"]["mean"] == 23


def test_summary_rejects_unmatched_seeds_or_reordered_operating_points(experiment_directory):
    path = experiment_directory / "paired-307/comparison.json"
    paired = json.loads(path.read_text())
    paired["conditions"]["matched"] = False
    write(path, paired)
    with pytest.raises(ValueError, match="matched completed"):
        summarize_experiments(experiment_directory)
    paired["conditions"]["matched"] = True
    paired["methods"]["joint"]["operating_points"].reverse()
    write(path, paired)
    with pytest.raises(ValueError, match="operating-point order"):
        summarize_experiments(experiment_directory)


def test_summary_requires_non_test_artifacts_and_never_averages_partial_test(experiment_directory):
    source = experiment_directory / "joint-206/val.json"
    row = json.loads(source.read_text())
    row["split"] = "test"
    write(source.with_name("test.json"), row)
    result = summarize_experiments(experiment_directory)
    assert result["main"]["joint"]["test"] is None
    missing = next(row for row in result["missing_test"] if row["method"] == "joint")
    assert missing["missing_files"] == ["joint-307/test.json", "joint-409/test.json"]
    (experiment_directory / "cssr-409/normal201.json").unlink()
    with pytest.raises(FileNotFoundError, match="normal201.json"):
        summarize_experiments(experiment_directory)


def test_empty_scientific_populations_stay_undefined_without_averaging_two_seeds(experiment_directory):
    for seed in (206, 307, 409):
        path = experiment_directory / f"paired-{seed}/comparison.json"
        paired = json.loads(path.read_text())
        paired["instance_coverage"]["instance_counts"][0] = 0
        for rows in paired["instance_coverage"]["methods"].values():
            for row in rows:
                row["recall"][0] = None
        if seed == 307:
            paired["confident_unknown_points"] = 0
            for method in paired["methods"].values():
                for row in method["operating_points"]:
                    for key in list(row):
                        if key.startswith("confident_unknown_"):
                            row[key] = None
        write(path, paired)
    path = experiment_directory / "joint-307/normal201.json"
    normal = json.loads(path.read_text())
    stratum = normal["predictive"]["by_range"][3]
    stratum.update(queries=0, query_share=0, nll=None, coverage90=None, width90_m=None, median_mae_m=None)
    write(path, normal)
    result = summarize_experiments(experiment_directory)
    recall = result["confident_unknown"]["methods"]["joint"][2]["confident_unknown_recall"]
    assert recall["mean"] is None and recall["std"] is None and recall["undefined_reason"]
    assert recall["per_seed"] == {"206": 50., "307": None, "409": 70.}
    assert result["instance_coverage"]["methods"]["joint"][2]["recall"]["mean"] == [None, 50., 50., 50., 50., 50.]
    assert result["predictive"]["joint"]["by_range"][3]["nll"]["mean"] is None
    assert result["predictive"]["joint"]["by_range"][3]["query_share"]["mean"] == pytest.approx(80 / 3)
    json.dumps(result, allow_nan=False)
    normal["predictive"]["nll"] = float("nan")
    write(path, normal)
    with pytest.raises(ValueError, match="finite seed-level"):
        summarize_experiments(experiment_directory)
