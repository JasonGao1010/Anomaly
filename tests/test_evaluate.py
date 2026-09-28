"""SERVE inference and official evaluation preserve original point identities."""

import numpy as np
import pytest
import torch
from torch import nn
import json
import sys
import hashlib

import src.evaluate as evaluation
from src.model import NORMAL_VERSION
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


@pytest.fixture
def hypothesis_model():
    class Model(nn.Module):
        mode = "normal_hypothesis"
        variant = "joint"

        def forward(self, sample):
            return sample["xyzi"][:, 0]

        def predict(self, sample):
            score = self(sample)
            return dict(score=score, semantic=torch.full_like(score, 18, dtype=torch.long))

    return Model().eval()


def test_infer_exports_stu19_in_original_slots(tmp_path, hypothesis_model):
    points = np.array([[10., 0, 0, .2], [0., 0, 0, 0], [5., 0, 0, .4]], np.float32)
    path = tmp_path / "206" / "velodyne" / "000000.bin"
    path.parent.mkdir(parents=True)
    points.tofile(path)
    result, timing = evaluation.infer(hypothesis_model, path, torch.device("cpu"), return_semantics=True)
    np.testing.assert_array_equal(result["score"], [10., 0., 5.])
    np.testing.assert_array_equal(result["semantic"], [18, -1, 18])
    assert timing["real_points"] == 2 and timing["slots"] == 3
    record = dict(scan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), slots=3, points=2)
    score, _ = evaluation.infer(hypothesis_model, path, torch.device("cpu"), record=record, partition="test")
    np.testing.assert_array_equal(score, result["score"])
    record["scan_sha256"] = "changed"
    with pytest.raises(ValueError, match="changed after manifest"):
        evaluation.infer(hypothesis_model, path, torch.device("cpu"), record=record, partition="test")


def test_evaluation_records_official_and_all_return_scores(tmp_path, monkeypatch, hypothesis_model):
    x = torch.tensor([6., 7., 8., 9., 10., 5., 4., 12.])
    xyzi = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)), -1)
    sample = dict(index=0, xyzi=xyzi, voxel_xyzi=xyzi,
        targets=torch.tensor([1, 1, 1, 1, 1, 0, 0, -1], dtype=torch.int8),
        slots=torch.tensor([2, 3, 7, 8, 9, 11, 15, 18]))
    monkeypatch.setattr(evaluation, "PreparedScans", lambda manifest: [sample])
    monkeypatch.setattr(evaluation, "Scans", lambda manifest: [{key: sample[key].numpy() for key in ("slots", "targets")}])
    monkeypatch.setattr(evaluation, "memory_available", lambda: 10**12)
    manifest = dict(kind="val", sha256="unit-fixture", records=[
        dict(eligible=True, normal=2, anomaly=5, points=8, slots=19)])
    measured = evaluation.evaluate(hypothesis_model, manifest, torch.device("cpu"), workers=0,
        score_path=tmp_path / "val.npy", record_points=True)
    reference = PointOODMetricsCalculator()
    reference.update(xyzi[:, :3].numpy(), x.numpy(), np.array([2, 2, 2, 2, 2, 1, 1, 0]))
    assert measured["metrics"] == reference.compute_metrics()
    assert measured["scans"] == 1 and measured["points"] == 7
    np.testing.assert_array_equal(np.load(tmp_path / "val.npy"), x[:7].numpy())
    np.testing.assert_array_equal(np.load(tmp_path / "val_all.npy"), x.numpy())
    identities = np.load(tmp_path / "val_points.npy")
    np.testing.assert_array_equal(identities["slot"], sample["slots"].numpy())
    np.testing.assert_array_equal(identities["target"], sample["targets"].numpy())
    assert (tmp_path / "val_records.json").is_file()
    assert not (tmp_path / "val.json").exists()


def test_validate_command_keeps_metric_metadata_and_identity_records_separate(tmp_path, monkeypatch, hypothesis_model):
    import src.train as training
    x = torch.arange(3, 10, dtype=torch.float32)
    sample = dict(index=0, xyzi=torch.stack((x, x * 0, x * 0, x * 0), -1),
                  slots=torch.arange(7), targets=torch.tensor([0, 0, 1, 1, 1, 1, 1], dtype=torch.int8))
    manifest = dict(kind="val", sha256="test-population", records=[
        dict(eligible=True, normal=2, anomaly=5, points=7, slots=7)])
    saved = dict(version=NORMAL_VERSION, frozen=True,
                 config=dict(seed=206, variant="joint", architecture="test", score_version="test"),
                 normal201=dict(mean_iou_gt=.5, scans=1))
    monkeypatch.setattr(evaluation, "PreparedScans", lambda manifest: [sample])
    monkeypatch.setattr(evaluation, "Scans", lambda manifest: [{key: sample[key].numpy() for key in ("slots", "targets")}])
    monkeypatch.setattr(evaluation, "load_model", lambda *args: (hypothesis_model, saved))
    monkeypatch.setattr(evaluation, "load_manifest", lambda *args: manifest)
    monkeypatch.setattr(evaluation, "memory_available", lambda: 10**12)
    monkeypatch.setattr(training, "disk_check", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "validate", "--checkpoint", "model.pt", "--manifest", "manifest.json",
        "--output", str(output), "--device", "cpu", "--workers", "0", "--record-points"])
    evaluation.main()
    metadata = json.loads(output.read_text())
    records = json.loads((tmp_path / "evaluation_records.json").read_text())
    assert metadata["mode"] == "normal_hypothesis" and metadata["normal201"]["mean_iou_gt"] == .5
    assert metadata["seed"] == 206 and metadata["readout"] == "joint" and metadata["split"] == "val"
    assert records["identities"] == "evaluation_points.npy" and records["frames"][0]["metric_stop"] == 7
    recomputed = evaluation.recompute_metrics(tmp_path / "evaluation.npy", manifest)
    assert recomputed["independent_metrics"] == metadata["metrics"]
    identities = np.load(tmp_path / "evaluation_points.npy", mmap_mode="r+")
    identities["target"][[0, 2]] = identities["target"][[2, 0]]
    identities.flush()
    with pytest.raises(ValueError, match="original scan"):
        evaluation.recompute_metrics(tmp_path / "evaluation.npy", manifest)
    monkeypatch.setattr(sys, "argv", ["evaluate", "validate", "--checkpoint", "model.pt", "--manifest", "manifest.json",
        "--output", str(tmp_path / "bad.npy"), "--device", "cpu", "--workers", "0", "--record-points"])
    with pytest.raises(SystemExit):
        evaluation.main()
    assert not (tmp_path / "bad.npy").exists()


def test_instance_coverage_counts_scan_pairs_and_retains_sparse_objects():
    identity_type = [("official", "?"), ("target", "i1"), ("instance", "i4")]
    identities = np.array([(True, 1, 7), (True, 1, 8), (True, 1, 8),
                           (False, 1, 8), (True, 0, 0), (True, 1, 7),
                           (True, 1, 7), (True, 1, 0)], dtype=identity_type)
    frames = [dict(start=0, stop=5, metric_start=0, metric_stop=4),
              dict(start=5, stop=8, metric_start=4, metric_stop=7)]
    # The repeated instance 7 contributes separately in each scan. The masked
    # point and unassigned ID cannot change an instance's valid-return count.
    records = np.zeros(7, dtype=[("score", "f4")])
    records["score"] = [1, 1, 0, 0, 0, 0, 1]
    methods = {"joint": dict(operating_points=[dict(threshold=.5, requested_fpr=.01, actual_fpr=0)])}
    result = evaluation.instance_coverage(identities, frames, {"joint": records}, methods)
    assert result["instance_counts"] == [3, 0, 0, 0, 0, 0]
    assert result["methods"]["joint"][0]["detected"][0] == 2
    assert result["methods"]["joint"][0]["recall"][0] == pytest.approx(2 / 3)
    assert result["unassigned_anomaly_points"] == 1
    assert result["complete"] is False and "positive instance ID" in result["population"]


def test_normal_corrections_use_independent_reference_and_keep_absent_candidates():
    truth = np.array([0, 0, 1, 1, 1])
    predictions = dict(semantic=np.array([0, 1, 0, 1, 1]), joint=np.array([18, 0, 1, 1, 1]))
    result = evaluation.normal_decision_counts(truth, predictions)["joint"]
    assert result["corrected"] == 2 and result["introduced"] == 1
    assert result["confusion"][0, 18] == 1
    metrics = evaluation.normal_semantic_metrics(result["confusion"])
    assert metrics["point_accuracy"] == .8
    assert metrics["mean_iou_gt"] == .75
    assert result["confusion"].sum() == len(truth)


def test_predictive_summary_pools_queries_and_rejects_dropped_intervals():
    stats = dict(prediction_count=[1, 3], finite_interval_count=[1, 3],
                 nll_sum=[-1, -9], coverage90_count=[1, 2],
                 width90_m_sum=[2, 12], abs_median_error_m_sum=[.1, 1.5])
    result = evaluation.predictive_summary(stats)
    assert result["queries"] == 4 and result["nll"] == -2.5
    assert result["coverage90"] == .75 and result["width90_m"] == 3.5
    assert result["median_mae_m"] == .4
    assert evaluation.predictive_summary({})["nll"] is None
    stats["finite_interval_count"] = [1, 2]
    with pytest.raises(ValueError, match="cannot be omitted"):
        evaluation.predictive_summary(stats)


def test_seed_aggregation_uses_sample_deviation_and_one_scientific_condition():
    results = [dict(seed=seed, variant="joint", readout="joint", split="val", manifest_sha256="same",
                    architecture="test", score_version="test", metrics=dict(AP=value, AUROC=90., FPR95=2.))
               for seed, value in zip((206, 307, 409), (10., 12., 14.), strict=True)]
    measured = evaluation.aggregate_results(results)
    assert measured["metrics"]["AP"]["mean"] == 12
    assert measured["metrics"]["AP"]["std"] == 2
    for result, value in zip(results, (.6, .62, .64), strict=True):
        result["normal201"] = dict(mean_iou_gt=value)
    assert evaluation.aggregate_results(results)["metrics"]["normal_mIoU"]["mean"] == 62
    results[2].pop("normal201")
    with pytest.raises(ValueError, match="all three"):
        evaluation.aggregate_results(results)
    results[2]["normal201"] = dict(mean_iou_gt=.64)
    results[2]["readout"] = "appearance"
    with pytest.raises(ValueError, match="readout"):
        evaluation.aggregate_results(results)
    results[2]["seed"] = 206
    with pytest.raises(ValueError, match="each seed"):
        evaluation.aggregate_results(results)


def test_benchmark_default_warmup_and_full_ordered_pass(monkeypatch, hypothesis_model):
    calls = []
    hypothesis_model.register_buffer("reference", torch.ones(2, dtype=torch.float64))

    def fake_infer(model, scan, device, **kwargs):
        calls.append(scan)
        return None, dict(seconds=.01 if scan == "a" else .03, read_seconds=.002)

    monkeypatch.setattr(evaluation, "infer", fake_infer)
    result = evaluation.benchmark(hypothesis_model, ["a", "b"], torch.device("cpu"))
    assert calls == ["a", "b"] * 26
    assert result["warmup_scans"] == 50 and result["measured_scans"] == 2
    assert result["precision"] == "FP32" and result["mean_ms"] == 20
    assert hypothesis_model.reference.dtype == torch.float64


def test_benchmark_manifest_times_scans_without_anomalies(tmp_path, monkeypatch, hypothesis_model):
    manifest = dict(kind="val", records=[dict(scan="first.bin", eligible=False),
                                         dict(scan="second.bin", eligible=True)])
    monkeypatch.setattr(evaluation, "load_model", lambda *args: (hypothesis_model, {}))
    monkeypatch.setattr(evaluation, "load_manifest", lambda *args: manifest)
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setattr(evaluation, "benchmark", lambda model, scans, device, **kwargs:
                        dict(scans=[str(path) for path in scans]))
    output = tmp_path / "runtime.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "benchmark", "--checkpoint", "model.pt", "--manifest", "val.json",
                                      "--output", str(output), "--device", "cpu"])
    evaluation.main()
    assert json.loads(output.read_text())["scans"] == ["first.bin", "second.bin"]
