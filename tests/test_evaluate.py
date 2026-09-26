"""SERVE inference and official evaluation preserve original point identities."""

import numpy as np
import pytest
import torch
from torch import nn

import src.evaluate as evaluation
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


@pytest.fixture
def hypothesis_model():
    class Model(nn.Module):
        mode = "normal_hypothesis"

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
    score, _ = evaluation.infer(hypothesis_model, path, torch.device("cpu"))
    np.testing.assert_array_equal(score, result["score"])


def test_evaluation_records_official_and_all_return_scores(tmp_path, monkeypatch, hypothesis_model):
    x = torch.tensor([6., 7., 8., 9., 10., 5., 4., 12.])
    xyzi = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)), -1)
    sample = dict(index=0, xyzi=xyzi, voxel_xyzi=xyzi,
        targets=torch.tensor([1, 1, 1, 1, 1, 0, 0, -1], dtype=torch.int8),
        slots=torch.tensor([2, 3, 7, 8, 9, 11, 15, 18]))
    monkeypatch.setattr(evaluation, "PreparedScans", lambda manifest: [sample])
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
