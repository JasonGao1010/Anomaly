"""Feature-evidence checkpoints preserve point identities and official metric scope."""

import numpy as np
import pytest
import torch
from torch import nn

import src.evaluate as evaluation
import src.model as models
from src.normal import EVIDENCE_VERSION, FeatureEvidence
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


@pytest.fixture
def evidence_model(monkeypatch):
    class Perception(nn.Module):
        def __init__(self, pretrained=None):
            super().__init__()
            self.register_buffer("anchor", torch.tensor(0.))

        def encode(self, sample):
            features = torch.zeros(len(sample["xyzi"]), 252)
            features[:, 0] = sample["xyzi"][:, 0]
            logits = torch.zeros(len(features), 16)
            logits[:, 0] = 1
            return dict(features=features, logits=logits)

    monkeypatch.setattr(models, "FrozenPerception", Perception)

    def create(residual=False):
        scorer = FeatureEvidence(residual=residual)
        with torch.no_grad():
            scorer.semantic.weight.zero_()
            scorer.semantic.bias.zero_()
            scorer.semantic.bias[18] = 1
            scorer.anomaly.weight.zero_()
            scorer.anomaly.weight[0, 0] = 1
            scorer.anomaly.bias.zero_()
        return models.FrozenSupport(scorer=scorer).eval()

    return create


@pytest.mark.parametrize("residual", [False, True])
def test_load_feature_evidence_preserves_capacity_and_rejects_unselected_weights(tmp_path, evidence_model, residual):
    model = evidence_model(residual)
    saved = dict(version=EVIDENCE_VERSION, mode="frozen_support", frozen=True, selected=True, complete=True,
        config=dict(version=EVIDENCE_VERSION, residual=residual, initial_sha256=models.WEIGHTS_SHA256),
        model=model.state_dict())
    path = tmp_path / "model.pt"
    torch.save(saved, path)
    loaded, metadata = evaluation.load_model(path, torch.device("cpu"))
    assert isinstance(loaded.scorer, FeatureEvidence)
    assert (loaded.scorer.residual is not None) == residual
    assert metadata["version"] == EVIDENCE_VERSION and not loaded.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert not torch.backends.cuda.matmul.allow_tf32
    saved["selected"] = False
    torch.save(saved, path)
    with pytest.raises(ValueError, match="completed model selection"):
        evaluation.load_model(path, torch.device("cpu"))
    saved["selected"] = True
    saved["config"]["residual"] = None
    torch.save(saved, path)
    with pytest.raises(ValueError, match="residual capacity"):
        evaluation.load_model(path, torch.device("cpu"))


def test_feature_evidence_infer_exports_learned_stu19_in_original_slots(tmp_path, evidence_model):
    # Two returns cannot support an eight-neighbor density estimate; it is unused.
    points = np.array([[10., 0, 0, .2], [0., 0, 0, 0], [5., 0, 0, .4]], np.float32)
    path = tmp_path / "206" / "velodyne" / "000000.bin"
    path.parent.mkdir(parents=True)
    points.tofile(path)
    model = evidence_model()
    result, timing = evaluation.infer(model, path, torch.device("cpu"), return_semantics=True)
    np.testing.assert_array_equal(result["score"], [10., 0., 5.])
    np.testing.assert_array_equal(result["semantic"], [18, -1, 18])
    assert timing["real_points"] == 2 and timing["slots"] == 3
    score, _ = evaluation.infer(model, path, torch.device("cpu"))
    np.testing.assert_array_equal(score, result["score"])


def test_feature_evidence_evaluation_records_official_and_all_return_scores(tmp_path, monkeypatch, evidence_model):
    x = torch.tensor([6., 7., 8., 9., 10., 5., 4., 12.])
    xyzi = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)), -1)
    sample = dict(index=0, xyzi=xyzi, voxel_xyzi=xyzi, conditions=torch.zeros(8, 2),
        targets=torch.tensor([1, 1, 1, 1, 1, 0, 0, -1], dtype=torch.int8),
        slots=torch.tensor([2, 3, 7, 8, 9, 11, 15, 18]))

    def prepared(manifest, **options):
        assert options["frozen"] and options["range_only"]
        return [sample]

    monkeypatch.setattr(evaluation, "PreparedScans", prepared)
    monkeypatch.setattr(evaluation, "memory_available", lambda: 10**12)
    manifest = dict(kind="val", sha256="unit-fixture", records=[
        dict(eligible=True, normal=2, anomaly=5, points=8, slots=19)])
    measured = evaluation.evaluate(evidence_model(), manifest, torch.device("cpu"), workers=0,
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
