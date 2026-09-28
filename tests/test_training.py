"""SERVE normal training, point identity and official evaluation regressions."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch import nn

from src.data import Frame, point_targets, supervision, unified_labels
from src.model import scatter_scores, voxelize
from vendor.litept.pointrope import PointROPE
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


def test_evaluation_record_preserves_official_population_and_ignored_returns(tmp_path, monkeypatch):
    import src.evaluate as evaluation
    class Score(nn.Module):
        def forward(self, sample):
            return sample["prediction"]
    sample = dict(index=0, xyzi=torch.arange(36).reshape(9, 4).float() + 1,
                  slots=torch.tensor([0, 2, 3, 5, 8, 9, 12, 15, 18]),
                  targets=torch.tensor([1, 1, 1, 1, 1, 0, 0, -1, -1]),
                  prediction=torch.tensor([2., 1., 3., 1., 4., .5, 1.5, 100., -100.]))
    monkeypatch.setattr(evaluation, "PreparedScans", lambda manifest, **kwargs: [sample])
    monkeypatch.setattr(evaluation, "Scans", lambda manifest: [{key: sample[key].numpy() for key in ("slots", "targets")}])
    manifest = dict(kind="val", sha256="fixture", records=[dict(eligible=True, normal=2, anomaly=5, points=9, slots=19)])
    expected = evaluation.evaluate(Score(), manifest, torch.device("cpu"), 0)
    actual = evaluation.evaluate(Score(), manifest, torch.device("cpu"), 0, tmp_path / "val1.npy", record_points=True)
    assert actual["metrics"] == expected["metrics"]
    np.testing.assert_array_equal(np.load(tmp_path / "val1.npy"), sample["prediction"][:7].numpy())
    np.testing.assert_array_equal(np.load(tmp_path / "val1_all.npy"), sample["prediction"].numpy())
    identities = np.load(tmp_path / "val1_points.npy")
    np.testing.assert_array_equal(identities["slot"], sample["slots"].numpy())
    np.testing.assert_array_equal(identities["target"], sample["targets"].numpy())
    independent = evaluation.recompute_metrics(tmp_path / "val1.npy", manifest)
    for name, value in expected["metrics"].items():
        assert independent["independent_metrics"][name] == pytest.approx(value, abs=1e-12, rel=0)
    assert independent["points"] == 7 and independent["normal_points"] == 2
    changed = np.load(tmp_path / "val1.npy", mmap_mode="r+")
    changed[0] += 1
    changed.flush()
    with pytest.raises(ValueError, match="original-point scores"):
        evaluation.recompute_metrics(tmp_path / "val1.npy", manifest)


def test_independent_score_ranks_match_official_ties_and_roc_vertex_removal():
    from src.evaluate import rank_metrics
    from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator
    cases = [
        (np.array([1., 1., 0., -1.], np.float32), np.array([1., 1., 0.], np.float32)),
        # Every intermediate mixed-class ROC vertex is collinear; simply using
        # the first raw threshold above 95% would report a different FPR95.
        (np.arange(100, dtype=np.float32), np.arange(100, dtype=np.float32)),
        # Exactly 19/20 recall must be passed, not accepted.
        (np.array([3., 1.], np.float32), np.r_[np.arange(22., 3., -1), 2.].astype(np.float32)),
        (np.array([2., 2.], np.float32), np.array([2., 2.], np.float32)),
    ]
    rng = np.random.default_rng(873)
    for size in (20, 100, 1000):
        for _ in range(12):
            cases.append((rng.integers(-size, size, size=size).astype(np.float32),
                          rng.integers(-size, size, size=size // 2).astype(np.float32)))
    for normal, anomaly in cases:
        official = PointOODMetricsCalculator()
        official.all_scores = [np.r_[normal, anomaly]]
        official.all_labels = [np.r_[np.zeros(len(normal), np.int8), np.ones(len(anomaly), np.int8)]]
        expected = official.compute_metrics()
        measured = rank_metrics(normal.copy(), anomaly)
        for name, value in expected.items():
            assert measured[name] == pytest.approx(value, abs=1e-12, rel=0), (name, normal, anomaly)
    assert rank_metrics(*[scores.copy() for scores in cases[1]])["FPR95"] == 100.
    assert rank_metrics(*[scores.copy() for scores in cases[2]])["FPR95"] == 50.


def test_official_mask_raw_mapping_and_strict_fpr95():
    xyz = np.array([[2.5, 0, 0], [50., 0, 0], [2.499, 0, 0], [50.001, 0, 0],
                    [0, 0, 0], [-3., 0, 0], [0, -4., 0], [0, 0, 5], [4., 0, 0],
                    [3., 0, 0], [3., 0, 0]], np.float32)
    raw = np.array([2, 2, 2, 2, 2, 2, 2, 2, 1, 40, 0], np.uint32)
    frame = Frame(0, np.column_stack((xyz, np.ones(len(xyz), np.float32))), np.eye(4), raw)
    np.testing.assert_array_equal(unified_labels(raw), [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0])
    scores = np.linspace(-1, 1, len(raw)).astype(np.float32)
    metric = PointOODMetricsCalculator()
    metric.update(xyz, scores, unified_labels(raw))
    targets = point_targets(frame)
    np.testing.assert_array_equal(metric.all_labels[0], targets[targets >= 0])
    np.testing.assert_array_equal(metric.all_scores[0], scores[targets >= 0])
    assert supervision(frame).anomaly_count == 5
    # TPR=19/20 must be passed, not accepted as the FPR95 operating point.
    labels = np.array([1] * 19 + [0, 1, 0])
    _, fpr, threshold = metric._calculate_auroc(np.arange(22, 0, -1), labels)
    assert fpr == .5 and threshold == 2
    a = metric.compute_metrics()
    metric.all_labels = [metric.all_labels[0].astype(np.int8)]
    assert metric.compute_metrics() == a


def test_all_return_voxels_origin_and_slot_scattering():
    points = np.array([[-.01, 0, 3, .1], [-.02, 0, 3, 1.5], [3., 0, 0, .4],
                       [70., 0, 0, .8], [3., 0, 0, .2]], np.float32)
    batch = voxelize(points)
    np.testing.assert_array_equal(batch["xyzi"], points)
    inv = batch["inverse"].numpy()
    assert inv[0] == inv[1] and inv[2] == inv[4]
    np.testing.assert_allclose(batch["voxel_xyzi"][inv[0]], points[:2].mean(0), atol=1e-7)
    raw_grid = np.floor(points[:, :3].astype(float) / .05).astype(int)
    translated = batch["grid"].numpy()[inv]
    for level in range(5):
        shift = translated // (2**level) - raw_grid // (2**level)
        assert np.all(shift == shift[0])
    np.testing.assert_allclose(batch["offset"], (points[:, :3] - (raw_grid + .5) * .05) / .05, atol=1e-6)
    output = scatter_scores(torch.arange(5.).requires_grad_(), torch.tensor([0, 2, 4, 6, 7]), 9)
    assert output[[1, 3, 5, 8]].tolist() == [0., 0., 0., 0.]
    assert output[[0, 2, 4, 6, 7]].tolist() == list(map(float, range(5)))
    output.sum().backward()
    with pytest.raises(ValueError, match="empty ray"):
        voxelize(np.zeros((1, 4), np.float32))
    # FP32 0.7 lies below the exact cell boundary; rounded division hides this.
    edge=np.float32(.7)
    boundary=np.array([[edge,0,3,.1],[np.nextafter(edge,np.float32(np.inf)),0,3,.1],
        [-edge,0,3,.1],[np.nextafter(-edge,np.float32(-np.inf)),0,3,.1]],np.float32)
    b=voxelize(boundary)
    torch.testing.assert_close(b["grid"][b["inverse"],0]-16,torch.tensor([13,14,-14,-15]),rtol=0,atol=0)


def test_spatial_codes_roundtrip_without_axis_or_bit_loss():
    from vendor.litept.serialization.default import encode, decode
    torch.manual_seed(73)
    for depth in (1,8,9,16):
        bound=2**depth
        grid=torch.cat((torch.randint(0,bound,(128,3)),torch.tensor([[0,0,0],[bound-1,bound-1,bound-1]])))
        batch=torch.arange(len(grid))%3
        for order in ("z","z-trans","hilbert","hilbert-trans"):
            code=encode(grid,batch,depth=depth,order=order)
            restored,groups=decode(code,depth=depth,order=order.removesuffix("-trans"))
            if order.endswith("-trans"):restored=restored[:,[1,0,2]]
            torch.testing.assert_close(restored,grid,rtol=0,atol=0)
            torch.testing.assert_close(groups,batch,rtol=0,atol=0)
            assert len(torch.unique(code))==len(torch.unique(torch.cat((batch[:,None],grid),1),dim=0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sparse_embedding_uses_centered_xyz_neighbors_under_autocast():
    from vendor.litept.model import Embedding,Point
    layer=Embedding(1,1).cuda()
    grid=torch.cartesian_prod(*(torch.arange(10,14,device="cuda") for _ in range(3)))
    values=torch.arange(1,len(grid)+1,device="cuda",dtype=torch.float32)[:,None]
    for displacement in (0,1):
        with torch.no_grad():
            layer.stem.conv.weight.zero_()
            layer.stem.conv.weight[0,2+displacement,2,2,0]=1.
        point=Point(coord=grid.float()*.05,grid_coord=grid,feat=values.clone(),
                    offset=torch.tensor([len(grid)],device="cuda"))
        point.sparsify()
        with torch.autocast("cuda",dtype=torch.bfloat16):result=layer(point)
        expected=values.clone() if displacement==0 else torch.where(grid[:,0,None]<13,values+16,0.)
        torch.testing.assert_close(result.feat,expected,rtol=0,atol=0)
        torch.testing.assert_close(result.sparse_conv_feat.indices[:,1:].long(),grid,rtol=0,atol=0)


def test_pointrope_matches_direct_axis_rotations_and_bounded_cache():
    torch.manual_seed(3)
    tokens = torch.randn(1, 14, 23, 18, requires_grad=True)
    positions = torch.randint(0, 2000, (1, 23, 3))
    module = PointROPE()
    actual = module(tokens, positions)
    axes = []
    for axis, part in enumerate(tokens.chunk(3, -1)):
        frequency = 1. / 100 ** (torch.arange(0, 6, 2).float() / 6)
        angles = positions[:, :, axis, None].float() * frequency
        angles = torch.cat((angles, angles), -1)[:, None]
        left, right = part.chunk(2, -1)
        axes.append(part * angles.cos() + torch.cat((-right, left), -1) * angles.sin())
    torch.testing.assert_close(actual, torch.cat(axes, -1), atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    assert torch.isfinite(tokens.grad).all()
    module(tokens, positions // 2)
    assert len(module.cache) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_attention_preserves_fp32_rotary_phases_under_bf16_autocast():
    from vendor.litept.model import Point, PointROPEAttention
    torch.manual_seed(73)
    attention = PointROPEAttention(36, 2, 16, 100.).cuda().eval()
    point = Point(feat=torch.randn(32, 36, device="cuda"),
                  grid_coord=torch.randint(700, 2300, (32, 3), device="cuda"),
                  offset=torch.tensor([32], device="cuda"))
    point.serialization(order=["z"])
    calls = []

    def compare_phases(module, inputs, result):
        tokens, positions = inputs
        axes = []
        for axis, part in enumerate(tokens.chunk(3, -1)):
            frequency = 1. / 100 ** (torch.arange(0, 6, 2, device="cuda").float() / 6)
            angles = positions[:, :, axis, None].float() * frequency
            angles = torch.cat((angles, angles), -1)[:, None]
            left, right = part.chunk(2, -1)
            axes.append(part * angles.cos() + torch.cat((-right, left), -1) * angles.sin())
        torch.testing.assert_close(result, torch.cat(axes, -1), atol=1e-6, rtol=1e-5)
        calls.append(result.shape)

    hook = attention.rope.register_forward_hook(compare_phases)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = attention(point)
            loss = result.feat.float().square().mean()
        loss.backward()
    finally:
        hook.remove()
    assert len(calls) == 2 and torch.isfinite(attention.qkv.weight.grad).all()


def test_normal_selection_prioritizes_fixed_gt_classes_before_likelihood():
    from src.train import normal_selection
    assert normal_selection(dict(mean_iou_gt=.8, objective=5.)) > normal_selection(dict(mean_iou_gt=.7, objective=-5.))
    assert normal_selection(dict(mean_iou_gt=.8, objective=4.)) > normal_selection(dict(mean_iou_gt=.8, objective=5.))
    for quality, objective in ((None, 0.), (.5, float("nan")), (float("inf"), 0.)):
        with pytest.raises(ValueError, match="ground-truth-class"):
            normal_selection(dict(mean_iou_gt=quality, objective=objective))


def test_normal_replay_preserves_budget_and_covers_refined_training_classes(tmp_path):
    from src.train import normal_order, normal_replay_pools
    labels = tmp_path / "training.label"
    np.array([0, 10, 50], dtype=np.uint32).tofile(labels)
    source = [{"refinement_classes": []} for _ in range(20)]
    for index, category in ((1, 6), (2, 6), (10, 9), (11, 9), (12, 12)):
        source[index]["refinement_classes"] = [category]
    target = [dict(source="normal_stu", scene="206", label=str(labels))]
    pools, present = normal_replay_pools(source, target)
    assert pools == {6: [1, 2], 9: [10, 11]} and present == [0, 12]
    with pytest.raises(ValueError, match="206 training"):
        normal_replay_pools(source, [dict(target[0], scene="201")])
    order = normal_order(20, 9, 8, 206, "target", pools)
    assert order == normal_order(20, 9, 8, 206, "target", pools)
    assert len(order) == 8 * (9 + 9 // 4)
    for epoch in range(8):
        assert sorted(index for turn, index, _ in order if turn == epoch and index < 9) == list(range(9))
    assert [visit for _, _, visit in order] == list(range(len(order)))
    replay = [index - 9 for _, index, _ in order if index >= 9]
    assert set(replay[::4]) == {1, 2} and set(replay[2::4]) == {10, 11}
    assert len(set(replay[1::2])) == len(replay[1::2])
    # Source pretraining does not depend on replay priorities or model variants.
    assert normal_order(20, 9, 1, 206, "source", pools) == normal_order(20, 9, 1, 206, "source")


def test_normal_reference_rejects_truncated_or_changed_comparison_budget(tmp_path):
    from src.train import normal_reference
    budget = {stage: dict(visits=visits, updates=visits // 2, order_identity=stage, schedule="fixed")
              for stage, visits in (("source", 20), ("target", 22))}
    config = dict(variant="semantic", architecture="method", budget=budget, seed=206,
                  source_mapping={2: (5,), 14: (1, 6)},
                  source_validation_indices=[0, 2], target_validation_indices=[1, 3])
    stages = {stage: dict(budget_complete=True, trained_frames=row["visits"], trained_updates=row["updates"],
                          planned_frames=row["visits"], planned_updates=row["updates"], selected_update=1)
              for stage, row in budget.items()}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "stages.json").write_text(json.dumps(stages))
    # Integer mapping keys survive JSON, and independent variants may choose different best updates.
    assert normal_reference(tmp_path, dict(config, variant="joint"))[0]["variant"] == "semantic"
    with pytest.raises(ValueError, match="initialization or training budget"):
        normal_reference(tmp_path, dict(config, seed=207))
    for field in ("source_validation_indices", "target_validation_indices"):
        with pytest.raises(ValueError, match=field):
            normal_reference(tmp_path, dict(config, **{field: [0, 1]}))
    for field, value in (("budget_complete", False), ("trained_frames", 20),
                         ("trained_updates", 10), ("planned_updates", 12)):
        changed = deepcopy(stages)
        changed["target"][field] = value
        (tmp_path / "stages.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="complete the common nominal"):
            normal_reference(tmp_path, config)


def test_normal_development_keeps_all_point_semantics_and_additive_diagnostics(monkeypatch):
    import src.data as data
    from src.train import normal_development

    class Dataset:
        def __init__(self, records, queries, seed=206):
            self.records = records

        def __getitem__(self, index):
            allowed = torch.zeros((4, 19), dtype=torch.bool)
            allowed[0, 0], allowed[1, 1], allowed[2, 1:3] = True, True, True
            return dict(allowed=allowed, xyzi=torch.tensor([[3., 0, 0, 0], [15, 0, 0, 0],
                                                            [35, 0, 0, 0], [60, 0, 0, 0]]),
                        observation=dict(group=torch.tensor([0, 1, 1, 2])))

    class Model:
        def eval(self):
            return self

        def loss(self, sample):
            self.development_prediction = torch.tensor([0, 0, 2, 1])
            self.development_semantic_prediction = torch.tensor([1, 1, 0, 1])
            self.development_diagnostics = dict(query_count=[1, 1], coarse_query_count=1)
            return torch.tensor(2.), dict(classification=torch.tensor(1.))

    monkeypatch.setattr(data, "NormalScans", Dataset)
    measured = normal_development(Model(), [{}, {}], [0, 1], torch.device("cpu"), 0, queries=1)
    assert measured["ground_truth_points"][:3] == [2, 2, 0]
    assert measured["mean_iou_gt"] == .25
    assert measured["joint_comparison"] == dict(set_points=6, corrected_points=4, worsened_points=2)
    assert measured["per_class_comparison"]["corrected_points"][0] == 2
    assert measured["observation_diagnostics"]["query_count"] == [2., 2.]
    for name in ("distance_metres", "returns_per_angular_cell"):
        assert sum(row[0] for row in measured["strata"][name]["counts"]) == 6


def test_normal_calibration_reuses_development_with_identical_samples_quantiles_and_rng(tmp_path, monkeypatch):
    import src.data as data
    import src.train as training
    from src.model import CALIBRATION_PROBABILITIES
    from src.train import normal_calibration, normal_development

    loads = []

    class Dataset:
        def __init__(self, records, queries, seed=206):
            self.records, self.queries = records, queries

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            loads.append((index, self.queries))
            count = 2800 + index * 13
            allowed = torch.zeros(count, 19, dtype=torch.bool)
            allowed[:, 0], allowed[::9, 0] = True, False
            xyzi = torch.zeros(count, 4)
            xyzi[:, 0] = torch.linspace(3., 45., count)
            return dict(index=index, xyzi=xyzi, allowed=allowed,
                        observation=dict(group=torch.arange(count) // 5))

    class Model(nn.Module):
        variant = "joint"

        def __init__(self):
            super().__init__()
            self.register_buffer("calibration", torch.zeros(len(CALIBRATION_PROBABILITIES)))
            self.register_buffer("calibrated", torch.tensor(False))
            self.reference_forwards = 0
            self.development_forwards = 0

        @staticmethod
        def raw_score(sample):
            return torch.arange(len(sample["xyzi"]), dtype=torch.float32).square() / 123 + sample["index"] * 17

        def loss(self, sample):
            self.development_forwards += 1
            self.development_raw_score = self.raw_score(sample)
            self.development_prediction = torch.zeros(len(sample["xyzi"]), dtype=torch.long)
            self.development_semantic_prediction = self.development_prediction
            self.development_diagnostics = dict(query_count=np.array([1.]))
            return self.development_raw_score.mean(), dict(classification=torch.tensor(1.))

        def components(self, sample, indices):
            self.reference_forwards += 1
            return dict(raw_score=self.raw_score(sample)[indices])

    monkeypatch.setattr(data, "NormalScans", Dataset)
    records = [dict(source="normal_stu", scene="201", frame=100 + index) for index in range(3)]
    device = torch.device("cpu")
    reused, fresh = Model(), Model()
    reference = dict(seed=206)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(172)
        normal_development(reused, records, [0, 1, 2], device, 0, queries=8192, calibration=reference)
        first = normal_calibration(reused, records, device, 0, tmp_path / "reused", reference=reference)
        reused_rng = torch.get_rng_state()
        assert loads == [(0, 8192), (1, 8192), (2, 8192)]
        assert reused.reference_forwards == 0
        loads.clear()
        torch.manual_seed(172)
        normal_development(fresh, records, [0, 1, 2], device, 0, queries=8192)
        second = normal_calibration(fresh, records, device, 0, tmp_path / "fresh")
        assert torch.equal(torch.get_rng_state(), reused_rng)
    assert loads == [(0, 8192), (1, 8192), (2, 8192), (0, 1), (1, 1), (2, 1)]
    assert fresh.reference_forwards == 3
    torch.testing.assert_close(reused.calibration, fresh.calibration, atol=0, rtol=0)
    assert reused.calibrated and fresh.calibrated
    for key in first:
        if key not in ("seconds", "inference"):
            assert first[key] == second[key], key
    independent = []
    for frame in range(3):
        sample = Dataset(records, 1)[frame]
        valid = np.flatnonzero(sample["allowed"].numpy().any(1))
        chosen = np.sort(np.random.default_rng(np.random.SeedSequence([206, frame])).choice(valid, 2048, replace=False))
        values = Model.raw_score(sample).numpy()[chosen]
        np.testing.assert_array_equal(reference["scores"][frame], values)
        independent.append(values)
    np.testing.assert_array_equal(reused.calibration.numpy(),
        np.quantile(np.concatenate(independent), CALIBRATION_PROBABILITIES).astype(np.float32))
    with pytest.raises(ValueError, match="model, seed and full normal population"):
        normal_calibration(reused, records, device, 0, tmp_path / "wrong", seed=207, reference=reference)
    clock = [10.]
    monkeypatch.setattr(training.time, "time", lambda: clock[0])
    before = len(loads)
    with pytest.raises(TimeoutError, match="before normal calibration"):
        normal_calibration(fresh, records, device, 0, tmp_path / "late", deadline=10.)
    assert len(loads) == before
    assert not (tmp_path / "late" / "calibration.json").exists()

    clock[0] = 0.
    original_loss = fresh.loss

    def finish_at_deadline(sample):
        result = original_loss(sample)
        clock[0] = 10.
        return result

    monkeypatch.setattr(fresh, "loss", finish_at_deadline)
    before = fresh.development_forwards
    with pytest.raises(TimeoutError, match="before complete normal development"):
        normal_development(fresh, records, [0, 1, 2], device, 0, deadline=10.)
    assert fresh.development_forwards == before + 1

    clock[0] = 0.
    original_components = fresh.components

    def reference_at_deadline(sample, indices):
        result = original_components(sample, indices)
        clock[0] = 10.
        return result

    monkeypatch.setattr(fresh, "components", reference_at_deadline)
    before = fresh.reference_forwards
    with pytest.raises(TimeoutError, match="before complete normal calibration"):
        normal_calibration(fresh, records, device, 0, tmp_path / "partial", deadline=10.)
    assert fresh.reference_forwards == before + 1
    assert not (tmp_path / "partial" / "calibration.json").exists()


def test_training_accumulates_frames_and_resumes_exact_rng_and_schedule(tmp_path, monkeypatch):
    import src.train as training
    import src.data as data

    class Dataset:
        def __init__(self, records, **kwargs):
            self.records = records

        def __getitem__(self, key):
            epoch, index, visit = key
            return dict(x=torch.tensor([[float(index + 1), float(epoch + 1)]]))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.dropout = nn.Dropout(.2)

        def loss(self, sample):
            value = self.linear(self.dropout(sample['x'])).square().mean()
            return value, dict(classification=value.detach())

    def optimizer(model, stage):
        return torch.optim.SGD([dict(params=list(model.parameters()), lr=.01, peak_lr=.01)])

    def development(model, *args, **kwargs):
        model.eval()
        # Evaluation may consume RNG; it must not change subsequent training.
        torch.rand(9)
        return dict(mean_iou_gt=.5, objective=float(model.linear.weight.detach().square().sum()))

    monkeypatch.setattr(data, 'NormalScans', Dataset)
    monkeypatch.setattr(training, 'normal_optimizer', optimizer)
    monkeypatch.setattr(training, 'normal_development', development)
    monkeypatch.setattr(training, 'disk_check', lambda *args: {})
    config = dict(batch=2, queries=19, seed=206, eval_every=2, target_eval_every=2)
    order = [(0, i, i) for i in range(6)] + [(1, 0, 6)]
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    full.mkdir()
    interrupted.mkdir()
    torch.manual_seed(99)
    reference = Model()
    candidate = deepcopy(reference)
    training.seed_all(206)
    full_stages = {}
    training.train_stage(reference, list(range(6)), [{}], [0], order, config, 'source',
                         full, torch.device('cpu'), 0, stages=full_stages, checkpoint_every=2)
    expected_rng = training.rng_state(torch.device('cpu'))
    save = training.atomic_save

    def stop_after_checkpoint(path, value):
        save(path, value)
        if path.name == 'last.pt' and value['trained_updates'] == 2:
            raise InterruptedError('test interruption after a complete saved update')

    monkeypatch.setattr(training, 'atomic_save', stop_after_checkpoint)
    training.seed_all(206)
    stages = {}
    with pytest.raises(InterruptedError):
        training.train_stage(candidate, list(range(6)), [{}], [0], order, config, 'source',
                             interrupted, torch.device('cpu'), 0, stages=stages, checkpoint_every=2)
    monkeypatch.setattr(training, 'atomic_save', save)
    checkpoint = torch.load(interrupted / 'last.pt', weights_only=False)
    training.train_stage(candidate, list(range(6)), [{}], [0], order, config, 'source',
                         interrupted, torch.device('cpu'), 0, stages=stages,
                         resume=checkpoint, checkpoint_every=2)
    assert stages == full_stages
    assert stages['source']['trained_frames'] == 7
    assert stages['source']['trained_updates'] == 4
    assert stages['source']['budget_complete']
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
    assert torch.equal(expected_rng['torch'], torch.get_rng_state())
    assert (full / 'training.jsonl').read_text() == (interrupted / 'training.jsonl').read_text()


def test_source_development_uses_distinct_scene_centers_and_fixed_update_rates():
    from src.train import development_indices, learning_rate_factor, normal_order
    rows = [dict(scene=f'{scene:03d}') for scene in range(150) for _ in range(5)]
    chosen = development_indices(rows, 50, by_scene=True)
    assert len(chosen) == len({rows[index]['scene'] for index in chosen}) == 50
    assert all(index % 5 == 2 for index in chosen)
    assert development_indices([{}] * 682, 68) == np.linspace(0, 681, 68, dtype=int).tolist()
    assert len(normal_order(28130, 449, 2, 206, 'source')) // 2 == 28130
    assert len(normal_order(28130, 449, 8, 206, 'target')) // 2 == 2244
    assert learning_rate_factor(50, 1000) == 1
    assert learning_rate_factor(1000, 1000) == .05


def test_optimizer_keeps_shared_prototypes_with_appearance_and_slows_measurement():
    from src.train import normal_optimizer, OPTIMIZATION

    model = nn.Module()
    model.backbone = nn.Linear(2, 2)
    model.embedding = nn.Linear(2, 2)
    model.hypotheses = nn.Module()
    model.hypotheses.queries = nn.Parameter(torch.randn(3, 2))
    model.hypotheses.surface = nn.Linear(2, 3)
    for stage in ('source', 'target'):
        optimizer = normal_optimizer(model, stage)
        groups = {group['name']: group for group in optimizer.param_groups}
        assigned = {id(p): name for name, group in groups.items() for p in group['params']}
        assert len(assigned) == sum(len(group['params']) for group in groups.values())
        assert set(assigned) == {id(p) for p in model.parameters()}
        assert assigned[id(model.hypotheses.queries)] == 'appearance'
        assert assigned[id(model.hypotheses.surface.weight)] == 'measurement'
        assert assigned[id(model.backbone.weight)] == 'backbone'
        for name, group in groups.items():
            assert group['lr'] == OPTIMIZATION[stage][name] == group['peak_lr']
        assert groups['backbone']['lr'] < groups['measurement']['lr'] < groups['appearance']['lr']
