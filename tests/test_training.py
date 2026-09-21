"""Scientific-semantic regressions for the fixed V4 training implementation."""

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.data import (Frame, Scans, MANIFEST_VERSION, VERSION, load_manifest, point_targets, read_delta,
                      STUSequence, restore_delta, supervision, unified_labels)
from src.evaluate import better, summarize
from src.model import (CHANNELS, Conditional, Interaction, Segmentor, balanced_loss, prepare_scan,
                       ranking_loss, rank_sample, RecallThreshold, scatter_scores, voxelize)
from src.train import (EPOCHS, configuration, effective_batches, epoch_order, lr_factor, optimizer_for, pilot_order,
                       forward_loss, ranking_weight, rng_state, restore_rng, seed_all)


def test_metric_proxies_match_complete_small_ranking_and_ignore_unlabelled():
    scores = torch.tensor([-.2, 2., .7, -1., 1.2, 999.], requires_grad=True)
    labels = torch.tensor([0, 1, 1, 0, 0, -1])
    loss, detail = ranking_loss(scores, labels, 7)
    p, n = scores[labels == 1], scores[labels == 0]
    precisions = []
    for i, anchor in enumerate(p):
        ahead = 1 + sum(torch.sigmoid(other - anchor) for j, other in enumerate(p) if i != j)
        precisions.append(ahead / (ahead + sum(torch.sigmoid(other - anchor) for other in n)))
    ap = 1 - torch.stack(precisions).mean()
    auc = torch.stack([F.softplus(normal - anomaly) for anomaly in p for normal in n]).mean()
    t = detail["threshold"]
    fpr = torch.sigmoid(n - t).mean()
    torch.testing.assert_close(detail["ap"], ap)
    torch.testing.assert_close(detail["auc"], auc)
    torch.testing.assert_close(detail["fpr95"], fpr)
    torch.testing.assert_close(loss, ap + .1 * auc + .1 * fpr)
    assert abs(float(detail["recall"]) - .95) < 2e-6
    loss.backward()
    assert scores.grad[-1] == 0 and torch.isfinite(scores.grad).all()
    assert abs(float(scores.grad.sum())) < 2e-7


def test_recall_threshold_implicit_gradient_and_positive_tail_direction():
    from scipy.optimize import brentq
    from scipy.special import expit
    positive = torch.tensor([-1.3, -.2, .8, 2.1], requires_grad=True)
    threshold = RecallThreshold.apply(positive, 1., .95)
    threshold.backward()
    differences = []
    values = positive.detach().numpy().astype(float)
    for i in range(len(positive)):
        direction = np.zeros_like(values)
        direction[i] = .001
        roots = [brentq(lambda t: expit(values + sign * direction - t).mean() - .95, -40, 40)
                 for sign in (1, -1)]
        differences.append((roots[0] - roots[1]) / .002)
    torch.testing.assert_close(positive.grad, torch.tensor(differences, dtype=torch.float32), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(positive.grad.sum(), torch.tensor(1.))
    assert positive.grad[0] > positive.grad[-1] > 0
    positive.grad = None
    negative = torch.tensor([-.3, 1.], requires_grad=True)
    fpr = torch.sigmoid(negative - RecallThreshold.apply(positive, 1., .95)).mean()
    fpr.backward()
    assert torch.all(positive.grad < 0) and torch.all(negative.grad > 0)
    assert abs(float(positive.grad.sum() + negative.grad.sum())) < 1e-7


def test_rank_sampling_inclusion_weights_and_independent_random_stream():
    torch.manual_seed(71)
    state = torch.get_rng_state().clone()
    p, n = torch.linspace(-2, 3, 1000), torch.arange(9000, dtype=torch.float32)
    a, selected, weights, top = rank_sample(p, n, 91)
    assert len(a) == 256 and len(a.unique()) == 256
    assert len(selected) == 4096 and len(selected.unique()) == 4096 and top == 512
    assert set(selected[:512].tolist()) == set(range(8488, 9000))
    torch.testing.assert_close(weights.sum(), torch.tensor(9000.))
    torch.testing.assert_close(weights[512:], torch.full((3584,), 8488 / 3584))
    assert torch.equal(torch.get_rng_state(), state)
    for first, repeated in zip((a, selected, weights), rank_sample(p, n, 91)[:3]):
        assert torch.equal(first, repeated)
    for count in (1, 511, 512, 4096):
        _, selected, weights, _ = rank_sample(p[:1], n[:count], 91)
        assert len(selected) == count and torch.equal(weights, torch.ones(count))


def test_cross_scan_ranking_reaches_both_graphs_and_bce_only_pairs():
    class Score(nn.Module):
        def forward(self, sample):
            return sample["score"]
    positive = torch.tensor([.2, .7], requires_grad=True)
    negative = torch.tensor([-.1, .5, 1.], requires_grad=True)
    pair = [dict(score=positive, targets=torch.ones(2, dtype=torch.long)),
            dict(score=negative, targets=torch.zeros(3, dtype=torch.long))]
    counts = torch.tensor([3, 2])
    combined, details = forward_loss(Score(), pair, counts, rank_weight=1., rank_seed=8)
    bce, _ = forward_loss(Score(), pair, counts)
    (combined - bce).backward()
    assert torch.all(positive.grad < 0) and torch.all(negative.grad > 0)
    assert details["positive"] == 2 and details["negative"] == 3
    only, details = ranking_loss(negative, pair[1]["targets"], 8)
    assert float(only.detach()) == 0 and details["threshold"] is None


def test_local_supervision_uses_original_slots_and_effective_batch_mean():
    class Score(nn.Module):
        def forward(self, sample):
            return sample["score"]
    local = dict(index=6006, slots=[7, 91], weight=1.)
    samples = [dict(index=i, slots=torch.tensor([7, 8, 91]), targets=torch.zeros(3, dtype=torch.long),
                    score=torch.tensor([-.2, 2., .8], requires_grad=True)) for i in (6006, 8, 6006, 9)]
    counts = torch.tensor([12, 0])
    extra = 0.
    for begin in (0, 2):
        pair = samples[begin:begin + 2]
        weighted, _ = forward_loss(Score(), pair, counts, local=local, local_count=4)
        original, _ = forward_loss(Score(), pair, counts)
        extra = extra + weighted - original
    expected = F.softplus(torch.cat([samples[i]["score"][[0, 2]] for i in (0, 2)])).mean()
    torch.testing.assert_close(extra, expected)
    extra.backward()
    for sample in samples:
        gradient = sample["score"].grad
        if sample["index"] == 6006:
            torch.testing.assert_close(gradient[[0, 2]], sample["score"][[0, 2]].detach().sigmoid() / 4)
            assert gradient[1] == 0
        else:
            assert torch.equal(gradient, torch.zeros_like(gradient))
    samples[0]["targets"][0] = -1
    with pytest.raises(ValueError, match="same verified normal returns"):
        forward_loss(Score(), samples[:2], counts, local=local, local_count=2)


def test_point_record_preserves_training_and_original_point_identity(tmp_path):
    from src.train import point_record, record_scan, record_state
    class Score(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.5), nn.Linear(3, 1))
        def forward(self, sample):
            return self.layers(sample["xyzi"]).squeeze(-1)
    seed_all(18)
    model = Score().train()
    samples = [dict(index=i, xyzi=torch.arange(12).reshape(4, 3).float() + i,
                    slots=torch.tensor([0, 7, 8, 91]), targets=torch.tensor([0, 1, -1, i])) for i in range(2)]
    counts = torch.tensor([3, 3])
    initial, random = deepcopy(model.state_dict()), rng_state(torch.device("cpu"))
    expected, _ = forward_loss(model, samples, counts, rank_weight=1, rank_seed=37)
    expected.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    expected_state, expected_rng = deepcopy(model.state_dict()), rng_state(torch.device("cpu"))
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    restore_rng(random, torch.device("cpu"))
    manifest = dict(sha256="fixture", records=[dict(points=4), dict(points=4)])
    recorded = point_record(tmp_path, manifest, [0, 1], model, torch.device("cpu"), 1)
    record_state(recorded, 0)
    actual, details = forward_loss(model, samples, counts, rank_weight=1, rank_seed=37, record_points=True)
    actual.backward()
    for visit, (sample, scores) in enumerate(zip(samples, details["point_scores"])):
        record_scan(recorded, sample, scores, visit)
    record_state(recorded, 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_state[name], atol=0, rtol=0)
    assert torch.equal(rng_state(torch.device("cpu"))["torch"], expected_rng["torch"])
    for array in recorded["arrays"].values():
        array.flush()
    reopened = point_record(tmp_path, manifest, [0, 1], model, torch.device("cpu"), 1, resume=True)
    np.testing.assert_array_equal(reopened["arrays"]["train"], torch.cat(details["point_scores"]).numpy())
    np.testing.assert_array_equal(reopened["arrays"]["points"]["slot"], np.tile([0, 7, 8, 91], 2))
    np.testing.assert_array_equal(reopened["arrays"]["points"]["target"], torch.cat([s["targets"] for s in samples]).numpy())
    with pytest.raises(ValueError, match="input point population"):
        record_scan(reopened, samples[0], details["point_scores"][0][:-1], 0)
    continuation = tmp_path / "continuation"
    continuation.mkdir()
    shared = point_record(continuation, manifest, [1, 0], model, torch.device("cpu"), 1, identity_source=tmp_path)
    record_scan(shared, samples[1], details["point_scores"][1], 0)
    assert not shared["arrays"]["points"].flags.writeable
    assert (continuation / "points.npy").stat().st_ino == (tmp_path / "points.npy").stat().st_ino
    changed = dict(samples[1], targets=torch.zeros_like(samples[1]["targets"]))
    with pytest.raises(ValueError, match="point identity or label"):
        record_scan(shared, changed, details["point_scores"][1], 0)


def test_evaluation_record_preserves_official_population_and_ignored_returns(tmp_path, monkeypatch):
    import src.evaluate as evaluation
    class Score(nn.Module):
        def forward(self, sample):
            return sample["prediction"]
    sample = dict(index=0, xyzi=torch.arange(36).reshape(9, 4).float() + 1,
                  slots=torch.tensor([0, 2, 3, 5, 8, 9, 12, 15, 18]),
                  targets=torch.tensor([1, 1, 1, 1, 1, 0, 0, -1, -1]),
                  prediction=torch.tensor([2., 1., 3., 1., 4., .5, 1.5, 100., -100.]))
    monkeypatch.setattr(evaluation, "PreparedScans", lambda manifest: [sample])
    manifest = dict(kind="val", sha256="fixture", records=[dict(eligible=True, normal=2, anomaly=5, points=9)])
    expected = evaluation.evaluate(Score(), manifest, torch.device("cpu"), 0)
    actual = evaluation.evaluate(Score(), manifest, torch.device("cpu"), 0, tmp_path / "val1.npy", record_points=True)
    assert actual["metrics"] == expected["metrics"]
    np.testing.assert_array_equal(np.load(tmp_path / "val1.npy"), sample["prediction"][:7].numpy())
    np.testing.assert_array_equal(np.load(tmp_path / "val1_all.npy"), sample["prediction"].numpy())
    identities = np.load(tmp_path / "val_points.npy")
    np.testing.assert_array_equal(identities["slot"], sample["slots"].numpy())
    np.testing.assert_array_equal(identities["target"], sample["targets"].numpy())


def test_local_observation_preserves_rng_buffers_modes_and_next_training_update():
    from src.train import observe_local
    class Score(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.5), nn.Linear(3, 1))
        def forward(self, sample):
            return self.layers(sample["xyzi"]).squeeze(-1)
    seed_all(9)
    model = Score().train()
    sample = dict(xyzi=torch.arange(12).reshape(4, 3).float(), slots=torch.tensor([0, 7, 8, 91]),
                  targets=torch.zeros(4, dtype=torch.long))
    local = dict(slots=[7, 91])
    initial = deepcopy(model.state_dict())
    saved_rng = rng_state(torch.device("cpu"))
    expected = model(sample)
    expected.sum().backward()
    expected_state = deepcopy(model.state_dict())
    expected_grad = [p.grad.clone() for p in model.parameters()]
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    restore_rng(saved_rng, torch.device("cpu"))
    first = observe_local(model, sample, local, torch.device("cpu"))
    assert first == observe_local(model, sample, local, torch.device("cpu"))
    assert all(m.training for m in model.modules())
    actual = model(sample)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], atol=0, rtol=0)
    for parameter, gradient in zip(model.parameters(), expected_grad):
        torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)
    assert ranking_weight(1, 1000) == ranking_weight(100, 1000) == 0
    assert ranking_weight(150, 1000) == pytest.approx(.5)
    assert ranking_weight(200, 1000) == ranking_weight(1000, 1000) == 1
from vendor.litept.pointrope import PointROPE
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


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


def test_pilot_normal_rule_and_balanced_source_schedule():
    xyz = np.tile(np.array([[4., 0, 0, .5]], np.float32), (10, 1))
    for anomaly in (0, 1, 4, 5):
        frame = Frame(0, xyz, np.eye(4), np.array([2] * anomaly + [40] * (10 - anomaly), np.uint32))
        assert supervision(frame, allow_normal=True).eligible == (anomaly == 0 or anomaly >= 5)
        assert supervision(frame).eligible == (anomaly >= 5)
    records = [dict(group=group) for group in ("base", "targeted", "normal_nuscenes", "normal_stu")
               for _ in range(37)]
    order = pilot_order(dict(records=records), 0, 500)
    assert order == pilot_order(dict(records=records), 0, 500)
    assert len(order) == 4000
    for batch in effective_batches(order):
        assert [sum(records[i]["group"] == group for i in batch) for group in
                ("base", "targeted", "normal_nuscenes", "normal_stu")] == [4, 2, 1, 1]


def test_native_two_passes_preserve_every_record_and_partial_final_batch():
    from src.data import NATIVE_VERSION
    manifest = dict(version=NATIVE_VERSION, records=[dict(group="normal_nuscenes") for _ in range(19)])
    order = pilot_order(manifest, 0, 5)
    assert len(order) == 38 and sorted(order[:19]) == sorted(order[19:]) == list(range(19))
    assert len(list(effective_batches(order))[-1]) == 6
    assert order == pilot_order(manifest, 0, 5)
    with pytest.raises(ValueError, match="exactly twice"):
        pilot_order(manifest, 0, 6)


def test_conditional_interaction_learns_from_sampling_and_each_context_scale():
    torch.manual_seed(41)
    layer = Conditional()
    detail = torch.randn(7, 128, requires_grad=True)
    xyz = torch.randn(7, 3)
    sizes = [7, 5, 3, 2, 1, 7]
    features = [torch.randn(n, c, requires_grad=True) for n, c in zip(sizes, (*CHANNELS, 72))]
    indices = [torch.arange(7) % n for n in sizes]
    coords = [torch.randn(n, 3) for n in sizes]
    output = layer(detail, xyz, indices, coords, features)
    (output * torch.randn_like(output)).sum().backward()
    assert detail.grad.abs().sum() > 0
    assert all(x.grad.abs().sum() > 0 for x in features)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    for update in layer.layers:
        assert update["query"].weight.grad.abs().sum() > 0
    changed = layer(detail.detach() + torch.randn_like(detail), xyz, indices, coords, features)
    assert not torch.allclose(output, changed)


def test_nuscenes_raw_order_intensity_and_ignored_context(tmp_path):
    from src.data import nuscenes_mapping, read_nuscenes
    categories = json.loads(Path("results/data/labels.json").read_text())["nuscenes"]
    (tmp_path / "lidarseg").mkdir()
    (tmp_path / "lidarseg/category.json").write_text(json.dumps(
        [dict(index=row["raw"], name=row["name"]) for row in categories]))
    mapping = nuscenes_mapping(tmp_path)
    assert mapping == categories
    assert len(mapping) == 32 and sum(row["target"] for row in mapping) == 20
    indices = {row["name"]: row["raw"] for row in mapping}
    raw = np.array([[4, 0, 0, 255, 0], [70, 0, 0, 127.5, 31],
                    [5, 0, 0, 128, 15], [6, 0, 0, 10, 3]], np.float32)
    labels = np.array([indices[name] for name in
                       ("vehicle.car", "vehicle.car", "static.other", "noise")], np.uint8)
    scan, label = tmp_path / "scan.bin", tmp_path / "label.bin"
    raw.tofile(scan); labels.tofile(label)
    frame = read_nuscenes(dict(scan=scan, label=label, frame=0), mapping)
    np.testing.assert_array_equal(frame.xyzi[:, :3], raw[:, :3])
    np.testing.assert_array_equal(frame.xyzi[:, 3], raw[:, 3] / 255.)
    np.testing.assert_array_equal(point_targets(frame), [0, -1, -1, -1])
    assert frame.actual.all() and supervision(frame, allow_normal=True).eligible


def test_paired_control_preserves_p1_updates_and_microbatch_positions():
    records = ([dict(group="base") for _ in range(4000)] +
               [dict(group=group) for group in ("targeted", "normal_nuscenes", "normal_stu") for _ in range(37)])
    manifest = dict(records=records)
    reference = pilot_order(manifest, 0, 500)
    sampling = dict(base=6, normal_nuscenes=1, normal_stu=1)
    paired = pilot_order(manifest, 0, 500, sampling=sampling, paired=True)
    old_base = {i for i in reference if records[i]["group"] == "base"}
    replacements = []
    for left, right in zip(effective_batches(reference), effective_batches(paired)):
        assert [sum(records[i]["group"] == g for i in right) for g in sampling] == [6, 1, 1]
        for before, after in zip(left, right):
            if records[before]["group"] == "targeted":
                assert records[after]["group"] == "base" and after not in old_base
                replacements.append(after)
            else:
                assert before == after
    assert len(set(replacements)) == len(replacements) == 1000
    assert paired == pilot_order(manifest, 0, 500, sampling=sampling, paired=True)
    extended = pilot_order(manifest, 0, 2000, sampling=sampling, paired=True)
    assert extended[:4000] == paired
    assert extended[4000:] == pilot_order(manifest, 0, 1500, sampling=sampling, segment=1)
    assert len(extended) == 16000
    for batch in effective_batches(extended):
        assert [sum(records[i]["group"] == g for i in batch) for g in sampling] == [6, 1, 1]
    with pytest.raises(ValueError, match="paired control"):
        pilot_order(manifest, 0, 500, sampling=sampling, segment=500, paired=True)


def test_background_replacement_preserves_positions_and_excludes_check_logs(tmp_path):
    from collections import Counter
    from src.data import PILOT_VERSION, identity, replace_background
    records = [dict(group=g, frame=i, normal=7, anomaly=5 if g == "base" else 0)
               for g, size in (("base", 18), ("normal_nuscenes", 5), ("normal_stu", 3))
               for i in range(size)]
    reference = dict(version=PILOT_VERSION, kind="train", records=records, mapping=[],
                     normal_manifest="original-normal", split=dict(check=["hold"], logs={"hold": "blocked"}))
    reference["sha256"] = identity(reference)
    reference_path = tmp_path / "train.json"
    reference_path.write_text(json.dumps(reference))
    order = []
    for step in range(31):
        batch = [(step * 6 + i) % 18 for i in range(6)] + [23 + step % 3]
        batch.insert(step % 8, 18 + step % 5)
        order.extend(batch)
    sampling_path = tmp_path / "sampling.json"
    sampling_path.write_text(json.dumps(dict(train_manifest=reference["sha256"], order=order)))
    rows = [dict(source="nuscenes", group="normal_nuscenes", subset="train", normal=10, anomaly=0,
                 scene=f"scene-{scene}", token=f"{scene}-{frame}", log_token=f"log-{scene}")
            for scene in range(11) for frame in range(8)]
    normal = dict(version=PILOT_VERSION, kind="normal", mapping=[], reference_manifest="original-normal",
                  records=rows, split=dict(check=["hold"], excluded_logs=["blocked"]))
    normal_path = tmp_path / "normal.json"
    normal_path.write_text(json.dumps(dict(normal, sha256=identity(normal))))
    expanded = replace_background(reference, normal_path, reference_path)
    actual = pilot_order(expanded, 0, 31, background=dict(manifest=reference_path, sampling=sampling_path))
    replacements = []
    for before, after in zip(order, actual):
        if records[before]["group"] == "normal_nuscenes":
            replacements.append(expanded["records"][after])
        else:
            assert records[before] == expanded["records"][after]
    counts = Counter(r["scene"] for r in replacements)
    assert len(replacements) == len({r["token"] for r in replacements}) == 31
    assert len(counts) == 11 and max(counts.values()) - min(counts.values()) == 1
    rows[0]["log_token"] = "blocked"
    normal_path.write_text(json.dumps(dict(normal, sha256=identity(normal))))
    with pytest.raises(ValueError, match="held-out-log"):
        replace_background(reference, normal_path, reference_path)


@pytest.mark.parametrize("sizes", [[(3, 20), (11, 2), (1, 1)], [(0, 5), (0, 3)], [(8, 0)]])
def test_global_class_weighted_accumulation(sizes):
    torch.manual_seed(7)
    x = torch.randn(sum(sum(s) for s in sizes), 3, dtype=torch.float64)
    labels = torch.cat([torch.tensor([0] * n + [1] * a) for n, a in sizes])
    counts = torch.tensor([int((labels == k).sum()) for k in (0, 1)], dtype=torch.int64)
    a = nn.Linear(3, 1, dtype=torch.float64)
    b = deepcopy(a)
    full = balanced_loss(a(x).squeeze(-1), labels, counts)
    full.backward()
    start, terms = 0, []
    for normal, anomaly in sizes:
        stop = start + normal + anomaly
        term = balanced_loss(b(x[start:stop]).squeeze(-1), labels[start:stop], counts)
        terms.append(term.detach())
        term.backward()
        start = stop
    torch.testing.assert_close(sum(terms), full.detach(), atol=1e-7, rtol=1e-7)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-7)
    direct = torch.stack([F.softplus((1 if k == 0 else -1) * a(x).squeeze(-1)[labels == k].float()).mean()
                          for k in (0, 1) if counts[k]]).mean()
    torch.testing.assert_close(full, direct)


def test_distributed_tail_and_order_are_unique_and_method_independent():
    for count in (1, 7, 8, 9, 17, 40152):
        order = epoch_order(count, 2, 2, 13)
        assert len(set(order)) == count and sorted(order) == list(range(count))
        for ranks in (1, 2, 3, 4, 8):
            per_rank = [list(effective_batches(order, r, ranks)) for r in range(ranks)]
            assert all(len(groups) == math.ceil(count / 8) for groups in per_rank)
            combined = [i for groups in per_rank for batch in groups for i in batch]
            assert sorted(combined) == list(range(count))
    assert epoch_order(31, 0, 2, 0) != epoch_order(31, 0, 2, 1)
    assert epoch_order(31, 0, 1, 0) != epoch_order(31, 0, 2, 0)


def test_schedule_selection_optimizer_and_random_state():
    assert EPOCHS == (8, 4)
    for total in (2000, 40152, 20076):
        warmup = math.ceil(.05 * total)
        assert lr_factor(1, total) == .1
        assert lr_factor(warmup, total) == 1.
        assert lr_factor(total, total) == .01
        assert lr_factor(warmup + 1, total) < 1.
    m = dict(AP=1., FPR95=9., AUROC=50.)
    assert not better(m, m)
    assert better(dict(m, AP=1. + 1e-12), m)
    assert better(dict(m, FPR95=8.), m)
    assert better(dict(m, AUROC=51.), m)
    assert not better(dict(m, AP=.99, FPR95=0., AUROC=99.), m)
    model = Segmentor("attention")
    for stage in (1, 2):
        optimizer = optimizer_for(model, stage)
        ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        lookup = {id(p): g for g in optimizer.param_groups for p in g["params"]}
        for name, parameter in model.named_parameters():
            if name.endswith(".bias") or parameter.ndim == 1:
                assert lookup[id(parameter)]["weight_decay"] == 0.
        assert lookup[id(model.backbone.embedding.stem.conv.weight)]["peak_lr"] == (2e-4 if stage == 1 else 2e-5)
        assert lookup[id(model.detail[0].weight)]["peak_lr"] == (2e-3 if stage == 1 else 2e-5)
        assert lookup[id(model.interaction_weight.weight)]["peak_lr"] == (2e-3 if stage == 1 else 2e-4)
    seed_all(2)
    state = rng_state(torch.device("cpu"))
    first = torch.rand(5), np.random.rand(5)
    restore_rng(state, torch.device("cpu"))
    torch.testing.assert_close(torch.rand(5), first[0], atol=0, rtol=0)
    np.testing.assert_array_equal(np.random.rand(5), first[1])


def test_interaction_formula_and_zero_initialized_head():
    torch.manual_seed(3)
    module = Interaction("attention")
    features = [torch.randn(7, c, requires_grad=True) for c in CHANNELS]
    coords = [torch.randn(7, 3) for _ in CHANNELS]
    indices = [torch.tensor([0, 2, 2, 6]) for _ in CHANNELS]
    xyz, sampling = torch.randn(4, 3), torch.randn(4, 64)
    keys, values = module.project(features)
    actual = module(xyz, sampling, indices, coords, keys, values)
    distance = xyz.norm(dim=-1, keepdim=True)
    sensor = module.position(torch.cat((distance / 50, xyz / distance), -1))
    query = module.query(torch.cat((sampling, sensor), -1)).reshape(4, 4, 16)
    bias = module.relative(torch.stack([(xyz - c[i]) / (.05 * 2**l)
                                        for l, (c, i) in enumerate(zip(coords, indices))], 1))
    expected = torch.empty_like(actual)
    for point in range(4):
        heads = []
        for head in range(4):
            logits, val = [], []
            for level in range(5):
                idx = indices[level][point]
                logits.append(query[point, head] @ keys[level][idx].reshape(4, 16)[head] / 4 + bias[point, level, head])
                val.append(values[level][idx].reshape(4, 16)[head])
            heads.append((torch.stack(logits).softmax(0)[:, None] * torch.stack(val)).sum(0))
        expected[point] = module.output(torch.cat(heads))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual.sum().backward()
    assert all(f.grad is not None and torch.isfinite(f.grad).all() for f in features)
    model = Segmentor()
    for mode in ("attention", "fusion"):
        branch = deepcopy(model)
        branch.add_interaction(mode)
        hidden = torch.randn(4, 128)
        r = torch.randn(4, 64)
        torch.testing.assert_close(model.head(hidden),
                                   branch.head[2](branch.head[1](branch.head[0](hidden) + branch.interaction_weight(r))),
                                   atol=0, rtol=0)
    assert torch.count_nonzero(model.adapter[-1].weight) == 0


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


def test_manifest_decoding_on_real_saved_scans():
    path = Path("assets/train.json")
    if not path.is_file():
        pytest.skip("build the real-data manifests first")
    manifest = load_manifest(path, "train")
    dataset = Scans(manifest)
    # Select distinct source times and all anomaly-count boundary cases present.
    indices = {0, len(dataset) - 1, max(range(len(dataset)), key=lambda i: dataset.records[i]["points"])}
    for target in (5, 6, 20):
        indices.add(next(i for i, r in enumerate(dataset.records) if r["anomaly"] == target))
    for index in sorted(indices):
        sample = dataset[index]
        row = dataset.records[index]
        original = STUSequence(manifest["data_root"])[row["frame"]]
        independent = restore_delta(row["delta"], original, row["world"])
        np.testing.assert_array_equal(sample["xyzi"], independent.xyzi[independent.actual])
        np.testing.assert_array_equal(sample["targets"], point_targets(independent)[independent.actual])
        np.testing.assert_array_equal(sample["slots"], independent.return_slots)
        assert len(sample["xyzi"]) == row["points"]
        assert (sample["targets"] == 1).sum() == row["anomaly"]
        np.testing.assert_array_equal(dataset[index]["xyzi"], sample["xyzi"])
    assert len(manifest["worlds"]) == 240
    assert len({(r["world"], r["frame"]) for r in manifest["records"]}) == len(dataset)
    assert all(r["anomaly"] >= 5 for r in manifest["records"])
    real = load_manifest("assets/val.json", "val")
    assert manifest["version"] == real["version"] == MANIFEST_VERSION == "AJAE-V4-F240-R1"
    config = configuration(manifest, real, torch.device("cpu"), 1)
    assert config["version"] == VERSION == "AJAE-V4-F240-R2"
    assert config["data_version"] == MANIFEST_VERSION
    assert config["seeds"] == [0] and config["epochs"] == (8, 4)
    reader = Scans(real)
    index = next(i for i, r in enumerate(real["records"]) if r["eligible"])
    sample = reader[index]
    official = PointOODMetricsCalculator()
    scores = np.arange(len(sample["xyzi"]), dtype=np.float32)  # Diagnostic ranks, not scientific predictions.
    truth = np.where(sample["targets"] < 0, 0, sample["targets"] + 1)
    official.update(sample["xyzi"][:, :3], scores, truth)
    assert len(official.all_labels[0]) == real["records"][index]["normal"] + real["records"][index]["anomaly"]


class _ToyModel(nn.Module):
    """Tiny implementation fixture; never used for the V4 scientific experiment."""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.backbone = nn.Linear(3, 4)
        self.dropout = nn.Dropout(.2)
        self.head = nn.Linear(4, 1)
        if mode != "base":
            self.interaction = nn.Linear(3, 4)
            self.interaction_weight = nn.Linear(4, 4, bias=False)
            nn.init.zeros_(self.interaction_weight.weight)

    def load_pretrained(self, path):
        return {"fixture": True}

    def forward(self, sample):
        hidden = self.backbone(sample["xyzi"])
        if self.mode != "base":
            hidden = hidden + self.interaction_weight(self.interaction(sample["xyzi"]))
        return self.head(self.dropout(hidden)).squeeze(-1)


class _ToyScans:
    def __init__(self, manifest):
        self.records = manifest["records"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        normal, anomaly = self.records[index]["normal"], self.records[index]["anomaly"]
        x = torch.arange((normal + anomaly) * 3).reshape(-1, 3).float() / 30 + index / 40
        return dict(xyzi=x, targets=torch.tensor([0] * normal + [1] * anomaly))


@pytest.mark.parametrize("start_update,material", [(500,False),(1000,False),(1000,True)])
def test_native_branch_restores_moments_rng_sampling_and_remaining_schedule(tmp_path, monkeypatch, start_update, material):
    import copy
    import src.train as training
    from src.data import NATIVE_VERSION, file_sha256
    monkeypatch.setattr(training, "Segmentor", _ToyModel)
    monkeypatch.setattr(training, "PreparedScans", _ToyScans)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    device = torch.device("cpu")
    manifest = dict(version=NATIVE_VERSION, sha256="fixture-train", records=[
        dict(group="anomaly_stu", normal=5, anomaly=5) for _ in range(6185)])
    order = pilot_order(manifest, 0, 1547)
    sampling = tmp_path / "sampling.json"
    sampling.write_text(json.dumps(dict(order=order)))
    training.seed_all(7)
    model = _ToyModel("conditional")
    optimizer = optimizer_for(model, 1)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    for state in optimizer.state.values():
        state["step"].fill_(start_update)
    metrics = dict(AP=75., FPR95=.2, AUROC=99.9)
    parent = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                  scaler={}, rng=[training.rng_state(device)],
                  validation=dict(metrics=metrics, manifest_sha256="fixture-val"))
    initial = tmp_path / "best.pt"
    torch.save(parent, initial)
    config = dict(version=NATIVE_VERSION, updates=start_update+500, schedule_updates=1547, start_update=start_update,
                  eval_every=500, epochs=None, microbatch=2, branch="lr" if start_update==500 else "control", recipe="native", objective="metrics",
                  loss=dict(auc_weight=.1, fpr95_weight=.1), lr_scale=.3,
                  reference_sampling=str(sampling), initial_sha256=file_sha256(initial))
    if material:
        metrics = dict(AP=78., FPR95=.1, AUROC=99.95)
        config.update(branch="material-control", updates=start_update+40, eval_every=40,
                      initial_validation=dict(metrics=metrics, manifest_sha256="fixture-val"))
    optimizer.zero_grad(set_to_none=True)
    counts = torch.tensor([40,40], dtype=torch.int64)
    dataset = _ToyScans(manifest)
    offset = start_update*8
    for k in range(4):
        loss,_ = training.forward_loss(model,[dataset[i] for i in order[offset+k*2:offset+2+k*2]],counts,
                                       rank_weight=.25,rank_seed=(start_update+1)*8+k)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    for group in optimizer.param_groups:
        group["lr"] = group["peak_lr"] * lr_factor(start_update+1,1547) * .3
    optimizer.step()
    expected_rng = training.rng_state(device)
    args = SimpleNamespace(output=tmp_path / "branch", initial=initial, resume=False, workers=0,
                           save_every=500, score_path=None)
    monkeypatch.setattr(training, "STOP", True)
    assert not training.train_stage(args,manifest,dict(sha256="fixture-val"),0,"conditional",device,config)
    saved = torch.load(args.output / "0/conditional/last.pt",weights_only=False)
    assert saved["planned_updates"] == saved["successful_updates"] == start_update+1
    assert all(int(s["step"])==start_update+1 for s in saved["optimizer"]["state"].values())
    assert saved["best_metrics"] == metrics and saved["epoch_frames"] == 8
    assert torch.equal(saved["rng"][0]["torch"],expected_rng["torch"])
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,saved["model"][name],rtol=0,atol=0)


def test_uniform_native_continuation_keeps_moments_rng_and_full_passes(tmp_path, monkeypatch):
    import src.train as training
    from src.data import NATIVE_VERSION, file_sha256
    monkeypatch.setattr(training, "Segmentor", _ToyModel)
    monkeypatch.setattr(training, "PreparedScans", _ToyScans)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    device = torch.device("cpu")
    manifest = dict(version=NATIVE_VERSION, sha256="fixture-train", records=[
        dict(group="anomaly_stu", normal=5, anomaly=5) for _ in range(11)])
    order = pilot_order(manifest, 0, 3, segment=1)
    assert order == epoch_order(11, 0, 1, 2) + epoch_order(11, 0, 1, 3)
    assert all(order.count(i) == 2 for i in range(11))
    assert training.continuation_factor(1, 3) == pytest.approx(.1)
    assert training.continuation_factor(3, 3) == pytest.approx(.01)
    seed_all(7)
    model = _ToyModel("conditional")
    optimizer = optimizer_for(model, 1)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    for value in optimizer.state.values():
        value["step"].fill_(1547)
    metrics = dict(AP=75., FPR95=.2, AUROC=99.9)
    config = dict(version=NATIVE_VERSION, updates=3, eval_every=3, epochs=2, microbatch=2,
        recipe="native", objective="metrics", loss=dict(auc_weight=.1, fpr95_weight=.1),
        world_size=1, train_manifest="fixture-train", sampling="two passes", sampling_segment=1,
        optimizer_state="inherit", parent_updates=1547, continuation_schedule=dict(warmup=False),
        initial_validation=dict(metrics=metrics, manifest_sha256="fixture-val"))
    initial = tmp_path / "initial.pt"
    torch.save(dict(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
        scaler={}, rng=[rng_state(device)], config=dict(config, sampling_segment=0), successful_updates=1547), initial)
    config["initial_sha256"] = file_sha256(initial)
    optimizer.zero_grad(set_to_none=True)
    data = _ToyScans(manifest)
    for k in range(4):
        loss, _ = forward_loss(model, [data[i] for i in order[k*2:k*2+2]], torch.tensor([40, 40]),
                               rank_weight=.25, rank_seed=1548*8+k)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    for group in optimizer.param_groups:
        group["lr"] = group["peak_lr"] * training.continuation_factor(1, 3)
    optimizer.step()
    expected_rng = rng_state(device)
    args = SimpleNamespace(output=tmp_path / "continued", initial=initial, resume=True,
                           workers=0, save_every=100, score_path=None)
    monkeypatch.setattr(training, "STOP", True)
    assert not training.train_stage(args, manifest, dict(sha256="fixture-val"), 0, "conditional", device, config)
    saved = torch.load(args.output / "0/conditional/last.pt", weights_only=False)
    assert {int(s["step"]) for s in saved["optimizer"]["state"].values()} == {1548}
    assert saved["planned_updates"] == 1 and saved["best_metrics"] == metrics
    assert torch.equal(saved["rng"][0]["torch"], expected_rng["torch"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
    calls = []
    def validate(*unused, **kwargs):
        calls.append(1)
        return dict(metrics=metrics, manifest_sha256="fixture-val")
    monkeypatch.setattr(training, "validate_all", validate)
    monkeypatch.setattr(training, "STOP", False)
    assert training.train_stage(args, manifest, dict(sha256="fixture-val"), 0, "conditional", device, config)
    saved = torch.load(args.output / "0/conditional/last.pt", weights_only=False)
    assert len(calls) == 1 and saved["complete"] and saved["planned_updates"] == 3
    assert {int(s["step"]) for s in saved["optimizer"]["state"].values()} == {1550}
    report = json.loads((args.output / "0/conditional/epoch1.json").read_text())
    assert report["frames"] == 22


def test_material_sampling_reuses_every_other_source_position():
    from src.train import material_order
    manifest = dict(records=[dict(group="normal_stu" if i % 7 == 0 else "anomaly_stu") for i in range(80)])
    order = np.random.default_rng(19).permutation(80).tolist()
    changed = material_order(order, manifest, [0], 2, 8)
    assert changed[:16] == order[:16] and changed[64:] == order[64:]
    for position in range(16, 64):
        assert changed[position] == (0 if manifest["records"][order[position]]["group"] == "normal_stu" else order[position])
    assert material_order(order, manifest, [0], 2, 8) == changed


def test_hard_sampling_preserves_six_positions_sources_and_prefix():
    from src.train import hard_order
    groups = ("normal_nuscenes","anomaly_nuscenes","normal_stu","anomaly_stu")
    manifest = dict(records=[dict(group=groups[i%4]) for i in range(128)])
    pool = dict(groups={g:[i for i in range(128) if groups[i%4]==g] for g in groups})
    original = np.random.default_rng(29).permutation(128).tolist()
    changed = hard_order(original,manifest,pool,2,14)
    assert changed[:16]==original[:16] and changed[112:]==original[112:]
    for start in range(16,112,8):
        assert sum(a!=b for a,b in zip(original[start:start+8],changed[start:start+8]))==2
        assert len(set(changed[start:start+8]))==8
        assert [manifest['records'][i]['group'] for i in changed[start:start+8]]==[manifest['records'][i]['group'] for i in original[start:start+8]]
    assert changed==hard_order(original,manifest,pool,2,14)


def test_pilot_budget_single_validation_and_resume(tmp_path, monkeypatch):
    import src.train as training
    from src.data import PILOT_VERSION
    monkeypatch.setattr(training, "Segmentor", _ToyModel)
    monkeypatch.setattr(training, "PreparedScans", _ToyScans)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    evaluations = []
    metrics = dict(AP=2., AUROC=51., FPR95=93., threshold=.5)
    def validate(*args, score_path=None):
        evaluations.append(1)
        return dict(metrics=metrics, manifest_sha256="fixture-val")
    monkeypatch.setattr(training, "validate_all", validate)
    initial = tmp_path / "initial.pt"
    torch.save(dict(mode="base", model=_ToyModel("base").state_dict(), complete=True, selected=True,
                    validation=dict(metrics=metrics, manifest_sha256="fixture-val")), initial)
    manifest = dict(records=[dict(group=group, normal=5, anomaly=5 if group in ("base", "targeted") else 0)
                            for group in ("base", "targeted", "normal_nuscenes", "normal_stu")
                            for _ in range(9)])
    config = dict(version=PILOT_VERSION, updates=7, epochs=None, fixture=True)
    args = SimpleNamespace(output=tmp_path / "full", initial=initial, resume=True, workers=0, save_every=500, score_path=None)
    val, device = dict(sha256="fixture-val"), torch.device("cpu")
    monkeypatch.setattr(training, "STOP", False)
    assert training.train_stage(args, manifest, val, 0, "pilot", device, config)
    full = torch.load(args.output / "0/pilot/last.pt", weights_only=False)
    assert len(evaluations) == 1 and full["planned_updates"] == full["successful_updates"] == 7
    assert full["final_metrics"] == metrics and full["complete"]
    args.output = tmp_path / "resumed"
    training.STOP = True
    assert not training.train_stage(args, manifest, val, 0, "pilot", device, config)
    assert len(evaluations) == 1
    training.STOP = False
    assert training.train_stage(args, manifest, val, 0, "pilot", device, config)
    assert len(evaluations) == 2
    resumed = torch.load(args.output / "0/pilot/last.pt", weights_only=False)
    for name, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][name], atol=0, rtol=0)


def test_continuation_keeps_optimizer_and_best_across_intervals(tmp_path, monkeypatch):
    import src.train as training
    monkeypatch.setattr(training, "Segmentor", _ToyModel)
    monkeypatch.setattr(training, "PreparedScans", _ToyScans)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    sampling = dict(base=4, targeted=2, normal_nuscenes=1, normal_stu=1)
    manifest = dict(sha256="fixture-train", records=[
        dict(group=group, normal=5, anomaly=5 if group in ("base", "targeted") else 0)
        for group in sampling for _ in range(9)])
    initial = tmp_path / "initial.pt"
    base_metrics = dict(AP=2., AUROC=51., FPR95=93.)
    torch.save(dict(mode="base", model=_ToyModel("base").state_dict(), complete=True, selected=True,
                    validation=dict(metrics=base_metrics, manifest_sha256="fixture-val")), initial)
    config = dict(updates=7, sampling=sampling, world_size=1, train_manifest="fixture-train", fixture=True)
    args = SimpleNamespace(output=tmp_path / "parent", initial=initial, resume=True, workers=0, save_every=500, score_path=None)
    val, device = dict(sha256="fixture-val"), torch.device("cpu")
    def validate_parent(model, *unused, score_path=None):
        model.eval()
        return dict(metrics=dict(base_metrics, AP=3.), manifest_sha256="fixture-val")
    monkeypatch.setattr(training, "validate_all", validate_parent)
    monkeypatch.setattr(training, "STOP", False)
    assert training.train_stage(args, manifest, val, 0, "pilot", device, config)
    args.initial = args.output / "0/pilot/best.pt"
    config = dict(config, updates=9, eval_every=3, optimizer_state="inherit", sampling_segment=7)
    calls = []
    stop_after_first = False
    def validate(model, *unused, score_path=None):
        model.eval()
        ap = [4., 2., 3.5][len(calls)]
        calls.append(ap)
        if stop_after_first and len(calls) == 1:
            training.STOP = True
        return dict(metrics=dict(base_metrics, AP=ap), manifest_sha256="fixture-val")
    monkeypatch.setattr(training, "validate_all", validate)
    args.output = tmp_path / "full"
    assert training.train_stage(args, manifest, val, 0, "pilot", device, config)
    full = torch.load(args.output / "0/pilot/last.pt", weights_only=False)
    best = torch.load(args.output / "0/pilot/best.pt", weights_only=False)
    assert calls == [4., 2., 3.5] and full["successful_updates"] == 9
    assert best["best_epoch"] == 1 and best["successful_updates"] == 3
    assert all(int(state["step"]) == 16 for state in full["optimizer"]["state"].values())
    assert all(g["lr"] == g["peak_lr"] * .01 for g in full["optimizer"]["param_groups"])
    calls.clear()
    stop_after_first = True
    args.output = tmp_path / "resumed"
    assert not training.train_stage(args, manifest, val, 0, "pilot", device, config)
    stopped = torch.load(args.output / "0/pilot/last.pt", weights_only=False)
    assert stopped["epoch"] == 1 and stopped["next_batch"] == 1 and stopped["successful_updates"] == 4
    training.STOP = False
    assert training.train_stage(args, manifest, val, 0, "pilot", device, config)
    resumed = torch.load(args.output / "0/pilot/last.pt", weights_only=False)
    assert calls == [4., 2., 3.5]
    for name, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][name], atol=0, rtol=0)
    old_order = pilot_order(manifest, 0, 500, sampling=dict(base=8))
    assert len(old_order) == 4000 and all(manifest["records"][i]["group"] == "base" for i in old_order)
    assert lr_factor(1, 1500) == .1 and lr_factor(75, 1500) == 1 and lr_factor(1500, 1500) == .01
    assert lr_factor(500, 1500) > .01


def test_actual_training_loop_resume_inheritance_epoch_zero_and_tail(tmp_path, monkeypatch):
    import src.train as training
    monkeypatch.setattr(training, "Segmentor", _ToyModel)
    monkeypatch.setattr(training, "PreparedScans", _ToyScans)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    def validate(model, *args, score_path=None):
        model.eval()
        return dict(metrics=dict(AP=2., AUROC=51., FPR95=93., threshold=.5))
    monkeypatch.setattr(training, "validate_all", validate)
    manifest = dict(records=[dict(normal=3 + i % 3, anomaly=5 + i % 2) for i in range(19)])
    config = dict(fixture=True, val_manifest="fixture-val", train_manifest="fixture-train")
    device = torch.device("cpu")
    args = SimpleNamespace(output=tmp_path / "full", resume=True, weights=None, workers=0, save_every=500, score_path=None)
    monkeypatch.setattr(training, "STOP", False)
    assert training.train_stage(args, manifest, {}, 0, "base", device, config)
    full = torch.load(args.output / "0/base/last.pt", weights_only=False)
    assert full["epoch"] == 8 and full["planned_updates"] == full["successful_updates"] == 24
    assert all(group["lr"] == group["peak_lr"] * .01 for group in full["optimizer"]["param_groups"])
    args.output = tmp_path / "resumed"
    training.STOP = True
    assert not training.train_stage(args, manifest, {}, 0, "base", device, config)
    interrupted = torch.load(args.output / "0/base/last.pt", weights_only=False)
    assert interrupted["planned_updates"] == 1 and interrupted["epoch_frames"] == 8
    training.STOP = False
    assert training.train_stage(args, manifest, {}, 0, "base", device, config)
    resumed = torch.load(args.output / "0/base/last.pt", weights_only=False)
    for name, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][name], atol=0, rtol=0)
    for method in ("attention", "continue", "fusion"):
        assert training.train_stage(args, manifest, {}, 0, method, device, config)
        last = torch.load(args.output / f"0/{method}/last.pt", weights_only=False)
        assert last["epoch"] == 4 and last["planned_updates"] == last["successful_updates"] == 12
        assert all(group["lr"] == group["peak_lr"] * .01 for group in last["optimizer"]["param_groups"])
        selected = torch.load(args.output / f"0/{method}/best.pt", weights_only=False)
        assert selected["best_epoch"] == selected["epoch"] == 0
        assert selected["complete"] and selected["selected"]
        parent = torch.load(args.output / "0/base/best.pt", weights_only=False)
        for name, value in parent["model"].items():
            torch.testing.assert_close(value, selected["model"][name], atol=0, rtol=0)
    summarize(args.output)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["version"] == VERSION and summary["seeds"] == [0]
    assert summary["repeat_uncertainty_estimated"] is False
    assert set(summary["methods"]) == {"base", "attention", "continue", "fusion"}
    assert summary["methods"]["base"]["epoch"] == 1
    for method in ("attention", "continue", "fusion"):
        assert summary["methods"][method]["epoch"] == 0
        assert summary["methods"][method]["validation_improved_from_epoch0"] is False
        assert "population_std" not in summary["methods"][method]
    with pytest.raises(ValueError, match="sole experiment seed"):
        training.train_stage(args, manifest, {}, 1, "base", device, config)
    result_path = args.output / "0/attention/result.json"
    result = json.loads(result_path.read_text())
    result_path.write_text(json.dumps(dict(result, version=MANIFEST_VERSION)))
    with pytest.raises(ValueError, match="incorrect experiment identity"):
        summarize(args.output)
    result_path.write_text(json.dumps(dict(result, complete=False)))
    with pytest.raises(ValueError, match="unfinished experiment"):
        summarize(args.output)


def _distributed_gradient_worker(rank, rendezvous, output):
    import torch.distributed as distributed
    from src.train import sync_gradients
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    torch.manual_seed(4)
    model = nn.Linear(3, 1)
    # One rank has no microbatch in the final global batch; it must contribute zero.
    x = torch.arange(15).float().reshape(5, 3) / 10
    labels = torch.tensor([0, 0, 1, 1, 1])
    counts = torch.tensor([2, 3], dtype=torch.int64)
    if rank == 0:
        balanced_loss(model(x).squeeze(-1), labels, counts).backward()
    sync_gradients(model)
    if rank == 0:
        torch.save([p.grad for p in model.parameters()], output)
    distributed.destroy_process_group()


def test_distributed_gradient_sum_with_empty_final_rank(tmp_path):
    import torch.multiprocessing as multiprocessing
    rendezvous = "file://" + str(tmp_path / "rendezvous")
    output = str(tmp_path / "gradients.pt")
    multiprocessing.spawn(_distributed_gradient_worker, args=(rendezvous, output), nprocs=2, join=True)
    torch.manual_seed(4)
    model = nn.Linear(3, 1)
    x = torch.arange(15).float().reshape(5, 3) / 10
    balanced_loss(model(x).squeeze(-1), torch.tensor([0, 0, 1, 1, 1]), torch.tensor([2, 3])).backward()
    for expected, observed in zip(model.parameters(), torch.load(output, weights_only=True)):
        torch.testing.assert_close(expected.grad, observed, atol=0, rtol=0)
