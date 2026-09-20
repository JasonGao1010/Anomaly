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
from src.model import (CHANNELS, Interaction, Segmentor, balanced_loss, prepare_scan,
                       scatter_scores, voxelize)
from src.train import (EPOCHS, configuration, effective_batches, epoch_order, lr_factor, optimizer_for, pilot_order,
                       rng_state, restore_rng, seed_all)
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
    with pytest.raises(ValueError, match="paired control"):
        pilot_order(manifest, 0, 500, sampling=sampling, segment=500, paired=True)


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
    for total in (40152, 20076):
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


def test_pilot_budget_single_validation_and_resume(tmp_path, monkeypatch):
    import src.train as training
    from src.data import PILOT_VERSION
    monkeypatch.setattr(training, "Segmentor", _ToyModel)
    monkeypatch.setattr(training, "PreparedScans", _ToyScans)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    evaluations = []
    metrics = dict(AP=2., AUROC=51., FPR95=93., threshold=.5)
    def validate(*args):
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
    args = SimpleNamespace(output=tmp_path / "full", initial=initial, resume=True, workers=0, save_every=500)
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
    args = SimpleNamespace(output=tmp_path / "parent", initial=initial, resume=True, workers=0, save_every=500)
    val, device = dict(sha256="fixture-val"), torch.device("cpu")
    def validate_parent(model, *unused):
        model.eval()
        return dict(metrics=dict(base_metrics, AP=3.), manifest_sha256="fixture-val")
    monkeypatch.setattr(training, "validate_all", validate_parent)
    monkeypatch.setattr(training, "STOP", False)
    assert training.train_stage(args, manifest, val, 0, "pilot", device, config)
    args.initial = args.output / "0/pilot/best.pt"
    config = dict(config, updates=9, eval_every=3, optimizer_state="inherit", sampling_segment=7)
    calls = []
    stop_after_first = False
    def validate(model, *unused):
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
    def validate(model, *args):
        model.eval()
        return dict(metrics=dict(AP=2., AUROC=51., FPR95=93., threshold=.5))
    monkeypatch.setattr(training, "validate_all", validate)
    manifest = dict(records=[dict(normal=3 + i % 3, anomaly=5 + i % 2) for i in range(19)])
    config = dict(fixture=True, val_manifest="fixture-val", train_manifest="fixture-train")
    device = torch.device("cpu")
    args = SimpleNamespace(output=tmp_path / "full", resume=True, weights=None, workers=0, save_every=500)
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
