from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.data import FrozenFrame
from src.model import RelationLayer, ScanTransform, load_config, shell_neighbors
from src.scene import PointLabels, make_source_frame
from src.train import (Requests, auxiliary_fraction, detection_loss, keep_loss,
                       query_rows, tail_loss, training_forward)
from vendor.litept.pointrope import PointROPE


def scan_fixture():
    xyz = np.array([[0, 0, 0], [10, 0, 0], [10.02, 0, 0], [10.02, 0, 0],
                    [10.1, .01, 0], [10.3, .1, 0], [11, .2, .05], [30, 0, 0],
                    [0, 0, 3]], np.float32)
    xyzi = np.column_stack((xyz, np.linspace(.1, .9, len(xyz)))).astype(np.float32)
    packed = np.full(len(xyzi), 40, np.uint32)
    target = np.full(len(xyzi), 8, np.uint8)
    target[4] = 255
    labels = PointLabels(packed, packed.astype(np.uint16), np.zeros(len(xyzi), np.uint16), target)
    return make_source_frame(3, xyzi, np.eye(4), labels, partition="fixture", sequence_id=1)


def test_annular_search_matches_exhaustive_distances_boundaries_and_ties():
    rng = np.random.default_rng(8)
    # Include exact shell boundaries, dense support and repeated equal distances.
    xyz = np.r_[rng.normal(size=(300, 3)), [[0, 0, 0], [.25, 0, 0], [.75, 0, 0],
                                         [2, 0, 0], [-.25, 0, 0], [10, 0, 0]]]
    first = rng.permutation(len(xyz)).astype(np.int32)
    radii = [.25, .75, 2.]
    ids, distances = shell_neighbors(xyz, first, radii, 16)
    for i in range(len(xyz)):
        d2 = np.sum((xyz - xyz[i]) ** 2, axis=1)
        inner = 0.
        for shell, radius in enumerate(radii):
            valid = np.flatnonzero((d2 > inner) & (d2 <= radius**2))
            order = valid[np.lexsort((first[valid], d2[valid]))][:16]
            expected = np.full(16, -1)
            expected[:len(order)] = first[order]
            np.testing.assert_array_equal(ids[i, shell * 16:(shell + 1) * 16], expected)
            np.testing.assert_allclose(distances[i, shell * 16:shell * 16 + len(order)], np.sqrt(d2[order]), atol=1e-14)
            inner = radius**2


def test_transform_is_label_and_pose_blind_and_keeps_original_returns():
    source = scan_fixture()
    transform = ScanTransform(load_config())
    scan = transform(source)
    other_pose = np.eye(4)
    other_pose[:3, 3] = [20, 10, -3]
    unlabelled = make_source_frame(8, source.xyzi.copy(), other_pose, partition="fixture", sequence_id=9)
    blind = transform(unlabelled)
    for key in scan:
        torch.testing.assert_close(scan[key], blind[key], rtol=0, atol=0)
    np.testing.assert_array_equal(scan["source_slot"], source.real_slots)
    assert len(scan["xyzi"]) == source.real_count == 8
    # Duplicates keep separate output rows but do not count as multiple support positions.
    assert scan["geometry_inverse"][1] == scan["geometry_inverse"][2]
    assert scan["voxel_inverse"][0] == scan["voxel_inverse"][1]
    assert not torch.equal(scan["point_offset"][0], scan["point_offset"][1])
    assert (scan["neighbors"][scan["geometry_inverse"][6:]] == -1).all()
    assert (scan["condition"][6:, -1] == 0).all()
    torch.testing.assert_close(scan["basis"].transpose(1, 2) @ scan["basis"],
                               torch.eye(3).expand(8, 3, 3), rtol=1e-6, atol=1e-6)
    assert all(torch.isfinite(value).all() for value in scan.values())
    assert (scan["sensing_scale"] > 0).all()


def test_relationship_chunks_null_support_and_gradients_match():
    torch.manual_seed(2)
    scan = ScanTransform(load_config())(scan_fixture())
    layer = RelationLayer().train()
    z = torch.randn(8, 64, requires_grad=True)
    h = torch.randn(8, 64, requires_grad=True)
    rows = torch.arange(8)
    expected = layer(z, h, scan, rows, conditioned=True, chunk=8, recompute=False)
    expected.sum().backward()
    gradients = [p.grad.clone() for p in [z, h, *layer.parameters()]]
    for p in [z, h, *layer.parameters()]:
        p.grad = None
    actual = layer(z, h, scan, rows, conditioned=True, chunk=3, recompute=True)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for p, gradient in zip([z, h, *layer.parameters()], gradients, strict=True):
        torch.testing.assert_close(p.grad, gradient, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(actual).all()
    # Isolated points continue through the residual update and have a task gradient.
    assert torch.count_nonzero(z.grad[6:]) > 0


def test_plain_relations_have_no_explicit_sensing_input():
    torch.manual_seed(11)
    scan = ScanTransform(load_config())(scan_fixture())
    layer, z, h = RelationLayer().eval(), torch.randn(8, 64), torch.randn(8, 64)
    query = torch.arange(8)
    changed = dict(scan, condition=scan["condition"] + 5, sensing_scale=scan["sensing_scale"] * 3)
    torch.testing.assert_close(layer.part(z, h, scan, query, False),
                               layer.part(z, h, changed, query, False), rtol=0, atol=0)
    assert not torch.allclose(layer.part(z, h, scan, query, True), layer.part(z, h, changed, query, True))


def test_original_points_in_one_voxel_can_receive_different_scores(monkeypatch):
    from types import SimpleNamespace
    from src.model import AJAE
    import vendor.litept.model as backbone

    class FixedContext(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, point):
            return SimpleNamespace(feat=torch.zeros(len(point["feat"]), 72))

    monkeypatch.setattr(backbone, "LitePT", FixedContext)
    config = load_config()
    scan = ScanTransform(config)(scan_fixture())
    initial = None
    for mode in ("none", "plain", "conditioned"):
        config["model"]["relation_mode"] = mode
        torch.manual_seed(12)
        model = AJAE(config).eval()
        if initial is None:
            initial = deepcopy(model.state_dict())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        scores = model(scan)
        assert len(scores) == 8 and torch.isfinite(scores).all()
        assert scan["voxel_inverse"][0] == scan["voxel_inverse"][1]
        assert scores[0] != scores[1]


def test_pointrope_matches_upstream_table_and_preserves_shared_inputs():
    torch.manual_seed(5)
    tokens = torch.randn(2, 3, 7, 42, requires_grad=True)
    positions = torch.randint(0, 300, (2, 7, 3))
    before = tokens.detach().clone()
    actual = PointROPE()(tokens, positions)
    d = 14
    freq = 1 / 100 ** (torch.arange(0, d, 2).float() / d)
    table = torch.arange(300).float()[:, None] * freq
    phase = torch.cat((table, table), -1)
    expected = []
    for axis, part in enumerate(tokens.chunk(3, -1)):
        x, y = part.chunk(2, -1)
        angle = phase[positions[..., axis]][:, None]
        expected.append(part * angle.cos() + torch.cat((-y, x), -1) * angle.sin())
    expected = torch.cat(expected, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.autograd.grad(actual.square().sum() + tokens.sum(), tokens)[0]
    torch.testing.assert_close(grad, 2 * tokens + 1, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(tokens, before, rtol=0, atol=0)


def test_normal_pairs_exclude_replacements_ignored_points_and_occlusions():
    original = scan_fixture()
    xyzi = original.xyzi.copy()
    packed = original.labels.packed.copy()
    semantic_target = original.labels.semantic_target.copy()
    xyzi[1] = [9.9, 0, 0, .7]
    packed[1] = 2 + (60001 << 16)
    xyzi[5] = 0
    packed[5] = 0
    semantic_target[[1, 5]] = 255
    labels = PointLabels(packed, (packed & 65535).astype(np.uint16), (packed >> 16).astype(np.uint16), semantic_target)
    inserted = np.arange(len(xyzi)) == 1
    occluded = np.isin(np.arange(len(xyzi)), [1, 5])
    after = make_source_frame(3, xyzi, np.eye(4), labels, partition="fixture", sequence_id=1)
    frozen = FrozenFrame(after, "a" * 64, inserted, occluded)
    queries = query_rows(frozen, original, load_config()["training"], np.random.default_rng(1))
    assert set(queries["keep_slot"]) == {2, 3, 6, 7, 8}
    assert set(queries["target"].tolist()) == {0, 1}
    np.testing.assert_array_equal(after.real_slots[queries["query"][queries["keep_index"]]], queries["keep_slot"])
    np.testing.assert_array_equal(original.real_slots[queries["original_query"]], queries["keep_slot"])


def test_task_loss_empty_classes_pair_control_and_cross_frame_ranking():
    scores = torch.tensor([-80., 0., 80., 2.], requires_grad=True)
    target = torch.tensor([0, 0, 0, -1])
    loss = detection_loss(scores, target)
    torch.testing.assert_close(loss, .5 * F.softplus(scores[:3]).mean())
    loss.backward()
    assert scores.grad[-1] == 0 and torch.isfinite(scores.grad).all()
    a = torch.tensor([2., -3.], requires_grad=True)
    b = torch.tensor([-1., 1.], requires_grad=True)
    keep_loss(a, b).backward()
    assert a.grad[0] > 0 and a.grad[1] == 0 and b.grad[0] == 0 and b.grad[1] > 0
    torch.testing.assert_close(keep_loss(a, b, "mean"), .5 * (F.softplus(a).mean() + F.softplus(b).mean()))
    s = torch.tensor([1., -2., 3., -4.], requires_grad=True)
    y, frames = torch.tensor([0, 0, 1, 1]), torch.tensor([1, 2, 1, 2])
    ranking, counts = tail_loss(s, y, frames, load_config()["loss"], torch.Generator().manual_seed(7))
    ranking.backward()
    assert (s.grad[y == 0] > 0).all() and (s.grad[y == 1] < 0).all()
    assert counts["cross_frame_pairs"] > 0
    empty, counts = tail_loss(s, torch.zeros(4), frames, load_config()["loss"], torch.Generator())
    assert empty == 0 and counts["pairs"] == 0


def test_scan_recomputation_preserves_gradient_and_batchnorm_updates():
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.GELU(), nn.Linear(8, 1))

        def forward(self, scan, query):
            return self.net(scan["x"])[query]

    torch.manual_seed(7)
    model = SmallModel().train()
    reference = deepcopy(model)
    rows = [(dict(x=torch.randn(12, 4)), torch.tensor([1, 4, 9])) for _ in range(4)]
    expected = torch.cat([reference(x, q) for x, q in rows])
    actual = torch.cat([training_forward(model, x, q) for x, q in rows])
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    for p, q in zip(model.buffers(), reference.buffers(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_scan_draws_queries_and_resume_do_not_change_with_loss_switches():
    config = load_config()
    probabilities = np.array([.2, .3, .5])
    original = list(Requests(probabilities, config, 250))
    off = deepcopy(config)
    off["loss"]["keep_weight"] = off["loss"]["tail_weight"] = 0
    assert [r[:2] for r in original] == [r[:2] for r in Requests(probabilities, off, 250)]
    assert list(Requests(probabilities, config, 250, 150)) == original[300:]
    assert auxiliary_fraction(0, config["training"]) == 0
    assert auxiliary_fraction(200, config["training"]) == 1
