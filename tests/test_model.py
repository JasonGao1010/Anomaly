from copy import deepcopy
from io import BytesIO

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


def test_historical_cells_keep_duplicates_counts_moments_and_all_two_meter_support():
    from src.model import support_cells
    rng = np.random.default_rng(31)
    xyz = np.r_[np.array([[10.399999, 0., 0.]]), [10.4, 0., 0.] + rng.uniform(-1.9, 1.9, (80, 3))]
    xyz = np.r_[xyz, xyz[[0]]]
    xyzi = np.column_stack((xyz, np.linspace(.1, 1.1, len(xyz)))).astype(np.float32)
    _, first = np.unique(xyzi[:, :3], axis=0, return_index=True)
    cells = support_cells(xyzi, [.05, .2, .8], first)
    for scale in range(3):
        assert cells[f"cell_{scale}_count"].sum() == len(xyzi)
        assert cells[f"cell_{scale}_positions"].sum() == len(first)
        for i in range(len(cells[f"cell_{scale}_count"])):
            points = xyzi[cells[f"cell_{scale}_inverse"] == i]
            np.testing.assert_allclose(cells[f"cell_{scale}_centroid"][i], points[:, :3].astype(np.float64).mean(0), atol=1e-6)
            np.testing.assert_allclose(cells[f"cell_{scale}_statistics"][i, -2:],
                [points[:, 3].astype(np.float64).mean(), points[:, 3].astype(np.float64).var()], atol=1e-7)
    for i, point in enumerate(xyzi[:, :3].astype(np.float64)):
        inside = np.linalg.norm(xyzi[:, :3].astype(np.float64) - point, axis=1) <= 2
        own = cells["cell_2_inverse"][i]
        assert set(cells["cell_2_inverse"][inside]) <= set(cells["cell_2_adjacent"][own])


def test_historical_relation_has_uncapped_gradient_path_and_zero_value_null():
    from src.model import MultiscaleRelation
    config = load_config()
    config["model"].update(relation_mode="multiscale", cell_sizes_m=[.05, .2, .8])
    source = scan_fixture()
    scan = ScanTransform(config)(source)
    z = torch.randn(len(source.real_slots), 64, requires_grad=True)
    h = torch.randn_like(z)
    layer = MultiscaleRelation()
    query = torch.tensor([1, 3])
    full = layer(z, h, scan, torch.arange(len(z)), chunk=3, recompute=True)
    part = layer(z, h, scan, query, chunk=1, recompute=False)
    torch.testing.assert_close(part, full[query], rtol=2e-5, atol=2e-5)
    part.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)
    unsupported = {k: v.clone() for k, v in scan.items()}
    for scale in range(3): unsupported[f"cell_{scale}_adjacent"].fill_(-1)
    evidence = []
    out = layer(z.detach(), h, unsupported, query, chunk=3, recompute=False,
                trace=lambda name, value: evidence.append(value) if name.endswith("evidence") else None)
    assert torch.isfinite(out).all() and all(torch.count_nonzero(value) == 0 for value in evidence)
    # More than16 distinct nearby returns all enter cell means, including a late return.
    xyz = np.column_stack((10 + np.arange(40) * .002, np.zeros(40), np.zeros(40), np.ones(40))).astype(np.float32)
    packed = np.full(40, 40, np.uint32)
    labels = PointLabels(packed, packed.astype(np.uint16), np.zeros(40, np.uint16), np.zeros(40, np.uint8))
    dense = make_source_frame(1, xyz, np.eye(4), labels, partition="train", sequence_id=206)
    scan = ScanTransform(config)(dense)
    z = torch.randn(40, 64, requires_grad=True)
    layer(z, torch.zeros_like(z), scan, torch.tensor([0]), chunk=1, recompute=False).square().sum().backward()
    assert z.grad[39].norm() > 0


def test_v3_inherits_only_compatible_encoders():
    from src.model import transfer_parent, _mlp
    parent, model = nn.Module(), nn.Module()
    for name in ("backbone", "context", "point"):
        setattr(parent, name, nn.Linear(2, 2))
        setattr(model, name, nn.Linear(2, 2))
    parent.base_head = _mlp(136, 128, 1)
    model.head = _mlp(208, 128, 1)
    before = deepcopy(model.head.state_dict())
    model.config = dict(relation_mode="pyramid")
    info = transfer_parent(model, dict(step=1152, model=parent.state_dict()))
    torch.testing.assert_close(model.head.state_dict(), before, atol=0, rtol=0)
    for name in ("backbone", "context", "point"):
        torch.testing.assert_close(getattr(model, name).state_dict(), getattr(parent, name).state_dict(), atol=0, rtol=0)
    assert not info["score_head_inherited"] and not info["final_parent_score_preserved"]


@pytest.mark.parametrize("frame", [0, 1, 2, 3])
def test_v3_verified_copy_mapping_is_label_blind_and_preserves_distinct_returns(frame):
    from src.render import DUPLICATE_201_RAY_LAYOUT
    from src.train import experiment_config, load_experiment
    config = experiment_config(load_experiment("protocol/v3.json"))
    template = np.zeros((131072, 4), np.float32)
    template[:3] = [[10, 0, 0, .1], [10.02, 0, 0, .2], [10.02, 0, 0, .9]]
    runs = DUPLICATE_201_RAY_LAYOUT[frame][1]
    xyzi = np.concatenate([template[start:start + length] for _, length, start in runs])
    source = make_source_frame(frame, xyzi, np.eye(4), partition="train", sequence_id=201)
    transform = ScanTransform(config)
    scan = transform(source)
    assert len(scan["xyzi"]) == 3
    np.testing.assert_array_equal(scan["xyzi"][scan["record_inverse"]], xyzi[source.real_slots])
    assert scan["voxel_count"].sum() == 3 and scan["voxel_positions"].sum() == 2
    assert set(scan) == {"xyzi", "source_slot", "point_offset", "voxel_inverse", "voxel_xyzi", "grid_coord",
                         "record_inverse", "voxel_count", "voxel_positions", "condition"}
    packed = np.arange(len(xyzi), dtype=np.uint32) % 53
    labels = PointLabels(packed, packed.astype(np.uint16), np.zeros(len(xyzi), np.uint16), np.zeros(len(xyzi), np.uint8))
    labeled = make_source_frame(frame, xyzi, np.eye(4), labels, partition="train", sequence_id=201)
    torch.testing.assert_close(scan, transform(labeled), atol=0, rtol=0)
    xyzi[-131072, 3] += 1
    changed = make_source_frame(frame, xyzi, np.eye(4), partition="train", sequence_id=201)
    assert len(transform(changed)["xyzi"]) == changed.real_count
    ordinary = make_source_frame(frame, source.xyzi, np.eye(4), partition="train", sequence_id=206)
    assert len(transform(ordinary)["xyzi"]) == source.real_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LitePT requires CUDA sparse convolution")
def test_v3_backbone_levels_point_paths_chunks_and_adaptation_gradients():
    from src.model import AJAE, to_device
    from src.train import experiment_config, load_experiment, set_trainable, parameter_family, _Numerics
    config = experiment_config(load_experiment("protocol/v3.json"))
    torch.manual_seed(53)
    model = AJAE(config).cuda().eval()
    scan = to_device(ScanTransform(config)(scan_fixture()), "cuda")
    voxels = scan["voxel_xyzi"]
    inputs = dict(feat=voxels, coord=voxels[:, :3], grid_coord=scan["grid_coord"],
                  offset=torch.tensor([len(voxels)], device="cuda"))
    with torch.no_grad():
        plain = model.backbone(inputs).feat
        output, levels = model.backbone(inputs, embedding_residual=voxels.new_zeros((len(voxels), 36)), return_pyramid=True)
        torch.testing.assert_close(plain, output.feat, atol=0, rtol=0)
        assert [level["feat"].shape[1] for level in levels] == [72, 72, 144, 252, 504]
        index = scan["voxel_inverse"]
        for level in levels:
            if "pooling_inverse" in level:
                index = level["pooling_inverse"][index]
            torch.testing.assert_close(level["grid_coord"][index],
                (scan["grid_coord"][scan["voxel_inverse"]] // level["stride"]).to(level["grid_coord"].dtype), atol=0, rtol=0)
        full = model(scan)
        query = torch.tensor([3, 0, 5, 0], device="cuda")
        model.config["query_chunk"] = 1
        torch.testing.assert_close(model(scan, query), full[query], atol=2e-5, rtol=2e-5)
        assert full[0] != full[1] and full[1] != full[2]
    model.train()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for step in (128, 129):
        set_trainable(model, config, step)
        model.zero_grad(set_to_none=True)
        with _Numerics(model) as trace:
            training_forward(model, scan, query, trace=trace).square().mean().backward()
        assert "head.output" in trace.summary()
        assert model.detail[-1].weight.grad.norm() > 0
        assert torch.count_nonzero(model.detail[0].weight.grad) == 0
        for name, parameter in model.named_parameters():
            if step == 128 and parameter_family(name) != "new_modules":
                assert parameter.grad is None
            elif parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        if step == 129:
            assert model.backbone.embedding.stem.conv.weight.grad.norm() > 0
    torch.testing.assert_close(model.state_dict(), before, atol=0, rtol=0)
    # Nonzero detail must reach both the dense and sparse representations at stage zero.
    with torch.no_grad():
        model.detail[-1].bias.fill_(.01)
    def consistent(_module, args):
        torch.testing.assert_close(args[0].feat, args[0].sparse_conv_feat.features, atol=0, rtol=0)
    hook = model.backbone.enc.enc0.register_forward_pre_hook(consistent)
    try:
        with torch.no_grad():
            assert torch.isfinite(model(scan)).all()
    finally:
        hook.remove()


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


def test_saved_transform_is_independent_of_current_calibration(monkeypatch, tmp_path):
    import src.model as model_module

    config, source = load_config(), scan_fixture()
    transform = ScanTransform(config)
    expected = transform(source)
    stream = BytesIO()
    torch.save(transform.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    monkeypatch.setattr(model_module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "protocol").mkdir()
    (tmp_path / "protocol/data.json").write_text("{}")

    def forbidden_calibration(*args, **kwargs):
        raise AssertionError("restoration must not read current ray calibration")

    monkeypatch.setattr(model_module, "calibrated_ray_grid", forbidden_calibration)
    restored = ScanTransform(config, state=state)
    state["elevations"][0] += .01
    exported = restored.state_dict()
    exported["elevation_step"][0] *= 2
    for key, value in restored(source).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_saved_transform_rejects_invalid_calibration():
    config = load_config()
    state = ScanTransform(config).state_dict()
    invalid = [dict(state, format="unknown"), dict(state, elevations=state["elevations"].float()),
               dict(state, elevation_step=state["elevation_step"][:-1]),
               dict(state, azimuth_step=torch.tensor([.01], dtype=torch.float64))]
    for key, value in (("elevations", float("nan")), ("elevations", 2.), ("elevation_step", 0.)):
        altered = deepcopy(state)
        altered[key][0] = value
        invalid.append(altered)
    altered = deepcopy(state)
    altered["elevations"][1] = altered["elevations"][0]
    invalid.append(altered)
    invalid.append(dict(state, azimuth_step=torch.tensor(0., dtype=torch.float64)))
    for altered in invalid:
        with pytest.raises(ValueError):
            ScanTransform(config, state=altered)


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
    traced = {}
    def observe(name, value):
        traced.setdefault(name, []).append(value.detach().clone())
    for p in [z, h, *layer.parameters()]:
        p.grad = None
    observed = layer(z, h, scan, rows, conditioned=True, chunk=8, recompute=False, trace=observe)
    observed.sum().backward()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    for p, gradient in zip([z, h, *layer.parameters()], gradients, strict=True):
        torch.testing.assert_close(p.grad, gradient, rtol=0, atol=0)
    assert {"attention_scores", "residual", "output"} <= traced.keys()


def test_plain_relations_have_no_explicit_sensing_input():
    torch.manual_seed(11)
    scan = ScanTransform(load_config())(scan_fixture())
    layer, z, h = RelationLayer().eval(), torch.randn(8, 64), torch.randn(8, 64)
    query = torch.arange(8)
    changed = dict(scan, condition=scan["condition"] + 5, sensing_scale=scan["sensing_scale"] * 3)
    torch.testing.assert_close(layer.part(z, h, scan, query, False),
                               layer.part(z, h, changed, query, False), rtol=0, atol=0)
    assert not torch.allclose(layer.part(z, h, scan, query, True), layer.part(z, h, changed, query, True))


def test_modulation_control_preserves_coordinates_and_removes_only_direct_conditions():
    torch.manual_seed(13)
    scan = ScanTransform(load_config())(scan_fixture())
    layer, z, h = RelationLayer().eval(), torch.randn(8, 64), torch.randn(8, 64)
    query, seen = torch.arange(8), {}
    handles = [getattr(layer, name).register_forward_pre_hook(
        lambda module, args, name=name: seen.setdefault(name, []).append(args[0].detach().clone()))
        for name in ("edge", "modulation", "null")]
    on = layer.part(z, h, scan, query, True, True)
    off = layer.part(z, h, scan, query, True, False)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(seen["edge"][0], seen["edge"][1], rtol=0, atol=0)
    assert torch.count_nonzero(seen["modulation"][1]) == 0
    torch.testing.assert_close(seen["null"][0][:, :-8], seen["null"][1][:, :-8], rtol=0, atol=0)
    assert torch.count_nonzero(seen["null"][1][:, -8:]) == 0
    assert not torch.allclose(on, off)
    changed = dict(scan, condition=scan["condition"] + 5)
    torch.testing.assert_close(off, layer.part(z, h, changed, query, True, False), rtol=0, atol=0)
    assert not torch.allclose(on, layer.part(z, h, changed, query, True, True))
    # Sensing normalization remains observable when explicit modulation is disabled.
    changed = dict(scan, sensing_scale=scan["sensing_scale"] * 3)
    assert not torch.allclose(off, layer.part(z, h, changed, query, True, False))


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
    for mode, modulation in (("none", True), ("plain", True), ("conditioned", True), ("conditioned", False)):
        config["model"]["relation_mode"] = mode
        config["model"]["condition_modulation"] = modulation
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
        query = torch.tensor([0, 5])
        diagnostics = model(scan, query, return_features=True)
        torch.testing.assert_close(diagnostics["score"], model(scan, query), rtol=0, atol=0)
        torch.testing.assert_close(diagnostics["score"],
            diagnostics["base_score"] + diagnostics["relation_score"], rtol=0, atol=0)
        if mode == "none":
            assert torch.count_nonzero(diagnostics["relation_score"]) == 0
        components = model.predict(scan_fixture(), prepared=scan, components=True)
        np.testing.assert_array_equal(components["final"].restore(scan_fixture())[scan_fixture().real_slots],
                                      scores.detach().numpy())
        assert all(np.array_equal(value.source_slot, scan_fixture().real_slots) for value in components.values())
        torch.testing.assert_close(diagnostics["point"],
            model.point(torch.cat((scan["xyzi"], scan["point_offset"]), -1)), rtol=0, atol=0)
        shared = tuple(diagnostics[name] for name in ("context", "point"))
        gradients = torch.autograd.grad(diagnostics["score"].sum(), shared)
        for representation, gradient in zip(shared, gradients, strict=True):
            assert representation.shape == (8, 64) and gradient.shape == representation.shape
            assert torch.isfinite(gradient).all() and gradient.norm() > 0
        assert model(scan, query[:0]).shape == (0,)
        with pytest.raises(ValueError, match="nonempty"):
            model(scan, query[:0], return_features=True)
        if mode == "conditioned" and not modulation:
            changed = dict(scan, condition=scan["condition"] + 5)
            assert not torch.allclose(scores, model(changed))  # Heads retain the same conditions.


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
