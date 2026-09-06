from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.data import FrozenWindowDataset, PredictionBatch, WindowPartition
from src.model import AJAE, joint_voxelize
from src.protocol import FrameSpan, SequenceSpec, load_protocol
from src.scene import PointLabels, SceneWindow, assemble_window, make_source_frame


def test_nre_support_formula_bounded_correction_and_ignore_gradients():
    from types import SimpleNamespace
    from src.model import NREHead, NREEvidence
    from src.train import nre_loss

    torch.manual_seed(23)
    head = NREHead((0, 8, 19))
    shallow = torch.randn(3, 36, requires_grad=True)
    deep = torch.randn(3, 72, requires_grad=True)
    inverse, detail = torch.tensor([0, 0, 1, 2]), torch.randn(4, 9)
    evidence = NREEvidence(*head(shallow, deep, inverse, detail))
    h = (
        head.fusion(torch.cat((shallow[inverse], deep[inverse], detail), 1))
        .detach()
        .numpy()
    )
    mu = head.prototypes.detach().numpy()
    q = (h / np.linalg.norm(h, axis=1, keepdims=True)) @ (
        mu / np.linalg.norm(mu, axis=2, keepdims=True)
    ).reshape(-1, 64).T
    expected = -np.log(np.exp((q.astype(np.float64) - 0.5) / 0.07).mean(axis=1))
    np.testing.assert_allclose(evidence.support.detach(), expected, atol=2e-6)
    assert torch.equal(evidence.score, evidence.support)
    for x in (evidence.score, evidence.support, evidence.normal_logits):
        x.retain_grad()
    target, groups = torch.tensor([0, 1, -1, 0]), torch.tensor([0, -1, -1, 19])
    statistics = dict(pi_normal=1, pi_anomaly=0.5, class_weights=[1, 2, 3])
    loss, parts, _ = nre_loss(
        evidence, target, groups, SimpleNamespace(head=head), statistics, 999
    )
    normal_mean = F.softplus(evidence.score[[0, 3]]).mean()
    anomaly_mean = F.softplus(-evidence.score[1])
    torch.testing.assert_close(parts["detection"], 0.5 * normal_mean + anomaly_mean)
    logp = evidence.normal_logits.log_softmax(1)
    expected_sem = -(logp[0, 0] + 3 * logp[3, 2]) / (4 * np.log(3))
    torch.testing.assert_close(parts["semantic"], expected_sem)
    loss.backward()
    assert evidence.score.grad[2] == evidence.support.grad[2] == 0
    assert torch.count_nonzero(evidence.normal_logits.grad[2]) == 0
    for p in (
        shallow,
        deep,
        head.prototypes,
        head.acceptance,
        head.fusion[0].weight,
        head.correction[-1].weight,
    ):
        assert p.grad is not None and p.grad.abs().sum() > 0
    with torch.no_grad():
        for bias in (-50, 50):
            head.correction[-1].bias.fill_(bias)
            values = NREEvidence(*head(shallow, deep, inverse, detail))
            assert torch.all((values.score - values.support).abs() <= 2 + 1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LitePT requires CUDA")
def test_nre_shallow_features_preserve_backbone_rows_and_legacy_output():
    model = AJAE().cuda().eval()
    inputs = joint_voxelize(_window(512), device="cuda")
    captured = []
    hook = model.backbone.enc[0].register_forward_hook(
        lambda _m, _a, point: captured.append(point.feat.detach().clone())
    )
    with torch.inference_mode():
        torch.manual_seed(23)
        original = model.backbone(inputs.backbone_input())
        torch.manual_seed(23)
        decoded, shallow = model.backbone(inputs.backbone_input(), return_shallow=True)
    hook.remove()
    assert torch.equal(original.feat, decoded.feat)
    assert torch.equal(decoded.grid_coord, inputs.grid_coord)
    assert torch.equal(shallow, captured[0]) and torch.equal(shallow, captured[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LitePT requires CUDA")
@pytest.mark.parametrize("large_branch", ["proj", "proj_skip"])
def test_decoder_projection_preserves_finite_batchnorm_under_autocast(large_branch):
    import copy
    import spconv.pytorch as spconv
    from vendor.litept.litept.model import GridUnpooling, Point

    layer = GridUnpooling(1, 1, 1, norm_layer=torch.nn.BatchNorm1d).cuda().train()
    with torch.no_grad():
        for branch in (layer.proj, layer.proj_skip):
            branch[0].weight.fill_(2)
            branch[0].bias.zero_()
    reference = copy.deepcopy(layer)
    large = torch.arange(40000, 60000, 5000, device="cuda").float()[:, None]
    small = torch.arange(1, 5, device="cuda").float()[:, None]

    def point():
        deep, skip = (large, small) if large_branch == "proj" else (small, large)
        parent = Point(feat=skip.clone().requires_grad_())
        indices = torch.zeros((4, 4), dtype=torch.int32, device="cuda")
        indices[:, 1] = torch.arange(4, device="cuda")
        parent.sparse_conv_feat = spconv.SparseConvTensor(
            parent.feat, indices, [4, 1, 1], batch_size=1
        )
        return Point(
            feat=deep.clone().requires_grad_(),
            pooling_parent=parent,
            pooling_inverse=torch.arange(3, -1, -1, device="cuda"),
        )

    with torch.autocast("cuda", dtype=torch.float16):
        # The old half-precision projection overflows before normalization.
        assert not torch.isfinite(getattr(layer, large_branch)[0](large)).all()
        actual = layer(point()).feat
    expected = reference(point()).feat
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (actual[:, 0] * torch.arange(1, 5, device="cuda")).sum().backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in layer.parameters()
    )
    assert all(b.isfinite().all() for b in layer.buffers())


def _window(count: int = 8, *, start: int = 0, labels: bool = True) -> SceneWindow:
    rng = np.random.default_rng(19)
    sources = []
    for scan in range(5):
        xyzi = rng.uniform((-8, -8, -2, 0), (8, 8, 2, 1), (count, 4)).astype(np.float32)
        xyzi[:3, :3] = (1.011 + scan * 0.001, 0.012, 0.013)
        if scan == 2:
            xyzi[:3, 0] += 0.1
        xyzi[3, :3] = (-0.01, 0.012, 0.013)
        semantic = np.full(count, 40, dtype=np.uint16)
        semantic[1:3] = (2, 0)
        truth = (
            PointLabels(
                semantic.astype(np.uint32), semantic, np.zeros(count, dtype=np.uint16)
            )
            if labels
            else None
        )
        sources.append(
            make_source_frame(
                start + scan,
                xyzi,
                np.eye(4, dtype=np.float64),
                truth,
                partition="train",
                sequence_id=206,
            )
        )
    return assemble_window(
        SequenceSpec("train", 206, "fixture", True, FrameSpan(start, start + 5)),
        start,
        tuple(range(start, start + 5)),
        sources,
        observation_sequence_id=f"fixture/{start}",
    )


def test_joint_voxels_match_independent_all_point_reference() -> None:
    window = _window()
    before = window.points.coordinates.copy()
    inputs = joint_voxelize(window)
    transferred = inputs.to("cuda" if torch.cuda.is_available() else "cpu").to("cpu")
    assert transferred.source_points is window.points
    for field in fields(inputs):
        value = getattr(inputs, field.name)
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(transferred, field.name))
    assert inputs.features.shape[1] == inputs.point_features.shape[1] == 9
    assert inputs.point_to_voxel.shape == (window.points.count,)
    assert torch.count_nonzero(inputs.backbone_input()["batch"]) == 0
    assert np.array_equal(before, window.points.coordinates)

    cells = np.floor(before.astype(np.float64) / 0.05).astype(np.int64)
    groups: dict[tuple[int, ...], list[int]] = {}
    for row, cell in enumerate(cells):
        groups.setdefault(tuple(cell), []).append(row)
    assert len(inputs.features) == len(groups)
    for cell, rows in groups.items():
        voxel = int(inputs.point_to_voxel[rows[0]])
        assert torch.all(inputs.point_to_voxel[rows] == voxel)
        expected = np.zeros(9, dtype=np.float32)
        expected[:3] = before[rows].mean(axis=0, dtype=np.float64)
        expected[3] = window.points.features[rows, 0].mean(dtype=np.float64)
        expected[4 + window.points.scan_group[rows]] = 1
        np.testing.assert_allclose(
            inputs.features[voxel], expected, atol=1e-6, rtol=1e-5
        )
        np.testing.assert_array_equal(
            inputs.grid_coord[voxel], cell - cells.min(axis=0)
        )
    shared = int(inputs.point_to_voxel[0])
    np.testing.assert_array_equal(inputs.features[shared, 4:], (1, 1, 0, 1, 1))
    np.testing.assert_array_equal(inputs.point_features[:, 3:4], window.points.features)
    np.testing.assert_array_equal(
        inputs.point_features[:, 4:].argmax(dim=1), window.points.scan_group
    )
    np.testing.assert_allclose(
        inputs.point_features[:, :3],
        (before - inputs.coordinates[inputs.point_to_voxel].numpy()) / 0.05,
        atol=1e-6,
        rtol=1e-5,
    )


def test_scan_intensity_permutation_is_label_free_and_updates_both_inputs():
    from collections import OrderedDict
    from src.intensity import permute_window, scan_permutation

    window = _window(40)
    inputs = joint_voxelize(window)
    sources = [member.source for member in window.frames]
    last = sources[-1]
    next_source = make_source_frame(
        5, last.xyzi, last.lidar_pose, last.labels, partition="train", sequence_id=206
    )
    overlapping = assemble_window(
        SequenceSpec("train", 206, "fixture", True, FrameSpan(0, 6)),
        1,
        (1, 2, 3, 4, 5),
        [*sources[1:], next_source],
        observation_sequence_id=window.observation_sequence_id,
    )
    cache = OrderedDict()
    for seed in (0, 1, 2):
        changed, prepared = permute_window(window, inputs, seed, cache)
        reference = joint_voxelize(changed)
        for field in fields(inputs):
            actual = getattr(prepared, field.name)
            if isinstance(actual, torch.Tensor):
                assert torch.equal(actual, getattr(reference, field.name))
        for name in ("coordinates", "grid_coord", "point_to_voxel"):
            assert torch.equal(getattr(inputs, name), getattr(prepared, name))
        assert torch.equal(inputs.features[:, 4:], prepared.features[:, 4:])
        assert torch.equal(inputs.point_features[:, :3], prepared.point_features[:, :3])
        assert changed.labels is window.labels
        assert not torch.equal(
            inputs.point_features[:, 3], prepared.point_features[:, 3]
        )
        for original, shuffled in zip(window.frames, changed.frames, strict=True):
            source = original.source
            slots = source.real_slots
            bins = np.floor(
                np.linalg.norm(source.xyzi[slots, :3], axis=1).astype(np.float64) / 2.5
            )
            for group in np.unique(bins):
                np.testing.assert_array_equal(
                    np.sort(source.xyzi[slots[bins == group], 3]),
                    np.sort(shuffled.source.xyzi[slots[bins == group], 3]),
                )
            first = scan_permutation(
                source.xyzi,
                slots,
                window.observation_sequence_id,
                source.frame_id,
                seed,
            )
            np.testing.assert_array_equal(
                first,
                scan_permutation(
                    source.xyzi,
                    slots,
                    window.observation_sequence_id,
                    source.frame_id,
                    seed,
                ),
            )
        repeated, _ = permute_window(window, inputs, seed, OrderedDict())
        np.testing.assert_array_equal(repeated.points.features, changed.points.features)
        shifted, _ = permute_window(
            overlapping, joint_voxelize(overlapping), seed, OrderedDict()
        )
        for a, b in zip(changed.frames[1:], shifted.frames[:4], strict=True):
            np.testing.assert_array_equal(a.source.xyzi, b.source.xyzi)


def test_labels_and_absolute_identities_never_enter_features() -> None:
    labelled = _window()
    unlabelled = _window(start=70, labels=False)
    a, b = joint_voxelize(labelled), joint_voxelize(unlabelled)
    for field in fields(a):
        value = getattr(a, field.name)
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(b, field.name))
    # Normal, anomaly and ignore coexist inside a single voxel without voting.
    assert a.point_to_voxel[:3].unique().numel() == 1
    np.testing.assert_array_equal(labelled.labels.anomaly_target[:3], (0, 1, -1))


@pytest.mark.parametrize("voxel_size", [0, -0.05, float("nan"), float("inf")])
def test_invalid_voxel_size_is_rejected(voxel_size: float) -> None:
    with pytest.raises(ValueError, match="voxel_size"):
        joint_voxelize(_window(), voxel_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LitePT requires CUDA")
def test_official_backbone_and_all_point_head_forward_backward(tmp_path: Path) -> None:
    torch.manual_seed(23)
    model = AJAE().cuda().train()
    assert model.backbone.num_stages == 5
    assert model.backbone.enc_conv == (True, True, True, False, False)
    assert model.backbone.enc_attn == (False, False, False, True, True)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.head[0].in_features == 81 and model.head[0].out_features == 32
    window = _window(512)
    inputs = joint_voxelize(window, device="cuda")
    model.eval()
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(91)
        uncached = model(window)
        torch.manual_seed(91)
        cached = model(window, inputs=inputs)
        torch.testing.assert_close(cached, uncached, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="different points"):
        model(_window(512), inputs=inputs)
    model.train()
    logits = model(window)
    assert logits.shape == (window.points.count,)
    assert torch.isfinite(logits).all()
    logits.retain_grad()
    target = torch.tensor(window.labels.anomaly_target, device="cuda")
    valid = target != -1
    F.binary_cross_entropy_with_logits(logits[valid], target[valid].float()).backward()
    assert torch.all(logits.grad[~valid] == 0)
    assert torch.all(logits.grad[valid] != 0)
    for module in (model.backbone, model.head):
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) > 0 for g in gradients)

    # Shared voxel context need not imply identical point scores.
    inputs = joint_voxelize(window, device="cuda")
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
        model.head[0].weight[0, 75] = 1  # Individual intensity, not voxel mean.
        model.head[2].weight[0, 0] = 1
        point_input = torch.cat(
            (torch.zeros((3, 72), device="cuda"), inputs.point_features[:3]), dim=1
        )
        assert model.head(point_input).unique().numel() == 3
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        record = model.predict(window)
    assert record.anomaly_score.shape == (window.points.count,)
    np.testing.assert_array_equal(record.online_mask, window.current_mask)
    path = tmp_path / "prediction.npz"
    metadata = record.save(path, window=window)
    restored = PredictionBatch.load(
        path, window=window, expected_sha256=metadata["file_sha256"]
    )
    np.testing.assert_array_equal(restored.anomaly_score, record.anomaly_score)


@pytest.mark.skipif(
    not os.environ.get("AJAE_STU_ROOT") or not torch.cuda.is_available(),
    reason="set AJAE_STU_ROOT to run full frozen-window GPU integration",
)
def test_complete_frozen_windows_on_gpu() -> None:
    protocol = load_protocol()
    root = Path(os.environ["AJAE_STU_ROOT"])
    torch.manual_seed(23)
    model = AJAE().cuda()
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    voxel_counts = []
    model.backbone.register_forward_pre_hook(
        lambda module, args: voxel_counts.append(len(args[0]["coord"]))
    )
    for pool_name in ("train", "validation"):
        dataset = FrozenWindowDataset(root, protocol, pool_name=pool_name)
        assert dataset.gradient_updates_allowed == (pool_name == "train")
        windows = [(pool_name, dataset[0])]
        if pool_name == "validation":
            windows.append(
                (
                    "normal_201",
                    next(iter(WindowPartition(dataset.source_sequence, 4, 4))),
                )
            )
        for name, window in windows:
            training = name == "train"
            model.train(training)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            with (
                torch.set_grad_enabled(training),
                torch.autocast("cuda", dtype=torch.float16),
            ):
                logits = model(window)
                assert logits.shape == (window.points.count,)
                assert torch.isfinite(logits).all()
                if training:
                    target = torch.tensor(window.labels.anomaly_target, device="cuda")
                    valid = target != -1
                    loss = F.binary_cross_entropy_with_logits(
                        logits[valid], target[valid].float()
                    )
            if training:
                scaler.scale(loss).backward()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                for stage in (*model.backbone.enc.children(), model.head):
                    assert any(
                        torch.count_nonzero(p.grad) > 0 for p in stage.parameters()
                    )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            scores = torch.sigmoid(logits.detach().float()).cpu().numpy()
            record = PredictionBatch.from_window(window, scores)
            assert record.online_mask.sum() == window.current_mask.sum()
            print(
                {
                    "view": name,
                    "points": window.points.count,
                    "voxels": voxel_counts[-1],
                    "current_points": int(window.current_mask.sum()),
                    "backward": training,
                    "seconds": round(elapsed, 3),
                    "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
                }
            )
            del logits, record
            if training:
                del loss


@pytest.mark.skipif(
    not os.environ.get("AJAE_STU_ROOT") or not torch.cuda.is_available(),
    reason="set AJAE_STU_ROOT for the frozen streaming-input equivalence check",
)
def test_streamed_frozen_input_matches_original_preparation() -> None:
    from src.evaluate import prepare_window
    from src.train import fixed_check, prepare_samples, training_samples

    protocol = load_protocol()
    root = Path(os.environ["AJAE_STU_ROOT"])
    dataset = FrozenWindowDataset(
        root, protocol, pool_name="train", segment_cache_bytes=256 * 2**20
    )
    selected = training_samples(protocol.training_pool, full=True)
    model = AJAE().cuda().eval()
    payload = torch.load(
        "runs/train/initial.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model"], strict=True)
    for sample in (selected[0], selected[384], selected[0]):
        reference, expected, target = prepare_samples(root, protocol, [sample], 1)[0]
        window, actual, _, _ = prepare_window(dataset, None, sample)
        cells = np.floor(window.points.coordinates.astype(np.float64) / 0.05)
        cells -= cells.min(axis=0)
        grid, inverse, counts = np.unique(
            cells.astype(np.int64), axis=0, return_inverse=True, return_counts=True
        )
        np.testing.assert_array_equal(actual.grid_coord.numpy(), grid)
        np.testing.assert_array_equal(actual.point_to_voxel.numpy(), inverse)
        np.testing.assert_array_equal(
            np.bincount(actual.point_to_voxel.numpy()), counts
        )
        for axis in range(4):
            values = (
                window.points.coordinates[:, axis]
                if axis < 3
                else window.points.features[:, 0]
            )
            means = (
                np.bincount(inverse, weights=values, minlength=len(grid)) / counts
            ).astype(np.float32)
            np.testing.assert_array_equal(actual.features[:, axis].numpy(), means)
        for name in (
            "coordinates",
            "features",
            "source_frame",
            "source_slot",
            "scan_group",
        ):
            np.testing.assert_array_equal(
                getattr(window.points, name), getattr(reference.points, name)
            )
        np.testing.assert_array_equal(window.labels.anomaly_target, target.numpy())
        assert dataset._cached_bytes <= 256 * 2**20
        assert (
            sum(len(segment._frame_cache) for segment, _ in dataset._segments.values())
            <= 5
        )
        for field in fields(actual):
            value = getattr(actual, field.name)
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    value, getattr(expected, field.name), atol=0, rtol=0
                )
        with fixed_check(model, 23):
            direct = model(window).cpu()
        with fixed_check(model, 23):
            streamed = model(window, inputs=actual.to("cuda")).cpu()
        torch.testing.assert_close(streamed, direct, atol=1e-6, rtol=1e-5)
        print(
            {
                "current_frame": window.current_frame_id,
                "points": window.points.count,
                "max_absolute_logit_difference": float((streamed - direct).abs().max()),
            }
        )


def test_rope_prefix_cache_matches_exact_tables_without_accumulating_lengths():
    from vendor.litept.libs.pointrope.pointrope_torch import PointROPE

    rope = PointROPE(freq=100.0)
    for length in (17, 257, 31, 513, 200):
        inv = 1.0 / (100.0 ** (torch.arange(0, 4, 2).float() / 4))
        phase = torch.einsum("i,j->ij", torch.arange(length).float(), inv)
        phase = torch.cat((phase, phase), dim=-1)
        cos, sin = rope.get_cos_sin(4, length, torch.device("cpu"), torch.float32)
        assert torch.equal(cos, phase.cos()) and torch.equal(sin, phase.sin())
        assert len(rope.cache) == 1
        assert next(iter(rope.cache.values()))[0].shape[0] <= 2 * max(513, length)
