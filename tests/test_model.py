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
