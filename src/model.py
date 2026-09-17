"""V3: per-return detail, LitePT-S context, and one unbounded anomaly logit."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch_scatter

from vendor.litept.model import LitePT, PointROPEAttention


SEED = 20260917


def configure_runtime():
    """Use the same arithmetic in training and standalone checkpoint evaluation."""
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@dataclass
class ScanInput:
    features: np.ndarray
    grid: np.ndarray
    coord: np.ndarray
    inverse: np.ndarray
    order: np.ndarray
    ptr: np.ndarray


def prepare_scan(xyzi):
    """Only measured XYZI enters the model; labels and pair identities stay outside."""
    xyzi = np.asarray(xyzi)
    if xyzi.dtype != np.float32 or xyzi.ndim != 2 or xyzi.shape[1] != 4:
        raise ValueError("model input must be float32[N,4]")
    if (
        not len(xyzi)
        or not np.isfinite(xyzi).all()
        or np.any(np.all(xyzi[:, :3] == 0, axis=1))
    ):
        raise ValueError("model input must contain finite actual returns")
    # Sensor-anchored cells, including negative coordinates, use floor, not truncation.
    cells = np.floor(xyzi[:, :3].astype(np.float64) / 0.05).astype(np.int64)
    offset = (xyzi[:, :3] - (cells + 0.5) * 0.05).astype(np.float32)
    grid, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    ptr = np.r_[0, np.cumsum(counts)]
    coord = np.add.reduceat(xyzi[order, :3], ptr[:-1], axis=0) / counts[:, None]
    # A multiple-of-16 shift preserves every nested physical grid while satisfying spconv.
    grid -= (grid.min(0) // 16) * 16
    return ScanInput(
        np.concatenate((xyzi, offset), axis=1),
        grid,
        coord.astype(np.float32),
        inverse,
        order,
        ptr,
    )


class V3(nn.Module):
    def __init__(self, mechanism="A", seed=SEED):
        super().__init__()
        if mechanism not in {"A", "B", "C"}:
            raise ValueError(
                "mechanism must be A (position), B (plain), or C (constant)"
            )
        self.mechanism, self.seed = mechanism, seed
        # Construct the same tensors before removing B's condition modules. This keeps
        # every shared A/B/C tensor identical without relying on RNG call-count accidents.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.point = nn.Sequential(nn.Linear(7, 64), nn.LayerNorm(64), nn.GELU())
            self.backbone = LitePT(
                condition_mode="constant" if mechanism == "C" else "position"
            )
            self.context = nn.Sequential(nn.Linear(72, 64), nn.LayerNorm(64), nn.GELU())
            self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
            nn.init.zeros_(self.head[-1].bias)
        if mechanism == "B":
            for module in self.modules():
                if isinstance(module, PointROPEAttention):
                    module.condition = None

    def forward(self, scans):
        device = next(self.parameters()).device
        values, coords, grids, batches, inverses, details = [], [], [], [], [], []
        voxel_count = 0
        for batch, scan in enumerate(scans):
            features = torch.as_tensor(scan.features, device=device)
            detail = self.point(features)
            order = torch.as_tensor(scan.order, device=device)
            ptr = torch.as_tensor(scan.ptr, device=device)
            values.append(torch_scatter.segment_csr(detail[order], ptr, reduce="mean"))
            coords.append(torch.as_tensor(scan.coord, device=device))
            grids.append(torch.as_tensor(scan.grid, device=device))
            batches.append(
                torch.full((len(scan.grid),), batch, device=device, dtype=torch.long)
            )
            inverses.append(torch.as_tensor(scan.inverse, device=device) + voxel_count)
            details.append(detail)
            voxel_count += len(scan.grid)
        point = self.backbone(
            dict(
                feat=torch.cat(values),
                coord=torch.cat(coords),
                grid_coord=torch.cat(grids),
                batch=torch.cat(batches),
            )
        )
        context = self.context(point.feat)
        return [
            self.head(torch.cat((detail, context[inverse]), dim=1)).squeeze(-1)
            for detail, inverse in zip(details, inverses)
        ]


def compatible_parameter(name):
    return (
        name.startswith(("backbone.enc.", "backbone.dec."))
        and ".condition." not in name
    )


def transfer(model, path, route):
    """Named encoder/decoder roles define compatibility; equal shapes alone do not."""
    if route not in {"P", "T"}:
        raise ValueError("transfer route must be P or T")
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if (
        route == "P"
        and digest != "95f151f6edcfbf315cd06df6afd261f2a2fde300d3c693dd26b1305d642ecc30"
    ):
        raise ValueError("P must be the published LitePT-S nuScenes weight file")
    allowed = [
        getattr,
        torch.optim.lr_scheduler.OneCycleLR,
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        type(np.dtype("uint32")),
        type(np.dtype("int64")),
    ]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    if route == "P":
        source = {
            key.removeprefix("module."): value
            for key, value in saved["state_dict"].items()
        }
    else:
        if saved.get("step") != 1152:
            raise ValueError("T requires the declared 1152 checkpoint")
        source = saved["model"]
    names = [name for name, _ in model.named_parameters() if compatible_parameter(name)]
    target = model.state_dict()
    selected = {}
    for name in names:
        if name not in source or source[name].shape != target[name].shape:
            raise ValueError(f"missing or incompatible backbone role: {name}")
        if (
            source[name].dtype != target[name].dtype
            or not torch.isfinite(source[name]).all()
        ):
            raise ValueError(f"invalid backbone tensor: {name}")
        selected[name] = source[name]
    model.load_state_dict(selected, strict=False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
    return dict(
        route=route,
        source=str(path.resolve()),
        sha256=digest,
        loaded=names,
        new_parameters=[n for n, _ in model.named_parameters() if n not in selected],
        discarded=sorted(set(source) - set(selected)),
        reset_bn_statistics=True,
    )


def optimizer(model, backbone_lr=1e-5, new_lr=1e-4):
    groups = {}
    for name, parameter in model.named_parameters():
        backbone = compatible_parameter(name)
        # Only linear/conv kernels decay; all affine biases and normalization stay free.
        decay = 0.01 if parameter.ndim > 1 and name.endswith("weight") else 0.0
        key = (backbone, decay)
        group = groups.setdefault(
            key,
            dict(
                params=[],
                names=[],
                weight_decay=decay,
                lr=backbone_lr if backbone else new_lr,
                peak_lr=backbone_lr if backbone else new_lr,
                role="backbone" if backbone else "new",
            ),
        )
        group["params"].append(parameter)
        group["names"].append(name)
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999), eps=1e-8)


def lr_factor(update, halvings=()):
    if update < 1 or any(step <= 256 for step in halvings):
        raise ValueError(
            "updates start at 1; the first 256 updates remain on the plateau"
        )
    warmup = 0.1 + 0.9 * (update - 1) / 15 if update <= 16 else 1.0
    return warmup * 0.5 ** sum(update >= step for step in halvings)


def paired_loss(
    positive, negative, positive_target, negative_target, instance, balance="instance"
):
    """Per-request risk; empty terms stay zero and never redistribute their weights."""
    if balance not in {"instance", "frame"}:
        raise ValueError("balance must be instance or frame")
    anomaly = positive_target == 1
    zero = positive.sum() * 0
    a = zero
    if anomaly.any():
        if balance == "instance":
            ids = instance[anomaly]
            if (ids <= 0).any():
                raise ValueError(
                    "instance-balanced training requires reliable positive IDs"
                )
            a = torch.stack(
                [
                    nn.functional.softplus(-positive[anomaly][ids == j]).mean()
                    for j in torch.unique(ids)
                ]
            ).mean()
        else:
            a = nn.functional.softplus(-positive[anomaly]).mean()
    normal = positive_target == 0
    nplus = nn.functional.softplus(positive[normal]).mean() if normal.any() else zero
    normal = negative_target == 0
    nminus = (
        nn.functional.softplus(negative[normal]).mean()
        if normal.any()
        else negative.sum() * 0
    )
    return 0.5 * a + 0.25 * nplus + 0.25 * nminus, torch.stack((a, nplus, nminus))
