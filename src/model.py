"""V3: per-return detail, LitePT-S context, and one unbounded anomaly logit."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch_scatter
from scipy.spatial import cKDTree
from torch.utils.checkpoint import checkpoint

from vendor.litept.model import LitePT


SEED = 20260917
METHOD = "ajae-v3-original-point-relations"
LOSS_TERMS = (
    "anomaly",
    "inserted_normal",
    "original_normal",
    "inserted_mean",
    "inserted_tail",
    "original_mean",
    "original_tail",
)


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
    neighbors: np.ndarray


def nearest_returns(xyz, workers=4):
    """Self plus 31 other returns; file-order identities break exact distance ties."""
    xyz = np.asarray(xyz, dtype=np.float64)
    count, width = len(xyz), min(32, len(xyz))
    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz, k=min(width + 1, count), workers=workers)
    if count == 1:
        return np.zeros((1, 1), dtype=np.int32)
    # One extra candidate reveals ties crossing the 31-neighbor boundary.
    distances[indices == np.arange(count)[:, None]] = np.inf
    order = np.lexsort((indices, distances), axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    distances = np.take_along_axis(distances, order, axis=1)
    other = indices[:, : width - 1].copy()
    if count > width:
        tied = np.flatnonzero(distances[:, width - 2] == distances[:, width - 1])
        candidates = tree.query_ball_point(
            xyz[tied], np.nextafter(distances[tied, width - 2], np.inf), workers=workers
        )
        for row, candidate in zip(tied, candidates):
            candidate = np.asarray(candidate, dtype=np.int64)
            candidate = candidate[candidate != row]
            distance = np.square(xyz[candidate] - xyz[row]).sum(1)
            other[row] = candidate[np.lexsort((candidate, distance))[: width - 1]]
    return np.column_stack((np.arange(count), other)).astype(np.int32)


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
        nearest_returns(xyzi[:, :3]),
    )


class RelationDecoder(nn.Module):
    """One raw-return block. Query chunks always read keys from the entire scan."""

    def __init__(self, mechanism, chunk_size=2048):
        super().__init__()
        self.mechanism, self.chunk_size = mechanism, chunk_size
        self.recompute = True
        self.norm = nn.LayerNorm(128)
        if mechanism == "D0":
            self.mlp = nn.Sequential(
                nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128)
            )
        else:
            self.qkv = nn.Linear(128, 384, bias=False)
            self.phi = nn.Sequential(
                nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 128, bias=False)
            )
            self.proj = nn.Linear(128, 128, bias=False)
            self.norm2 = nn.LayerNorm(128)
            self.ffn = nn.Sequential(
                nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128)
            )

    def message(self, query, key, value, xyz, center, neighbors):
        delta = xyz[neighbors] - center[:, None, :]
        position = center / 50.0
        condition = position if self.mechanism == "D3" else torch.zeros_like(position)
        geometry = self.phi(
            torch.cat((condition[:, None, :].expand_as(delta), delta), -1)
        )
        # Neighbor zero is self: reuse its phi(c, 0) and make e_ii exactly zero.
        origin = geometry[:, 0]
        geometry = (geometry - origin[:, None, :]).reshape(len(center), -1, 4, 32)
        keys = key[neighbors].reshape_as(geometry) + geometry
        logits = (query.reshape(-1, 1, 4, 32) * keys).sum(-1) / 32**0.5
        weights = logits.softmax(dim=1)
        message = (
            weights[..., None] * (value[neighbors].reshape_as(geometry) + geometry)
        ).sum(1)
        message = message.flatten(1)
        if self.mechanism == "D2":
            message = (
                message
                + self.phi(torch.cat((position, torch.zeros_like(position)), -1))
                - origin
            )
        return message

    def forward(self, features, xyz, neighbors):
        normalized = self.norm(features)
        if self.mechanism == "D0":
            return features + self.mlp(normalized)
        query, key, value = self.qkv(normalized).chunk(3, -1)
        messages = []
        for start in range(0, len(features), self.chunk_size):
            stop = start + self.chunk_size
            args = (
                query[start:stop],
                key,
                value,
                xyz,
                xyz[start:stop],
                neighbors[start:stop],
            )
            message = (
                checkpoint(
                    self.message, *args, use_reentrant=False, preserve_rng_state=False
                )
                if self.recompute and self.training and torch.is_grad_enabled()
                else self.message(*args)
            )
            messages.append(message)
        updated = features + self.proj(torch.cat(messages))
        return updated + self.ffn(self.norm2(updated))


class V3(nn.Module):
    def __init__(self, mechanism="D3", seed=SEED):
        super().__init__()
        if mechanism not in {"D0", "D1", "D2", "D3"}:
            raise ValueError("mechanism must be D0, D1, D2, or D3")
        self.mechanism, self.seed = mechanism, seed
        # Build common modules first: D0's different parameter count cannot shift them.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.point = nn.Sequential(nn.Linear(7, 64), nn.LayerNorm(64), nn.GELU())
            self.backbone = LitePT()
            self.context = nn.Sequential(nn.Linear(72, 64), nn.LayerNorm(64), nn.GELU())
            self.fusion = nn.Sequential(
                nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU()
            )
            self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))
            self.decoder = RelationDecoder(mechanism)

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
        scores = []
        for scan, detail, inverse in zip(scans, details, inverses):
            fused = self.fusion(torch.cat((detail, context[inverse]), dim=1))
            decoded = self.decoder(
                fused,
                torch.as_tensor(scan.features[:, :3], device=device),
                torch.as_tensor(scan.neighbors, device=device, dtype=torch.long),
            )
            scores.append(self.head(decoded).squeeze(-1))
        return scores


def compatible_parameter(name):
    return name.startswith(("backbone.enc.", "backbone.dec."))


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


def tail_weights(losses, alpha=0.01):
    """Empirical upper-tail mass, including fractional and equal-boundary weights."""
    if not 0 < alpha <= 1:
        raise ValueError("tail fraction must lie in (0,1]")
    if not len(losses):
        return torch.zeros_like(losses)
    mass = max(1.0, alpha * len(losses))
    boundary = losses.detach().topk(int(np.ceil(mass)), sorted=False).values.min()
    above, tied = losses.detach() > boundary, losses.detach() == boundary
    remaining = mass - above.sum()
    return (above.to(losses.dtype) + tied * (remaining / tied.sum())) / mass


def normal_risk(scores, tail_weight=0.5):
    if not 0 <= tail_weight <= 1:
        raise ValueError("normal tail weight must lie in [0,1]")
    losses = nn.functional.softplus(scores)
    mean = losses.mean() if len(losses) else scores.sum() * 0
    tail = (losses * tail_weights(losses)).sum()
    return (1 - tail_weight) * mean + tail_weight * tail, mean, tail


def paired_loss(
    positive,
    negative,
    positive_target,
    negative_target,
    instance,
    balance="instance",
    tail_weight=0.5,
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
    nplus, mean_plus, tail_plus = normal_risk(
        positive[positive_target == 0], tail_weight
    )
    nminus, mean_minus, tail_minus = normal_risk(
        negative[negative_target == 0], tail_weight
    )
    return 0.5 * a + 0.25 * nplus + 0.25 * nminus, torch.stack(
        (a, nplus, nminus, mean_plus, tail_plus, mean_minus, tail_minus)
    )
