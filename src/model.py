"""Joint five-scan LitePT-S input and full-window point anomaly prediction."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import PredictionBatch
from .scene import SceneWindow, WindowPoints


@dataclass(frozen=True)
class JointVoxels:
    """Label-free voxel inputs and inverse mapping in original point order."""

    coordinates: torch.Tensor
    grid_coord: torch.Tensor
    features: torch.Tensor
    point_to_voxel: torch.Tensor
    point_features: torch.Tensor
    source_points: WindowPoints
    voxel_size: float

    def to(self, device: torch.device | str) -> JointVoxels:
        """Move only deterministic inputs; preserve the exact source-point identity."""
        return replace(
            self,
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "coordinates",
                    "grid_coord",
                    "features",
                    "point_to_voxel",
                    "point_features",
                )
            },
        )

    def backbone_input(self) -> dict[str, torch.Tensor]:
        return {
            "coord": self.coordinates,
            "grid_coord": self.grid_coord,
            "feat": self.features,
            # The entire window is one scene, not five independent batches.
            "batch": torch.zeros(
                len(self.coordinates), dtype=torch.long, device=self.coordinates.device
            ),
        }


def joint_voxelize(
    window: SceneWindow,
    voxel_size: float = 0.05,
    *,
    device: torch.device | str = "cpu",
) -> JointVoxels:
    """Aggregate all returns without resampling, relabelling or realigning them."""

    if not isinstance(window, SceneWindow):
        raise TypeError("joint voxelization requires a complete SceneWindow")
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    points = window.points
    if points.count == 0 or points.features.shape != (points.count, 1):
        raise ValueError("a window must contain visible points with raw intensity")

    # Anchor physical cells at the current LiDAR origin, including negative xyz.
    # Float64 division/sums avoid introducing avoidable boundary/mean roundoff.
    cells = np.floor(points.coordinates.astype(np.float64) / voxel_size)
    cells -= cells.min(axis=0)
    if cells.max() >= 2**16:
        raise ValueError("voxel extent exceeds LitePT's 16-bit spatial encoding")
    cells = cells.astype(np.int64)
    # A collision-free 48-bit key preserves the same lexicographic voxel order.
    keys = (cells[:, 0] << 32) | (cells[:, 1] << 16) | cells[:, 2]
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(first)
    grid = cells[order[first]]
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(first, dtype=np.int64) - 1
    counts = np.diff(np.r_[starts, len(order)])
    count = len(grid)
    means = np.empty((count, 4), dtype=np.float32)
    for axis in range(4):
        values = points.coordinates[:, axis] if axis < 3 else points.features[:, 0]
        means[:, axis] = np.bincount(inverse, weights=values, minlength=count) / counts

    # These are scan-hit flags, not visibility estimates or anomaly labels.
    scan = points.scan_group.astype(np.int64)
    hits = np.zeros((count, 5), dtype=np.float32)
    hits[inverse, scan] = 1.0
    features = np.concatenate((means, hits), axis=1)
    point_features = np.concatenate(
        (
            (points.coordinates - means[inverse, :3]) / voxel_size,
            points.features,
            np.eye(5, dtype=np.float32)[scan],
        ),
        axis=1,
    )
    return JointVoxels(
        coordinates=torch.from_numpy(np.ascontiguousarray(means[:, :3])).to(device),
        grid_coord=torch.from_numpy(grid).to(device),
        features=torch.from_numpy(features).to(device),
        point_to_voxel=torch.from_numpy(inverse).to(device),
        point_features=torch.from_numpy(point_features).to(device),
        source_points=points,
        voxel_size=voxel_size,
    )


@dataclass
class NREEvidence:
    score: torch.Tensor
    support: torch.Tensor
    normal_logits: torch.Tensor
    correction: torch.Tensor
    prototype_choice: torch.Tensor


class NREHead(nn.Module):
    """Normal support and a bounded correction share one learned point representation."""

    def __init__(self, active_groups):
        super().__init__()
        groups = tuple(active_groups)
        if (
            len(groups) < 2
            or tuple(sorted(set(groups))) != groups
            or not all(0 <= g < 20 for g in groups)
        ):
            raise ValueError(
                "normal support needs at least two distinct supervised groups"
            )
        self.register_buffer("active_groups", torch.tensor(groups, dtype=torch.long))
        self.fusion = nn.Sequential(
            nn.Linear(117, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, 64)
        )
        self.prototypes = nn.Parameter(torch.randn(len(groups), 4, 64))
        self.acceptance = nn.Parameter(torch.zeros(len(groups)))
        self.correction = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    @property
    def thresholds(self):
        return 0.2 + 0.6 * self.acceptance.sigmoid()

    def forward(self, shallow, deep, inverse, detail):
        h = self.fusion(torch.cat((shallow[inverse], deep[inverse], detail), dim=1))
        # Directional support, log-sum-exp and final logits always use float32.
        with torch.autocast(h.device.type, enabled=False):
            h = h.float()
            q = (
                F.normalize(h, dim=1)
                @ F.normalize(self.prototypes.float(), dim=2).flatten(0, 1).T
            )
            q = q.reshape(-1, len(self.active_groups), 4)
            a = torch.logsumexp(
                (q - self.thresholds[None, :, None]) / 0.07, dim=2
            ) - math.log(4)
            support = math.log(len(self.active_groups)) - torch.logsumexp(a, dim=1)
            correction = 2 * torch.tanh(self.correction(h).squeeze(1))
        return support + correction, support, a, correction, q.argmax(2).to(torch.uint8)


class AJAE(nn.Module):
    """One joint LitePT-S backbone with the NRE or historical binary point head."""

    def __init__(
        self, voxel_size: float = 0.05, *, normal_groups=None, point_chunk_size=65536
    ) -> None:
        super().__init__()
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        # Keep data-only tools usable without loading optional CUDA dependencies.
        from vendor.litept.litept.model import LitePT

        self.voxel_size = voxel_size
        # All other defaults are the official semantic LitePT-S configuration.
        self.backbone = LitePT(in_channels=9)
        self.head = (
            NREHead(normal_groups)
            if normal_groups is not None
            else nn.Sequential(nn.Linear(81, 32), nn.GELU(), nn.Linear(32, 1))
        )
        self.score_kind = "logit" if normal_groups is not None else "probability"
        if point_chunk_size < 1:
            raise ValueError("point chunk size must be positive")
        self.point_chunk_size = point_chunk_size

    def forward(
        self,
        window: SceneWindow,
        *,
        inputs: JointVoxels | None = None,
        return_evidence=False,
    ) -> torch.Tensor | NREEvidence:
        """Return one trainable logit per original point, including history."""

        device = next(self.parameters()).device
        if inputs is None:
            inputs = joint_voxelize(window, self.voxel_size, device=device)
        elif (
            inputs.source_points is not window.points
            or inputs.voxel_size != self.voxel_size
            or inputs.features.device != device
        ):
            # Reuse only deterministic inputs for this exact immutable window.
            raise ValueError(
                "prepared voxels belong to different points, grid or device"
            )
        # spconv 2.3.8 eval bypasses its AMP weight cast. Keep eval inputs/weights
        # in float32; the official attention's internal float16 path is unchanged.
        precision = (
            nullcontext()
            if self.training
            else torch.autocast(device.type, enabled=False)
        )
        with precision:
            nre = isinstance(self.head, NREHead)
            if nre:
                decoded, shallow = self.backbone(
                    inputs.backbone_input(), return_shallow=True
                )
                if shallow.shape != (len(inputs.coordinates), 36):
                    raise RuntimeError(
                        "shallow encoder features must retain original voxel rows"
                    )
            else:
                decoded = self.backbone(inputs.backbone_input())
            if decoded.feat.shape != (len(inputs.coordinates), 72) or not torch.equal(
                decoded.grid_coord, inputs.grid_coord
            ):
                raise RuntimeError(
                    "LitePT decoder did not preserve the input voxel rows"
                )
            if nre:
                chunks = []
                for start in range(
                    0, len(inputs.point_to_voxel), self.point_chunk_size
                ):
                    stop = start + self.point_chunk_size
                    args = (
                        shallow,
                        decoded.feat,
                        inputs.point_to_voxel[start:stop],
                        inputs.point_features[start:stop],
                    )
                    # Recompute only the small head during backward; never repeat the backbone.
                    chunks.append(
                        checkpoint(self.head, *args, use_reentrant=False)
                        if self.training and torch.is_grad_enabled()
                        else self.head(*args)
                    )
                evidence = NREEvidence(
                    *(torch.cat(items) for items in zip(*chunks, strict=True))
                )
                return evidence if return_evidence else evidence.score
            if return_evidence:
                raise ValueError("normal evidence is available only for AJAE-NRE")
            # Labels never vote at voxel level: distinct points share context only.
            features = torch.cat(
                (decoded.feat[inputs.point_to_voxel], inputs.point_features), dim=1
            )
            return self.head(features).squeeze(-1)

    @torch.inference_mode()
    def predict(self, window: SceneWindow) -> PredictionBatch:
        """Retain all point scores and their identities; filter online only later."""

        if self.training:
            raise RuntimeError("call model.eval() before prediction")
        logits = self(window).float()
        scores = (
            (logits if self.score_kind == "logit" else logits.sigmoid()).cpu().numpy()
        )
        return PredictionBatch.from_window(window, scores, score_kind=self.score_kind)
