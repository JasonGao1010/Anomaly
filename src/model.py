"""Joint five-scan LitePT-S input and full-window point anomaly prediction."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

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
    grid, inverse, counts = np.unique(
        cells.astype(np.int64), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
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
        coordinates=torch.tensor(means[:, :3], device=device),
        grid_coord=torch.tensor(grid, dtype=torch.long, device=device),
        features=torch.tensor(features, device=device),
        point_to_voxel=torch.tensor(inverse, dtype=torch.long, device=device),
        point_features=torch.tensor(point_features, device=device),
        source_points=points,
        voxel_size=voxel_size,
    )


class AJAE(nn.Module):
    """Randomly initialized semantic LitePT-S plus an 81 -> 32 -> 1 point head."""

    def __init__(self, voxel_size: float = 0.05) -> None:
        super().__init__()
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        # Keep data-only tools usable without loading optional CUDA dependencies.
        from vendor.litept.litept.model import LitePT

        self.voxel_size = voxel_size
        # All other defaults are the official semantic LitePT-S configuration.
        self.backbone = LitePT(in_channels=9)
        self.head = nn.Sequential(nn.Linear(81, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(
        self, window: SceneWindow, *, inputs: JointVoxels | None = None
    ) -> torch.Tensor:
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
            decoded = self.backbone(inputs.backbone_input())
            if decoded.feat.shape != (len(inputs.coordinates), 72) or not torch.equal(
                decoded.grid_coord, inputs.grid_coord
            ):
                raise RuntimeError(
                    "LitePT decoder did not preserve the input voxel rows"
                )
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
        scores = torch.sigmoid(self(window).float()).cpu().numpy()
        return PredictionBatch.from_window(window, scores)
