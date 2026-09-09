"""V1: one label-free scan, a LitePT-S context encoder and point-level geometry."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_scatter import segment_csr

from vendor.litept.model import LitePT
from .supervision import ScanGeometry


def model_input(xyzi, slots, config, scale_parameters, workers=1):
    """Voxelize context only; every original return keeps its own geometry and output."""
    xyzi = np.asarray(xyzi, np.float32)
    geometry = ScanGeometry(xyzi, slots, scale_parameters, workers)
    grid = np.floor(xyzi[:, :3].astype(np.float64) / config["voxel_size_m"]).astype(np.int32)
    grid -= grid.min(axis=0)
    unique, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable").astype(np.int64)
    pointer = np.r_[0, np.cumsum(counts)].astype(np.int64)
    neighbors = geometry.neighbors[geometry.inverse]
    valid = neighbors >= 0
    # Distinct positions use their first original return, including the 201 aliases.
    rows = geometry.first[np.maximum(neighbors, 0)]
    arrays = dict(xyzi=xyzi, grid_coord=unique, voxel_inverse=inverse.astype(np.int64),
                  voxel_count=counts.astype(np.float32), voxel_order=order, voxel_pointer=pointer,
                  neighbors=rows.astype(np.int64),
                  neighbor_valid=valid, **geometry.scale_arrays())
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in arrays.items()}


class V1(nn.Module):
    """Only predicted boundary/surface evidence enters the common anomaly head."""

    def __init__(self, config, epsilon=1e-6):
        super().__init__()
        self.config, self.epsilon = config, epsilon
        self.enhancements = dict(config["enhancements"])
        self.backbone = LitePT(**config["backbone"])
        width = config["geometry_channels"]
        self.edge = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(2 * width + 2, width), nn.LayerNorm(width), nn.GELU())
        fused = config["fusion_channels"]
        self.fuse = nn.Sequential(nn.Linear(config["backbone"]["dec_channels"][0] + width + 7, fused),
                                  nn.LayerNorm(fused), nn.GELU())
        self.boundary = nn.Linear(fused, 1)
        self.surface = nn.Linear(fused, 1)
        self.anomaly = nn.Sequential(nn.Linear(fused + 2, config["head_channels"]), nn.GELU(),
                                     nn.Linear(config["head_channels"], 1))
        self.register_buffer("feature_divisor", torch.tensor(config["feature_divisor"], dtype=torch.float32))

    def forward(self, scan):
        xyzi = scan["xyzi"]
        inverse, counts = scan["voxel_inverse"], scan["voxel_count"]
        feat = xyzi / self.feature_divisor
        # Sorted segments give every point equal weight without unordered atomic sums.
        voxels = segment_csr(feat[scan["voxel_order"]], scan["voxel_pointer"], reduce="mean")
        coord = segment_csr(xyzi[scan["voxel_order"], :3], scan["voxel_pointer"], reduce="mean")
        point = self.backbone(dict(feat=voxels, coord=coord, grid_coord=scan["grid_coord"],
                                   batch=torch.zeros(len(counts), device=xyzi.device, dtype=torch.long)))
        context = point.feat[inverse]
        if self.enhancements["sampling"]:
            geometry = self.local_geometry(scan)
        else:
            geometry = xyzi.new_zeros((len(xyzi), self.config["geometry_channels"]))
        point_offset = (xyzi[:, :3] - coord[inverse]) / self.config["voxel_size_m"]
        fused = self.fuse(torch.cat((context, geometry, feat, point_offset), dim=1))
        with torch.autocast(device_type=xyzi.device.type, enabled=False):
            fused = fused.float()
            boundary = (self.boundary(fused).squeeze(-1).sigmoid() if self.enhancements["boundary"]
                        else fused.new_zeros(len(fused)))
            surface = (self.surface(fused).squeeze(-1) if self.enhancements["surface"]
                       else fused.new_zeros(len(fused)))
            # The shared point features form a complete detection path even with all enhancements disabled.
            logits = self.anomaly(torch.cat((fused, boundary[:, None], surface[:, None]), dim=1)).squeeze(-1)
        return dict(logits=logits, boundary=boundary, surface=surface)

    def local_geometry(self, scan):
        xyzi = scan["xyzi"]
        neighbor = xyzi[scan["neighbors"]]
        displacement = neighbor[..., :3] - xyzi[:, None, :3]
        valid = scan["neighbor_valid"]
        scale_valid = scan["sampling_scale_valid"]
        normalized = displacement / (scan["sampling_scale"][:, None, None] + self.epsilon)
        normalized = torch.where(scale_valid[:, None, None], normalized, 0.)
        edge = torch.cat((displacement, normalized, neighbor[..., 3:] - xyzi[:, None, 3:],
                          scale_valid[:, None, None].expand(-1, valid.shape[1], 1)), dim=-1)
        edge = torch.where(valid[..., None], edge, 0.)
        edge = self.edge(edge)
        mean = (edge * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        maximum = edge.masked_fill(~valid[..., None], -torch.inf).amax(1)
        maximum = torch.where(valid.any(1)[:, None], maximum, 0.)
        return self.geometry(torch.cat((mean, maximum, scale_valid[:, None],
                                        torch.log1p(scan["sampling_scale"])[:, None]), dim=1))
