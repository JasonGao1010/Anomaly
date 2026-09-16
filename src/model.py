"""Point detail and backbone-scale context, with independent historical inference."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .data import FramePrediction
from .protocol import PROJECT_ROOT
from .render import calibrated_ray_grid

SENSOR_CONDITIONS = ("log_range", "ray_x", "ray_y", "ray_z", "log_azimuth_scale",
    "log_elevation_scale", "log_azimuth_step", "log_elevation_step", "log_radial_scale")
CELL_STATISTICS = ("log_returns", "log_positions", "centroid_x", "centroid_y", "centroid_z",
    "cov_xx", "cov_xy", "cov_xz", "cov_yy", "cov_yz", "cov_zz", "intensity_mean", "intensity_variance")
OBSERVATION_CONDITIONS = ("delta", "delta_valid") + tuple(
    f"cell_{scale}_{name}" for scale in range(3) for name in CELL_STATISTICS)


def load_config(path=PROJECT_ROOT / "protocol/model.json"):
    return validate_config(json.loads(Path(path).read_text()))


def validate_config(config):
    v3 = config.get("format") == "ajae-v3"
    if (config.get("format") not in {"ajae-v1", "ajae-v3"}
            or config["initialization"] not in {"random_no_external_weights", "nuscenes_litept_s"}):
        raise ValueError("expected a declared AJAE initialization")
    m, t, loss = config["model"], config["training"], config["loss"]
    if config["initialization"] == "nuscenes_litept_s":
        source = config.get("pretrained", {})
        if (source.get("repository") != "prs-eth/LitePT"
                or source.get("revision") != "a8e76e92efbb2061639f5c683968bc5d248ee002"
                or source.get("filename") != "nuscenes-semseg-litept-small-v1m1/model/model_best.pth"
                or source.get("sha256") != "95f151f6edcfbf315cd06df6afd261f2a2fde300d3c693dd26b1305d642ecc30"
                or m.get("intensity_transform") != "identity_raw_stu_no_clip"
                or m.get("backbone_batchnorm") not in {"train_batch_eval_running", "fixed_parent_running"}
                or not 0 < t.get("backbone_learning_rate", 0) <= t["learning_rate"]):
            raise ValueError("incomplete pretrained source, input rule or fine-tuning configuration")
    elif "pretrained" in config or "backbone_learning_rate" in t:
        raise ValueError("random initialization cannot silently inherit a pretrained recipe")
    if v3:
        pyramid = m["relation_mode"] == "pyramid"
        if pyramid:
            obsolete = {"relation_layers", "cell_sizes_m", "context_radius_m", "radial_scale_m",
                "minimum_sampling_m", "scale_neighbors", "scale_radius_m", "scale_minimum_neighbors",
                "relation_chunk", "checkpoint_relations", "neighbors_per_shell", "condition_modulation"}
            if (config["initialization"] != "nuscenes_litept_s"
                    or m["backbone"] != "LitePT-S" or m["channels"] != 64
                    or m["attention_heads"] != 4 or m["voxel_m"] != .05
                    or m["query_chunk"] < 1 or not isinstance(m["checkpoint_fusion"], bool)
                    or m["backbone_batchnorm"] != "fixed_parent_running"
                    or m["backbone_drop_path"] != 0 or m["backbone_shuffle_orders"]
                    or obsolete.intersection(m)):
                raise ValueError("invalid V3 backbone pyramid or inherited normalization")
        elif (config["initialization"] != "nuscenes_litept_s"
                or m["backbone"] != "LitePT-S" or m["channels"] != 64
                or m["relation_layers"] != 2 or m["relation_mode"] != "multiscale"
                or m["cell_sizes_m"] != [.05, .2, .8] or m["attention_heads"] != 4
                or m["context_radius_m"] != 2. or m["voxel_m"] != .05
                or m["scale_neighbors"] != 6 or m["scale_minimum_neighbors"] != 3
                or m["scale_radius_m"] != .5 or m["relation_chunk"] < 1
                or min(m["radial_scale_m"], m["minimum_sampling_m"]) <= 0
                or m["backbone_batchnorm"] != "fixed_parent_running"
                or m["backbone_drop_path"] != 0 or m["backbone_shuffle_orders"]
                or "neighbors_per_shell" in m or "condition_modulation" in m):
            raise ValueError("invalid V3 multiscale structure or inherited normalization")
        if (t["batch_frames"] != 2 or t["accumulation_steps"] != 4 or t["workers"] < 0
                or t["binary_view"] != "official_range_v3" or t["anomaly_queries"] != 2048
                or t["augmentation"] != "none_preserve_physical_sensor_conditions"
                or t["conditions"] != dict(radius_m=2., minimum_neighbors=8)
                or t["group_queries"] != dict(normal=[4096, 2048, 2048], raw=[2048, 1024, 1024], weights=[.5, .25, .25])
                or t["learning_rate_schedule"] != dict(kind="v3_adapt_joint", freeze_updates=128,
                    warmup_updates=32, decay_start=160, total_updates=1024, start_factor=.1, end_factor=.1)
                or [t[k] for k in ("learning_rate", "inherited_learning_rate", "backbone_learning_rate")] != [1e-4, 5e-6, 2e-6]
                or t["gradient_clip"] != 1. or t["weight_decay"] != .01 or t["save_every"] != 128
                or loss != dict(kind="paired_binary_population", coefficients=[.5, .25, .25])):
            raise ValueError("invalid V3 population risk, request budget or update schedule")
        return config
    if (m["backbone"] != "LitePT-S" or m["relation_mode"] not in {"none", "plain", "conditioned"}
            or not isinstance(m.get("condition_modulation"), bool)
            or m["radii_m"] != [0.25, 0.75, 2.0]
            or m["channels"] != 64 or m["relation_layers"] != 2
            or m["neighbors_per_shell"] < 1 or m["relation_chunk"] < 1
            or m["backbone_drop_path"] != 0 or m["backbone_shuffle_orders"]):
        raise ValueError("invalid V1 structure or stochastic paired-scan encoder")
    if (min(m["voxel_m"], m["radial_scale_m"], m["minimum_sampling_m"]) <= 0
            or not 1 <= m["scale_minimum_neighbors"] <= m["scale_neighbors"] <= m["neighbors_per_shell"]
            or not m["radii_m"][0] <= m["scale_radius_m"] <= m["radii_m"][1]):
        raise ValueError("invalid physical geometry scales")
    if (t["batch_frames"] < 2 or t["workers"] < 0
            or min(t[k] for k in ("normal_queries", "anomaly_queries", "keep_queries", "save_every")) < 1
            or min(t[k] for k in ("learning_rate", "keep_near_m", "gradient_clip")) <= 0
            or not 0 <= t["keep_near_fraction"] <= 1
            or t["warmup_steps"] < 0 or t["ramp_steps"] < 1
            or t["augmentation"] != "none_preserve_physical_sensor_conditions"):
        raise ValueError("invalid sampling or optimization configuration")
    schedule = t.get("learning_rate_schedule")
    if schedule is not None and (
            set(schedule) != {"kind", "warmup_updates", "total_updates", "start_factor", "end_factor"}
            or schedule["kind"] != "linear_warmup_cosine"
            or not isinstance(schedule["warmup_updates"], int) or not isinstance(schedule["total_updates"], int)
            or not 2 <= schedule["warmup_updates"] < schedule["total_updates"]
            or not 0 < schedule["start_factor"] <= 1 or not 0 < schedule["end_factor"] <= 1):
        raise ValueError("invalid continuous learning-rate schedule")
    if (loss["keep_mode"] not in {"worst", "mean", "increase"}
            or loss.get("anomaly_reduction", "frame") not in {"frame", "point"}
            or min(loss["keep_weight"], loss["tail_weight"], loss["margin"]) < 0
            or loss["temperature"] <= 0 or loss["pairs_per_tail"] < 1
            or any(not 0 < loss[k] <= 1 for k in ("normal_tail_fraction", "anomaly_tail_fraction"))):
        raise ValueError("invalid task-loss configuration")
    if "group_queries" in t or "conditions" in t:
        groups, conditions = t.get("group_queries", {}), t.get("conditions", {})
        if (groups != dict(normal=[4096, 2048, 2048], keep=[512, 256, 256], weights=[.5, .25, .25])
                or conditions != dict(radius_m=2., minimum_neighbors=8, minimum_anomaly_rays=5,
                    minimum_normal_positions=5, anomaly_range_m=[2.5, 50.], mixture=.2)
                or t["batch_frames"] != 2 or t["normal_queries"] != 8192
                or t["keep_queries"] != 1024 or t["anomaly_queries"] != 2048
                or loss["keep_mode"] != "mean" or loss["keep_weight"] != 1. or loss["tail_weight"] != 0.):
            raise ValueError("V2 requires the declared groups, conditions and full mean protection")
    return config


def inherit_backbone(model, config, path):
    """Transfer every backbone tensor; no official optimizer, schedule or classifier survives."""
    validate_config(config)
    if config["initialization"] != "nuscenes_litept_s":
        raise ValueError("external weights require their declared initialization")
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != config["pretrained"]["sha256"]:
        raise ValueError("checkpoint does not match the pinned official weight file")
    # The official training file contains scheduler and NumPy scalar metadata.
    # Use a limited unpickler allowlist only after verifying its published content digest.
    allowed = [getattr, torch.optim.lr_scheduler.OneCycleLR,
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
        type(np.dtype("float64"))]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    weights = saved["state_dict"]
    return transfer_backbone(model, weights)


def transfer_backbone(model, weights):
    prefix = "module.backbone."
    target = model.backbone.state_dict()
    expected = {prefix + name for name in target} | {"module.seg_head.weight", "module.seg_head.bias"}
    if set(weights) != expected:
        raise ValueError(f"official state keys differ: missing={sorted(expected - set(weights))}, "
                         f"unexpected={sorted(set(weights) - expected)}")
    selected = {name: weights[prefix + name] for name in target}
    shapes = {"module.seg_head.weight": (16, 72), "module.seg_head.bias": (16,)}
    for name, tensor in weights.items():
        expected_shape = target[name[len(prefix):]].shape if name.startswith(prefix) else shapes[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected_shape or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid official tensor: {name}")
        if name.startswith(prefix) and tensor.dtype != target[name[len(prefix):]].dtype:
            raise ValueError(f"official tensor dtype differs: {name}")
    model.backbone.load_state_dict(selected, strict=True)
    if any(not torch.equal(value.cpu(), selected[name]) for name, value in model.backbone.state_dict().items()):
        raise ValueError("loaded backbone differs from the official tensors")
    return dict(loaded_keys=len(target), parameters=sum(p.numel() for p in model.backbone.parameters()),
        missing_keys=[], unexpected_keys=[], shape_mismatches=[], exact_tensor_equality=True,
        tensors={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in selected.items()},
        discarded_classifier={name: list(shape) for name, shape in shapes.items()},
        discarded_training_state=["optimizer", "scheduler", "scaler", "epoch", "best_metric_value"])


def transfer_parent(model, saved):
    """Inherit representations; the detail adapter, fusion and score head remain new."""
    if model.config["relation_mode"] != "pyramid" or saved["step"] != 1152:
        raise ValueError("new V3 transfer requires the declared1152 parent")
    weights = saved["model"]
    for name in ("backbone", "context", "point"):
        prefix = name + "."
        getattr(model, name).load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
    return dict(encoders=["backbone", "context", "point"], score_head_inherited=False,
                final_parent_score_preserved=False)


@njit(parallel=True)
def _shell_neighbors(xyz, first, lower, upper, nodes, permutation, radii2, k):
    """Exact annular nearest neighbors: prune both outside and fully inside a shell."""
    n, shells = len(xyz), len(radii2)
    indices = np.full((n, shells * k), -1, np.int32)
    distances = np.full((n, shells * k), np.inf)
    for row in prange(n):
        stack = np.empty(64, np.int32)
        stack[0], pending = 0, 1
        while pending:
            pending -= 1
            node = stack[pending]
            minimum, maximum = 0., 0.
            for axis in range(3):
                a, b = lower[node, axis] - xyz[row, axis], upper[node, axis] - xyz[row, axis]
                minimum += max(a, -b, 0.) ** 2
                maximum += max(a * a, b * b)
            possible, inner = False, 0.
            for shell in range(shells):
                cap = min(radii2[shell], distances[row, (shell + 1) * k - 1])
                # A tiny bound cushion avoids pruning an equal-distance tie by roundoff.
                possible |= minimum <= cap + 1e-12 and maximum + 1e-12 > inner
                inner = radii2[shell]
            if not possible:
                continue
            left, right, begin, end = nodes[node]
            if left >= 0:
                near_left, near_right = 0., 0.
                for axis in range(3):
                    near_left += max(lower[left, axis] - xyz[row, axis], xyz[row, axis] - upper[left, axis], 0.) ** 2
                    near_right += max(lower[right, axis] - xyz[row, axis], xyz[row, axis] - upper[right, axis], 0.) ** 2
                stack[pending] = right if near_left <= near_right else left
                stack[pending + 1] = left if near_left <= near_right else right
                pending += 2
                continue
            for pos in range(begin, end):
                j = permutation[pos]
                d2 = 0.
                for axis in range(3):
                    d2 += (xyz[j, axis] - xyz[row, axis]) ** 2
                if d2 == 0 or d2 > radii2[-1]:
                    continue
                shell = 0
                while d2 > radii2[shell]:
                    shell += 1
                lo, hi, slot = shell * k, (shell + 1) * k - 1, first[j]
                if d2 > distances[row, hi] or (d2 == distances[row, hi] and slot >= indices[row, hi]):
                    continue
                while hi > lo and (d2 < distances[row, hi - 1]
                                   or (d2 == distances[row, hi - 1] and slot < indices[row, hi - 1])):
                    distances[row, hi] = distances[row, hi - 1]
                    indices[row, hi] = indices[row, hi - 1]
                    hi -= 1
                distances[row, hi], indices[row, hi] = d2, slot
    return indices, np.sqrt(distances)


def shell_neighbors(xyz, first, radii, k):
    tree = cKDTree(xyz)
    lower, upper, nodes = [], [], []

    def visit(node):
        index = len(nodes)
        points = xyz[tree.indices[node.start_idx:node.end_idx]]
        lower.append(points.min(0))
        upper.append(points.max(0))
        nodes.append([-1, -1, node.start_idx, node.end_idx])
        if node.split_dim >= 0:
            nodes[index][:2] = [visit(node.lesser), visit(node.greater)]
        return index

    if not len(xyz):
        return np.empty((0, len(radii) * k), np.int32), np.empty((0, len(radii) * k))
    visit(tree.tree)
    return _shell_neighbors(xyz, first.astype(np.int32), np.array(lower), np.array(upper),
                            np.array(nodes, np.int32), tree.indices, np.square(radii), k)


def support_cells(xyzi, sizes, first):
    """Each scale aggregates original returns directly; coordinate deduplication only counts support."""
    xyz = xyzi[:, :3].astype(np.float64)
    result = {}
    for scale, size in enumerate(sizes):
        grid = np.floor(xyz / size).astype(np.int64)
        cells, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
        n = len(cells)
        offset = xyz - (grid + .5) * size
        mean = np.column_stack([np.bincount(inverse, weights=offset[:, axis], minlength=n) / counts for axis in range(3)])
        centered = offset - mean[inverse]
        covariance = np.column_stack([np.bincount(inverse, weights=centered[:, a] * centered[:, b], minlength=n) / counts
                                      for a, b in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))])
        intensity = xyzi[:, 3].astype(np.float64)
        intensity_mean = np.bincount(inverse, weights=intensity, minlength=n) / counts
        variance = np.bincount(inverse, weights=(intensity - intensity_mean[inverse]) ** 2, minlength=n) / counts
        positions = np.bincount(inverse[first], minlength=n)
        centroid = (cells + .5) * size + mean
        statistics = np.column_stack((np.log1p(counts), np.log1p(positions), centroid, covariance, intensity_mean, variance))
        reach = 3 if scale == 2 else 1
        offsets = np.stack(np.meshgrid(*([np.arange(-reach, reach + 1)] * 3), indexing="ij"), -1).reshape(-1, 3)
        adjacent = np.full((n, len(offsets)), -1, np.int32)
        if n:
            origin = cells.min(0)
            shape = cells.max(0) - origin + 1
            if int(shape[0]) * int(shape[1]) * int(shape[2]) >= 2**62:
                raise ValueError("support-cell index exceeds exact integer capacity")
            def pack(values):
                return (values[..., 0] * shape[1] + values[..., 1]) * shape[2] + values[..., 2]
            keys = pack(cells - origin)
            for begin in range(0, n, 2048):
                candidate = cells[begin:begin + 2048, None] - origin + offsets
                packed = pack(candidate)
                found = np.searchsorted(keys, packed).clip(max=n - 1)
                valid = ((candidate >= 0) & (candidate < shape)).all(-1) & (keys[found] == packed)
                adjacent[begin:begin + len(candidate)] = np.where(valid, found, -1)
        for name, values in dict(inverse=inverse, offset=(offset / size).astype(np.float32),
                count=counts.astype(np.float32), positions=positions, grid=cells,
                centroid=centroid.astype(np.float32), statistics=statistics.astype(np.float32), adjacent=adjacent).items():
            result[f"cell_{scale}_{name}"] = values
    return result


class ScanTransform:
    """Label-free preprocessing; every nonzero source return retains its own output row."""

    def __init__(self, config, *, state=None, workers=1):
        self.config, self.workers = deepcopy(config["model"]), workers
        set_num_threads(workers)
        if state is None:
            data = json.loads((PROJECT_ROOT / "protocol/data.json").read_text())
            grid = calibrated_ray_grid(PROJECT_ROOT / data["calibration"]["rays"])
            elevations = np.unique(np.sort(grid.beam_elevation_rad))
            state = dict(format="ajae-scan-transform-v1",
                elevations=torch.tensor(elevations, dtype=torch.float64),
                elevation_step=torch.tensor(np.gradient(elevations), dtype=torch.float64),
                azimuth_step=torch.tensor(2 * np.pi / grid.columns, dtype=torch.float64))
        if (not isinstance(state, dict)
                or set(state) != {"format", "elevations", "elevation_step", "azimuth_step"}
                or state["format"] != "ajae-scan-transform-v1"
                or any(not isinstance(state[key], torch.Tensor) or state[key].dtype != torch.float64
                       for key in ("elevations", "elevation_step", "azimuth_step"))):
            raise ValueError("invalid saved scan-transform state")
        elevations, steps, azimuth = (state[key].detach().cpu().numpy().copy()
            for key in ("elevations", "elevation_step", "azimuth_step"))
        if (elevations.ndim != 1 or len(elevations) < 2 or steps.shape != elevations.shape
                or azimuth.ndim != 0 or not np.isfinite(elevations).all()
                or not np.isfinite(steps).all() or not np.isfinite(azimuth)
                or np.any(np.diff(elevations) <= 0) or np.any(np.abs(elevations) > np.pi / 2)
                or np.any((steps <= 0) | (steps > np.pi)) or not 0 < azimuth <= 2 * np.pi):
            raise ValueError("invalid saved ray angles or sampling intervals")
        # A restored model uses its actual angular calibration, never the current data files.
        self.elevations, self.elevation_step, self.azimuth_step = elevations, steps, float(azimuth)

    def state_dict(self):
        return dict(format="ajae-scan-transform-v1",
            elevations=torch.tensor(self.elevations, dtype=torch.float64),
            elevation_step=torch.tensor(self.elevation_step, dtype=torch.float64),
            azimuth_step=torch.tensor(self.azimuth_step, dtype=torch.float64))

    def __call__(self, source):
        slots = source.observation_slots.copy()
        m = self.config
        # Preserve the physical sensor axes in both coordinates and directional conditions.
        xyzi = source.xyzi[slots].copy()
        xyz = xyzi[:, :3].astype(np.float64)
        n = len(xyz)
        unique, first, inverse = np.unique(xyz, axis=0, return_index=True, return_inverse=True)
        grid = np.floor(xyz / m["voxel_m"]).astype(np.int64)
        cells, voxel_inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
        voxel_xyzi = np.zeros((len(cells), 4), np.float64)
        np.add.at(voxel_xyzi, voxel_inverse, xyzi)
        voxel_xyzi /= counts[:, None]
        offset = xyz - (grid + .5) * m["voxel_m"]
        # Shift by a multiple of the entire pooling stride, preserving voxel membership.
        if len(cells):
            cells = cells - np.floor_divide(cells.min(axis=0), 16) * 16
        if np.any(cells >= 65536):
            raise ValueError("complete scan exceeds LitePT serialization extent; no points were discarded")
        arrays = dict(xyzi=xyzi, source_slot=slots, record_inverse=source.record_inverse.copy(),
            point_offset=offset.astype(np.float32),
            voxel_inverse=voxel_inverse, voxel_xyzi=voxel_xyzi.astype(np.float32), grid_coord=cells.astype(np.int32))
        r = np.linalg.norm(xyz, axis=1)
        u = xyz / r[:, None]
        theta = np.arcsin(np.clip(u[:, 2], -1, 1))
        beam = np.abs(theta[:, None] - self.elevations).argmin(axis=1)
        if m["relation_mode"] == "pyramid":
            arrays.update(voxel_count=counts.astype(np.float32),
                voxel_positions=np.bincount(voxel_inverse[first], minlength=len(cells)).astype(np.float32),
                condition=np.column_stack((np.log1p(r), u, np.full(n, self.azimuth_step),
                                           self.elevation_step[beam])).astype(np.float32))
            return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}

        # Historical inference retains its original physical statistics and support rules.
        if m["relation_mode"] == "multiscale":
            # This capped query defines delta only; relation support has no point-count cap.
            close = cKDTree(unique).query(unique, k=list(range(2, m["scale_neighbors"] + 2)), workers=self.workers)[0]
            close[close > m["scale_radius_m"]] = np.inf
        else:
            indices, distances = shell_neighbors(unique, first, m["radii_m"], m["neighbors_per_shell"])
            close = np.sort(np.where(distances <= m["scale_radius_m"], distances, np.inf), axis=1)
            close = close[:, :m["scale_neighbors"]]
        count = np.isfinite(close).sum(axis=1)
        valid = count >= m["scale_minimum_neighbors"]
        delta = np.zeros(len(unique))
        for k in range(m["scale_minimum_neighbors"], m["scale_neighbors"] + 1):
            take = count == k
            delta[take] = np.median(close[take, :k], axis=1)

        rho = np.linalg.norm(u[:, :2], axis=1)
        az = np.column_stack((-u[:, 1], u[:, 0], np.zeros(n))) / np.maximum(rho[:, None], 1e-12)
        az[rho < 1e-12] = [0, 1, 0]
        basis = np.stack((u, az, np.cross(u, az)), axis=2)
        az_m = np.maximum(r * rho * self.azimuth_step, m["minimum_sampling_m"])
        el_m = np.maximum(r * self.elevation_step[beam], m["minimum_sampling_m"])
        scales = np.column_stack((np.full(n, m["radial_scale_m"]), az_m, el_m))
        conditions = np.column_stack((np.log(r), u, np.log(az_m), np.log(el_m), delta[inverse], valid[inverse]))

        arrays.update(condition=conditions.astype(np.float32), basis=basis.astype(np.float32),
                      sensing_scale=scales.astype(np.float32), geometry_inverse=inverse)
        if m["relation_mode"] == "multiscale":
            arrays.update(support_cells(xyzi, m["cell_sizes_m"], first))
            arrays["sensor_condition"] = np.column_stack((conditions[:, :6], np.full(n, np.log(self.azimuth_step)),
                np.log(self.elevation_step[beam]), np.full(n, np.log(m["radial_scale_m"])))).astype(np.float32)
            arrays["observation_condition"] = np.column_stack((conditions[:, 6:],
                *[arrays[f"cell_{i}_statistics"][arrays[f"cell_{i}_inverse"]] for i in range(3)])).astype(np.float32)
        else:
            arrays["neighbors"] = indices
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}


def to_device(scan, device):
    return {key: value.to(device, non_blocking=True) for key, value in scan.items()}


def _mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class RelationLayer(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(2 * channels, channels)
        self.key = nn.Linear(2 * channels, channels)
        self.edge = _mlp(channels + 7, channels, channels)
        self.modulation = _mlp(16, channels, 2 * channels)
        self.bias = nn.Linear(channels, 1)
        self.null = _mlp(2 * channels + 8, channels, 3)
        self.update = _mlp(5 * channels, 2 * channels, channels)

    def part(self, z, h, scan, query, conditioned, condition_modulation=True, trace=None):
        ids = scan["neighbors"][scan["geometry_inverse"][query]].long()
        valid = ids >= 0
        ids = ids.clamp_min(0)
        zi, zj = self.norm(z[query]), self.norm(z[ids])
        hi, hj = h[query], h[ids]
        difference = scan["xyzi"][ids, :3] - scan["xyzi"][query, None, :3]
        ci, cj = scan["condition"][query], scan["condition"][ids]
        if conditioned:
            physical = torch.einsum("nkj,njl->nkl", difference, scan["basis"][query])
            sensing = physical / scan["sensing_scale"][query, None]
            if not condition_modulation:
                # Keep the sensing coordinates; remove only direct condition-dependent modulation.
                ci, cj = torch.zeros_like(ci), torch.zeros_like(cj)
        else:
            physical = sensing = difference
            # Same-sized ordinary relation uses learned point features, not sensing descriptors.
            ci, cj = zi[:, :8], zj[..., :8]
        intensity = scan["xyzi"][ids, 3:4] - scan["xyzi"][query, None, 3:4]
        edge = self.edge(torch.cat((physical, sensing, intensity, zj - zi[:, None]), dim=-1))
        gamma, beta = self.modulation(torch.cat((ci[:, None].expand_as(cj), cj), dim=-1)).chunk(2, dim=-1)
        values = torch.nn.functional.gelu(edge * (1 + gamma.tanh()) + beta)
        q, k = self.query(torch.cat((zi, hi), -1)), self.key(torch.cat((zj, hj), -1))
        scores = (q[:, None] * k).sum(-1) / z.shape[1]**.5 + self.bias(values).squeeze(-1)
        if trace is not None:
            trace("sensing", sensing[valid])
            trace("values", values[valid])
            trace("attention_scores", scores[valid])
        scores = scores.masked_fill(~valid, -torch.inf).reshape(len(query), 3, -1)
        null = self.null(torch.cat((zi, hi, ci), -1)).unsqueeze(-1)
        # Every shell has a finite null logit, including completely unsupported queries.
        alpha = torch.softmax(torch.cat((scores, null), -1).float(), -1)[..., :-1]
        values = values.reshape(len(query), 3, -1, z.shape[1])
        evidence = (alpha[..., None] * values).sum(2).flatten(1)
        residual = self.update(torch.cat((hi, zi, evidence), -1))
        result = z[query] + residual
        if trace is not None:
            trace("null_scores", null)
            trace("attention_weights", alpha)
            trace("residual", residual)
            trace("output", result)
        return result

    def forward(self, z, h, scan, query, *, conditioned, chunk, recompute, condition_modulation=True, trace=None):
        parts = []
        for q in query.split(chunk):
            def compute(z, h, q):
                return self.part(z, h, scan, q, conditioned, condition_modulation, trace)
            parts.append(checkpoint(compute, z, h, q, use_reentrant=False)
                         if recompute and self.training and torch.is_grad_enabled()
                         else compute(z, h, q))
        return torch.cat(parts) if parts else z[:0]


class MultiscaleRelation(nn.Module):
    """All-return cell summaries and complete per-query cell softmax, including a zero-value null."""

    def __init__(self, channels=64, heads=4):
        super().__init__()
        self.channels, self.heads = channels, heads
        self.norm = nn.LayerNorm(channels)
        self.cell_point = nn.ModuleList([_mlp(channels + 3, channels, channels) for _ in range(3)])
        self.cell_projection = nn.ModuleList([nn.Sequential(nn.Linear(2 * channels + len(CELL_STATISTICS), channels),
            nn.LayerNorm(channels), nn.GELU()) for _ in range(3)])
        query_size = 2 * channels + len(SENSOR_CONDITIONS)
        self.query = nn.ModuleList([nn.Linear(query_size, channels) for _ in range(3)])
        self.key_value = nn.ModuleList([nn.Linear(channels, 2 * channels) for _ in range(3)])
        self.edge = _mlp(9, 16, heads)
        self.null = _mlp(query_size, channels, 3 * heads)
        self.update = _mlp(5 * channels, 2 * channels, channels)

    def pool(self, z, scan, scale):
        inverse = scan[f"cell_{scale}_inverse"]
        count = scan[f"cell_{scale}_count"]
        feature = self.cell_point[scale](torch.cat((z, scan[f"cell_{scale}_offset"]), -1))
        total = feature.new_zeros((len(count), self.channels)).index_add(0, inverse, feature)
        maximum = feature.new_full(total.shape, -torch.inf).scatter_reduce(0,
            inverse[:, None].expand_as(feature), feature, reduce="amax", include_self=True)
        return self.cell_projection[scale](torch.cat((total / count[:, None], maximum,
                                                      scan[f"cell_{scale}_statistics"]), -1))

    def part(self, z, h, scan, query, keys, values, trace=None):
        zi, hi = z[query], h[query]
        inputs = torch.cat((zi, hi, scan["sensor_condition"][query]), -1)
        null = self.null(inputs).reshape(len(query), 3, self.heads)
        evidence = []
        for scale, size in enumerate((.05, .2, .8)):
            own = scan[f"cell_{scale}_inverse"][query]
            adjacent = scan[f"cell_{scale}_adjacent"][own].long()
            row, column = torch.where(adjacent >= 0)
            cell = adjacent[row, column]
            if scale == 2:
                # Exact cell/ball intersection uses box geometry, never centroid ranking.
                point = scan["xyzi"][query[row], :3].double()
                lower = scan[f"cell_{scale}_grid"][cell].double() * size
                distance = torch.maximum(torch.maximum(lower - point, point - (lower + size)), torch.zeros_like(point))
                valid = distance.square().sum(-1) <= 4. + 1e-12
                row, cell = row[valid], cell[valid]
            q = self.query[scale](inputs).reshape(len(query), self.heads, -1)
            displacement = scan[f"cell_{scale}_centroid"][cell] - scan["xyzi"][query[row], :3]
            sensing = torch.einsum("ei,eij->ej", displacement, scan["basis"][query[row]]) / scan["sensing_scale"][query[row]]
            edge = torch.cat((displacement, sensing, displacement.new_full((len(cell), 1), size),
                              scan[f"cell_{scale}_statistics"][cell, :2]), -1)
            logits = (q[row] * keys[scale][cell]).sum(-1) / (self.channels / self.heads) ** .5 + self.edge(edge)
            index = row[:, None].expand(-1, self.heads)
            # Null participates in the same denominator; unsupported evidence is exactly zero.
            maximum = null[:, scale].detach().clone().scatter_reduce(0, index, logits.detach(), reduce="amax")
            weights = torch.exp(logits - maximum[row])
            denominator = torch.exp(null[:, scale] - maximum).index_add(0, row, weights)
            weighted = (weights / denominator[row])[..., None] * values[scale][cell]
            pooled = z.new_zeros((len(query), self.heads, self.channels // self.heads)).index_add(0, row, weighted)
            evidence.append(pooled.flatten(1))
            if trace is not None:
                trace(f"scale{scale}.logits", logits)
                trace(f"scale{scale}.null", null[:, scale])
                trace(f"scale{scale}.evidence", pooled)
        return self.update(torch.cat((hi, zi, *evidence), -1))

    def forward(self, z, h, scan, query, *, chunk, recompute, trace=None):
        normalized = self.norm(z)
        keys, values = [], []
        for scale in range(3):
            # Learning features and key/value tensors live only in this layer's current forward.
            unit = self.pool(normalized, scan, scale)
            key, value = self.key_value[scale](unit).reshape(len(unit), 2, self.heads, -1).unbind(1)
            keys.append(key)
            values.append(value)
        parts = []
        for query_part in query.split(chunk):
            def compute(z, h, q, *kv):
                return self.part(z, h, scan, q, kv[:3], kv[3:], trace)
            parts.append(checkpoint(compute, normalized, h, query_part, *keys, *values, use_reentrant=False)
                if recompute and self.training and torch.is_grad_enabled()
                else compute(normalized, h, query_part, *keys, *values))
        # Preserve the unnormalized point residual rather than silently normalizing it at every layer.
        return z[query] + torch.cat(parts) if parts else z[:0]


class ScaleFusion(nn.Module):
    """Choose among backbone levels; their features already contain spatial context."""

    def __init__(self, channels=64, heads=4):
        super().__init__()
        self.heads = heads
        self.position = _mlp(4, 32, channels)
        self.query = nn.Linear(2 * channels + 16, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)

    def forward(self, point, finest, condition, xyz, indices, levels, keys, values, voxel_m, trace=None):
        count = len(point)
        query = self.query(torch.cat((point, finest, condition), -1)).reshape(count, self.heads, -1)
        selected_keys, selected_values = [], []
        for index, level, key, value in zip(indices, levels, keys, values, strict=True):
            relative = (xyz - level["coord"][index]) / (voxel_m * level["stride"])
            position = self.position(torch.cat((relative, relative.new_full((count, 1), np.log2(level["stride"]))), -1))
            # W_k(g + e) = (W_k g + b_k) + W_k e; count the bias exactly once.
            selected_keys.append(key[index] + nn.functional.linear(position, self.key.weight))
            selected_values.append(value[index])
        key = torch.stack(selected_keys, 1).reshape(count, len(levels), self.heads, -1)
        value = torch.stack(selected_values, 1).reshape_as(key)
        weights = torch.softmax((query[:, None] * key).sum(-1) / query.shape[-1] ** .5, dim=1)
        if trace is not None:
            trace("scale_weights", weights)
        return self.output((weights[..., None] * value).sum(1).flatten(1))


class AJAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_config(config)
        from vendor.litept.model import LitePT
        self.config = deepcopy(config["model"])
        m, c = self.config, self.config["channels"]
        self.backbone = LitePT(in_channels=4, drop_path=m["backbone_drop_path"],
                               shuffle_orders=m["backbone_shuffle_orders"])
        self.context = nn.Sequential(nn.Linear(72, c), nn.LayerNorm(c), nn.GELU())
        self.point = nn.Sequential(nn.Linear(7, c), nn.LayerNorm(c), nn.GELU())
        if m["relation_mode"] == "pyramid":
            # Keep new parameters outside the inherited backbone's optimizer family.
            self.detail = nn.Sequential(nn.Linear(2 * c + 1, c), nn.LayerNorm(c), nn.GELU(), nn.Linear(c, 36))
            nn.init.zeros_(self.detail[-1].weight)
            nn.init.zeros_(self.detail[-1].bias)
            self.scales = nn.ModuleList([nn.Sequential(nn.Linear(width, c), nn.LayerNorm(c), nn.GELU())
                                        for width in (72, 144, 252, 504)])
            self.condition = nn.Sequential(nn.Linear(6, 16), nn.LayerNorm(16), nn.GELU())
            self.fusion = ScaleFusion(c, m["attention_heads"])
            self.head = _mlp(3 * c + 16, 128, 1)
            nn.init.zeros_(self.head[-1].bias)
        elif m["relation_mode"] == "multiscale":
            self.relations = nn.ModuleList([MultiscaleRelation(c, m["attention_heads"]) for _ in range(m["relation_layers"])])
            self.head = _mlp(3 * c + len(SENSOR_CONDITIONS) + len(OBSERVATION_CONDITIONS), 128, 1)
        else:
            # Historical checkpoints retain their original independent inference function.
            self.relations = nn.ModuleList([RelationLayer(c) for _ in range(m["relation_layers"])])
            self.base_head = _mlp(2 * c + 8, 128, 1)
            self.relation_head = _mlp(2 * c + 8, 128, 1)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.config.get("backbone_batchnorm") == "fixed_parent_running":
            # Freeze inherited statistics in every forward; affine parameters still learn.
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, scan, query=None, *, return_features=False, trace=None):
        n, m = len(scan["source_slot"]), self.config
        all_rows = torch.arange(n, device=scan["xyzi"].device)
        query = all_rows if query is None else query
        if query.ndim != 1 or query.dtype != torch.long or torch.any((query < 0) | (query >= n)):
            raise ValueError("query rows must address real returns in this complete scan")
        if not n or not len(query):
            if return_features:
                raise ValueError("shared-feature diagnostics require a nonempty real-point query")
            head = self.head if m["relation_mode"] in {"pyramid", "multiscale"} else self.base_head
            return scan["xyzi"][:0, 0] + head[-1].weight.sum() * 0
        if m["relation_mode"] == "pyramid":
            return self.pyramid_forward(scan, query, return_features=return_features, trace=trace)
        # Both declared recipes preserve STU intensity, including values above one.
        # Pretraining changes initialization, not the frozen raw-return representation.
        voxels = scan["voxel_xyzi"]
        encoded = self.backbone(dict(feat=voxels, coord=voxels[:, :3], grid_coord=scan["grid_coord"],
            offset=torch.tensor([len(voxels)], dtype=torch.long, device=voxels.device)))
        h = self.context(encoded.feat)[scan["voxel_inverse"]]
        z = self.point(torch.cat((scan["xyzi"], scan["point_offset"]), -1))
        # Expose the shared graph nodes before relation updates, without detaching them.
        features = dict(context=h, point=z) if return_features else None
        if m["relation_mode"] == "multiscale":
            original = z[query]
            for index, layer in enumerate(self.relations):
                z = layer(z, h, scan, query if index == len(self.relations) - 1 else all_rows,
                          chunk=m["relation_chunk"], recompute=m["checkpoint_relations"], trace=trace)
            score = self.head(torch.cat((h[query], original, z, scan["sensor_condition"][query],
                                        scan["observation_condition"][query]), -1)).squeeze(-1).float()
            return dict(score=score, relation=z, **features) if return_features else score
        base = self.base_head(torch.cat((h[query], z[query], scan["condition"][query]), -1)).squeeze(-1)
        if m["relation_mode"] == "none":
            score = base.float()
            return dict(score=score, base_score=score, relation_score=torch.zeros_like(score),
                        **features) if return_features else score
        for index, layer in enumerate(self.relations):
            # First-layer support is updated for the full scan, even for sampled loss queries.
            z = layer(z, h, scan, query if index == len(self.relations) - 1 else all_rows,
                conditioned=m["relation_mode"] == "conditioned", chunk=m["relation_chunk"],
                recompute=m["checkpoint_relations"], condition_modulation=m["condition_modulation"],
                trace=(lambda name, value, i=index: trace(f"relations.{i}.{name}", value)) if trace is not None else None)
        correction = self.relation_head(torch.cat((h[query], z, scan["condition"][query]), -1)).squeeze(-1)
        score = (base + correction).float()
        return dict(score=score, base_score=base.float(), relation_score=correction.float(),
                    **features) if return_features else score

    def pyramid_forward(self, scan, query, *, return_features=False, trace=None):
        m = self.config
        point = self.point(torch.cat((scan["xyzi"], scan["point_offset"]), -1))
        inverse, count = scan["voxel_inverse"], scan["voxel_count"]
        total = point.new_zeros((len(count), point.shape[1])).index_add(0, inverse, point)
        maximum = point.new_full(total.shape, -torch.inf).scatter_reduce(
            0, inverse[:, None].expand_as(point), point, reduce="amax", include_self=True)
        detail = self.detail(torch.cat((total / count[:, None], maximum, scan["voxel_positions"].log1p()[:, None]), -1))
        voxels = scan["voxel_xyzi"]
        _, levels = self.backbone(dict(feat=voxels, coord=voxels[:, :3], grid_coord=scan["grid_coord"],
            offset=torch.tensor([len(voxels)], dtype=torch.long, device=voxels.device)),
            embedding_residual=detail, return_pyramid=True)
        units = [self.context(levels[0]["feat"])] + [projection(level["feat"])
            for projection, level in zip(self.scales, levels[1:], strict=True)]
        indices = [inverse]
        for level in levels[1:]:
            indices.append(level["pooling_inverse"][indices[-1]])
        context = units[0][inverse]
        condition = self.condition(scan["condition"])
        keys, values = [self.fusion.key(g) for g in units], [self.fusion.value(g) for g in units]

        def compute(rows):
            p, h, c = point[rows], context[rows], condition[rows]
            fused = self.fusion(p, h, c, scan["xyzi"][rows, :3], [ids[rows] for ids in indices],
                                levels, keys, values, m["voxel_m"], trace)
            score = self.head(torch.cat((h, p, fused, c), -1)).squeeze(-1).float()
            return (score, fused) if return_features else score

        parts = [checkpoint(compute, rows, use_reentrant=False)
                 if m["checkpoint_fusion"] and self.training and torch.is_grad_enabled() else compute(rows)
                 for rows in query.split(m["query_chunk"])]
        score = torch.cat([part[0] for part in parts] if return_features else parts)
        if return_features:
            fused = torch.cat([part[1] for part in parts])
            return dict(score=score, fusion=fused, context=context, point=point)
        return score

    @torch.no_grad()
    def predict(self, source, transform=None, *, prepared=None, components=False):
        if self.training:
            raise ValueError("prediction requires model.eval()")
        if components and self.config["relation_mode"] in {"pyramid", "multiscale"}:
            raise ValueError("V3 has one unified score; legacy base/relation components do not exist")
        if prepared is None:
            prepared = transform(source)
        elif (not np.array_equal(prepared["source_slot"], source.observation_slots)
              or not np.array_equal(prepared["record_inverse"], source.record_inverse)
              or not np.array_equal(prepared["xyzi"], source.xyzi[source.observation_slots])):
            raise ValueError("prepared inference input differs from the complete source returns")
        scan = to_device(prepared, next(self.parameters()).device)
        output = self(scan, return_features=components)
        # Components share one complete forward and the same physical return identities.
        scores = ({name: output[key] for name, key in
                   (("base", "base_score"), ("relation", "relation_score"), ("final", "score"))}
                  if components else {"final": output})
        result = {}
        for name, values in scores.items():
            # Expand only at the I/O boundary; all model statistics see one complete scan.
            result[name] = FramePrediction(source.partition, source.sequence_id, source.frame_id,
                                           source.real_slots, values.cpu().numpy()[source.record_inverse])
            result[name].validate(source)
        return result if components else result["final"]
