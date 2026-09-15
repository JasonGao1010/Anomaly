"""AJAE V1: full-scan context and original-return conditional relationships."""

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


def load_config(path=PROJECT_ROOT / "protocol/model.json"):
    return validate_config(json.loads(Path(path).read_text()))


def validate_config(config):
    if (config.get("format") != "ajae-v1"
            or config["initialization"] not in {"random_no_external_weights", "nuscenes_litept_s"}):
        raise ValueError("expected a declared AJAE V1 initialization")
    m, t, loss = config["model"], config["training"], config["loss"]
    if config["initialization"] == "nuscenes_litept_s":
        source = config.get("pretrained", {})
        if (source.get("repository") != "prs-eth/LitePT"
                or source.get("revision") != "a8e76e92efbb2061639f5c683968bc5d248ee002"
                or source.get("filename") != "nuscenes-semseg-litept-small-v1m1/model/model_best.pth"
                or source.get("sha256") != "95f151f6edcfbf315cd06df6afd261f2a2fde300d3c693dd26b1305d642ecc30"
                or m.get("intensity_transform") != "identity_raw_stu_no_clip"
                or m.get("backbone_batchnorm") != "train_batch_eval_running"
                or not 0 < t.get("backbone_learning_rate", 0) <= t["learning_rate"]):
            raise ValueError("incomplete pretrained source, input rule or fine-tuning configuration")
    elif "pretrained" in config or "backbone_learning_rate" in t:
        raise ValueError("random initialization cannot silently inherit a pretrained recipe")
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
        slots = source.real_slots.copy()
        xyzi = source.xyzi[slots].copy()
        xyz = xyzi[:, :3].astype(np.float64)
        m, n = self.config, len(xyz)
        unique, first, inverse = np.unique(xyz, axis=0, return_index=True, return_inverse=True)
        indices, distances = shell_neighbors(unique, first, m["radii_m"], m["neighbors_per_shell"])
        close = np.sort(np.where(distances <= m["scale_radius_m"], distances, np.inf), axis=1)
        close = close[:, :m["scale_neighbors"]]
        count = np.isfinite(close).sum(axis=1)
        valid = count >= m["scale_minimum_neighbors"]
        delta = np.zeros(len(unique))
        for k in range(m["scale_minimum_neighbors"], m["scale_neighbors"] + 1):
            take = count == k
            delta[take] = np.median(close[take, :k], axis=1)

        r = np.linalg.norm(xyz, axis=1)
        u = xyz / r[:, None]
        rho = np.linalg.norm(u[:, :2], axis=1)
        az = np.column_stack((-u[:, 1], u[:, 0], np.zeros(n))) / np.maximum(rho[:, None], 1e-12)
        az[rho < 1e-12] = [0, 1, 0]
        basis = np.stack((u, az, np.cross(u, az)), axis=2)
        theta = np.arcsin(np.clip(u[:, 2], -1, 1))
        beam = np.abs(theta[:, None] - self.elevations).argmin(axis=1)
        az_m = np.maximum(r * rho * self.azimuth_step, m["minimum_sampling_m"])
        el_m = np.maximum(r * self.elevation_step[beam], m["minimum_sampling_m"])
        scales = np.column_stack((np.full(n, m["radial_scale_m"]), az_m, el_m))
        conditions = np.column_stack((np.log(r), u, np.log(az_m), np.log(el_m), delta[inverse], valid[inverse]))

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
        arrays = dict(xyzi=xyzi, source_slot=slots, point_offset=offset.astype(np.float32),
            condition=conditions.astype(np.float32), basis=basis.astype(np.float32),
            sensing_scale=scales.astype(np.float32), neighbors=indices,
            geometry_inverse=inverse, voxel_inverse=voxel_inverse,
            voxel_xyzi=voxel_xyzi.astype(np.float32), grid_coord=cells.astype(np.int32))
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
        self.relations = nn.ModuleList([RelationLayer(c) for _ in range(m["relation_layers"])])
        self.base_head = _mlp(2 * c + 8, 128, 1)
        self.relation_head = _mlp(2 * c + 8, 128, 1)

    def forward(self, scan, query=None, *, return_features=False, trace=None):
        n, m = len(scan["xyzi"]), self.config
        all_rows = torch.arange(n, device=scan["xyzi"].device)
        query = all_rows if query is None else query
        if query.ndim != 1 or query.dtype != torch.long or torch.any((query < 0) | (query >= n)):
            raise ValueError("query rows must address real returns in this complete scan")
        if not n or not len(query):
            if return_features:
                raise ValueError("shared-feature diagnostics require a nonempty real-point query")
            return scan["xyzi"][:0, 0] + self.base_head[-1].weight.sum() * 0
        # Both declared recipes preserve STU intensity, including values above one.
        # Pretraining changes initialization, not the frozen raw-return representation.
        voxels = scan["voxel_xyzi"]
        encoded = self.backbone(dict(feat=voxels, coord=voxels[:, :3], grid_coord=scan["grid_coord"],
            offset=torch.tensor([len(voxels)], dtype=torch.long, device=voxels.device)))
        h = self.context(encoded.feat)[scan["voxel_inverse"]]
        z = self.point(torch.cat((scan["xyzi"], scan["point_offset"]), -1))
        # Expose the shared graph nodes before relation updates, without detaching them.
        features = dict(context=h, point=z) if return_features else None
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

    @torch.no_grad()
    def predict(self, source, transform=None, *, prepared=None, components=False):
        if self.training:
            raise ValueError("prediction requires model.eval()")
        if prepared is None:
            prepared = transform(source)
        elif (not np.array_equal(prepared["source_slot"], source.real_slots)
              or not np.array_equal(prepared["xyzi"], source.xyzi[source.real_slots])):
            raise ValueError("prepared inference input differs from the complete source returns")
        scan = to_device(prepared, next(self.parameters()).device)
        output = self(scan, return_features=components)
        # Components share one complete forward and the same physical return identities.
        scores = ({name: output[key] for name, key in
                   (("base", "base_score"), ("relation", "relation_score"), ("final", "score"))}
                  if components else {"final": output})
        result = {}
        for name, values in scores.items():
            result[name] = FramePrediction(source.partition, source.sequence_id, source.frame_id,
                                           source.real_slots, values.cpu().numpy())
            result[name].validate(source)
        return result if components else result["final"]
