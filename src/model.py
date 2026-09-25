"""V4 full-return LitePT-S segmentation with sampling-conditioned context."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import segment_csr

from .data import file_sha256
from .normal import (NORMAL_MODES, NormalField, Compatibility, joint_nll,
                     SCALES, RAY_CHUNK, LOWER, UPPER)
from vendor.litept.model import LitePT, Point


LITEPT_COMMIT = "436d04801c8151faebe66a1b2d368a9711e7e6aa"
WEIGHTS_REVISION = "a8e76e92efbb2061639f5c683968bc5d248ee002"
WEIGHTS_SHA256 = "95f151f6edcfbf315cd06df6afd261f2a2fde300d3c693dd26b1305d642ecc30"
CHANNELS = (36, 72, 144, 252, 504)
GRID_SIZE = .05
POINT_CHUNK = 65536
RELATION_CHUNK = 4096
RELATION_MODES = ("local_attention", "relation", "relation_no_condition", "relation_no_difference")
NORMAL_VERSION = "AJAE-normal-evidence"
NORMAL_ARCHITECTURE = "multimode48_blind_density"
NORMAL_SCORE_VERSION = "joint_density_logtail"
NORMAL_VARIANTS = ("joint", "semantic", "separate")
NORMAL_LOSS_WEIGHTS = dict(classification=1., semantic=.5, normal=.05, geometry=.2, context=.2)
CALIBRATION_PROBABILITIES = np.r_[np.linspace(0., .9, 33), 1 - np.geomspace(.1, 1e-4, 97)[1:]]
CALIBRATION_LEVELS = -np.log1p(-CALIBRATION_PROBABILITIES)


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden, eps=1e-5),
                         nn.GELU(), nn.Linear(hidden, outputs))


def voxelize(xyzi, *, official=False):
    """Preserve every point identity while constructing the selected voxel input."""
    xyzi = np.asarray(xyzi)
    if xyzi.dtype != np.float32 or xyzi.ndim != 2 or xyzi.shape[1] != 4:
        raise ValueError("input must be float32[N,4]")
    if not len(xyzi) or not np.isfinite(xyzi).all() or np.any(~np.any(xyzi[:, :3] != 0, axis=1)):
        raise ValueError("voxelization requires finite real returns, with no empty ray slots")
    grid = np.floor(xyzi[:, :3].astype(np.float64) / GRID_SIZE).astype(np.int64)
    # One stable lexicographic sort preserves both voxel and within-voxel sum order.
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))
    ordered_grid = grid[order]
    starts = np.r_[True, np.any(ordered_grid[1:] != ordered_grid[:-1], axis=1)]
    pointer = np.r_[np.flatnonzero(starts), len(grid)].astype(np.int64)
    counts = np.diff(pointer)
    unique = ordered_grid[pointer[:-1]]
    inverse = np.empty(len(grid), dtype=np.int64)
    inverse[order] = np.cumsum(starts) - 1
    mean = (np.add.reduceat(xyzi[order].astype(np.float64), pointer[:-1], axis=0)
            / counts[:, None])
    if official:
        # Deterministic single-view representatives, not the official multi-view inference.
        groups = inverse[order]
        distance = np.square(xyzi[order, :3].astype(np.float64) - mean[groups, :3]).sum(1)
        nearest = np.minimum.reduceat(distance, pointer[:-1])
        candidates = np.where(distance == nearest[groups], order, len(xyzi))
        representatives = np.minimum.reduceat(candidates, pointer[:-1])
        voxel_xyzi = xyzi[representatives]
        shift = unique.min(axis=0)
    else:
        voxel_xyzi = mean.astype(np.float32)
        # A multiple of 16 preserves the sensor-origin grid at all four pooling steps.
        shift = (unique.min(axis=0) // 16) * 16
    return dict(xyzi=torch.from_numpy(xyzi), grid=torch.from_numpy(unique - shift),
                voxel_xyzi=torch.from_numpy(voxel_xyzi), inverse=torch.from_numpy(inverse),
                order=torch.from_numpy(order), pointer=torch.from_numpy(pointer),
                offset=torch.from_numpy(((xyzi[:, :3].astype(np.float64)
                                          - (grid + .5) * GRID_SIZE) / GRID_SIZE).astype(np.float32)))


def relation_neighbors(xyz, neighbors=8):
    """Union of spatial and ray-direction kNN; no labels, padding returns or free-space inference."""
    xyz = np.asarray(xyz, dtype=np.float64)
    count = len(xyz)
    width = min(neighbors + 1, count)
    direction = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12)
    indices = []
    for coordinates in (xyz, direction):
        _, nearest = cKDTree(coordinates).query(coordinates, k=list(range(1, width + 1)), workers=1)
        nearest[nearest == np.arange(count)[:, None]] = count
        # Stable compaction removes self even when coincident directions precede it.
        order = np.argsort(nearest == count, axis=1, kind="stable")
        indices.append(np.take_along_axis(nearest, order, axis=1)[:, :min(neighbors, count - 1)])
    joined = np.concatenate(indices, axis=1)
    # Sorting also removes overlap between spatial and angular neighbors.
    joined.sort(axis=1)
    valid = joined != count
    if joined.shape[1] > 1:
        valid[:, 1:] &= joined[:, 1:] != joined[:, :-1]
    result = np.full((count, 2 * neighbors), -1, dtype=np.int64)
    result[:, :joined.shape[1]] = np.where(valid, joined, -1)
    return torch.from_numpy(result)


def prepare_scan(sample, *, relations=False, official=False):
    result = voxelize(sample["xyzi"], official=official)
    if relations:
        result["neighbors"] = relation_neighbors(result["voxel_xyzi"][:, :3].numpy())
    for name in ("targets", "slots"):
        result[name] = torch.from_numpy(sample[name])
    result["slot_count"], result["index"] = sample["slot_count"], sample["index"]
    return result


def to_device(sample, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor)
            else to_device(v, device) if isinstance(v, dict) else v
            for k, v in sample.items()}


class FrozenPerception(nn.Module):
    """Fixed official nuScenes semantics and point-aligned multilevel features."""

    def __init__(self, pretrained="assets/nuscenes.pth"):
        super().__init__()
        self.backbone = LitePT(shuffle_orders=False, fp32_attention=True)
        self.seg_head = nn.Linear(72, 16)
        if file_sha256(pretrained) != WEIGHTS_SHA256:
            raise ValueError("checkpoint differs from the pinned official nuScenes LitePT-S weights")
        saved = torch.load(pretrained, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
        self.load_state_dict({key.removeprefix("module."): value for key, value in saved.items()}, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Freezing weights also requires fixed normalization statistics and stochastic layers.
        return super().train(False)

    @torch.no_grad()
    def encode(self, sample, indices=None):
        inverse = sample["inverse"]
        if indices is not None:
            if not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.dtype != torch.long:
                raise ValueError("point indices must be a one-dimensional int64 tensor")
            inverse = inverse[indices]
        with torch.autocast(sample["voxel_xyzi"].device.type, enabled=False):
            xyzi = sample["voxel_xyzi"].float()
            point = Point(coord=xyzi[:, :3], feat=xyzi, grid_coord=sample["grid"],
                          grid_size=GRID_SIZE,
                          offset=torch.tensor([len(xyzi)], device=xyzi.device))
            point.sparsify()
            point = self.backbone.embedding(point)
            ancestry = inverse
            features = []
            for level, encoder in enumerate(self.backbone.enc):
                point = encoder(point)
                if level:
                    ancestry = point.pooling_inverse[ancestry]
                if level in (0, 2):
                    # Gather before unpooling replaces the parent Point's feature field.
                    features.append(point.feat[ancestry].float())
            point = self.backbone.dec(point)
            features.append(point.feat[inverse].float())
            return dict(features=torch.cat(features, dim=-1),
                        logits=self.seg_head(point.feat.float())[inverse].float())


class FrozenSupport(nn.Module):
    """Learn normal support without changing the official 16-class perception model."""

    def __init__(self, scorer=None, pretrained="assets/nuscenes.pth"):
        super().__init__()
        from .normal import FeatureSupport
        self.mode = "frozen_support"
        self.perception = FrozenPerception(pretrained)
        self.scorer = FeatureSupport() if scorer is None else scorer

    def forward(self, sample):
        with torch.autocast(sample["voxel_xyzi"].device.type, enabled=False):
            encoded = self.perception.encode(sample)
            return self.scorer(encoded["features"], sample["conditions"].float())


class Interaction(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("attention", "fusion"):
            raise ValueError(mode)
        self.mode = mode
        self.position = mlp(4, 32, 64)
        self.projections = nn.ModuleList(nn.Linear(c, 64) for c in CHANNELS)
        self.scale = nn.Parameter(torch.zeros(5, 64))
        self.norm = nn.LayerNorm(64, eps=1e-5)
        if mode == "attention":
            self.query = nn.Linear(128, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)
            self.relative = mlp(3, 16, 4)
            self.output = nn.Linear(64, 64)
        else:
            self.fusion = mlp(463, 64, 64)

    def project(self, features):
        # Reuse keys/values across every point sharing a voxel.
        levels = [self.norm(project(feat) + self.scale[i])
                  for i, (project, feat) in enumerate(zip(self.projections, features))]
        if self.mode == "attention":
            return tuple(self.key(x) for x in levels), tuple(self.value(x) for x in levels)
        return tuple(levels), ()

    def forward(self, xyz, sampling, indices, coords, keys, values):
        distance = torch.linalg.vector_norm(xyz.float(), dim=-1, keepdim=True)
        sensor = self.position(torch.cat((distance / 50, xyz / distance.clamp_min(1e-12)), -1))
        relative = torch.stack([(xyz - c[idx]) / (GRID_SIZE * 2**level)
                                for level, (c, idx) in enumerate(zip(coords, indices))], 1)
        gathered = torch.stack([k[idx] for k, idx in zip(keys, indices)], 1)
        if self.mode == "fusion":
            return self.fusion(torch.cat((sampling, sensor, gathered.flatten(1), relative.flatten(1)), -1))
        query = self.query(torch.cat((sampling, sensor), -1)).reshape(-1, 4, 16)
        key = gathered.reshape(-1, 5, 4, 16)
        value = torch.stack([v[idx] for v, idx in zip(values, indices)], 1).reshape(-1, 5, 4, 16)
        # FP32 softmax protects the five-scale probabilities in mixed precision.
        logits = (query[:, None].float() * key.float()).sum(-1) / 4
        weights = (logits + self.relative(relative).float()).softmax(dim=1)
        result = (weights[..., None] * value.float()).sum(1).flatten(1)
        return self.output(result.to(sampling.dtype))


class Conditional(nn.Module):
    """Update one voxel state by sampling-conditioned attention to six context scales."""

    def __init__(self):
        super().__init__()
        self.detail = nn.Linear(128, 64)
        self.sensor = mlp(4, 32, 64)
        self.projections = nn.ModuleList(nn.Linear(c, 64) for c in (*CHANNELS, 72))
        self.scale = nn.Parameter(torch.zeros(6, 64))
        self.relative = mlp(3, 32, 64)
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            query=nn.Linear(64, 64), key=nn.Linear(64, 64), value=nn.Linear(64, 64),
            output=nn.Linear(64, 64), norm=nn.LayerNorm(64),
            feedforward=mlp(64, 128, 64), final_norm=nn.LayerNorm(64),
        )) for _ in range(2))

    def forward(self, pooled, xyz, indices, coords, features):
        distance = torch.linalg.vector_norm(xyz.float(), dim=-1, keepdim=True)
        state = self.detail(pooled) + self.sensor(torch.cat((distance / 50, xyz / distance.clamp_min(1e-12)), -1))
        tokens = []
        for level, (project, feat, coord, index) in enumerate(zip(self.projections, features, coords, indices)):
            relative = (coord[index] - xyz) / (GRID_SIZE * 2**min(level, 4))
            tokens.append(project(feat)[index] + self.scale[level] + self.relative(relative))
        tokens = torch.stack(tokens, 1)
        for layer in self.layers:
            query = layer["query"](state).reshape(-1, 4, 16)
            keys = layer["key"](tokens).reshape(-1, 6, 4, 16)
            values = layer["value"](tokens).reshape(-1, 6, 4, 16)
            # Six tokens per voxel: no quadratic attention across all scene points.
            weights = ((query[:, None].float() * keys.float()).sum(-1) / 4).softmax(1)
            context = (weights[..., None] * values.float()).sum(1).flatten(1)
            state = layer["norm"](state + layer["output"](context.to(state.dtype)))
            state = layer["final_norm"](state + layer["feedforward"](state))
        return state


class Relation(nn.Module):
    """Matched edge inputs/capacity; compare normalized attention with signed differences."""

    def __init__(self, mode):
        super().__init__()
        if mode not in RELATION_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.query = nn.Linear(64, 128)
        self.key = nn.Linear(64, 128)
        self.value = nn.Linear(64, 64)
        self.geometry = mlp(3, 32, 72)
        self.condition = mlp(10, 32, 8)
        self.output = nn.Linear(128, 64)
        self.norm = nn.LayerNorm(64)

    def messages(self, state, xyz, radii, rays, neighbors, query, keys, values, begin, end):
        indices = neighbors[begin:end]
        valid = indices >= 0
        indices = indices.clamp_min(0)
        origin, other = xyz[begin:end, None].float(), xyz[indices].float()
        delta = other - origin  # Sensor coordinates in metres, before any learned scaling.
        ri, rj = radii[begin:end, None].expand_as(radii[indices]), radii[indices]
        ui, uj = rays[begin:end, None].expand_as(rays[indices]), rays[indices]
        condition = torch.cat((ri / 50, rj / 50, ui, uj, (rj - ri) / 50, (ui * uj).sum(-1, keepdim=True)), -1)
        if self.mode == "relation_no_condition":
            condition = torch.zeros_like(condition)
        geometry = self.geometry(delta)
        logits = (query[begin:end, None].float() * keys[indices].float()).reshape(
            end - begin, indices.shape[1], 8, 16).sum(-1) / 4
        logits = logits + geometry[..., :8].float() + self.condition(condition).float()
        # Mask padding after softmax as well, so a singleton scan has zero messages.
        weights = logits.masked_fill(~valid[..., None], -1e9).softmax(1) * valid[..., None]
        common = (weights[..., :4, None] * values[indices].float().reshape(
            end - begin, indices.shape[1], 4, 16)).sum(1).flatten(1)
        difference = F.gelu(values[indices] - values[begin:end, None] + geometry[..., 8:])
        if self.mode != "local_attention":
            # Bounded signed contributions retain deviations instead of averaging them away.
            weights = torch.tanh(logits) * valid[..., None] / valid.sum(1).clamp_min(1)[:, None, None]
        contrast = (weights[..., 4:, None] * difference.float().reshape(
            end - begin, indices.shape[1], 4, 16)).sum(1).flatten(1)
        if self.mode == "relation_no_difference":
            contrast = contrast * 0
        return self.norm(state[begin:end] + self.output(torch.cat((common, contrast), -1).to(state.dtype)))

    def forward(self, state, xyz, neighbors, *, recompute=True):
        # Project node features once; only bounded edge chunks are materialized.
        query, keys, values = self.query(state), self.key(state), self.value(state)
        radii = torch.linalg.vector_norm(xyz.float(), dim=-1, keepdim=True).clamp_min(1e-12)
        rays = xyz.float() / radii
        result = []
        for begin in range(0, len(state), RELATION_CHUNK):
            end = min(begin + RELATION_CHUNK, len(state))
            def block(s, q, k, v, start=begin, stop=end):
                return self.messages(s, xyz, radii, rays, neighbors, q, k, v, start, stop)
            if self.training and recompute and torch.is_grad_enabled():
                result.append(checkpoint(block, state, query, keys, values, use_reentrant=False))
            else:
                result.append(block(state, query, keys, values))
        return torch.cat(result)


class NormalHypothesis(nn.Module):
    """One class evidence jointly supports normal semantics and unknown detection."""

    def __init__(self, variant="joint"):
        super().__init__()
        from .normal import SemanticHypotheses
        if variant not in NORMAL_VARIANTS:
            raise ValueError(f"unknown normal perception variant: {variant}")
        self.variant = variant
        self.mode = "normal_hypothesis"
        self.backbone = LitePT(shuffle_orders=False, fp32_attention=True)
        self.detail = mlp(7, 32, 32)
        self.embedding = mlp(104, 96, 48)
        self.hypotheses = SemanticHypotheses()
        self.appearance_modes = nn.Parameter(torch.randn(19, 4, 48) * .05)
        self.register_buffer("calibration", torch.zeros(len(CALIBRATION_PROBABILITIES)))
        self.register_buffer("calibrated", torch.tensor(False))

    @staticmethod
    def validate_checkpoint(saved, *, require_calibrated=False):
        config = saved.get("config", {})
        if (saved.get("version") != NORMAL_VERSION
                or config.get("architecture") != NORMAL_ARCHITECTURE
                or config.get("score_version") != NORMAL_SCORE_VERSION
                or config.get("variant") not in NORMAL_VARIANTS):
            raise ValueError("incompatible normal model: initialize this method from the official nuScenes backbone")
        state = saved.get("model", {})
        if state.get("calibration", torch.empty(0)).shape != (len(CALIBRATION_PROBABILITIES),):
            raise ValueError("incompatible normal reference shape")
        if require_calibrated and not bool(state.get("calibrated", False)):
            raise ValueError("normal reference calibration is required for anomaly inference")

    def load_pretrained(self, path):
        # Only the official backbone initializes this new normal-only experiment.
        return Segmentor.load_pretrained(self, path)

    def loss_weights(self):
        weights = dict(NORMAL_LOSS_WEIGHTS)
        if self.variant == "semantic":
            weights.update(geometry=0., context=0.)
        return weights

    def semantic(self, sample, *, modes=False):
        point = self.backbone(dict(coord=sample["voxel_xyzi"][:, :3], feat=sample["voxel_xyzi"],
            grid_coord=sample["grid"], grid_size=GRID_SIZE,
            offset=torch.tensor([len(sample["grid"])], device=sample["xyzi"].device)))
        xyzi = sample["xyzi"]
        detail = self.detail(torch.cat((xyzi[:, :3] / 50, xyzi[:, 3:4], sample["offset"]), -1))
        features = self.embedding(torch.cat((point.feat[sample["inverse"]], detail), -1))
        # Several equally scaled appearance modes preserve normal intra-class diversity.
        centers = F.normalize(self.hypotheses.queries[:, None] + self.appearance_modes, dim=-1) * (48 ** .5)
        centers = centers.flatten(0, 1)
        distance = (features.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None]
                    - 2 * features @ centers.T).clamp_min(0) / 48
        costs = -distance.reshape(-1, 19, 4) / .2
        energy = -(torch.logsumexp(costs, -1) - np.log(4.))
        return (energy, costs[sample["queries"]].softmax(-1)) if modes else energy

    def components(self, sample, indices=None, *, semantic_energy=None):
        from .normal import geometry_energy
        semantic_energy = self.semantic(sample) if semantic_energy is None else semantic_energy
        if indices is None:
            indices = torch.arange(len(sample["xyzi"]), device=semantic_energy.device)
        semantic_energy = semantic_energy[indices]
        prediction = None if self.variant == "semantic" else self.hypotheses(sample["observation"], indices)
        geometry = torch.zeros_like(semantic_energy) if prediction is None else geometry_energy(prediction)
        energy = semantic_energy + geometry
        return dict(logits=-energy, semantic_energy=semantic_energy, geometry_energy=geometry,
                    energy=energy, raw_score=energy.amin(-1), prediction=prediction)

    def loss(self, sample):
        from .normal import allowed_loss, hypothesis_loss, observation_diagnostics
        indices, allowed = sample["queries"], sample["allowed"]
        if self.training:
            semantic_energy = self.semantic(sample)
        else:
            semantic_energy, appearance = self.semantic(sample, modes=True)
        parts = self.components(sample, indices if self.training else None, semantic_energy=semantic_energy)
        if self.training:
            logits, prediction = parts["logits"], parts["prediction"]
        else:
            # Development measures the exact full-point classifier used at inference.
            self.development_prediction = parts["logits"].argmax(-1).detach()
            self.development_semantic_prediction = semantic_energy.argmin(-1).detach()
            self.development_raw_score = parts["raw_score"].detach()
            logits = parts["logits"][indices]
            prediction = ({key: value[indices] for key, value in parts["prediction"].items()}
                          if parts["prediction"] is not None else None)
            self.development_diagnostics = observation_diagnostics(
                prediction, sample["observation"], indices, allowed[indices], appearance)
        # The control removes observation costs from supervised class competition.
        # Shared class parameters and predictive losses remain; inference uses E.
        if self.variant != "joint":
            logits = -semantic_energy[indices]
        classification = allowed_loss(logits, allowed[indices])
        semantic = allowed_loss(-semantic_energy, allowed)
        valid = allowed[indices].any(1)
        if bool(valid.any()):
            admitted = allowed[indices][valid]
            support = (-semantic_energy[indices][valid]).masked_fill(~admitted, -torch.inf)
            normal = (-torch.logsumexp(support, -1) + admitted.sum(-1).float().log()).mean()
        else:
            normal = semantic_energy.sum() * 0
        geometry, context = (hypothesis_loss(prediction, allowed[indices]) if prediction is not None
                             else (semantic_energy.sum() * 0, semantic_energy.sum() * 0))
        terms = dict(classification=classification, semantic=semantic, normal=normal, geometry=geometry, context=context)
        loss = sum(self.loss_weights()[key] * value for key, value in terms.items())
        details = {key: value.detach() for key, value in terms.items()}
        details.update(supervised=allowed.any(1).sum().detach(), queries=len(indices),
                       supported=prediction["supported"].sum().detach() if prediction is not None else 0)
        return loss, details

    def calibrate_score(self, raw_score):
        if not bool(self.calibrated):
            raise ValueError("normal reference calibration must be fitted before anomaly inference")
        knots = self.calibration.to(raw_score)
        if not bool(torch.isfinite(knots).all()) or bool((knots[1:] < knots[:-1]).any()):
            raise ValueError("normal reference must be finite and nondecreasing")
        levels = raw_score.new_tensor(CALIBRATION_LEVELS)
        # Merge tied quantiles using the right empirical tail level. No epsilon-width
        # intervals: those can magnify a constant reference into infinite scores.
        knots, counts = torch.unique_consecutive(knots, return_counts=True)
        levels = levels[counts.cumsum(0) - 1]
        if len(knots) == 1:
            return levels[0] + (raw_score - knots[0]) / knots[0].abs().clamp_min(1)
        indices = torch.searchsorted(knots, raw_score.contiguous()).clamp(1, len(knots) - 1)
        fraction = (raw_score - knots[indices - 1]) / (knots[indices] - knots[indices - 1])
        # A shared monotone transform retains the joint ranking and extrapolates tails.
        return levels[indices - 1] + fraction * (levels[indices] - levels[indices - 1])

    def predict(self, sample):
        with torch.autocast(sample["xyzi"].device.type, enabled=False):
            parts = self.components(sample)
            return dict(semantic=parts["energy"].argmin(-1),
                        semantic_only=parts["semantic_energy"].argmin(-1),
                        raw_score=parts["raw_score"], score=self.calibrate_score(parts["raw_score"]),
                        confidence=(-parts["energy"]).softmax(-1).amax(-1),
                        semantic_confidence=(-parts["semantic_energy"]).softmax(-1).amax(-1),
                        semantic_score=parts["semantic_energy"].amin(-1))

    def forward(self, sample):
        return self.predict(sample)["score"]


class Segmentor(nn.Module):
    def __init__(self, mode="field", *, recompute=True):
        super().__init__()
        if mode in ("density", "geometry"):
            raise ValueError("retired normal predictor; use its historical code revision to load that checkpoint")
        self.mode, self.recompute = mode, recompute
        self.backbone = LitePT(shuffle_orders=False, fp32_attention=mode == "field")
        self.detail = mlp(7, 64, 64)
        self.adapter = mlp(128, 64, 36)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        if mode == "conditional" or mode in (*RELATION_MODES, *NORMAL_MODES):
            self.conditional = Conditional()
            self.point_detail = nn.Linear(64, 64)
            self.point_position = mlp(3, 32, 64)
            self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        else:
            self.sampling = mlp(100, 64, 64)
            self.context = nn.Linear(72, 64)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.interaction = None
        self.interaction_weight = None
        if mode not in ("base", "conditional", *RELATION_MODES, *NORMAL_MODES):
            self.add_interaction(mode)
        self.relation = None
        if mode in RELATION_MODES:
            # Preserve all shared N1 initialization and the post-construction random stream.
            with torch.random.fork_rng(devices=[]):
                self.relation = Relation(mode)
        self.normal = None
        if mode in NORMAL_MODES:
            with torch.random.fork_rng(devices=[]):
                self.normal = NormalField(recompute=recompute)
                self.compatibility = Compatibility()
                self.head = nn.Sequential(nn.Linear(163, 128), nn.LayerNorm(128), nn.GELU(),
                                          nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 1))

    def add_interaction(self, mode):
        if self.interaction is not None:
            raise ValueError("only one interaction layer is permitted")
        self.interaction = Interaction(mode)
        self.interaction_weight = nn.Linear(64, 64, bias=False)
        nn.init.zeros_(self.interaction_weight.weight)
        self.mode = mode

    def _checkpoint(self, function, *args):
        if self.training and self.recompute and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)
        return function(*args)

    def load_pretrained(self, path):
        path = Path(path)
        digest = file_sha256(path)
        if digest != WEIGHTS_SHA256:
            raise ValueError("checkpoint differs from the pinned official nuScenes LitePT-S weights")
        saved = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        state = {k.removeprefix("module.backbone."): v for k, v in saved.items()
                 if k.startswith("module.backbone.")}
        ignored = sorted(k for k in saved if not k.startswith("module.backbone."))
        if ignored != ["module.seg_head.bias", "module.seg_head.weight"]:
            raise ValueError(f"unexpected pretrained parameters: {ignored}")
        self.backbone.load_state_dict(state, strict=True)
        return dict(sha256=digest, revision=WEIGHTS_REVISION, loaded=sorted(state),
                    removed=ignored, trainable=sum(p.numel() for p in self.backbone.parameters()))

    def forward(self, sample, *, normal_loss=False, query_indices=None):
        if query_indices is not None and self.mode != "field":
            raise ValueError("selected point queries require the field model")
        # FP32 avoids amplification of sparse-kernel rounding at BF16 boundaries.
        # Legacy checkpoints keep their original externally selected precision.
        with torch.autocast(sample["xyzi"].device.type,
                            enabled=torch.is_autocast_enabled(sample["xyzi"].device.type) and self.mode != "field"):
            fields = self.normal(sample["observation"]) if self.normal is not None else None
            xyzi, inverse = sample["xyzi"], sample["inverse"]
            queries = (torch.arange(len(xyzi), device=xyzi.device) if query_indices is None
                       else query_indices)
            if queries.ndim != 1 or queries.dtype != torch.long:
                raise ValueError("point queries must be a vector of original point indices")
            detail = torch.cat([
                self._checkpoint(self.detail, torch.cat((xyzi[start:start + POINT_CHUNK, :3] / 50,
                                                          xyzi[start:start + POINT_CHUNK, 3:4],
                                                          sample["offset"][start:start + POINT_CHUNK]), -1))
                for start in range(0, len(xyzi), POINT_CHUNK)])
            ordered = detail[sample["order"]].float()
            with torch.autocast(xyzi.device.type, enabled=False):
                pooled = torch.cat((segment_csr(ordered, sample["pointer"], reduce="mean"),
                                    segment_csr(ordered, sample["pointer"], reduce="max")), -1)
            point = Point(coord=sample["voxel_xyzi"][:, :3], feat=sample["voxel_xyzi"],
                          grid_coord=sample["grid"], grid_size=GRID_SIZE,
                          offset=torch.tensor([len(sample["grid"])], device=xyzi.device))
            point.sparsify()
            point = self.backbone.embedding(point)
            point.feat = point.feat + self.adapter(pooled)
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
            features, coords, ancestors, voxel_ancestors = [], [], [], []
            ancestry = inverse
            voxel_ancestry = torch.arange(len(sample["grid"]), device=xyzi.device)
            for level, encoder in enumerate(self.backbone.enc):
                point = encoder(point)
                if level:
                    ancestry = point.pooling_inverse[ancestry]
                    voxel_ancestry = point.pooling_inverse[voxel_ancestry]
                # Store tensors before decoder mutation of the Point containers.
                features.append(point.feat)
                coords.append(point.coord)
                ancestors.append(ancestry)
                voxel_ancestors.append(voxel_ancestry)
            point = self.backbone.dec(point)
            if self.mode == "conditional" or self.mode in (*RELATION_MODES, *NORMAL_MODES):
                unified = self._checkpoint(self.conditional, pooled, sample["voxel_xyzi"][:, :3],
                    voxel_ancestors + [voxel_ancestors[0]], coords + [point.coord], features + [point.feat])
                if self.relation is not None:
                    unified = self.relation(unified, sample["voxel_xyzi"][:, :3], sample["neighbors"],
                                            recompute=self.recompute)
                output, probabilities = [], []
                chunk = RAY_CHUNK if fields is not None else POINT_CHUNK
                for chosen in queries.split(chunk):
                    if not len(chosen):
                        continue
                    def score_points(e, context, offset, indices=chosen):
                        state = context + self.point_detail(e) + self.point_position(offset)
                        if fields is not None:
                            # Casting BEFORE the compatibility/head computation preserves
                            # FP32 ranking precision; a cast of BF16 logits would not.
                            with torch.autocast(xyzi.device.type, enabled=False):
                                hidden, prob = self.compatibility(state.float(), sample["observation"], fields, indices)
                                return self.head(hidden).squeeze(-1), prob
                        return self.head(state).squeeze(-1).float()
                    result = self._checkpoint(score_points, detail[chosen], unified[inverse[chosen]],
                                              sample["offset"][chosen])
                    output.append(result[0] if fields is not None else result)
                    if fields is not None and normal_loss and sample.get("normal_training", False):
                        probabilities.append(result[1])
                scores = torch.cat(output) if output else unified.sum(-1)[:0]
                if normal_loss:
                    auxiliary = scores.sum() * 0
                    if "normal_reference" in sample:
                        reference = sample["normal_reference"]
                        auxiliary = self.normal.likelihood(reference["observation"], reference["targets"])
                    elif probabilities:
                        observation = sample["observation"]
                        selected = ((sample["targets"] == 0) & (observation["distance"] >= LOWER)
                                    & (observation["distance"] <= UPPER))
                        # The scene encoders still see every input point. Only the
                        # independent final queries are sparse; block targets stay intact.
                        prob = torch.cat(probabilities)
                        prob = prob.new_zeros((len(xyzi), len(SCALES), prob.shape[-1])).index_copy(0, queries, prob)
                        covered = torch.zeros_like(selected).index_fill(0, queries, True)
                        if bool((selected & ~covered).any()):
                            raise ValueError("normal auxiliary queries omitted supervised normal points")
                        with torch.autocast(xyzi.device.type, enabled=False):
                            auxiliary = sum(joint_nll(prob[:, i], fields[str(size)]["log_weights"],
                                                      observation["grids"][str(size)], selected)
                                            for i, size in enumerate(SCALES)) / len(SCALES)
                    return scores, auxiliary
                return scores
            context = self.context(point.feat)
            keys, values = self.interaction.project(features) if self.interaction is not None else ((), ())

            def score(start, end, e):
                indices = [a[start:end] for a in ancestors]
                sampling = self.sampling(torch.cat((e, features[0][indices[0]]), -1))
                hidden = self.head[0](torch.cat((sampling, context[indices[0]]), -1))
                if self.interaction is not None:
                    interaction = self.interaction(xyzi[start:end, :3], sampling, indices, coords, keys, values)
                    hidden = hidden + self.interaction_weight(interaction)
                return self.head[2](self.head[1](hidden)).squeeze(-1).float()

            output = []
            for start in range(0, len(xyzi), POINT_CHUNK):
                end = min(start + POINT_CHUNK, len(xyzi))
                # Bind block bounds; checkpoint recomputation occurs after this loop.
                def block(e, begin=start, stop=end):
                    return score(begin, stop, e)
                output.append(self._checkpoint(block, detail[start:end]))
            return torch.cat(output)


def scatter_scores(scores, slots, slot_count):
    if scores.ndim != 1 or len(scores) != len(slots):
        raise ValueError("each real return must have exactly one prediction")
    output = scores.new_zeros(slot_count)
    return output.scatter(0, slots.long(), scores)


def balanced_loss(logits, targets, counts, control_mask=None):
    """Effective-batch means, optionally reserving normal mass for insertion controls."""
    if counts.dtype != torch.int64 or counts.shape != (2,):
        raise ValueError("class counts must be int64[normal, anomaly]")
    if control_mask is not None:
        if control_mask.shape != targets.shape or control_mask.dtype != torch.bool:
            raise ValueError("normal-control mask must match point targets")
        if bool((control_mask & (targets != 0)).any()):
            raise ValueError("normal controls must be verified normal targets")
        # Preserve equal normal/anomaly mass, while giving inserted normal
        # objects half of the normal mass instead of diluting them in road points.
        groups = ((targets == 1, -1, .5), (control_mask, 1, .25),
                  ((targets == 0) & ~control_mask, 1, .25))
        result, mass = logits.float().sum() * 0, 0.
        for selected, sign, weight in groups:
            if bool(selected.any()):
                result = result + weight * F.softplus(sign * logits[selected].float()).mean()
                mass += weight
        if not mass:
            raise ValueError("training scan has no supervised points")
        return result / mass
    present = int((counts > 0).sum())
    if not present:
        raise ValueError("effective batch contains no supervised points")
    result = logits.float().sum() * 0
    for label, sign in ((0, 1), (1, -1)):
        if counts[label] > 0:
            result = result + F.softplus(sign * logits[targets == label].float()).sum() / (present * counts[label])
    return result


class RecallThreshold(torch.autograd.Function):
    """Implicit derivative of mean(sigmoid((positive - t) / tau)) = recall."""

    @staticmethod
    def forward(ctx, positive, tau, recall):
        low, high = positive.min() - 32 * tau, positive.max() + 32 * tau
        for _ in range(40):
            middle = (low + high) / 2
            above = torch.sigmoid((positive - middle) / tau).mean() > recall
            low, high = torch.where(above, middle, low), torch.where(above, high, middle)
        threshold = (low + high) / 2
        scaled = (positive - threshold) / tau
        # log-space normalization also handles a sharply separated positive tail.
        derivative = (F.logsigmoid(scaled) + F.logsigmoid(-scaled)).softmax(0)
        ctx.save_for_backward(derivative)
        return threshold

    @staticmethod
    def backward(ctx, gradient):
        (derivative,) = ctx.saved_tensors
        return gradient * derivative, None, None


def rank_sample(positive, negative, seed):
    """Uniform positive anchors; certain top negatives plus inverse-probability sampling."""
    generator = torch.Generator(device=positive.device).manual_seed(seed)
    anchors = (torch.randperm(len(positive), device=positive.device, generator=generator)[:256]
               if len(positive) > 256 else torch.arange(len(positive), device=positive.device))
    if len(negative) <= 4096:
        return positive[anchors], negative, torch.ones_like(negative), min(512, len(negative))
    top = negative.detach().topk(512).indices
    remaining = torch.ones(len(negative), device=negative.device, dtype=torch.bool)
    remaining[top] = False
    rest = remaining.nonzero().flatten()
    selected = rest[torch.randperm(len(rest), device=negative.device, generator=generator)[:3584]]
    indices = torch.cat((top, selected))
    weights = torch.cat((negative.new_ones(512), negative.new_full((3584,), len(rest) / 3584)))
    return positive[anchors], negative[indices], weights, 512


def ranking_loss(logits, targets, seed, tau=1., *, auc_weight=.1, fpr95_weight=.1, return_terms=False):
    """One supplied score pool, with all positives and weighted sampled negatives."""
    with torch.autocast(logits.device.type, enabled=False):
        positive, negative = logits[targets == 1].float(), logits[targets == 0].float()
        zero = logits.float().sum() * 0
        if not len(positive) or not len(negative):
            details = dict(ap=zero.detach(), auc=zero.detach(), fpr95=zero.detach(),
                              positive=len(positive), negative=len(negative), anchors=0, negatives=0,
                              top=0, random_weight=0., threshold=None, recall=None)
            if return_terms:
                details["terms"] = dict(ap=zero, auc=zero, fpr95=zero)
            return zero, details
        anchors, sampled, weights, top = rank_sample(positive, negative, seed)
        # Each anchor is a member of P: subtract its self-comparison sigmoid(0).
        positive_rank = .5 + torch.sigmoid((positive[None, :] - anchors[:, None]) / tau).sum(1)
        difference = (sampled[None, :] - anchors[:, None]) / tau
        negative_rank = (torch.sigmoid(difference) * weights).sum(1)
        ap = 1 - (positive_rank / (positive_rank + negative_rank)).mean()
        auc = (F.softplus(difference) * weights).sum(1).mean() / len(negative)
        threshold = RecallThreshold.apply(positive, tau, .95)
        recall = torch.sigmoid((positive - threshold) / tau).mean()
        fpr95 = (torch.sigmoid((sampled - threshold) / tau) * weights).sum() / len(negative)
        details = dict(
            ap=ap.detach(), auc=auc.detach(), fpr95=fpr95.detach(),
            positive=len(positive), negative=len(negative), anchors=len(anchors), negatives=len(sampled),
            top=top, random_weight=(len(negative) - 512) / 3584 if len(negative) > 4096 else 1.,
            threshold=threshold.detach(), recall=recall.detach())
        if return_terms:
            details["terms"] = dict(ap=ap, auc=auc, fpr95=fpr95)
        return ap + auc_weight * auc + fpr95_weight * fpr95, details
