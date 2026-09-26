"""SERVE class support from appearance and target-excluded return verification."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import file_sha256
from vendor.litept.model import LitePT


LITEPT_COMMIT = "436d04801c8151faebe66a1b2d368a9711e7e6aa"
WEIGHTS_REVISION = "a8e76e92efbb2061639f5c683968bc5d248ee002"
WEIGHTS_SHA256 = "95f151f6edcfbf315cd06df6afd261f2a2fde300d3c693dd26b1305d642ecc30"
GRID_SIZE = .05
POINT_CHUNK = 65536
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


def voxelize(xyzi):
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
    voxel_xyzi = mean.astype(np.float32)
    # A multiple of 16 preserves the sensor-origin grid at all four pooling steps.
    shift = (unique.min(axis=0) // 16) * 16
    return dict(xyzi=torch.from_numpy(xyzi), grid=torch.from_numpy(unique - shift),
                voxel_xyzi=torch.from_numpy(voxel_xyzi), inverse=torch.from_numpy(inverse),
                order=torch.from_numpy(order), pointer=torch.from_numpy(pointer),
                offset=torch.from_numpy(((xyzi[:, :3].astype(np.float64)
                                          - (grid + .5) * GRID_SIZE) / GRID_SIZE).astype(np.float32)))


def prepare_scan(sample):
    result = voxelize(sample["xyzi"])
    for name in ("targets", "slots"):
        result[name] = torch.from_numpy(sample[name])
    result["slot_count"], result["index"] = sample["slot_count"], sample["index"]
    return result


def to_device(sample, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor)
            else to_device(v, device) if isinstance(v, dict) else v
            for k, v in sample.items()}


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


def scatter_scores(scores, slots, slot_count):
    if scores.ndim != 1 or len(scores) != len(slots):
        raise ValueError("each real return must have exactly one prediction")
    output = scores.new_zeros(slot_count)
    return output.scatter(0, slots.long(), scores)
