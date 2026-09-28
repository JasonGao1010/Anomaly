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
NORMAL_VERSION = "SERVE-3"
NORMAL_ARCHITECTURE = "multimode48_return_verification"
NORMAL_SCORE_VERSION = "joint_density_logtail"
NORMAL_VARIANTS = ("joint", "semantic", "separate", "standard", "cssr", "target_available",
                   "single_component", "no_nll", "no_compactness")
NORMAL_READOUTS = ("joint", "appearance", "common_density", "independent_minima", "energy", "softmax")
NORMAL_LOSS_WEIGHTS = dict(classification=1., semantic=.5, normal=.05, geometry=.2, context=.2)
CALIBRATION_PROBABILITIES = np.r_[np.linspace(0., .9, 33), 1 - np.geomspace(.1, 1e-4, 97)[1:]]
CALIBRATION_LEVELS = -np.log1p(-CALIBRATION_PROBABILITIES)


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden, eps=1e-5),
                         nn.GELU(), nn.Linear(hidden, outputs))


def voxelize(xyzi, *, rng=None):
    """Preserve every point identity while constructing the selected voxel input."""
    xyzi = np.asarray(xyzi)
    if xyzi.dtype != np.float32 or xyzi.ndim != 2 or xyzi.shape[1] != 4:
        raise ValueError("input must be float32[N,4]")
    if not len(xyzi) or not np.isfinite(xyzi).all() or np.any(~np.any(xyzi[:, :3] != 0, axis=1)):
        raise ValueError("voxelization requires finite real returns, with no empty ray slots")
    grid = np.floor(xyzi[:, :3].astype(np.float64) / GRID_SIZE).astype(np.int64)
    # Stable grouping preserves original point identities, including multiple returns.
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))
    ordered_grid = grid[order]
    starts = np.r_[True, np.any(ordered_grid[1:] != ordered_grid[:-1], axis=1)]
    pointer = np.r_[np.flatnonzero(starts), len(grid)].astype(np.int64)
    counts = np.diff(pointer)
    unique = ordered_grid[pointer[:-1]]
    inverse = np.empty(len(grid), dtype=np.int64)
    inverse[order] = np.cumsum(starts) - 1
    # Use a real return as in pretraining; evaluation fixes the first original point.
    selected = pointer[:-1] if rng is None else pointer[:-1] + rng.integers(counts)
    voxel_xyzi = xyzi[order[selected]]
    # Match the pretrained pooling origin without translating physical coordinates.
    shift = unique.min(axis=0)
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
        self.backbone = LitePT(shuffle_orders=True, fp32_attention=True)
        self.detail = mlp(7, 32, 32)
        self.embedding = mlp(104, 96, 48)
        self.appearance_modes = nn.Parameter(torch.randn(19, 4, 48) * .05)
        # Optional heads must not change common initialization or the training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((torch.initial_seed() + 104729) % (2 ** 63))
            self.hypotheses = SemanticHypotheses(components=1 if variant == "single_component" else 3,
                                               target_available=variant == "target_available")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((torch.initial_seed() + 130363) % (2 ** 63))
            if variant == "standard":
                self.classifier = nn.Linear(48, 19)
            if variant == "cssr":
                self.autoencoders = nn.ModuleList(nn.Sequential(nn.Linear(48, 12), nn.Tanh(),
                                                                nn.Linear(12, 48)) for _ in range(19))
                self.register_buffer("cssr_mean", torch.zeros(19, 48, dtype=torch.float64))
                self.register_buffer("cssr_gram", torch.zeros(19, 48, 48, dtype=torch.float64))
                self.register_buffer("cssr_location", torch.zeros(3, dtype=torch.float64))
                self.register_buffer("cssr_scale", torch.zeros(3, dtype=torch.float64))
                self.register_buffer("cssr_fitted", torch.tensor(False))
        self.register_buffer("calibration", torch.zeros(len(CALIBRATION_PROBABILITIES)))
        self.register_buffer("calibrated", torch.tensor(False))

    def train(self, mode=True):
        super().train(mode)
        # Pooling stages keep their own order setting; evaluation must fix all of them.
        for module in self.backbone.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = mode
        return self

    @staticmethod
    def validate_checkpoint(saved, *, require_calibrated=False, require_fitted=False):
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
        if require_fitted and config.get("variant") == "cssr" and not bool(state.get("cssr_fitted", False)):
            raise ValueError("CSSR normal activation statistics must be fitted before unknown inference")

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
        if self.variant in ("semantic", "standard", "cssr"):
            weights.update(geometry=0., context=0.)
        if self.variant in ("standard", "cssr", "no_compactness"):
            weights["normal"] = 0.
        if self.variant == "standard":
            weights["semantic"] = 0.
        if self.variant == "no_nll":
            weights["geometry"] = 0.
        return weights

    def features(self, sample):
        point = self.backbone(dict(coord=sample["voxel_xyzi"][:, :3], feat=sample["voxel_xyzi"],
            grid_coord=sample["grid"], grid_size=GRID_SIZE,
            offset=torch.tensor([len(sample["grid"])], device=sample["xyzi"].device)))
        xyzi = sample["xyzi"]
        detail = self.detail(torch.cat((xyzi[:, :3] / 50, xyzi[:, 3:4], sample["offset"]), -1))
        return self.embedding(torch.cat((point.feat[sample["inverse"]], detail), -1))

    def reconstruction_error(self, features):
        return torch.stack([(features - autoencoder(features)).abs().sum(-1)
                            for autoencoder in self.autoencoders], -1)

    def feature_energy(self, features, *, modes=False, mode_indices=None):
        if self.variant == "standard":
            return -self.classifier(features)
        if self.variant == "cssr":
            return .1 * self.reconstruction_error(features)
        # Several equally scaled appearance modes preserve normal intra-class diversity.
        centers = F.normalize(self.hypotheses.queries[:, None] + self.appearance_modes, dim=-1) * (48 ** .5)
        centers = centers.flatten(0, 1)
        distance = (features.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None]
                    - 2 * features @ centers.T).clamp_min(0) / 48
        costs = -distance.reshape(-1, 19, 4) / .2
        energy = -(torch.logsumexp(costs, -1) - np.log(4.))
        if modes:
            selected = costs if mode_indices is None else costs[mode_indices]
            return energy, selected.softmax(-1)
        return energy

    def semantic(self, sample, *, modes=False):
        if modes and self.variant in ("standard", "cssr"):
            raise ValueError("appearance mode responsibilities require the multimodal appearance head")
        return self.feature_energy(self.features(sample), modes=modes,
                                   mode_indices=sample["queries"] if modes else None)

    def components(self, sample, indices=None, *, semantic_energy=None, features=None, require_reference=True):
        from .normal import geometry_energy
        if self.variant == "cssr" and features is None:
            features = self.features(sample)
            semantic_energy = self.feature_energy(features)
        semantic_energy = self.semantic(sample) if semantic_energy is None else semantic_energy
        if indices is None:
            indices = torch.arange(len(sample["xyzi"]), device=semantic_energy.device)
        semantic_energy = semantic_energy[indices]
        prediction = (None if self.variant in ("semantic", "standard", "cssr")
                      else self.hypotheses(sample["observation"], indices))
        geometry = torch.zeros_like(semantic_energy) if prediction is None else geometry_energy(prediction)
        energy = semantic_energy + geometry
        raw_score = energy.amin(-1)
        if self.variant == "standard":
            raw_score = -torch.logsumexp(-energy, -1)
        if self.variant == "cssr":
            if not bool(self.cssr_fitted) and require_reference:
                raise ValueError("CSSR normal activation statistics must be fitted before unknown inference")
            raw_score = (self.cssr_score(features[indices], energy / .1)
                         if bool(self.cssr_fitted) else None)
        return dict(logits=-energy, semantic_energy=semantic_energy, geometry_energy=geometry,
                    energy=energy, raw_score=raw_score, prediction=prediction)

    def loss(self, sample):
        from .normal import allowed_loss, hypothesis_loss, observation_diagnostics
        indices, allowed = sample["queries"], sample["allowed"]
        features = self.features(sample) if self.variant == "cssr" else None
        if self.variant in ("standard", "cssr"):
            semantic_energy = self.feature_energy(features) if features is not None else self.semantic(sample)
        elif self.training:
            semantic_energy = self.semantic(sample)
        else:
            semantic_energy, appearance = self.semantic(sample, modes=True)
        parts = self.components(sample, indices if self.training else None, semantic_energy=semantic_energy,
                                features=features, require_reference=False)
        if self.training:
            logits, prediction = parts["logits"], parts["prediction"]
        else:
            # Development measures the exact full-point classifier used at inference.
            self.development_prediction = parts["logits"].argmax(-1).detach()
            self.development_semantic_prediction = semantic_energy.argmin(-1).detach()
            self.development_raw_score = (parts["raw_score"].detach() if parts["raw_score"] is not None else None)
            logits = parts["logits"][indices]
            prediction = ({key: value[indices] for key, value in parts["prediction"].items()}
                          if parts["prediction"] is not None else None)
            self.development_diagnostics = ({} if self.variant in ("standard", "cssr") else observation_diagnostics(
                prediction, sample["observation"], indices, allowed[indices], appearance))
        # The control removes observation costs from supervised class competition.
        # Shared class parameters and predictive losses remain; inference uses E.
        if self.variant in ("semantic", "separate", "standard", "cssr"):
            logits = -semantic_energy[indices]
        classification = (allowed_loss(-semantic_energy, allowed) if self.variant == "standard"
                          else allowed_loss(logits, allowed[indices]))
        semantic = classification if self.variant == "standard" else allowed_loss(-semantic_energy, allowed)
        valid = allowed[indices].any(1)
        if self.variant not in ("standard", "cssr") and bool(valid.any()):
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

    def cssr_support(self, features, errors):
        """First- and second-order activation support use the predicted class."""
        values = features.double().abs()
        classes = errors.argmin(-1)
        relative = -errors.double().amin(-1) / (values.sum(-1).square() + 1e-8)
        first = (values * self.cssr_mean[classes]).sum(-1)
        # Evaluate b^T G_c b by class to avoid an N x D x D temporary.
        second = torch.zeros_like(first)
        for category in classes.unique():
            selected = classes == category
            group = values[selected]
            second[selected] = ((group @ self.cssr_gram[category]) * group).sum(-1)
        return torch.stack((relative, first, second), -1)

    def cssr_score(self, features, errors):
        return -((self.cssr_support(features, errors) - self.cssr_location)
                 / (self.cssr_scale + 1e-8)).sum(-1).to(features.dtype)

    @torch.no_grad()
    def fit_cssr_reference(self, unaugmented, augmented, device):
        """Fit the paper's two normal-training references with bounded memory."""
        if self.variant != "cssr":
            raise ValueError("activation references only apply to the CSSR-style control")
        self.eval()
        count = torch.zeros(19, dtype=torch.float64, device=device)
        sums = torch.zeros_like(self.cssr_mean)
        grams = torch.zeros_like(self.cssr_gram)
        frames = 0
        for sample in unaugmented:
            sample = to_device(sample, device)
            features = self.features(sample)[sample["allowed"].any(1)]
            classes = self.reconstruction_error(features).argmin(-1)
            values = features.double().abs()
            count += torch.bincount(classes, minlength=19)
            sums.index_add_(0, classes, values)
            for category in classes.unique():
                group = values[classes == category]
                grams[category] += group.T @ group
            frames += 1
        if not bool(count.sum()):
            raise ValueError("CSSR statistics require nonempty normal training points")
        mean = sums / count.clamp_min(1)[:, None]
        self.cssr_mean.copy_(mean / (mean.sum(0) + 1e-8))
        self.cssr_gram.copy_(grams / count.clamp_min(1)[:, None, None])
        # Parallel Welford updates avoid cancellation in the activation variances.
        total, augmented_frames = 0, 0
        location = torch.zeros_like(self.cssr_location)
        squared = torch.zeros_like(location)
        for sample in augmented:
            sample = to_device(sample, device)
            features = self.features(sample)[sample["allowed"].any(1)]
            support = self.cssr_support(features, self.reconstruction_error(features))
            size = len(support)
            if size:
                current = support.mean(0)
                delta = current - location
                squared += (support - current).square().sum(0) + delta.square() * total * size / (total + size)
                location += delta * size / (total + size)
                total += size
            augmented_frames += 1
        if not total or not bool(torch.isfinite(location).all() & torch.isfinite(squared).all()):
            raise ValueError("CSSR standardization requires finite augmented normal training points")
        self.cssr_location.copy_(location)
        self.cssr_scale.copy_((squared / total).clamp_min(0).sqrt())
        self.cssr_fitted.fill_(True)
        return dict(unaugmented_frames=frames, augmented_frames=augmented_frames,
                    unaugmented_points=int(count.sum()), augmented_points=total,
                    predicted_class_points=count.long().cpu().tolist(),
                    population="reliably labeled normal source and target training returns; all returns remain encoder inputs",
                    location=location.cpu().tolist(), scale=self.cssr_scale.cpu().tolist())

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

    def _readout(self, parts, readout):
        if readout not in NORMAL_READOUTS:
            raise ValueError(f"unknown inference readout: {readout}")
        energy = parts["energy"]
        raw_score = parts["raw_score"]
        if self.variant == "standard":
            if readout not in ("joint", "energy", "softmax"):
                raise ValueError("the standard classifier supports energy and softmax readouts")
            if readout == "softmax":
                raw_score = 1 - (-energy).softmax(-1).amax(-1)
        elif readout in ("energy", "softmax"):
            raise ValueError("energy and softmax readouts require the standard classifier")
        elif self.variant == "cssr" and readout != "joint":
            raise ValueError("the CSSR-style control uses its fitted activation score")
        elif readout == "appearance":
            energy = parts["semantic_energy"]
            raw_score = energy.amin(-1)
        elif readout == "common_density":
            # Averaging densities, rather than energies, preserves normalization.
            common = (-torch.logsumexp(-parts["geometry_energy"], -1) + np.log(19.)).clamp_min(0)
            energy = parts["semantic_energy"] + common[:, None]
            raw_score = parts["semantic_energy"].amin(-1) + common
        elif readout == "independent_minima":
            raw_score = parts["semantic_energy"].amin(-1) + parts["geometry_energy"].amin(-1)
        score = (self.calibrate_score(raw_score)
                 if readout == "joint" and bool(self.calibrated) else raw_score)
        label_energy = parts["semantic_energy"] if readout == "common_density" else energy
        return dict(semantic=label_energy.argmin(-1), semantic_only=parts["semantic_energy"].argmin(-1),
                    raw_score=raw_score, score=score,
                    confidence=(-label_energy).softmax(-1).amax(-1),
                    semantic_confidence=(-parts["semantic_energy"]).softmax(-1).amax(-1),
                    semantic_score=parts["semantic_energy"].amin(-1))

    def predict(self, sample, readout="joint"):
        with torch.autocast(sample["xyzi"].device.type, enabled=False):
            return self._readout(self.components(sample), readout)

    def predict_readouts(self, sample):
        with torch.autocast(sample["xyzi"].device.type, enabled=False):
            parts = self.components(sample)
            readouts = ("joint", "appearance", "common_density", "independent_minima")
            if self.variant == "standard":
                readouts = ("energy", "softmax")
            elif self.variant in ("semantic", "cssr"):
                readouts = ("joint",)
            return {name: self._readout(parts, name) for name in readouts}

    def forward(self, sample):
        return self.predict(sample)["score"]


def scatter_scores(scores, slots, slot_count):
    if scores.ndim != 1 or len(scores) != len(slots):
        raise ValueError("each real return must have exactly one prediction")
    output = scores.new_zeros(slot_count)
    return output.scatter(0, slots.long(), scores)
