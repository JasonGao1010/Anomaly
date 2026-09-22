"""Target-blind angular context and joint normal-return distributions."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import segment_csr


NORMAL_MODES = ("density", "geometry")
AZIMUTH_BINS, ELEVATION_BINS = 180, 90
HYPOTHESES = 4


def angular_observation(xyzi, *, origins=None, directions=None):
    """Build angular blocks before any voxelization or spatial-neighbor search.

    Default rays share the reference-frame origin. These are effective scan rays,
    not recovered firing poses of a moving sensor. Directions carry no range.
    """
    xyz = np.asarray(xyzi[:, :3], dtype=np.float64)
    origins = np.zeros_like(xyz) if origins is None else np.asarray(origins, dtype=np.float64)
    distance = np.linalg.norm(xyz - origins, axis=1)
    if np.any(distance <= 0) or not np.isfinite(distance).all():
        raise ValueError("normal prediction requires finite positive return ranges")
    directions = ((xyz - origins) / distance[:, None] if directions is None
                  else np.asarray(directions, dtype=np.float64))
    if origins.shape != xyz.shape or directions.shape != xyz.shape:
        raise ValueError("ray metadata must identify every actual return")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-6):
        raise ValueError("ray directions must have unit length")
    azimuth = np.arctan2(directions[:, 1], directions[:, 0])
    elevation = np.arcsin(np.clip(directions[:, 2], -1, 1))
    az = np.floor((azimuth + np.pi) * AZIMUTH_BINS / (2 * np.pi)).astype(np.int64) % AZIMUTH_BINS
    el = np.floor((elevation + np.pi / 2) * ELEVATION_BINS / np.pi).astype(np.int64)
    el = el.clip(0, ELEVATION_BINS - 1)
    cells, group, counts = np.unique(el * AZIMUTH_BINS + az, return_inverse=True, return_counts=True)
    order = np.argsort(group, kind="stable")
    ca = ((cells % AZIMUTH_BINS + .5) / AZIMUTH_BINS * 2 - 1) * np.pi
    ce = ((cells // AZIMUTH_BINS + .5) / ELEVATION_BINS - .5) * np.pi
    center = np.column_stack((np.cos(ce) * np.cos(ca), np.cos(ce) * np.sin(ca), np.sin(ce)))
    tangent = np.column_stack((-np.sin(ca), np.cos(ca), np.zeros_like(ca)))
    vertical = np.cross(center, tangent)
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            row = cells // AZIMUTH_BINS + dy
            neighbor = row * AZIMUTH_BINS + (cells % AZIMUTH_BINS + dx) % AZIMUTH_BINS
            neighbors.append(np.where((row >= 0) & (row < ELEVATION_BINS), neighbor, -1))
    relative = np.column_stack(((azimuth - ca[group] + np.pi) % (2 * np.pi) - np.pi,
                                elevation - ce[group]))
    # Pointwise encoding and block-local pooling precede exclusion. No BatchNorm,
    # global feature, target-cell count, target voxel or spatial kNN is allowed.
    features = np.column_stack((xyz / 50, xyzi[:, 3], directions, origins / 50, np.log(distance)))
    values = dict(features=features, origins=origins, directions=directions, log_range=np.log(distance),
                  relative=relative, center=center, tangent=tangent, vertical=vertical,
                  cells=cells, group=group, order=order,
                  pointer=np.r_[0, np.cumsum(counts)], neighbors=np.stack(neighbors, 1))
    return {key: torch.from_numpy(value.astype(np.int64 if key in
            ("cells", "group", "order", "pointer", "neighbors") else np.float32))
            for key, value in values.items()}


def network(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, outputs))


class NormalDistribution(nn.Module):
    """Four region-level hypotheses; ordinary ray regression or shared planes."""

    def __init__(self, mode):
        super().__init__()
        if mode not in NORMAL_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.encoder = network(11, 32, 32)
        self.context = network(8 * 34 + 3, 96, 96)
        self.weights = nn.Linear(96, HYPOTHESES)
        self.planes = nn.Linear(96, HYPOTHESES * 4) if mode == "geometry" else None
        self.regression = network(96 + 8, 64, HYPOTHESES * 2) if mode == "density" else None
        output = self.planes if self.planes is not None else self.regression[-1]
        nn.init.zeros_(output.weight)
        with torch.no_grad():
            output.bias.zero_()
            output.bias.reshape(HYPOTHESES, -1)[:, 0] = torch.linspace(-.15, .15, HYPOTHESES)
            output.bias.reshape(HYPOTHESES, -1)[:, -1] = -2.

    def forward(self, observation):
        # Keep probability calculations and geometry in FP32 even under N1 AMP.
        with torch.autocast(observation["features"].device.type, enabled=False):
            return self.predict(observation)

    def predict(self, o):
        features = self.encoder(o["features"].float())
        pooled = segment_csr(torch.cat((features, o["log_range"][:, None]), 1)[o["order"]],
                             o["pointer"], reduce="mean")
        population = (o["pointer"][1:] - o["pointer"][:-1]).float()
        pooled = torch.cat((pooled, population.log1p()[:, None]), 1)
        table = pooled.new_zeros((AZIMUTH_BINS * ELEVATION_BINS + 1, pooled.shape[1]))
        table = table.index_copy(0, o["cells"], pooled)
        neighbors = o["neighbors"]
        surrounding = table[torch.where(neighbors < 0, table.shape[0] - 1, neighbors)]
        present = surrounding[..., -1] > 0
        count = present.sum(1).clamp_min(1)
        base = (surrounding[..., -2] * present).sum(1) / count
        base = torch.where(present.any(1), base, torch.full_like(base, math.log(15.)))
        context = self.context(torch.cat((surrounding.flatten(1), o["center"]), 1))
        log_weights = self.weights(context).log_softmax(-1)
        group = o["group"]
        if self.planes is not None:
            raw = self.planes(context).reshape(-1, HYPOTHESES, 4)
            radius = (base[:, None] + raw[..., 0]).clamp(-2., 7.).exp()
            slopes = 20 * raw[..., 1:3].tanh()
            normals = (o["center"][:, None] - slopes[..., :1] * o["tangent"][:, None]
                       - slopes[..., 1:] * o["vertical"][:, None])
            mu, valid = ray_planes(radius[group], normals[group], o["origins"], o["directions"])
            scale = (.015 + F.softplus(raw[..., 3])).clamp_max(2.)[group]
        else:
            query = torch.cat((o["relative"], o["directions"], o["origins"] / 50), 1)
            raw = self.regression(torch.cat((context[group], query), 1)).reshape(-1, HYPOTHESES, 2)
            mu = (base[group, None] + raw[..., 0]).clamp(-2., 7.)
            scale = (.015 + F.softplus(raw[..., 1])).clamp_max(2.)
            valid = torch.ones_like(mu, dtype=torch.bool)
        return dict(mu=mu, scale=scale, valid=valid, log_weights=log_weights, group=group)


def ray_planes(offset, normal, origin, direction):
    """One plane n.x=b jointly fixes all of its ray intersections."""
    numerator = offset - (normal * origin[:, None]).sum(-1)
    denominator = (normal * direction[:, None]).sum(-1)
    valid = (numerator > 0) & (denominator > 0)
    mu = numerator.clamp_min(1e-8).log() - denominator.clamp_min(1e-8).log()
    return mu, valid


def component_log_prob(prediction, log_range):
    # Laplace observation noise gives a proper density and robust linear tails.
    residual = (log_range[:, None] - prediction["mu"]) / prediction["scale"]
    value = -residual.abs() - (2 * prediction["scale"]).log()
    # Invalid forward intersections put their probability on the no-hit outcome.
    return value.masked_fill(~prediction["valid"], -1e4)


def joint_nll(prediction, log_range, selected):
    log_prob = component_log_prob(prediction, log_range)
    group, weights = prediction["group"], prediction["log_weights"]
    total = log_prob.new_zeros(weights.shape).index_add(0, group, log_prob * selected[:, None])
    counts = log_prob.new_zeros(len(weights)).index_add(0, group, selected.float())
    used = counts > 0
    # Sum ray log-likelihoods BEFORE marginalizing the shared hypothesis.
    nll = -torch.logsumexp(weights + total, -1)
    return nll[used].sum() / counts.sum().clamp_min(1)


def point_evidence(prediction, log_range):
    log_prob = component_log_prob(prediction, log_range)
    group, prior = prediction["group"], prediction["log_weights"]
    total = log_prob.new_zeros(prior.shape).index_add(0, group, log_prob)
    # Scoring may compare measured returns; prediction parameters never see them.
    # Leave this point out when testing which shared hypothesis explains its peers.
    posterior = (prior[group] + total[group] - log_prob).log_softmax(-1)
    marginal = -torch.logsumexp(prior[group] + log_prob, -1)
    conditional = -torch.logsumexp(posterior + log_prob, -1)
    probabilities = posterior.exp()
    mean = (probabilities * prediction["mu"]).sum(-1)
    variance = (probabilities * (2 * prediction["scale"].square()
                + (prediction["mu"] - mean[:, None]).square())).sum(-1)
    residual = (log_range - mean) / variance.sqrt().clamp_min(.015)
    entropy = -(probabilities * posterior).sum(-1)
    # No regional joint NLL is broadcast to individual points.
    return torch.stack((marginal, conditional, residual, residual.abs(),
                        .5 * variance.clamp_min(1e-8).log(), entropy), -1).clamp(-30, 30)


def quantiles(prediction, probabilities=(.05, .5, .95)):
    """Marginal predictive intervals, computed before observing any target range."""
    mu, scale = prediction["mu"], prediction["scale"]
    weights = prediction["log_weights"][prediction["group"]].exp() * prediction["valid"]
    if bool((weights.sum(-1) < max(probabilities)).any()):
        raise ValueError("normal prediction assigns too much probability to no forward intersection")
    low, high = (mu - 30 * scale).amin(-1), (mu + 30 * scale).amax(-1)
    outputs = []
    for probability in probabilities:
        left, right = low.clone(), high.clone()
        for _ in range(28):
            midpoint = (left + right) / 2
            delta = (midpoint[:, None] - mu) / scale
            cdf = torch.where(delta < 0, .5 * delta.clamp_max(0).exp(),
                              1 - .5 * (-delta).clamp_max(0).exp())
            below = (cdf * weights).sum(-1) < probability
            left, right = torch.where(below, midpoint, left), torch.where(below, right, midpoint)
        outputs.append(((left + right) / 2).exp())
    return torch.stack(outputs, -1)


@torch.no_grad()
def prediction_metrics(model, sample):
    observation = sample["observation"]
    prediction = model(observation)
    selected = sample["targets"] == 0
    # Evaluate the normal-only held-out labels, including every selected ray.
    subset = {k: v if k == "log_weights" else v[selected] for k, v in prediction.items()}
    intervals = quantiles(subset)
    actual = observation["log_range"][selected].exp()
    error = intervals[:, 1] - actual
    covered = (actual >= intervals[:, 0]) & (actual <= intervals[:, 2])
    return dict(points=len(actual), mae_m=float(error.abs().mean()),
                mse_m2=float(error.square().mean()), coverage90=float(covered.float().mean()),
                width90_m=float((intervals[:, 2] - intervals[:, 0]).mean()),
                joint_nll=float(joint_nll(prediction, observation["log_range"], selected)))
