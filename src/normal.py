"""Target-blind Gaussian return fields and analytic conditional ray observations."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import segment_csr


NORMAL_MODES = ("field",)
SCALES = (1, 2, 4)
HYPOTHESES, KERNELS = 4, 8
LOWER, UPPER = 2.5, 50.
RAY_CHUNK, BLOCK_CHUNK = 2048, 256
HYPOTHESIS_CHUNK = 1024
MIN_RETURN_SCALE = .001
LOG_RETURN_PEAK = math.log(2 / (math.pi * math.sqrt(3) * MIN_RETURN_SCALE))


def angular_observation(xyzi, *, origins=None, directions=None):
    """Rebuild effective rays from the actual (possibly transformed) coordinates.

    Without firing metadata, rays share the scan reference origin. Block identity
    uses directions only; target ranges, counts and intensity never select context.
    """
    xyz = np.asarray(xyzi[:, :3], dtype=np.float64)
    origin = np.zeros_like(xyz) if origins is None else np.asarray(origins, dtype=np.float64)
    if origin.shape != xyz.shape:
        raise ValueError("ray origins must identify every actual return")
    distance = np.linalg.norm(xyz - origin, axis=1)
    if not len(xyz) or np.any(distance <= 0) or not np.isfinite(xyzi).all() or not np.isfinite(distance).all():
        raise ValueError("normal prediction requires finite real returns and positive ranges")
    rays = (xyz - origin) / distance[:, None]
    if directions is not None:
        supplied = np.asarray(directions, dtype=np.float64)
        if supplied.shape != rays.shape or not np.allclose(supplied, rays, atol=1e-6, rtol=0):
            raise ValueError("ray metadata disagrees with current coordinates; rebuild after augmentation")
        rays = supplied
    azimuth = (np.rad2deg(np.arctan2(rays[:, 1], rays[:, 0])) + 180) % 360
    elevation = np.rad2deg(np.arcsin(np.clip(rays[:, 2], -1, 1))) + 90
    # Derive every scale from one integer grid so nesting is exact at boundaries.
    az = np.floor(azimuth).astype(np.int64)
    el = np.floor(elevation).astype(np.int64).clip(0, 179)
    grids = {}
    for size in SCALES:
        width, height = 360 // size, 180 // size
        cells, group, counts = np.unique((el // size) * width + az // size,
                                          return_inverse=True, return_counts=True)
        row, col = cells // width, cells % width
        ca, ce = np.deg2rad((col + .5) * size - 180), np.deg2rad((row + .5) * size - 90)
        center = np.column_stack((np.cos(ce) * np.cos(ca), np.cos(ce) * np.sin(ca), np.sin(ce)))
        tangent = np.column_stack((-np.sin(ca), np.cos(ca), np.zeros_like(ca)))
        # Grid positions are fixed, even if all target measurements change.
        position = np.column_stack((np.cos(ca), np.sin(ca), ce / (np.pi / 2), np.full(len(cells), size / 4)))
        table = np.full(width * height, len(cells), dtype=np.int64)
        table[cells] = np.arange(len(cells))
        neighbors = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == dy == 0:
                    continue
                valid = (row + dy >= 0) & (row + dy < height)
                lookup = np.clip(row + dy, 0, height - 1) * width + (col + dx) % width
                neighbors.append(np.where(valid, table[lookup], len(cells)))
        grids[str(size)] = dict(cells=cells, group=group, order=np.argsort(group, kind="stable"),
            pointer=np.r_[0, np.cumsum(counts)], neighbors=np.stack(neighbors, 1), position=position,
            basis=np.stack((center, tangent, np.cross(center, tangent)), -1))
    values = dict(features=np.column_stack((xyz / UPPER, xyzi[:, 3], rays)),
                  origins=origin, directions=rays, distance=distance, grids=grids)

    def tensors(value):
        if isinstance(value, dict):
            return {key: tensors(item) for key, item in value.items()}
        return torch.from_numpy(value.astype(np.int64 if value.dtype.kind in "iu" else np.float32))
    return tensors(values)


def network(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, outputs))


def hypothesis_observation(xyzi, cell_degrees=.5):
    """Independent angular cells; withheld cell values never construct its state."""
    xyz = np.asarray(xyzi[:, :3], dtype=np.float64)
    distance = np.linalg.norm(xyz, axis=1)
    if not len(xyz) or np.any(distance <= 0) or not np.isfinite(xyzi).all():
        raise ValueError("normal hypotheses require finite actual returns")
    if cell_degrees <= 0 or not np.isclose(180 / cell_degrees, round(180 / cell_degrees)):
        raise ValueError("angular cell size must divide 180 degrees")
    rays = xyz / distance[:, None]
    azimuth = (np.rad2deg(np.arctan2(rays[:, 1], rays[:, 0])) + 180) % 360
    elevation = np.rad2deg(np.arcsin(np.clip(rays[:, 2], -1, 1))) + 90
    width, height = round(360 / cell_degrees), round(180 / cell_degrees)
    col = np.floor(azimuth / cell_degrees).astype(np.int64) % width
    row = np.floor(elevation / cell_degrees).astype(np.int64).clip(0, height - 1)
    cells, group, counts = np.unique(row * width + col, return_inverse=True, return_counts=True)
    row, col = cells // width, cells % width
    ca, ce = np.deg2rad((col + .5) * cell_degrees - 180), np.deg2rad((row + .5) * cell_degrees - 90)
    table = np.full(width * height, len(cells), dtype=np.int64)
    table[cells] = np.arange(len(cells))
    neighbors = []
    # Two radii expose local shape and its surroundings without another backbone.
    for radius in (1, 3):
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            valid = (row + dy * radius >= 0) & (row + dy * radius < height)
            ids = np.clip(row + dy * radius, 0, height - 1) * width + (col + dx * radius) % width
            neighbors.append(np.where(valid, table[ids], len(cells)))
    da = (np.deg2rad(azimuth - 180) - ca[group] + np.pi) % (2 * np.pi) - np.pi
    de = np.deg2rad(elevation - 90) - ce[group]
    da, de = da / np.deg2rad(cell_degrees / 2), de / np.deg2rad(cell_degrees / 2)
    values = dict(features=np.column_stack((xyz / 50, xyzi[:, 3], rays)),
                  log_distance=np.log(distance), group=group, order=np.argsort(group, kind="stable"),
                  pointer=np.r_[0, np.cumsum(counts)], neighbors=np.stack(neighbors, 1),
                  position=np.column_stack((np.cos(ca), np.sin(ca), ce / (np.pi / 2))),
                  offset=np.column_stack((da, de, da * da, da * de, de * de)), cells=cells)
    return {key: torch.from_numpy(value.astype(np.int64 if value.dtype.kind in "iu" else np.float32))
            for key, value in values.items()}


class SemanticHypotheses(nn.Module):
    """Nineteen normal explanations, each predicting a held-out return surface."""

    def __init__(self):
        super().__init__()
        self.encoder = network(7, 32, 32)
        self.pool = nn.Linear(64, 48)
        self.position = nn.Linear(3, 48)
        self.queries = nn.Parameter(torch.randn(19, 48) * .1)
        self.empty = nn.Parameter(torch.zeros(48))
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            attention=nn.MultiheadAttention(48, 3, batch_first=True),
            norm=nn.LayerNorm(48), feedforward=network(48, 96, 48), final_norm=nn.LayerNorm(48),
        )) for _ in range(2))
        self.surface = nn.Linear(48, 3 * 8)
        self.belief = nn.Linear(48, 1)
        nn.init.normal_(self.surface.weight, std=.001)
        nn.init.zeros_(self.surface.bias)
        with torch.no_grad():
            self.surface.bias.reshape(3, 8)[:, 6] = -2.

    def encode(self, observation):
        features = self.encoder(observation["features"])
        ordered = features[observation["order"]]
        pointer = observation["pointer"]
        pooled = torch.cat((segment_csr(ordered, pointer, reduce="mean"),
                            segment_csr(ordered, pointer, reduce="max")), -1)
        tokens = self.pool(pooled) + self.position(observation["position"])
        depth = segment_csr(observation["log_distance"][observation["order"]], pointer, reduce="mean")
        return tokens, depth

    def project_context(self, tokens):
        # Project each independent cell once, rather than once per neighboring query.
        context = torch.cat((tokens, self.empty[None]))
        return [F.linear(context, layer["attention"].in_proj_weight[48:],
                         layer["attention"].in_proj_bias[48:]).chunk(2, -1)
                for layer in self.layers]

    def propose(self, observation, encoded, groups, projected=None):
        tokens, depth = encoded
        neighbors = observation["neighbors"][groups]
        present = neighbors < len(tokens)
        # An explicit empty token keeps completely isolated cells numerically valid.
        neighbors = F.pad(neighbors, (0, 1), value=len(tokens))
        mask = torch.cat((present, ~present.any(1, keepdim=True)), 1)
        projected = self.project_context(tokens) if projected is None else projected
        # The very same prototypes classify observed points and generate normal geometry.
        prototypes = F.normalize(self.queries, dim=-1) * math.sqrt(48)
        state = prototypes[None] + self.position(observation["position"][groups])[:, None]
        for layer, (keys, values) in zip(self.layers, projected):
            attention = layer["attention"]
            query = F.linear(state, attention.in_proj_weight[:48], attention.in_proj_bias[:48])
            query = query.reshape(len(groups), 19, 3, 16).transpose(1, 2)
            key = keys[neighbors].reshape(len(groups), -1, 3, 16).transpose(1, 2)
            value = values[neighbors].reshape(len(groups), -1, 3, 16).transpose(1, 2)
            update = F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None, None])
            update = attention.out_proj(update.transpose(1, 2).reshape(len(groups), 19, 48))
            state = layer["norm"](state + update)
            state = layer["final_norm"](state + layer["feedforward"](state))
        base = (F.pad(depth, (0, 1))[neighbors[:, :-1]] * present).sum(1) / present.sum(1).clamp_min(1)
        base = torch.where(present.any(1), base, torch.full_like(base, math.log(20)))
        raw = self.surface(state).reshape(-1, 19, 3, 8)
        # Unbounded, normalized quadratic coefficients can represent the steep
        # angular range changes of distant ground without a hand-set slope limit.
        return dict(mean=base[:, None, None] + raw[..., 0],
                    slope=raw[..., 1:6], scale=MIN_RETURN_SCALE + F.softplus(raw[..., 6]),
                    weight=raw[..., 7].log_softmax(-1), belief=self.belief(state).squeeze(-1),
                    supported=present.any(1))

    def forward(self, observation, indices):
        encoded = self.encode(observation)
        if not len(indices):
            zero = encoded[0].sum() * 0
            return dict(log_prob=zero.expand(0, 19, 3), log_compatibility=zero.expand(0, 19, 3),
                        weight=zero.expand(0, 19, 3), belief=zero.expand(0, 19),
                        supported=observation["group"].new_empty(0, dtype=torch.bool),
                        group=observation["group"][indices], scale=zero.expand(0, 19, 3),
                        mean=zero.expand(0, 19, 3))
        projected = self.project_context(encoded[0])
        groups, inverse = observation["group"][indices].unique(sorted=True, return_inverse=True)
        parts = [self.propose(observation, encoded, part, projected) for part in groups.split(HYPOTHESIS_CHUNK)]
        fields = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
        selected = {key: value[inverse] for key, value in fields.items()}
        mean = selected["mean"] + (selected["slope"] * observation["offset"][indices, None, None]).sum(-1)
        standardized = (observation["log_distance"][indices, None, None] - mean) / selected["scale"]
        log_compatibility = -2 * torch.log1p(standardized.square() / 3)
        # This proper density is defined on log distance, with respect to d(log r).
        log_prob = -math.log(math.pi * math.sqrt(3) / 2) - selected["scale"].log() + log_compatibility
        return dict(log_prob=log_prob, weight=selected["weight"], belief=selected["belief"],
                    log_compatibility=log_compatibility,
                    supported=selected["supported"], group=observation["group"][indices],
                    scale=selected["scale"], mean=mean)


def geometry_energy(prediction):
    """Normalized density support; retain the scale penalty in every mixture mode.

    The common bound is the peak of a minimum-scale Student-t. Its additive
    log constant cannot change class competition and keeps both supports <= 1.
    """
    energy = LOG_RETURN_PEAK - torch.logsumexp(prediction["weight"] + prediction["log_prob"], -1)
    return torch.where(prediction["supported"][:, None], energy.clamp_min(0), 0.)


def allowed_loss(logits, allowed):
    """Class-set balanced supervision without inventing unobserved fine labels."""
    valid = allowed.any(1)
    if not bool(valid.any()):
        return logits.sum() * 0
    admitted = allowed[valid]
    nll = -torch.logsumexp(logits[valid].log_softmax(-1).masked_fill(~admitted, -torch.inf), -1)
    bits = (admitted.long() * (2 ** torch.arange(admitted.shape[1], device=logits.device))).sum(1)
    _, group, counts = bits.unique(return_inverse=True, return_counts=True)
    return (nll / counts[group]).sum() / len(counts)


def hypothesis_loss(prediction, allowed):
    """Proper pointwise conditional density, including ambiguous normal label sets."""
    valid = allowed.any(1) & prediction["supported"]
    if not bool(valid.any()):
        zero = prediction["log_prob"].sum() * 0 + prediction["belief"].sum() * 0
        return zero, zero
    admitted = allowed[valid]
    # Normalize the latent class prior inside the observed label set. Two returns
    # carrying the same coarse label may still belong to different fine classes.
    prior = prediction["belief"][valid].masked_fill(~admitted, -torch.inf).log_softmax(-1)
    probability = prediction["log_prob"][valid] + prediction["weight"][valid] + prior[..., None]
    nll = -torch.logsumexp(probability.flatten(1), -1)
    return nll.mean(), allowed_loss(prediction["belief"][valid], admitted)


@torch.no_grad()
def observation_diagnostics(prediction, observation, indices, allowed, appearance):
    """Additive held-out diagnostics; coarse labels never become fine-class truth."""
    device = allowed.device
    count = allowed.sum(-1)
    valid = (count == 1)
    classes = allowed.long().argmax(-1)
    result = dict(query_count=torch.bincount(classes[valid], minlength=19),
                  appearance_mode_mass=torch.zeros(19, 4, device=device, dtype=torch.float64),
                  appearance_mode_entropy_sum=torch.zeros(19, device=device, dtype=torch.float64))
    responsibilities = appearance[valid, classes[valid]].double()
    result["appearance_mode_mass"].index_add_(0, classes[valid], responsibilities)
    result["appearance_mode_entropy_sum"].index_add_(0, classes[valid],
        -(responsibilities * responsibilities.clamp_min(1e-300).log()).sum(-1))
    for key in ("prediction_count", "nll_sum", "coverage90_count", "width90_m_sum",
                "abs_median_error_m_sum", "finite_interval_count", "geometry_mode_entropy_sum",
                "mean_pair_separation_sum"):
        result[key] = torch.zeros(19, device=device, dtype=torch.float64)
    result["geometry_mode_mass"] = torch.zeros(19, 3, device=device, dtype=torch.float64)
    result["geometry_prior_mass"] = torch.zeros(19, 3, device=device, dtype=torch.float64)
    result["geometry_scale_sum"] = torch.zeros(19, 3, device=device, dtype=torch.float64)
    result.update(coarse_query_count=0, coarse_nll_sum=0., unsupported_query_count=0)
    if prediction is None:
        return {key: value.cpu().numpy() if torch.is_tensor(value) else value for key, value in result.items()}
    supported = prediction["supported"]
    result["unsupported_query_count"] = int(((count > 0) & ~supported).sum())
    coarse = (count > 1) & supported
    if bool(coarse.any()):
        prior = prediction["belief"][coarse].double().masked_fill(~allowed[coarse], -torch.inf).log_softmax(-1)
        lp = prediction["weight"][coarse].double() + prediction["log_prob"][coarse].double() + prior[..., None]
        result["coarse_query_count"] = int(coarse.sum())
        result["coarse_nll_sum"] = float(-torch.logsumexp(lp.flatten(1), -1).sum())
    valid &= supported
    c = classes[valid]
    if len(c):
        mean = prediction["mean"][valid, c].double()
        scale = prediction["scale"][valid, c].double()
        weights = prediction["weight"][valid, c].double().softmax(-1)
        posterior = (prediction["weight"][valid, c].double() + prediction["log_prob"][valid, c].double()).softmax(-1)
        value = observation["log_distance"][indices][valid].double()

        def cdf(z):
            u = (z[..., None] - mean[:, None]) / (scale[:, None] * math.sqrt(3))
            return ((.5 + (u.atan() + u / (1 + u.square())) / math.pi) * weights[:, None]).sum(-1)

        # Exact t3 component quantiles bracket every corresponding mixture quantile.
        probabilities = value.new_tensor([.05, .5, .95])
        component = mean[:, None] + scale[:, None] * value.new_tensor([-2.3533634348018233, 0., 2.3533634348018233])[None, :, None]
        left, right = component.amin(-1), component.amax(-1)
        for _ in range(40):
            middle = (left + right) * .5
            below = cdf(middle) < probabilities
            left, right = torch.where(below, middle, left), torch.where(below, right, middle)
        quantiles = ((left + right) * .5).exp()
        finite = torch.isfinite(quantiles).all(-1)
        pit = cdf(value[:, None]).squeeze(-1)
        quantities = dict(prediction_count=torch.ones_like(value),
            nll_sum=-torch.logsumexp(prediction["weight"][valid, c].double() + prediction["log_prob"][valid, c].double(), -1),
            coverage90_count=((pit >= .05) & (pit <= .95)).double(),
            geometry_mode_entropy_sum=-(posterior * posterior.clamp_min(1e-300).log()).sum(-1),
            mean_pair_separation_sum=(mean[:, 0] - mean[:, 1]).abs() + (mean[:, 0] - mean[:, 2]).abs() + (mean[:, 1] - mean[:, 2]).abs(),
            finite_interval_count=finite.double())
        for key, values in quantities.items():
            result[key].index_add_(0, c, values)
        result["geometry_mode_mass"].index_add_(0, c, posterior)
        result["geometry_prior_mass"].index_add_(0, c, weights)
        result["geometry_scale_sum"].index_add_(0, c, scale)
        result["width90_m_sum"].index_add_(0, c[finite], quantiles[finite, 2] - quantiles[finite, 0])
        result["abs_median_error_m_sum"].index_add_(0, c[finite], (quantiles[finite, 1] - value[finite].exp()).abs())
    return {key: value.cpu().numpy() if torch.is_tensor(value) else value for key, value in result.items()}


class NormalField(nn.Module):
    """One four-hypothesis field per block, inferred exclusively from 24 neighbors."""

    def __init__(self, *, recompute=True):
        super().__init__()
        self.recompute = recompute
        self.encoder = network(7, 64, 64)
        self.pool = nn.Linear(128, 128)
        self.position = nn.Linear(4, 128)
        self.empty = nn.Parameter(torch.randn(128) * .02)
        self.queries = nn.Parameter(torch.randn(HYPOTHESES, 128) * .02)
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            attention=nn.MultiheadAttention(128, 4, batch_first=True),
            norm=nn.LayerNorm(128), feedforward=network(128, 256, 128), final_norm=nn.LayerNorm(128),
        )) for _ in range(2))
        self.weights = nn.Linear(128, 1)
        self.parameters_out = nn.Linear(128, KERNELS * 10)
        nn.init.normal_(self.parameters_out.weight, std=.001)
        with torch.no_grad():
            bias = self.parameters_out.bias.reshape(KERNELS, 10)
            bias.zero_()
            bias[:, 0] = torch.linspace(LOWER / UPPER, 1., KERNELS)
            # Broad initialization is an optimization choice, not a noise model.
            bias[:, [3, 5, 8]] = math.log(math.expm1(20 / UPPER))
            bias[:, 9] = -2.

    def _run(self, function, *args):
        if self.training and self.recompute and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def project_context(self, tokens):
        """Project each independent block once before repeated neighbor gathering."""
        context = torch.cat((tokens, tokens.new_zeros((1, 128)), self.empty[None]))
        return tuple(F.linear(context, layer["attention"].in_proj_weight[128:],
                              layer["attention"].in_proj_bias[128:]) for layer in self.layers)

    def decode(self, projected, neighbors, position, basis):
        count = len(projected[0]) - 2
        present = neighbors < count
        empty = ~present.any(1)
        # Independent projections mix no blocks; self is still excluded before attention.
        indices = torch.cat((neighbors, neighbors.new_full((len(neighbors), 1), count + 1)), 1)
        mask = torch.cat((~present, ~empty[:, None]), 1)
        state = self.queries[None] + self.position(position)[:, None]
        for layer, context in zip(self.layers, projected):
            attention = layer["attention"]
            query = F.linear(state, attention.in_proj_weight[:128], attention.in_proj_bias[:128])
            key, value = context[indices].chunk(2, -1)
            query, key, value = (item.reshape(len(neighbors), -1, 4, 32).transpose(1, 2)
                                 for item in (query, key, value))
            update = F.scaled_dot_product_attention(query, key, value,
                attn_mask=~mask[:, None, None], dropout_p=attention.dropout if self.training else 0.)
            update = update.transpose(1, 2).reshape(len(neighbors), HYPOTHESES, 128)
            update = attention.out_proj(update)
            state = layer["norm"](state + update)
            state = layer["final_norm"](state + layer["feedforward"](state))
        raw = self.parameters_out(state).reshape(-1, HYPOTHESES, KERNELS, 10)
        center = UPPER * torch.einsum("bij,bkmj->bkmi", basis, raw[..., :3])
        a, b, c, d, e, f = raw[..., 3:9].unbind(-1)
        zero = torch.zeros_like(a)
        # A A^T is a full covariance. The floor only prevents singular solves.
        diagonal = lambda x: UPPER * F.softplus(x) + 1e-4
        chol = torch.stack((diagonal(a), zero, zero, UPPER * b, diagonal(c), zero,
                            UPPER * d, UPPER * e, diagonal(f)), -1).reshape(*a.shape, 3, 3)
        inverse = torch.linalg.solve_triangular(chol, torch.eye(3, device=chol.device).expand_as(chol), upper=False)
        # log(softplus(x)) must remain finite for very negative amplitudes.
        log_amplitude = torch.where(raw[..., 9] < -20, raw[..., 9],
                                    F.softplus(raw[..., 9].clamp_min(-20)).log()) - math.log(UPPER)
        weights = self.weights(state).squeeze(-1).log_softmax(-1)
        return center, inverse, log_amplitude, weights

    def forward(self, observation):
        with torch.autocast(observation["features"].device.type, enabled=False):
            features = torch.cat([self._run(self.encoder, part) for part in observation["features"].float().split(65536)])
            fields = {}
            for size in SCALES:
                grid = observation["grids"][str(size)]
                ordered = features[grid["order"]]
                pooled = torch.cat((segment_csr(ordered, grid["pointer"], reduce="mean"),
                                    segment_csr(ordered, grid["pointer"], reduce="max")), -1)
                tokens = self.pool(pooled) + self.position(grid["position"])
                projected = self.project_context(tokens)
                parts = [self._run(self.decode, projected, grid["neighbors"][start:start + BLOCK_CHUNK],
                                   grid["position"][start:start + BLOCK_CHUNK], grid["basis"][start:start + BLOCK_CHUNK])
                         for start in range(0, len(tokens), BLOCK_CHUNK)]
                fields[str(size)] = dict(zip(("center", "inverse", "log_amplitude", "log_weights"),
                                              (torch.cat(items) for items in zip(*parts))))
            return fields

    def likelihood(self, observation, targets):
        """Normal auxiliary loss on a clean companion, without another backbone pass."""
        distance = observation["distance"]
        selected = (targets == 0) & (distance >= LOWER) & (distance <= UPPER)
        indices = selected.nonzero().flatten()
        if not len(indices):
            # Preserve zero gradients, including optimizer decay semantics.
            return sum(parameter.sum() * 0 for parameter in self.parameters())
        fields = self(observation)
        losses = []
        for size in SCALES:
            grid, field = observation["grids"][str(size)], fields[str(size)]
            parts = []
            for chosen in indices.split(RAY_CHUNK):
                def probability(group, origin, direction, measured, parameters=field):
                    return ray_log_prob(*ray_parameters(parameters, group, origin, direction), measured)
                parts.append(self._run(probability, grid["group"][chosen],
                    observation["origins"][chosen], observation["directions"][chosen], distance[chosen]))
            # Keep the full block membership and its original averaging rule.
            probability = distance.new_zeros((len(distance), HYPOTHESES)).index_copy(0, indices, torch.cat(parts))
            losses.append(joint_nll(probability, field["log_weights"], grid, selected))
        return torch.stack(losses).mean()


def ray_parameters(field, group, origin, direction):
    """Restrict each 3D Gaussian to a ray; covariance inverses are cached by block."""
    inverse = field["inverse"][group]
    v = torch.einsum("nkmij,nj->nkmi", inverse, direction)
    w = torch.einsum("nkmij,nkmj->nkmi", inverse, origin[:, None, None] - field["center"][group])
    vv = v.square().sum(-1)
    mu = -(v * w).sum(-1) / vv
    tau = vv.rsqrt()
    log_h = field["log_amplitude"][group] - .5 * (w + mu[..., None] * v).square().sum(-1)
    return mu, tau, log_h


class LogNormalCDF(torch.autograd.Function):
    """Keep log-CDF values, but avoid cancellation in their negative-tail derivative."""

    @staticmethod
    def forward(ctx, value):
        result = torch.special.log_ndtr(value)
        ctx.save_for_backward(value, result)
        return result

    @staticmethod
    def backward(ctx, gradient):
        value, result = ctx.saved_tensors
        # phi(x)/Phi(x) = sqrt(2/pi)/erfcx(-x/sqrt(2)) for x <= 0.
        # Safe inputs matter: where() still evaluates both derivative branches.
        tail = math.sqrt(2 / math.pi) / torch.special.erfcx(-value.clamp_max(0) / math.sqrt(2))
        central = (-.5 * value.clamp_min(0).square() - result.clamp_min(-math.log(2))
                   - .5 * math.log(2 * math.pi)).exp()
        return gradient * torch.where(value < 0, tail, central)


def log_normal_mass(mu, tau, lower, upper):
    """Stable log Gaussian interval mass, including narrow intervals in either tail."""
    width = (upper - lower) / tau
    middle = ((upper - mu) + (lower - mu)) / (2 * tau)
    left, right = middle - width / 2, middle + width / 2
    # Reflect the positive tail before subtracting CDFs near one.
    positive = middle > 0
    lo = LogNormalCDF.apply(torch.where(positive, -right, left))
    hi = LogNormalCDF.apply(torch.where(positive, -left, right))
    delta = lo - hi
    safe_delta = torch.where(delta < 0, delta, torch.full_like(delta, -1.))
    ordinary = hi + torch.log(-torch.expm1(safe_delta))
    narrow = (width < .01) & (middle.abs() * width < .01)
    # The midpoint integral expansion avoids catastrophic CDF cancellation.
    local_width = torch.where(narrow, width, torch.ones_like(width))
    local_middle = torch.where(narrow, middle, torch.zeros_like(middle))
    # Reorder the same polynomial: m*w stays small even in a very distant tail.
    w2, u2 = local_width.square(), (local_middle * local_width).square()
    correction = (u2 - w2) / 24 + (u2.square() - 6 * u2 * w2 + 3 * w2.square()) / 1920
    local = local_width.clamp_min(torch.finfo(width.dtype).tiny).log() - .5 * local_middle.square() - .5 * math.log(2 * math.pi)
    local = local + torch.log1p(correction)
    return torch.where(width > 0, torch.where(narrow, local, ordinary), torch.full_like(local, -torch.inf))


def log_event_ratio(log_H):
    """log((1-exp(-H))/H), retaining the normalized limit as H tends to zero."""
    small = log_H < math.log(.01)
    H_small = log_H.clamp_max(math.log(.01)).exp()
    approximation = -H_small / 2 + H_small.square() / 24
    safe_log_H = log_H.clamp_min(math.log(.01))
    # Above this bound the log correction already rounds to zero; keep -log_H.
    maximum = math.log(-math.log(torch.finfo(log_H.dtype).tiny))
    regular = torch.log(-torch.expm1(-safe_log_H.clamp_max(maximum).exp())) - safe_log_H
    return torch.where(small, approximation, regular)


def ray_log_prob(mu, tau, log_h, distance, lower=LOWER, upper=UPPER):
    """Conditional first-event log density in metres, with its low-rate limit."""
    r = distance[:, None, None]
    shift = log_h.amax(-1, keepdim=True)
    relative = log_h - shift
    log_hazard = torch.logsumexp(relative - .5 * ((r - mu) / tau).square(), -1)
    integral = relative + tau.log() + .5 * math.log(2 * math.pi)
    log_total = torch.logsumexp(integral + log_normal_mass(mu, tau, lower, upper), -1)
    log_H = shift.squeeze(-1) + log_total
    # Avoid evaluating log(0) in an inactive autograd branch at the lower bound.
    b = torch.where(r > lower, r, torch.full_like(r, (lower + upper) / 2))
    log_partial = torch.logsumexp(integral + log_normal_mass(mu, tau, lower, b), -1)
    H_r = torch.where(distance[:, None] > lower, (shift.squeeze(-1) + log_partial).exp(), 0.)
    value = log_hazard - log_total - H_r - log_event_ratio(log_H)
    valid = (distance >= lower) & (distance <= upper)
    return value.masked_fill(~valid[:, None], -torch.inf)


@torch.no_grad()
def ray_quantiles(mu, tau, log_h, log_weights, probabilities=(.05, .5, .95)):
    """Analytic-CDF inversion of the context-prior mixture; no measured ranges."""
    shift = log_h.amax(-1, keepdim=True)
    integral = log_h - shift + tau.log() + .5 * math.log(2 * math.pi)
    total = torch.logsumexp(integral + log_normal_mass(mu, tau, LOWER, UPPER), -1)
    correction = log_event_ratio(shift.squeeze(-1) + total)
    left = mu.new_full((len(mu), len(probabilities)), LOWER)
    right = torch.full_like(left, UPPER)
    probability = torch.as_tensor(probabilities, device=mu.device)
    for _ in range(26):
        middle = (left + right) / 2
        partial = torch.logsumexp(integral[:, None] + log_normal_mass(
            mu[:, None], tau[:, None], LOWER, middle[:, :, None, None]), -1)
        log_cdf = partial - total[:, None] + log_event_ratio(shift.squeeze(-1)[:, None] + partial) - correction[:, None]
        cdf = (log_weights[:, None] + log_cdf).logsumexp(-1).exp()
        below = cdf < probability
        left, right = torch.where(below, middle, left), torch.where(below, right, middle)
    return (left + right) / 2


def joint_nll(log_prob, log_weights, grid, selected):
    """Marginalize ONE hypothesis after summing all valid rays, then average blocks."""
    masked = torch.where(selected[:, None], log_prob, 0.)
    total = segment_csr(masked[grid["order"]], grid["pointer"], reduce="sum")
    counts = segment_csr(selected.float()[grid["order"]], grid["pointer"], reduce="sum")
    nll = -torch.logsumexp(log_weights + total, -1) / counts.clamp_min(1)
    used = counts > 0
    return torch.where(used, nll, 0.).sum() / used.sum().clamp_min(1)


@torch.no_grad()
def prediction_metrics(model, sample):
    """Diagnostic coverage/width on unmodified normal scans, never a training gate."""
    if not sample.get("normal_training", False):
        raise ValueError("normal diagnostics require an unmodified real-normal scan")
    observation = sample["observation"]
    selected = ((sample["targets"] == 0) & (observation["distance"] >= LOWER)
                & (observation["distance"] <= UPPER))
    indices = selected.nonzero().flatten()
    if not len(indices):
        raise ValueError("normal diagnostic scan has no valid normal targets")
    fields, result = model(observation), {}
    with torch.autocast(indices.device.type, enabled=False):
        for size in SCALES:
            grid, field = observation["grids"][str(size)], fields[str(size)]
            log_prob = observation["distance"].new_zeros((len(selected), HYPOTHESES))
            totals = log_prob.new_zeros(3)
            for chosen in indices.split(RAY_CHUNK):
                group = grid["group"][chosen]
                params = ray_parameters(field, group, observation["origins"][chosen], observation["directions"][chosen])
                actual = observation["distance"][chosen]
                log_prob[chosen] = ray_log_prob(*params, actual)
                intervals = ray_quantiles(*params, field["log_weights"][group])
                totals += torch.stack(((intervals[:, 1] - actual).abs().sum(),
                    ((actual >= intervals[:, 0]) & (actual <= intervals[:, 2])).sum(),
                    (intervals[:, 2] - intervals[:, 0]).sum()))
            result[str(size)] = dict(zip(("mae_m", "coverage90", "width90_m"), (totals / len(indices)).tolist()))
            result[str(size)]["joint_nll"] = float(joint_nll(log_prob, field["log_weights"], grid, selected))
    return dict(points=len(indices), scales=result)


class Compatibility(nn.Module):
    """Compare complete context hypotheses before learning a pointwise decision."""

    def __init__(self):
        super().__init__()
        self.hypotheses = network(2, 16, 64)
        self.scale = nn.Parameter(torch.randn(len(SCALES), 64) * .02)
        self.query = nn.Linear(64, 32)
        self.range = network(1, 16, 32)

    def forward(self, state, observation, fields, indices):
        distance = observation["distance"][indices]
        query = (self.query(state.float()) + self.range(distance[:, None] / UPPER)).reshape(-1, 4, 8)
        features, marginals, probabilities = [], [], []
        valid = (distance >= LOWER) & (distance <= UPPER)
        for index, size in enumerate(SCALES):
            group = observation["grids"][str(size)]["group"][indices]
            field = fields[str(size)]
            mu, tau, log_h = ray_parameters(field, group, observation["origins"][indices],
                                            observation["directions"][indices])
            weights = field["log_weights"][group]
            # Outside protocol support, keep point logits but provide no density
            # evidence. Such returns remain context and are never NLL targets.
            prob = ray_log_prob(mu, tau, log_h, distance.clamp(LOWER, UPPER))
            density = torch.where(valid[:, None], prob, 0.)
            # The complete ray density already integrates every coexisting kernel.
            hypotheses = self.hypotheses(torch.stack((density, weights), -1)) + self.scale[index]
            key, value = (part.reshape(-1, HYPOTHESES, 4, 8) for part in hypotheses.chunk(2, -1))
            # This is learned evidence aggregation, not a posterior update of the
            # normal field. Shared maps make both kernel and hypothesis order arbitrary.
            attention = ((query[:, None] * key).sum(-1) / math.sqrt(8)).softmax(1)
            features.append((attention[..., None] * value).sum(1).flatten(1))
            marginals.append(torch.where(valid, torch.logsumexp(weights + prob, -1), 0.))
            probabilities.append(prob)
        return torch.cat((state.float(), *features, torch.stack(marginals, -1)), -1), torch.stack(probabilities, 1)
