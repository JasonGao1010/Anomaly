"""Class-conditioned return densities and normal semantic diagnostics for SERVE."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import segment_csr


HYPOTHESIS_CHUNK = 1024
MIN_RETURN_SCALE = .001
LOG_RETURN_PEAK = math.log(2 / (math.pi * math.sqrt(3) * MIN_RETURN_SCALE))


def normal_semantic_metrics(confusion):
    """Truth indexes rows; every false positive contributes to its predicted class union."""
    matrix = torch.as_tensor(confusion, dtype=torch.float64, device="cpu")
    if matrix.shape != (19, 19) or bool((matrix < 0).any()) or not bool(matrix.sum()):
        raise ValueError("normal semantics require a nonempty 19-class confusion matrix")
    support, predicted, correct = matrix.sum(1), matrix.sum(0), matrix.diag()
    union = support + predicted - correct
    return dict(points=int(matrix.sum()), confusion=matrix.long().tolist(),
        iou=[float(correct[c]/union[c]) if union[c] else None for c in range(19)],
        mean_iou_gt=float((correct[support>0]/union[support>0]).mean()),
        mean_iou_present=float((correct[union>0]/union[union>0]).mean()),
        point_accuracy=float(correct.sum()/matrix.sum()),
        definition="pooled point confusion; mean_iou_gt averages ground-truth-present classes, including classes without training references")


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
