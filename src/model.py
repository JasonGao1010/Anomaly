"""V4: full-return LitePT-S segmentation and one five-scale interaction."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import segment_csr

from .data import file_sha256
from vendor.litept.model import LitePT, Point


LITEPT_COMMIT = "436d04801c8151faebe66a1b2d368a9711e7e6aa"
WEIGHTS_REVISION = "a8e76e92efbb2061639f5c683968bc5d248ee002"
WEIGHTS_SHA256 = "95f151f6edcfbf315cd06df6afd261f2a2fde300d3c693dd26b1305d642ecc30"
CHANNELS = (36, 72, 144, 252, 504)
GRID_SIZE = .05
POINT_CHUNK = 65536


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden, eps=1e-5),
                         nn.GELU(), nn.Linear(hidden, outputs))


def voxelize(xyzi):
    """All-return means on an origin-anchored grid; preserve every point identity."""
    xyzi = np.asarray(xyzi)
    if xyzi.dtype != np.float32 or xyzi.ndim != 2 or xyzi.shape[1] != 4:
        raise ValueError("input must be float32[N,4]")
    if not len(xyzi) or not np.isfinite(xyzi).all() or np.any(~np.any(xyzi[:, :3] != 0, axis=1)):
        raise ValueError("voxelization requires finite real returns, with no empty ray slots")
    grid = np.floor(xyzi[:, :3].astype(np.float64) / GRID_SIZE).astype(np.int64)
    unique, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    pointer = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    mean = (np.add.reduceat(xyzi[order].astype(np.float64), pointer[:-1], axis=0)
            / counts[:, None]).astype(np.float32)
    # A multiple of 16 preserves the sensor-origin grid at all four pooling steps.
    shift = (unique.min(axis=0) // 16) * 16
    return dict(xyzi=torch.from_numpy(xyzi), grid=torch.from_numpy(unique - shift),
                voxel_xyzi=torch.from_numpy(mean), inverse=torch.from_numpy(inverse),
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
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()}


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


class Segmentor(nn.Module):
    def __init__(self, mode="base", *, recompute=True):
        super().__init__()
        self.mode, self.recompute = mode, recompute
        self.backbone = LitePT(shuffle_orders=False)
        self.detail = mlp(7, 64, 64)
        self.adapter = mlp(128, 64, 36)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        if mode == "conditional":
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
        if mode not in ("base", "conditional"):
            self.add_interaction(mode)

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

    def forward(self, sample):
        xyzi, inverse = sample["xyzi"], sample["inverse"]
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
        if self.mode == "conditional":
            unified = self._checkpoint(self.conditional, pooled, sample["voxel_xyzi"][:, :3],
                voxel_ancestors + [voxel_ancestors[0]], coords + [point.coord], features + [point.feat])
            output = []
            for start in range(0, len(xyzi), POINT_CHUNK):
                end = min(start + POINT_CHUNK, len(xyzi))
                def score_points(e, context, offset):
                    state = context + self.point_detail(e) + self.point_position(offset)
                    return self.head(state).squeeze(-1).float()
                output.append(self._checkpoint(score_points, detail[start:end], unified[inverse[start:end]],
                                               sample["offset"][start:end]))
            return torch.cat(output)
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


def balanced_loss(logits, targets, counts):
    """Global effective-batch class means, not the mean of microbatch means."""
    if counts.dtype != torch.int64 or counts.shape != (2,):
        raise ValueError("class counts must be int64[normal, anomaly]")
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
    """One two-scan proxy, with all positives and weighted sampled negatives."""
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
