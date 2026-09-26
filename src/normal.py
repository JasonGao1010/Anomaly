"""Normal observation models and distributions of frozen point features."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import segment_csr


SUPPORT_VERSION = "AJAE-frozen-support"
INSTANCE_VERSION = "AJAE-instance-support"
EVIDENCE_VERSION = "AJAE-feature-evidence"


class FeatureEvidence(nn.Module):
    """Joint normal semantics and anomaly evidence from one shared residual feature."""

    dimensions = 252
    classes = 19
    point_chunk = 16384

    def __init__(self, residual=True):
        super().__init__()
        self.register_buffer("location", torch.zeros(self.dimensions, dtype=torch.float32))
        self.register_buffer("whitener", torch.eye(self.dimensions, dtype=torch.float32))
        self.residual = nn.Sequential(
            nn.Linear(self.dimensions, 128, dtype=torch.float32), nn.GELU(),
            nn.Linear(128, self.dimensions, dtype=torch.float32)) if residual else None
        # Initial predictions preserve the independently fitted linear readouts.
        if self.residual is not None:
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        self.semantic = nn.Linear(self.dimensions, self.classes, dtype=torch.float32)
        self.anomaly = nn.Linear(self.dimensions, 1, dtype=torch.float32)

    def encode(self, features):
        if features.ndim != 2 or features.shape[1] != self.dimensions:
            raise ValueError("feature evidence requires shape [N, 252]")
        with torch.autocast(features.device.type, enabled=False):
            x = (features.float() - self.location) @ self.whitener
            return x if self.residual is None else x + self.residual(x)

    def components(self, features):
        if features.ndim != 2 or features.shape[1] != self.dimensions:
            raise ValueError("feature evidence requires shape [N, 252]")
        logits, scores = [], []
        with torch.autocast(features.device.type, enabled=False):
            for start in range(0, len(features), self.point_chunk):
                z = self.encode(features[start:start+self.point_chunk])
                logits.append(self.semantic(z))
                scores.append(self.anomaly(z).squeeze(-1))
            return dict(logits=torch.cat(logits) if logits else features.new_empty((0, self.classes), dtype=torch.float32),
                        score=torch.cat(scores) if scores else features.new_empty(0, dtype=torch.float32))

    def normal_logits(self, features):
        return self.components(features)["logits"]

    def forward(self, features, conditions=None):
        return self.components(features)["score"]


def support_conditions(xyzi, indices=None, *, range_only=False):
    """Measured range, with spacing computed only for models that consume it."""
    xyz = np.asarray(xyzi[:, :3], dtype=np.float64)
    selected = xyz if indices is None else xyz[np.asarray(indices)]
    if len(xyz) < (1 if range_only else 9) or not np.isfinite(xyz).all():
        raise ValueError("normal support has too few or nonfinite returns")
    distance = np.linalg.norm(selected, axis=1)
    if np.any(distance <= 0):
        raise ValueError("range must be positive at actual returns")
    if range_only:
        return np.column_stack((np.log(distance), np.zeros_like(distance))).astype(np.float32)
    from scipy.spatial import cKDTree
    spacing = cKDTree(xyz).query(selected, k=[9], workers=1)[0][:, 0]
    # A millimetre floor handles coincident returns without an infinite logarithm.
    return np.column_stack((np.log(distance), np.log(np.maximum(spacing, .001)))).astype(np.float32)


class FeatureSupport(nn.Module):
    """Normal class mixtures of joint multilevel features and observed spacing.

    Range can condition the mean and covariance scale. Spacing is scored, so unusual
    sparsity is not silently removed as a nuisance. All targets are frozen.
    """

    def __init__(self, modes=4, features=252, classes=19):
        super().__init__()
        self.modes, self.features, self.classes = modes, features, classes
        dimensions = features + 1
        self.register_buffer("location", torch.zeros(dimensions))
        self.register_buffer("scale", torch.ones(dimensions))
        self.register_buffer("range_location", torch.tensor(0.))
        self.register_buffer("range_scale", torch.tensor(1.))
        self.register_buffer("coefficients", torch.zeros(classes, 3, dimensions))
        self.register_buffer("log_variance_coefficients", torch.zeros(classes, 3))
        self.register_buffer("centers", torch.zeros(classes, modes, dimensions))
        self.register_buffer("precision_cholesky", torch.eye(dimensions).repeat(classes, 1, 1))
        self.register_buffer("log_volume", torch.zeros(classes))
        self.register_buffer("log_weights", torch.full((classes, modes), -torch.inf))
        self.register_buffer("present", torch.zeros(classes, dtype=torch.bool))

    def class_energy(self, features, conditions):
        if features.ndim != 2 or features.shape[1] != self.features or conditions.shape != (len(features), 2):
            raise ValueError("support features and measured conditions must identify the same points")
        if not bool(self.present.any()):
            raise ValueError("normal support has not been fitted")
        if not torch.isfinite(features).all() or not torch.isfinite(conditions).all():
            raise ValueError("nonfinite normal support input")
        with torch.autocast(features.device.type, enabled=False):
            values = torch.cat((features.float(), conditions[:, 1:2].float()), -1)
            values = (values - self.location) / self.scale
            distance = (conditions[:, 0].float() - self.range_location) / self.range_scale
            basis = torch.stack((torch.ones_like(distance), distance, distance.square()), -1)
            result = values.new_full((len(values), self.classes), torch.inf)
            for category in self.present.nonzero().flatten().tolist():
                precision = self.precision_cholesky[category]
                centers = self.centers[category] @ precision
                norm = centers.square().sum(-1)[None]
                for start in range(0, len(values), 16384):
                    stop = min(start + 16384, len(values))
                    residual = values[start:stop] - basis[start:stop] @ self.coefficients[category]
                    white = residual @ precision
                    square = (white.square().sum(-1, keepdim=True) + norm - 2 * white @ centers.T).clamp_min(0)
                    log_variance = 4 * torch.tanh((basis[start:stop] @ self.log_variance_coefficients[category]) / 4)
                    component = self.log_weights[category] - .5 * square * torch.exp(-log_variance[:, None])
                    # The feature standardization Jacobian is common to all classes
                    # and points; omitting that constant cannot alter score ranking.
                    result[start:stop, category] = (self.log_volume[category]
                        + .5 * values.shape[1] * (math.log(2 * math.pi) + log_variance)
                        - torch.logsumexp(component, -1))
            return result

    def forward(self, features, conditions):
        # Equal class priors avoid penalizing a valid but uncommon normal class.
        return self.class_energy(features, conditions).amin(-1)


class ScoreCalibration(nn.Module):
    """Normal score quantiles by frozen class, optionally continuous in log range."""

    def __init__(self, range_bandwidth=0.):
        super().__init__()
        if not math.isfinite(range_bandwidth) or range_bandwidth < 0:
            raise ValueError("calibration range bandwidth must be finite and nonnegative")
        self.register_buffer("probabilities", torch.tensor(
            [.01, .05, .1, .25, .5, .75, .9, .95, .99, .995, .999], dtype=torch.float64))
        self.register_buffer("knots", torch.zeros(17, 11))
        self.register_buffer("levels", torch.zeros(17, 11))
        self.register_buffer("lengths", torch.zeros(17, dtype=torch.long))
        self.register_buffer("counts", torch.zeros(17, dtype=torch.long))
        self.register_buffer("groups", torch.zeros(17, dtype=torch.long))
        self.register_buffer("minimum_points", torch.tensor(2048, dtype=torch.long))
        self.register_buffer("fitted", torch.tensor(False))
        self.register_buffer("enabled", torch.tensor(False))
        # Keep the default state identical to existing class-only checkpoints.
        if range_bandwidth > 0:
            self.register_buffer("range_bandwidth", torch.tensor(range_bandwidth, dtype=torch.float64))
            self.register_buffer("range_anchors", torch.linspace(
                math.log(2.5), math.log(50), 16, dtype=torch.float64))
            self.register_buffer("range_knots", torch.zeros(17, 16, 11))
            self.register_buffer("range_groups_enabled", torch.zeros(17, dtype=torch.bool))

    @staticmethod
    def _validate(scores, predicted, conditions):
        if (scores.ndim != 1 or predicted.shape != scores.shape
                or conditions.shape != (len(scores), 2)):
            raise ValueError("calibration values must identify the same points")
        if predicted.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("calibration requires integer frozen 16-class predictions")
        if bool(((predicted < 0) | (predicted >= 16)).any()):
            raise ValueError("calibration requires frozen 16-class predictions")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("normal calibration scores must be finite")

    @torch.no_grad()
    def fit(self, scores, predicted, conditions):
        scores = torch.as_tensor(scores).detach().to(device="cpu", dtype=torch.float64)
        predicted = torch.as_tensor(predicted).detach().cpu()
        conditions = torch.as_tensor(conditions).detach().cpu()
        self._validate(scores, predicted, conditions)
        if len(scores) < 2:
            raise ValueError("normal calibration requires at least two scores")
        probabilities = self.probabilities.cpu().numpy()
        quantile_levels = -np.log1p(-probabilities)
        knots = torch.zeros_like(self.knots, device="cpu")
        levels = torch.zeros_like(self.levels, device="cpu")
        counts = torch.zeros_like(self.counts, device="cpu")
        lengths = torch.zeros_like(self.lengths, device="cpu")
        groups = torch.zeros_like(self.groups, device="cpu")
        minimum = int(self.minimum_points)
        conditional = hasattr(self, "range_anchors")
        if conditional:
            if not bool(torch.isfinite(conditions[:, 0]).all()):
                raise ValueError("normal calibration log range must be finite")
            range_knots = torch.zeros_like(self.range_knots, device="cpu")
            range_enabled = torch.zeros_like(self.range_groups_enabled, device="cpu")
            range_fallback = [len(self.range_anchors)] * 17
            anchors = self.range_anchors.cpu().numpy()
            bandwidth = float(self.range_bandwidth)
        for group in range(17):
            selected = slice(None) if group == 0 else predicted == group - 1
            values = scores[selected]
            counts[group] = len(values)
            if group and len(values) < minimum:
                continue
            quantiles = np.quantile(values.numpy(), probabilities).astype(np.float32)
            unique, repetitions = np.unique(quantiles, return_counts=True)
            if len(unique) < 2:
                if group == 0:
                    raise ValueError("global normal calibration needs distinct score quantiles")
                continue
            # Merge atoms at their right quantile; never invent epsilon-width bins.
            size = len(unique)
            knots[group, :size] = torch.from_numpy(unique)
            knots[group, size:] = float(unique[-1])
            levels[group, :size] = torch.from_numpy(quantile_levels[repetitions.cumsum() - 1])
            lengths[group], groups[group] = size, group
            # Range adjustment needs class-specific references. Sparse classes
            # retain the original range-independent global fallback.
            if conditional and group > 0 and size == len(probabilities):
                # Atoms keep their original merged class CDF. Interpolate only
                # strictly increasing quantiles, without artificial epsilon bins.
                range_enabled[group] = True
                range_knots[group] = torch.from_numpy(quantiles)
                order = np.argsort(values.numpy(), kind="stable")
                ordered_scores = values.numpy()[order]
                distances = conditions[selected, 0].double().numpy()[order]
                for anchor_index, anchor in enumerate(anchors):
                    weights = np.exp(-.5 * ((distances - anchor) / bandwidth) ** 2)
                    weight_sum, square_sum = weights.sum(), np.dot(weights, weights)
                    if square_sum == 0 or weight_sum ** 2 / square_sum < minimum:
                        continue
                    indices = np.searchsorted(np.cumsum(weights), probabilities * weight_sum, side="left")
                    local = ordered_scores[indices.clip(max=len(order) - 1)].astype(np.float32)
                    if np.any(np.diff(local) <= 0):
                        continue
                    range_knots[group, anchor_index] = torch.from_numpy(local)
                    range_fallback[group] -= 1
        for name, value in (("knots", knots), ("levels", levels), ("counts", counts),
                            ("lengths", lengths), ("groups", groups)):
            getattr(self, name).copy_(value)
        self.fitted.fill_(True)
        self.enabled.fill_(False)
        report = dict(points=len(scores), minimum_points=minimum,
                    class_counts=counts[1:].tolist(),
                    fallback_classes=(groups[1:] == 0).nonzero().flatten().tolist(),
                    class_quantile_counts=lengths[1:].tolist(),
                    conditions="frozen official 16-class prediction only; no range or spacing bins")
        if conditional:
            self.range_knots.copy_(range_knots)
            self.range_groups_enabled.copy_(range_enabled)
            report.update(range_bandwidth=bandwidth, range_groups_enabled=range_enabled.tolist(),
                          range_fallback_anchors=range_fallback,
                          conditions="continuous log range for sufficiently populated frozen 16-class groups; "
                                     "sparse classes use the range-independent global CDF; no spacing")
        return report

    def forward(self, scores, predicted, conditions):
        if not bool(self.enabled):
            return scores
        if not bool(self.fitted):
            raise ValueError("normal score calibration has not been fitted")
        if predicted is None:
            raise ValueError("calibration requires frozen 16-class predictions")
        self._validate(scores, predicted, conditions)
        rows = self.groups[predicted.long() + 1]
        knots, levels = self.knots[rows], self.levels[rows]
        if hasattr(self, "range_anchors"):
            if not bool(torch.isfinite(conditions[:, 0]).all()):
                raise ValueError("normal calibration log range must be finite")
            distance = conditions[:, 0].to(knots)
            anchors = self.range_anchors.to(knots)
            right = torch.searchsorted(anchors, distance.contiguous(), right=True).clamp(1, len(anchors) - 1)
            fraction = ((distance - anchors[right - 1]) / (anchors[right] - anchors[right - 1])).clamp(0, 1)
            local = torch.lerp(self.range_knots[rows, right - 1], self.range_knots[rows, right], fraction[:, None])
            # Saved flags remain authoritative, including historical global fits.
            knots = torch.where(self.range_groups_enabled[rows, None], local, knots)
        upper = torch.searchsorted(knots, scores.contiguous()[:, None], right=True).flatten()
        upper = torch.minimum(upper.clamp_min(1), self.lengths[rows] - 1)
        index = torch.arange(len(scores), device=scores.device)
        low_x, high_x = knots[index, upper - 1], knots[index, upper]
        low_y, high_y = levels[index, upper - 1], levels[index, upper]
        # Continue the first/last positive slope beyond the reference score range.
        return low_y + (scores - low_x) * ((high_y - low_y) / (high_x - low_x))


@torch.no_grad()
def select_score_calibration(reference, holdout, bandwidths, device, current=None):
    """Select only on independent normal tails; stream queries to bound GPU memory."""
    import copy
    ref_score, ref_predicted, ref_conditions = reference
    scores, predicted, conditions = holdout
    ref_score = torch.as_tensor(ref_score).cpu()
    scores, predicted = torch.as_tensor(scores).cpu(), torch.as_tensor(predicted).cpu().long()
    conditions = torch.as_tensor(conditions).cpu()
    raw_threshold = float(np.quantile(ref_score.numpy(), .99))
    counts = torch.bincount(predicted, minlength=16)
    raw_false = torch.bincount(predicted[scores > raw_threshold], minlength=16)
    populated = counts >= 2048
    if not bool(populated.any()):
        raise ValueError("normal calibration selection has no sufficiently populated holdout group")
    raw_rates = raw_false.double() / counts.clamp_min(1)
    raw_error = float((raw_rates[populated] - .01).abs().mean())
    candidates = []
    if current is not None and bool(current.enabled):
        candidates.append(("previous", copy.deepcopy(current).to(device), None))
    for bandwidth in bandwidths:
        candidate = ScoreCalibration(range_bandwidth=bandwidth).to(device)
        fit = candidate.fit(ref_score, ref_predicted, ref_conditions)
        candidate.enabled.fill_(True)
        candidates.append(("full_normal" if current is not None else "normal", candidate, fit))
    reports, best, best_error, best_source, best_bandwidth = [], None, math.inf, None, None
    for source, candidate, fit in candidates:
        false = torch.zeros(16, dtype=torch.long, device=device)
        for start in range(0, len(scores), 65536):
            stop = start + 65536
            group = predicted[start:stop].to(device)
            values = candidate(scores[start:stop].to(device), group, conditions[start:stop].to(device))
            false += torch.bincount(group[values > math.log(100)], minlength=16)
        rates = false.cpu().double() / counts.clamp_min(1)
        error = float((rates[populated] - .01).abs().mean())
        bandwidth = float(candidate.range_bandwidth) if hasattr(candidate, "range_bandwidth") else 0.
        rows = [dict(category=c, points=int(counts[c]), raw_fpr01=float(raw_rates[c]),
                     calibrated_fpr01=float(rates[c])) for c in range(16) if counts[c]]
        reports.append(dict(source=source, range_bandwidth=bandwidth, classes=rows,
                            fit=fit, mean_class_tail_error=error))
        if error < best_error:
            best, best_error, best_source, best_bandwidth = candidate, error, source, bandwidth
        print(f"normal calibration {source} bandwidth={bandwidth:g}: tail error={error:.6f}", flush=True)
    best.enabled.fill_(best_error < raw_error)
    return best, dict(enabled=bool(best.enabled), range_bandwidth=best_bandwidth,
        selected_source=best_source if bool(best.enabled) else "raw", candidates=reports,
        raw_mean_class_tail_error=raw_error, calibrated_mean_class_tail_error=best_error,
        normal_blocks_only=True,
        selection="mean absolute deviation from 1% normal FPR across frozen prediction groups with at least 2048 holdout points; whole 64-frame blocks remain disjoint")


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


class InstanceSupport(nn.Module):
    """One observed normal instance jointly supports all frozen feature levels."""

    dimensions = 252
    classes = 19
    point_chunk = 4096

    def __init__(self, memory_size=16384, temporal_window=16):
        super().__init__()
        if memory_size < 1:
            raise ValueError("normal instance memory must be nonempty")
        if temporal_window < 1:
            raise ValueError("normal temporal exclusion window must be positive")
        self.temporal_window = temporal_window
        self.register_buffer("location", torch.zeros(self.dimensions))
        self.register_buffer("whitener", torch.eye(self.dimensions))
        self.register_buffer("memory", torch.zeros(memory_size, self.dimensions))
        self.register_buffer("memory_allowed", torch.zeros(memory_size, self.classes, dtype=torch.bool))
        self.register_buffer("memory_source", torch.full((memory_size,), -1, dtype=torch.long))
        self.register_buffer("memory_frame", torch.full((memory_size,), -1, dtype=torch.long))
        self.register_buffer("temperature", torch.tensor(1.))
        self.transform = nn.Parameter(torch.eye(self.dimensions))
        self.calibration = ScoreCalibration()
        self.register_buffer("_encoded_memory", torch.empty(0, self.dimensions), persistent=False)
        self.register_buffer("_memory_norm", torch.empty(0), persistent=False)
        self._cache_key = None
        self._label_key = None
        self._class_indices = None

    def _invalidate_cache(self):
        self._cache_key = None

    def train(self, mode=True):
        if mode:
            self._invalidate_cache()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate_cache()
        self._label_key = None
        return super()._load_from_state_dict(*args, **kwargs)

    def encode(self, features):
        if features.ndim != 2 or features.shape[1] != self.dimensions:
            raise ValueError("normal instance features must have shape [N, 252]")
        with torch.autocast(features.device.type, enabled=False):
            return (features.float() - self.location) @ (self.whitener @ self.transform)

    @torch.no_grad()
    def project_metric(self):
        # Keep every whitened direction: squared distances remain between
        # 0.25 and 4 times their untrained values, with no learned support radius.
        with torch.autocast(self.transform.device.type, enabled=False):
            left, singular, right = torch.linalg.svd(self.transform.float(), full_matrices=False)
            self.transform.copy_((left * singular.clamp(.5, 2)[None]) @ right)
        self._invalidate_cache()
        return self

    def _state_key(self):
        return (self.memory.device, self.memory.dtype,
                self.location._version, self.whitener._version,
                self.transform._version, self.memory._version)

    @torch.no_grad()
    def freeze_metric(self):
        self._encoded_memory = self.encode(self.memory).detach()
        self._memory_norm = self._encoded_memory.square().sum(-1)
        if not bool(torch.isfinite(self._memory_norm).all()):
            raise ValueError("normal memory has nonfinite transformed features")
        self._cache_key = self._state_key()
        return self

    def _bank(self):
        if self.training or torch.is_grad_enabled():
            # Query and reference transformations both differentiate through T.
            values = self.encode(self.memory)
            return values, values.square().sum(-1)
        if self._cache_key != self._state_key():
            self.freeze_metric()
        return self._encoded_memory, self._memory_norm

    def _members(self):
        key = (self.memory_allowed.device, self.memory_allowed._version)
        if self._label_key != key:
            counts = self.memory_allowed.sum(-1)
            self._class_indices = [((counts == 1) & self.memory_allowed[:, category]).nonzero().flatten()
                                   for category in range(self.classes)]
            self._label_key = key
        return self._class_indices

    @staticmethod
    def _validate_identity(features, conditions, source, frame):
        if features.ndim != 2 or features.shape[1] != 252:
            raise ValueError("normal instance features must have shape [N, 252]")
        if conditions is not None and conditions.shape != (len(features), 2):
            raise ValueError("normal support conditions must identify the same points")
        if (source is None) != (frame is None):
            raise ValueError("source and frame identities must be supplied together")
        if source is not None:
            if source.shape != (len(features),) or frame.shape != (len(features),):
                raise ValueError("source and frame identities must identify the same points")
            if source.dtype not in (torch.int32, torch.int64) or frame.dtype not in (torch.int32, torch.int64):
                raise ValueError("source and frame identities must be integer tensors")

    def _distance(self, encoded, bank, norm, source=None, frame=None):
        distance = (encoded.square().sum(-1, keepdim=True) + norm[None]
                    - 2 * encoded @ bank.T).clamp_min(0)
        if source is not None:
            # Source frames carry nuScenes scene IDs; target frames carry the
            # original 206 indices. Exclude adjacent target frames across blocks.
            nearby = torch.where(source[:, None] == 1,
                (frame[:, None] - self.memory_frame[None]).abs() < self.temporal_window,
                frame[:, None] == self.memory_frame[None])
            excluded = (source[:, None] == self.memory_source[None]) & nearby
            distance = distance.masked_fill(excluded, torch.inf)
        return distance

    def class_energy(self, features, conditions=None, source=None, frame=None):
        """Fine-class distances; coarse labels never create false fine anchors."""
        self._validate_identity(features, conditions, source, frame)
        with torch.autocast(features.device.type, enabled=False):
            bank, norm = self._bank()
            members = self._members()
            chunks = []
            for start in range(0, len(features), self.point_chunk):
                stop = start + self.point_chunk
                distance = self._distance(self.encode(features[start:stop]), bank, norm,
                    None if source is None else source[start:stop],
                    None if frame is None else frame[start:stop])
                chunks.append(torch.stack([distance[:, index].amin(-1) if len(index)
                    else distance.new_full((len(distance),), torch.inf) for index in members], -1))
            return torch.cat(chunks) if chunks else features.new_empty((0, self.classes), dtype=torch.float32)

    def raw_score(self, features, conditions=None, source=None, frame=None):
        self._validate_identity(features, conditions, source, frame)
        with torch.autocast(features.device.type, enabled=False):
            valid = self.memory_allowed.any(-1)
            if not bool(valid.any()):
                raise ValueError("normal instance memory has no trusted normal points")
            bank, norm = self._bank()
            chunks = []
            for start in range(0, len(features), self.point_chunk):
                stop = start + self.point_chunk
                distance = self._distance(self.encode(features[start:stop]), bank, norm,
                    None if source is None else source[start:stop],
                    None if frame is None else frame[start:stop])
                # All trusted normal instances support rejection, including
                # coarse labels. One identical reference supplies all 252 values.
                chunks.append(distance.masked_fill(~valid[None], torch.inf).amin(-1))
            return torch.cat(chunks) if chunks else features.new_empty(0, dtype=torch.float32)

    def forward(self, features, conditions, predicted=None):
        return self.calibration(self.raw_score(features, conditions), predicted, conditions)


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
