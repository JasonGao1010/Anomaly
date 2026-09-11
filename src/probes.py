"""Normal-only empirical geometry probes; fitting uses train/206 only."""

from __future__ import annotations

import numpy as np


GEOMETRY = ("surface_residual", "roughness", "normal_change", "variation", "linearity", "planarity")
NORMALIZED = ("residual_scaled", "roughness_scaled", *GEOMETRY[2:])
DIRECT = ("range", "ray_z")
SAMPLING = (*DIRECT, "scale")
FEATURES = (*SAMPLING, "neighbor_count", *GEOMETRY, *NORMALIZED[:2])
MIN_POINTS, MIN_FRAMES = 256, 20
RANGE_EDGES = np.array([2.5, 5, 10, 20, 35, 50.000001], dtype=np.float64)
DIRECTION_EDGES = np.array([-1.000001, -.3, -.15, -.075, -.025, .025, .1, .3, 1.000001])


def _arrays(data):
    arrays = {name: np.asarray(data[name]) for name in (*FEATURES, "frame", "source_slot")}
    size = len(arrays["frame"])
    if any(values.ndim != 1 or len(values) != size for values in arrays.values()):
        raise ValueError("Feature arrays must be one-dimensional and share point identities")
    for name in ("frame", "source_slot"):
        if not np.isfinite(arrays[name]).all() or np.any(arrays[name] != np.floor(arrays[name])):
            raise ValueError(f"{name} must contain finite integer identities")
    return arrays


def _bin(values, edges):
    result = np.full(len(values), -1, dtype=np.int16)
    if not len(edges):
        return result
    valid = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    if len(edges) == 1:
        result[valid] = 0
    else:
        # The last observed scale belongs to the last bin; tails are unsupported.
        result[valid] = np.minimum(np.searchsorted(edges, values[valid], side="right") - 1, len(edges) - 2)
    return result


def _cells(data, edges):
    radius, direction, scale = (_bin(data[name], edges[name]) for name in DIRECT + ("scale",))
    n_direction, n_scale = len(edges["ray_z"]) - 1, max(1, len(edges["scale"]) - 1)
    angular = np.where((radius >= 0) & (direction >= 0), radius * n_direction + direction, -1)
    sampling = np.where((angular >= 0) & (scale >= 0), angular * n_scale + scale, -1)
    return {"range": radius, "direction": angular, "sampling": sampling}


def _bounds(values):
    values = values[np.isfinite(values)]
    return [float(values.min()), float(values.max())] if len(values) else None


def _fit_field(values, frames):
    valid = np.isfinite(values)
    count, frame_count = int(valid.sum()), len(np.unique(frames[valid]))
    trusted = count >= MIN_POINTS and frame_count >= MIN_FRAMES
    metadata = dict(points=count, frames=frame_count, bounds=_bounds(values), trusted=trusted)
    return np.sort(values[valid]).copy() if trusted else np.empty(0, values.dtype), metadata


def fit_reference(data):
    """Fit normal train/206 points; the caller must enforce that source boundary.

    Point occurrences weight empirical distributions equally. Distinct frames only
    set a minimum support requirement; they are not independent sample guarantees.
    """
    arrays = _arrays(data)
    scales = arrays["scale"][np.isfinite(arrays["scale"])]
    edges = {"range": RANGE_EDGES.copy(), "ray_z": DIRECTION_EDGES.copy(),
             "scale": np.unique(np.quantile(scales, np.linspace(0, 1, 6))) if len(scales) else np.array([])}
    metadata = dict(source="train/206 normal", rows=len(arrays["frame"]),
                    frames=len(np.unique(arrays["frame"])), frame_bounds=_bounds(arrays["frame"]),
                    source_slot_bounds=_bounds(arrays["source_slot"]),
                    weighting="equal point occurrences; minimum distinct-frame support",
                    min_points=MIN_POINTS, min_frames=MIN_FRAMES,
                    edges={name: values.tolist() for name, values in edges.items()},
                    global_fields={}, cells={})
    reference = dict(edges=edges, global_fields={}, conditional={}, metadata=metadata)
    for name in FEATURES:
        reference["global_fields"][name], metadata["global_fields"][name] = _fit_field(arrays[name], arrays["frame"])
    for mode, cells in _cells(arrays, edges).items():
        reference["conditional"][mode], metadata["cells"][mode] = {}, {}
        # Group once, avoiding one full-length boolean scan per cell and feature.
        order = np.argsort(cells, kind="stable")
        groups, starts, counts = np.unique(cells[order], return_index=True, return_counts=True)
        for cell, start, count in zip(groups, starts, counts):
            if cell < 0:
                continue
            indices = order[start:start + count]
            fields, stats = {}, {}
            for name in GEOMETRY:
                fields[name], stats[name] = _fit_field(arrays[name][indices], arrays["frame"][indices])
            reference["conditional"][mode][int(cell)] = fields
            metadata["cells"][mode][str(cell)] = dict(
                points=int(count), frames=len(np.unique(arrays["frame"][indices])),
                fields=stats, trusted_all=all(item["trusted"] for item in stats.values()))
    return reference


def tail_score(values, ordered):
    """Inclusive two-sided empirical rarity; a descriptive score, not a p-value."""
    values = np.asarray(values)
    result = np.full(values.shape, np.nan, dtype=np.float32)
    if not len(ordered):
        return result
    valid = np.isfinite(values)
    # Including ties in both tails makes constant normal values non-anomalous.
    lower = np.searchsorted(ordered, values[valid], side="right")
    upper = len(ordered) - np.searchsorted(ordered, values[valid], side="left")
    result[valid] = 1 - np.minimum(1, 2 * np.minimum(lower, upper) / len(ordered))
    return result


def _maximum(fields, names):
    # NaN propagation enforces the same required geometry dimensions in B and C.
    result = fields[names[0]].copy()
    for name in names[1:]:
        np.maximum(result, fields[name], out=result)
    return result


def score_reference(data, reference):
    """Return composite scores, per-feature scores, and conditional cell identities.

    Unsupported condition cells remain missing. Geometry beyond normal reference
    extrema is scored by its empirical rarity, not mistaken for missing support.
    """
    arrays = _arrays(data)
    marginal = {name: tail_score(arrays[name], reference["global_fields"][name])
                for name in FEATURES if name != "neighbor_count"}
    scores = dict(A_range=marginal["range"], A_direct=_maximum(marginal, DIRECT),
                  A_sampling=_maximum(marginal, SAMPLING), B_geometry=_maximum(marginal, GEOMETRY),
                  B_normalized=_maximum(marginal, NORMALIZED))
    individual = {"B_geometry": {name: marginal[name] for name in GEOMETRY},
                  "B_normalized": {name: marginal[name] for name in NORMALIZED}}
    cells = _cells(arrays, reference["edges"])
    for mode, identities in cells.items():
        fields = {name: np.full(len(identities), np.nan, np.float32) for name in GEOMETRY}
        order = np.argsort(identities, kind="stable")
        groups, starts, counts = np.unique(identities[order], return_index=True, return_counts=True)
        for cell, start, count in zip(groups, starts, counts):
            normal = reference["conditional"][mode].get(int(cell))
            if normal is None:
                continue
            indices = order[start:start + count]
            for name in GEOMETRY:
                fields[name][indices] = tail_score(arrays[name][indices], normal[name])
        key = "C_" + mode
        individual[key], scores[key] = fields, _maximum(fields, GEOMETRY)
    return scores, individual, cells


def _matched_field(values, targets, weights, frames, denominators):
    valid = np.isfinite(values)
    order = np.argsort(values[valid], kind="stable")
    x, label, weight = values[valid][order], targets[valid][order], weights[valid][order]
    starts = np.flatnonzero(np.r_[True, x[1:] != x[:-1]]) if len(x) else np.array([], int)
    unique = x[starts]
    distributions, result = [], {}
    for target, name in enumerate(("normal", "anomaly")):
        chosen = (targets == target) & valid
        stats = dict(denominators[target], valid=int(chosen.sum()),
                     weighted_valid=float(weights[chosen].sum()), valid_frames=len(np.unique(frames[chosen])))
        stats.update(missing=stats["sampled"] - stats["valid"],
                     weighted_missing=float(weights[(targets == target) & ~valid].sum()),
                     p10=None, median=None, p90=None)
        mass = np.add.reduceat(np.where(label == target, weight, 0), starts) if len(starts) else np.array([])
        cumulative = mass.cumsum()
        if len(cumulative) and cumulative[-1] > 0:
            for key, quantile in (("p10", .1), ("median", .5), ("p90", .9)):
                stats[key] = float(unique[np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])
        result[name] = stats
        distributions.append((mass, cumulative))
    (normal, normal_cdf), (anomaly, anomaly_cdf) = distributions
    result.update(ks=None, auc_greater=None)
    if len(normal_cdf) and normal_cdf[-1] > 0 and anomaly_cdf[-1] > 0:
        result["ks"] = float(np.max(np.abs(normal_cdf / normal_cdf[-1] - anomaly_cdf / anomaly_cdf[-1])))
        # Every tie block contributes half its normal mass to the positive-direction AUC.
        result["auc_greater"] = float(np.sum(anomaly * (normal_cdf - .5 * normal)) / normal_cdf[-1] / anomaly_cdf[-1])
    return result


def descriptive_matches(sample, reference):
    """Describe weighted val label distributions without fitting or choosing direction.

    Normal sampling weights expand uniformly sampled points within each frame.
    Unsupported condition values are included in 'all', never pooled as one cell.
    """
    arrays = _arrays(sample)
    sequence, target, weight = (np.asarray(sample[name]) for name in ("sequence", "target", "weight"))
    size = len(arrays["frame"])
    if any(value.ndim != 1 or len(value) != size for value in (sequence, target, weight)):
        raise ValueError("Descriptive labels, weights and sequences must share point identities")
    if not np.isin(target, (0, 1)).all() or not (np.isfinite(weight) & (weight > 0)).all():
        raise ValueError("Descriptive targets must be 0/1 and weights finite and positive")
    if not size:
        return []
    weight = weight.astype(np.float64, copy=False)
    # A frame number alone is not an identity when several sequences are pooled.
    sequence_id = np.unique(sequence, return_inverse=True)[1]
    frame = arrays["frame"].astype(np.int64)
    frame_id = sequence_id * (int(frame.max()) - int(frame.min()) + 1) + frame - frame.min()
    rows = []
    cells = {"all": np.zeros(size, np.int16), **_cells(arrays, reference["edges"])}
    for mode, identities in cells.items():
        order = np.argsort(identities, kind="stable")
        groups, starts, counts = np.unique(identities[order], return_index=True, return_counts=True)
        for cell, start, count in zip(groups, starts, counts):
            if cell < 0:
                continue
            indices = order[start:start + count]
            labels, weights, frames = target[indices], weight[indices], frame_id[indices]
            denominators = []
            for label in (0, 1):
                chosen = labels == label
                denominators.append(dict(sampled=int(chosen.sum()), weighted_total=float(weights[chosen].sum()),
                                         frames=len(np.unique(frames[chosen]))))
            support = reference["metadata"]["cells"].get(mode, {}).get(str(cell), {})
            for feature in (*GEOMETRY, *NORMALIZED[:2]):
                trusted = (reference["metadata"]["global_fields"][feature]["trusted"] if mode == "all"
                           else support.get("trusted_all", False))
                row = dict(conditioning=mode, cell=None if mode == "all" else int(cell),
                           feature=feature, reference_trusted=trusted)
                row.update(_matched_field(arrays[feature][indices], labels, weights, frames, denominators))
                rows.append(row)
    return rows


def subgroup_diagnostics(sample, reference, thresholds):
    """Describe fixed-threshold detection by observable conditions, without AP.

    All official anomalies are retained; normal rows are uniformly sampled per
    frame with inverse sampling weights. Normal FPR and coverage are estimates.
    Recall among covered anomalies is exact; thresholds are not FPR-matched here.
    """
    from .geometry import comparisons

    arrays = _arrays(sample)
    sequence, target, weight = (np.asarray(sample[name]) for name in ("sequence", "target", "weight"))
    size = len(arrays["frame"])
    if any(value.ndim != 1 or len(value) != size for value in (sequence, target, weight)):
        raise ValueError("Subgroup labels, weights and sequences must share point identities")
    if not np.isin(target, (0, 1)).all() or not (np.isfinite(weight) & (weight > 0)).all():
        raise ValueError("Subgroup targets must be 0/1 and weights finite and positive")
    if np.any(weight[target == 1] != 1):
        raise ValueError("Subgroup diagnostics require every anomaly with unit weight")
    if not size:
        return []
    weight = weight.astype(np.float64, copy=False)
    normal, anomaly = target == 0, target == 1
    sequence_id = np.unique(sequence, return_inverse=True)[1]
    frame = arrays["frame"].astype(np.int64)
    identity = sequence_id * (int(frame.max()) - int(frame.min()) + 1) + frame - frame.min()
    _, inverse = np.unique(identity, return_inverse=True)
    # Every anomaly is present, so this is the full official count for each frame.
    frame_anomalies = np.bincount(inverse[anomaly], minlength=int(inverse.max()) + 1)
    frame_counts = frame_anomalies[inverse]
    strata = []
    for axis, values, edges in (
        ("range", arrays["range"], np.array([2.5, 10, 20, 35, 50.])),
        ("returns", frame_counts, np.array([5, 20, 100, 500, np.inf])),
        ("scale", arrays["scale"], reference["edges"]["scale"]),
    ):
        # Slot zero retains missing or out-of-reference conditions explicitly.
        bins = _bin(values, edges) + 1
        count = max(1, len(edges) - 1) + 1 if len(edges) else 1
        strata.append((axis, edges, bins, count,
                       np.bincount(bins[normal], minlength=count),
                       np.bincount(bins[normal], weights=weight[normal], minlength=count),
                       np.bincount(bins[anomaly], minlength=count)))
    scores, individual, _ = score_reference(arrays, reference)
    rows = []
    for cohort, methods, covered in comparisons(scores, individual):
        if cohort.startswith("feature/"):
            break
        covered_normal, covered_anomaly = covered & normal, covered & anomaly
        for axis, edges, bins, count, normal_n, normal_total, anomaly_total in strata:
            normal_covered_n = np.bincount(bins[covered_normal], minlength=count)
            normal_covered = np.bincount(bins[covered_normal], weights=weight[covered_normal], minlength=count)
            anomaly_covered = np.bincount(bins[covered_anomaly], minlength=count)
            for method, score in methods.items():
                threshold = thresholds[cohort + "|" + method]
                accepted = np.zeros(size, bool) if threshold is None else covered & (score >= threshold)
                fp_mask, tp_mask = accepted & normal, accepted & anomaly
                normal_fp_n = np.bincount(bins[fp_mask], minlength=count)
                normal_fp = np.bincount(bins[fp_mask], weights=weight[fp_mask], minlength=count)
                anomaly_tp = np.bincount(bins[tp_mask], minlength=count)
                for group in range(count):
                    if not normal_n[group] and not anomaly_total[group]:
                        continue
                    missing = group == 0
                    index = group - 1
                    lower = None if missing else float(edges[index])
                    upper = None if missing else float(edges[min(index + 1, len(edges) - 1)])
                    upper = upper if upper is None or np.isfinite(upper) else None
                    n_total, n_covered = float(normal_total[group]), float(normal_covered[group])
                    a_total, a_covered = int(anomaly_total[group]), int(anomaly_covered[group])
                    rows.append(dict(
                        axis=axis, group="missing_or_outside_reference" if missing else str(index),
                        lower=lower, upper=upper, upper_inclusive=not missing and index == count - 2 and upper is not None,
                        cohort=cohort, model=method, threshold=threshold,
                        threshold_source="train/206", fpr_matched=False,
                        normal_sampled_total=int(normal_n[group]), normal_weighted_total=n_total,
                        normal_sampled_covered=int(normal_covered_n[group]), normal_weighted_covered=n_covered,
                        normal_sampled_fp=int(normal_fp_n[group]), normal_weighted_fp=float(normal_fp[group]),
                        anomaly_total=a_total, anomaly_covered=a_covered, anomaly_tp=int(anomaly_tp[group]),
                        normal_coverage_estimate_percent=100 * n_covered / n_total if n_total else None,
                        anomaly_coverage_percent=100 * a_covered / a_total if a_total else None,
                        normal_fpr_estimate_percent=100 * float(normal_fp[group]) / n_covered if n_covered else None,
                        anomaly_recall_covered_percent=100 * int(anomaly_tp[group]) / a_covered if a_covered else None,
                        anomaly_detected_fraction_all_percent=100 * int(anomaly_tp[group]) / a_total if a_total else None,
                    ))
    return rows
