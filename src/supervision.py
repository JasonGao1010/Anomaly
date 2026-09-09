"""Reusable single-scan supervision and a fixed, real-sample implementation pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree, ConvexHull, QhullError

from .data import FrozenDataset, FrozenFrame, _atomic_json, host_disk, source_identity
from .render import calibrated_ray_grid, canonical_ray_slots_for_source, shape_from_dict, shape_geometry


REASONS = {
    "boundary": (
        "query_label_unknown_or_conflicting", "sampling_scale_unavailable",
        "local_label_support_unreliable", "no_observed_interface_path",
    ),
    "sampling": (
        "unknown_label", "sampling_scale_unavailable", "local_unknown_labels",
        "lost_label_group_support", "lost_spatial_span",
    ),
    "surface": (
        "unknown_query_label", "insufficient_reference_support",
        "rank_deficient_or_nonfinite", "insufficient_inliers",
        "plane_fit_unreliable", "outside_reference_hull",
        "insufficient_visible_support", "outside_visible_hull",
        "visible_support_unreliable",
    ),
}


def loss_point_weights(labels, valid, *, dtype, boundary_target=None, region_weights=(.75, .25)):
    """Normalize within classes, then regions; input rows belong to one batch/view."""
    import torch

    if labels.ndim != 1 or valid.shape != labels.shape or valid.dtype != torch.bool:
        raise ValueError("loss labels and boolean validity must be aligned point vectors")
    if boundary_target is None:
        regions = ((valid, 1.0),)
    else:
        if boundary_target.shape != labels.shape or len(region_weights) != 2 or min(region_weights) <= 0:
            raise ValueError("boundary loss needs aligned targets and two positive region weights")
        regions = ((valid & (boundary_target < 1), region_weights[0]),
                   (valid & (boundary_target == 1), region_weights[1]))
    # Keep reductions in at least float32, even under mixed-precision prediction.
    dtype = torch.float64 if dtype == torch.float64 else torch.float32
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    active_weight = weights.new_zeros(())
    for region, weight in regions:
        groups = torch.stack((region & (labels == 0), region & (labels == 1)))
        counts = groups.sum(dim=1)
        present = (counts > 0).sum()
        within = (groups.to(dtype) / counts.clamp_min(1).to(dtype)[:, None]).sum(dim=0)
        weights += weight * within / present.clamp_min(1)
        active_weight += weight * (present > 0)
    return weights / active_weight.clamp_min(torch.finfo(dtype).tiny)


def boundary_loss(prediction, target, labels, valid, region_weights=(.75, .25)):
    """C1: balanced L1 within each region, with a 3:1 band/saturation default."""
    weights = loss_point_weights(labels, valid, dtype=prediction.dtype,
                                 boundary_target=target, region_weights=region_weights)
    selected = valid & ((labels == 0) | (labels == 1))
    # Mask before arithmetic: an ignored NaN must not contaminate the loss or gradient.
    error = (prediction[selected] - target.detach()[selected]).abs()
    return (error * weights[selected]).sum()


def sampling_loss(sparse_logits, dense_logits, dense_row, labels, valid):
    """C2: balanced squared score consistency; the dense branch receives no gradient."""
    weights = loss_point_weights(labels, valid, dtype=sparse_logits.dtype)
    selected = valid & ((labels == 0) | (labels == 1))
    sparse = sparse_logits[selected].to(weights.dtype).sigmoid()
    dense = dense_logits[dense_row[selected]].detach().to(weights.dtype).sigmoid()
    return ((sparse - dense).square() * weights[selected]).sum()


def surface_loss(prediction, target, labels, valid):
    """C3: equal normal/anomaly means of the valid surface-offset absolute error."""
    weights = loss_point_weights(labels, valid, dtype=prediction.dtype)
    selected = valid & ((labels == 0) | (labels == 1))
    error = (prediction[selected] - target.detach()[selected]).abs()
    return (error * weights[selected]).sum()


def detection_loss(logits, labels):
    """Detection is supervised even when every auxiliary mask is false."""
    import torch
    valid = labels >= 0
    weights = loss_point_weights(labels, valid, dtype=logits.dtype)
    error = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[valid].float(), labels[valid].float(), reduction="none")
    return (error * weights[valid]).sum()


class ScanGeometry:
    """Label-free geometry; duplicate positions retain every original input row."""

    def __init__(self, xyzi, slots, parameters, workers=1):
        self.xyzi = np.asarray(xyzi)
        self.slots = np.asarray(slots, np.int32)
        if self.xyzi.shape != (len(self.slots), 4) or np.any(np.diff(self.slots) <= 0):
            raise ValueError("view requires xyzi and ascending unique original slots")
        if not np.isfinite(self.xyzi).all() or np.any(np.all(self.xyzi[:, :3] == 0, axis=1)):
            raise ValueError("view must contain only finite actual returns")
        self.xyz, self.first, self.inverse = np.unique(
            self.xyzi[:, :3].astype(np.float64), axis=0,
            return_index=True, return_inverse=True,
        )
        self.representative = self.slots[self.first]
        self.tree = cKDTree(self.xyz)
        self.workers = workers
        self.parameters = parameters
        count, k = len(self.xyz), parameters["neighbors"]
        radius = parameters["radius_m"]
        self.neighbors = np.full((count, k), -1, np.int32)
        distances = np.full((count, k), np.inf)
        if count:
            # One extra neighbor exposes a tie at the retained-neighbor boundary.
            _, ids = self.tree.query(
                self.xyz, k=np.arange(1, k + 3),
                distance_upper_bound=np.nextafter(radius, np.inf), workers=workers,
            )
            padded_xyz = np.vstack((self.xyz, [np.inf, np.inf, np.inf]))
            measured = np.linalg.norm(padded_xyz[ids] - self.xyz[:, None], axis=2)
            measured[(ids == np.arange(count)[:, None]) | (measured > radius)] = np.inf
            representatives = np.r_[self.representative, np.iinfo(np.int32).max]
            order = np.lexsort((representatives[ids], measured), axis=1)
            measured = np.take_along_axis(measured, order, axis=1)
            ids = np.take_along_axis(ids, order, axis=1)
            distances[:] = measured[:, :k]
            self.neighbors[:] = np.where(np.isfinite(distances), ids[:, :k], -1)
            tied = np.flatnonzero(np.isfinite(measured[:, k]) & (measured[:, k - 1] == measured[:, k]))
            for i in tied:
                candidates = np.asarray(self.tree.query_ball_point(
                    self.xyz[i], np.nextafter(measured[i, k], np.inf)), np.int32)
                ds = np.linalg.norm(self.xyz[candidates] - self.xyz[i], axis=1)
                keep = (ds > 0) & (ds <= radius)
                candidates, ds = candidates[keep], ds[keep]
                take = np.lexsort((self.representative[candidates], ds))[:k]
                self.neighbors[i] = candidates[take]
                distances[i] = ds[take]
        counts = np.isfinite(distances).sum(axis=1)
        self.scale_valid = counts >= parameters["minimum_distinct_neighbors"]
        self.delta = np.zeros(count)
        rows = np.flatnonzero(self.scale_valid)
        self.delta[rows] = np.maximum(
            (distances[rows, (counts[rows] - 1) // 2] + distances[rows, counts[rows] // 2]) / 2,
            parameters["epsilon_m"],
        )

    def groups(self, labels):
        labels = np.asarray(labels, np.int8)
        if labels.shape != self.slots.shape or not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError("labels must be the aligned frozen binary detection targets")
        low = np.full(len(self.xyz), 2, np.int8)
        high = np.full(len(self.xyz), -2, np.int8)
        np.minimum.at(low, self.inverse, labels)
        np.maximum.at(high, self.inverse, labels)
        return np.where(low == high, low, -1).astype(np.int8)

    def scale_arrays(self):
        return dict(sampling_scale=self.delta[self.inverse].astype(np.float32),
                    sampling_scale_valid=self.scale_valid[self.inverse])


def boundary_targets(geometry, labels, parameters):
    """Half-edge seeds followed by shortest paths strictly within each label."""
    g = geometry
    groups = g.groups(labels)
    n = len(groups)
    radius = np.minimum(g.parameters["radius_m"], parameters["scale_multiplier"] * g.delta)
    same_count = np.zeros(n, np.int32)
    unknown_count = np.zeros(n, np.int32)
    for label in (0, 1):
        rows = np.flatnonzero(groups == label)
        if len(rows):
            same_count[rows] = cKDTree(g.xyz[rows]).query_ball_point(
                g.xyz[rows], radius[rows], return_length=True, workers=g.workers)
    if np.any(groups < 0):
        unknown_count = cKDTree(g.xyz[groups < 0]).query_ball_point(
            g.xyz, radius, return_length=True, workers=g.workers)
    reason = np.zeros(n, np.uint16)
    reason[groups < 0] |= 1
    reason[~g.scale_valid] |= 2
    reason[(unknown_count > 0) | (same_count < parameters["minimum_same_label_positions"])] |= 4
    eligible = reason == 0
    u = np.repeat(np.arange(n), g.neighbors.shape[1])
    v = g.neighbors.ravel()
    keep = (v >= 0) & (u < v)
    u, v = u[keep], v[keep]
    keep = eligible[u] & eligible[v] & np.any(g.neighbors[v] == u[:, None], axis=1)
    u, v = u[keep], v[keep]
    length = np.linalg.norm(g.xyz[u] - g.xyz[v], axis=1)
    keep = length <= np.minimum(radius[u], radius[v])
    u, v, length = u[keep], v[keep], length[keep]
    cross = groups[u] != groups[v]
    a, b = u[cross], v[cross]
    normal = np.where(groups[a] == 0, a, b)
    anomaly = np.where(groups[a] == 1, a, b)
    edges = np.column_stack((g.representative[normal], g.representative[anomaly]))
    if len(edges):
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    seed = np.full(n, np.inf)
    np.minimum.at(seed, a, length[cross] / 2)
    np.minimum.at(seed, b, length[cross] / 2)
    starts = np.flatnonzero(np.isfinite(seed))
    su, sv, sw = u[~cross], v[~cross], length[~cross]
    # A virtual source sets unequal initial seed distances without crossing labels.
    graph = csr_matrix((np.r_[sw, sw, seed[starts]],
                        (np.r_[su, sv, np.full(len(starts), n)], np.r_[sv, su, starts])),
                       shape=(n + 1, n + 1))
    distance, predecessor = dijkstra(graph, directed=True, indices=n, return_predecessors=True)
    distance = distance[:n]
    valid = eligible & np.isfinite(distance)
    reason[eligible & ~valid] |= 8
    target = np.zeros(n)
    target[valid] = np.minimum(distance[valid] / (
        parameters["scale_multiplier"] * g.delta[valid] + g.parameters["epsilon_m"]), 1)
    return dict(boundary_edges=edges.astype(np.int32),
                boundary_distance=target[g.inverse].astype(np.float32),
                boundary_valid=valid[g.inverse], boundary_ignore_reason=reason[g.inverse],
                _groups=groups, _distance=distance, _predecessor=predecessor,
                _seed=seed, _u=su, _v=sv, _length=sw,
                _local_unknown=unknown_count, _same_count=same_count)


def thinning_pair(sample, original, grid, world_seed, level, seed):
    """Remove rays, never re-render returns or use labels to choose a phase."""
    raw_ray = canonical_ray_slots_for_source(original, grid)
    canonical = grid.canonical_ray_by_slot[raw_ray]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([
        seed, original.sequence_id, world_seed, original.frame_id, level["level_index"]])))
    b, c = level["beam_stride"], level["column_stride"]
    phase_b, phase_c = int(rng.integers(b)), int(rng.integers(c))
    ray_keep = ((canonical // grid.columns) % b == phase_b) & ((canonical % grid.columns) % c == phase_c)
    rows = np.flatnonzero(ray_keep[sample.source.real_slots]).astype(np.int32)
    return dict(sparse_source_slot=sample.source.real_slots[rows].astype(np.int32),
                dense_row=rows, ray_keep=ray_keep,
                phase=dict(**level, phase_b=phase_b, phase_c=phase_c))


@njit
def _group_extents(xyz, groups, indptr, indices):
    n = len(indptr) - 1
    counts = np.zeros((n, 2), np.int32)
    spans = np.zeros((n, 2))
    unknown = np.zeros(n, np.bool_)
    for i in range(n):
        lower = np.full((2, 3), np.inf)
        upper = np.full((2, 3), -np.inf)
        for j in indices[indptr[i]:indptr[i + 1]]:
            group = groups[j]
            if group < 0:
                unknown[i] = True
                continue
            counts[i, group] += 1
            for k in range(3):
                lower[group, k] = min(lower[group, k], xyz[j, k])
                upper[group, k] = max(upper[group, k], xyz[j, k])
        for group in range(2):
            if counts[i, group]:
                spans[i, group] = np.sqrt(np.sum((upper[group] - lower[group]) ** 2))
    return counts, spans, unknown


def _csr_neighbors(neighborhoods):
    lengths = np.fromiter((len(a) for a in neighborhoods), np.int64, count=len(neighborhoods))
    return np.r_[0, np.cumsum(lengths)], np.concatenate(neighborhoods).astype(np.int64) if lengths.sum() else np.empty(0, np.int64)


def local_evidence(geometry, labels, radius, block_size=512):
    """Bound neighborhood memory while retaining the complete current scan."""
    g = geometry
    groups = g.groups(labels)
    count = np.zeros((len(g.xyz), 2), np.int32)
    span = np.zeros((len(g.xyz), 2))
    unknown = np.zeros(len(g.xyz), bool)
    for start in range(0, len(g.xyz), block_size):
        stop = min(start + block_size, len(g.xyz))
        neighbors = g.tree.query_ball_point(g.xyz[start:stop], radius, workers=g.workers)
        ptr, ids = _csr_neighbors(neighbors)
        count[start:stop], span[start:stop], unknown[start:stop] = _group_extents(g.xyz, groups, ptr, ids)
    return dict(count=count, span=span, unknown=unknown)


def sampling_targets(dense, sparse, dense_labels, sparse_labels, pair, parameters, dense_evidence=None):
    """Keep score consistency only where each observed label group retains support."""
    de = dense_evidence if dense_evidence is not None else local_evidence(dense, dense_labels, parameters["radius_m"])
    se = local_evidence(sparse, sparse_labels, parameters["radius_m"])
    match = dense.inverse[pair["dense_row"][sparse.first]]
    groups = sparse.groups(sparse_labels)
    reason = np.zeros(len(sparse.xyz), np.uint16)
    reason[groups < 0] |= 1
    reason[~sparse.scale_valid | ~dense.scale_valid[match]] |= 2
    reason[se["unknown"] | de["unknown"][match]] |= 4
    present = de["count"][match] > 0
    enough = (de["count"][match] >= parameters["minimum_positions_per_present_label_group"]) & (se["count"] >= parameters["minimum_positions_per_present_label_group"])
    reason[np.any(present & ~enough, axis=1)] |= 8
    retained_span = (de["span"][match] > dense.parameters["epsilon_m"]) & (
        se["span"] >= parameters["minimum_retained_span_ratio"] * de["span"][match])
    reason[np.any(present & ~retained_span, axis=1)] |= 16
    return dict(sampling_consistency_valid=(reason == 0)[sparse.inverse],
                sampling_ignore_reason=reason[sparse.inverse], _dense_evidence=de,
                _sparse_evidence=se, _match=match)


@njit
def _inside_hull(xy, tolerance):
    """Andrew's hull on lexicographically sorted XY; query is the origin."""
    n = len(xy)
    hull = np.empty(2 * n, np.int64)
    size = 0
    for j in range(n):
        while size >= 2:
            a, b = xy[hull[size - 2]], xy[hull[size - 1]]
            if (b[0] - a[0]) * (xy[j, 1] - a[1]) - (b[1] - a[1]) * (xy[j, 0] - a[0]) > 0:
                break
            size -= 1
        hull[size] = j
        size += 1
    lower = size + 1
    for j in range(n - 2, -1, -1):
        while size >= lower:
            a, b = xy[hull[size - 2]], xy[hull[size - 1]]
            if (b[0] - a[0]) * (xy[j, 1] - a[1]) - (b[1] - a[1]) * (xy[j, 0] - a[0]) > 0:
                break
            size -= 1
        hull[size] = j
        size += 1
    size -= 1
    if size < 3:
        return False
    for j in range(size):
        a, b = xy[hull[j]], xy[hull[(j + 1) % size]]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = np.sqrt(dx * dx + dy * dy)
        if length == 0 or dy * a[0] - dx * a[1] < -tolerance * length:
            return False
    return True


@njit
def _xy_eigenvalue(xy):
    centered = xy - np.array([np.mean(xy[:, 0]), np.mean(xy[:, 1])])
    covariance = centered.T @ centered / (len(xy) - 1)
    return np.linalg.eigvalsh(covariance)[0]


@njit
def _plane_fit(points):
    matrix = np.ones((len(points), 3))
    matrix[:, :2] = points[:, :2]
    # Match NumPy's default rank cutoff, independent of LAPACK thread count.
    coefficients, _, rank, _ = np.linalg.lstsq(
        matrix, points[:, 2].copy(), np.finfo(np.float64).eps * max(len(points), 3))
    return coefficients, rank, matrix


@njit(parallel=True)
def _surface_chunk(queries, ground, visible, indptr, indices, query_known, p, visible_minimums):
    minimum, residual_floor, mad_multiplier, mad_normalization, rmse_limit, slope_limit, eigen_limit, tolerance = p
    n, views = query_known.shape
    levels = len(visible_minimums)
    visible_floor = np.min(visible_minimums)
    offset = np.zeros((n, views, levels))
    reason = np.zeros((n, views, levels), np.uint16)
    inlier_count = np.zeros(n, np.int32)
    seen_count = np.zeros((n, views), np.int32)
    quality = np.zeros(n, np.uint8)
    seen_quality = np.zeros((n, views, levels), np.uint8)
    for i in prange(n):
        ids = indices[indptr[i]:indptr[i + 1]]
        for view in range(views):
            if not query_known[i, view]:
                reason[i, view] = 1
        if not np.any(query_known[i]):
            continue
        failure = 0
        if len(ids) < minimum:
            failure = 2
        else:
            points = ground[ids].copy()
            points[:, 0] -= queries[i, 0]
            points[:, 1] -= queries[i, 1]
            coef, rank, matrix = _plane_fit(points)
            if rank != 3 or not np.isfinite(coef).all():
                failure = 4
            else:
                residual = points[:, 2] - matrix @ coef
                mad = mad_normalization * np.median(np.abs(residual - np.median(residual)))
                keep = np.abs(residual) <= max(residual_floor, mad_multiplier * mad)
                points, ids = points[keep], ids[keep]
                inlier_count[i] = len(points)
                if len(points) < minimum:
                    failure = 8
                else:
                    coef, rank, matrix = _plane_fit(points)
                    if rank != 3 or not np.isfinite(coef).all():
                        failure = 4
                    else:
                        residual = points[:, 2] - matrix @ coef
                        rmse = np.sqrt(np.mean(residual ** 2))
                        slope = np.sqrt(coef[0] ** 2 + coef[1] ** 2)
                        eigen = _xy_eigenvalue(points[:, :2])
                        quality[i] = (int(not np.isfinite(rmse) or rmse > rmse_limit)
                                      + 2 * int(not np.isfinite(slope) or slope > slope_limit)
                                      + 4 * int(not np.isfinite(eigen) or eigen < eigen_limit))
                        if quality[i]:
                            failure = 16
                        elif not _inside_hull(points[:, :2], tolerance):
                            failure = 32
        if failure:
            for view in range(views):
                if query_known[i, view]:
                    reason[i, view] = failure
            continue
        for view in range(views):
            if not query_known[i, view]:
                continue
            keep = visible[ids, view]
            seen = points[keep]
            seen_count[i, view] = len(seen)
            seen_failure, rejection = 0, 0
            if len(seen) >= visible_floor:
                # Both thresholds share this unchanged reference plane and geometry check.
                _, rank, seen_matrix = _plane_fit(seen)
                if rank != 3:
                    seen_failure = 4
                else:
                    eigen = _xy_eigenvalue(seen[:, :2])
                    rmse = np.sqrt(np.mean((seen[:, 2] - seen_matrix @ coef) ** 2))
                    rejection = (int(not np.isfinite(rmse) or rmse > rmse_limit)
                                 + 4 * int(not np.isfinite(eigen) or eigen < eigen_limit))
                    if rejection:
                        seen_failure = 256
                    elif not _inside_hull(seen[:, :2], tolerance):
                        seen_failure = 128
            for level in range(levels):
                if len(seen) < visible_minimums[level]:
                    reason[i, view, level] = 64
                else:
                    reason[i, view, level] = seen_failure
                    seen_quality[i, view, level] = rejection
                    if seen_failure == 0:
                        offset[i, view, level] = coef[2] - queries[i, 2]
    return offset, reason, inlier_count, seen_count, quality, seen_quality


def surface_reference_data(original, sample, pairs, parameters):
    slots = original.real_slots
    slots = slots[(original.labels.semantic_target[slots] != 255)
                  & np.isin(original.labels.semantic[slots], parameters["ground_semantics"])]
    ground, inverse = np.unique(original.xyzi[slots, :3].astype(np.float64), axis=0, return_inverse=True)
    unchanged = ~(sample.inserted_mask | sample.occluded_original_mask | sample.source.zero_slot_mask)
    visible = np.zeros((len(ground), 1 + len(pairs)), bool)
    for view, keep in enumerate([np.ones(original.slot_count, bool)] + [p["ray_keep"] for p in pairs]):
        # Any surviving alias suffices; aliases never increase distinct support counts.
        np.logical_or.at(visible[:, view], inverse, unchanged[slots] & keep[slots])
    return ground, visible


def surface_targets(original, sample, geometries, labels, pairs, parameters, visible_minimums=None, reference_cache=None):
    """Return {visible minimum: view targets}, sharing every reference plane."""
    thresholds = np.asarray(visible_minimums if visible_minimums is not None
                            else [parameters["minimum_visible_support_points"]], np.int64)
    if thresholds.ndim != 1 or not len(thresholds) or np.any(thresholds < 3) or len(np.unique(thresholds)) != len(thresholds):
        raise ValueError("visible support thresholds must be distinct integers of at least three")
    dense = geometries[0]
    ground, visible = surface_reference_data(original, sample, pairs, parameters)
    tree = cKDTree(ground[:, :2])
    n, views = len(dense.xyz), len(geometries)
    match = [np.arange(n)] + [dense.inverse[p["dense_row"][g.first]] for g, p in zip(geometries[1:], pairs)]
    known = np.zeros((n, views), bool)
    for view, (g, y, rows) in enumerate(zip(geometries, labels, match)):
        known[rows, view] = g.groups(y) >= 0
    counts = tree.query_ball_point(dense.xyz[:, :2], parameters["xy_radius_m"],
                                   return_length=True, workers=dense.workers)
    reason = np.repeat(np.where(known, 2, 1)[..., None], len(thresholds), axis=2).astype(np.uint16)
    offset = np.zeros((n, views, len(thresholds)))
    inlier_count = np.zeros(n, np.int32)
    seen_count = np.zeros((n, views), np.int32)
    quality = np.zeros(n, np.uint8)
    seen_quality = np.zeros((n, views, len(thresholds)), np.uint8)
    affected = np.ones(n, bool)
    if reference_cache is not None:
        if (reference_cache["source"] != source_identity(original) or reference_cache["parameters"] != parameters
                or list(thresholds) != [parameters["minimum_visible_support_points"]]):
            raise ValueError("normal-surface reuse has different scientific inputs")
        changed = sample.inserted_mask | sample.occluded_original_mask
        affected = ~known.any(axis=1)
        np.logical_or.at(affected, dense.inverse, changed[dense.slots])
        changed_returns = changed & ~original.zero_slot_mask
        if np.any(changed_returns):
            nearest = cKDTree(original.xyzi[changed_returns, :3].astype(float)).query(dense.xyz, workers=dense.workers)[0]
            affected |= nearest == 0  # Removing a conflicting alias can change query-label availability.
        ground_slots = reference_cache["ground_slots"]
        changed_ground = ground_slots[changed[ground_slots]]
        if len(changed_ground):
            # Only queries whose original reference disk intersects a changed support can differ.
            distance = cKDTree(original.xyzi[changed_ground, :2].astype(float)).query(dense.xyz[:, :2], workers=dense.workers)[0]
            affected |= distance <= np.nextafter(parameters["xy_radius_m"], np.inf)
        for pair in pairs:
            key = _phase_key(pair["phase"])
            if not np.array_equal(pair["ray_keep"], reference_cache["views"][key]["ray_keep"]):
                raise ValueError("surface cache uses a different exact ray mask")
    rows = np.flatnonzero((counts >= parameters["minimum_reference_support_points"]) & known.any(axis=1) & affected)
    p = np.array([parameters["minimum_reference_support_points"], parameters["minimum_inlier_residual_m"],
                  parameters["mad_multiplier"], parameters["mad_normalization"],
                  parameters["maximum_plane_rmse_m"], np.tan(np.deg2rad(parameters["maximum_slope_degrees"])),
                  parameters["minimum_xy_covariance_eigenvalue_m2"], parameters["convex_hull_tolerance_m"]])
    start = 0
    while start < len(rows):
        # Bound the ragged neighbor list as well as the number of simultaneous fits.
        cumulative = np.cumsum(counts[rows[start:start + 1024]])
        size = max(1, min(len(cumulative), int(np.searchsorted(cumulative, 500_000)) + 1))
        chunk = rows[start:start + size]
        neighbors = tree.query_ball_point(dense.xyz[chunk, :2], parameters["xy_radius_m"],
                                          workers=dense.workers, return_sorted=True)
        ptr, ids = _csr_neighbors(neighbors)
        values = _surface_chunk(dense.xyz[chunk], ground, visible, ptr, ids, known[chunk], p, thresholds)
        offset[chunk], reason[chunk], inlier_count[chunk], seen_count[chunk], quality[chunk], seen_quality[chunk] = values
        start += size
    results = {}
    for level, threshold in enumerate(thresholds):
        results[int(threshold)] = []
        for view, (g, rows) in enumerate(zip(geometries, match)):
            back = rows[g.inverse]
            results[int(threshold)].append(dict(surface_offset_z=offset[back, view, level].astype(np.float32),
                                                surface_valid=reason[back, view, level] == 0,
                                                surface_ignore_reason=reason[back, view, level],
                                                surface_reference_count=counts[back].astype(np.int32),
                                                surface_inlier_count=inlier_count[back],
                                                surface_seen_count=seen_count[back, view],
                                                surface_plane_rejection=quality[back],
                                                surface_seen_rejection=seen_quality[back, view, level]))
            if reference_cache is not None:
                cached = reference_cache["views"]["base" if view == 0 else _phase_key(pairs[view-1]["phase"])]
                unchanged_rows = ~affected[back]
                for name, array in results[int(threshold)][view].items():
                    array[unchanged_rows] = cached[name][g.slots[unchanged_rows]]
    return results


def _phase_key(phase):
    return f'{phase["level_index"]}/{phase["phase_b"]}/{phase["phase_c"]}'


def normal_surface_cache(original, grid, protocol, workers):
    """Fit a normal source once for all declared phases; reuse only unaffected query disks."""
    empty = np.zeros(original.slot_count, bool)
    sample = FrozenFrame(original, source_identity(original), empty, empty)
    slots = original.real_slots
    canonical = grid.canonical_ray_by_slot[canonical_ray_slots_for_source(original, grid)]
    pairs = []
    for level in protocol["supervision"]["C2"]["levels"]:
        for b in range(level["beam_stride"]):
            for c in range(level["column_stride"]):
                keep = (((canonical // grid.columns) % level["beam_stride"] == b)
                        & ((canonical % grid.columns) % level["column_stride"] == c))
                rows = np.flatnonzero(keep[slots]).astype(np.int32)
                pairs.append(dict(ray_keep=keep, dense_row=rows, sparse_source_slot=slots[rows],
                                  phase=dict(**level, phase_b=b, phase_c=c)))
    scale = protocol["supervision"]["common"]["sampling_scale"]
    geometries = [ScanGeometry(original.xyzi[slots], slots, scale, workers)]
    geometries.extend(ScanGeometry(original.xyzi[p["sparse_source_slot"]], p["sparse_source_slot"], scale, workers) for p in pairs)
    labels = [sample.anomaly_target[g.slots] for g in geometries]
    parameters = protocol["supervision"]["C3"]["parameters"]
    values = surface_targets(original, sample, geometries, labels, pairs, parameters)[parameters["minimum_visible_support_points"]]
    views = {}
    for i, (g, target) in enumerate(zip(geometries, values)):
        record = {}
        for name, value in target.items():
            record[name] = np.zeros(original.slot_count, value.dtype)
            record[name][g.slots] = value
        record["ray_keep"] = np.ones(original.slot_count, bool) if i == 0 else pairs[i-1]["ray_keep"]
        views["base" if i == 0 else _phase_key(pairs[i-1]["phase"])] = record
    ground_slots = slots[(original.labels.semantic_target[slots] != 255)
                         & np.isin(original.labels.semantic[slots], parameters["ground_semantics"])]
    return dict(source=source_identity(original), parameters=parameters, ground_slots=ground_slots, views=views)


def compute_supervision(sample, original, grid, world_seed, protocol, workers=1, progress=None, surface_cache=None):
    """Training-only targets; each geometry object sees its complete current scan."""
    config = protocol["supervision"]
    slots = sample.source.real_slots
    labels = [sample.anomaly_target[slots]]
    pairs = [thinning_pair(sample, original, grid, world_seed, level, protocol["seed"])
             for level in config["C2"]["levels"]]
    geometries = [ScanGeometry(sample.source.xyzi[slots], slots, config["common"]["sampling_scale"], workers)]
    for pair in pairs:
        rows = pair["dense_row"]
        geometries.append(ScanGeometry(geometries[0].xyzi[rows], pair["sparse_source_slot"],
                                       config["common"]["sampling_scale"], workers))
        labels.append(labels[0][rows])
    result = []
    for g, y in zip(geometries, labels):
        result.append(dict(**g.scale_arrays(), **boundary_targets(g, y, config["C1"]["parameters"])))
    if progress:
        progress("scale_and_boundary")
    evidence = local_evidence(geometries[0], labels[0], config["C2"]["evidence_parameters"]["radius_m"])
    for g, y, pair, target in zip(geometries[1:], labels[1:], pairs, result[1:]):
        target.update(sampling_targets(geometries[0], g, labels[0], y, pair,
                                       config["C2"]["evidence_parameters"], evidence))
        target.update(sparse_source_slot=pair["sparse_source_slot"], dense_row=pair["dense_row"])
    if progress:
        progress("sampling_pairs")
    surfaces = surface_targets(original, sample, geometries, labels, pairs, config["C3"]["parameters"], reference_cache=surface_cache)
    for target, surface in zip(result, surfaces[config["C3"]["parameters"]["minimum_visible_support_points"]]):
        target.update(surface)
    if progress:
        progress("surface")
    return geometries, labels, pairs, result


def surface_probe(point, original, sample, view_slots, parameters):
    """Independent NumPy/SciPy check of one target, including original-slot visibility."""
    p = parameters
    slots = original.real_slots
    slots = slots[(original.labels.semantic_target[slots] != 255)
                  & np.isin(original.labels.semantic[slots], p["ground_semantics"])]
    slots = slots[np.linalg.norm(original.xyzi[slots, :2].astype(np.float64) - point[:2], axis=1) <= p["xy_radius_m"]]
    ground, inverse = np.unique(original.xyzi[slots, :3].astype(np.float64), axis=0, return_inverse=True)
    record = dict(query_xyz_m=point.tolist(), reference_positions=len(ground), valid=False)
    minimum = p["minimum_reference_support_points"]
    if len(ground) < minimum:
        return dict(record, reason="insufficient_reference_support")
    matrix = np.column_stack((ground[:, :2] - point[:2], np.ones(len(ground))))
    coef, _, rank, _ = np.linalg.lstsq(matrix, ground[:, 2], rcond=None)
    if rank != 3 or not np.isfinite(coef).all():
        return dict(record, reason="rank_deficient_or_nonfinite")
    residual = ground[:, 2] - matrix @ coef
    mad = p["mad_normalization"] * np.median(np.abs(residual - np.median(residual)))
    inlier = np.abs(residual) <= max(p["minimum_inlier_residual_m"], p["mad_multiplier"] * mad)
    record["inlier_positions"] = int(inlier.sum())
    if inlier.sum() < minimum:
        return dict(record, reason="insufficient_inliers")
    coef, _, rank, _ = np.linalg.lstsq(matrix[inlier], ground[inlier, 2], rcond=None)
    if rank != 3 or not np.isfinite(coef).all():
        return dict(record, reason="rank_deficient_or_nonfinite")
    residual = ground[:, 2] - matrix @ coef
    rmse = float(np.sqrt(np.mean(residual[inlier] ** 2)))
    eigen = float(np.linalg.eigvalsh(np.cov(matrix[inlier, :2], rowvar=False))[0])
    slope = float(np.linalg.norm(coef[:2]))
    record.update(plane_abc=coef.tolist(), reference_rmse_m=rmse, reference_xy_eigenvalue_m2=eigen,
                  surface_z_m=float(coef[2]), offset_z_m=float(coef[2] - point[2]))
    if rmse > p["maximum_plane_rmse_m"] or slope > np.tan(np.deg2rad(p["maximum_slope_degrees"])) or eigen < p["minimum_xy_covariance_eigenvalue_m2"]:
        return dict(record, reason="plane_fit_unreliable")
    def inside(xy):
        try:
            return bool(np.all(ConvexHull(xy).equations[:, -1] <= p["convex_hull_tolerance_m"]))
        except QhullError:
            return False
    if not inside(matrix[inlier, :2]):
        return dict(record, reason="outside_reference_hull")
    unchanged = ~(sample.inserted_mask[slots] | sample.occluded_original_mask[slots])
    present = np.isin(slots, view_slots)
    assert np.array_equal(sample.source.xyzi[slots[unchanged]], original.xyzi[slots[unchanged]])
    visible = np.zeros(len(ground), bool)
    np.logical_or.at(visible, inverse, unchanged & present)
    seen = visible & inlier
    record["seen_positions"] = int(seen.sum())
    if seen.sum() < p["minimum_visible_support_points"]:
        return dict(record, reason="insufficient_visible_support")
    if np.linalg.matrix_rank(matrix[seen]) != 3:
        return dict(record, reason="rank_deficient_or_nonfinite")
    seen_eigen = float(np.linalg.eigvalsh(np.cov(matrix[seen, :2], rowvar=False))[0])
    seen_rmse = float(np.sqrt(np.mean(residual[seen] ** 2)))
    record.update(seen_rmse_m=seen_rmse, seen_xy_eigenvalue_m2=seen_eigen)
    if seen_rmse > p["maximum_plane_rmse_m"] or seen_eigen < p["minimum_xy_covariance_eigenvalue_m2"]:
        return dict(record, reason="visible_support_unreliable")
    if not inside(matrix[seen, :2]):
        return dict(record, reason="outside_visible_hull")
    return dict(record, valid=True, reason="valid")


def _probe_rows(labels, valid):
    rows = list(np.flatnonzero(labels == 1))
    for state in (False, True):
        candidates = np.flatnonzero((labels == 0) & (valid == state))
        if len(candidates):
            rows.extend(candidates[[0, len(candidates) // 2, -1]].tolist())
    return np.unique(rows)


def audit_supervision(sample, original, geometries, labels, pairs, targets, config):
    """Global graph identities plus independent local checks, never coverage gates."""
    summary = dict(boundary=[], sampling=[], surface=[])
    for view, (g, y, target) in enumerate(zip(geometries, labels, targets)):
        for prefix, continuous in (("boundary", "boundary_distance"), ("surface", "surface_offset_z")):
            assert np.array_equal(target[prefix + "_valid"], target[prefix + "_ignore_reason"] == 0)
            assert np.all(target[continuous][~target[prefix + "_valid"]] == 0)
            assert np.isfinite(target[continuous]).all()
        edges = target["boundary_edges"]
        erows = np.searchsorted(g.slots, edges)
        assert np.all(y[erows[:, 0]] == 0) and np.all(y[erows[:, 1]] == 1)
        distance, seed = target["_distance"], target["_seed"]
        u, v, length = target["_u"], target["_v"], target["_length"]
        assert np.all(target["_groups"][u] == target["_groups"][v])
        # Bellman inequalities and a strictly descending predecessor prove the stored paths.
        assert np.all(distance <= seed + 1e-12)
        assert np.all(distance[v] <= distance[u] + length + 1e-12)
        assert np.all(distance[u] <= distance[v] + length + 1e-12)
        finite = np.flatnonzero(np.isfinite(distance))
        previous = target["_predecessor"][finite]
        from_seed = previous == len(g.xyz)
        np.testing.assert_allclose(distance[finite[from_seed]], seed[finite[from_seed]], rtol=0, atol=1e-12)
        a, b = finite[~from_seed], previous[~from_seed]
        assert np.all(b >= 0)
        assert np.all(target["_groups"][a] == target["_groups"][b])
        assert np.all(np.any(g.neighbors[a] == b[:, None], axis=1))
        np.testing.assert_allclose(distance[a], distance[b] + np.linalg.norm(g.xyz[a] - g.xyz[b], axis=1), rtol=0, atol=1e-12)
        probes = []
        for row in _probe_rows(y, target["boundary_valid"]):
            i = g.inverse[row]
            ds = np.linalg.norm(g.xyz - g.xyz[i], axis=1)
            candidates = np.flatnonzero((ds > 0) & (ds <= g.parameters["radius_m"]))
            ids = candidates[np.lexsort((g.representative[candidates], ds[candidates]))[:g.parameters["neighbors"]]]
            np.testing.assert_array_equal(g.neighbors[i][g.neighbors[i] >= 0], ids)
            if g.scale_valid[i]:
                assert abs(g.delta[i] - np.median(ds[ids])) < 1e-12
            if y[row] == 1 and target["boundary_valid"][row] and len(probes) < 3:
                probes.append(dict(source_slot=int(g.slots[row]), xyz_m=g.xyz[i].tolist(),
                                   distance_m=float(distance[i]), scale_m=float(g.delta[i]),
                                   target=float(target["boundary_distance"][row]), label=int(y[row])))
        summary["boundary"].append(dict(view=view, graph_positions=len(g.xyz),
                                         interface_edges=len(edges), finite_paths=len(finite),
                                         scale_probes=len(_probe_rows(y, target["boundary_valid"])), examples=probes))
        surface_examples = []
        for row in _probe_rows(y, target["surface_valid"]):
            probe = surface_probe(g.xyzi[row, :3].astype(np.float64), original, sample, g.slots, config["C3"]["parameters"])
            assert probe["valid"] == bool(target["surface_valid"][row]), (view, int(g.slots[row]), probe)
            if probe["valid"]:
                np.testing.assert_allclose(target["surface_offset_z"][row], probe["offset_z_m"], rtol=1e-6, atol=1e-6)
            else:
                assert target["surface_ignore_reason"][row] & (1 << REASONS["surface"].index(probe["reason"]))
            if len(surface_examples) < 2 or (y[row] == 1 and len(surface_examples) < 5):
                surface_examples.append(dict(source_slot=int(g.slots[row]), label=int(y[row]), **probe))
        summary["surface"].append(dict(view=view, probes=len(_probe_rows(y, target["surface_valid"])), examples=surface_examples))
        if view == 0:
            continue
        pair, dense = pairs[view - 1], geometries[0]
        np.testing.assert_array_equal(g.slots, dense.slots[pair["dense_row"]])
        np.testing.assert_array_equal(g.xyzi, dense.xyzi[pair["dense_row"]])
        np.testing.assert_array_equal(y, labels[0][pair["dense_row"]])
        assert np.array_equal(target["sampling_consistency_valid"], target["sampling_ignore_reason"] == 0)
        dg, sg = dense.groups(labels[0]), g.groups(y)
        p = config["C2"]["evidence_parameters"]
        examples = []
        for row in _probe_rows(y, target["sampling_consistency_valid"]):
            si, di = g.inverse[row], dense.inverse[pair["dense_row"][row]]
            ds = np.asarray(dense.tree.query_ball_point(g.xyz[si], p["radius_m"]), np.int64)
            ss = np.asarray(g.tree.query_ball_point(g.xyz[si], p["radius_m"]), np.int64)
            valid = (sg[si] >= 0 and g.scale_valid[si] and dense.scale_valid[di]
                     and np.all(dg[ds] >= 0) and np.all(sg[ss] >= 0))
            counts, spans = [], []
            for group in (0, 1):
                dxyz, sxyz = dense.xyz[ds[dg[ds] == group]], g.xyz[ss[sg[ss] == group]]
                counts.append([len(dxyz), len(sxyz)])
                dl = float(np.linalg.norm(np.ptp(dxyz, axis=0))) if len(dxyz) else 0.0
                sl = float(np.linalg.norm(np.ptp(sxyz, axis=0))) if len(sxyz) else 0.0
                spans.append([dl, sl])
                if len(dxyz):
                    valid &= (min(len(dxyz), len(sxyz)) >= p["minimum_positions_per_present_label_group"]
                              and dl > dense.parameters["epsilon_m"] and sl >= p["minimum_retained_span_ratio"] * dl)
            assert bool(valid) == target["sampling_consistency_valid"][row]
            if y[row] == 1 and len(examples) < 3:
                examples.append(dict(source_slot=int(g.slots[row]), valid=bool(valid),
                                     scale_dense_m=float(dense.delta[di]), scale_sparse_m=float(g.delta[si]),
                                     normal_anomaly_counts_dense_sparse=counts, normal_anomaly_spans_dense_sparse_m=spans))
        summary["sampling"].append(dict(view=view, exact_correspondences=len(y),
                                         probes=len(_probe_rows(y, target["sampling_consistency_valid"])), examples=examples))
    return summary


def summarize_view(geometry, labels, targets, dense_labels=None):
    result = dict(input_returns=len(labels), distinct_positions=len(geometry.xyz),
                  ignored_detection_returns=int(np.sum(labels < 0)))
    for label, name in ((0, "normal"), (1, "anomaly")):
        selected = labels == label
        entry = dict(total=int(selected.sum()), distinct_positions=len(np.unique(geometry.inverse[selected])),
                     scale_valid=int(np.sum(selected & targets["sampling_scale_valid"])))
        for prefix, valid_name in (("boundary", "boundary_valid"), ("surface", "surface_valid"),
                                    ("sampling", "sampling_consistency_valid")):
            if valid_name not in targets:
                continue
            reason = targets[prefix + "_ignore_reason"]
            valid = selected & targets[valid_name]
            values = dict(valid=int(valid.sum()),
                          reasons={key: int(np.sum(selected & ((reason & (1 << i)) != 0)))
                                   for i, key in enumerate(REASONS[prefix])})
            if prefix == "boundary":
                values["non_saturated"] = int(np.sum(valid & (targets["boundary_distance"] < 1)))
            if prefix == "surface":
                values["negative_offset"] = int(np.sum(valid & (targets["surface_offset_z"] < 0)))
                values["plane_rejection"] = {key: int(np.sum(selected & ((targets["surface_plane_rejection"] & bit) != 0)))
                                             for key, bit in (("rmse", 1), ("slope", 2), ("xy_extent", 4))}
                values["seen_rejection"] = {key: int(np.sum(selected & ((targets["surface_seen_rejection"] & bit) != 0)))
                                            for key, bit in (("rmse", 1), ("xy_extent", 4))}
            entry[prefix] = values
        if dense_labels is not None:
            original_count = int(np.sum(dense_labels == label))
            entry["sampling"].update(original=original_count, retained=entry["total"],
                                      retention=entry["total"] / original_count if original_count else None,
                                      valid_among_retained=entry["sampling"]["valid"] / entry["total"] if entry["total"] else None)
        result[name] = entry
    return result


def _conditions(record, height):
    n, r = record["count"], record.get("range")
    conditions = []
    if n == 0:
        conditions.append("normal")
    if 1 <= n <= 4:
        conditions.append("one_to_four")
    if 5 <= n <= 19:
        conditions.append("five_to_nineteen")
    if n >= 100:
        conditions.append("dense")
    if n and 35 <= r <= 50:
        conditions.append("far")
    if n and height <= .2:
        conditions.append("low")
    if record["slots"] != 131072:
        conditions.append("special_slots")
        if n:
            conditions.append("special_anomaly")
    return conditions


def _aggregate(records):
    result = {}
    conditions = ["all", "train", "validation"] + sorted({c for r in records for c in r["conditions"]})
    for condition in conditions:
        chosen = [r for r in records if condition == "all" or condition == r["sample"]["split"] or condition in r["conditions"]]
        views = []
        for view in range(3):
            row = {}
            for label in ("normal", "anomaly"):
                parts = [r["views"][view][label] for r in chosen]
                counts = dict(total=sum(p["total"] for p in parts), distinct_positions=sum(p["distinct_positions"] for p in parts),
                              scale_valid=sum(p["scale_valid"] for p in parts))
                for target in ("boundary", "surface", "sampling"):
                    if target not in parts[0]:
                        continue
                    target_parts = [p[target] for p in parts]
                    counts[target] = {key: sum(p[key] for p in target_parts)
                                      for key in target_parts[0] if isinstance(target_parts[0][key], int)}
                    counts[target]["valid_fraction"] = counts[target]["valid"] / counts["total"] if counts["total"] else None
                    for key in ("reasons", "plane_rejection", "seen_rejection"):
                        if key in target_parts[0]:
                            counts[target][key] = {reason: sum(p[key][reason] for p in target_parts)
                                                   for reason in target_parts[0][key]}
                    if target == "sampling":
                        origin = counts[target]["original"]
                        counts[target]["retention"] = counts["total"] / origin if origin else None
                row[label] = counts
            views.append(row)
        result[condition] = dict(frames=len(chosen), views=views)
    return result


def _loss_probe(parts, region_weights):
    """Evaluate constants on saved targets, not a trained model or synthetic scores."""
    import torch

    labels = torch.from_numpy(np.concatenate([p["labels"] for p in parts]))
    distance = torch.from_numpy(np.concatenate([p["distance"] for p in parts]).astype(np.float64))
    valid = torch.from_numpy(np.concatenate([p["boundary_valid"] for p in parts]))
    weights = loss_point_weights(labels, valid, dtype=torch.float64,
                                 boundary_target=distance, region_weights=region_weights)
    d, w = distance[valid].numpy(), weights[valid].numpy()
    order = np.argsort(d, kind="stable")
    best = float(d[order[np.searchsorted(np.cumsum(w[order]), .5)]]) if len(d) else None
    one = torch.ones_like(distance)
    constant_one = float(boundary_loss(one, distance, labels, valid, region_weights))
    best_loss = float(boundary_loss(torch.full_like(distance, best), distance, labels, valid, region_weights)) if best is not None else None
    groups = {}
    for region, mask in (("band", valid & (distance < 1)), ("saturated", valid & (distance == 1))):
        groups[region] = {name: dict(points=int((mask & (labels == y)).sum()),
                                    weight=float(weights[mask & (labels == y)].sum()))
                          for y, name in ((0, "normal"), (1, "anomaly"))}
    report = dict(C1=dict(valid=len(d), non_saturated=int(np.sum(d < 1)),
                         original_constant_one_loss=float(np.mean(1 - d)) if len(d) else None,
                         original_constant_one_bound=float(np.mean(d < 1)) if len(d) else None,
                         original_optimal_constant=float(np.median(d)) if len(d) else None,
                         grouped_constant_one_loss=constant_one,
                         grouped_optimal_constant=best, grouped_optimal_constant_loss=best_loss,
                         groups=groups))
    if len(d):
        assert best < 1 and constant_one > best_loss
        # An extra copy of every saturated group changes counts, not its total weight.
        extra = valid & (distance == 1)
        torch.testing.assert_close(boundary_loss(
            torch.cat((one, one[extra])), torch.cat((distance, distance[extra])),
            torch.cat((labels, labels[extra])), torch.cat((valid, valid[extra])), region_weights),
            torch.tensor(constant_one, dtype=torch.float64), rtol=0, atol=1e-12)
    for key, mask_key in (("C2", "sampling_valid"), ("C3", "surface_valid")):
        if mask_key not in parts[0]:
            continue
        mask = torch.from_numpy(np.concatenate([p[mask_key] for p in parts]))
        w = loss_point_weights(labels, mask, dtype=torch.float64)
        counts = {name: int((mask & (labels == y)).sum()) for y, name in ((0, "normal"), (1, "anomaly"))}
        total = sum(counts.values())
        report[key] = dict(valid=counts, original_anomaly_weight=counts["anomaly"] / total if total else None,
                           grouped_weight={name: float(w[labels == y].sum()) for y, name in ((0, "normal"), (1, "anomaly"))})
    return report


def compare_pilot(protocol, data_root, workers):
    """Read the fixed pilot, compare one visible-support variable, and check losses."""
    import torch

    torch.set_num_threads(1)
    config = protocol["supervision"]
    plan = config["pilot"]["comparison"]
    baseline = Path(config["pilot"]["output"])
    output = Path(plan["output"])
    parameters = config["C3"]["parameters"]
    thresholds = plan["visible_support_points"]
    if thresholds != [20, 10] or parameters["minimum_reference_support_points"] != 20 or parameters["minimum_visible_support_points"] != 20:
        raise ValueError("this comparison holds reference/default minima at 20 and changes visible support only")
    previous = json.loads((baseline / "summary.json").read_text())
    if [r["sample"] for r in previous["records"]] != config["pilot"]["samples"]:
        raise ValueError("comparison must use exactly the already computed fixed pilot")
    old_parameters = dict(previous["parameters"]["surface"])
    old_parameters["minimum_reference_support_points"] = old_parameters.pop("minimum_distinct_support_points")
    old_parameters["minimum_visible_support_points"] = 20
    if parameters != old_parameters:
        raise ValueError("reference fitting or other geometric conditions changed")
    disk_before = host_disk()
    datasets = {split: FrozenDataset(protocol["supervision"]["pilot"]["dataset_directory"], data_root, split) for split in ("train", "validation")}
    records, loss_parts = [], [[], [], []]
    start = time.monotonic()
    for number, old_record in enumerate(previous["records"]):
        selection = old_record["sample"]
        dataset = datasets[selection["split"]]
        index = next(i for i, (p, _, frame) in enumerate(dataset.samples)
                     if p.parent.parent.name == selection["world"] and frame == selection["frame"])
        if hashlib.sha256(dataset.samples[index][0].read_bytes()).hexdigest() != old_record["binding"]["frozen_delta_sha256"]:
            raise ValueError("frozen base frame changed since the original pilot")
        sample, original = dataset[index], dataset.sequence[selection["frame"]]
        relative = Path(selection["split"]) / selection["world"] / f"{selection['frame']:06d}.npz"
        source_path = baseline / relative
        before = hashlib.sha256(source_path.read_bytes()).hexdigest()
        with np.load(source_path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        if arrays["source_identity"].item() != source_identity(original) or arrays["world_identity"].item() != sample.world_identity:
            raise ValueError("pilot auxiliaries identify a different normal source or frozen world")
        geometries, labels, pairs = [], [], []
        for view in range(3):
            slots = arrays[f"view_{view}/source_slot"]
            geometries.append(ScanGeometry(sample.source.xyzi[slots], slots, config["common"]["sampling_scale"], workers))
            labels.append(sample.anomaly_target[slots])
            if view == 0:
                np.testing.assert_array_equal(slots, sample.source.real_slots)
            else:
                keep = np.zeros(original.slot_count, bool)
                keep[slots] = True
                row = arrays[f"view_{view}/dense_row"]
                np.testing.assert_array_equal(slots, geometries[0].slots[row])
                pairs.append(dict(dense_row=row, ray_keep=keep))
            part = dict(labels=labels[-1], distance=arrays[f"view_{view}/boundary_distance"],
                        boundary_valid=arrays[f"view_{view}/boundary_valid"], surface_valid=arrays[f"view_{view}/surface_valid"])
            if view:
                part["sampling_valid"] = arrays[f"view_{view}/sampling_consistency_valid"]
            loss_parts[view].append(part)
        compared = surface_targets(original, sample, geometries, labels, pairs, parameters, thresholds)
        views, probes = [], []
        artifact = dict(format=np.asarray("stu-visible-support-comparison"),
                        source_identity=arrays["source_identity"], world_identity=arrays["world_identity"],
                        baseline_auxiliary_sha256=np.asarray(before), minimum_reference_support_points=np.asarray(20),
                        minimum_visible_support_points=np.asarray(10), formal_default=np.asarray(False))
        for view, (g, y, a, b) in enumerate(zip(geometries, labels, compared[20], compared[10])):
            for key, value in a.items():
                np.testing.assert_array_equal(value, arrays[f"view_{view}/{key}"])
            assert np.all(~a["surface_valid"] | b["surface_valid"])
            np.testing.assert_array_equal(a["surface_offset_z"][a["surface_valid"]], b["surface_offset_z"][a["surface_valid"]])
            restored = b["surface_valid"] & ~a["surface_valid"]
            candidates = a["surface_ignore_reason"] == 64
            row = {}
            for name, mask in (("normal", y == 0), ("anomaly", y == 1),
                               ("low_anomaly", (y == 1) & ("low" in old_record["conditions"]))):
                reason = b["surface_ignore_reason"]
                row[name] = dict(total=int(mask.sum()), valid_20=int(np.sum(mask & a["surface_valid"])),
                                 valid_10=int(np.sum(mask & b["surface_valid"])), recovered=int(np.sum(mask & restored)),
                                 recovered_distinct_positions=len(np.unique(g.inverse[mask & restored])),
                                 quantity_failed_20=int(np.sum(mask & candidates)),
                                 remaining_after_quantity_failure={key: int(np.sum(mask & candidates & ((reason & (1 << k)) != 0)))
                                                                  for k, key in enumerate(REASONS["surface"])},
                                 remaining_geometry_rejection={key: int(np.sum(mask & candidates & ((b["surface_seen_rejection"] & bit) != 0)))
                                                               for key, bit in (("rmse", 1), ("xy_extent", 4))})
            views.append(row)
            check_rows = list(np.flatnonzero((y == 1) & candidates))
            for mask in (restored, candidates & ~restored):
                normal_rows = np.flatnonzero((y == 0) & mask)
                if len(normal_rows):
                    check_rows.extend(normal_rows[[0, len(normal_rows) // 2, -1]])
            for point in np.unique(check_rows):
                probe = surface_probe(g.xyzi[point, :3].astype(np.float64), original, sample, g.slots,
                                      dict(parameters, minimum_visible_support_points=10))
                assert probe["valid"] == bool(b["surface_valid"][point]), (selection, view, int(g.slots[point]), probe)
                if probe["valid"]:
                    np.testing.assert_allclose(b["surface_offset_z"][point], probe["offset_z_m"], rtol=1e-6, atol=1e-6)
                else:
                    assert b["surface_ignore_reason"][point] & (1 << REASONS["surface"].index(probe["reason"]))
                probes.append(dict(view=view, source_slot=int(g.slots[point]), label=int(y[point]), **probe))
            artifact[f"view_{view}/source_slot"] = g.slots
            artifact[f"view_{view}/recovered_source_slot"] = g.slots[restored]
            artifact.update({f"view_{view}/{key}": value for key, value in b.items()})
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **artifact)
        with np.load(destination, allow_pickle=False) as saved:
            for key, value in artifact.items():
                np.testing.assert_array_equal(saved[key], value)
        assert before == hashlib.sha256(source_path.read_bytes()).hexdigest()
        record = dict(sample=selection, conditions=old_record["conditions"], views=views, probes=probes,
                      baseline_auxiliary_sha256=before, baseline_20_arrays_identical=True,
                      auxiliary_bytes=destination.stat().st_size)
        _atomic_json(destination.with_suffix(".json"), record)
        records.append(record)
        print(json.dumps(dict(sample=number, stage="visible_20_vs_10", seconds=round(time.monotonic() - start, 2),
                              recovered=[{name: counts["recovered"] for name, counts in view.items()} for view in views])), flush=True)
        host_disk()
    groups = {}
    for name in ("all", "train", "validation"):
        selected = [r for r in records if name == "all" or r["sample"]["split"] == name]
        groups[name] = []
        for view in range(3):
            totals = {}
            for label in ("normal", "anomaly", "low_anomaly"):
                parts = [r["views"][view][label] for r in selected]
                totals[label] = {key: sum(p[key] for p in parts) for key, value in parts[0].items() if isinstance(value, int)}
                for key in ("remaining_after_quantity_failure", "remaining_geometry_rejection"):
                    totals[label][key] = {reason: sum(p[key][reason] for p in parts) for reason in parts[0][key]}
            groups[name].append(totals)
    weights = config["loss"]["C1"]
    report = dict(plan=plan, loss_rules=config["loss"], reference_parameters=parameters,
                  scope="同一批 14 帧、42 个视图的损失计算性质及 C3 单变量对照；未训练。",
                  baseline="results/supervision/pilot/summary.json at 9a0ecb5", records=records, groups=groups,
                  loss_checks=[_loss_probe(parts, (weights["band_weight"], weights["saturated_weight"])) for parts in loss_parts],
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  counts="实际文件回波行；低矮异常是异常子集，不能与异常总数相加。三个视图分别统计；失效原因按首次失败条件记录。",
                  seconds=time.monotonic() - start, workers=workers,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  disk_before=disk_before, disk_after=host_disk())
    _atomic_json(output / "summary.json", report)
    print(json.dumps(dict(comparison=str(output / "summary.json"), frames=len(records))), flush=True)


def training_identity(protocol):
    config = protocol["supervision"]
    parameters = dict(seed=protocol["seed"], scale=config["common"]["sampling_scale"],
                      boundary=config["C1"]["parameters"], levels=config["C2"]["levels"],
                      sampling=config["C2"]["evidence_parameters"], surface=config["C3"]["parameters"])
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("supervision.py", "data.py", "scene.py", "render.py", "protocol.py")}
    rays = hashlib.sha256(Path(protocol["calibration"]["rays"]).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(dict(parameters=parameters, implementation=implementation, rays=rays),
                                    sort_keys=True).encode()).hexdigest()


def auxiliary_path(protocol, split, frozen_path):
    return Path(protocol["training"]["supervision_directory"]) / split / frozen_path.parent.parent.name / frozen_path.name


def auxiliary_binding(sample, original, frozen_path, identity):
    return dict(configuration=identity, world=sample.world_identity, source=source_identity(original),
                sequence=original.sequence_id, frame=original.frame_id,
                delta=hashlib.sha256(frozen_path.read_bytes()).hexdigest())


def load_training_auxiliary(path, sample, original, frozen_path, identity):
    """Verify the full scientific identity before exposing aligned training-only arrays."""
    with np.load(path, allow_pickle=False) as saved:
        if (saved["format"].item() != "stu-v1-training-supervision"
                or json.loads(saved["binding"].item()) != auxiliary_binding(sample, original, frozen_path, identity)):
            raise ValueError("auxiliary cache belongs to a different world, source or supervision definition")
        views = [{key.split("/", 1)[1]: saved[key] for key in saved.files if key.startswith(f"view_{v}/")}
                 for v in range(3)]
    if not np.array_equal(views[0]["source_slot"], sample.source.real_slots):
        raise ValueError("auxiliary base slots differ from the frozen current scan")
    for v, view in enumerate(views):
        slots = view["source_slot"]
        if v:
            rows = view["dense_row"]
            if (np.any(np.diff(rows) <= 0) or np.any(rows < 0) or np.any(rows >= len(views[0]["source_slot"]))
                    or not np.array_equal(slots, views[0]["source_slot"][rows])):
                raise ValueError("sparse-to-base source-slot correspondence is not exact")
        for name in ("boundary_distance", "boundary_valid", "surface_offset_z", "surface_valid"):
            if view[name].shape != slots.shape:
                raise ValueError("auxiliary target rows differ from view slots")
        for target, valid in (("boundary_distance", "boundary_valid"), ("surface_offset_z", "surface_valid")):
            if view[valid].dtype != np.bool_ or not np.isfinite(view[target][view[valid]]).all():
                raise ValueError("valid auxiliary supervision is not finite")
    return views


def prepare_training_sample(dataset, index, split, protocol, identity, threads, *, original=None, reuse=None, surface_cache=None):
    frozen_path, world_identity, frame = dataset.samples[index]
    original = dataset.sequence[frame] if original is None else original
    sample = FrozenFrame.load(frozen_path, original, world_identity)
    destination = auxiliary_path(protocol, split, frozen_path)
    if destination.exists():
        load_training_auxiliary(destination, sample, original, frozen_path, identity)
        return dict(bytes=destination.stat().st_size, reused_file=True, reused_calculation=False)
    definition = json.loads((frozen_path.parent.parent / "world.json").read_text())["world"]
    grid = calibrated_ray_grid(protocol["calibration"]["rays"])
    pairs = [thinning_pair(sample, original, grid, definition["seed"], level, protocol["seed"])
             for level in protocol["supervision"]["C2"]["levels"]]
    # Within this one original frame, identical full scans and phases have identical targets.
    # The saved identity still belongs to each individual world; no cross-source reuse occurs.
    key = hashlib.sha256()
    key.update((identity + source_identity(original)).encode())
    for array in (sample.source.xyzi, sample.anomaly_target, sample.inserted_mask, sample.occluded_original_mask):
        key.update(array.tobytes())
    key.update(json.dumps([pair["phase"] for pair in pairs], sort_keys=True).encode())
    key = key.hexdigest()
    reused = reuse is not None and key in reuse
    if reused:
        arrays = dict(reuse[key])
    else:
        geometries, labels, pairs, targets = compute_supervision(sample, original, grid, definition["seed"], protocol, threads,
                                                               surface_cache=surface_cache)
        arrays = {}
        for v, (geometry, target) in enumerate(zip(geometries, targets)):
            arrays[f"view_{v}/source_slot"] = geometry.slots
            for name in ("boundary_distance", "boundary_valid", "surface_offset_z", "surface_valid"):
                arrays[f"view_{v}/{name}"] = target[name]
            if v:
                for name in ("dense_row", "sampling_consistency_valid"):
                    arrays[f"view_{v}/{name}"] = target[name]
                arrays[f"view_{v}/phase"] = np.asarray(json.dumps(pairs[v-1]["phase"], sort_keys=True))
        if reuse is not None:
            reuse[key] = dict(arrays)
    arrays.update(format=np.asarray("stu-v1-training-supervision"),
                  binding=np.asarray(json.dumps(auxiliary_binding(sample, original, frozen_path, identity), sort_keys=True)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        with np.load(temporary, allow_pickle=False) as saved:
            for name, array in arrays.items():
                if not np.array_equal(saved[name], array):
                    raise ValueError("saved training supervision changed target values")
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(bytes=destination.stat().st_size, reused_file=False, reused_calculation=reused)


def _training_worker(protocol, data_root, threads):
    global _training_protocol, _training_datasets, _training_threads, _training_identity
    _training_protocol, _training_threads = protocol, threads
    _training_identity = training_identity(protocol)
    set_num_threads(threads)
    _training_datasets = {s: FrozenDataset(protocol["dataset"]["directory"], data_root, s) for s in ("train", "validation")}


def _prepare_source_frame(selection):
    split, frame = selection
    dataset = _training_datasets[split]
    original = dataset.sequence[frame]
    start, cpu = time.monotonic(), time.process_time()
    indices = [i for i, (_, _, f) in enumerate(dataset.samples) if f == frame]
    reused = {}
    pending = any(not auxiliary_path(_training_protocol, split, dataset.samples[i][0]).exists() for i in indices)
    cache = (normal_surface_cache(original, calibrated_ray_grid(_training_protocol["calibration"]["rays"]),
                                  _training_protocol, _training_threads) if pending else None)
    records = [prepare_training_sample(dataset, i, split, _training_protocol, _training_identity, _training_threads,
                                       original=original, reuse=reused, surface_cache=cache) for i in indices]
    return dict(split=split, frame=frame, samples=len(records), bytes=sum(r["bytes"] for r in records),
                reused_files=sum(r["reused_file"] for r in records), reused_calculations=sum(r["reused_calculation"] for r in records),
                seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def prepare_training_dataset(protocol, data_root, jobs, threads):
    identity = training_identity(protocol)
    output = Path(protocol["training"]["supervision_directory"])
    output.mkdir(parents=True, exist_ok=True)
    datasets = {s: FrozenDataset(protocol["dataset"]["directory"], data_root, s) for s in ("train", "validation")}
    disk = host_disk()
    stored = sum(p.stat().st_size for p in output.rglob("*.npz"))
    storage = protocol["training"]["storage"]
    if disk["SizeRemaining"] - (storage["peak_additional_bytes"] - stored) < disk["reserve_bytes"]:
        raise OSError("the remaining preparation, prediction and metric peak would enter the E: reserve")
    expected = {s: len(d) for s, d in datasets.items()}
    manifest_path = output / "manifest.json"
    manifest = dict(format="stu-v1-training-supervision", configuration=identity, dataset=protocol["dataset"]["directory"],
                    expected=expected, status="preparing", jobs=jobs, threads=threads, disk_before=disk)
    if manifest_path.exists() and json.loads(manifest_path.read_text())["configuration"] != identity:
        raise ValueError("the training cache directory belongs to a different supervision implementation")
    _atomic_json(manifest_path, manifest)
    schedule = [(s, f) for s, d in datasets.items() for f in d.sequence.frame_ids]
    start, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn"),
                             initializer=_training_worker, initargs=(protocol, data_root, threads)) as pool:
        futures = [pool.submit(_prepare_source_frame, selection) for selection in schedule]
        for future in as_completed(futures):
            records.append(future.result())
            completed = sum(r["samples"] for r in records)
            print(json.dumps(dict(event="supervision", completed=completed, total=sum(expected.values()),
                                  elapsed_seconds=round(time.monotonic()-start, 1), **records[-1])), flush=True)
            if len(records) % 10 == 0:
                current_disk = host_disk()
                if (sum(r["bytes"] for r in records) > storage["supervision_budget_bytes"]
                        or current_disk["SizeRemaining"] < current_disk["reserve_bytes"] + 3_000_000_000):
                    for pending in futures:
                        pending.cancel()
                    raise OSError("preparation stopped before exceeding its storage allocation")
    manifest.update(status="complete", completed={s: sum(r["samples"] for r in records if r["split"] == s) for s in expected},
                    seconds=time.monotonic()-start, bytes=sum(r["bytes"] for r in records),
                    reused_calculations=sum(r["reused_calculations"] for r in records),
                    cpu_seconds=sum(r["cpu_seconds"] for r in records),
                    maximum_worker_rss_bytes=max(r["peak_rss_bytes"] for r in records), disk_after=host_disk())
    if manifest["completed"] != expected:
        raise ValueError("supervision preparation missed final-list samples")
    _atomic_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("protocol/v1.json"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--workers", type=int, default=min(16, len(os.sched_getaffinity(0))))
    parser.add_argument("--compare", action="store_true", help="compare losses and visible support on the existing fixed pilot")
    parser.add_argument("--full", action="store_true", help="prepare training targets for every final-list scan")
    parser.add_argument("--jobs", type=int, default=1, help="process count for final-dataset preparation")
    args = parser.parse_args()
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        parser.error("workers must fit current CPU affinity")
    set_num_threads(args.workers)
    protocol = json.loads(args.protocol.read_text())
    if args.full:
        if args.jobs < 1 or args.jobs * args.workers > len(os.sched_getaffinity(0)):
            parser.error("jobs times workers must fit CPU affinity")
        prepare_training_dataset(protocol, args.data_root, args.jobs, args.workers)
        return
    if args.compare:
        compare_pilot(protocol, args.data_root, args.workers)
        return
    config = protocol["supervision"]
    output = Path(config["pilot"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    disk_before = host_disk()
    implementation = hashlib.sha256(b"".join(
        Path(__file__).with_name(name).read_bytes()
        for name in ("supervision.py", "data.py", "scene.py", "render.py", "protocol.py"))).hexdigest()
    parameters = dict(seed=protocol["seed"], scale=config["common"]["sampling_scale"],
                      boundary=config["C1"]["parameters"], levels=config["C2"]["levels"],
                      sampling=config["C2"]["evidence_parameters"], surface=config["C3"]["parameters"])
    parameters["scale"] = {k: v for k, v in parameters["scale"].items() if isinstance(v, (int, float))}
    parameter_identity = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    datasets = {split: FrozenDataset(protocol["supervision"]["pilot"]["dataset_directory"], args.data_root, split) for split in ("train", "validation")}
    grid = calibrated_ray_grid(protocol["calibration"]["rays"])
    rays_identity = hashlib.sha256(Path(protocol["calibration"]["rays"]).read_bytes()).hexdigest()
    records = []
    for number, selection in enumerate(config["pilot"]["samples"]):
        dataset = datasets[selection["split"]]
        index = next(i for i, (p, _, frame) in enumerate(dataset.samples)
                     if p.parent.parent.name == selection["world"] and frame == selection["frame"])
        path, world_identity, frame = dataset.samples[index]
        world_dir = path.parent.parent
        world = json.loads((world_dir / "world.json").read_text())["world"]
        manifest = json.loads((world_dir / "manifest.json").read_text())["frames"][frame]
        height = shape_geometry(shape_from_dict(world["objects"][0]["shape"]))["height_m"]
        directory = output / selection["split"] / selection["world"]
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / f"{frame:06d}.json"
        auxiliary_path = summary_path.with_suffix(".npz")
        start, cpu_start = time.monotonic(), time.process_time()
        sample, original = dataset[index], dataset.sequence[frame]
        binding = dict(implementation_sha256=implementation, parameters_sha256=parameter_identity,
                       frozen_delta_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), world_identity=world_identity,
                       source_identity=source_identity(original), rays_sha256=rays_identity)
        if summary_path.exists() and auxiliary_path.exists():
            previous = json.loads(summary_path.read_text())
            if previous["binding"] == binding and previous["sample"] == selection:
                records.append(previous)
                print(json.dumps(dict(sample=number, reused=True), ensure_ascii=False), flush=True)
                continue
        assert int(np.sum(sample.anomaly_target == 1)) == manifest["count"]
        def progress(stage):
            print(json.dumps(dict(sample=number, split=selection["split"], world=selection["world"], frame=frame,
                                  stage=stage, seconds=round(time.monotonic() - start, 2)), ensure_ascii=False), flush=True)
        geometries, labels, pairs, targets = compute_supervision(sample, original, grid, world["seed"], protocol, args.workers, progress)
        checks = audit_supervision(sample, original, geometries, labels, pairs, targets, config)
        assert binding["frozen_delta_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        arrays = dict(format=np.asarray("stu-single-scan-supervision"), source_identity=np.asarray(source_identity(original)),
                      world_identity=np.asarray(world_identity), parameters_sha256=np.asarray(parameter_identity),
                      implementation_sha256=np.asarray(implementation))
        for view, (g, target) in enumerate(zip(geometries, targets)):
            arrays[f"view_{view}/source_slot"] = g.slots
            arrays.update({f"view_{view}/{key}": value for key, value in target.items() if not key.startswith("_")})
        for view, pair in enumerate(pairs, 1):
            arrays[f"view_{view}/sampling_level_and_phase"] = np.asarray(json.dumps(pair["phase"], sort_keys=True))
        np.savez_compressed(auxiliary_path, **arrays)
        # Validate the saved artifact, including every row correspondence, before summarizing it.
        with np.load(auxiliary_path, allow_pickle=False) as saved:
            for key, array in arrays.items():
                np.testing.assert_array_equal(saved[key], array)
        record = dict(sample=selection, binding=binding, source_identity=source_identity(original),
                      source_partition=original.partition, source_sequence=original.sequence_id,
                      original_slot_count=original.slot_count, manifest=manifest, physical_height_m=height,
                      conditions=_conditions(manifest, height), phases=[p["phase"] for p in pairs],
                      views=[summarize_view(g, y, target, labels[0] if i else None)
                             for i, (g, y, target) in enumerate(zip(geometries, labels, targets))],
                      geometry_checks=checks, frozen_delta_unchanged=True,
                      seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu_start,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      auxiliary_bytes=auxiliary_path.stat().st_size)
        _atomic_json(summary_path, record)
        records.append(record)
        progress("saved_and_checked")
        host_disk()
        del sample, original, geometries, labels, pairs, targets, arrays
    report = dict(scope=config["pilot"]["scope"], selection=config["pilot"], implementation_sha256=implementation,
                  parameters=parameters, parameters_sha256=parameter_identity,
                  information_use=protocol["information_use"], workers=args.workers,
                  reason_encoding={target: {name: 1 << i for i, name in enumerate(names)} for target, names in REASONS.items()},
                  reason_interpretation="C1/C2 可多因并存；C3 记录计算顺序中首先失败的条件。平面质量位为 RMSE=1、坡度=2、水平展开=4。",
                  counts="以实际文件回波行计数；distinct_positions 另列不同坐标数。条件可重叠，不代表全池覆盖率。视图 0 为基础，1/2 为规定删减级别。",
                  disk_before=disk_before, disk_after=host_disk(), records=records, groups=_aggregate(records))
    _atomic_json(output / "summary.json", report)
    print(json.dumps(dict(summary=str(output / "summary.json"), frames=len(records),
                          auxiliary_bytes=sum(r["auxiliary_bytes"] for r in records)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
