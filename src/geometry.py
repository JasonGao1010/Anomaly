"""Compute scan geometry, frozen-data diagnostics and normal-only conditional probes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor
import csv
import ctypes
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import tempfile
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree, ConvexHull, QhullError

from .data import _atomic_json, host_disk
from .evaluate import evaluation_targets
from .profile import GEOMETRY_PARAMETERS, observed_geometry
from .probes import FEATURES, GEOMETRY, fit_reference, score_reference
from .protocol import load_protocol
from .render import canonical_ray_slots_for_source
from .scene import STUSequence, LabelMode


SEED, NORMAL_QUERIES = 20260911, 8192
FLAGS = ("valid", "condition_valid", "normal_valid", "normal_change_valid")
DTYPE = np.dtype([("source_slot", "i4"), ("frame", "i4"), ("target", "i1"),
                  ("semantic", "u2"), *[(k, "f4") for k in FEATURES],
                  *[(k, "?") for k in FLAGS]])
READER = {}
REFERENCE, THRESHOLDS = None, None
GROUP_DTYPE = np.dtype([("bits", "u4"), ("count", "i8"), ("positive", "i8")])
SAMPLE_DTYPE = np.dtype(DTYPE.descr + [("sequence", "i4"), ("weight", "f8")])


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
            raise ValueError("labels must be aligned frozen normal/anomaly labels")
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
    """Measure whether each observed label group retains local support after thinning."""
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


def surface_targets(original, sample, geometries, labels, pairs, parameters, visible_minimums=None):
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
    rows = np.flatnonzero((counts >= parameters["minimum_reference_support_points"]) & known.any(axis=1))
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
    return results


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


def summarize_view(geometry, labels, targets, dense_labels=None):
    result = dict(input_returns=len(labels), distinct_positions=len(geometry.xyz),
                  ignored_label_returns=int(np.sum(labels < 0)))
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



def reader(data_root, partition, sequence):
    key = (str(data_root), partition, sequence)
    if READER.get("key") != key:
        READER.clear()
        READER.update(key=key, source=STUSequence.open(
            data_root, protocol=load_protocol(), partition=partition,
            sequence_id=sequence, label_mode=LabelMode.REQUIRED))
    return READER["source"]


def extract_frame(job):
    data_root, output, partition, sequence, frame_id = job
    started = time.monotonic()
    source = reader(data_root, partition, sequence)[frame_id]
    semantic = source.labels.semantic
    target = evaluation_targets(source.xyzi[:, :3], semantic)
    eligible = int(np.count_nonzero(target == 1)) >= 5
    row = dict(partition=partition, sequence=sequence, frame=frame_id,
               actual_returns=len(source.real_slots), eligible=eligible,
               official_normal=int(np.count_nonzero(target == 0)),
               official_anomaly=int(np.count_nonzero(target == 1)), rows=0)
    if partition == "train" or eligible:
        slots = source.real_slots
        distance = np.linalg.norm(source.xyzi[slots, :3], axis=1)
        query = slots[(distance >= 2.5) & (distance <= 50)]
        if partition == "train" and len(query) > NORMAL_QUERIES:
            # Sampling precedes label filtering; every scan uses its own random stream.
            rng = np.random.default_rng(np.random.SeedSequence([SEED, sequence, frame_id]))
            query = np.sort(rng.choice(query, NORMAL_QUERIES, replace=False))
        geometry = observed_geometry(source.xyzi[slots], slots, query_slots=query)
        if partition == "train":
            mapping = load_protocol().semantic_class_map
            keep = np.array([mapping.get(int(x), 255) != 255 for x in semantic[query]])
            labels = np.zeros(int(keep.sum()), np.int8)
        else:
            keep = target[query] >= 0
            labels = target[query][keep].astype(np.int8)
        values = np.empty(int(keep.sum()), DTYPE)
        for name in (*FEATURES, *FLAGS, "source_slot"):
            values[name] = geometry[name][keep]
        values["frame"], values["target"] = frame_id, labels
        values["semantic"] = semantic[query][keep]
        path = Path(output) / "features" / partition / str(sequence) / f"{frame_id:06d}.npy"
        np.save(path, values, allow_pickle=False)
        row.update(rows=len(values), geometry_valid=int(values["valid"].sum()),
                   all_geometry_valid=int(np.isfinite(np.column_stack([values[k] for k in GEOMETRY])).all(axis=1).sum()),
                   bytes=path.stat().st_size)
        if partition == "val" and (int((labels == 1).sum()) != row["official_anomaly"]
                                    or int((labels == 0).sum()) != row["official_normal"]):
            raise ValueError("observable point identities do not match the official point set")
    row.update(seconds=time.monotonic()-started,
               max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return row


def extract(data_root, output, workers):
    output = Path(output)
    if (output / "summary.json").exists() and not (output / "features").exists():
        raise FileExistsError("Geometry caches were removed; use a new --output directory for explicit recomputation")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "features"
    if destination.exists():
        raise FileExistsError("feature extraction already exists; use the analysis stage to reuse it")
    protocol = load_protocol()
    sequences = [("train", 206), ("train", 201), *[("val", s) for s in protocol.public_sequence_ids]]
    jobs = []
    for partition, sequence in sequences:
        source = reader(data_root, partition, sequence)
        (destination / partition / str(sequence)).mkdir(parents=True)
        jobs.extend((str(data_root), str(output), partition, sequence, f) for f in source.frame_ids)
    READER.clear()
    disk = host_disk()
    # Persistent features plus exact score-count aggregates and report peaks.
    peak = 26_000_000_000
    if disk["SizeRemaining"] - peak < disk["reserve_bytes"]:
        raise OSError("the 26 GB peak analysis budget would invade the physical E: reserve")
    config = dict(format="stu-observed-geometry", data_root=str(Path(data_root).resolve()),
                  parameters=GEOMETRY_PARAMETERS, seed=SEED, normal_queries_per_frame=NORMAL_QUERIES,
                  normal_selection="uniform current-scan range-valid slots, before semantic filtering",
                  normal_labels="valid normal_semantic_class_map classes only",
                  val_scope="every official eligible frame; every official valid point; neighborhoods use full actual scan",
                  normal_fit="train/206", normal_transfer="train/201", evaluation="public development val19",
                  workers=workers, cpu_affinity=len(os.sched_getaffinity(0)), library_threads=1,
                  peak_write_budget_bytes=peak, initial_disk=disk,
                  implementation_pilot=dict(frames=[0,224,448], partition="train", sequence=206,
                                            label_mode="forbidden", workers_1_vs_4="all returned arrays exactly equal",
                                            single_thread_seconds=[1.0193,1.0448,.9602],
                                            four_thread_seconds=[.7588,.7548,.7421]))
    _atomic_json(output / "configuration.json", config)
    start, records, minimum_disk = time.monotonic(), [], disk["SizeRemaining"]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for row in pool.map(extract_frame, jobs, chunksize=4):
            records.append(row)
            if len(records) % 100 == 0 or len(records) == len(jobs):
                volume = host_disk()
                minimum_disk = min(minimum_disk, volume["SizeRemaining"])
                print(json.dumps(dict(stage="geometry", completed=len(records), total=len(jobs),
                                      seconds=round(time.monotonic()-start,2),
                                      disk_remaining=volume["SizeRemaining"])), flush=True)
    _atomic_json(output / "extraction.json", dict(
        frames=records, seconds=time.monotonic()-start, minimum_disk_remaining=minimum_disk,
        final_disk=host_disk(), bytes=sum(r.get("bytes",0) for r in records)))


def as_data(values):
    return {key: values[key] for key in values.dtype.names}


def score_groups(scores, labels):
    """Compress identical float32 scores exactly; never round into histogram bins."""
    scores = np.asarray(scores, np.float32)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("score groups require finite empirical scores")
    bits, inverse, count = np.unique(scores.view(np.uint32), return_inverse=True, return_counts=True)
    result = np.empty(len(bits), GROUP_DTYPE)
    result["bits"], result["count"] = bits, count
    result["positive"] = np.bincount(inverse[labels == 1], minlength=len(bits))
    return result


def merge_groups(parts):
    if not parts:
        return np.empty(0, GROUP_DTYPE)
    values = np.concatenate(parts)
    if not len(values):
        return values
    values = values[np.argsort(values["bits"],kind="stable")]
    starts = np.r_[0, np.flatnonzero(values["bits"][1:] != values["bits"][:-1])+1]
    result = np.empty(len(starts), GROUP_DTYPE)
    result["bits"] = values["bits"][starts]
    for name in ("count", "positive"):
        result[name] = np.add.reduceat(values[name], starts)
    return result


def subtract_groups(total, removed):
    result = total.copy()
    indices = np.searchsorted(result["bits"], removed["bits"])
    if not np.array_equal(result["bits"][indices], removed["bits"]):
        raise ValueError("sequence scores are absent from pooled score identities")
    for name in ("count", "positive"):
        result[name][indices] -= removed[name]
    if np.any(result["positive"] < 0) or np.any(result["count"] < result["positive"]):
        raise ValueError("negative sequence-subtraction count")
    return result[result["count"] > 0]


def group_metrics(groups):
    from .evaluate import metrics_from_groups
    positive = int(groups["positive"].sum())
    negative = int(groups["count"].sum()) - positive
    descending = groups[::-1]
    iterator = ((a["bits"], a["count"], a["positive"])
                for start in range(0,len(descending),1<<18)
                if len(a := descending[start:start+(1<<18)]))
    return metrics_from_groups(iterator, positive=positive, negative=negative)


def normal_threshold(groups):
    descending = groups[::-1]
    counts = np.cumsum(descending["count"])
    feasible = np.flatnonzero(counts <= .01 * int(groups["count"].sum()))
    return float(descending["bits"][feasible[-1]].view(np.float32)) if len(feasible) else None


def threshold_counts(groups, threshold):
    take = np.zeros(len(groups), bool) if threshold is None else groups["bits"].view(np.float32) >= threshold
    positive, total = int(groups["positive"].sum()), int(groups["count"].sum())
    tp, detected = int(groups["positive"][take].sum()), int(groups["count"][take].sum())
    return dict(normal=total-positive, anomaly=positive, tp=tp, fp=detected-tp,
                FPR=100*(detected-tp)/(total-positive) if total>positive else None,
                recall=100*tp/positive if positive else None)


def comparisons(scores, individual):
    for mode, control in (("range","A_range"),("direction","A_direct"),("sampling","A_sampling")):
        names = (control,"B_geometry","C_"+mode)
        finite = np.logical_and.reduce([np.isfinite(scores[k]) for k in names])
        yield mode, {k:scores[k] for k in names}, finite
    finite = np.isfinite(scores["B_geometry"]) & np.isfinite(scores["B_normalized"])
    yield "normalization", {k:scores[k] for k in ("B_geometry","B_normalized")}, finite
    for feature in GEOMETRY:
        for mode in ("range","direction","sampling"):
            names = ("B_geometry","C_"+mode)
            fields = {k:individual[k][feature] for k in names}
            finite = np.logical_and.reduce([np.isfinite(a) for a in fields.values()])
            yield f"feature/{feature}/{mode}", fields, finite


def merge_count_archives(chunks,destination,keys):
    """Stream one exact comparison at a time; inputs are bounded frame blocks."""
    with ExitStack() as stack:
        archives=[stack.enter_context(np.load(p,allow_pickle=False)) for p in chunks]
        archive=stack.enter_context(ZipFile(destination,"w",ZIP_DEFLATED,allowZip64=True))
        for key_index,key in enumerate(keys):
            total=np.empty(0,GROUP_DTYPE)
            for part in archives:
                total=merge_groups([total,part[str(key_index)]])
            with archive.open(str(key_index)+".npy","w",force_zip64=True) as member:
                np.lib.format.write_array(member,total,allow_pickle=False)
            del total
            ctypes.CDLL(None).malloc_trim(0)


def process_scores(job):
    output, partition, sequence = job
    output = Path(output)
    directory = output / "counts" / partition / str(sequence)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"summary.json").exists():
        if not (directory/"groups.npz").exists():
            raise ValueError("completed scoring is missing its exact counts")
        return dict(partition=partition,sequence=sequence,seconds=0.,max_rss_bytes=0,reused_complete=True)
    pending, coverage, confusion = defaultdict(list), defaultdict(lambda:np.zeros(4,np.int64)), defaultdict(lambda:np.zeros(3,np.int64))
    workspace=tempfile.TemporaryDirectory(prefix="counts_",dir=directory)
    chunks=[]
    candidates, samples, frames = [], [], []
    started = time.monotonic()
    paths = sorted((output/"features"/partition/str(sequence)).glob("*.npy"))
    for index, path in enumerate(paths):
        values = np.load(path,allow_pickle=False)
        scores, individual, cells = score_reference(as_data(values),REFERENCE)
        labels = values["target"]
        anomaly_count = int((labels==1).sum())
        strata = {"all":np.ones(len(values),bool)}
        range_bins = np.searchsorted([10,20,35],values["range"],side="right")
        for k in range(4):
            strata["range_"+str(k)] = range_bins==k
        if partition=="val":
            strata["returns_"+str(np.searchsorted([20,100,500],anomaly_count,side="right"))] = strata["all"]
        for cohort, methods, finite in comparisons(scores,individual):
            for group, mask in strata.items() if not cohort.startswith("feature/") else [("all",strata["all"])]:
                key = cohort+"|"+group
                coverage[key] += [int(np.sum(mask & (labels==0))),int(np.sum(mask & (labels==1))),
                                  int(np.sum(mask & finite & (labels==0))),int(np.sum(mask & finite & (labels==1)))]
            for method, score in methods.items():
                key = cohort+"|"+method
                pending[key].append(score_groups(score[finite],labels[finite]))
                threshold = None if THRESHOLDS is None else THRESHOLDS.get(key)
                detected = finite & (score>=threshold) if threshold is not None else np.zeros(len(values),bool)
                if not cohort.startswith("feature/"):
                    for semantic in np.unique(values["semantic"][finite & (labels==0)]):
                        subset = (values["semantic"]==semantic) & (labels==0) & finite
                        confusion[key+"|"+str(int(semantic))] += [int(subset.sum()),int((subset & detected).sum()),0]
                    if partition=="val":
                        frames.append(dict(sequence=sequence,frame=int(values["frame"][0]),cohort=cohort,method=method,
                                           normal=int(np.sum(finite & (labels==0))),anomaly=int(np.sum(finite & (labels==1))),
                                           fp=int(np.sum(detected & (labels==0))),tp=int(np.sum(detected & (labels==1)))))
                        if method in ("C_direction","C_sampling"):
                            normal=np.flatnonzero(detected & (labels==0))
                            if len(normal):
                                top=normal[np.argmax(score[normal])]
                                candidates.append(dict(sequence=sequence,frame=int(values["frame"][top]),cohort=cohort,
                                                       method=method,fp=len(normal),slot=int(values["source_slot"][top]),
                                                       score=float(score[top])))
        if partition=="val":
            normal=np.flatnonzero(labels==0)
            rng=np.random.default_rng(np.random.SeedSequence([SEED,sequence,int(values["frame"][0]),91]))
            take=np.sort(rng.choice(normal,min(2048,len(normal)),replace=False))
            selected=np.r_[np.flatnonzero(labels==1),take]
            sample=np.empty(len(selected),SAMPLE_DTYPE)
            for name in DTYPE.names:
                sample[name]=values[name][selected]
            sample["sequence"],sample["weight"]=sequence,1.
            sample["weight"][sample["target"]==0]=len(normal)/len(take) if len(take) else 0
            samples.append(sample)
        # Fixed-size blocks prevent 47 growing sequence distributions occupying RAM together.
        if (index+1)%8==0 or index+1==len(paths):
            keys=sorted(pending)
            chunk=Path(workspace.name)/f"{len(chunks):04d}.npz"
            with ZipFile(chunk,"w",ZIP_DEFLATED,allowZip64=True) as archive:
                for key_index,key in enumerate(keys):
                    group=merge_groups(pending.pop(key))
                    with archive.open(str(key_index)+".npy","w",force_zip64=True) as member:
                        np.lib.format.write_array(member,group,allow_pickle=False)
                del group
            chunks.append(chunk)
            pending.clear()
            ctypes.CDLL(None).malloc_trim(0)
        if (index+1)%25==0 or index+1==len(paths):
            print(json.dumps(dict(stage="score_frames",partition=partition,sequence=sequence,
                                  completed=index+1,total=len(paths),seconds=round(time.monotonic()-started,2))),flush=True)
    merge_count_archives(chunks,directory/"groups.npz",keys)
    workspace.cleanup()
    if samples:
        np.save(output/"samples"/f"{sequence}.npy",np.concatenate(samples),allow_pickle=False)
    totals=dict(partition=partition,sequence=sequence,keys=keys,frames=frames,
                coverage={k:v.tolist() for k,v in coverage.items()},
                confusion={k:v.tolist() for k,v in confusion.items()},candidates=candidates,
                seconds=time.monotonic()-started,
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    _atomic_json(directory/"summary.json",totals)
    return dict(partition=partition,sequence=sequence,seconds=totals["seconds"],max_rss_bytes=totals["max_rss_bytes"])


def load_counts(output,partition,sequence):
    directory=Path(output)/"counts"/partition/str(sequence)
    summary=json.loads((directory/"summary.json").read_text())
    with np.load(directory/"groups.npz") as data:
        groups={k:data[str(i)] for i,k in enumerate(summary["keys"])}
    return groups,summary


def csv_rows(path,rows):
    if not rows:
        return
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output,workers=6):
    global REFERENCE,THRESHOLDS
    output=Path(output)
    if not (output / "features/train/206").is_dir():
        raise FileNotFoundError("Geometry feature caches are unavailable; retained tables are not recomputed automatically")
    config=json.loads((output/"configuration.json").read_text())
    extraction=json.loads((output/"extraction.json").read_text())
    disk=host_disk()
    if disk["SizeRemaining"]-14_000_000_000<disk["reserve_bytes"]:
        raise OSError("score counts and temporary blocks would invade the physical E: reserve")
    if config["parameters"]!=GEOMETRY_PARAMETERS or config["normal_fit"]!="train/206":
        raise ValueError("feature definition or fitting source changed")
    start=time.monotonic()
    normal=np.concatenate([np.load(p,allow_pickle=False) for p in sorted((output/"features/train/206").glob("*.npy"))])
    if np.any(normal["target"]!=0) or len(np.unique(normal["frame"]))!=449:
        raise ValueError("reference requires all 449 train/206 frames and only valid normal labels")
    REFERENCE=fit_reference(as_data(normal))
    existing=output/"reference.json"
    if existing.exists() and json.loads(existing.read_text())!=REFERENCE["metadata"]:
        raise ValueError("normal reference changed; cached scoring cannot be reused")
    _atomic_json(existing,REFERENCE["metadata"])
    del normal
    THRESHOLDS=None
    process_scores((str(output),"train",206))
    fit_groups,_=load_counts(output,"train",206)
    THRESHOLDS={key:normal_threshold(group) for key,group in fit_groups.items()}
    _atomic_json(output/"thresholds.json",dict(source="train/206 fitted normal scores",normal_fpr_limit=.01,values=THRESHOLDS))
    from .coverage import save_geometry_reference
    save_geometry_reference(output, REFERENCE)
    del fit_groups
    gc.collect()
    # Release allocator-retained fit/count workspaces before creating reader workers.
    ctypes.CDLL(None).malloc_trim(0)
    (output/"samples").mkdir(exist_ok=True)
    jobs=[(str(output),"train",201),*[(str(output),"val",s) for s in load_protocol().public_sequence_ids]]
    # Fork shares immutable normal reference arrays; each worker holds one scan.
    resources=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("fork")) as pool:
        for row in pool.map(process_scores,jobs,chunksize=1):
            resources.append(row)
            disk=host_disk()
            print(json.dumps(dict(stage="scores",**row,disk_remaining=disk["SizeRemaining"])),flush=True)
    from .profile_report import sampling_increment, summarize_geometry
    summarize_geometry(output,extraction)
    increment=sampling_increment(output,REFERENCE)
    from .probes import descriptive_matches
    sample=np.concatenate([np.load(p,allow_pickle=False) for p in sorted((output/"samples").glob("*.npy"))])
    matched=descriptive_matches(as_data(sample),REFERENCE)
    _atomic_json(output/"matched.json",dict(
        scope="descriptive weighted sample of official eligible val frames; not official metrics",
        normal_sampling="up to 2048 uniform normal slots per frame with inverse sampling probability weights",
        anomaly_sampling="all official anomaly points",rows=matched))
    from .probes import subgroup_diagnostics
    csv_rows(output/"tables/subgroups.csv",subgroup_diagnostics(as_data(sample),REFERENCE,THRESHOLDS))
    del sample
    from .profile_report import geometry_cases, plot_geometry
    plot_geometry(output)
    geometry_cases(output,config["data_root"],REFERENCE)
    _atomic_json(output/"execution.json",dict(seconds=time.monotonic()-start,score_workers=workers,
                                              workers=resources,final_disk=host_disk(),
                                              sampling_increment={k:v for k,v in increment.items() if k!="rows"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--output", type=Path, default=Path("results/geometry"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--stage", choices=("extract", "analyze", "all"), default="all")
    args = parser.parse_args()
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        parser.error("workers exceed the current CPU affinity")
    if args.stage in ("extract", "all"):
        extract(args.data_root, args.output, args.workers)
    if args.stage in ("analyze", "all"):
        analyze(args.output, min(args.workers,6))


if __name__ == "__main__":
    main()
