"""Full public-val observation profile, without loading a model or predictions."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
import torch

from .data import FrozenSyntheticSegment, _atomic_json, load_pool_manifest
from .model import joint_voxelize
from .protocol import load_protocol
from .scene import STUSequence, LabelMode
from .train import host_disk


QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
COUNT_BINS = (0, 1, 5, 20, 100, 500, float("inf"))
DISTANCE_BINS = (0, 2.5, 10, 20, 35, 50, float("inf"))
FRACTION_BINS = (0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.000001)
LENGTH_BINS = (0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 50, float("inf"))
INTENSITY_BINS = (-float("inf"), 0, 0.05, 0.1, 0.2, 0.4, 0.6, 1, 2, float("inf"))
GROUND = (40, 44, 48, 49, 60)
STATIC = (40, 44, 48, 49, 50, 51, 60, 71, 72, 80, 81)
NAMES = ("normal", "anomaly", "ignore")
BIT_COUNTS = np.array([i.bit_count() for i in range(32)], np.uint8)


def categories(semantic):
    return np.where(semantic == 2, 1, np.where(semantic == 0, 2, 0))


class Ledger:
    """Exact weighted empirical counts; bounded bins for billion-member residuals."""

    def __init__(self):
        self.series = {}

    def add(
        self,
        factor,
        metric,
        values,
        *,
        scope="all_frames",
        group="all",
        unit="frame",
        bins=COUNT_BINS,
        resolution=0.0,
        categorical=False,
        summarize=False,
    ):
        key = "|".join((factor, metric, scope, group))
        item = self.series.setdefault(
            key,
            dict(
                factor=factor,
                metric=metric,
                scope=scope,
                group=group,
                unit=unit,
                bins=list(bins),
                resolution=resolution,
                categorical=categorical,
                counts=Counter(),
                frame_weights=Counter(),
                n=0,
                denominator=0,
                frames=0,
                valid_frames=0,
                total=0.0,
                frame_total=0.0,
                minimum=None,
                maximum=None,
                bin_counts=np.zeros(len(bins) - 1, np.int64),
                bin_frame_weights=np.zeros(len(bins) - 1, np.float64),
            ),
        )
        values = np.asarray(values, np.float64).reshape(-1)
        item["frames"] += 1
        item["denominator"] += len(values)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        item["valid_frames"] += 1
        item["n"] += len(values)
        total = float(values.sum(dtype=np.float64))
        item["total"] += total
        item["frame_total"] += total / len(values)
        minimum, maximum = float(values.min()), float(values.max())
        item["minimum"] = (
            minimum if item["minimum"] is None else min(item["minimum"], minimum)
        )
        item["maximum"] = (
            maximum if item["maximum"] is None else max(item["maximum"], maximum)
        )
        # No model input is rounded. Fine bins bound only reported quantiles.
        encoded = (
            np.floor(values / resolution).astype(np.int64) if resolution else values
        )
        unique, counts = np.unique(encoded, return_counts=True)
        item["counts"].update(dict(zip(unique.tolist(), counts.tolist())))
        item["frame_weights"].update(
            dict(zip(unique.tolist(), (counts / len(values)).tolist()))
        )
        histogram = np.histogram(values, bins)[0]
        item["bin_counts"] += histogram
        item["bin_frame_weights"] += histogram / len(values)
        if summarize:
            selected = unique[
                np.searchsorted(
                    np.cumsum(counts), np.asarray(QUANTILES) * len(values), side="left"
                )
            ]
            return dict(
                n=len(values),
                mean=total / len(values),
                minimum=minimum,
                maximum=maximum,
                quantiles=dict(zip(map(str, QUANTILES), selected.tolist())),
            )

    def save(self, path):
        arrays, catalog = {}, []
        for i, (key, item) in enumerate(sorted(self.series.items())):
            values = np.array(sorted(item["counts"]), np.float64)
            arrays[f"x{i}"] = values
            arrays[f"c{i}"] = np.array([item["counts"][x] for x in values], np.int64)
            arrays[f"w{i}"] = np.array(
                [item["frame_weights"][x] for x in values], np.float64
            )
            meta = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in item.items()
                if k not in ("counts", "frame_weights")
            }
            catalog.append(dict(key=key, **meta))
        arrays["catalog"] = np.asarray(json.dumps(catalog, allow_nan=True))
        np.savez_compressed(path, **arrays)


def describe(meta, values, weights, *, mean=None, bin_weights=None):
    total = float(weights.sum())
    result = {
        k: v
        for k, v in meta.items()
        if k not in ("total", "frame_total", "bin_frame_weights")
    }
    result["missing_fraction"] = (
        (meta["denominator"] - meta["n"]) / meta["denominator"]
        if meta["denominator"]
        else None
    )
    result["empty_frame_fraction"] = (
        (meta["frames"] - meta["valid_frames"]) / meta["frames"]
        if meta["frames"]
        else None
    )
    result["status"] = "observed" if total else "no_reliable_values"
    result["mean"] = None if meta["categorical"] else mean
    result["quantiles"] = {}
    if total and not meta["categorical"]:
        indexes = np.searchsorted(
            np.cumsum(weights), np.array(QUANTILES) * total, side="left"
        )
        selected = values[np.minimum(indexes, len(values) - 1)]
        for q, x in zip(QUANTILES, selected, strict=True):
            if meta["resolution"]:
                low = max(float(x * meta["resolution"]), meta["minimum"])
                high = min(float((x + 1) * meta["resolution"]), meta["maximum"])
                result["quantiles"][str(q)] = dict(
                    value=(low + high) / 2, lower=low, upper=high
                )
            else:
                result["quantiles"][str(q)] = dict(
                    value=float(x), lower=float(x), upper=float(x)
                )
    if meta["categorical"]:
        result["categories"] = (
            {
                str(int(x)): float(w / total)
                for x, w in zip(values, weights, strict=True)
            }
            if total
            else {}
        )
    histogram = np.asarray(meta["bin_counts"] if bin_weights is None else bin_weights)
    result["bin_fraction"] = (
        (histogram / total).tolist() if total else [None] * len(histogram)
    )
    return result


def frame_geometry(source, ledger, sequence, frame):
    slots = source.real_slots
    xyzi = source.xyzi[slots]
    xyz, intensity = xyzi[:, :3], xyzi[:, 3]
    semantic, instance = source.labels.semantic[slots], source.labels.instance[slots]
    label = categories(semantic)
    distance = np.linalg.norm(xyz, axis=1)
    inside = (distance >= 2.5) & (distance <= 50)
    anomaly = label == 1
    a = int(anomaly.sum())
    p = int((anomaly & inside).sum())
    state = 0 if not a else 1 if not p else 2 if p < 5 else 3
    record = dict(
        sequence=sequence,
        frame=frame,
        slots=source.slot_count,
        visible=len(slots),
        zero_slots=source.slot_count - len(slots),
        normal=int((label == 0).sum()),
        ignore=int((label == 2).sum()),
        anomaly=a,
        anomaly_in_range=p,
        normal_in_range=int(((label == 0) & inside).sum()),
        state=state,
        anomaly_distance_median=float(np.median(distance[anomaly])) if a else None,
        anomaly_in_range_distance_median=float(np.median(distance[anomaly & inside]))
        if p
        else None,
        unknown_instance_points=int(np.count_nonzero(anomaly & (instance == 0))),
    )
    record["instance_count"] = (
        len(np.unique(instance[anomaly]))
        if not record["unknown_instance_points"]
        else None
    )
    for metric in ("slots", "visible", "zero_slots", "normal", "ignore"):
        ledger.add(
            "A02",
            metric,
            [record[metric]],
            bins=(0, 1000, 10000, 50000, 100000, 131073, float("inf")),
        )
    for name, numerator, denominator in (
        ("visible_fraction", len(slots), source.slot_count),
        ("zero_slot_fraction", source.slot_count - len(slots), source.slot_count),
        ("ignore_fraction", record["ignore"], len(slots)),
        ("normal_fraction", record["normal"], len(slots)),
        ("anomaly_fraction", a, len(slots)),
    ):
        record[name] = numerator / denominator
        ledger.add("A02", name, [record[name]], bins=FRACTION_BINS)
    ledger.add(
        "A03", "current_state", [state], categorical=True, bins=np.arange(-0.5, 4.5)
    )
    for factor, metric, value in (
        ("A03", "current_unseen", state == 0),
        ("A04", "outside_only", state == 1),
        ("A05", "one_to_four", state == 2),
        ("A05", "eligible", state == 3),
    ):
        ledger.add(factor, metric, [value], categorical=True, bins=(-0.5, 0.5, 1.5))
        ledger.add(
            factor,
            metric,
            [value],
            scope="complete_frames" if frame >= 4 else "startup_frames",
            categorical=True,
            bins=(-0.5, 0.5, 1.5),
        )
    for metric in ("anomaly", "anomaly_in_range"):
        ledger.add("B01", metric, [record[metric]])
    for metric in ("anomaly_distance_median", "anomaly_in_range_distance_median"):
        ledger.add(
            "B02",
            metric,
            [np.nan if record[metric] is None else record[metric]],
            bins=DISTANCE_BINS,
        )
    ledger.add(
        "B05",
        "visible_instance_count",
        [record["instance_count"] if record["instance_count"] is not None else np.nan],
    )
    ledger.add(
        "D01",
        "raw_semantic",
        semantic,
        unit="point",
        categorical=True,
        bins=np.arange(-0.5, 260.5),
    )
    ledger.add(
        "D01",
        "normal_semantic",
        semantic[label == 0],
        unit="point",
        categorical=True,
        bins=np.arange(-0.5, 260.5),
    )
    # Source-frame intensity is counted once, including ignored and out-of-range returns.
    radial = np.minimum(
        np.floor(distance.astype(np.float64) / 2.5).astype(np.int32), 20
    )
    radial[distance == 50] = 19
    for g, name in enumerate(NAMES):
        mask = label == g
        record[f"{name}_intensity"] = ledger.add(
            "C01",
            "intensity",
            intensity[mask],
            group=name,
            unit="point",
            bins=INTENSITY_BINS,
            summarize=True,
        )
        raw = intensity[mask]
        grid = (np.rint(raw.astype(np.float64) * 3500) / 3500).astype(np.float32)
        ledger.add(
            "C02",
            "grid_match",
            raw == grid,
            group=name,
            unit="point",
            categorical=True,
            bins=(-0.5, 0.5, 1.5),
        )
        ledger.add(
            "C02",
            "zero",
            raw == 0,
            group=name,
            unit="point",
            categorical=True,
            bins=(-0.5, 0.5, 1.5),
        )
        for b in range(21):
            ledger.add(
                "C03",
                "intensity",
                intensity[mask & (radial == b)],
                group=f"{name}:r{b}",
                unit="point",
                bins=INTENSITY_BINS,
            )
    ax = xyz[anomaly].astype(np.float64)
    az = np.arctan2(ax[:, 1], ax[:, 0])
    el = np.arctan2(ax[:, 2], np.linalg.norm(ax[:, :2], axis=1))
    points = dict(
        sequence=np.full(
            a, sequence, dtype=np.int32 if isinstance(sequence, int) else None
        ),
        frame=np.full(a, frame, np.int32),
        slot=slots[anomaly],
        instance=instance[anomaly],
        distance=distance[anomaly],
        intensity=intensity[anomaly],
        inside=inside[anomaly],
    )
    for c, name in enumerate(("x", "y", "z")):
        points[name] = ax[:, c]
        ledger.add(
            "B04",
            name,
            ax[:, c],
            unit="anomaly_point",
            bins=(-float("inf"), -50, -20, -5, -1, 0, 1, 5, 20, 50, float("inf")),
        )
    for name, values in (("azimuth", az), ("elevation", el)):
        points[name] = values
        ledger.add(
            "B04",
            name,
            values,
            unit="anomaly_point",
            bins=np.linspace(-np.pi, np.pi, 25),
        )
    ledger.add(
        "B02", "distance", distance[anomaly], unit="anomaly_point", bins=DISTANCE_BINS
    )
    normal = label == 0
    tree = cKDTree(xyz[normal]) if a and normal.any() else None
    nearest = np.full(a, np.nan)
    contrast = np.full(a, np.nan)
    spread = np.full(a, np.nan)
    density = np.zeros(a, np.int32)
    nearest_class = np.full(a, -1, np.int32)
    neighborhood = []
    if tree is not None:
        nearest, index = tree.query(ax, k=1, workers=1)
        normal_intensity = intensity[normal]
        normal_semantic = semantic[normal]
        nearest_class = normal_semantic[index].astype(np.int32)
        neighborhood = tree.query_ball_point(ax, 0.5, workers=1)
        for i, ids in enumerate(neighborhood):
            density[i] = len(ids)
            if len(ids) >= 3:
                background = normal_intensity[ids].astype(np.float64)
                contrast[i] = float(points["intensity"][i]) - background.mean()
                spread[i] = background.std()
        union = (
            np.unique(np.concatenate([x for x in neighborhood if len(x)]))
            if any(len(x) for x in neighborhood)
            else np.empty(0, np.int64)
        )
        bg_sem = normal_semantic[union]
    else:
        bg_sem = np.empty(0, np.uint16)
    ledger.add(
        "C04",
        "intensity_contrast",
        contrast,
        unit="anomaly_point",
        bins=(-float("inf"), -0.5, -0.2, -0.05, 0, 0.05, 0.2, 0.5, float("inf")),
    )
    ledger.add(
        "C04",
        "background_intensity_std",
        spread,
        unit="anomaly_point",
        bins=INTENSITY_BINS,
    )
    ledger.add(
        "D01",
        "neighbor_semantic",
        bg_sem,
        unit="unique_neighbor_point_per_frame",
        categorical=True,
        bins=np.arange(-0.5, 260.5),
    )
    ledger.add(
        "D02",
        "nearest_normal_distance",
        nearest,
        unit="anomaly_point",
        bins=LENGTH_BINS,
    )
    ledger.add("D02", "normal_neighbors_r0.5", density, unit="anomaly_point")
    points.update(
        nearest_normal=nearest,
        nearest_normal_semantic=nearest_class,
        background_neighbors=density,
        intensity_contrast=contrast,
        background_intensity_std=spread,
    )
    record["neighbor_road_fraction"] = (
        float(np.mean(bg_sem == 40)) if len(bg_sem) else None
    )
    records = []
    nn = np.full(a, np.nan)
    neighbor_count = np.full(a, np.nan)
    ground_xyz = (
        xyz[np.isin(semantic, GROUND)].astype(np.float64) if a else np.empty((0, 3))
    )
    ground_tree = cKDTree(ground_xyz[:, :2]) if len(ground_xyz) else None
    for iid in np.unique(instance[anomaly & (instance > 0)]):
        chosen = points["instance"] == iid
        obj = ax[chosen]
        unique = np.unique(obj, axis=0)
        item = dict(
            sequence=sequence,
            frame=frame,
            instance=int(iid),
            points=len(obj),
            unique_points=len(unique),
            shape_status="reliable_visible_extent"
            if len(unique) >= 10
            else "fewer_than_10_distinct_returns",
        )
        metrics = {
            key: np.nan
            for key in (
                "length",
                "width",
                "height",
                "aspect",
                "linearity",
                "planarity",
                "scattering",
                "azimuth_span",
                "elevation_span",
                "azimuth_gap",
            )
        }
        if len(obj) >= 2:
            otree = cKDTree(obj)
            nn[chosen] = otree.query(obj, k=2, workers=1)[0][:, 1]
            neighbor_count[chosen] = (
                otree.query_ball_point(obj, 0.25, return_length=True, workers=1) - 1
            )
        if len(unique) >= 10:
            spans = np.ptp(obj, axis=0)
            metrics.update(
                length=float(max(spans[:2])),
                width=float(min(spans[:2])),
                height=float(spans[2]),
            )
            metrics["aspect"] = (
                metrics["length"] / metrics["width"] if metrics["width"] > 0 else np.nan
            )
            eigen = np.linalg.eigvalsh(np.cov(obj, rowvar=False))[::-1]
            if eigen[0] > 1e-12:
                metrics.update(
                    linearity=float((eigen[0] - eigen[1]) / eigen[0]),
                    planarity=float((eigen[1] - eigen[2]) / eigen[0]),
                    scattering=float(eigen[2] / eigen[0]),
                )
            angles = np.unique(np.mod(az[chosen], 2 * np.pi))
            gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
            metrics.update(
                azimuth_span=float(2 * np.pi - gaps.max()),
                elevation_span=float(np.ptp(el[chosen])),
                azimuth_gap=float(np.max(np.delete(gaps, np.argmax(gaps))))
                if len(gaps) > 1
                else np.nan,
            )
        item.update(
            {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}
        )
        ground = ground_relation(obj, ground_xyz, ground_tree)
        item.update(ground)
        records.append(item)
    # Normalize all instances in the same source frame together for frame weights.
    for factor, keys, edges in (
        ("B05", ("length", "width", "height", "aspect"), LENGTH_BINS),
        ("B06", ("linearity", "planarity", "scattering"), FRACTION_BINS),
        (
            "B07",
            ("azimuth_span", "elevation_span", "azimuth_gap"),
            (0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, np.pi, 2 * np.pi),
        ),
        (
            "D02",
            ("ground_height_p05", "ground_height_median"),
            (
                -float("inf"),
                -0.2,
                -0.05,
                0,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2,
                5,
                float("inf"),
            ),
        ),
    ):
        for key in keys:
            ledger.add(
                factor,
                key,
                [r[key] if r[key] is not None else np.nan for r in records],
                unit="instance_frame",
                bins=edges,
            )
    ledger.add(
        "B06",
        "same_instance_neighbor_distance",
        nn,
        unit="anomaly_point",
        bins=LENGTH_BINS,
    )
    ledger.add(
        "B06", "same_instance_neighbors_r0.25", neighbor_count, unit="anomaly_point"
    )
    points.update(
        same_instance_neighbor_distance=nn, same_instance_neighbors=neighbor_count
    )
    return record, records, points


def ground_relation(obj, ground, tree):
    result = dict(
        ground_status="no_ground_support",
        ground_height_p05=None,
        ground_height_median=None,
        ground_support=0,
        ground_rmse=None,
    )
    if tree is None:
        return result
    center = obj[:, :2].mean(axis=0)
    ids = tree.query_ball_point(center, 2.0)
    result["ground_support"] = len(ids)
    if len(ids) < 20:
        result["ground_status"] = "fewer_than_20_ground_returns"
        return result
    ground = ground[ids]
    xy = ground[:, :2] - center
    design = np.column_stack((xy, np.ones(len(xy))))
    coefficients = np.linalg.lstsq(design, ground[:, 2], rcond=None)[0]
    residual = ground[:, 2] - design @ coefficients
    mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    keep = np.abs(residual) <= max(0.05, 3 * mad)
    if keep.sum() < 20:
        result["ground_status"] = "insufficient_plane_inliers"
        return result
    coefficients = np.linalg.lstsq(design[keep], ground[keep, 2], rcond=None)[0]
    rmse = float(np.sqrt(np.mean((ground[keep, 2] - design[keep] @ coefficients) ** 2)))
    result["ground_rmse"] = rmse
    try:
        hull = ConvexHull(xy[keep])
        contained = bool(np.all(hull.equations[:, -1] <= 1e-10))
    except QhullError:
        contained = False
    if (
        rmse > 0.05
        or np.linalg.norm(coefficients[:2]) > np.tan(np.deg2rad(20))
        or np.linalg.eigvalsh(np.cov(xy[keep], rowvar=False))[0] < 0.01
        or not contained
    ):
        result["ground_status"] = "plane_fit_or_spatial_support_unreliable"
        return result
    height = obj[:, 2] - (
        np.column_stack((obj[:, :2] - center, np.ones(len(obj)))) @ coefficients
    )
    result.update(
        ground_status="local_plane_proxy",
        ground_height_p05=float(np.quantile(height, 0.05)),
        ground_height_median=float(np.median(height)),
    )
    return result


def window_geometry(window, inputs, frame_rows, ledger, sequence, frame):
    scope = "complete_windows" if frame >= 4 else "startup_windows"
    xyz = window.points.coordinates
    semantic = window.labels.semantic
    label = categories(semantic)
    scan = window.points.scan_group
    current = window.current_mask
    inverse = inputs.point_to_voxel.numpy()
    v = len(inputs.features)
    features = inputs.features.numpy()
    means = inputs.coordinates.numpy()
    counts = np.bincount(inverse, minlength=v)
    member = [np.bincount(inverse[label == g], minlength=v) for g in range(3)]
    current_counts = np.bincount(inverse[current], minlength=v)
    current_normal = np.bincount(inverse[current & (label == 0)], minlength=v)
    current_anomaly = np.bincount(inverse[current & (label == 1)], minlength=v)
    ca = current_anomaly > 0
    ja = member[1] > 0
    cv = current_counts > 0
    mix = (member[0] > 0).astype(np.uint8) + 2 * (member[2] > 0)
    hits = (features[:, 4:] @ np.array((16, 8, 4, 2, 1), np.float32)).astype(np.uint8)
    anomaly_hits = np.zeros(v, np.uint8)
    np.bitwise_or.at(
        anomaly_hits,
        inverse[label == 1],
        (1 << (4 - scan[label == 1])).astype(np.uint8),
    )
    new_normal = ca & (current_normal == 0) & (member[0] > 0)
    raw = [r["anomaly"] for r in frame_rows[-5:]]
    official = [r["anomaly_in_range"] for r in frame_rows[-5:]]
    padded = [None] * (5 - len(raw)) + raw
    official_padded = [None] * (5 - len(raw)) + official
    pattern = "".join("-" if x is None else str(int(x > 0)) for x in padded)
    history = sum(raw[:-1])
    support = sum(x > 0 for x in raw[:-1])
    rec = dict(
        sequence=sequence,
        frame=frame,
        scope=scope,
        anomaly_vector=padded,
        anomaly_in_range_vector=official_padded,
        visibility_pattern=pattern,
        all_unseen=sum(raw) == 0,
        history_anomaly=history,
        history_visible_scans=support,
        history_anomaly_fraction=history / sum(raw) if sum(raw) else None,
        voxels=v,
        current_voxels=int(cv.sum()),
        current_anomaly_voxels=int(ca.sum()),
        joint_anomaly_voxels=int(ja.sum()),
        joint_anomaly_voxel_increment=int(ja.sum() - ca.sum()),
        current_anomaly_compression=raw[-1] / ca.sum() if ca.any() else None,
        joint_anomaly_compression=sum(raw) / ja.sum() if ja.any() else None,
        current_anomaly_normal_mix_voxels=int((ca & (member[0] > 0)).sum()),
        current_anomaly_ignore_mix_voxels=int((ca & (member[2] > 0)).sum()),
        history_new_normal_voxels=int(new_normal.sum()),
        history_new_normal_points=int(current_anomaly[new_normal].sum()),
    )
    for key in ("history_anomaly", "history_visible_scans"):
        ledger.add(
            "E01",
            key,
            [rec[key]],
            scope=scope,
            bins=COUNT_BINS if key == "history_anomaly" else np.arange(-0.5, 5.5),
        )
    ledger.add(
        "E01",
        "history_anomaly_fraction",
        [
            rec["history_anomaly_fraction"]
            if rec["history_anomaly_fraction"] is not None
            else np.nan
        ],
        scope=scope,
        bins=FRACTION_BINS,
    )
    ledger.add(
        "A06",
        "whole_window_unseen",
        [sum(raw) == 0],
        scope=scope,
        categorical=True,
        bins=(-0.5, 0.5, 1.5),
    )
    if frame >= 4:
        ledger.add(
            "E02",
            "visibility_pattern",
            [int(pattern, 2)],
            scope=scope,
            categorical=True,
            bins=np.arange(-0.5, 32.5),
        )
    transform = (
        window.current_pose.current_from_world @ window.frames[0].source.lidar_pose
    )
    rec["translation_m"] = float(np.linalg.norm(transform[:3, 3])) if frame else 0.0
    rec["rotation_rad"] = (
        float(np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1)))
        if frame
        else 0.0
    )
    ledger.add(
        "E04", "translation", [rec["translation_m"]], scope=scope, bins=LENGTH_BINS
    )
    ledger.add(
        "E04",
        "rotation",
        [rec["rotation_rad"]],
        scope=scope,
        bins=(0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, np.pi),
    )
    for key in (
        "current_anomaly_voxels",
        "joint_anomaly_voxels",
        "joint_anomaly_voxel_increment",
    ):
        ledger.add("F01", key, [rec[key]], scope=scope)
    for key in ("current_anomaly_compression", "joint_anomaly_compression"):
        ledger.add(
            "F01",
            key,
            [rec[key] if rec[key] is not None else np.nan],
            scope=scope,
            bins=(1, 1.01, 1.1, 1.5, 2, 3, 5, 10, 50, float("inf")),
        )
    ledger.add(
        "F02",
        "all_voxel_label_presence",
        (member[0] > 0) + 2 * (member[1] > 0) + 4 * (member[2] > 0),
        scope=scope,
        unit="voxel",
        categorical=True,
        bins=np.arange(-0.5, 8.5),
    )
    ledger.add(
        "F02",
        "anomaly_voxel_mix",
        mix[ja],
        scope=scope,
        unit="anomaly_voxel",
        categorical=True,
        bins=np.arange(-0.5, 4.5),
    )
    cur_anomaly_inverse = inverse[current & (label == 1)]
    ledger.add(
        "F02",
        "current_anomaly_point_mix",
        mix[cur_anomaly_inverse],
        scope=scope,
        unit="current_anomaly_point",
        categorical=True,
        bins=np.arange(-0.5, 4.5),
    )
    for g, name in enumerate(NAMES):
        ledger.add(
            "F02",
            "member_fraction",
            member[g][ja] / counts[ja],
            scope=scope,
            group=name,
            unit="anomaly_voxel",
            bins=FRACTION_BINS,
        )
    ledger.add(
        "F03",
        "new_normal",
        new_normal[ca],
        scope=scope,
        unit="current_anomaly_voxel",
        categorical=True,
        bins=(-0.5, 0.5, 1.5),
    )
    ledger.add(
        "F03",
        "new_normal",
        new_normal[cur_anomaly_inverse],
        scope=scope,
        unit="current_anomaly_point",
        group="point_weighted",
        categorical=True,
        bins=(-0.5, 0.5, 1.5),
    )
    for s in range(4):
        h = np.bincount(inverse[(scan == s) & (label == 0)], minlength=v) > 0
        ledger.add(
            "F03",
            f"normal_from_scan_{s}",
            h[new_normal],
            scope=scope,
            unit="newly_mixed_anomaly_voxel",
            categorical=True,
            bins=(-0.5, 0.5, 1.5),
        )
    ledger.add(
        "F04",
        "scan_hits",
        hits,
        scope=scope,
        unit="voxel",
        categorical=True,
        bins=np.arange(-0.5, 32.5),
    )
    ledger.add(
        "F04",
        "scan_hits",
        hits[ca],
        scope=scope,
        group="current_anomaly",
        unit="voxel",
        categorical=True,
        bins=np.arange(-0.5, 32.5),
    )
    ledger.add(
        "F04",
        "anomaly_history_hits",
        BIT_COUNTS[anomaly_hits[ca] & 30],
        scope=scope,
        unit="current_anomaly_voxel",
        categorical=True,
        bins=np.arange(-0.5, 5.5),
    )
    cur = np.flatnonzero(current)
    ci = inverse[cur]
    current_means = np.column_stack(
        [
            np.bincount(
                ci,
                weights=xyz[cur, c] if c < 3 else window.points.features[cur, 0],
                minlength=v,
            )[cv]
            / current_counts[cv]
            for c in range(4)
        ]
    ).astype(np.float32)
    shift = np.linalg.norm(means[cv] - current_means[:, :3], axis=1)
    intensity_shift = features[cv, 3] - current_means[:, 3]
    residual = np.linalg.norm(xyz[cur] - means[ci], axis=1)
    for group, selected in (
        ("all_current", np.ones(cv.sum(), bool)),
        ("current_anomaly", ca[cv]),
    ):
        ledger.add(
            "F05",
            "mean_displacement",
            shift[selected],
            scope=scope,
            group=group,
            unit="current_voxel",
            bins=LENGTH_BINS,
            resolution=0.0001,
        )
        ledger.add(
            "F05",
            "mean_intensity_shift",
            intensity_shift[selected],
            scope=scope,
            group=group,
            unit="current_voxel",
            bins=(-float("inf"), -0.2, -0.05, -0.01, 0, 0.01, 0.05, 0.2, float("inf")),
            resolution=0.0001,
        )
    for g, name in enumerate(NAMES):
        ledger.add(
            "F05",
            "point_residual",
            residual[label[cur] == g],
            scope=scope,
            group=name,
            unit="current_point",
            bins=LENGTH_BINS,
            resolution=0.0001,
        )
    rec["anomaly_mean_displacement_median"] = (
        float(np.median(shift[ca[cv]])) if ca.any() else None
    )
    rec["normal_mix_fraction"] = (
        rec["current_anomaly_normal_mix_voxels"] / ca.sum() if ca.any() else None
    )
    rec["ignore_mix_fraction"] = (
        rec["current_anomaly_ignore_mix_voxels"] / ca.sum() if ca.any() else None
    )
    rec["new_normal_fraction"] = (
        rec["history_new_normal_voxels"] / ca.sum() if ca.any() else None
    )
    sampled, matched, distances = static_overlap(
        xyz, semantic, current, window.spec.sequence_id, window.current_frame_id
    )
    rec.update(
        static_sampled=sampled,
        static_matched=matched,
        static_match_fraction=matched / sampled if sampled else None,
    )
    ledger.add(
        "E05",
        "sampled_static_distance",
        distances,
        scope=scope,
        unit="sampled_current_static_point",
        bins=(0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, float("inf")),
        resolution=0.0001,
    )
    ledger.add(
        "E05",
        "static_match_fraction",
        [rec["static_match_fraction"] if sampled else np.nan],
        scope=scope,
        bins=FRACTION_BINS,
    )
    # Save anomaly-specific measurements only; all background members were counted.
    detail = dict(
        voxel_normal_fraction=member[0][cur_anomaly_inverse]
        / counts[cur_anomaly_inverse],
        voxel_anomaly_fraction=member[1][cur_anomaly_inverse]
        / counts[cur_anomaly_inverse],
        voxel_ignore_fraction=member[2][cur_anomaly_inverse]
        / counts[cur_anomaly_inverse],
        voxel_mix=mix[cur_anomaly_inverse],
        voxel_hits=hits[cur_anomaly_inverse],
        history_new_normal=new_normal[cur_anomaly_inverse],
        point_residual=residual[label[cur] == 1],
    )
    return rec, detail


def static_overlap(xyz, semantic, current, sequence, frame):
    ids = np.flatnonzero(current & np.isin(semantic, STATIC))
    if not frame or not len(ids):
        return 0, 0, np.empty(0)
    if len(ids) > 2048:
        ids = np.sort(
            np.random.default_rng(
                np.random.SeedSequence([20260906, sequence, frame])
            ).choice(ids, 2048, replace=False)
        )
    distances = np.full(len(ids), np.nan)
    for raw in np.unique(semantic[ids]):
        chosen = semantic[ids] == raw
        history = (~current) & (semantic == raw)
        if history.any():
            found = cKDTree(xyz[history]).query(
                xyz[ids[chosen]], k=1, distance_upper_bound=0.2, workers=1
            )[0]
            found[~np.isfinite(found)] = np.nan
            distances[chosen] = found
    return len(ids), int(np.isfinite(distances).sum()), distances


def stages(frames, sequence):
    visible = np.array([r["anomaly"] > 0 for r in frames])
    starts = np.r_[0, np.flatnonzero(visible[1:] != visible[:-1]) + 1]
    ends = np.r_[starts[1:], len(frames)]
    output = []
    for start, end in zip(starts, ends, strict=True):
        kind = (
            "visible"
            if visible[start]
            else "entire_unseen"
            if start == 0 and end == len(frames)
            else "prefix"
            if start == 0
            else "tail"
            if end == len(frames)
            else "gap"
        )
        output.append(
            dict(
                sequence=sequence,
                start=int(start),
                end=int(end - 1),
                length=int(end - start),
                kind=kind,
                left_censored=bool(start == 0),
                right_censored=bool(end == len(frames)),
            )
        )
        for frame in range(start, end):
            frames[frame]["stage"] = kind
            frames[frame]["first_in_visible_run"] = bool(
                visible[start] and frame == start
            )
    return output


def profile_sequence(data_root, output, sequence, limit=None, segment_record=None):
    started = time.monotonic()
    cpu_started = time.process_time()
    torch.set_num_threads(1)
    directory = Path(output) / str(sequence)
    if (directory / "summary.json").exists():
        return json.loads((directory / "summary.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    synthetic = segment_record is not None
    source_id = segment_record["source_sequence_id"] if synthetic else sequence
    source = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train" if synthetic else "val",
        sequence_id=source_id,
        label_mode=LabelMode.REQUIRED,
    )
    source._cache_frames = 5
    segment = None
    if synthetic:
        segment = FrozenSyntheticSegment(
            protocol.path.parent / segment_record["file"],
            source,
            segment_record["file_sha256"],
        )
        for key in ("world_identity", "synthetic_sequence_id", "segment_index"):
            if segment.metadata[key] != segment_record[key]:
                raise ValueError(f"frozen world identity differs: {key}")
    ledger = Ledger()
    frames = []
    windows = []
    instances = []
    point_chunks = []
    frame_ids = segment.frame_ids if synthetic else tuple(range(source.frame_count))
    if limit is not None:
        frame_ids = frame_ids[:limit]
    count = len(frame_ids)
    for frame, source_frame in enumerate(frame_ids):
        raw = (
            segment.frame(source_frame)
            if synthetic
            else source.source_frame(source_frame)
        )
        record, objects, points = frame_geometry(raw, ledger, sequence, frame)
        if synthetic:
            # Local indices define world boundaries; source IDs preserve raw observation identity.
            for row in (record, *objects):
                row.update(
                    source_frame=source_frame,
                    source_sequence=source_id,
                    version=segment_record["synthetic_sequence_id"],
                    segment=segment_record["segment_index"],
                )
        frames.append(record)
        instances.extend(objects)
        if not synthetic or frame >= 4:
            window = (
                segment.window(source_frame - 4)
                if synthetic
                else source.for_output(frame)
            )
            inputs = joint_voxelize(window)
            record, detail = window_geometry(
                window, inputs, frames, ledger, sequence, frame
            )
            if synthetic:
                record.update(
                    source_frame=source_frame,
                    source_sequence=source_id,
                    version=segment_record["synthetic_sequence_id"],
                    segment=segment_record["segment_index"],
                    source_frames=list(window.frame_ids),
                )
            windows.append(record)
            points.update(detail)
            del window, inputs, detail
        else:
            # Initial context is observed once, but has no legal synthetic output window.
            for key in (
                "voxel_normal_fraction",
                "voxel_anomaly_fraction",
                "voxel_ignore_fraction",
                "voxel_mix",
                "voxel_hits",
                "history_new_normal",
                "point_residual",
            ):
                points[key] = np.full(len(points["frame"]), np.nan)
        if len(points["frame"]):
            point_chunks.append(points)
        del points, raw
        if (frame + 1) % 100 == 0:
            print(
                json.dumps(
                    dict(
                        event="sequence_progress",
                        sequence=sequence,
                        frames=frame + 1,
                        total=count,
                    )
                ),
                flush=True,
            )
    episodes = stages(frames, sequence)
    if synthetic:
        for row in episodes:
            row.update(
                source_start=frame_ids[row["start"]], source_end=frame_ids[row["end"]]
            )
    for kind in ("visible", "prefix", "gap", "tail", "entire_unseen"):
        ledger.add(
            "E03",
            "stage_length",
            [x["length"] for x in episodes if x["kind"] == kind],
            scope="sequence_runs",
            group=kind,
            unit="observed_run",
            bins=(0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf")),
        )
    stage_codes = {
        name: i
        for i, name in enumerate(("prefix", "visible", "gap", "tail", "entire_unseen"))
    }
    for row in frames:
        ledger.add(
            "E03",
            "stage",
            [stage_codes[row["stage"]]],
            categorical=True,
            bins=np.arange(-0.5, 5.5),
        )
        ledger.add(
            "E03",
            "first_in_visible_run",
            [row["first_in_visible_run"]],
            categorical=True,
            bins=(-0.5, 0.5, 1.5),
        )
    for name, rows in (
        ("frames", frames),
        ("windows", windows),
        ("instances", instances),
        ("stages", episodes),
    ):
        with (directory / f"{name}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    if point_chunks:
        np.savez_compressed(
            directory / "anomaly.npz",
            **{
                key: np.concatenate([p[key] for p in point_chunks])
                for key in point_chunks[0]
            },
        )
    ledger.save(directory / "histograms.npz")
    states = np.bincount([r["state"] for r in frames], minlength=4)
    result = dict(
        sequence=sequence,
        frames=count,
        first_frame=frame_ids[0],
        last_frame=frame_ids[-1],
        complete_windows=max(0, count - 4),
        startup_windows=0 if synthetic else min(4, count),
        states=states.tolist(),
        slots=sum(r["slots"] for r in frames),
        visible=sum(r["visible"] for r in frames),
        ignore=sum(r["ignore"] for r in frames),
        normal=sum(r["normal"] for r in frames),
        anomaly=sum(r["anomaly"] for r in frames),
        anomaly_in_range=sum(r["anomaly_in_range"] for r in frames),
        unknown_instance_points=sum(r["unknown_instance_points"] for r in frames),
        instance_frame_count=len(instances),
        labelled_instance_count=len({r["instance"] for r in instances}),
        whole_window_unseen=sum(
            r["all_unseen"] for r in windows if r["scope"] == "complete_windows"
        ),
        timestamp_available=False,
        wall_seconds=time.monotonic() - started,
        cpu_seconds=time.process_time() - cpu_started,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    if synthetic:
        result.update(
            world=segment_record["world_identity"],
            version=segment_record["synthetic_sequence_id"],
            source_sequence=source_id,
            segment=segment_record["segment_index"],
            context_frames=min(4, count),
        )
    _atomic_json(directory / "summary.json", result)
    return result


def profile_pools(args, disk):
    """Profile frozen worlds independently and reuse the completed real observation profile."""
    protocol = load_protocol()
    real_spec = json.loads((args.real_profile / "spec.json").read_text())
    if not (args.real_profile / "summary.json").is_file():
        raise ValueError(
            "completed real profile is required; raw val is never opened here"
        )
    jobs = []
    for pool in (protocol.training_pool, protocol.validation_pool):
        manifest = load_pool_manifest(protocol, pool)
        directory = args.output / pool.name
        directory.mkdir(parents=True, exist_ok=True)
        records = {
            f"{r['synthetic_sequence_index']:03d}_{r['segment_index']:02d}": dict(
                r, source_sequence_id=pool.source_sequence_id
            )
            for r in manifest["segments"]
        }
        spec = dict(real_spec)
        spec.update(
            population=pool.name,
            sequences=list(records),
            records=records,
            frames=sum(
                r["frame_range_inclusive"][1] - r["frame_range_inclusive"][0] + 1
                for r in records.values()
            ),
            complete_windows=pool.total_window_count,
            source="frozen sparse observations reconstructed on train/206 or train/201; no generation",
            source_sequence_id=pool.source_sequence_id,
            synthetic_versions=pool.synthetic_sequence_count,
            world_count=pool.world_count,
            manifest=str(protocol.pool_manifest_path(pool.name)),
            boundaries="each world independently; first four local frames are context only; no startup windows",
            entity_weighting="sequence_equal denotes equal worlds, not independent roads or versions",
            point_intensity="original float32 exact unique values; each synthetic frame once per world; versions share raw background",
            static_seed_identity="unchanged seed 20260906, raw source sequence ID, raw source frame ID",
            real_profile=str(args.real_profile),
            workers=args.workers,
            output_peak_budget_bytes=4 * 2**30,
            host_disk=disk,
        )
        path = directory / "spec.json"
        if path.exists():
            old = json.loads(path.read_text())
            if any(
                old[k] != spec[k] for k in spec if k not in ("workers", "host_disk")
            ):
                raise ValueError("saved synthetic profile definitions differ")
        else:
            _atomic_json(path, spec)
        jobs.extend((directory, key, record) for key, record in records.items())
    started = time.monotonic()
    pending = [job for job in jobs if not (job[0] / job[1] / "summary.json").exists()]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("fork")
    ) as pool:
        futures = {
            pool.submit(
                profile_sequence, args.data_root, directory, key, None, record
            ): (directory.name, key)
            for directory, key, record in pending
        }
        for future in as_completed(futures):
            row = future.result()
            print(
                json.dumps(
                    dict(event="world_completed", pool=futures[future][0], **row)
                ),
                flush=True,
            )
            volume = host_disk()
            if volume["SizeRemaining"] - 512 * 2**20 < volume["reserve_bytes"]:
                raise OSError("profile writes are approaching the host reserve")
    timing = dict(
        wall_seconds=time.monotonic() - started,
        scanned_worlds=len(pending),
        reused_worlds=len(jobs) - len(pending),
        host_disk=host_disk(),
    )
    print(json.dumps(dict(event="pool_scans_completed", **timing)), flush=True)
    if pending:
        _atomic_json(args.output / "execution.json", timing)
    from .profile_report import aggregate_profile, write_pool_tables, compare_profiles

    results = {}
    for name in ("train", "validation"):
        results[name] = aggregate_profile(args.output / name)
        write_pool_tables(args.output / name, results[name], args.tables / name)
    real = json.loads((args.real_profile / "summary.json").read_text())
    compare_profiles(results, real, args.output, args.tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--tables",
        type=Path,
        help="directory for the completed UTF-8 CSV tables",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="both frozen pools; reuse real profile"
    )
    parser.add_argument("--real-profile", type=Path, default=Path("runs/profile_v1"))
    parser.add_argument(
        "--pilot",
        type=int,
        help="first N frames of val/125, in a separate output directory",
    )
    args = parser.parse_args()
    args.output = args.output or Path(
        "runs/profile_pools" if args.synthetic else "runs/profile_v1"
    )
    args.tables = args.tables or Path("profiles" if args.synthetic else "profiles/real")
    torch.set_num_threads(1)
    disk = host_disk()
    args.output.mkdir(parents=True, exist_ok=True)
    if (
        disk["SizeRemaining"] - (4 if args.synthetic else 2) * 2**30
        < disk["reserve_bytes"]
    ):
        raise OSError("profile output budget would invade the E: reserve")
    if args.synthetic:
        if args.pilot:
            parser.error(
                "use a separate bounded world pilot before profiling both frozen pools"
            )
        profile_pools(args, disk)
        return
    if args.pilot:
        print(
            json.dumps(profile_sequence(args.data_root, args.output, 125, args.pilot)),
            flush=True,
        )
        return
    spec = dict(
        sequences=list(load_protocol().public_sequence_ids),
        frames=8659,
        complete_windows=8583,
        voxel_size_m=0.05,
        source="public val raw scans, labels and provided poses only; no model or prediction reads",
        neighborhoods=dict(
            background_radius_m=0.5,
            background_min_points=3,
            same_instance_radius_m=0.25,
            shape_min_distinct_points=10,
        ),
        ground=dict(
            semantics=GROUND,
            xy_radius_m=2,
            min_points=20,
            max_rmse_m=0.05,
            max_slope_degrees=20,
            min_xy_eigenvalue=0.01,
            centroid_inside_support=True,
        ),
        static_proxy=dict(
            semantics=STATIC,
            max_current_sample=2048,
            seed=20260906,
            match_radius_m=0.2,
            same_semantic=True,
            history="all arrived history in current coordinates",
        ),
        aggregation="inverse empirical CDF; observation, frame and sequence weights separately; 0.0001 fine bins only for massive residual/shift arrays with quantile bounds",
        point_intensity="original float32 exact unique values; all visible returns, no repeated source frames",
        radial_bins="2.5 m through 50 m inclusive; >50 separate; no clipping of intensity",
        workers=args.workers,
        output_peak_budget_bytes=2 * 2**30,
        host_disk=disk,
    )
    if (args.output / "spec.json").exists():
        previous = json.loads((args.output / "spec.json").read_text())
        if any(
            previous[k] != json.loads(json.dumps(spec[k]))
            for k in spec
            if k not in ("workers", "host_disk")
        ):
            raise ValueError("profile definitions differ from saved results")
    else:
        _atomic_json(args.output / "spec.json", spec)
    started = time.monotonic()
    reused = {
        seq
        for seq in spec["sequences"]
        if (args.output / str(seq) / "summary.json").exists()
    }
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("fork")
    ) as pool:
        futures = {
            pool.submit(profile_sequence, args.data_root, args.output, seq): seq
            for seq in spec["sequences"]
        }
        for future in as_completed(futures):
            row = future.result()
            event = (
                "sequence_reused" if row["sequence"] in reused else "sequence_completed"
            )
            print(json.dumps(dict(event=event, **row)), flush=True)
            host_disk()
    print(
        json.dumps(
            dict(
                event="profile_scan_stage_completed",
                wall_seconds=time.monotonic() - started,
                reused_sequences=len(reused),
                scanned_sequences=len(spec["sequences"]) - len(reused),
            )
        ),
        flush=True,
    )
    from .profile_report import aggregate_profile, write_tables, plot_profile

    result = aggregate_profile(args.output)
    write_tables(args.output, result, args.tables)
    plot_profile(args.output, result)
    print(
        json.dumps(dict(event="profile_completed", tables=str(args.tables))),
        flush=True,
    )


if __name__ == "__main__":
    main()
