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

from .data import _atomic_json, host_disk
from .protocol import load_protocol
from .scene import STUSequence, LabelMode


QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
COUNT_BINS = (0, 1, 5, 20, 100, 500, float("inf"))
DISTANCE_BINS = (0, 2.5, 10, 20, 35, 50, float("inf"))
FRACTION_BINS = (0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.000001)
LENGTH_BINS = (0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 50, float("inf"))
INTENSITY_BINS = (-float("inf"), 0, 0.05, 0.1, 0.2, 0.4, 0.6, 1, 2, float("inf"))
GROUND = (40, 44, 48, 49, 60)
NAMES = ("normal", "anomaly", "ignore")


def categories(semantic):
    return np.where(semantic == 2, 1, np.where(semantic == 0, 2, 0))


class Ledger:
    """Exact weighted empirical counts and bounded bins for scan geometry."""

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


def frame_geometry(source, ledger):
    if source.labels is None:
        raise ValueError(
            "geometry diagnostics require source semantic and instance labels"
        )
    sequence, frame = source.sequence_id, source.frame_id
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
        record[name] = numerator / denominator if denominator else None
        ledger.add(
            "A02", name, [record[name] if denominator else np.nan], bins=FRACTION_BINS
        )
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
        sequence=np.full(a, sequence, np.int32),
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
        if len(obj) >= 2:
            otree = cKDTree(obj)
            nn[chosen] = otree.query(obj, k=2, workers=1)[0][:, 1]
            neighbor_count[chosen] = (
                otree.query_ball_point(obj, 0.25, return_length=True, workers=1) - 1
            )
        metrics = visible_shape(obj)
        item.update(metrics)
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


def visible_shape(obj):
    """Describe the observed returns, never the unobserved solid object extent."""
    unique = np.unique(obj, axis=0)
    az = np.arctan2(obj[:, 1], obj[:, 0])
    el = np.arctan2(obj[:, 2], np.linalg.norm(obj[:, :2], axis=1))
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
        angles = np.unique(np.mod(az, 2 * np.pi))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
        metrics.update(
            azimuth_span=float(2 * np.pi - gaps.max()),
            elevation_span=float(np.ptp(el)),
            azimuth_gap=float(np.max(np.delete(gaps, np.argmax(gaps))))
            if len(gaps) > 1
            else np.nan,
        )
    return {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}


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


def profile_sequence(data_root, output, sequence, limit=None):
    """Read each current scan once; retain only single-scan observations."""
    started = time.monotonic()
    source = STUSequence.open(
        data_root,
        protocol=load_protocol(),
        partition="val",
        sequence_id=sequence,
        label_mode=LabelMode.REQUIRED,
    )
    directory = Path(output) / str(sequence)
    directory.mkdir(parents=True, exist_ok=True)
    frame_ids = source.frame_ids if limit is None else source.frame_ids[:limit]
    ledger, frames, instances, point_chunks = Ledger(), [], [], []
    for frame_id in frame_ids:
        record, objects, points = frame_geometry(source.source_frame(frame_id), ledger)
        frames.append(record)
        instances.extend(objects)
        point_chunks.append(points)
    for name, rows in (("frames", frames), ("instances", instances)):
        with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
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
    result = dict(
        format="stu-frame-profile",
        sequence=sequence,
        frames=len(frames),
        first_frame=frame_ids[0],
        last_frame=frame_ids[-1],
        states=np.bincount([r["state"] for r in frames], minlength=4).tolist(),
        **{
            key: sum(r[key] for r in frames)
            for key in (
                "slots",
                "visible",
                "zero_slots",
                "ignore",
                "normal",
                "anomaly",
                "anomaly_in_range",
                "unknown_instance_points",
            )
        },
        official_frames=sum(r["state"] == 3 for r in frames),
        official_anomaly_points=sum(
            r["anomaly_in_range"] for r in frames if r["state"] == 3
        ),
        official_normal_points=sum(
            r["normal_in_range"] for r in frames if r["state"] == 3
        ),
        instance_frame_count=len(instances),
        wall_seconds=time.monotonic() - started,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    _atomic_json(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/profile"))
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--sequence", type=int, action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    sequences = (
        tuple(args.sequence) if args.sequence else load_protocol().public_sequence_ids
    )
    if len(sequences) != len(set(sequences)):
        parser.error("duplicate sequences")
    for sequence in sequences:
        load_protocol().sequence("val", sequence)
    if (
        args.limit is not None
        or set(sequences) != set(load_protocol().public_sequence_ids)
    ) and args.output.resolve() == Path("results/profile").resolve():
        parser.error("subset profiles require a separate output directory")
    # Records and derived tables share one result root to keep their scope together.
    records, tables = args.output / "records", args.output / "tables"
    records.mkdir(parents=True, exist_ok=True)
    if not args.tables_only:
        disk = host_disk()
        # The compact distributions and anomaly-only records fit within this bound.
        if disk["SizeRemaining"] - 2 * 2**30 < disk["reserve_bytes"]:
            raise OSError("profile storage would invade the E: reserve")
        _atomic_json(
            records / "spec.json",
            dict(
                format="stu-frame-profile",
                sequences=list(sequences),
                limit=args.limit,
                source="current scans and labels from public STU validation",
                ground=dict(
                    semantics=GROUND,
                    xy_radius_m=2,
                    min_points=20,
                    max_rmse_m=0.05,
                    max_slope_degrees=20,
                    min_xy_eigenvalue=0.01,
                    centroid_inside_support=True,
                ),
            ),
        )
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(profile_sequence, args.data_root, records, seq, args.limit)
                for seq in sequences
            ]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
                host_disk()
    from .profile_report import aggregate_profile, write_tables

    result = aggregate_profile(records)
    write_tables(records, result, tables)
    print(json.dumps({"tables": str(tables), "totals": result["totals"]}), flush=True)


if __name__ == "__main__":
    main()
