"""Exact detection diagnostics and compact evaluation of a fixed paired experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time
from threading import local

import numpy as np

from .data import FrozenDataset, FrozenFrame, FramePrediction, _atomic_json, host_disk
from .evaluate import (APAttribution, evaluate_frames, evaluation_targets, exact_metrics,
                       official_frame, packed_scores, pooled_files)
from .protocol import load_protocol
from .scene import LabelMode, STUSequence
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator as Official


SCOPES = ("all", "stu_filtered")
LIMITS = (0.01, 0.001, 0.0001)
RANGES = ("<2.5", "[2.5,10)", "[10,20)", "[20,35)", "[35,50]", ">50")
FRAME_GROUPS = ("no_anomaly", "anomaly_not_eligible", "eligible")
REAL_READERS = local()
POINT_DTYPE = np.dtype([(name, kind) for name, kind in (
    ("sequence", "i4"), ("frame", "i4"), ("slot", "i4"), ("score", "f4"),
    ("precision", "f8"), ("q_global", "f8"), ("q_frame", "f8"), ("distance", "f4"))])
REAL_CASE_RULES = {
    "125_failure": "two eligible 125 frames with largest AP deficit; ties by frame ID",
    "125_success": "remaining eligible 125 frame with highest recall at global 1% FPR; ties by anomaly count descending, frame ID",
    "other_success": "eligible non-125 frame with highest recall at global 1% FPR; ties by anomaly count descending, sequence/frame ID",
    "normal_clusters": "two sequences with most FP at global 1% FPR; in each, densest one-metre cell of distinct FP positions; ties by frame ID then lexicographic cell; anchor highest score, then original slot",
    "display": "all anomaly bounds plus 2 m per axis for anomaly cases; 6 m radius with +/-1.2 m detail for normal cases; no geometry-based reselection",
}


def read_real(identity, data_root, run):
    sequence_id, frame_id = identity
    key = (str(Path(data_root).resolve()), sequence_id)
    # Readers own a mutable frame cache, so each loading thread has its own reader.
    if getattr(REAL_READERS, "key", None) != key:
        REAL_READERS.source = STUSequence.open(data_root, protocol=load_protocol(),
            partition="val", sequence_id=sequence_id, label_mode=LabelMode.REQUIRED)
        REAL_READERS.key = key
    source = REAL_READERS.source[frame_id]
    path = Path(run) / "predictions" / "real" / "val" / str(sequence_id) / f"{frame_id:06d}.npz"
    return source, FramePrediction.load(path, source)


def real_frames(identities, data_root, run, jobs):
    # Bound live complete scans; executor.map would eagerly retain thousands of scans.
    iterator = iter(identities)
    with ThreadPoolExecutor(jobs) as pool:
        pending = deque()
        for _ in range(jobs):
            identity = next(iterator, None)
            if identity is not None:
                pending.append(pool.submit(read_real, identity, data_root, run))
        while pending:
            yield pending.popleft().result()
            identity = next(iterator, None)
            if identity is not None:
                pending.append(pool.submit(read_real, identity, data_root, run))


def required_frame_fpr(normal_scores, anomaly_scores):
    """Empirical normal survival at each anomaly, including the entire score tie."""
    ordered = np.sort(normal_scores)
    if not len(ordered):
        raise ValueError("within-frame FPR is undefined without official normal points")
    return (len(ordered) - np.searchsorted(ordered, anomaly_scores, side="left")) / len(ordered)


def anomaly_summary(points, total_positive, thresholds):
    result = dict(anomaly_points=len(points),
        ap_deficit_pp=float(100 * np.sum(1 - points["precision"], dtype=np.float64) / total_positive))
    if not len(points):
        return result
    for name in ("q_global", "q_frame"):
        values = 100 * points[name]
        for key, value in zip(("min", "median", "p90", "p95", "max"), np.quantile(values, [0, .5, .9, .95, 1]), strict=True):
            result[f"{name}_{key}_percent"] = float(value)
        result[f"{name}_mean_percent"] = float(values.mean())
    high_global, high_frame = points["q_global"] > .01, points["q_frame"] > .01
    for name, mask in (("both_q_above_1pct", high_global & high_frame),
                       ("only_global_q_above_1pct", high_global & ~high_frame),
                       ("only_frame_q_above_1pct", ~high_global & high_frame),
                       ("both_q_at_most_1pct", ~high_global & ~high_frame)):
        result[name] = int(mask.sum())
    score = points["score"]
    result.update(tp_1=int(np.count_nonzero(score >= thresholds[0])),
        tp_95=int(np.count_nonzero(score >= thresholds[1])),
        below_t95=int(np.count_nonzero(score < thresholds[1])),
        at_t95=int(np.count_nonzero(score == thresholds[1])),
        q_global_above_10pct=int(np.count_nonzero(points["q_global"] > .1)))
    result["recall_1_percent"] = 100 * result["tp_1"] / len(points)
    result["recall_95_percent"] = 100 * result["tp_95"] / len(points)
    return result


def real_frame_attribution(identity, data_root, run, observer, metrics):
    source, prediction = read_real(identity, data_root, run)
    scores, target, eligible = official_frame(source, prediction)
    if not eligible:
        raise ValueError("attribution requested for a frame outside official evaluation")
    slots = np.flatnonzero(target == 1)
    points = np.empty(len(slots), POINT_DTYPE)
    points["sequence"] = identity[0]
    points["frame"] = identity[1]
    points["slot"] = slots
    points["score"] = scores[slots]
    points["precision"], points["q_global"] = observer.values(scores[slots])
    points["q_frame"] = required_frame_fpr(scores[target == 0], scores[slots])
    distance = np.linalg.norm(source.xyzi[:, :3], axis=1)
    points["distance"] = distance[slots]
    thresholds = [metrics["recall_at_fpr_limit"]["threshold"], metrics["official_high_recall"]["threshold"]]
    row = dict(sequence=identity[0], frame=identity[1], eligible=True,
        normal_points=int(np.count_nonzero(target == 0)),
        **anomaly_summary(points, metrics["anomaly_count"], thresholds))
    normal = target == 0
    groups = []
    bins = range_ids(distance)
    for index in range(1, 5):
        use = normal & (bins == index)
        groups.append(dict(sequence=identity[0], distance=RANGES[index], normal_points=int(use.sum()),
            fp_1=int(np.count_nonzero(use & (scores >= thresholds[0]))),
            fp_95=int(np.count_nonzero(use & (scores >= thresholds[1])))))
    row.update({key: sum(group[key] for group in groups) for key in ("fp_1", "fp_95")})
    # Spatial cells select displays only; official metrics keep every original slot.
    fp_slots = np.flatnonzero(normal & (scores >= thresholds[0]))
    candidate = None
    if len(fp_slots):
        xyz = np.unique(source.xyzi[fp_slots, :3], axis=0)
        cells, counts = np.unique(np.floor(xyz).astype(np.int32), axis=0, return_counts=True)
        cell = cells[np.argmax(counts)]
        members = fp_slots[np.all(np.floor(source.xyzi[fp_slots, :3]) == cell, axis=1)]
        anchor = int(members[np.argmax(scores[members])])
        candidate = dict(sequence=identity[0], frame=identity[1], slot=anchor,
            unique_fp_positions_in_cell=int(counts.max()), cell=cell.tolist(),
            center=source.xyzi[anchor, :3].astype(float).tolist())
    return row, points, groups, candidate


def select_real_cases(rows, sequences, candidates):
    failures = sorted((r for r in rows if r["eligible"] and r["sequence"] == 125),
                      key=lambda r: (-r["ap_deficit_pp"], r["frame"]))[:2]
    used = {(r["sequence"], r["frame"]) for r in failures}
    success_key = lambda r: (-r["recall_1_percent"], -r["anomaly_points"], r["sequence"], r["frame"])
    own = min((r for r in rows if r["eligible"] and r["sequence"] == 125 and (125, r["frame"]) not in used), key=success_key)
    other = min((r for r in rows if r["eligible"] and r["sequence"] != 125), key=success_key)
    cases = [dict(kind=kind, sequence=r["sequence"], frame=r["frame"])
             for kind, r in zip(("125_failure", "125_failure", "125_success", "other_success"), [*failures, own, other], strict=True)]
    leaders = sorted(sequences, key=lambda r: (-r["fp_1"], r["sequence"]))[:2]
    for leader in leaders:
        case = min((c for c in candidates if c and c["sequence"] == leader["sequence"]),
                   key=lambda c: (-c["unique_fp_positions_in_cell"], c["frame"], c["cell"]))
        cases.append(dict(kind="normal_cluster", **case))
    return cases


def write_table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def inspect_real_cases(cases, data_root, run, points, thresholds, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.colors import Normalize
    from scipy.spatial import cKDTree

    chinese = FontProperties(fname="/mnt/c/Windows/Fonts/simsun.ttc")
    english = FontProperties(fname="/mnt/c/Windows/Fonts/times.ttf")
    if chinese.get_name() != "SimSun" or english.get_name() != "Times New Roman":
        raise ValueError("required figure fonts are unavailable")
    fontManager.addfont(english.get_file())
    fontManager.addfont(chinese.get_file())
    matplotlib.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42, "axes.unicode_minus": False})
    details = []
    for index, case in enumerate(cases, 1):
        identity = case["sequence"], case["frame"]
        source, prediction = read_real(identity, data_root, run)
        scores, target, eligible = official_frame(source, prediction)
        if not eligible:
            raise ValueError("selected case is no longer officially eligible")
        xyz = source.xyzi[:, :3]
        anomaly = target == 1
        normal = target == 0
        if case["kind"] == "normal_cluster":
            center = np.asarray(case["center"])
            display = np.linalg.norm(xyz - center, axis=1) <= 6
        else:
            lower, upper = xyz[anomaly].min(axis=0), xyz[anomaly].max(axis=0)
            center = (lower + upper) / 2
            display = np.all((xyz >= lower-2) & (xyz <= upper+2), axis=1)
        display &= ~source.zero_slot_mask
        local_slots = np.flatnonzero(display)
        relative = xyz - center
        frame_points = points[(points["sequence"] == identity[0]) & (points["frame"] == identity[1])]
        accepted = scores >= thresholds[0]
        groups = [(display & (target < 0), "#dedede", "范围外或忽略", 2),
                  (display & normal & ~accepted, "#a8a8a8", "正常正确", 3),
                  (display & normal & accepted, "#8826a8", "正常误报", 8),
                  (display & anomaly & accepted, "#008641", "异常检出", 20),
                  (display & anomaly & ~accepted, "#d73027", "异常漏检", 20)]
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        for row, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            left, middle, right = axes[row]
            left.scatter(relative[display & ~anomaly, a], relative[display & ~anomaly, b], s=3, color="#aaaaaa", linewidths=0, rasterized=True)
            left.scatter(relative[display & anomaly, a], relative[display & anomaly, b], s=20, color="#0062b8", linewidths=0, rasterized=True)
            # One fixed raw-logit colour scale for all six displays; scores are never transformed.
            order = np.r_[local_slots[target[local_slots] != 1], local_slots[target[local_slots] == 1]]
            scatter = middle.scatter(relative[order, a], relative[order, b], s=np.where(target[order] == 1, 20, 4),
                c=scores[order], cmap="coolwarm", norm=Normalize(-15, 15), linewidths=0, rasterized=True)
            for use, color, label, size in groups:
                right.scatter(relative[use, a], relative[use, b], s=size, color=color, label=label, linewidths=0, rasterized=True)
            if case["kind"] == "normal_cluster":
                detail = right.inset_axes((.025, .035, .34, .42))
                near = np.all(np.abs(relative) <= 1.2, axis=1)
                for use, color, _, size in groups:
                    take = use & near
                    detail.scatter(relative[take, a], relative[take, b], s=size, color=color, linewidths=0, rasterized=True)
                detail.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal")
                detail.set_title("误报簇局部", fontproperties=chinese, fontsize=8)
                detail.tick_params(labelsize=7)
            for ax in (left, middle, right):
                ax.set_xlabel(f"{'xyz'[a]} (m)", fontproperties=english)
                ax.set_ylabel(f"{'xyz'[b]} (m)", fontproperties=english)
                ax.set_xlim(relative[display, a].min()-.1, relative[display, a].max()+.1)
                ax.set_ylim(relative[display, b].min()-.1, relative[display, b].max()+.1)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=.15)
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontproperties(english)
        for ax, title in zip(axes[0], ("真实异常标注为蓝色", "原始异常分数", "固定全局阈值下的检出与误报"), strict=True):
            ax.set_title(title, fontproperties=chinese, fontsize=12)
        handles, labels = axes[0, 2].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, prop=chinese)
        fig.suptitle("真实局部观测与模型排序", fontproperties=chinese, fontsize=17)
        fig.text(.5, .955, f"Case {index} | val/{identity[0]}/{identity[1]:06d} | t1={thresholds[0]:.6f} | t95={thresholds[1]:.6f} | N+={anomaly.sum()} | TP1={np.count_nonzero(anomaly & accepted)}",
                 ha="center", fontproperties=english)
        fig.tight_layout(rect=(0, .055, .93, .94))
        bar = fig.colorbar(scatter, cax=fig.add_axes((.945, .29, .013, .42)), extend="both")
        bar.set_label("Raw logit", fontproperties=english)
        for threshold, label in zip(thresholds, ("t1", "t95"), strict=True):
            bar.ax.axhline(threshold, color="black", linewidth=.8)
            bar.ax.text(-.4, threshold, label, ha="right", va="center", fontproperties=english)
        for tick in bar.ax.get_yticklabels():
            tick.set_fontproperties(english)
        stem = f"case_{index}"
        fig.savefig(output / f"{stem}.png", dpi=160)
        fig.savefig(output / f"{stem}.pdf")
        plt.close(fig)
        # Geometric descriptions concern observed returns, not an inferred ground surface.
        anomalous_xyz = xyz[anomaly]
        missed_xyz = xyz[anomaly & ~accepted]
        nearest_normal = cKDTree(xyz[normal]).query(anomalous_xyz, k=1)[0]
        normal_local = display & normal
        raw, counts = np.unique(source.labels.semantic[normal_local], return_counts=True)
        anomaly_world = anomalous_xyz.astype(np.float64) @ source.lidar_pose[:3, :3].T + source.lidar_pose[:3, 3]
        summary = dict(case=index, figure=f"{stem}.png", pdf=f"{stem}.pdf", center=center.tolist(),
            normal_points_local=int(normal_local.sum()), fp_1_local=int(np.count_nonzero(normal_local & accepted)),
            anomaly_points_local=int(np.count_nonzero(display & anomaly)),
            score_display_clipped_points=int(np.count_nonzero(display & ((scores < -15) | (scores > 15)))),
            normal_raw_codes=dict(zip(map(str, raw), map(int, counts), strict=True)),
            anomaly_unique_positions=len(np.unique(anomalous_xyz, axis=0)),
            anomaly_extent_xyz_m=np.ptp(anomalous_xyz, axis=0).astype(float).tolist(),
            missed_extent_xyz_m=np.ptp(missed_xyz, axis=0).astype(float).tolist() if len(missed_xyz) else None,
            anomaly_world_centroid=anomaly_world.mean(axis=0).tolist(),
            anomaly_distance_min_m=float(frame_points["distance"].min()),
            anomaly_distance_max_m=float(frame_points["distance"].max()),
            anomaly_score_quantiles=np.quantile(scores[anomaly], [0, .1, .5, .9, 1]).tolist(),
            normal_local_score_quantiles=np.quantile(scores[normal_local], [0, .1, .5, .9, 1]).tolist(),
            anomaly_nearest_normal_distance_quantiles_m=np.quantile(nearest_normal, [0, .5, .9, 1]).tolist(),
            nearest_anomaly_to_center_m=float(np.linalg.norm(anomalous_xyz-center, axis=1).min()),
            **anomaly_summary(frame_points, len(points), thresholds))
        if case["kind"] == "normal_cluster":
            cell = np.all(np.floor(xyz) == case["cell"], axis=1) & ~source.zero_slot_mask
            cell_normal = cell & normal
            cell_fp = cell_normal & accepted
            distance = np.linalg.norm(xyz, axis=1)
            raw, counts = np.unique(source.labels.semantic[cell_fp], return_counts=True)
            summary["selected_cell"] = dict(actual_points=int(cell.sum()), normal_points=int(cell_normal.sum()),
                fp_1=int(cell_fp.sum()), outside_range=int(np.count_nonzero(cell &
                    ((distance < Official.min_eval_distance) | (distance > Official.max_eval_distance)))),
                distance_min_m=float(distance[cell].min()), distance_max_m=float(distance[cell].max()),
                fp_extent_xyz_m=np.ptp(xyz[cell_fp], axis=0).tolist(),
                fp_raw_codes=dict(zip(map(str, raw), map(int, counts), strict=True)),
                fp_score_quantiles=np.quantile(scores[cell_fp], [0, .5, 1]).tolist())
        details.append({**case, **summary})
    _atomic_json(output / "cases.json", dict(rules=REAL_CASE_RULES, thresholds=thresholds,
        score_display="fixed linear raw logit [-15,15], colour-only clipping; left column marks true anomalies; t1 decisions use full unmodified values",
        cases=details))


def diagnose_real(args):
    started = time.monotonic()
    previous = json.loads((args.run / "real.json").read_text())
    binding = json.loads((args.run / "predictions" / "manifest.json").read_text())
    with (args.run / "epoch1.pt").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if binding != previous["binding"] or binding["checkpoint_sha256"] != digest:
        raise ValueError("saved real result and predictions do not bind to this epoch1 checkpoint")
    output = args.run / "attribution"
    output.mkdir(exist_ok=True)
    host_before = host_disk()
    peak = 8 * (previous["metrics"]["normal_count"] + previous["metrics"]["anomaly_count"]) + 100_000_000
    if host_before["SizeRemaining"] - peak < host_before["reserve_bytes"]:
        raise OSError("real attribution would consume the E: safety reserve")
    print(json.dumps(dict(event="real_start", jobs=args.jobs, peak_new_bytes=peak,
        case_rules=REAL_CASE_RULES, host=host_before)), flush=True)
    identities = []
    for sequence in load_protocol().public_sequence_ids:
        source = STUSequence.open(args.data_root, protocol=load_protocol(), partition="val",
                                  sequence_id=sequence, label_mode=LabelMode.REQUIRED)
        identities.extend((sequence, frame) for frame in source.frame_ids)
    count = 0
    def progress():
        nonlocal count
        count += 1
        if count % 500 == 0 or count == len(identities):
            print(json.dumps(dict(event="real_pool", frames=count, seconds=time.monotonic()-started, host=host_disk())), flush=True)
    observer = APAttribution()
    metrics, frame_rows = evaluate_frames(real_frames(identities, args.data_root, args.run, args.jobs),
        directory=output, observe=observer, check_resources=progress)
    if metrics != previous["metrics"] or frame_rows != previous["frames"]:
        raise ValueError("original real metrics or official frame membership did not reproduce exactly")
    observer.values(np.empty(0, np.float32))  # Freeze the small lookup before concurrent read-only queries.
    _atomic_json(output / "metrics.json", dict(binding=binding, metrics=metrics,
        original_result_exactly_reproduced=True))
    print(json.dumps(dict(event="real_metrics", metrics=metrics, seconds=time.monotonic()-started)), flush=True)
    eligible = [(r["sequence"], r["frame"]) for r in frame_rows if r["eligible"]]
    rows, point_blocks, candidates, joint = [], [], [], {}
    def analyze(identity):
        return real_frame_attribution(identity, args.data_root, args.run, observer, metrics)
    with ThreadPoolExecutor(args.jobs) as pool:
        for i, (row, points, groups, candidate) in enumerate(pool.map(analyze, eligible), 1):
            rows.append(row)
            point_blocks.append(points)
            candidates.append(candidate)
            for group in groups:
                key = group["sequence"], group["distance"]
                if key not in joint:
                    joint[key] = dict(sequence=key[0], distance=key[1], normal_points=0, fp_1=0, fp_95=0)
                for name in ("normal_points", "fp_1", "fp_95"):
                    joint[key][name] += group[name]
            if i % 300 == 0 or i == len(eligible):
                print(json.dumps(dict(event="real_attribute", frames=i, seconds=time.monotonic()-started, host=host_disk())), flush=True)
    points = np.concatenate(point_blocks)
    thresholds = [metrics["recall_at_fpr_limit"]["threshold"], metrics["official_high_recall"]["threshold"]]
    total = anomaly_summary(points, metrics["anomaly_count"], thresholds)
    if len(points) != metrics["anomaly_count"] or not np.isclose(total["ap_deficit_pp"], 100-metrics["AP"], rtol=0, atol=1e-10):
        raise ValueError("positive-point AP deficits do not sum to 100 minus AP")
    sequences = []
    for sequence in load_protocol().public_sequence_ids:
        frames = [r for r in rows if r["sequence"] == sequence]
        summary = dict(sequence=sequence, eligible_frames=len(frames),
            **anomaly_summary(points[points["sequence"] == sequence], metrics["anomaly_count"], thresholds))
        for key in ("normal_points", "fp_1", "fp_95"):
            summary[key] = sum(r[key] for r in frames)
        sequences.append(summary)
    for index, key in ((0, "1"), (1, "95")):
        reference = metrics["recall_at_fpr_limit" if index == 0 else "official_high_recall"]
        if sum(r[f"fp_{key}"] for r in rows) != reference["fp"] or total[f"tp_{key}"] != reference["tp"]:
            raise ValueError("global threshold counts do not match the saved operating point")
    if sum(r["normal_points"] for r in joint.values()) != metrics["normal_count"]:
        raise ValueError("joint distance groups lost official normal points")
    for table in (rows, sequences, list(joint.values())):
        for row in table:
            for suffix, reference in (("1", metrics["recall_at_fpr_limit"]), ("95", metrics["official_high_recall"])):
                row[f"fpr_{suffix}_percent"] = 100 * row[f"fp_{suffix}"] / row["normal_points"] if row["normal_points"] else None
                row[f"fp_{suffix}_share_percent"] = 100 * row[f"fp_{suffix}"] / reference["fp"]
            row["additional_fp_to_95"] = row["fp_95"] - row["fp_1"]
            if "ap_deficit_pp" in row:
                row["ap_deficit_share_percent"] = 100 * row["ap_deficit_pp"] / total["ap_deficit_pp"]
    cases = select_real_cases(rows, sequences, candidates)
    # Persist statistical selection before any local geometry is inspected or plotted.
    _atomic_json(output / "cases.json", dict(rules=REAL_CASE_RULES, thresholds=thresholds, cases=cases))
    row_lookup = {(r["sequence"], r["frame"]): r for r in rows}
    complete_rows = [row_lookup.get((r["sequence"], r["frame"]), {**r, "ap_deficit_pp": 0.0}) for r in frame_rows]
    write_table(output / "frames.csv", complete_rows)
    write_table(output / "sequences.csv", sequences)
    write_table(output / "joint.csv", list(joint.values()))
    np.savez_compressed(output / "anomalies.npz", **{name: points[name] for name in points.dtype.names})
    _atomic_json(output / "summary.json", dict(binding=binding, total=total, sequences=sequences,
        definitions=dict(scope="official eligible frames and original valid slots; no score changes",
            ap_deficit="100/N_positive * sum(1 - global precision at the complete anomaly-score tie); percentage points, not causal attribution",
            required_fpr="normal scores >= anomaly score, divided by normal count in global or same-frame official scope",
            tail="below_t95 is score < official pruned-ROC high-recall threshold; q_global_above_10pct is a separately declared tail descriptor",
            joint="sequence x sensor distance at unchanged global 1% FPR and official FPR95 thresholds",
            ineligible="frame inventory retained; ineligible points do not enter AP or required-FPR summaries"),
        seconds_before_plots=time.monotonic()-started, jobs=args.jobs, host_before=host_before, host_after=host_disk()))
    inspect_real_cases(cases, args.data_root, args.run, points, thresholds, output)
    print(json.dumps(dict(event="real_complete", directory=str(output), seconds=time.monotonic()-started, host=host_disk())), flush=True)


def scope_masks(xyzi, target):
    # Reuse only STU's observation filter, never its raw-semantic anomaly mapping.
    distance = np.linalg.norm(xyzi[:, :3], axis=1)
    inside = (distance >= Official.min_eval_distance) & (distance <= Official.max_eval_distance)
    eligible = int(np.count_nonzero(inside & (target == 1))) >= Official.min_num_points_to_eval
    return ((target >= 0), (target >= 0) & inside & eligible), distance, eligible


def range_ids(distance):
    result = np.searchsorted([2.5, 10, 20, 35], distance, side="right")
    result[distance > 50] = 5
    return result


def initialize(dataset_root, data_root, run, temporary, thresholds=None):
    global DATASET, RUN, TEMPORARY, THRESHOLDS, SAMPLES, WORLDS
    DATASET = FrozenDataset(dataset_root, data_root, "validation")
    RUN, TEMPORARY, THRESHOLDS = Path(run), Path(temporary), thresholds
    SAMPLES = defaultdict(list)
    WORLDS = list(dict.fromkeys(path.parent.parent.name for path, _, _ in DATASET.samples))
    if len(WORLDS) > 32:
        raise ValueError("world recurrence bitsets support at most 32 worlds")
    for path, identity, frame in DATASET.samples:
        SAMPLES[frame].append((path, identity, WORLDS.index(path.parent.parent.name)))
    keys = [(identity, frame) for _, identity, frame in DATASET.samples]
    if len(set(keys)) != len(keys) or len(WORLDS) * len(SAMPLES) != len(keys):
        raise ValueError("diagnosis requires distinct complete world/frame identities")


def read_world(original, entry):
    path, identity, world_index = entry
    frozen = FrozenFrame.load(path, original, identity)
    path = RUN / "predictions" / "synthetic" / WORLDS[world_index] / f"{original.frame_id:06d}.npz"
    prediction = FramePrediction.load(path, frozen.source)
    # Loading already validates exact original-slot coverage and scan identity.
    scores = np.zeros(frozen.source.slot_count, np.float32)
    scores[prediction.source_slot] = prediction.anomaly_score
    return frozen, scores


def pack_frame(frame):
    original = DATASET.sequence[frame]
    rows = []
    streams = [(TEMPORARY / f"{frame:06d}_{scope}.bin").open("wb") for scope in SCOPES]
    try:
        for entry in SAMPLES[frame]:
            frozen, scores = read_world(original, entry)
            target = frozen.anomaly_target
            masks, _, eligible = scope_masks(frozen.source.xyzi, target)
            row = dict(world=WORLDS[entry[2]], world_identity=entry[1], frame=frame,
                       actual_points=len(frozen.source.real_slots), eligible=eligible)
            for scope, use, stream in zip(SCOPES, masks, streams, strict=True):
                packed_scores(scores[use], target[use], score_kind="logit").tofile(stream)
                row[scope] = dict(normal=int(np.count_nonzero(use & (target == 0))),
                                  anomaly=int(np.count_nonzero(use & (target == 1))))
            rows.append(row)
    finally:
        for stream in streams:
            stream.close()
    return rows


def recurrence(seen, false_positive):
    """Distinct world membership, so coincident slots cannot inflate recurrence."""
    return dict(
        observed=int(np.count_nonzero(seen)),
        false_positive=[int(np.count_nonzero(row)) for row in false_positive],
        observed_world_histogram=np.bincount(np.bitwise_count(seen[seen != 0]), minlength=33).tolist(),
        fp_world_histograms=[np.bincount(np.bitwise_count(row[row != 0]), minlength=33).tolist()
                             for row in false_positive],
    )


def add_group(groups, kind, values, accepted):
    # Column 0 is the normal denominator; columns 1..3 use fixed global thresholds.
    unique, inverse = np.unique(values, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique))
    fps = [np.bincount(inverse[row], minlength=len(unique)) for row in accepted]
    for i, value in enumerate(unique):
        groups[kind][str(value)] += np.array([counts[i], *(row[i] for row in fps)], np.int64)


def analyze_frame(frame):
    original = DATASET.sequence[frame]
    normal_source = np.flatnonzero((original.labels.semantic_target != 255) & ~original.zero_slot_mask)
    # A coordinate key is local to this source frame, not a track across time.
    _, first, inverse = np.unique(original.xyzi[normal_source, :3], axis=0,
                                  return_index=True, return_inverse=True)
    positions = np.full(original.slot_count, -1, np.int64)
    positions[normal_source] = inverse
    slot_seen = np.zeros((2, original.slot_count), np.uint32)
    slot_fp = np.zeros((2, 3, original.slot_count), np.uint32)
    position_seen = np.zeros((2, len(first)), np.uint32)
    position_fp = np.zeros((2, 3, len(first)), np.uint32)
    maximum = np.full((2, original.slot_count), -np.inf, np.float32)
    maximum_world = np.full((2, original.slot_count), -1, np.int16)
    groups = [{kind: defaultdict(lambda: np.zeros(4, np.int64))
               for kind in ("semantic", "distance", "frame_type", "world", "official_membership")}
              for _ in SCOPES]
    totals = np.zeros((2, 2, 3), np.int64)
    for entry in SAMPLES[frame]:
        frozen, scores = read_world(original, entry)
        target = frozen.anomaly_target
        masks, distance, eligible = scope_masks(frozen.source.xyzi, target)
        frame_type = 2 if eligible else int(np.any(target == 1))
        world_bit = np.uint32(1 << entry[2])
        for s, use in enumerate(masks):
            positive = scores[use & (target == 1)]
            totals[s, 1] += (positive[None, :] >= THRESHOLDS[s, :, None]).sum(axis=1)
            slots = np.flatnonzero(use & (target == 0))
            if not len(slots):
                continue
            if np.any(positions[slots] < 0) or not np.array_equal(frozen.source.xyzi[slots], original.xyzi[slots]):
                raise ValueError("normal diagnosis rows must be unchanged original background slots")
            accepted = scores[slots][None, :] >= THRESHOLDS[s, :, None]
            totals[s, 0] += accepted.sum(axis=1)
            add_group(groups[s], "semantic", original.labels.semantic[slots], accepted)
            add_group(groups[s], "distance", range_ids(distance[slots]), accepted)
            nfp = np.r_[len(slots), accepted.sum(axis=1)]
            groups[s]["frame_type"][str(frame_type)] += nfp
            groups[s]["world"][WORLDS[entry[2]]] += nfp
            inside = (distance[slots] >= 2.5) & (distance[slots] <= 50)
            membership = inside.astype(np.int8) + 2 * int(eligible)
            add_group(groups[s], "official_membership", membership, accepted)
            slot_seen[s, slots] |= world_bit
            np.bitwise_or.at(position_seen[s], positions[slots], world_bit)
            improved = slots[scores[slots] > maximum[s, slots]]
            maximum[s, improved] = scores[improved]
            maximum_world[s, improved] = entry[2]
            for k, take in enumerate(accepted):
                selected = slots[take]
                slot_fp[s, k, selected] |= world_bit
                np.bitwise_or.at(position_fp[s, k], positions[selected], world_bit)
    result = dict(frame=frame, scopes={}, candidates=[])
    for s, scope in enumerate(SCOPES):
        count = sum((v for v in groups[s]["semantic"].values()), np.zeros(4, np.int64))
        top = []
        for k in range(3):
            repetitions = np.bitwise_count(slot_fp[s, k])
            selected = np.lexsort((np.arange(len(repetitions)), -maximum[s], -repetitions.astype(np.int16)))[:5]
            top.append([dict(slot=int(i), xyz=original.xyzi[i, :3].tolist(),
                             semantic=int(original.labels.semantic[i]),
                             fp_worlds=int(repetitions[i]),
                             observed_worlds=int(np.bitwise_count(slot_seen[s, i])))
                        for i in selected if repetitions[i]])
        result["scopes"][scope] = dict(
            counts=count.tolist(), fp=totals[s, 0].tolist(), tp=totals[s, 1].tolist(),
            groups={kind: {key: value.tolist() for key, value in table.items()}
                    for kind, table in groups[s].items()},
            slots=recurrence(slot_seen[s], slot_fp[s]),
            positions=recurrence(position_seen[s], position_fp[s]), top_slots=top,
        )
        # Choose visible clusters for inspection, counting coincident 201 slots once.
        available = np.flatnonzero(slot_fp[s, 1])
        for semantic in np.unique(original.labels.semantic[available]):
            slots = available[original.labels.semantic[available] == semantic]
            _, keep = np.unique(positions[slots], return_index=True)
            slots = slots[keep]
            cells = np.floor(original.xyzi[slots, :3]).astype(np.int32)
            unique, cell_index, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
            cell = int(np.argmax(counts))
            members = slots[cell_index == cell]
            center = unique[cell] + 0.5
            anchor = int(members[np.argmin(np.linalg.norm(original.xyzi[members, :3] - center, axis=1))])
            result["candidates"].append(dict(scope=scope, semantic=int(semantic), frame=frame,
                slot=anchor, world=WORLDS[int(maximum_world[s, anchor])],
                xyz=original.xyzi[anchor, :3].tolist(), score=float(maximum[s, anchor]),
                unique_fp_positions_in_one_metre_cell=int(counts[cell])))
    return result


def group_table(table, total_fp):
    return {key: dict(normal=int(value[0]), points={f"{limit:g}": dict(
        fp=int(value[k + 1]), FPR=100 * value[k + 1] / value[0] if value[0] else None,
        fp_share=100 * value[k + 1] / total_fp[k] if total_fp[k] else None)
        for k, limit in enumerate(LIMITS)}) for key, value in table.items()}


def aggregate(rows, metrics):
    result = {}
    for scope in SCOPES:
        tables = {kind: defaultdict(lambda: np.zeros(4, np.int64)) for kind in rows[0]["scopes"][scope]["groups"]}
        source_frames = {}
        identities = {kind: dict(observed=0, false_positive=np.zeros(3, np.int64),
                        observed_world_histogram=np.zeros(33, np.int64), fp_world_histograms=np.zeros((3, 33), np.int64))
                      for kind in ("slots", "positions")}
        fp, tp = np.zeros(3, np.int64), np.zeros(3, np.int64)
        for row in rows:
            data = row["scopes"][scope]
            fp += data["fp"]
            tp += data["tp"]
            source_frames[str(row["frame"])] = np.asarray(data["counts"], np.int64)
            for kind, table in data["groups"].items():
                for key, value in table.items():
                    tables[kind][key] += value
            for kind, totals in identities.items():
                for key in totals:
                    totals[key] += data[kind][key] if key == "observed" else np.asarray(data[kind][key], np.int64)
        expected = metrics[scope]
        for k, limit in enumerate(LIMITS):
            point = expected["operating_points"][f"{limit:g}"]
            if (int(fp[k]), int(tp[k])) != (point["fp"], point["tp"]):
                raise ValueError("grouped counts disagree with complete-tie global ranking")
        tables["source_frame"] = source_frames
        for table in tables.values():
            total = sum(table.values(), np.zeros(4, np.int64))
            if not np.array_equal(total, np.r_[expected["normal_count"], fp]):
                raise ValueError("group denominator or false-positive count does not partition the scope")
        for kind, counts in identities.items():
            identities[kind] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in counts.items()}
        source_counts = np.array(list(source_frames.values()))
        concentration = []
        for k in range(3):
            descending = np.sort(source_counts[:, k + 1])[::-1]
            concentration.append(dict(source_frames_with_fp=int(np.count_nonzero(descending)),
                top_frame_fp_share={str(n): 100 * int(descending[:n].sum()) / int(fp[k]) if fp[k] else None
                                   for n in (1, 5, 10, 50)},
                unique_slots=identities["slots"]["false_positive"][k],
                unique_positions=identities["positions"]["false_positive"][k]))
        result[scope] = dict(groups={kind: group_table(table, fp) for kind, table in tables.items()},
                             identities=identities, concentration=concentration)
    return result


def inspect_cases(rows, summary, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager

    chinese = FontProperties(fname="/mnt/c/Windows/Fonts/simsun.ttc")
    english = FontProperties(fname="/mnt/c/Windows/Fonts/times.ttf")
    if chinese.get_name() != "SimSun" or english.get_name() != "Times New Roman":
        raise ValueError("required figure fonts are unavailable")
    fontManager.addfont(english.get_file())
    fontManager.addfont(chinese.get_file())
    matplotlib.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42,
                                "axes.unicode_minus": False})
    candidates = [case for row in rows for case in row["candidates"]]
    selected = []
    for scope in SCOPES:
        semantic = summary[scope]["groups"]["semantic"]
        leaders = sorted(semantic, key=lambda key: -semantic[key]["points"]["0.001"]["fp"])[:3]
        for key in leaders:
            choices = [case for case in candidates if case["scope"] == scope and case["semantic"] == int(key)]
            if choices:
                selected.append(max(choices, key=lambda case: (case["unique_fp_positions_in_one_metre_cell"], -case["frame"])))
    cases = []
    for index, case in enumerate(selected):
        original = DATASET.sequence[case["frame"]]
        entry = next(item for item in SAMPLES[case["frame"]] if WORLDS[item[2]] == case["world"])
        frozen, scores = read_world(original, entry)
        target = frozen.anomaly_target
        masks, _, _ = scope_masks(frozen.source.xyzi, target)
        use = masks[SCOPES.index(case["scope"])]
        threshold = metrics[case["scope"]]["operating_points"]["0.001"]["threshold"]
        center = np.asarray(case["xyz"])
        local = (np.linalg.norm(frozen.source.xyzi[:, :3] - center, axis=1) <= 6) & ~frozen.source.zero_slot_mask
        local_slots = np.flatnonzero(local)
        # Plot original geometry once per exact position; score maxima preserve visible false positives.
        xyz, representative, inverse = np.unique(frozen.source.xyzi[local, :3], axis=0, return_index=True, return_inverse=True)
        local_fp = local & use & (target == 0) & (scores >= threshold)
        fp_position = np.zeros(len(xyz), bool)
        np.logical_or.at(fp_position, inverse, local_fp[local])
        anomaly_position = target[local_slots[representative]] == 1
        ignored_position = target[local_slots[representative]] == -1
        highlighted = (original.labels.semantic[local_slots[representative]] == case["semantic"]) & ~anomaly_position & ~ignored_position
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        groups = [(np.ones(len(xyz), bool), "#cccccc", "完整局部背景", 3),
                  (highlighted, "#3182bd", "所检正常类别", 5),
                  (fp_position, "#de2d26", "正常误报", 8),
                  (anomaly_position, "#00a651", "插入异常", 12)]
        relative = xyz - center
        for ax, (a, b) in zip(axes, ((0, 1), (0, 2), (1, 2)), strict=True):
            for take, color, label, size in groups:
                ax.scatter(relative[take, a], relative[take, b], s=size, c=color, label=label, linewidths=0)
            ax.scatter([0], [0], s=65, marker="x", color="black", linewidths=1)
            ax.set_xlabel(f"{'xyz'[a]} (m)", fontproperties=english)
            ax.set_ylabel(f"{'xyz'[b]} (m)", fontproperties=english)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(english)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, prop=chinese)
        fig.suptitle("正常高分点的局部几何核对", fontproperties=chinese)
        fig.text(.5, .91, f"{case['scope']} | train/201/{case['frame']:06d} | {case['world']} | raw={case['semantic']} | slot={case['slot']}",
                 ha="center", fontproperties=english)
        fig.tight_layout(rect=(0, .07, 1, .89))
        stem = f"case_{index + 1}"
        fig.savefig(output / f"{stem}.png", dpi=160)
        fig.savefig(output / f"{stem}.pdf")
        plt.close(fig)
        raw, counts = np.unique(original.labels.semantic[local & (target == 0)], return_counts=True)
        anomalies = frozen.source.xyzi[target == 1, :3]
        cases.append({**case, "world_identity": entry[1], "global_threshold": threshold,
            "radius_m": 6, "normal_semantics": dict(zip(map(str, raw), map(int, counts), strict=True)),
            "normal_points": int(np.count_nonzero(local & (target == 0))),
            "false_positive_points": int(np.count_nonzero(local_fp)),
            "anomaly_points": int(np.count_nonzero(local & (target == 1))),
            "unique_positions": len(xyz), "unique_fp_positions": int(fp_position.sum()),
            "relative_xyz_min": relative.min(axis=0).tolist(), "relative_xyz_max": relative.max(axis=0).tolist(),
            "nearest_inserted_return_m": float(np.linalg.norm(anomalies - center, axis=1).min()) if len(anomalies) else None,
            "figure": f"{stem}.png", "pdf": f"{stem}.pdf"})
    return cases


def _paired_metrics(stream, count, observe=None):
    """Sort one exact record file in place; never allocate a second pooled copy."""
    stream.flush()
    if stream.tell() != 8 * count:
        raise ValueError("compact metric records lost point identities")
    if not count:
        return exact_metrics(np.empty(0, np.uint64), score_kind="logit", fpr_limits=LIMITS)
    ordered = np.memmap(stream, dtype=np.uint64, mode="r+", shape=(count,))
    ordered.sort(kind="quicksort")
    result = exact_metrics(ordered, score_kind="logit", fpr_limits=LIMITS, observe=observe)
    del ordered
    return result


def _paired_thresholds(metrics):
    points = [metrics.get("operating_points", {}).get(f"{limit:g}", {}) for limit in LIMITS]
    points.append(metrics.get("official_high_recall", {}))
    return np.array([point.get("threshold") if point.get("threshold") is not None else np.inf
                     for point in points], np.float64)


def _paired_add(groups, scope, kind, values, score, labels, thresholds):
    """Count every original point at this arm's own complete-tie thresholds."""
    if not len(score):
        return
    if np.ndim(values) == 0:
        count = np.zeros(10, np.int64)
        count[:2] = np.bincount(labels, minlength=2)
        for index, threshold in enumerate(thresholds):
            count[2+2*index:4+2*index] = np.bincount(labels[score >= threshold], minlength=2)
        groups.setdefault((scope, kind, str(values)), np.zeros(10, np.int64))[:] += count
        return
    values = np.asarray(values)
    if values.shape != score.shape or labels.shape != score.shape:
        raise ValueError("fixed group values must share score and label identities")
    unique, inverse = np.unique(values, return_inverse=True)
    count = np.zeros((len(unique), 10), np.int64)
    for label in (0, 1):
        selected = labels == label
        count[:, label] = np.bincount(inverse[selected], minlength=len(unique))
        for index, threshold in enumerate(thresholds):
            accepted = selected & (score >= threshold)
            count[:, 2 + 2 * index + label] = np.bincount(inverse[accepted], minlength=len(unique))
    for value, row in zip(unique, count, strict=True):
        key = (scope, kind, str(value))
        groups.setdefault(key, np.zeros(10, np.int64))[:] += row


def _paired_group_rows(groups):
    rows = []
    for (scope, kind, value), count in sorted(groups.items()):
        row = dict(scope=scope, kind=kind, group=value, normal=int(count[0]), anomaly=int(count[1]))
        for index, name in enumerate(("1", "0.1", "0.01", "95")):
            fp, tp = map(int, count[2+2*index:4+2*index])
            row.update({f"fp_{name}": fp, f"tp_{name}": tp,
                        f"FPR_{name}": 100*fp/count[0] if count[0] else None,
                        f"recall_{name}": 100*tp/count[1] if count[1] else None})
        rows.append(row)
    return rows


def _paired_fixed_cases():
    """Reuse previously inspected identities; new model scores never select a case."""
    sources = {"synthetic": Path("results/v1/diagnosis/groups.json"),
               "real": Path("results/v1/attribution/cases.json"),
               "geometry": Path("results/geometry/cases.json"),
               "normal": Path("results/coverage/geometry/normal_cases.json")}
    return {name: json.loads(path.read_text())["cases"] for name, path in sources.items()}


def _paired_case_mask(case, xyz, labels, kind):
    if kind == "geometry":
        center = np.array([case[f"centre_{axis}_m"] for axis in "xyz"])
        return np.all(np.abs(xyz - center) <= 3, axis=1)
    if kind == "normal":
        return np.all(np.abs(xyz - np.asarray(case["center"])) <= 3, axis=1)
    if kind == "synthetic":
        return np.linalg.norm(xyz - np.asarray(case["xyz"]), axis=1) <= case["radius_m"]
    if case["kind"] == "normal_cluster":
        return np.linalg.norm(xyz - np.asarray(case["center"]), axis=1) <= 6
    positive = xyz[labels == 1]
    return np.all((xyz >= positive.min(axis=0)-2) & (xyz <= positive.max(axis=0)+2), axis=1)


def _paired_synthetic_masks(xyz, labels, target):
    """The existing 2 m and C3 protrusion proxies, independent of predictions."""
    from scipy.spatial import cKDTree
    near = np.zeros(len(labels), bool)
    normal = np.flatnonzero(labels == 0)
    anomaly = xyz[labels == 1].astype(np.float64)
    if len(normal) and len(anomaly):
        distance = cKDTree(anomaly).query(xyz[normal].astype(np.float64),
                    distance_upper_bound=np.nextafter(2., np.inf), workers=1)[0]
        near[normal] = distance <= 2
    valid = np.asarray(target["surface_valid"], bool)
    offset = np.asarray(target["surface_offset_z"])
    raised = (labels == 0) & valid & (offset >= -.2) & (offset <= -.05)
    return np.stack((near, raised, valid))


def _paired_cached_synthetic(row, frozen, scores, flags):
    """Restore compact records to the same sorted original slots used at inference."""
    target = frozen.anomaly_target
    slots = np.flatnonzero(target >= 0)
    n = row["all"]["normal"] + row["all"]["anomaly"]
    if len(slots) != n:
        raise ValueError("compact synthetic label denominator changed")
    scores.seek(4*row["offset"])
    score = np.fromfile(scores, np.float32, n)
    flags.seek(row["flag_offset"])
    packed = np.fromfile(flags, np.uint8, 3*((n+7)//8))
    if len(score) != n or len(packed) != 3*((n+7)//8):
        raise ValueError("truncated compact synthetic records")
    mask = np.unpackbits(packed.reshape(3, -1), axis=1, count=n).astype(bool)
    scopes, distance, eligible = scope_masks(frozen.source.xyzi[slots], target[slots])
    if eligible != row["eligible"]:
        raise ValueError("compact synthetic observation membership changed")
    for scope, use in zip(SCOPES, scopes, strict=True):
        if (int(np.sum(use & (target[slots] == 0))), int(np.sum(use & (target[slots] == 1)))) != (
                row[scope]["normal"], row[scope]["anomaly"]):
            raise ValueError("compact synthetic scope label counts changed")
    return slots, score, target[slots], mask, scopes, distance


def _paired_synthetic(protocol, data_root, run, model, cases):
    from .train import PreparedDataset, loader, predict, add_errors, finalize_errors
    from .supervision import detection_loss
    import torch

    dataset = PreparedDataset(protocol, data_root, "validation")
    if len(dataset) != 13640:
        raise ValueError("paired synthetic evaluation requires all 13640 validation frames")
    order = sorted(range(len(dataset)), key=lambda i: (dataset.frozen.samples[i][2], i))
    settings = protocol["training"]
    rows, errors, cache, groups = [], {}, {}, {}
    loss_sum = np.zeros(3)
    frame_key, reused, count = None, 0, 0
    started = time.monotonic()
    with tempfile.TemporaryFile(dir=run) as scores, tempfile.TemporaryFile(dir=run) as flags:
        with tempfile.TemporaryFile(dir=run) as packed:
            batches = loader(dataset, order, settings["loader_workers"], settings["seed"])
            for sample in batches:
                if sample["frame"] != frame_key:
                    cache.clear()
                    frame_key = sample["frame"]
                losses, dense = [], None
                for index, view in enumerate(sample["views"]):
                    key = view["fingerprint"]
                    if key in cache:
                        output = cache[key]
                        reused += 1
                    else:
                        output = predict(model, view["scan"])
                        cache[key] = output
                    add_errors(errors, index, output, view["target"], dense)
                    losses.append(float(detection_loss(torch.from_numpy(output["logits"]), view["target"]["labels"])))
                    if index == 0:
                        dense = output["logits"]
                loss_sum += losses
                source = sample["source"]
                target = sample["views"][0]["target"]
                labels = target["labels"].numpy()
                slots = source.real_slots
                valid = labels >= 0
                frame_masks, _, eligible = scope_masks(source.xyzi[slots], labels)
                masks = _paired_synthetic_masks(source.xyzi[slots, :3], labels, target)[:, valid]
                row = dict(index=sample["index"], world=sample["world"], world_identity=sample["identity"],
                           frame=sample["frame"], eligible=eligible, actual_points=len(slots),
                           offset=count, flag_offset=flags.tell(), detection_loss=float(np.mean(losses)))
                for scope, mask in zip(SCOPES, frame_masks, strict=True):
                    row[scope] = dict(normal=int(np.sum(mask & (labels == 0))),
                                      anomaly=int(np.sum(mask & (labels == 1))))
                dense[valid].astype(np.float32).tofile(scores)
                np.packbits(masks, axis=1).tofile(flags)
                packed_scores(dense[valid], labels[valid], score_kind="logit").tofile(packed)
                count += int(valid.sum())
                rows.append(row)
                if len(rows) % 200 == 0:
                    disk = host_disk()
                    if disk["SizeRemaining"] < disk["reserve_bytes"]:
                        raise OSError("compact synthetic evaluation reached the E: reserve")
                    print(json.dumps(dict(event="paired_synthetic", frames=len(rows), total=len(dataset),
                        reused_forwards=reused, seconds=time.monotonic()-started,
                        temporary_bytes=scores.tell()+flags.tell()+packed.tell(), host=disk)), flush=True)
            if len(rows) != len(dataset):
                raise ValueError("synthetic evaluation did not visit every prescribed frame")
            metrics = {"all": _paired_metrics(packed, count)}
        cache.clear()
        scores.flush()
        flags.flush()
        # Only compact scores and three bit masks survive inference. Replay fixed
        # scan identities, without recomputing model inputs or auxiliary targets.
        def frames():
            for row in rows:
                path, identity, frame = dataset.frozen.samples[row["index"]]
                if (identity, frame, path.parent.parent.name) != (row["world_identity"], row["frame"], row["world"]):
                    raise ValueError("compact synthetic score identity changed")
                original = dataset.frozen.sequence[frame]
                frozen = FrozenFrame.load(path, original, identity)
                yield row, frozen, *_paired_cached_synthetic(row, frozen, scores, flags)

        # The filtered sort is created only after the larger all-point sort closes.
        for scope_index, scope in enumerate(SCOPES):
            if scope_index:
                with tempfile.TemporaryFile(dir=run) as packed:
                    filtered_count = 0
                    for row, frozen, slots, score, labels, mask, scopes, distance in frames():
                        use = scopes[scope_index]
                        packed_scores(score[use], labels[use], score_kind="logit").tofile(packed)
                        filtered_count += int(use.sum())
                    metrics[scope] = _paired_metrics(packed, filtered_count)
            thresholds = _paired_thresholds(metrics[scope])
            for row, frozen, slots, score, labels, mask, scopes, distance in frames():
                use = scopes[scope_index]
                frame_type = 2 if row["eligible"] else int(np.any(labels == 1))
                for kind, value in (("all", "all"), ("world", row["world"]),
                                    ("source_frame", row["frame"]), ("frame_type", frame_type)):
                    _paired_add(groups, scope, kind, value, score[use], labels[use], thresholds)
                for kind, value in (("distance", range_ids(distance)),
                        ("official_membership", ((distance >= 2.5) & (distance <= 50)).astype(np.int8)+2*int(row["eligible"]))):
                    _paired_add(groups, scope, kind, value[use], score[use], labels[use], thresholds)
                normal = use & (labels == 0)
                _paired_add(groups, scope, "semantic", frozen.source.labels.semantic[slots][normal],
                            score[normal], labels[normal], thresholds)
                for name, selected in (("nearby_kept_normal", mask[0]), ("raised_normal", mask[1]),
                        ("nearby_raised_normal", mask[0] & mask[1]), ("surface_valid_normal", mask[2] & (labels == 0))):
                    take = use & selected
                    _paired_add(groups, scope, "normal_context", name, score[take], labels[take], thresholds)
                for index, case in enumerate(cases):
                    if case["scope"] == scope and (case["world"], case["frame"]) == (row["world"], row["frame"]):
                        if case["world_identity"] != row["world_identity"]:
                            raise ValueError("fixed synthetic case world changed")
                        take = use & _paired_case_mask(case, frozen.source.xyzi[slots, :3], labels, "synthetic")
                        _paired_add(groups, scope, "fixed_case", index+1, score[take], labels[take], thresholds)
            expected = metrics[scope]
            totals = groups[scope, "all", "all"]
            if tuple(totals[:2]) != (expected["normal_count"], expected["anomaly_count"]):
                raise ValueError("synthetic compact groups lost a label denominator")
            for k, point in enumerate([expected["operating_points"][f"{v:g}"] for v in LIMITS] + [expected["official_high_recall"]]):
                if tuple(totals[2+2*k:4+2*k]) != (point["fp"], point["tp"]):
                    raise ValueError("synthetic groups disagree with complete-tie working points")
    for row in rows:
        for key in ("index", "offset", "flag_offset"):
            row.pop(key)
    return dict(scope="all 13640 existing train/201 validation world frames; synthetic labels retained",
                metrics=metrics, frames=rows, groups=_paired_group_rows(groups),
                detection_loss=float(loss_sum.mean()/len(rows)), detection_loss_by_view=(loss_sum/len(rows)).tolist(),
                auxiliary=finalize_errors(errors), reused_forwards=reused, seconds=time.monotonic()-started)


def _paired_real(protocol, data_root, run, model, cases):
    from .train import RealDataset, loader, predict

    dataset = RealDataset(protocol, data_root)
    if len(dataset) != 8659:
        raise ValueError("paired real evaluation requires all 8659 prescribed scans")
    settings = protocol["training"]
    rows, blocks, groups, sequences = [], [], {}, {}
    reader = None
    def labelled(sequence, frame):
        nonlocal reader
        if reader is None or reader.spec.sequence_id != sequence:
            reader = STUSequence.open(data_root, protocol=load_protocol(), partition="val",
                                      sequence_id=sequence, label_mode=LabelMode.REQUIRED)
        return reader[frame]
    count, started = 0, time.monotonic()
    observer = APAttribution()
    with tempfile.TemporaryFile(dir=run) as saved:
        with tempfile.TemporaryFile(dir=run) as packed:
            batches = loader(dataset, list(range(len(dataset))), settings["loader_workers"], settings["seed"])
            for sample in batches:
                source = sample["source"]
                if source.labels is not None:
                    raise ValueError("real paired model input exposed labels")
                output = predict(model, sample["scan"])
                # The labelled reader is accessed only after this scan's forward.
                truth = labelled(source.sequence_id, source.frame_id)
                if not np.array_equal(source.xyzi, truth.xyzi):
                    raise ValueError("labelled evaluation scan differs from the model input")
                prediction = FramePrediction(source.partition, source.sequence_id, source.frame_id,
                                             source.real_slots, output["logits"])
                score, target, eligible = official_frame(truth, prediction)
                use = target >= 0
                row = dict(sequence=source.sequence_id, frame=source.frame_id, eligible=eligible,
                           actual_points=len(source.real_slots), normal_points=int(np.sum(target == 0)),
                           anomaly_points=int(np.sum(target == 1)), offset=count)
                rows.append(row)
                if eligible:
                    score[use].tofile(saved)
                    packed_scores(score[use], target[use], score_kind="logit").tofile(packed)
                    count += int(use.sum())
                    slots = np.flatnonzero(target == 1)
                    points = np.empty(len(slots), POINT_DTYPE)
                    points["sequence"], points["frame"], points["slot"] = source.sequence_id, source.frame_id, slots
                    points["score"] = score[slots]
                    points["q_frame"] = required_frame_fpr(score[target == 0], score[slots])
                    points["distance"] = np.linalg.norm(truth.xyzi[slots, :3], axis=1)
                    blocks.append(points)
                if len(rows) % 200 == 0:
                    disk = host_disk()
                    if disk["SizeRemaining"] < disk["reserve_bytes"]:
                        raise OSError("compact real evaluation reached the E: reserve")
                    print(json.dumps(dict(event="paired_real", frames=len(rows), total=len(dataset),
                        seconds=time.monotonic()-started, temporary_bytes=saved.tell()+packed.tell(), host=disk)), flush=True)
            if len(rows) != len(dataset):
                raise ValueError("real evaluation did not visit every prescribed scan")
            metrics = _paired_metrics(packed, count, observer)
        metrics.update(frames=len(rows), eligible_frames=sum(row["eligible"] for row in rows))
        points = np.concatenate(blocks) if blocks else np.empty(0, POINT_DTYPE)
        if len(points) != metrics["anomaly_count"]:
            raise ValueError("real anomaly records lost official slots")
        points["precision"], points["q_global"] = observer.values(points["score"])
        thresholds = _paired_thresholds(metrics)
        saved.flush()
        for sequence in load_protocol().public_sequence_ids:
            sequence_rows = [row for row in rows if row["sequence"] == sequence]
            sequence_count = 0
            with tempfile.TemporaryFile(dir=run) as packed:
                for row in sequence_rows:
                    if not row["eligible"]:
                        continue
                    source = labelled(sequence, row["frame"])
                    target = evaluation_targets(source.xyzi[:, :3], source.labels.semantic)
                    slots = np.flatnonzero(target >= 0)
                    labels = target[slots]
                    if (int(np.sum(labels == 0)), int(np.sum(labels == 1))) != (row["normal_points"], row["anomaly_points"]):
                        raise ValueError("real compact score membership changed")
                    saved.seek(4*row["offset"])
                    score = np.fromfile(saved, np.float32, len(slots))
                    if len(score) != len(slots):
                        raise ValueError("truncated real compact scores")
                    packed_scores(score, labels, score_kind="logit").tofile(packed)
                    sequence_count += len(score)
                    for kind, value in (("all", "all"), ("sequence", sequence), ("frame", f"{sequence}/{row['frame']}")):
                        _paired_add(groups, "official", kind, value, score, labels, thresholds)
                    distance = np.linalg.norm(source.xyzi[slots, :3], axis=1)
                    bins = range_ids(distance)
                    _paired_add(groups, "official", "distance", bins, score, labels, thresholds)
                    joint_bins = np.array([f"{sequence}/{v}" for v in range(len(RANGES))])[bins]
                    _paired_add(groups, "official", "sequence_distance", joint_bins,
                                score, labels, thresholds)
                    normal = labels == 0
                    _paired_add(groups, "official", "semantic", source.labels.semantic[slots][normal],
                                score[normal], labels[normal], thresholds)
                    for kind in ("real", "geometry"):
                        for index, case in enumerate(cases[kind]):
                            if (case["sequence"], case["frame"]) == (sequence, row["frame"]):
                                take = _paired_case_mask(case, source.xyzi[slots, :3], labels, kind)
                                _paired_add(groups, "official", "fixed_"+kind+"_case", index+1,
                                            score[take], labels[take], thresholds)
                seq_metrics = _paired_metrics(packed, sequence_count)
            seq_points = points[points["sequence"] == sequence]
            sequences[str(sequence)] = dict(metrics=seq_metrics, eligible_frames=sum(r["eligible"] for r in sequence_rows),
                attribution=anomaly_summary(seq_points, len(points), thresholds[[0, 3]]))
    totals = groups["official", "all", "all"]
    if tuple(totals[:2]) != (metrics["normal_count"], metrics["anomaly_count"]):
        raise ValueError("real compact group denominators differ from the official pool")
    for k, point in enumerate([metrics["operating_points"][f"{v:g}"] for v in LIMITS] + [metrics["official_high_recall"]]):
        if tuple(totals[2+2*k:4+2*k]) != (point["fp"], point["tp"]):
            raise ValueError("real groups disagree with official complete-tie working points")
    attribution = anomaly_summary(points, len(points), thresholds[[0, 3]])
    if not np.isclose(attribution["ap_deficit_pp"], 100-metrics["AP"], atol=1e-10, rtol=0):
        raise ValueError("positive-score precision no longer reconstructs official AP")
    for row in rows:
        row.pop("offset")
        if row["eligible"]:
            selected = (points["sequence"] == row["sequence"]) & (points["frame"] == row["frame"])
            row["attribution"] = anomaly_summary(points[selected], len(points), thresholds[[0, 3]])
    np.savez_compressed(run / "anomalies.npz", **{key: points[key] for key in POINT_DTYPE.names})
    return dict(scope="all prescribed val19 scans; exact original official eligible points", metrics=metrics,
                frames=rows, groups=_paired_group_rows(groups), sequences=sequences,
                attribution=attribution, seconds=time.monotonic()-started, model_inputs_label_free=True)


def _paired_original_cases(protocol, data_root, model, cases, metrics):
    from .model import model_input
    from .train import predict

    readers, groups, rows = {}, {}, []
    thresholds = _paired_thresholds(metrics)
    for index, case in enumerate(cases):
        sequence = case["sequence"]
        if sequence not in readers:
            readers[sequence] = STUSequence.open(data_root, protocol=load_protocol(), partition="train",
                                               sequence_id=sequence, label_mode=LabelMode.REQUIRED)
        source = readers[sequence][case["frame"]]
        slots = source.real_slots
        scan = model_input(source.xyzi[slots], slots, protocol["model"],
                           protocol["supervision"]["common"]["sampling_scale"])
        score = predict(model, scan)["logits"]
        normal = source.labels.semantic_target[slots] != 255
        labels = np.zeros(len(slots), np.int8)
        local = normal & _paired_case_mask(case, source.xyzi[slots, :3], labels, "normal")
        _paired_add(groups, "original_normal", "fixed_case", index+1, score[local], labels[local], thresholds)
        anchor = np.flatnonzero(slots == case["slot"])
        if len(anchor) != 1 or not normal[anchor[0]]:
            raise ValueError("fixed original normal anchor lost its identity")
        rows.append(dict(case=index+1, sequence=sequence, frame=case["frame"], slot=case["slot"],
                         kind=case["kind"], normal=int(local.sum()), anchor_score=float(score[anchor[0]]),
                         score_quantiles=np.quantile(score[local], [.1, .5, .9]).tolist() if local.any() else None))
    return dict(scope="six pre-existing normal cases; 206 is training source, 201 is normal-source transfer",
                threshold_source="this arm's complete all-label train/201 synthetic validation curve",
                thresholds=[float(value) if np.isfinite(value) else None for value in thresholds],
                groups=_paired_group_rows(groups), cases=rows)


def evaluate_paired_arm(protocol, data_root, run):
    """Evaluate one fixed checkpoint; temporary predictions disappear after reduction."""
    from .train import load_trained, bind_predictions, seed_all
    import torch

    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    seed_all(protocol["training"]["seed"])
    torch.set_num_threads(protocol["training"]["torch_threads"])
    if (run / "evaluation.json").exists():
        result = json.loads((run / "evaluation.json").read_text())
        if result["binding"] != bind_predictions(protocol, run) or not (run / "anomalies.npz").exists():
            raise ValueError("completed compact evaluation belongs to a different run")
        return result
    disk = host_disk()
    if disk["SizeRemaining"] - 13_000_000_000 < disk["reserve_bytes"]:
        raise OSError("compact paired evaluation needs 13 GB above the physical E: reserve")
    cases = _paired_fixed_cases()
    model = load_trained(protocol, run)
    started = time.monotonic()
    synthetic = _paired_synthetic(protocol, data_root, run, model, cases["synthetic"])
    originals = _paired_original_cases(protocol, data_root, model, cases["normal"], synthetic["metrics"]["all"])
    real = _paired_real(protocol, data_root, run, model, cases)
    result = dict(binding=bind_predictions(protocol, run), synthetic=synthetic, real=real, original_normal=originals,
        fixed_cases=cases, definitions=dict(score="unmodified float32 logit", thresholds="each arm and scope uses its own curve",
            normal_context="retained normal within 2 m of inserted return; existing C3 valid offset [-.2,-.05] m proxy",
            group_overlap="normal-context and fixed-case groups overlap; semantic is normal-only",
            cases="unchanged historical display regions; all scope-valid points, without geometry-validity or new-score selection",
            detection_loss="same class-balanced BCE, averaged over each frame's three views; no auxiliary loss comparison",
            q_frame="same-frame official normal scores >= the anomaly score, including complete ties",
            sequence_work_points="sequence metrics use their own sequence curve; sequence group counts use the global official thresholds",
            input_scope="model geometry sees complete scans; labels only select evaluation outputs",
            validation="20 existing train/201 worlds remain validation only; val19 remains public development evidence"),
        seconds=time.monotonic()-started, host_before=disk, host_after=host_disk())
    _atomic_json(run / "evaluation.json", result)
    return result


def _paired_compare_points(first, second):
    """Join anomaly scores by original slot, never by rank or floating-point value."""
    keys = ("sequence", "frame", "slot")
    ordered = []
    for points in (first, second):
        order = np.lexsort(tuple(points[key] for key in reversed(keys)))
        points = points[order]
        identities = np.column_stack([points[key] for key in keys])
        if len(identities) > 1 and np.any(np.all(identities[1:] == identities[:-1], axis=1)):
            raise ValueError("paired anomaly identities are duplicated")
        ordered.append((points, identities))
    (first, identities), (second, other) = ordered
    if not np.array_equal(identities, other) or not np.array_equal(first["distance"], second["distance"]):
        raise ValueError("paired official anomaly slot identities differ")
    rows = []
    for a, b in zip(first[first["sequence"] == 125], second[second["sequence"] == 125], strict=True):
        row = {key: int(a[key]) for key in keys}
        for field in ("q_frame", "q_global", "precision"):
            row.update({field+"_joint_percent": 100*float(a[field]),
                        field+"_detection_percent": 100*float(b[field]),
                        field+"_delta_pp": 100*(float(a[field])-float(b[field]))})
        rows.append(row)
    return rows


def compare_paired_runs(directory):
    """Compare complete arms at matching identities and independently read work points."""
    import copy

    directory = Path(directory)
    runs = [directory / name for name in ("joint", "detection")]
    protocols = [json.loads((run / "protocol.json").read_text()) for run in runs]
    if [p["training"].get("auxiliary_scale") for p in protocols] != [1., 0.]:
        raise ValueError("paired comparison requires auxiliary scales one and zero")
    common = copy.deepcopy(protocols)
    for protocol in common:
        protocol["training"].pop("auxiliary_scale")
    if common[0] != common[1]:
        raise ValueError("paired arm protocols differ beyond the auxiliary loss multiplier")
    arms = [json.loads((run / "evaluation.json").read_text()) for run in runs]
    if (arms[0]["binding"]["dataset_sha256"] != arms[1]["binding"]["dataset_sha256"]
            or arms[0]["fixed_cases"] != arms[1]["fixed_cases"]
            or arms[0]["definitions"] != arms[1]["definitions"]):
        raise ValueError("paired evaluation data, cases or definitions differ")
    for domain, keys in (("synthetic", ("world", "world_identity", "frame", "eligible", "actual_points", "all", "stu_filtered")),
                         ("real", ("sequence", "frame", "eligible", "actual_points", "normal_points", "anomaly_points"))):
        identities = [[{key: row[key] for key in keys} for row in arm[domain]["frames"]] for arm in arms]
        if identities[0] != identities[1]:
            raise ValueError(f"paired {domain} frame membership or point denominators differ")
    rows = []
    def append_metric(domain, scope, sequence, left, right):
        if any(left[key] != right[key] for key in ("normal_count", "anomaly_count")):
            raise ValueError("paired metric denominators differ")
        for metric in ("AP", "AUROC", "FPR95", "R1", "FPR1"):
            if metric in ("R1", "FPR1"):
                key = "recall" if metric == "R1" else "FPR"
                values = [(m.get("recall_at_fpr_limit") or {}).get(key) for m in (left, right)]
            else:
                values = [m[metric] for m in (left, right)]
            rows.append(dict(domain=domain, scope=scope, sequence=sequence, kind="metric", group="all", metric=metric,
                normal=left["normal_count"], anomaly=left["anomaly_count"], joint=values[0], detection=values[1],
                delta_pp=values[0]-values[1] if all(v is not None for v in values) else None,
                threshold_joint=(left.get("recall_at_fpr_limit") or {}).get("threshold") if metric in ("R1", "FPR1") else None,
                threshold_detection=(right.get("recall_at_fpr_limit") or {}).get("threshold") if metric in ("R1", "FPR1") else None))
    for scope in SCOPES:
        append_metric("synthetic", scope, 201, *(arm["synthetic"]["metrics"][scope] for arm in arms))
    append_metric("real", "official", "all", *(arm["real"]["metrics"] for arm in arms))
    if set(arms[0]["real"]["sequences"]) != set(arms[1]["real"]["sequences"]):
        raise ValueError("paired real sequence sets differ")
    for sequence in arms[0]["real"]["sequences"]:
        append_metric("real", "official", int(sequence), *(arm["real"]["sequences"][sequence]["metrics"] for arm in arms))
    for domain in ("synthetic", "real", "original_normal"):
        tables = [{(r["scope"], r["kind"], r["group"]): r for r in arm[domain]["groups"]} for arm in arms]
        if tables[0].keys() != tables[1].keys():
            raise ValueError("paired fixed group membership differs")
        for key, left in tables[0].items():
            right = tables[1][key]
            if (left["normal"], left["anomaly"]) != (right["normal"], right["anomaly"]):
                raise ValueError("paired fixed group point denominators differ")
            for metric in ("FPR_1", "recall_1", "FPR_95", "recall_95"):
                a, b = left[metric], right[metric]
                rows.append(dict(domain=domain, scope=key[0], sequence="", kind=key[1], group=key[2], metric=metric,
                    normal=left["normal"], anomaly=left["anomaly"], joint=a, detection=b,
                    delta_pp=a-b if a is not None and b is not None else None))
    points = []
    for run in runs:
        with np.load(run / "anomalies.npz", allow_pickle=False) as saved:
            array = np.empty(len(saved["slot"]), POINT_DTYPE)
            for name in POINT_DTYPE.names:
                array[name] = saved[name]
            arm = arms[len(points)]
            if len(array) != arm["real"]["metrics"]["anomaly_count"]:
                raise ValueError("paired anomaly table differs from the official positive denominator")
            points.append(array)
    paired = _paired_compare_points(*points)
    point_summary = {}
    for metric in ("q_frame", "q_global"):
        delta = np.array([row[metric+"_delta_pp"] for row in paired])
        point_summary[metric] = dict(points=len(delta), joint_lower=int(np.sum(delta < 0)),
            joint_higher=int(np.sum(delta > 0)), equal=int(np.sum(delta == 0)),
            mean_delta_pp=float(delta.mean()) if len(delta) else None,
            median_delta_pp=float(np.median(delta)) if len(delta) else None)
    result = dict(scope="one paired seed and one epoch; same official point identities and fixed diagnostic groups",
        delta="joint minus detection; AP/R1 higher is better, FPR95 and required FPR lower is better",
        work_points="each arm uses its own complete-tie curve threshold; subgroup FPRs need not match",
        sequence_work_points="per-sequence metrics use sequence curves; sequence groups use global official thresholds",
        sequence125_weighting="paired anomaly point occurrences, not equal frame weights or a mechanism attribution",
        denominator_check=True, normal_context="overlapping fixed proxies and cases, not an exhaustive structure taxonomy",
        frames=dict(real=len(arms[0]["real"]["frames"]), synthetic=len(arms[0]["synthetic"]["frames"])),
        common_detection_loss={name: arm["synthetic"]["detection_loss"] for name, arm in zip(("joint", "detection"), arms)},
        sequence125=point_summary, metric_rows=[r for r in rows if r["kind"] == "metric"],
        binding={name: arm["binding"] for name, arm in zip(("joint", "detection"), arms)},
        interpretation="total losses are not compared; geometric probe failures are not model predictions; val19 is development data")
    write_table(directory / "comparison.csv", rows)
    write_table(directory / "125.csv", paired)
    _atomic_json(directory / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("results/synthetic/experiment"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--run", type=Path, default=Path("results/v1"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--real", action="store_true", help="attribute saved real val19 ranking and inspect six fixed cases")
    parser.add_argument("--paired-run", type=Path, help="compactly evaluate this arm using its saved protocol.json")
    parser.add_argument("--compare", type=Path, help="compare complete joint and detection evaluations in this directory")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    if args.paired_run and args.compare:
        parser.error("choose one paired evaluation action")
    if args.paired_run:
        protocol = json.loads((args.paired_run / "protocol.json").read_text())
        evaluate_paired_arm(protocol, args.data_root, args.paired_run)
        return
    if args.compare:
        compare_paired_runs(args.compare)
        return
    if args.real:
        diagnose_real(args)
        return
    started = time.monotonic()
    previous = json.loads((args.run / "synthetic.json").read_text())
    binding = json.loads((args.run / "predictions" / "manifest.json").read_text())
    with (args.run / "epoch1.pt").open("rb") as stream:
        checkpoint_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = dict(checkpoint_sha256=checkpoint_digest,
                    dataset_sha256=hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest(), score="raw_float32_logit")
    if binding != expected:
        raise ValueError("prediction binding differs from checkpoint or final dataset")
    output = args.run / "diagnosis"
    output.mkdir(exist_ok=True)
    resources = host_disk()
    # Two record sets plus one sorted copy, with a small allowance for reports.
    peak = 24 * (previous["metrics"]["normal_count"] + previous["metrics"]["anomaly_count"]) + 100_000_000
    if resources["SizeRemaining"] - peak < resources["reserve_bytes"]:
        raise OSError("diagnostic sorting would consume the E: safety reserve")
    print(json.dumps(dict(event="start", jobs=args.jobs, peak_new_bytes=peak, host=resources)), flush=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        initargs = (args.dataset, args.data_root, args.run, temporary)
        initialize(*initargs)
        frames = sorted(SAMPLES)
        packed_rows = []
        with ProcessPoolExecutor(args.jobs, initializer=initialize, initargs=initargs) as pool:
            for i, rows in enumerate(pool.map(pack_frame, frames), 1):
                packed_rows.extend(rows)
                if i % 100 == 0 or i == len(frames):
                    print(json.dumps(dict(event="pack", source_frames=i, worlds=len(WORLDS), seconds=time.monotonic()-started,
                                          host=host_disk())), flush=True)
        metrics = {}
        for scope in SCOPES:
            paths = [Path(temporary) / f"{frame:06d}_{scope}.bin" for frame in frames]
            print(json.dumps(dict(event="sort", scope=scope, bytes=sum(p.stat().st_size for p in paths), host=host_disk())), flush=True)
            metrics[scope] = pooled_files(paths, score_kind="logit", fpr_limits=LIMITS)
            for path in paths:
                path.unlink()
            eligible = [r for r in packed_rows if scope == "all" or r["eligible"]]
            metrics[scope].update(frames=len(eligible), source_frames=len({r["frame"] for r in eligible}),
                worlds=len({r["world_identity"] for r in eligible}),
                anomaly_prevalence_percent=100 * metrics[scope]["anomaly_count"] / (metrics[scope]["normal_count"] + metrics[scope]["anomaly_count"]))
            for kind, key in (("normal", "normal_count"), ("anomaly", "anomaly_count")):
                if sum(row[scope][kind] for row in packed_rows) != metrics[scope][key]:
                    raise ValueError("world/frame counts disagree with metric pool")
            print(json.dumps(dict(event="metrics", scope=scope, metrics=metrics[scope])), flush=True)
        for key, value in previous["metrics"].items():
            if metrics["all"][key] != value:
                raise ValueError(f"original full-synthetic result did not reproduce: {key}")
        _atomic_json(output / "metrics.json", dict(binding=binding, metrics=metrics, frame_counts=packed_rows,
                                                    original_full_result_exactly_reproduced=True))
        thresholds = np.array([[metrics[scope]["operating_points"][f"{limit:g}"]["threshold"]
                                if metrics[scope]["operating_points"][f"{limit:g}"]["threshold"] is not None else np.inf
                                for limit in LIMITS] for scope in SCOPES], np.float64)
        rows = []
        with ProcessPoolExecutor(args.jobs, initializer=initialize, initargs=(*initargs, thresholds)) as pool:
            for i, row in enumerate(pool.map(analyze_frame, frames), 1):
                rows.append(row)
                if i % 100 == 0 or i == len(frames):
                    print(json.dumps(dict(event="groups", source_frames=i, seconds=time.monotonic()-started,
                                          host=host_disk())), flush=True)
        summary = aggregate(rows, metrics)
        cases = inspect_cases(rows, summary, metrics, output)
    _atomic_json(output / "groups.json", dict(binding=binding, scopes=summary, source_frames=rows, cases=cases,
        definitions=dict(distance_bins=RANGES, frame_types=FRAME_GROUPS,
            official_membership={"0": "ineligible/outside", "1": "ineligible/inside", "2": "eligible/outside", "3": "eligible/inside"},
            point_identity="train/201, source frame, original slot; independent of synthetic world",
            position_identity="train/201, source frame, exact original xyz; not a tracked surface across frames",
            recurrence="number of distinct worlds; coincident slots within one world count once",
            case_selection="top three normal semantic FP contributors per scope at 0.1% global FPR; densest one-metre cell of distinct FP positions across source frames; six-metre display only"),
        seconds=time.monotonic()-started, jobs=args.jobs, host_before=resources, host_after=host_disk()))
    print(json.dumps(dict(event="complete", seconds=time.monotonic()-started, directory=str(output))), flush=True)


if __name__ == "__main__":
    main()
